"""Capabilities chained through the catalogue's registry against the real Postgres stores."""
import dataclasses
import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import xarray as xr

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_grids import NJ, granule
from tests.test_harness_sources_monitors import KEY, FakeAqs, default_rows
from tests.test_harness_sources_monitors import SECRET as AQS_SECRET
from tests.test_harness_sources_satellite import NO2, TODAY, FakeService
from tests.test_harness_sources_satellite import SECRET as EDL_SECRET
from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.handles import HandleError, HandleRecord
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import Stores, ensure_tables, postgres_stores
from tta_backend.harness.testing import (
    FakePlaces,
    FixedClock,
    MemoryArtifacts,
    StaticCredentials,
    make_run_context,
    make_run_env,
)
from tta_backend.harness.trace import FailureCategory, RunNotFound, StepFailure

JUNE = {"start": "2024-06-01", "end": "2024-06-03"}
LATER = {"start": "2024-06-10", "end": "2024-06-12"}
EMAIL = "researcher@lab.edu"


def three_days(first: str, value: float) -> xr.Dataset:
    days = [f"{first[:8]}{int(first[8:]) + n:02d}" for n in range(3)]
    return xr.concat([granule(day=day, value=value + n * 1e14) for n, day in enumerate(days)], dim="time")


class Chain:
    """One user's run over shared stores, calling capabilities by name through the catalogue."""

    def __init__(self, env: RunEnv, artifacts: MemoryArtifacts, stores: Stores) -> None:
        self.env, self.artifacts, self.stores = env, artifacts, stores
        self.registry = build_registry()
        self.handles: list[str] = []

    async def call(self, name: str, params: dict[str, Any], step: str | None = None) -> Any:
        result = await invoke(self.env, self.registry.get(name), params, step=step or name)
        if result.handle:
            self.handles.append(result.handle)
        return result

    async def record(self, handle: str) -> HandleRecord:
        return await self.stores.handles.get(self.env.context.identity.user_id, handle)

    async def ancestors(self, handle: str) -> set[str]:
        found: set[str] = set()
        todo = [handle]
        while todo:
            for parent in (await self.record(todo.pop())).parents:
                if parent not in found:
                    found.add(parent)
                    todo.append(parent)
        return found


class EndToEndTests(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        aqs_client.reset_request_state()
        self.addCleanup(aqs_client.reset_request_state)
        self.service = FakeService()
        self.service.make_file = lambda day: three_days(day, 2e15 if day < "2024-06-05" else 3e15)
        self.earthdata = self.service.earthdata()
        self.earthdata.tools["get_provenance"] = lambda args: {"handle": args["handle"], "ancestors": [], "events": []}
        self.aqs = FakeAqs(default_rows())
        real_fetcher = aqs_client.fetcher
        for p in (
            patch.object(aqs_client, "fetcher", lambda credential: real_fetcher(credential, transport=self.aqs.transport)),
            patch.object(aqs_client.asyncio, "sleep", AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)

    @asynccontextmanager
    async def world(self) -> AsyncIterator[Any]:
        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.service.files[record.id]

        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)

                async def chain_for(user: str) -> Chain:
                    artifacts = MemoryArtifacts()
                    context = make_run_context(
                        user_id=user, stores=stores, artifacts=artifacts, clock=FixedClock(TODAY),
                        credentials=StaticCredentials({"earthdata": EDL_SECRET, "epa_aqs": AQS_SECRET}),
                    )
                    env = await make_run_env(
                        root, context=context, earthdata=self.earthdata, open_retrieval=open_retrieval,
                        places=FakePlaces({"new jersey": NJ}),
                    )
                    return Chain(env, artifacts, stores)

                yield chain_for, stores

    async def satellite(self, chain: Chain) -> dict[str, Any]:
        area = await chain.call("sources.areas.resolve", {"place": "New Jersey"})
        got = {}
        for key, period in (("june", JUNE), ("later", LATER)):
            got[key] = await chain.call("sources.satellite.get_data", {
                "quantity": NO2, "area": area.handle, "dates": period, "date_role": "range",
            }, step=f"get_{key}")
        means = {
            key: await chain.call("operations.aggregate_time", {"data": got[key].handle}, step=f"mean_{key}")
            for key in got
        }
        diff = await chain.call(
            "operations.difference", {"a": means["june"].handle, "b": means["later"].handle}, step="diff",
        )
        stat = await chain.call("operations.statistic", {"data": diff.handle, "stat": "mean"}, step="stat")
        chart = await chain.call("render.map", {"grid": diff.handle, "style": "diverging"}, step="map")
        lineage = await chain.call("explain.lineage", {"handles": [diff.handle]}, step="lineage")
        reliability = await chain.call("explain.reliability", {"handle": diff.handle}, step="reliability")
        return {"area": area, **got, "means": means, "diff": diff, "stat": stat, "chart": chart,
                "lineage": lineage, "reliability": reliability}

    async def test_the_satellite_chain_runs_records_and_explains_itself(self):
        async with self.world() as (chain_for, stores):
            chain = await chain_for("user-a")
            s = await self.satellite(chain)

            self.assertEqual(s["diff"].status, "done")
            self.assertGreater(s["diff"].summary["mean"], 0)
            self.assertEqual((s["stat"].status, s["chart"].status), ("done", "done"))

            entries = await stores.traces.entries("user-a", chain.env.run_id)
            done = [e for e in entries if e.status == "done"]
            self.assertEqual([(e.capability, e.step) for e in done], [
                ("sources.areas.resolve", "sources.areas.resolve"),
                ("sources.satellite.get_data", "get_june"),
                ("sources.satellite.get_data", "get_later"),
                ("operations.aggregate_time", "mean_june"),
                ("operations.aggregate_time", "mean_later"),
                ("operations.difference", "diff"),
                ("operations.statistic", "stat"),
                ("render.map", "map"),
                ("explain.lineage", "lineage"),
                ("explain.reliability", "reliability"),
            ])
            self.assertEqual([e.seq for e in entries], sorted({e.seq for e in entries}))

            diff = s["diff"].handle
            mean_june, mean_later = s["means"]["june"].handle, s["means"]["later"].handle
            self.assertEqual(await chain.ancestors(diff), {s["june"].handle, s["later"].handle, mean_june, mean_later})
            self.assertEqual((await chain.record(mean_june)).parents, (s["june"].handle,))

            for handle, period in ((s["june"].handle, JUNE), (s["later"].handle, LATER)):
                descriptor = (await chain.record(handle)).descriptor
                self.assertEqual((descriptor.product, descriptor.period.start, descriptor.period.end),
                                 ("TEMPO_NO2", period["start"], period["end"]))
                self.assertEqual(descriptor.region["area"], s["area"].handle)
                self.assertEqual(descriptor.masking["geometry"], "area")
            for handle in (mean_june, mean_later):
                descriptor = (await chain.record(handle)).descriptor
                self.assertEqual((descriptor.product, descriptor.quantity), ("TEMPO_NO2", NO2))
                self.assertEqual(descriptor.region["name"], "New Jersey")
                self.assertEqual(descriptor.masking["variable"], "vertical_column_troposphere")
            result = (await chain.record(diff)).descriptor
            self.assertEqual(result.product, "TEMPO_NO2")
            self.assertEqual(result.region["name"], "New Jersey")
            # Two different periods have no single period; the handle records both.
            self.assertIsNone(result.period)
            self.assertEqual((await chain.record(diff)).parameters["periods"], {
                "a": {"start": "2024-06-01", "end": "2024-06-03"},
                "b": {"start": "2024-06-10", "end": "2024-06-12"},
            })
            self.assertEqual(s["diff"].values["period_a"], {"start": "2024-06-01", "end": "2024-06-03"})
            self.assertEqual(s["diff"].values["period_b"], {"start": "2024-06-10", "end": "2024-06-12"})
            for handle, period in ((mean_june, JUNE), (mean_later, LATER)):
                kept = (await chain.record(handle)).descriptor.period
                self.assertEqual((kept.start, kept.end), (period["start"], period["end"]))
            self.assertEqual(len(result.masking["inputs"]), 2)
            self.assertEqual(result.masking["inputs"][0]["variable"], "vertical_column_troposphere")

            [chart_id] = s["chart"].charts
            self.assertEqual(chain.artifacts.stored[chart_id]["metadata"]["source_handles"], [diff])
            self.assertTrue(diff.startswith("grid_"))

            lineage = [step["handle"] for step in s["lineage"].values["steps"]]
            for handle in (s["june"].handle, s["later"].handle, diff):
                self.assertIn(handle, lineage)
            self.assertLess(lineage.index(s["june"].handle), lineage.index(diff))
            self.assertTrue(s["reliability"].values["has_evidence"])
            self.assertEqual(s["reliability"].summary["product"], "TEMPO_NO2")

    async def test_the_monitor_chain_and_a_regional_series_share_a_chart(self):
        async with self.world() as (chain_for, stores):
            chain = await chain_for("user-a")
            area = await chain.call("sources.areas.resolve", {"place": "New Jersey"})
            grid = await chain.call("sources.satellite.get_data", {
                "quantity": NO2, "area": area.handle, "dates": JUNE, "date_role": "range",
            }, step="get")
            found = await chain.call("sources.monitors.find", {
                "pollutant": "o3", "lat": 34.07, "lon": -118.22, "limit": 2, "start": "2024-06-01", "end": "2024-06-02",
            }, step="find")
            readings = await chain.call("sources.monitors.readings", {
                "pollutant": "o3", "points": found.handle, "start": "2024-06-01", "end": "2024-06-02",
            }, step="readings")
            alone = await chain.call("render.timeseries", {"series": [readings.handle]}, step="alone")
            regional = await chain.call("operations.regional_series", {"data": grid.handle}, step="regional")
            both = await chain.call("render.timeseries", {"series": [readings.handle, regional.handle]}, step="both")

            entries = await stores.traces.entries("user-a", chain.env.run_id)
            self.assertEqual([e.capability for e in entries if e.status == "done"], [
                "sources.areas.resolve", "sources.satellite.get_data", "sources.monitors.find",
                "sources.monitors.readings", "render.timeseries", "operations.regional_series", "render.timeseries",
            ])

            self.assertEqual(await chain.ancestors(regional.handle), {grid.handle})
            self.assertEqual(await chain.ancestors(readings.handle), {found.handle})
            monitors =(await chain.record(readings.handle)).descriptor
            self.assertEqual((monitors.product, monitors.period.start, monitors.period.end),
                             ("EPA AQS", "2024-06-01", "2024-06-02"))
            satellite = (await chain.record(regional.handle)).descriptor
            self.assertEqual((satellite.product, satellite.region["name"], satellite.masking["variable"]),
                             ("TEMPO_NO2", "New Jersey", "vertical_column_troposphere"))
            self.assertEqual((satellite.period.start, satellite.period.end), ("2024-06-01", "2024-06-03"))

            [alone_id], [both_id] = alone.charts, both.charts
            self.assertEqual(chain.artifacts.stored[alone_id]["metadata"]["source_handles"], [readings.handle])
            sources = chain.artifacts.stored[both_id]["metadata"]["source_handles"]
            self.assertEqual(sources, [readings.handle, regional.handle])
            self.assertTrue(all(h.startswith(("series_", "grid_")) for h in sources))

    async def test_a_second_user_reads_none_of_the_first_users_work(self):
        async with self.world() as (chain_for, stores):
            first = await chain_for("user-a")
            s = await self.satellite(first)
            second = await chain_for("user-b")

            for handle in (s["june"].handle, s["diff"].handle, s["area"].handle):
                with self.assertRaises(HandleError):
                    await stores.handles.get("user-b", handle)
                for name, params in (
                    ("explain.lineage", {"handles": [handle]}),
                    ("explain.reliability", {"handle": handle}),
                ):
                    with self.assertRaises(StepFailure) as caught:
                        await second.call(name, params)
                    self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)
            for name, params in (
                ("operations.statistic", {"data": s["diff"].handle}),
                ("render.map", {"grid": s["diff"].handle}),
            ):
                with self.assertRaises(StepFailure):
                    await second.call(name, params)

            with self.assertRaises(RunNotFound):
                await stores.traces.entries("user-b", first.env.run_id)
            with self.assertRaises(RunNotFound):
                await stores.traces.get_run("user-b", first.env.run_id)
            self.assertNotIn(first.env.run_id, [r.id for r in await stores.traces.list_runs("user-b", "conv-1")])
            self.assertEqual(second.handles, [])

    async def test_nothing_recorded_holds_a_credential(self):
        async with self.world() as (chain_for, stores):
            chain = await chain_for("user-a")
            await self.satellite(chain)
            found = await chain.call("sources.monitors.find", {
                "pollutant": "o3", "lat": 34.07, "lon": -118.22, "start": "2024-06-01", "end": "2024-06-02",
            })
            readings = await chain.call("sources.monitors.readings", {
                "pollutant": "o3", "points": found.handle, "start": "2024-06-01", "end": "2024-06-02",
            })
            await chain.call("render.timeseries", {"series": [readings.handle]})

            recorded = [
                json.dumps(dataclasses.asdict(entry), default=str)
                for entry in await stores.traces.entries("user-a", chain.env.run_id)
            ]
            recorded += [(await chain.record(handle)).model_dump_json() for handle in chain.handles]
            text = "\n".join(recorded)
            self.assertIn("sources.monitors.readings", text)
            self.assertGreater(len(chain.handles), 8)
            for secret in (KEY, EMAIL, EDL_SECRET, AQS_SECRET):
                self.assertNotIn(secret, text)
