"""Each shipped workflow run end to end in memory: the real capabilities on a fake earthdata service, AQS and NOAA archive."""
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from tests.test_harness_grids import layered_retrieval
from tests.test_harness_sources_monitors import FakeAqs, daily_row, monitor_row
from tests.test_harness_sources_monitors import SECRET as AQS_SECRET
from tests.test_harness_sources_satellite import NO2, FakeService
from tests.test_harness_sources_smoke import smoke_archive
from tests.test_harness_workflow_runner import AREA_ENTRY, Fixture, three_days
from tta_backend.datasets.registry import load_registry
from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.capabilities.sources import smoke
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.results import Mode, Result
from tta_backend.harness.workflows.loader import load_packs

PACKS = Path(__file__).resolve().parent.parent / "packs"
EDL_SECRET = "edl-packs-run-secret-5d2a"
JUNE = {"start": "2024-06-01", "end": "2024-06-03"}
JULY = {"start": "2024-06-10", "end": "2024-06-12"}
SMOKE_DATE = "2026-01-01"

GET_DATA = "sources.satellite.get_data"
MAP = "render.map"
TIMESERIES = "render.timeseries"
TABLE = "render.table"
POINT = "sources.satellite.get_point_series"
REGIONAL = "operations.regional_series"
STATISTIC = "operations.statistic"

# Two ozone monitors inside New Jersey; site A is above the 8-hour level on one day.
SITE_A = {"state_code": "34", "county_code": "003", "site_number": "0010", "latitude": 40.8, "longitude": -74.1,
          "local_site_name": "Bergen"}
SITE_B = {"state_code": "34", "county_code": "007", "site_number": "0003", "latitude": 39.9, "longitude": -75.0,
          "local_site_name": "Camden"}
# One plume over New Jersey and one far away.
PLUMES = [
    ((-75.0, 39.5, -74.0, 40.5), "Heavy", "2026001 1500", "2026001 1700"),
    ((-120.0, 35.0, -110.0, 40.0), "Light", "2026001 1200", "2026001 1400"),
]


def aqs_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "monitors": [monitor_row(SITE_A), monitor_row(SITE_B)],
        "sampleData": [],
        "dailyData": [
            daily_row(SITE_A, "2024-06-01", 0.075), daily_row(SITE_A, "2024-06-02", 0.065),
            daily_row(SITE_B, "2024-06-01", 0.060), daily_row(SITE_B, "2024-06-02", 0.055),
        ],
    }


def layered_for_new_jersey(day: str) -> xr.Dataset:
    ds = layered_retrieval(axes_as_variables=True, day=day)
    return ds.assign_coords(latitude=np.array([39.5, 40.5]), longitude=np.array([-75.0, -74.0]))


def step(capability: str, step_id: str, status: str = "done") -> tuple[str, str, str, str]:
    return ("use", capability, step_id, status)


def fetch(capability: str, step_id: str) -> list[tuple[str, str, str, str]]:
    """A retrieval is saved as waiting, then finishes."""
    return [step(capability, step_id, "waiting"), step(capability, step_id)]


class PackRun(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.service = FakeService()
        self.service.make_file = lambda day: three_days(day, 2e15 if day < "2024-06-05" else 3e15)
        self.aqs = FakeAqs(aqs_rows())
        registry = build_registry()
        self.catalogue = load_packs(PACKS, registry, load_registry())

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.service.files[record.id]

        self.f = Fixture(
            tmp.name, registry, self.catalogue, earthdata=self.service.earthdata(), open_retrieval=open_retrieval,
            credentials={"earthdata": EDL_SECRET, "epa_aqs": AQS_SECRET},
        )
        aqs_client.reset_request_state()
        self.addCleanup(aqs_client.reset_request_state)
        real_fetcher = aqs_client.fetcher
        patches = (
            patch.object(aqs_client, "fetcher", lambda credential: real_fetcher(credential, transport=self.aqs.transport)),
            patch.object(aqs_client.asyncio, "sleep", AsyncMock()),
            patch.object(smoke, "download_archive", AsyncMock(return_value=smoke_archive(PLUMES))),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def run_workflow(self, name: str, inputs: dict[str, Any]) -> tuple[Result, list[tuple[str, str | None, str | None, str]]]:
        result = await self.f.start(inputs, name=name)
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.mode, Mode.draft(name, 1))
        self.assertEqual(await self.f.run_status(result.trace_id), "done")
        return result, await self.f.entries(result.trace_id)

    def assert_charts_stored(self, result: Result, count: int = 1) -> None:
        self.assertEqual(len(result.charts), count)
        for chart in result.charts:
            self.assertIn(chart, self.f.artifacts.stored)

    def names(self, result: Result) -> list[str]:
        return [finding.name for finding in result.findings]


class SatelliteWorkflowRuns(PackRun):
    async def test_map(self):
        result, entries = await self.run_workflow("map", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        self.assertEqual(entries, [AREA_ENTRY, *fetch(GET_DATA, "data"), step(MAP, "figure")])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), [])
        self.assertEqual(result.handles, ("obs_1",))
        self.assertEqual(self.service.jobs, {"job_1": "obs_1"})

    async def test_map_with_a_named_product(self):
        result, entries = await self.run_workflow(
            "map", {"quantity": NO2, "area": "New Jersey", "dates": JUNE, "named": "TEMPO_NO2"},
        )
        self.assertEqual(entries, [AREA_ENTRY, *fetch(GET_DATA, "data"), step(MAP, "figure")])
        self.assert_charts_stored(result)
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2"))
        self.assertIn("it was named in the request", result.disclosures[0])

    async def test_timeseries_in_region_mode(self):
        result, entries = await self.run_workflow("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        # Steps ready together are written in the spec's file order, so a skipped branch interleaves with the live one.
        self.assertEqual(entries, [
            AREA_ENTRY,
            step(GET_DATA, "data", "waiting"),
            step(POINT, "point", "skipped"),
            step(GET_DATA, "data"),
            step(REGIONAL, "series"),
            step(TIMESERIES, "point_figure", "skipped"),
            step(STATISTIC, "point_average", "skipped"),
            step(TIMESERIES, "figure"),
            step(STATISTIC, "average"),
        ])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["days_with_data", "series_mean", "series_max", "series_min"])

    async def test_timeseries_in_point_mode(self):
        self.service.series = pd.DataFrame({
            "time": ["2024-06-01T15:00:00Z", "2024-06-02T15:00:00Z", "2024-06-03T15:00:00Z"],
            "vertical_column_troposphere": [1.0e15, 2.0e15, 3.0e15],
        })
        result, entries = await self.run_workflow(
            "timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE, "mode": "point"},
        )
        self.assertEqual(entries, [
            AREA_ENTRY,
            step(GET_DATA, "data", "skipped"),
            step(POINT, "point", "waiting"),
            step(POINT, "point"),
            step(REGIONAL, "series", "skipped"),
            step(TIMESERIES, "point_figure"),
            step(STATISTIC, "point_average"),
            step(TIMESERIES, "figure", "skipped"),
            step(STATISTIC, "average", "skipped"),
        ])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["point_days_with_data", "point_mean"])

    async def test_timeseries_with_a_named_product(self):
        result, _ = await self.run_workflow(
            "timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE, "named": "TEMPO_NO2"},
        )
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2"))
        self.assertIn("it was named in the request", result.disclosures[0])

    async def test_point_timeseries_with_a_named_product(self):
        self.service.series = pd.DataFrame({
            "time": ["2024-06-01T15:00:00Z", "2024-06-02T15:00:00Z"],
            "vertical_column_troposphere": [1.0e15, 2.0e15],
        })
        result, _ = await self.run_workflow(
            "timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE, "mode": "point", "named": "TEMPO_NO2"},
        )
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2"))

    async def test_vertical_profile_with_a_named_product(self):
        self.service.make_file = layered_for_new_jersey
        result, _ = await self.run_workflow(
            "vertical-profile",
            {"area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-01"}, "named": "TEMPO_O3PROF"},
        )
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_O3PROF"))
        self.assertIn("it was named in the request", result.disclosures[0])

    async def test_period_comparison_with_a_named_product_keeps_both_periods_on_it(self):
        result, _ = await self.run_workflow(
            "period-comparison",
            {"quantity": NO2, "area": "New Jersey", "first": JUNE, "second": JULY, "named": "TEMPO_NO2"},
        )
        self.assertEqual(result.status, "done", result.reason)
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2"))
        self.assertEqual(len(self.service.jobs), 2)

    async def test_vertical_profile(self):
        self.service.make_file = layered_for_new_jersey
        result, entries = await self.run_workflow(
            "vertical-profile", {"area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-01"}},
        )
        self.assertEqual(entries, [
            AREA_ENTRY, *fetch(GET_DATA, "data"),
            step("operations.vertical_profile", "profile"),
            step("render.profile", "figure"),
        ])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["layers"])

    async def test_period_comparison(self):
        result, entries = await self.run_workflow(
            "period-comparison", {"quantity": NO2, "area": "New Jersey", "first": JUNE, "second": JULY},
        )
        done = [e for e in entries if e[3] == "done"]
        self.assertEqual(entries[0], AREA_ENTRY)
        self.assertEqual(
            sorted(e[2] for e in done),
            sorted(["input.area", "first_data", "second_data", "first_mean", "second_mean", "diff", "figure", "stat"]),
        )
        self.assertEqual(len(self.service.jobs), 2)
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["mean_change", "percent_change", "largest_increase", "largest_decrease"])
        mean_change = next(f for f in result.findings if f.name == "mean_change")
        # The second period is the later, higher one, so the change is positive.
        self.assertAlmostEqual(float(mean_change.value) / 1e15, 1.0, places=6)


class CadenceWorkflowRuns(PackRun):
    """Version 2 of map, timeseries and period-comparison passes the requested cadence to every retrieval."""

    MONTHLY = "TROPOMI_NO2"

    async def start_v2(self, name: str, inputs: dict[str, Any]) -> Result:
        return await self.f.start({"quantity": NO2, "area": "New Jersey", **inputs}, name=name, version=2)

    async def test_a_daily_request_against_a_monthly_product_is_refused(self):
        cases = {
            "map": {"dates": JUNE},
            "timeseries": {"dates": JUNE},
            "period-comparison": {"first": JUNE, "second": JULY},
        }
        for name, inputs in cases.items():
            with self.subTest(name):
                result = await self.start_v2(name, {**inputs, "named": self.MONTHLY, "cadence": "daily"})
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertIn("so daily values are not available", result.reason or "")
                self.assertEqual(result.mode, Mode.draft(name, 2))

    async def test_a_daily_point_request_is_refused_before_any_retrieval(self):
        result = await self.start_v2(
            "timeseries", {"dates": JUNE, "mode": "point", "named": self.MONTHLY, "cadence": "daily"},
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("so daily values are not available", result.reason or "")
        self.assertEqual(self.service.jobs, {})

    async def test_a_monthly_request_is_accepted_by_a_monthly_product(self):
        result = await self.start_v2("map", {"dates": JUNE, "named": self.MONTHLY, "cadence": "monthly"})
        self.assertNotIn("values are not available", result.reason or "")

    async def test_no_cadence_keeps_the_monthly_product_as_before(self):
        result = await self.start_v2("map", {"dates": JUNE, "named": self.MONTHLY})
        self.assertNotIn("values are not available", result.reason or "")
        v1 = await self.f.start({"quantity": NO2, "area": "New Jersey", "dates": JUNE, "named": self.MONTHLY}, name="map")
        self.assertEqual(v1.status, result.status)

    async def test_version_one_ignores_a_cadence_input(self):
        result = await self.f.start(
            {"quantity": NO2, "area": "New Jersey", "dates": JUNE, "named": self.MONTHLY, "cadence": "daily"}, name="map",
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("not an input", result.reason or "")


class GroundWorkflowRuns(PackRun):
    inputs = {"pollutant": "o3", "area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-02"}}
    EXCEEDANCE_FINDINGS = [
        "monitors_found", "exceedance_days", "monitor_days_exceeding", "naaqs_level", "highest_exceeding_value",
    ]

    def expected(self, view: str) -> list[tuple[str, str, str, str]]:
        def status(shown: bool) -> str:
            return "done" if shown else "skipped"

        # Steps ready together are written in the spec's file order.
        return [
            AREA_ENTRY,
            step("sources.monitors.find", "find"),
            step("sources.monitors.readings", "readings", status(view == "series")),
            step("sources.monitors.daily_summary", "summary", status(view in ("map", "table"))),
            step("sources.monitors.exceedance_days", "exceedances"),
            step(TIMESERIES, "figure_series", status(view == "series")),
            step("render.monitor_map", "figure_map", status(view == "map")),
            step(TABLE, "figure_table", status(view == "table")),
            step(TABLE, "figure_exceedances", status(view == "exceedances")),
        ]

    async def run_view(self, view: str) -> Result:
        result, entries = await self.run_workflow("monitor-readings", {**self.inputs, "view": view})
        self.assertEqual(entries, self.expected(view))
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), self.EXCEEDANCE_FINDINGS)
        return result

    async def test_series_view(self):
        await self.run_view("series")

    async def test_map_view(self):
        await self.run_view("map")

    async def test_table_view(self):
        await self.run_view("table")

    async def test_exceedances_view(self):
        result = await self.run_view("exceedances")
        days = next(f for f in result.findings if f.name == "exceedance_days")
        self.assertEqual(days.value, 1)

    async def test_the_default_view_is_a_series(self):
        result, entries = await self.run_workflow("monitor-readings", self.inputs)
        self.assertEqual(entries, self.expected("series"))
        self.assert_charts_stored(result)

    async def test_version_3_gives_the_reading_level_in_every_view(self):
        for view in ("series", "map", "table", "exceedances"):
            with self.subTest(view=view):
                result = await self.f.start({**self.inputs, "view": view}, name="monitor-readings", version=3)
                self.assertEqual(result.status, "done", result.reason)
                names = self.names(result)
                self.assertIn("reading_mean", names)
                self.assertIn("reading_max", names)
                self.assert_charts_stored(result)


class SmokeWorkflowRuns(PackRun):
    async def test_over_an_area(self):
        result, entries = await self.run_workflow("smoke-map", {"date": SMOKE_DATE, "area": "New Jersey"})
        self.assertEqual(entries, [AREA_ENTRY, step("sources.smoke.get_extent", "extent"), step("render.polygon_map", "figure")])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["plumes", "smoke_area"])
        self.assertEqual(next(f for f in result.findings if f.name == "plumes").value, 1)

    async def test_the_whole_product_without_an_area(self):
        result, entries = await self.run_workflow("smoke-map", {"date": SMOKE_DATE})
        self.assertEqual(entries, [step("sources.smoke.get_extent", "extent"), step("render.polygon_map", "figure")])
        self.assert_charts_stored(result)
        self.assertEqual(self.names(result), ["plumes", "smoke_area"])
        self.assertEqual(next(f for f in result.findings if f.name == "plumes").value, 2)
        self.assertEqual(self.f.places.asked, [])


if __name__ == "__main__":
    unittest.main()
