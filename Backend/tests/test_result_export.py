"""A result's export and a harness chart's metadata, from real workflow runs on fake sources.

The map workflow draws its chart from a retrieval handle; the timeseries
workflow from a series the backend derived. Charts are stored through the
app's StoredArtifacts, so each carries the link to its run.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
import tempfile
import zipfile
from typing import Any

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from harness_chat_helpers import MemoryCharts, MemoryTranscripts, memory_repositories  # noqa: E402
from tests.test_harness_sources_satellite import NO2  # noqa: E402
from tests.test_harness_workflow_packs_run import JUNE, PackRun  # noqa: E402
from tta_backend.harness.handles.model import Descriptor, HandleRecord, ServicePayload  # noqa: E402
from tta_backend.harness.results import Mode, Result  # noqa: E402
from tta_backend.harness.testing import FakeEarthdata, env_factory  # noqa: E402
from tta_backend.harness.trace import NewEntry, Run, new_run_id  # noqa: E402
from tta_backend.services.artifact_store import ArtifactStore  # noqa: E402
from tta_backend.services.chart_service import ChartLink, ChartService  # noqa: E402
from tta_backend.services.harness_chat import StoredArtifacts  # noqa: E402
from tta_backend.services.result_export import (  # noqa: E402
    INTERPRETATION_NOTE,
    NO_DATA_SERVICE,
    ResultExporter,
    ResultNotFound,
    chart_citations,
    chart_provenance,
    rule_text,
)

CITATION = {
    "handle": "dataset_tempo", "doi": "10.5067/TEMPO",
    "collection_citations": [{"Creator": "NASA", "Title": "TEMPO NO2", "Publisher": "ASDC", "ReleaseDate": "2024"}],
}


def data_service() -> FakeEarthdata:
    return FakeEarthdata({
        "get_provenance": lambda args: {
            "handle": args["handle"], "ancestors": [{"handle": "dataset_tempo", "depth": 1}],
            "events": [{"event_type": "materialized", "detail": {}, "created_at": "2024-06-04T00:00:00Z"}],
        },
        "cite_dataset": lambda args: CITATION,
    })


class ExportCase(PackRun):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(MemoryTranscripts(), self.charts))
        self.frames: list[Any] = []
        self.chart_service = ChartService()
        artifacts = StoredArtifacts(
            self.chart_service, "conv-1", "user-a", self.frames.append, self.f.stores.traces,
        )
        self.f.context = dataclasses.replace(self.f.context, artifacts=artifacts)
        self.tables = ArtifactStore()
        self.earthdata = data_service()
        # The runs' derived payloads go where the exporter reads them.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

        async def open_retrieval(user_id: str, record: Any) -> Any:
            return self.service.files[record.id]

        self.f.build = env_factory(
            self.root, earthdata=self.service.earthdata(), open_retrieval=open_retrieval, places=self.f.places,
        )

    def exporter(self, earthdata: Any = "default") -> ResultExporter:
        return ResultExporter(
            self.f.stores, chart_service=self.chart_service, artifacts=self.tables, payload_root=self.root,
            earthdata=self.earthdata if earthdata == "default" else earthdata,
        )

    async def charted_run(self, name: str, inputs: dict[str, Any]) -> tuple[Result, dict[str, Any], ChartLink]:
        result, _ = await self.run_workflow(name, inputs)
        [chart_id] = result.charts
        payload = self.charts.rows[chart_id]
        link = ChartLink.of(payload)
        assert link is not None
        return result, payload, link


class MapExportTests(ExportCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.result, self.payload, self.link = await self.charted_run("map", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})

    async def test_the_stored_chart_is_linked_to_its_run_mode_and_retrieval_handle(self):
        self.assertEqual(self.link.run_id, self.result.trace_id)
        self.assertEqual(self.link.mode, Mode.draft("map", 1))
        self.assertEqual(self.link.handle, "obs_1")
        self.assertEqual(self.payload["metadata"]["run_id"], self.result.trace_id)

    async def test_methods_open_with_the_mode_and_give_a_sentence_per_step(self):
        markdown = await self.exporter().methods("user-a", self.result.trace_id)

        lines = markdown.splitlines()
        self.assertEqual(lines[0], "# Methods: Draft workflow: map v1")
        self.assertIn("**Mode:** Draft workflow map, version 1. The workflow has not been validated.", markdown)
        entries = await self.f.stores.traces.entries("user-a", self.result.trace_id)
        methods = [e.result["method"] for e in entries if e.status == "done" and e.kind == "use"]
        self.assertEqual(len(methods), 3)
        for number, method in enumerate(methods, 1):
            self.assertTrue(any(line.startswith(f"{number}. ") and line.endswith(method) for line in lines), method)
        self.assertNotIn("did not complete", markdown)
        self.assertIn(f"- TEMPO_NO2: {rule_text('TEMPO_NO2', 'highest_rank')}", markdown)
        self.assertIn("## Masking", markdown)
        self.assertIn("- `obs_1`:", markdown)
        self.assertIn("- `obs_1` (grid, TEMPO_NO2): ", markdown)
        self.assertIn("- NASA, TEMPO NO2, ASDC, 2024, doi:10.5067/TEMPO", markdown)
        self.assertNotIn("Interpretation", markdown)

    async def test_the_provenance_file_is_machine_readable(self):
        provenance = await self.exporter().provenance("user-a", self.result.trace_id)

        self.assertEqual(provenance["run_id"], self.result.trace_id)
        self.assertEqual(provenance["workflow"], {"name": "map", "version": 1})
        self.assertEqual(provenance["mode"]["label"], "Draft workflow: map v1")
        self.assertEqual(provenance["harness_release"], self.result.harness_release)
        [product] = provenance["products"]
        self.assertEqual((product["key"], product["rule"]), ("TEMPO_NO2", "highest_rank"))
        self.assertEqual([s["id"] for s in provenance["sources"]], ["obs_1", *[
            s["id"] for s in provenance["sources"][1:]]])
        self.assertIn("obs_1", [m["handle"] for m in provenance["masking"]])
        entries = await self.f.stores.traces.entries("user-a", self.result.trace_id)
        self.assertEqual([o["seq"] for o in provenance["operations"]], [e.seq for e in entries])
        self.assertEqual(provenance["citations"], [CITATION])
        self.assertEqual(provenance["figures"], list(self.result.charts))
        self.assertIsNone(provenance["interpretation"])
        json.dumps(provenance)

    async def test_the_archive_holds_the_figure_the_methods_and_the_provenance(self):
        content = await self.exporter().archive("user-a", self.result.trace_id)

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            self.assertIn("methods.md", names)
            self.assertIn("provenance.json", names)
            [figure] = [n for n in names if n.startswith("figures/")]
            self.assertTrue(figure.endswith(".png"))
            self.assertTrue(archive.read(figure).startswith(b"\x89PNG"))
            self.assertEqual(archive.read("methods.md").decode(), await self.exporter().methods("user-a", self.result.trace_id))

    async def test_another_users_run_is_not_found(self):
        for call in ("methods", "provenance", "archive"):
            with self.subTest(call), self.assertRaises(ResultNotFound):
                await getattr(self.exporter(), call)("user-b", self.result.trace_id)
        with self.assertRaises(ResultNotFound):
            await self.exporter().methods("user-a", "not-a-run")

    async def test_chart_provenance_names_the_run_the_mode_the_product_and_rule_and_the_handle(self):
        view = await chart_provenance(self.f.stores, "user-a", self.link, self.earthdata)

        self.assertEqual(view["run_id"], self.result.trace_id)
        self.assertEqual(view["mode"], Mode.draft("map", 1).view())
        self.assertEqual(view["handle"], "obs_1")
        self.assertEqual(view["product"], {
            "key": "TEMPO_NO2", "rule": "highest_rank",
            "rule_text": rule_text("TEMPO_NO2", "highest_rank"),
        })
        self.assertIn("dataset_tempo", [node["handle"] for node in view["nodes"]])
        self.assertNotIn("note", view)
        self.assertEqual(await chart_citations(self.f.stores, "user-a", self.link, self.earthdata), [CITATION])

    async def test_a_chart_drawn_from_several_handles_walks_all_of_them(self):
        stores, now = self.f.stores, self.f.context.clock.now()
        await stores.handles.record_retrieval(HandleRecord(
            id="obs_9", kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
            descriptor=Descriptor(summary="TROPOMI_NO2 over New Jersey", product="TROPOMI_NO2"),
            created_at=now, payload=ServicePayload(),
        ))
        other = new_run_id()
        await stores.traces.start_run(Run(
            id=other, user_id="user-a", conversation_id="conv-1", mode=Mode.exploratory(), started=now,
        ))
        await stores.traces.append("user-a", NewEntry(
            run_id=other, kind="use", status="done", started=now, seconds=1.0,
            capability="sources.satellite.get_data", params={},
            result={"status": "done", "handle": "obs_9", "product": "TROPOMI_NO2", "values": {"rule": "user_named"}},
        ))
        await stores.traces.finish_run("user-a", other, "done", now)
        tropomi = {**CITATION, "handle": "dataset_tropomi", "doi": "10.5067/TROPOMI"}
        earthdata = FakeEarthdata({
            "get_provenance": lambda args: {
                "handle": args["handle"], "events": [],
                "ancestors": [{"handle": "dataset_tropomi" if args["handle"] == "obs_9" else "dataset_tempo", "depth": 1}],
            },
            "cite_dataset": lambda args: tropomi if args["dataset_handle"] == "dataset_tropomi" else CITATION,
        })
        link = ChartLink(run_id=self.link.run_id, mode=self.link.mode, handle="obs_1", handles=("obs_1", "obs_9"))

        view = await chart_provenance(stores, "user-a", link, earthdata)

        self.assertEqual(view["handles"], ["obs_1", "obs_9"])
        nodes = [node["handle"] for node in view["nodes"]]
        for handle in ("obs_1", "obs_9", "dataset_tempo", "dataset_tropomi"):
            self.assertIn(handle, nodes)
        self.assertEqual(view["product"]["key"], "TEMPO_NO2")
        self.assertEqual(view["products"], [
            {"key": "TEMPO_NO2", "rule": "highest_rank", "rule_text": rule_text("TEMPO_NO2", "highest_rank")},
            {"key": "TROPOMI_NO2", "rule": "user_named", "rule_text": rule_text("TROPOMI_NO2", "user_named")},
        ])
        self.assertEqual(await chart_citations(stores, "user-a", link, earthdata), [CITATION, tropomi])

    async def test_a_chart_link_keeps_every_source_handle(self):
        stored = ChartLink(run_id=self.link.run_id, mode=self.link.mode, handle="obs_1", handles=("obs_1", "obs_9"))
        self.assertEqual(ChartLink.of({"metadata": {"harness": stored.to_json()}}), stored)
        # A link written with only its first handle still reads.
        only_first = {"run_id": self.link.run_id, "mode": self.link.mode.model_dump(mode="json"), "handle": "obs_1"}
        self.assertEqual(ChartLink.of({"metadata": {"harness": only_first}}).handles, ("obs_1",))

    async def test_a_chart_no_longer_stored_is_named_as_a_figure_in_the_archive(self):
        del self.charts.rows[self.result.charts[0]]
        content = await self.exporter().archive("user-a", self.result.trace_id)

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            left_out = archive.read("data/NOT_INCLUDED.txt").decode()
        self.assertIn(f"Figure {self.result.charts[0]}: it is no longer stored.", left_out)
        self.assertNotIn("Table", left_out)

    async def test_without_the_data_service_the_documents_say_what_is_left_out(self):
        markdown = await self.exporter(earthdata=None).methods("user-a", self.result.trace_id)
        self.assertIn(NO_DATA_SERVICE, markdown)
        self.assertNotIn("### Citations", markdown)
        view = await chart_provenance(self.f.stores, "user-a", self.link, None)
        self.assertEqual(view["note"], NO_DATA_SERVICE)
        self.assertEqual(view["product"]["key"], "TEMPO_NO2")


class TimeseriesExportTests(ExportCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.result, self.payload, self.link = await self.charted_run(
            "timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE},
        )

    async def test_the_chart_renders_a_derived_series_whose_product_rule_is_found_behind_it(self):
        self.assertTrue(self.link.handle and self.link.handle.startswith("series_"))
        view = await chart_provenance(self.f.stores, "user-a", self.link, self.earthdata)
        self.assertEqual(view["handle"], self.link.handle)
        self.assertEqual((view["product"]["key"], view["product"]["rule"]), ("TEMPO_NO2", "highest_rank"))
        handles = [node["handle"] for node in view["nodes"]]
        self.assertIn(self.link.handle, handles)
        self.assertIn("dataset_tempo", handles)

    async def test_the_archive_holds_the_figure_and_the_series_as_csv(self):
        content = await self.exporter().archive("user-a", self.result.trace_id)

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            [figure] = [n for n in names if n.startswith("figures/")]
            self.assertTrue(archive.read(figure).startswith(b"\x89PNG"))
            data = [n for n in names if n.startswith("data/") and n.endswith(".csv")]
            self.assertTrue(data)
            first = archive.read(data[0]).decode().splitlines()
            self.assertEqual(first[0], "series,time,value,units,handle")
            self.assertEqual(len(first) - 1, len(self.payload["times"]))
            for name in data[1:]:
                self.assertIn("time", archive.read(name).decode().splitlines()[0])
        provenance = await self.exporter().provenance("user-a", self.result.trace_id)
        derived = [h for h in provenance["handles"] if h["origin"] == "derived"]
        self.assertIn(self.link.handle, [h["id"] for h in derived])
        self.assertTrue(all(h["operation"] for h in derived))


class InterpretationTests(ExportCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

    async def test_a_model_written_interpretation_is_its_own_section_marked_as_such(self):
        run_id = new_run_id()
        stores = self.f.stores
        await stores.traces.start_run(Run(
            id=run_id, user_id="user-a", conversation_id="conv-1", mode=Mode.exploratory(),
            started=self.f.context.clock.now(),
        ))
        await stores.results.put("user-a", Result(
            status="incomplete", mode=Mode.exploratory(), trace_id=run_id, reason="The exploration ran out of budget.",
            interpretation="NO2 looks higher in the north.",
        ))

        markdown = await self.exporter().methods("user-a", run_id)

        self.assertIn("**Mode:** Exploratory.", markdown)
        self.assertIn("Status: incomplete. The exploration ran out of budget.", markdown)
        head, _, interpretation = markdown.partition("## Interpretation (model-written)")
        self.assertNotIn("NO2 looks higher", head)
        self.assertIn(INTERPRETATION_NOTE, interpretation)
        self.assertIn("NO2 looks higher in the north.", interpretation)
        provenance = await self.exporter().provenance("user-a", run_id)
        self.assertEqual(provenance["interpretation"], {"text": "NO2 looks higher in the north.", "written_by": "model"})
        self.assertIsNone(provenance["workflow"])


class FormulaCellTests(ExportCase):
    async def test_text_a_spreadsheet_would_run_as_a_formula_is_quoted_and_numbers_are_left_alone(self):
        from tta_backend.services.payload_figures import csv_bytes

        rows = [["site", "value", "note"], ["=HYPERLINK(\"x\")", -1.5, "+1 more"], ["@sum", "-2.5", "\tTab"],
                ["-cmd", 3, "plain"]]
        import csv

        read = list(csv.reader(io.StringIO(csv_bytes(rows).decode())))
        self.assertEqual(read, [
            ["site", "value", "note"], ["'=HYPERLINK(\"x\")", "-1.5", "'+1 more"], ["'@sum", "-2.5", "'\tTab"],
            ["'-cmd", "3", "plain"],
        ])

    async def test_a_derived_tables_rows_in_the_archive_are_quoted_the_same_way(self):
        import pandas as pd

        from tta_backend.services.result_export import _frame_rows

        rows = _frame_rows(pd.DataFrame({"name": ["=1+1", "Trenton"], "value": [-3.0, 4.0]}))
        from tta_backend.services.payload_figures import csv_bytes

        self.assertEqual(csv_bytes(rows or []).decode().splitlines(), ["name,value", "'=1+1,-3.0", "Trenton,4.0"])

    async def test_a_stored_tables_csv_is_quoted_the_same_way(self):
        ref = self.tables.put_table("Sites", ["site", "value"], [{"site": "=cmd|' /C calc'!A0", "value": -1}])
        await self.tables.claim(ref.id, "user-a", "conv-1")
        chunks = [chunk async for chunk in self.tables.iter_csv_chunks(ref.id, "user-a")]
        self.assertEqual(b"".join(chunks).decode().splitlines(), ["site,value", "'=cmd|' /C calc'!A0,-1"])
