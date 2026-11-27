"""The result export endpoints and the chart endpoints for a chart a harness run drew, through the app."""
from __future__ import annotations

import datetime
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from typing import Any

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from harness_chat_helpers import MemoryCharts, MemoryTranscripts, PlaybackSupervisor, make_chat, memory_repositories  # noqa: E402
from tta_backend.earthdata_mcp.connection import STATE_READY  # noqa: E402
from tta_backend.harness.handles.model import Descriptor, HandleRecord, ServicePayload  # noqa: E402
from tta_backend.harness.results import Mode, Result  # noqa: E402
from tta_backend.harness.storage import memory_stores  # noqa: E402
from tta_backend.harness.trace import NewEntry, Run, new_run_id  # noqa: E402
from tta_backend.services.chart_service import ChartLink  # noqa: E402
from tta_backend.services.result_export import NO_DATA_SERVICE, rule_text  # noqa: E402

_REQUIRED = ["fastapi", "httpx", "jwt"]
NOW = datetime.datetime(2024, 7, 1, 12, tzinfo=datetime.UTC)
SERIES = "series_" + "1" * 32


@unittest.skipIf(
    any(importlib.util.find_spec(module) is None for module in _REQUIRED),
    "endpoint dependencies are not installed",
)
class ResultExportEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import httpx

        import tta_backend.api as api

        self.httpx = httpx
        self.api = api
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stores = memory_stores()
        chat = make_chat(PlaybackSupervisor(), tmp.name, stores=self.stores)
        chat.payload_root = tmp.name
        api.app.state.agent = chat
        self.addCleanup(setattr, api.app.state, "agent", None)
        # The data service is connected but offers no lineage or citation tool, so both are left out and the
        # documents say so.
        previous = getattr(api.app.state, "earthdata_mcp_manager", None)
        api.app.state.earthdata_mcp_manager = SimpleNamespace(state=STATE_READY, tools={})
        self.addCleanup(setattr, api.app.state, "earthdata_mcp_manager", previous)
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(MemoryTranscripts(), self.charts))
        self.enterContext(auth_helpers.patch_verifier())
        self.owner = {"Authorization": f"Bearer {auth_helpers.make_token('user-1')}"}
        self.other = {"Authorization": f"Bearer {auth_helpers.make_token('user-2')}"}
        self.run_id = await self.seed_run()

    async def seed_run(self) -> str:
        run_id = new_run_id()
        mode = Mode.validated("map", 2)
        await self.stores.traces.start_run(Run(
            id=run_id, user_id="user-1", conversation_id="thread-1", mode=mode, started=NOW,
        ))
        await self.stores.handles.record_retrieval(HandleRecord(
            id="obs_1", kind="grid", origin="retrieval", user_id="user-1", conversation_id="thread-1",
            descriptor=Descriptor(summary="TEMPO_NO2 over New Jersey", product="TEMPO_NO2",
                                  masking={"variable": "vertical_column_troposphere"}),
            created_at=NOW, payload=ServicePayload(),
        ))
        await self.stores.traces.append("user-1", NewEntry(
            run_id=run_id, kind="use", status="done", started=NOW, seconds=1.0,
            capability="sources.satellite.get_data", step="data", params={"area": "obs_1"},
            result={"status": "done", "handle": "obs_1", "kind": "grid", "product": "TEMPO_NO2",
                    "values": {"rule": "highest_rank"}, "method": "TEMPO_NO2 was retrieved over New Jersey."},
        ))
        await self.stores.traces.finish_run("user-1", run_id, "done", NOW)
        await self.stores.results.put("user-1", Result(status="done", mode=mode, trace_id=run_id, handles=("obs_1",)))
        return run_id

    async def seed_chart(self, chart_type: str, handle: str, **payload: Any) -> str:
        link = ChartLink(run_id=self.run_id, mode=Mode.validated("map", 2), handle=handle)
        stored = await self.charts.save_chart("thread-1", {
            "type": chart_type, "title": f"A {chart_type}",
            "metadata": {"source_handles": [handle], "run_id": self.run_id, "harness": link.to_json()},
            **payload,
        }, "user-1")
        return str(stored["chart_id"])

    async def get(self, path: str, headers: dict[str, str]) -> Any:
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path, headers=headers)

    async def test_the_owner_gets_the_methods_the_provenance_and_the_archive(self):
        methods = await self.get(f"/result/{self.run_id}/methods.md", self.owner)
        self.assertEqual(methods.status_code, 200, methods.text)
        self.assertTrue(methods.headers["content-type"].startswith("text/markdown"))
        self.assertTrue(methods.text.startswith("# Methods: map v2"))
        self.assertIn("**Mode:** Validated workflow map, version 2.", methods.text)
        self.assertIn("1. `data`: TEMPO_NO2 was retrieved over New Jersey.", methods.text)
        self.assertIn(NO_DATA_SERVICE, methods.text)

        provenance = await self.get(f"/result/{self.run_id}/provenance.json", self.owner)
        self.assertEqual(provenance.status_code, 200)
        body = provenance.json()
        self.assertEqual((body["run_id"], body["workflow"]), (self.run_id, {"name": "map", "version": 2}))
        self.assertEqual(body["products"], [{"key": "TEMPO_NO2", "rule": "highest_rank",
                                             "rule_text": rule_text("TEMPO_NO2", "highest_rank")}])

        archive = await self.get(f"/result/{self.run_id}/export.zip", self.owner)
        self.assertEqual(archive.status_code, 200)
        self.assertEqual(archive.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as opened:
            self.assertEqual(opened.read("methods.md").decode(), methods.text)
            self.assertIn("provenance.json", opened.namelist())

    async def test_another_users_run_and_a_malformed_id_are_404(self):
        for path in ("methods.md", "provenance.json", "export.zip"):
            with self.subTest(path):
                self.assertEqual((await self.get(f"/result/{self.run_id}/{path}", self.other)).status_code, 404)
                self.assertEqual((await self.get(f"/result/run_nope/{path}", self.owner)).status_code, 404)

    async def test_a_harness_time_series_exports_csv_and_png_from_its_payload(self):
        chart_id = await self.seed_chart(
            "timeseries", SERIES, times=["2024-06-01", "2024-06-02"], values=[1.0, 2.0], units="ppb",
            series=[{"label": "NO2", "units": "ppb", "axis": "y", "times": ["2024-06-01", "2024-06-02"],
                     "values": [1.0, 2.0], "handle": SERIES}],
        )

        csv = await self.get(f"/chart/{chart_id}/export.csv", self.owner)
        self.assertEqual(csv.status_code, 200, csv.text)
        self.assertEqual(csv.text.splitlines(), [
            "series,time,value,units,handle", f"NO2,2024-06-01,1.0,ppb,{SERIES}", f"NO2,2024-06-02,2.0,ppb,{SERIES}",
        ])
        png = await self.get(f"/chart/{chart_id}/export.png", self.owner)
        self.assertEqual(png.status_code, 200)
        self.assertTrue(png.content.startswith(b"\x89PNG"))
        netcdf = await self.get(f"/chart/{chart_id}/export.nc", self.owner)
        self.assertEqual(netcdf.status_code, 409)
        self.assertIn(f"/result/{self.run_id}/export.zip", netcdf.json()["detail"])
        self.assertEqual((await self.get(f"/chart/{chart_id}/export.csv", self.other)).status_code, 404)

    async def test_while_the_data_service_is_down_a_harness_charts_csv_and_citations_are_still_served(self):
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="connecting", tools={})
        chart_id = await self.seed_chart(
            "timeseries", SERIES, times=["2024-06-01"], values=[1.0], units="ppb",
            series=[{"label": "NO2", "units": "ppb", "times": ["2024-06-01"], "values": [1.0], "handle": SERIES}],
        )

        csv = await self.get(f"/chart/{chart_id}/export.csv", self.owner)
        self.assertEqual(csv.status_code, 200, csv.text)
        self.assertEqual(csv.text.splitlines()[0], "series,time,value,units,handle")
        citations = await self.get(f"/chart/{chart_id}/citations", self.owner)
        self.assertEqual(citations.status_code, 200, citations.text)
        self.assertEqual(citations.json(), {"citations": [], "note": NO_DATA_SERVICE})
        netcdf = await self.get(f"/chart/{chart_id}/export.nc", self.owner)
        self.assertEqual(netcdf.status_code, 409)
        # A chart that needs the data service still says it is unavailable.
        plain = await self.charts.save_chart("thread-1", {
            "type": "heatmap", "title": "Not a harness chart", "export": {"type": "heatmap"},
            "metadata": {"source_handles": ["obs_1"]},
        }, "user-1")
        self.assertEqual((await self.get(f"/chart/{plain['chart_id']}/export.csv", self.owner)).status_code, 503)
        self.assertEqual((await self.get(f"/chart/{plain['chart_id']}/citations", self.owner)).status_code, 503)

    async def test_a_deleted_conversations_results_are_not_found(self):
        from tta_backend.harness.deletion import delete_conversation

        self.assertEqual((await self.get(f"/result/{self.run_id}/methods.md", self.owner)).status_code, 200)
        await delete_conversation(self.stores, None, "user-1", "thread-1")

        for path in ("methods.md", "provenance.json", "export.zip"):
            with self.subTest(path):
                self.assertEqual((await self.get(f"/result/{self.run_id}/{path}", self.owner)).status_code, 404)

    async def test_a_harness_map_draws_a_png_and_refuses_csv_with_a_reason(self):
        chart_id = await self.seed_chart(
            "heatmap", "obs_1", lats=[40.0, 41.0], lons=[-75.0, -74.0], values=[[1.0, None], [2.0, 3.0]],
            vmin=1.0, vmax=3.0, units="ppb",
        )

        png = await self.get(f"/chart/{chart_id}/export.png", self.owner)
        self.assertEqual(png.status_code, 200)
        self.assertTrue(png.content.startswith(b"\x89PNG"))
        csv = await self.get(f"/chart/{chart_id}/export.csv", self.owner)
        self.assertEqual(csv.status_code, 409)
        self.assertIn("no CSV export", csv.json()["detail"])

    async def test_a_harness_charts_provenance_and_methods_come_from_its_run(self):
        chart_id = await self.seed_chart("heatmap", "obs_1", lats=[40.0], lons=[-75.0], values=[[1.0]])

        response = await self.get(f"/chart/{chart_id}/provenance", self.owner)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["run_id"], self.run_id)
        self.assertEqual(body["mode"], {"kind": "validated", "label": "map v2", "workflow": {"name": "map", "version": 2}})
        self.assertEqual(body["handle"], "obs_1")
        self.assertEqual(body["product"], {"key": "TEMPO_NO2", "rule": "highest_rank",
                                           "rule_text": rule_text("TEMPO_NO2", "highest_rank")})
        self.assertIn("nodes", body)
        methods = await self.get(f"/chart/{chart_id}/methods.md", self.owner)
        self.assertEqual(methods.status_code, 200)
        self.assertTrue(methods.text.startswith("# Methods: map v2"))
        self.assertEqual((await self.get(f"/chart/{chart_id}/provenance", self.other)).status_code, 404)
        # Overlay and frames were never stored for a harness map; each says so.
        overlay = await self.get(f"/chart/{chart_id}/overlay.png", self.owner)
        self.assertEqual(overlay.status_code, 404)
        self.assertIn("no rendered overlay", overlay.json()["detail"])
        frames = await self.get(f"/chart/{chart_id}/frames.f32.gz", self.owner)
        self.assertEqual(frames.status_code, 404)
        self.assertIn("no stored frame stack", frames.json()["detail"])


if __name__ == "__main__":
    unittest.main()
