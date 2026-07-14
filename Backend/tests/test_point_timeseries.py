"""
tests/test_point_timeseries.py
================================
PRD T20: the model-facing point_timeseries tool (tools/satellite_tools/
retrieval_tools.py::make_point_timeseries) — drives it at the tool seam
against the fake MCP, asserting external behavior only: a chart event with
the series, a compact tool result carrying the artifact id and source
handle, jobs-panel-visible progress, an over-span refusal with no
submission, AOI reuse (no second geocoding path), and a failed job's
message surfaced verbatim.

Prior art: test_retrieval_composites.py (gate/await patterns), test_open_
handle.py (Parquet fixtures via HandleVolume), test_satellite_tools_
factory.py (chart artifact assertions).
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "pyarrow"]


def _make_series_table():
    import pyarrow as pa

    table = pa.table({
        "time": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "no2": [3.0, 1.0, 2.0],
    })
    return table.replace_schema_metadata({b"units": b"mol/m^2"})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "point_timeseries test dependencies are not installed",
)
class PointTimeseriesToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)
        self.volume.add_parquet("cube_ts_1", _make_series_table)

        self.aoi_calls = []
        self.submit_calls = []

        async def define_area_of_interest(location, workspace_id):
            self.aoi_calls.append(location)
            return {"handle": "aoi_newark", "location": location}

        async def retrieve_timeseries(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
            self.submit_calls.append({
                "dataset_handle": dataset_handle, "time_range": time_range, "variables": variables,
                "aoi_handle": aoi_handle, "point_sample": point_sample,
            })
            return {"job_handle": "job_cube_ts_1"}

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "define_area_of_interest": define_area_of_interest,
            "retrieve_timeseries": retrieve_timeseries,
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    def _tool(self):
        from tools.satellite_tools.factory import build_satellite_tools

        tools = {t.name: t for t in build_satellite_tools(self.mcp_tools)}
        return tools["point_timeseries"]

    async def test_produces_a_timeseries_chart_with_the_sampled_series(self):
        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        tool = self._tool()
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            result = await tool.ainvoke({
                "dataset_handle": "dataset_1",
                "location": "Newark, NJ",
                "time_range": "2024-01-01/2024-01-31",
                "variable": "no2",
            })
        payload = json.loads(result)

        self.assertNotIn("error", payload)

        full = emitted["payload"]
        self.assertEqual(full["type"], "timeseries")
        self.assertEqual(full["variable"], "no2")
        self.assertEqual(full["units"], "mol/m^2")
        # Sorted by time even though the fixture table is out of order.
        self.assertEqual(full["times"], ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"])
        self.assertEqual(full["values"], [1.0, 2.0, 3.0])
        self.assertEqual(full["metadata"]["source_handles"], ["cube_ts_1"])

        self.assertEqual(payload["render_type"], "timeseries")
        self.assertEqual(payload["source_handles"], ["cube_ts_1"])
        self.assertTrue(payload["chart_id"].startswith("ts_"))
        ref = payload["_artifact_refs"][0]
        self.assertEqual(ref["type"], "timeseries")
        self.assertEqual(ref["metadata"]["source_handles"], ["cube_ts_1"])

    async def test_reuses_the_same_aoi_tool_the_agent_path_uses(self):
        tool = self._tool()
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda payload: None):
            await tool.ainvoke({
                "dataset_handle": "dataset_1",
                "location": "Newark, NJ",
                "time_range": "2024-01-01/2024-01-31",
                "variable": "no2",
            })

        self.assertEqual(self.aoi_calls, ["Newark, NJ"])
        self.assertEqual(len(self.submit_calls), 1)
        self.assertEqual(self.submit_calls[0]["aoi_handle"], "aoi_newark")
        self.assertTrue(self.submit_calls[0]["point_sample"])

    async def test_emits_a_jobs_panel_visible_progress_event(self):
        import utils.streaming as streaming

        seen = []
        token = streaming._job_progress_emitter.set(lambda data: seen.append(data))
        tool = self._tool()
        try:
            with patch("tools.satellite_tools.plot_tools.emit_chart", lambda payload: None):
                await tool.ainvoke({
                    "dataset_handle": "dataset_1",
                    "location": "Newark, NJ",
                    "time_range": "2024-01-01/2024-01-31",
                    "variable": "no2",
                })
        finally:
            streaming._job_progress_emitter.reset(token)

        self.assertTrue(seen)
        self.assertEqual(seen[-1]["status"], "ready")

    async def test_refuses_an_over_span_request_without_submitting(self):
        tool = self._tool()
        result = await tool.ainvoke({
            "dataset_handle": "dataset_1",
            "location": "Newark, NJ",
            "time_range": "2020-01-01/2024-01-31",
            "variable": "no2",
        })
        payload = json.loads(result)

        self.assertIn("error", payload)
        self.assertEqual(payload["error"]["category"], "too_large")
        self.assertEqual(self.aoi_calls, [])
        self.assertEqual(self.submit_calls, [])

    async def test_surfaces_a_failed_job_message(self):
        async def failing_get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": job_handle, "status": "failed", "message": "appeears: provider rejected request"}

        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        async def define_area_of_interest(location, workspace_id):
            return {"handle": "aoi_1", "location": location}

        async def retrieve_timeseries(**kwargs):
            return {"job_handle": "job_x"}

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "define_area_of_interest": define_area_of_interest,
            "retrieve_timeseries": retrieve_timeseries,
            "get_retrieval_status": failing_get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        mcp_tools = await load_raw_mcp_tools(settings)

        from tools.satellite_tools.factory import build_satellite_tools

        tool = {t.name: t for t in build_satellite_tools(mcp_tools)}["point_timeseries"]
        result = await tool.ainvoke({
            "dataset_handle": "dataset_1",
            "location": "Newark, NJ",
            "time_range": "2024-01-01/2024-01-31",
            "variable": "no2",
        })
        payload = json.loads(result)

        # Matches await_retrieval's own contract: a failed job is passed
        # through verbatim, not raised or reshaped into an {"error": ...}
        # envelope — the model reads status/message itself.
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["message"], "appeears: provider rejected request")


if __name__ == "__main__":
    unittest.main()
