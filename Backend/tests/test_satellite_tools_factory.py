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

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite tools factory test dependencies are not installed",
)
class SatelliteToolsFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    def _tool(self, name):
        from tools.satellite_tools.factory import build_satellite_tools

        tools = {t.name: t for t in build_satellite_tools(self.mcp_tools)}
        return tools[name]

    async def test_plot_singular_produces_a_heatmap_artifact_from_a_handle(self):
        import xarray as xr

        def make_dataset():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_1", make_dataset)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        plot_singular = self._tool("plot_singular")
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            result = await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})
        payload = json.loads(result)

        self.assertNotIn("error", payload)

        # The full grid still reaches the frontend chart/artifact pipeline.
        full = emitted["payload"]
        self.assertEqual(full["type"], "heatmap")
        self.assertEqual(full["variable"], "no2")
        self.assertEqual(full["units"], "mol/m^2")
        self.assertEqual(full["metadata"]["source_handles"], ["obs_1"])

        # The model-facing result is the compact summary (T13).
        self.assertEqual(payload["render_type"], "heatmap")
        self.assertEqual(payload["variable"], "no2")
        self.assertEqual(payload["units"], "mol/m^2")
        self.assertEqual(payload["source_handles"], ["obs_1"])
        self.assertNotIn("values", payload)
        self.assertNotIn("lats", payload)
        self.assertTrue(payload["chart_id"].startswith("map_"))
        ref = payload["_artifact_refs"][0]
        self.assertEqual(ref["id"], payload["chart_id"])
        self.assertEqual(ref["type"], "map")
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_1"])

    async def test_plot_multiple_produces_a_multi_panel_artifact_from_handles(self):
        import xarray as xr

        def make_dataset():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_a", make_dataset)
        self.volume.add_zarr("obs_b", make_dataset)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        plot_multiple = self._tool("plot_multiple")
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            result = await plot_multiple.ainvoke({
                "handles": ["obs_a", "obs_b"],
                "locations": ["global", "global"],
            })
        payload = json.loads(result)

        self.assertNotIn("error", payload)

        full = emitted["payload"]
        self.assertEqual(full["type"], "heatmap_multi")
        self.assertEqual(len(full["panels"]), 2)
        self.assertEqual(full["metadata"]["source_handles"], ["obs_a", "obs_b"])

        self.assertEqual(payload["render_type"], "heatmap_multi")
        self.assertEqual(payload["source_handles"], ["obs_a", "obs_b"])
        self.assertNotIn("panels", payload)
        self.assertTrue(payload["chart_id"].startswith("cmp_"))
        ref = payload["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual([p["handle"] for p in ref["metadata"]["panels"]], ["obs_a", "obs_b"])

    async def test_conduct_temporal_statistic_produces_a_timeseries_artifact_from_a_handle(self):
        import numpy as np
        import xarray as xr

        def make_dataset():
            return xr.Dataset(
                {
                    "no2": (
                        ("time", "lat", "lon"),
                        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                        {"units": "mol/m^2"},
                    )
                },
                coords={
                    "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                    "lat": [10.0, 20.0],
                    "lon": [30.0, 40.0],
                },
            )

        self.volume.add_zarr("obs_ts", make_dataset)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = self._tool("conduct_temporal_statistic")
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            result = await conduct_temporal_statistic.ainvoke({"handle": "obs_ts", "location": "global", "stat": "mean"})
        payload = json.loads(result)

        self.assertNotIn("error", payload)

        full = emitted["payload"]
        self.assertEqual(full["type"], "timeseries")
        self.assertEqual(len(full["times"]), 2)
        self.assertEqual(full["metadata"]["source_handles"], ["obs_ts"])

        self.assertEqual(payload["render_type"], "timeseries")
        self.assertEqual(payload["grid_dims"], [2])
        self.assertEqual(payload["source_handles"], ["obs_ts"])
        self.assertNotIn("times", payload)
        self.assertTrue(payload["chart_id"].startswith("ts_"))
        ref = payload["_artifact_refs"][0]
        self.assertEqual(ref["type"], "timeseries")
        self.assertEqual(ref["metadata"]["series"][0]["source_kind"], "satellite")

    async def test_compute_statistic_tool_produces_stats_from_a_handle(self):
        import xarray as xr

        def make_dataset():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_stat", make_dataset)

        compute_statistic_tool = self._tool("compute_statistic_tool")
        result = await compute_statistic_tool.ainvoke({"handle": "obs_stat", "location": "global", "stats": ["mean", "max"]})
        payload = json.loads(result)

        self.assertNotIn("error", payload)
        self.assertEqual(payload["mean"], 2.5)
        self.assertEqual(payload["max"], 4.0)
        self.assertEqual(payload["source_handles"], ["obs_stat"])

    async def test_find_daily_peak_locates_the_peak_from_a_handle(self):
        import xarray as xr

        def make_dataset():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 9.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_peak", make_dataset)

        find_daily_peak = self._tool("find_daily_peak")
        result = await find_daily_peak.ainvoke({"handle": "obs_peak", "location": "global"})
        payload = json.loads(result)

        self.assertNotIn("error", payload)
        self.assertEqual(payload["peak_value"], 9.0)
        self.assertEqual(payload["peak_lat"], 20.0)
        self.assertEqual(payload["peak_lon"], 40.0)
        self.assertEqual(payload["source_handles"], ["obs_peak"])

    async def test_factory_registers_the_t07_validation_tools(self):
        self.assertIsNotNone(self._tool("validate_against_ground"))
        self.assertIsNotNone(self._tool("exceedance_overlay"))

    async def test_factory_registers_the_t08_compare_tool(self):
        self.assertIsNotNone(self._tool("compare"))


if __name__ == "__main__":
    unittest.main()
