"""
tests/test_satellite_tools_masking_execution.py
==================================================
T25 masking-execution fix: the honesty-guard commit (321d507) proved that
before this fix, every real tool path opened a Dataset, extracted just the
science DataArray, and lost the sibling QA-flag variable before it ever
reached AggregationService.aggregate() -- so no tool actually ran QA masking
despite collections.yaml pinning qa_good_values for TEMPO_NO2/TEMPO_HCHO/etc.
The existing test_aggregation_service.py unit tests hid this gap by passing
a full Dataset straight to aggregate(), a shape no production tool call
ever takes.

These are integration tests at the tool layer, mirroring production shape:
open a real Dataset (science var + sibling QA-flag var) through the same
HandleVolume/open_handle seam every tool uses, call the actual plot/stat/
compare tool, and assert (a) bad-quality-flagged pixels are actually
dropped from the computed result, not just from provenance metadata, and
(b) the reported qa_status truthfully says "verified" (a pinned collections.
yaml rule) rather than the honesty guard's "not applied" downgrade.

The registry match is driven by the opened Dataset's global ``short_name``
attribute (datasets/mask_info.py::col_info_for_short_name) -- "TEMPO_NO2_L3"
matches collections.yaml's TEMPO_NO2 entry (quality_flag_var=
main_data_quality_flag, qa_good_values=[0]).
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

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]


def _tempo_no2_dataset(xr, values, flags, lat=(10.0, 20.0), lon=(30.0, 40.0), time=None):
    """A TEMPO_NO2-shaped Dataset: science var + sibling QA-flag var, with
    the ``short_name`` global attribute col_info_for_short_name matches
    against collections.yaml's TEMPO_NO2 entry."""
    if time is None:
        data_vars = {
            "vertical_column_troposphere": (("lat", "lon"), values, {"units": "molecules/cm^2"}),
            "main_data_quality_flag": (("lat", "lon"), flags),
        }
        coords = {"lat": list(lat), "lon": list(lon)}
    else:
        import numpy as np

        data_vars = {
            "vertical_column_troposphere": (("time", "lat", "lon"), values, {"units": "molecules/cm^2"}),
            "main_data_quality_flag": (("time", "lat", "lon"), flags),
        }
        coords = {"time": np.array(list(time), dtype="datetime64[ns]"), "lat": list(lat), "lon": list(lon)}
    return xr.Dataset(data_vars, coords=coords, attrs={"short_name": "TEMPO_NO2_L3"})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "masking-execution integration test dependencies are not installed",
)
class MaskingExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)
        self._align_handler = None

        async def _align(source_handles, method="outer", workspace_id="default"):
            return await self._align_handler(source_handles)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
            "align": _align,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def test_plot_singular_drops_bad_flag_pixels_and_reports_verified_qa(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import make_plot_singular

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        plot_singular = make_plot_singular(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        masking = full["provenance"]["masking"]
        self.assertEqual(masking["qa_status"], "verified")
        self.assertEqual(masking["qa_source"], "collections_yaml")

        # Bad-flag pixels (lon=40.0 column, flag=1) are actually dropped from
        # the rendered grid -- not just disclosed in provenance.
        flat_values = [v for row in full["values"] for v in row if v is not None]
        self.assertTrue(all(v in (1.0, 3.0) for v in flat_values), flat_values)

    async def test_compute_statistic_tool_excludes_bad_flag_pixels_from_the_mean(self):
        import xarray as xr
        from tools.satellite_tools.stat_tools import make_compute_statistic_tool

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_1", "location": "global", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        # Good cells (flag=0): 1.0, 3.0 -> mean 2.0. Unmasked mean would be 2.5.
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertEqual(result["n_pixels"], 2)
        self.assertEqual(result["aggregation_meta"]["masking"]["qa_status"], "verified")

    async def test_find_daily_peak_excludes_a_bad_flag_pixel_even_though_it_is_numerically_highest(self):
        import xarray as xr
        from tools.satellite_tools.stat_tools import make_find_daily_peak

        def make_ds():
            # The numerically highest raw value (99.0) carries a bad flag;
            # the true peak once masked is the good-flag 3.0 cell.
            return _tempo_no2_dataset(
                xr, values=[[1.0, 99.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        find_daily_peak = make_find_daily_peak(self.mcp_tools)
        raw = await find_daily_peak.ainvoke({"handle": "obs_1", "location": "global"})
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["peak_value"], 3.0)
        self.assertEqual(result["aggregation_meta"]["masking"]["qa_status"], "verified")

    async def test_conduct_temporal_statistic_masks_every_time_step_and_reports_verified_qa(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
                time=["2024-01-01", "2024-01-02"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        # Good cells per step (flag=0): step0 -> [1.0, 3.0] mean=2.0;
        # step1 -> [5.0, 7.0] mean=6.0. Unmasked means would be 2.5/6.5.
        self.assertEqual(full["values"], [2.0, 6.0])
        self.assertEqual(full["masking"]["qa_status"], "verified")
        self.assertEqual(full["masking"]["qa_source"], "collections_yaml")

    async def test_compare_region_mode_masks_bad_flag_pixels_on_both_sides(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_a():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        def make_b():
            return _tempo_no2_dataset(
                xr, values=[[10.0, 20.0], [30.0, 40.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_a", make_a)
        self.volume.add_zarr("obs_b", make_b)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await compare.ainvoke({
                "handle_a": "obs_a", "handle_b": "obs_b", "mode": "region",
                "label_a": "A", "label_b": "B",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        full = emitted["payload"]
        # Good cells (flag=0) only: A -> [1.0, 3.0] mean=2.0 (unmasked 2.5);
        # B -> [10.0, 30.0] mean=20.0 (unmasked 25.0).
        self.assertAlmostEqual(full["stats"]["A"]["mean"], 2.0)
        self.assertAlmostEqual(full["stats"]["B"]["mean"], 20.0)


if __name__ == "__main__":
    unittest.main()
