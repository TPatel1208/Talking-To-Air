"""
Tests for tools/satellite_tools/comparison_tools.py (PRD T08 — region/period
comparison workflow).

Hermetic at the analysis-tool seam: synthetic aligned/misaligned cube
fixtures exercised through the module's own helpers (prior art:
test_validation_tools.py testing validation_tools' helpers directly).
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain", "numpy", "pandas", "xarray"]
FULL_TOOL_REQUIRED_MODULES = REQUIRED_MODULES + [
    "langchain_mcp_adapters", "fastmcp", "uvicorn", "zarr", "httpx",
]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class VariableMismatchTests(unittest.TestCase):
    def test_same_variable_name_is_not_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _variable_mismatch_error

        da_a = xr.DataArray([1.0], name="no2")
        da_b = xr.DataArray([2.0], name="no2")

        self.assertIsNone(_variable_mismatch_error(da_a, da_b))

    def test_different_variable_names_are_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _variable_mismatch_error

        da_a = xr.DataArray([1.0], name="no2")
        da_b = xr.DataArray([2.0], name="hcho")

        error = _variable_mismatch_error(da_a, da_b)

        self.assertIsNotNone(error)
        self.assertIn("no2", error)
        self.assertIn("hcho", error)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class DifferenceTests(unittest.TestCase):
    def test_difference_is_period_b_minus_period_a(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _difference

        da_a = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))
        da_b = xr.DataArray([[5.0, 5.0], [5.0, 5.0]], dims=("lat", "lon"))

        diff = _difference(da_a, da_b)

        self.assertEqual(diff.values.tolist(), [[4.0, 3.0], [2.0, 1.0]])

    def test_a_cell_missing_on_either_side_is_excluded_from_the_difference(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _difference

        da_a = xr.DataArray([1.0, np.nan, 3.0], dims=("x",))
        da_b = xr.DataArray([10.0, 20.0, np.nan], dims=("x",))

        diff = _difference(da_a, da_b)

        self.assertEqual(diff.values[0], 9.0)
        self.assertTrue(np.isnan(diff.values[1]))
        self.assertTrue(np.isnan(diff.values[2]))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class AnomalyStatsTests(unittest.TestCase):
    def test_mean_difference_and_percent_change_match_hand_computed_values(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

        # a: mean 2.0 -> b: mean 3.0. diff mean = 1.0, percent change = 50%.
        da_a = xr.DataArray([1.0, 2.0, 3.0], dims=("x",))
        da_b = xr.DataArray([2.0, 3.0, 4.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertEqual(stats["n_cells"], 3)
        self.assertAlmostEqual(stats["mean_difference"], 1.0)
        self.assertAlmostEqual(stats["percent_change"], 50.0)
        self.assertNotIn("area_exceeding_threshold", stats)

    def test_cells_missing_on_either_side_are_excluded_from_stats(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

        da_a = xr.DataArray([1.0, np.nan], dims=("x",))
        da_b = xr.DataArray([2.0, 5.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertEqual(stats["n_cells"], 1)
        self.assertAlmostEqual(stats["mean_difference"], 1.0)

    def test_area_exceeding_threshold_counts_cells_at_or_above_the_magnitude(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

        da_a = xr.DataArray([0.0, 0.0, 0.0, 0.0], dims=("x",))
        da_b = xr.DataArray([1.0, 2.0, 5.0, 10.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=5.0)

        self.assertEqual(stats["area_exceeding_threshold"]["n_cells"], 2)
        self.assertAlmostEqual(stats["area_exceeding_threshold"]["fraction"], 0.5)
        self.assertEqual(stats["area_exceeding_threshold"]["threshold"], 5.0)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class SplitAlignedTests(unittest.TestCase):
    def test_splits_a_two_source_aligned_cube_into_its_two_arrays_in_order(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _split_aligned

        da = xr.DataArray(
            [[[1.0, 2.0]], [[10.0, 20.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": [0, 1], "lat": [10.0], "lon": [30.0, 40.0]},
        )

        da_a, da_b = _split_aligned(da)

        self.assertEqual(da_a.values.tolist(), [[1.0, 2.0]])
        self.assertEqual(da_b.values.tolist(), [[10.0, 20.0]])

    def test_rejects_an_aligned_result_without_a_source_dimension(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _split_aligned

        da = xr.DataArray([[1.0, 2.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [30.0, 40.0]})

        with self.assertRaises(ValueError):
            _split_aligned(da)

    def test_rejects_an_aligned_result_with_the_wrong_number_of_sources(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _split_aligned

        da = xr.DataArray(
            [[[1.0]], [[2.0]], [[3.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": [0, 1, 2], "lat": [10.0], "lon": [30.0]},
        )

        with self.assertRaises(ValueError):
            _split_aligned(da)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class RegionStatsTests(unittest.TestCase):
    def test_computes_basic_stats_over_valid_cells(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _region_stats

        da = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))

        stats = _region_stats(da)

        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["n_pixels"], 4)

    def test_returns_none_when_no_valid_cells(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _region_stats

        da = xr.DataArray([np.nan, np.nan], dims=("x",))

        self.assertIsNone(_region_stats(da))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class EmptyOverlapTests(unittest.TestCase):
    def test_returns_none_when_data_has_finite_values(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _empty_overlap_error

        da = xr.DataArray([1.0, 2.0], dims=("x",))

        self.assertIsNone(_empty_overlap_error(da, "A"))

    def test_returns_an_error_naming_the_side_when_all_values_are_missing(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _empty_overlap_error

        da = xr.DataArray([np.nan, np.nan], dims=("x",))

        error = _empty_overlap_error(da, "A")

        self.assertIsNotNone(error)
        self.assertIn("A", error)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class DisjointPeriodsTests(unittest.TestCase):
    def test_overlapping_time_ranges_are_not_disjoint(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-01", periods=2)})
        da_b = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-02", periods=2)})

        self.assertIsNone(_disjoint_periods_error(da_a, da_b))

    def test_non_overlapping_time_ranges_are_rejected(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-01", periods=2)})
        da_b = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-06-01", periods=2)})

        self.assertIsNotNone(_disjoint_periods_error(da_a, da_b))

    def test_no_time_dimension_on_either_side_is_not_disjoint(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("x",))
        da_b = xr.DataArray([1.0, 2.0], dims=("x",))

        self.assertIsNone(_disjoint_periods_error(da_a, da_b))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in FULL_TOOL_REQUIRED_MODULES),
    "full compare tool test dependencies are not installed",
)
class CompareToolTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_region_mode_produces_side_by_side_panels_on_a_shared_scale(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_a():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_b():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[10.0, 20.0], [30.0, 40.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
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
                "label_a": "Newark", "label_b": "Philly",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)

        # (a) the full panel/stats detail still reaches the frontend pipeline,
        # out-of-band from the model-facing return value (T13).
        full = emitted["payload"]
        self.assertEqual(full["type"], "heatmap_multi")
        self.assertEqual(full["mode"], "n-panel")
        self.assertEqual(len(full["panels"]), 2)
        # Shared color scale across both panels (region mode never differences).
        self.assertEqual(full["panels"][0]["vmin"], full["panels"][1]["vmin"])
        self.assertEqual(full["panels"][0]["vmax"], full["panels"][1]["vmax"])
        self.assertAlmostEqual(full["stats"]["Newark"]["mean"], 2.5)
        self.assertAlmostEqual(full["stats"]["Philly"]["mean"], 25.0)
        self.assertNotIn("difference", full)

        # (b) the model-facing result is the compact summary.
        self.assertEqual(result["render_type"], "heatmap_multi")
        for bulky_key in ("panels", "stats", "mode"):
            self.assertNotIn(bulky_key, result)

        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual(ref["metadata"]["mode"], "n-panel")
        self.assertEqual([p["handle"] for p in ref["metadata"]["panels"]], ["obs_a", "obs_b"])
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_a", "obs_b"])

    async def test_period_mode_differences_b_minus_a_via_mcp_align(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_a():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_b():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[2.0, 4.0], [6.0, 8.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_aligned():
            return xr.Dataset(
                {"no2": (
                    ("source", "lat", "lon"),
                    [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]],
                    {"units": "mol/m^2"},
                )},
                coords={"source": [0, 1], "lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_june25", make_a)
        self.volume.add_zarr("obs_june26", make_b)
        self.volume.add_zarr("cube_aligned1", make_aligned)

        async def _align(source_handles):
            self.assertEqual(source_handles, ["obs_june25", "obs_june26"])
            return {"handle": "cube_aligned1", "status": "ok", "alignment_report": {"method": "outer"}}

        self._align_handler = _align

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await compare.ainvoke({
                "handle_a": "obs_june25", "handle_b": "obs_june26", "mode": "period",
                "label_a": "June 2025", "label_b": "June 2026",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)

        # (a) the full difference grid/stats still reach the frontend pipeline.
        full = emitted["payload"]
        self.assertEqual(full["mode"], "difference")
        # b - a: [[1,2],[3,4]] doubled -> diff = a itself: [[1,2],[3,4]]
        self.assertEqual(full["difference"]["values"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(full["stats"]["mean_difference"], 2.5)
        self.assertAlmostEqual(full["stats"]["percent_change"], 100.0)
        # Diverging, zero-centered scale.
        self.assertAlmostEqual(full["difference"]["vmin"], -full["difference"]["vmax"])

        # (b) the model-facing result is the compact summary, using the
        # difference grid's own dimensions/value range (T13).
        self.assertEqual(result["render_type"], "heatmap_multi")
        self.assertEqual(result["grid_dims"], [2, 2])
        self.assertAlmostEqual(result["vmin"], full["difference"]["vmin"])
        self.assertAlmostEqual(result["vmax"], full["difference"]["vmax"])
        for bulky_key in ("panels", "stats", "mode", "difference"):
            self.assertNotIn(bulky_key, result)

        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual(ref["metadata"]["mode"], "difference")
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_june25", "obs_june26", "cube_aligned1"])

    async def test_mismatched_variables_are_rejected_with_a_plain_explanation(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_no2():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_hcho():
            return xr.Dataset(
                {"hcho": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_no2", make_no2)
        self.volume.add_zarr("obs_hcho", make_hcho)

        compare = comparison_tools.make_compare(self.mcp_tools)
        raw = await compare.ainvoke({"handle_a": "obs_no2", "handle_b": "obs_hcho", "mode": "region"})
        result = json.loads(raw)

        self.assertIn("error", result)
        self.assertIn("no2", result["error"])
        self.assertIn("hcho", result["error"])

    async def test_an_unknown_mode_is_rejected(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_ds():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_x", make_ds)
        self.volume.add_zarr("obs_y", make_ds)

        compare = comparison_tools.make_compare(self.mcp_tools)
        raw = await compare.ainvoke({"handle_a": "obs_x", "handle_b": "obs_y", "mode": "bogus"})
        result = json.loads(raw)

        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
