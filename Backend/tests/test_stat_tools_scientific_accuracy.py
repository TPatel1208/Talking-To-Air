"""
tests/test_stat_tools_scientific_accuracy.py
==============================================
Two scientific-accuracy doctrines for the statistics tools:

1. **Extremes are computed over every observation, not the time-mean field.**
   The old shape reduced time with a per-cell mean first and then took
   max/min over that field — so "what was the max NO2?" over a multi-day
   handle answered with the max of the period-AVERAGE map, systematically
   understating real extremes. max/min compose exactly across time (the max
   of per-cell maxima IS the max over all observations), so they now reduce
   time with the same stat. find_daily_peak gets the same treatment.

2. **Regional means are cos(latitude) area-weighted.** On a lat/lon grid,
   cell area shrinks by cos(latitude) toward the poles; an unweighted mean
   over cells over-weights high latitudes (a few percent for a CONUS-scale
   region, badly wrong for continental/global ones). The basis for every
   reported statistic is disclosed in ``stat_basis``.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]

_HERMETIC_MODULES = ["numpy", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in _HERMETIC_MODULES),
    "area_weighted_mean test dependencies are not installed",
)
class AreaWeightedMeanTests(unittest.TestCase):
    """Hermetic — the ONE regional-mean definition, no fake MCP needed."""

    def test_weights_cells_by_cosine_of_latitude(self):
        import xarray as xr
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        # Two lat rows: value 0.0 at the equator (weight 1.0), 1.0 at 60°N
        # (weight 0.5) -> weighted mean 1/3, where the unweighted mean is 0.5.
        da = xr.DataArray(
            [[0.0, 0.0], [1.0, 1.0]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0, 20.0]},
        )

        self.assertAlmostEqual(area_weighted_mean(da), 1.0 / 3.0, places=6)

    def test_nan_cells_are_excluded_not_zero_filled(self):
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        da = xr.DataArray(
            [[np.nan, np.nan], [1.0, 3.0]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0, 20.0]},
        )

        # Only the 60°N row survives; equal weights within a row -> plain mean.
        self.assertAlmostEqual(area_weighted_mean(da), 2.0, places=6)

    def test_all_nan_padding_around_the_data_changes_nothing(self):
        """The regional mean must depend on the cells that carry data, not on
        how much empty grid surrounds them: masking a small AOI out of a
        continental granule and out of a tight crop of that same granule is
        the same question. Real granules publish float32 latitudes, and
        weights built in that dtype make the answer drift with the number of
        zero-weight cells summed -- 5e-7 on a real TEMPO New Jersey mean."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        lats = np.arange(38.79, 41.35, 0.02).astype(np.float32)
        lons = np.arange(-75.57, -73.88, 0.02).astype(np.float32)
        rng = np.random.default_rng(7)
        values = rng.random((len(lats), len(lons)))
        padded = xr.DataArray(
            values, dims=("lat", "lon"),
            coords={"lat": ("lat", lats), "lon": ("lon", lons)},
        )
        # Blank everything outside a small interior window, then take the same
        # window as a tight array: identical data, very different padding.
        window = {"lat": slice(40, 70), "lon": slice(10, 40)}
        blanked = padded.where(False)
        blanked[window["lat"], window["lon"]] = padded[window["lat"], window["lon"]]
        tight = padded.isel(window)

        np.testing.assert_allclose(
            area_weighted_mean(blanked), area_weighted_mean(tight), rtol=1e-12
        )

    def test_falls_back_to_unweighted_mean_without_a_latitude_axis(self):
        import xarray as xr
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        da = xr.DataArray([1.0, 2.0, 3.0], dims=("x",))

        self.assertAlmostEqual(area_weighted_mean(da), 2.0, places=6)

    def test_raises_when_no_finite_values_remain(self):
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        da = xr.DataArray(
            [[np.nan], [np.nan]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0]},
        )

        with self.assertRaises(ValueError):
            area_weighted_mean(da)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "stat-tools scientific accuracy test dependencies are not installed",
)
class StatToolsTimeCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

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

    def _add_two_day_dataset(self, handle, values_by_day):
        """A two-timestep TEMPO_NO2-shaped Dataset (all flags good) on
        lat=(10, 20), lon=(30, 40)."""
        from test_satellite_tools_masking_execution import _tempo_no2_dataset
        import xarray as xr

        good_flags = [[[0, 0], [0, 0]] for _ in values_by_day]

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=values_by_day, flags=good_flags,
                time=("2024-06-01", "2024-06-02")[: len(values_by_day)],
            )

        self.volume.add_zarr(handle, make_ds)

    async def test_max_and_min_are_true_extremes_over_all_observations(self):
        """Day 1: [[1,2],[3,4]]; day 2: [[9,0.5],[0.5,0.5]]. The per-cell
        time-mean field is [[5,1.25],[1.75,2.25]] — its max is 5.0 and its
        min is 1.25, but the TRUE extremes over every observation are 9.0
        and 0.5. The old max-of-the-mean-field shape reported 5.0/1.25."""
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        self._add_two_day_dataset("obs_2day", [
            [[1.0, 2.0], [3.0, 4.0]],
            [[9.0, 0.5], [0.5, 0.5]],
        ])

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_2day", "location": "global", "stats": ["max", "min"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["max"], 9.0)
        self.assertAlmostEqual(result["min"], 0.5)
        self.assertIn("all timesteps", result["stat_basis"]["max"])

    async def test_mean_is_area_weighted_and_its_basis_is_disclosed(self):
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        self._add_two_day_dataset("obs_1day", [[[1.0, 1.0], [3.0, 3.0]]])

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_1day", "location": "global", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected = (1.0 * w10 + 3.0 * w20) / (w10 + w20)
        self.assertAlmostEqual(result["mean"], expected)
        self.assertIn("cos(latitude)", result["stat_basis"]["mean"])

    async def test_find_daily_peak_reports_the_peak_over_all_observations(self):
        """Day 2 carries a 10.0 spike that the period-mean field flattens to
        7.0 — the reported peak must be the observed 10.0."""
        from tta_backend.tools.satellite_tools.stat_tools import make_find_daily_peak

        self._add_two_day_dataset("obs_peak", [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 0.5], [0.5, 10.0]],
        ])

        find_daily_peak = make_find_daily_peak(self.mcp_tools)
        raw = await find_daily_peak.ainvoke({"handle": "obs_peak", "location": "global"})
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["peak_value"], 10.0)
        self.assertAlmostEqual(result["peak_lat"], 20.0)
        self.assertAlmostEqual(result["peak_lon"], 40.0)


if __name__ == "__main__":
    unittest.main()
