"""
tests/test_stat_tools_longitude_normalization.py
==================================================
The stats tools skipped the 0..360 -> -180..180 longitude normalization
every other masking path performs. plot_singular, plot_multiple, and
compare all call _normalize_longitudes on the opened Dataset before
geometry masking; compute_statistic_tool and find_daily_peak masked the
raw array directly. On a 0..360 dataset a western-hemisphere region
(negative longitudes) rasterizes entirely outside the grid, the geometry
mask is all-False, and the tools report "No valid data found ... region
may be outside the data bbox" for data that is fully present — the same
handle plots fine but refuses to compute stats.

These tests mirror test_satellite_tools_masking_execution.py's production
shape: a TEMPO_NO2 Dataset (science var + sibling QA-flag var) opened
through the HandleVolume/open_handle seam, on a 0..360 longitude grid over
the continental US, masked with the offline 'usa' region (lon -125..-66).
The fixtures keep bad QA flags so they also pin the reason normalization
must happen on the whole Dataset before the DataArray is extracted (the
plot_singular convention): normalizing only the extracted science array
would leave the sibling flag variable on 0..360 and QA alignment would
produce an empty intersection instead of a mask.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

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


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "stat-tools longitude normalization test dependencies are not installed",
)
class StatToolsLongitudeNormalizationTests(unittest.IsolatedAsyncioTestCase):
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

    def _add_us_dataset_on_0_360_longitudes(self, values, flags):
        """A TEMPO_NO2-shaped Dataset over the continental US whose
        longitudes use the 0..360 convention: 245/255 are -115/-105 once
        normalized, squarely inside the offline 'usa' region box
        (-125..-66 lon, 24..50 lat)."""
        from test_satellite_tools_masking_execution import _tempo_no2_dataset
        import xarray as xr

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=values, flags=flags,
                lat=(30.0, 35.0), lon=(245.0, 255.0),
            )

        self.volume.add_zarr("obs_1", make_ds)

    async def test_compute_statistic_tool_normalizes_0_360_longitudes_before_masking(self):
        from tools.satellite_tools.stat_tools import make_compute_statistic_tool

        self._add_us_dataset_on_0_360_longitudes(
            values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
        )

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_1", "location": "usa", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        # Good cells (flag=0): 1.0 at lat=30, 3.0 at lat=35, cos(latitude)
        # area-weighted. A da-only normalization would break QA alignment
        # (empty intersection with the still-0..360 flag var); no
        # normalization at all yields "No valid data found".
        import math

        w30, w35 = math.cos(math.radians(30.0)), math.cos(math.radians(35.0))
        expected_mean = (1.0 * w30 + 3.0 * w35) / (w30 + w35)
        self.assertAlmostEqual(result["mean"], expected_mean)
        self.assertEqual(result["n_pixels"], 2)
        self.assertEqual(result["aggregation_meta"]["masking"]["qa_status"], "verified")

    async def test_find_daily_peak_normalizes_0_360_longitudes_before_masking(self):
        from tools.satellite_tools.stat_tools import make_find_daily_peak

        # The numerically highest raw value (99.0) carries a bad flag; the
        # true peak once masked is the good-flag 3.0 cell at lat=35, lon=245
        # (-115 once normalized).
        self._add_us_dataset_on_0_360_longitudes(
            values=[[1.0, 99.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
        )

        find_daily_peak = make_find_daily_peak(self.mcp_tools)
        raw = await find_daily_peak.ainvoke({"handle": "obs_1", "location": "usa"})
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["peak_value"], 3.0)
        self.assertAlmostEqual(result["peak_lat"], 35.0)
        # The reported peak longitude uses the -180..180 convention the rest
        # of the app (charts, geocoding) speaks — not the source's 0..360.
        self.assertAlmostEqual(result["peak_lon"], -115.0)


if __name__ == "__main__":
    unittest.main()
