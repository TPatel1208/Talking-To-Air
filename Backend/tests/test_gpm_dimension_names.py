"""
tests/test_gpm_dimension_names.py
===================================
GPM-style HDF5 products (IMERG) carry no netCDF dimension scales, so both
NetCDF engines open them with engine-invented placeholder dims
(``phony_dim_N``) — the science variable ends up with NO lat/lon
coordinates, and every mask/plot/stat call refused with "Could not find
lat/lon coordinates. Dims present: ['time','dim0'...]" while the same
handle retrieved and opened fine (QA 2026-07-17 major finding: a GPM plot
died at the stats step with a generic internal error, and the follow-up
point query answered a false "no data found").

The files DO declare their real dims: every variable carries GPM's
per-variable ``DimensionNames`` attribute ("time,lon,lat"). Honoring that
declaration is coordinate discovery, the same doctrine as T24's
CF-metadata-primary lat/lon identification. These tests write a real
dimension-scale-less HDF5 with h5py (no xarray writer can produce one) and
drive it through the public open/stat seams.
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
    "numpy", "xarray", "h5py", "shapely", "rasterio", "affine",
]


def _write_gpm_granule(path: str) -> None:
    """A minimal GPM-IMERG-shaped granule: /Grid group, no dimension scales,
    per-variable DimensionNames attrs — precipitation(time, lon, lat) on a
    10-degree global grid, constant 2.0 mm/hr."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        grid = f.create_group("Grid")

        precip = grid.create_dataset(
            "precipitation", data=np.full((1, 36, 18), 2.0, dtype=np.float32)
        )
        # Real IMERG files store this attr as fixed-length bytes.
        precip.attrs["DimensionNames"] = np.bytes_(b"time,lon,lat")
        precip.attrs["units"] = "mm/hr"

        lon = grid.create_dataset("lon", data=np.arange(-175.0, 185.0, 10.0, dtype=np.float32))
        lon.attrs["DimensionNames"] = "lon"
        lon.attrs["units"] = "degrees_east"

        lat = grid.create_dataset("lat", data=np.arange(-85.0, 95.0, 10.0, dtype=np.float32))
        lat.attrs["DimensionNames"] = "lat"
        lat.attrs["units"] = "degrees_north"

        time = grid.create_dataset("time", data=np.array([1_400_000_000], dtype=np.int64))
        time.attrs["DimensionNames"] = "time"
        time.attrs["units"] = "seconds since 1980-01-06 00:00:00"


def _write_3cmb_style_granule(path: str) -> None:
    """GPM_3CMB_DAY shape (live 2026-07-17): the ROOT group carries header
    string variables (InputFileNames...), and the science data lives in
    nested /Grids/G1/... groups. The old "root has data_vars => genuinely
    flat file" early-return returned ONLY the header strings and silently
    discarded every science group."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        f.create_dataset("InputFileNames", data=np.array([b"3B-DAY.GPM...HDF5"]))
        g1 = f.create_group("Grids/G1")
        var = g1.create_dataset("precipTotRate", data=np.full((4, 3), 1.5, dtype=np.float32))
        var.attrs["DimensionNames"] = "lnL,ltL"
        var.attrs["units"] = "mm/hr"


def _write_undeclared_phony_granule(path: str) -> None:
    """The unrecoverable cousin: no dimension scales AND no DimensionNames —
    nothing on disk names the dims, so refusal is the honest answer."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        grid = f.create_group("Grid")
        var = grid.create_dataset("mystery", data=np.ones((4, 5), dtype=np.float32))
        var.attrs["units"] = "1"


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "GPM dimension-names test dependencies are not installed",
)
class GpmDimensionNamesTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_open_handle_recovers_real_dim_names_from_dimension_names_attrs(self):
        from services.open_handle import open_handle
        from utils.geo_utils import find_lat_coord, find_lon_coord

        self.volume.add_hdf5("obs_gpm", _write_gpm_granule)

        ds = await open_handle("obs_gpm", self.mcp_tools)
        da = ds["precipitation"]

        self.assertIn("lat", da.dims, f"dims still placeholders: {list(da.dims)}")
        self.assertIn("lon", da.dims)
        self.assertEqual(find_lat_coord(da), "lat")
        self.assertEqual(find_lon_coord(da), "lon")

    async def test_compute_statistic_tool_answers_stats_for_a_gpm_shaped_granule(self):
        from tools.satellite_tools.stat_tools import make_compute_statistic_tool

        self.volume.add_hdf5("obs_gpm", _write_gpm_granule)

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_gpm", "location": "usa", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["mean"], 2.0, places=5)  # float32 source values
        self.assertGreater(result["n_pixels"], 0)

    async def test_root_header_variables_do_not_hide_the_science_groups(self):
        from services.open_handle import open_handle

        self.volume.add_hdf5("obs_3cmb", _write_3cmb_style_granule)

        ds = await open_handle("obs_3cmb", self.mcp_tools)

        self.assertIn(
            "precipTotRate", ds.data_vars,
            f"science groups were discarded; only {list(ds.data_vars)} survived",
        )
        # Declared dims applied even though the file offers no coordinate
        # arrays for them — identification can then refuse honestly.
        self.assertEqual(set(ds["precipTotRate"].dims), {"lnL", "ltL"})

    async def test_stats_refuse_with_a_typed_error_when_nothing_declares_the_dims(self):
        # Refusal path: no scales, no DimensionNames — the tool must answer
        # a classified unsupported_grid error, never crash off-taxonomy and
        # never claim "no data found" for data that is right there.
        from tools.satellite_tools.stat_tools import make_compute_statistic_tool

        self.volume.add_hdf5("obs_mystery", _write_undeclared_phony_granule)

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_mystery", "location": "usa", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertIn("error", result)
        self.assertEqual(result["error"]["category"], "unsupported_grid")
        self.assertNotIn("No valid data", str(result["error"]))


if __name__ == "__main__":
    unittest.main()
