import asyncio
import importlib.util
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

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "open_handle test dependencies are not installed",
)
class OpenHandleZarrTests(unittest.IsolatedAsyncioTestCase):
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
        self.tools = await load_raw_mcp_tools(settings)

    async def test_open_handle_opens_zarr_handle_into_dataset_with_expected_variables(self):
        import xarray as xr

        from services.open_handle import open_handle

        def make_dataset():
            return xr.Dataset({"no2": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])})

        self.volume.add_zarr("obs_1", make_dataset)

        ds = await open_handle("obs_1", self.tools)

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("no2", ds.data_vars)

    async def test_open_handle_opens_zipped_zarr_transform_export(self):
        """The MCP's transform tools (compare/regrid) export derived cubes as
        a *zipped* Zarr store (``cube.zarr.zip``, media_type
        ``application/zarr``) — never a directory store. zarr-python 3
        dropped v2's ZipStore-from-suffix inference, so a plain
        ``xr.open_zarr(path)`` reads the zip file as an empty directory and
        every compare dies with "No group found in store ... at path ''"
        (live TEMPO NO2 Texas compare, 2026-07-16)."""
        import xarray as xr

        from services.open_handle import open_handle

        def make_dataset():
            return xr.Dataset(
                {"product__vertical_column_troposphere": (("time", "y", "x"), [[[1.0, 2.0], [3.0, 4.0]]])}
            )

        self.volume.add_zarr_zip("cube_transform_1", make_dataset)

        ds = await open_handle("cube_transform_1", self.tools)

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("product__vertical_column_troposphere", ds.data_vars)

    async def test_open_handle_opens_parquet_handle_into_arrow_table(self):
        import pyarrow as pa

        from services.open_handle import open_handle

        def make_table():
            return pa.table({"lat": [1.0, 2.0], "lon": [3.0, 4.0], "no2": [5.0, 6.0]})

        self.volume.add_parquet("cube_1", make_table)

        table = await open_handle("cube_1", self.tools)

        self.assertIsInstance(table, pa.Table)
        self.assertEqual(table.column_names, ["lat", "lon", "no2"])
        self.assertEqual(table.num_rows, 2)

    async def test_open_handle_emits_an_open_stage_status(self):
        """T19: open_handle is the single seam every plot/statistics tool
        passes through to reach an opened dataset — narrating "open" here
        covers every caller (including stat_tools/comparison_tools/
        validation_tools, which have no emit_status calls of their own)
        without touching each tool individually."""
        import xarray as xr

        from services.open_handle import open_handle
        import utils.streaming as streaming

        def make_dataset():
            return xr.Dataset({"no2": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])})

        self.volume.add_zarr("obs_open_stage", make_dataset)

        seen = []

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await open_handle("obs_open_stage", self.tools)
        finally:
            streaming._status_emitter.reset(token)

        self.assertIn("open", [s["stage"] for s in seen])

    async def test_open_handle_recovers_from_eviction_via_rematerialize(self):
        import xarray as xr

        from services.open_handle import open_handle

        def make_dataset():
            return xr.Dataset({"no2": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])})

        self.volume.add_zarr("obs_2", make_dataset)
        self.volume.evict("obs_2")

        ds = await open_handle("obs_2", self.tools)

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("no2", ds.data_vars)
        self.assertEqual(self.volume.rematerialize_calls["obs_2"], 1)

    async def test_open_handle_self_heals_a_ready_but_unreadable_export_via_rematerialize(self):
        """The observed intermittent plot failure: export_result reports
        "ready" but the file on disk isn't a readable NetCDF/HDF5 (an error-
        response body or an incomplete/empty file saved in place of the
        granule). A manual retry works because a fresh retrieval writes real
        data -- so open_handle re-materializes once and re-opens itself,
        rather than surfacing the failure and making the user retry by hand."""
        import xarray as xr

        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        self.volume.add_netcdf("obs_corrupt", {None: make_root, "product": make_product_group})
        self.volume.corrupt("obs_corrupt")  # ready, but body is an HTML error page

        ds = await open_handle("obs_corrupt", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertEqual(self.volume.rematerialize_calls["obs_corrupt"], 1)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "open_handle test dependencies are not installed",
)
class OpenHandleEventLoopOffloadTests(unittest.IsolatedAsyncioTestCase):
    """T16: opening a handle (xr.open_zarr et al.) is CPU/IO work that used
    to run straight on the event loop, freezing every concurrent stream for
    its duration. Hermetic per PRD Testing Decisions: no thread-pool
    internals inspected — only that a concurrent trivial coroutine keeps
    making progress while a (patched-slow) open call is in flight."""

    async def test_open_handle_does_not_block_a_concurrent_coroutine(self):
        import time
        from unittest.mock import AsyncMock, patch

        import xarray as xr

        from services import open_handle as open_handle_module
        from services.open_handle import open_handle

        def slow_open_zarr(path):
            time.sleep(0.6)
            return xr.Dataset({"no2": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])})

        tick_count = 0

        async def ticker():
            nonlocal tick_count
            for _ in range(20):
                await asyncio.sleep(0.03)
                tick_count += 1

        fast_export = AsyncMock(return_value={
            "status": "ready", "storage_uri": "file:///obs_slow.zarr", "media_type": "zarr",
        })
        with patch.object(open_handle_module, "_export", fast_export), \
             patch("xarray.open_zarr", side_effect=slow_open_zarr):
            start = time.monotonic()
            ds, _ = await asyncio.gather(open_handle("obs_slow", {}), ticker())
            elapsed = time.monotonic() - start

        # ticker() and slow_open_zarr each take ~0.6s. If _open ran on the
        # event loop the two would serialize (~1.5s, measured); offloaded to
        # a thread they overlap (~1.0s, measured) — 1.25s cleanly separates
        # the two on this environment's own timer overhead.
        self.assertLess(elapsed, 1.25)
        self.assertEqual(tick_count, 20)
        self.assertIn("no2", ds.data_vars)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "open_handle test dependencies are not installed",
)
class OpenHandleRecoveryExhaustedTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_handle_surfaces_mcp_error_verbatim_after_one_failed_rematerialize(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings
        from services.open_handle import OpenHandleError, open_handle

        calls = {"rematerialize": 0}

        async def export_result(handle, workspace_id):
            return {"handle": handle, "status": "expired", "message": "handle evicted"}

        async def rematerialize(handle, workspace_id):
            calls["rematerialize"] += 1
            return {"job_handle": "job_x", "obs_handle": handle, "status": "queued"}

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": job_handle,
                "status": "failed",
                "message": "harmony: provider GES_DISC rejected rematerialize request",
            }

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": export_result,
            "rematerialize": rematerialize,
            "get_retrieval_status": get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)

        with self.assertRaises(OpenHandleError) as ctx:
            await open_handle("obs_evicted", tools)

        self.assertIn("harmony: provider GES_DISC rejected rematerialize request", str(ctx.exception))
        self.assertEqual(calls["rematerialize"], 1)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "open_handle test dependencies are not installed",
)
class OpenHandleClassifiedErrorTests(unittest.IsolatedAsyncioTestCase):
    """T18: a classified MCP outcome (e.g. a contract-shaped tool failure)
    is a distinct thing from OpenHandleError's own eviction-recovery
    failure — open_handle() must let it propagate as MCPToolError, not
    swallow or relabel it, so a caller can tell "the handle couldn't be
    recovered" apart from "the MCP call itself was malformed"."""

    async def test_open_handle_lets_a_classified_mcp_error_propagate_distinct_from_open_handle_error(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.results import MCPToolError
        from config.settings import Settings
        from services.open_handle import OpenHandleError, open_handle

        async def export_result(handle, workspace_id="default"):
            raise ValueError("a shape the classifier has never seen before")

        server = FakeEarthdataMCPServer(build_fake_mcp({"export_result": export_result}))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)

        with self.assertRaises(MCPToolError) as ctx:
            await open_handle("obs_1", tools)

        self.assertEqual(ctx.exception.category, "contract")
        self.assertNotIsInstance(ctx.exception, OpenHandleError)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle grouped-netcdf test dependencies are not installed",
)
class OpenHandleGroupedNetcdfTests(unittest.IsolatedAsyncioTestCase):
    """Some providers (e.g. TEMPO L3) nest their science variables under an
    HDF5 subgroup such as /product and leave the root group's data_vars
    empty. Before the fix, xr.open_dataset(path) alone returned that empty
    root dataset and every downstream plot/statistics tool failed with
    "Dataset has no data variables." even though the granule had real data."""

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
        self.tools = await load_raw_mcp_tools(settings)

    async def test_open_handle_descends_into_a_subgroup_when_root_has_no_data_vars(self):
        import xarray as xr

        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        self.volume.add_netcdf("obs_tempo", {None: make_root, "product": make_product_group})

        ds = await open_handle("obs_tempo", self.tools)

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("vertical_column_troposphere", ds.data_vars)

    async def test_open_handle_merges_multiple_non_empty_subgroups(self):
        import xarray as xr

        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        def make_qa_group():
            return xr.Dataset({
                "qa_value": (("lat", "lon"), [[0.0, 0.0], [0.0, 0.0]]),
            })

        self.volume.add_netcdf("obs_tempo_multi", {
            None: make_root,
            "product": make_product_group,
            "qa_statistics": make_qa_group,
        })

        ds = await open_handle("obs_tempo_multi", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertIn("qa_value", ds.data_vars)

    async def test_open_handle_stamps_each_variables_source_group(self):
        """Merging groups by bare name destroys group membership, which is
        classification evidence (variable_roles' qa_statistics/geolocation/
        product priors). Each merged variable must carry its source group as
        a ``group_path`` attr so post-open classification (T36 evidence) sees
        the same group a describe_dataset inventory name would."""
        import xarray as xr

        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        def make_qa_group():
            return xr.Dataset({
                "max_vertical_column_sample": (("lat", "lon"), [[0.0, 0.0], [0.0, 0.0]]),
            })

        self.volume.add_netcdf("obs_tempo_stamped", {
            None: make_root,
            "product": make_product_group,
            "qa_statistics": make_qa_group,
        })

        ds = await open_handle("obs_tempo_stamped", self.tools)

        self.assertEqual(ds["vertical_column_troposphere"].attrs.get("group_path"), "product")
        self.assertEqual(ds["max_vertical_column_sample"].attrs.get("group_path"), "qa_statistics")

    async def test_open_handle_qualifies_leaf_name_collisions_instead_of_silently_overriding(self):
        """The same leaf name in two groups used to merge with
        compat="override" — whichever group iterated first silently won, so
        a plausible number could come from the wrong variable. Colliding
        variables must keep their group-qualified names, so an explicit
        qualified request resolves exactly and a bare ambiguous one is
        refused with candidates (T25 doctrine), never guessed."""
        import xarray as xr

        from preprocessing.aggregation_service import AggregationService, VariableChoiceRequired
        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset({
                "vertical_column": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        def make_support_group():
            return xr.Dataset({
                "vertical_column": (("lat", "lon"), [[100.0, 200.0], [300.0, 400.0]]),
            })

        self.volume.add_netcdf("obs_collide", {
            None: make_root,
            "product": make_product_group,
            "support_data": make_support_group,
        })

        ds = await open_handle("obs_collide", self.tools)

        # Both survive, group-qualified — neither silently shadowed.
        self.assertIn("product/vertical_column", ds.data_vars)
        self.assertIn("support_data/vertical_column", ds.data_vars)
        self.assertNotIn("vertical_column", ds.data_vars)

        # An explicit qualified request resolves to exactly that group's data.
        service = AggregationService()
        da = service.to_dataarray(ds, variable="product/vertical_column")
        self.assertEqual(float(da.values[0][0]), 1.0)

        # A bare ambiguous request refuses with candidates, never guesses --
        # T49: as the deterministic picker short-circuit, whose compact
        # tool-result still names the qualified candidates.
        with self.assertRaises(VariableChoiceRequired) as ctx:
            service.to_dataarray(ds, variable=None)
        self.assertIn("product/vertical_column", ctx.exception.mcp_error.message)
        self.assertIn("support_data/vertical_column", ctx.exception.mcp_error.message)

    async def test_open_handle_leaves_a_genuinely_flat_netcdf_dataset_untouched(self):
        import xarray as xr

        from services.open_handle import open_handle

        def make_flat():
            return xr.Dataset({"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])})

        self.volume.add_netcdf("obs_flat", {None: make_flat})

        ds = await open_handle("obs_flat", self.tools)

        self.assertIn("no2", ds.data_vars)

    async def test_open_handle_promotes_lat_lon_from_a_sibling_group_to_coordinates(self):
        """TEMPO L3 (and similar grouped products like OMI L3) split their
        science variable and its lon/lat into separate sibling subgroups
        (/product and /geolocation) rather than nesting lon/lat under the
        science group. Before the fix, both groups' variables merged in as
        plain data_vars -- so find_lat_coord/find_lon_coord (which only
        look at .coords) came up empty, and AggregationService.to_dataarray
        could even pick "longitude" as the primary variable by accident
        (dict iteration order put it before the real science variable)."""
        import xarray as xr

        from services.open_handle import open_handle
        from preprocessing.aggregation_service import AggregationService
        from utils.geo_utils import find_lat_coord, find_lon_coord

        def make_root():
            return xr.Dataset()

        def make_geolocation_group():
            return xr.Dataset({
                "longitude": (("mirror_step", "xtrack"), [[-100.0, -99.0], [-100.0, -99.0]]),
                "latitude": (("mirror_step", "xtrack"), [[30.0, 30.0], [31.0, 31.0]]),
            })

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("mirror_step", "xtrack"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        self.volume.add_netcdf("obs_tempo_geo", {
            None: make_root,
            "geolocation": make_geolocation_group,
            "product": make_product_group,
        })

        ds = await open_handle("obs_tempo_geo", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertNotIn("longitude", ds.data_vars)
        self.assertIn("latitude", ds.coords)
        self.assertIn("longitude", ds.coords)

        da = AggregationService().to_dataarray(ds)
        self.assertEqual(da.name, "vertical_column_troposphere")
        self.assertEqual(find_lat_coord(da), "latitude")
        self.assertEqual(find_lon_coord(da), "longitude")

    async def test_open_handle_promotes_cf_identified_latlon_with_unusual_names(self):
        """T24: the promotion site keys on CF metadata, not a name allowlist,
        so a grouped product whose lat/lon are named 'y'/'x' (a spelling the
        allowlist would never guess) but carry standard_name latitude/
        longitude is still attached and resolvable -- covering datasets not
        on disk by the contract they publish against."""
        import xarray as xr

        from services.open_handle import open_handle
        from preprocessing.aggregation_service import AggregationService
        from utils.geo_utils import find_lat_coord, find_lon_coord

        def make_root():
            return xr.Dataset()

        def make_geolocation_group():
            return xr.Dataset({
                "x": (("mirror_step", "xtrack"), [[-100.0, -99.0], [-100.0, -99.0]], {"standard_name": "longitude"}),
                "y": (("mirror_step", "xtrack"), [[30.0, 30.0], [31.0, 31.0]], {"standard_name": "latitude"}),
            })

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (("mirror_step", "xtrack"), [[1.0, 2.0], [3.0, 4.0]]),
            })

        self.volume.add_netcdf("obs_cf_named", {
            None: make_root,
            "geolocation": make_geolocation_group,
            "product": make_product_group,
        })

        ds = await open_handle("obs_cf_named", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertIn("y", ds.coords)
        self.assertIn("x", ds.coords)
        self.assertNotIn("y", ds.data_vars)

        da = AggregationService().to_dataarray(ds)
        self.assertEqual(da.name, "vertical_column_troposphere")
        self.assertEqual(find_lat_coord(da), "y")
        self.assertEqual(find_lon_coord(da), "x")

    async def test_open_handle_keeps_root_group_coordinates_when_science_var_is_in_a_subgroup(self):
        """TEMPO L3, subset to a single variable, splits its science
        variable into /product but leaves latitude/longitude/time as
        coordinate variables in the *root* group (no data_vars there).
        Before the fix, _open_netcdf popped that coord-only root group and
        returned /product alone -- so the DataArray arrived with zero
        coordinates and every plot/statistics tool failed with "Could not
        find lat/lon coordinates. Available coords: []" even though the
        grid was in the file. Reproduced against the real
        252241949_TEMPO_NO2_L3_V04_...subsetted.nc4 granule."""
        import xarray as xr

        from services.open_handle import open_handle
        from preprocessing.aggregation_service import AggregationService
        from utils.geo_utils import find_lat_coord, find_lon_coord

        def make_root():
            # coord-only root: lat/lon/time as coordinate variables, no data_vars
            return xr.Dataset(coords={
                "longitude": ("longitude", [-75.0, -74.0]),
                "latitude": ("latitude", [40.0, 41.0]),
                "time": ("time", [0]),
            })

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (
                    ("time", "latitude", "longitude"),
                    [[[1.0, 2.0], [3.0, 4.0]]],
                ),
            })

        self.volume.add_netcdf("obs_tempo_l3_rootcoords", {
            None: make_root,
            "product": make_product_group,
        })

        ds = await open_handle("obs_tempo_l3_rootcoords", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertIn("latitude", ds.coords)
        self.assertIn("longitude", ds.coords)

        da = AggregationService().to_dataarray(ds)
        self.assertEqual(da.name, "vertical_column_troposphere")
        self.assertEqual(find_lat_coord(da), "latitude")
        self.assertEqual(find_lon_coord(da), "longitude")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle grouped-netcdf test dependencies are not installed",
)
class OpenHandleNetcdfBundleTests(unittest.IsolatedAsyncioTestCase):
    """The MCP ships every OPeNDAP subset and every multi-granule Harmony
    result as ``application/netcdf-bundle+zip`` — a zip of NetCDF granule
    subsets. Before the fix, _open's ``"netcdf" in mt`` substring check
    matched that media type and fed the zip to the NetCDF engines, which
    failed with "not a readable NetCDF/HDF5 dataset ... retrying the
    retrieval typically resolves it" — a misleading message that sent the
    agent (and users) into retry loops that could never succeed. Reproduced
    against real /data/harmony/*/result.nc.zip and /data/opendap/*/
    subset.nc.zip exports."""

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
        self.tools = await load_raw_mcp_tools(settings)

    @staticmethod
    def _make_granule(day: int):
        import numpy as np
        import xarray as xr

        def factory():
            return xr.Dataset(
                {"no2": (("time", "latitude", "longitude"), [[[1.0 * day, 2.0], [3.0, 4.0]]])},
                coords={
                    "time": [np.datetime64(f"2026-07-{day:02d}T12:00:00")],
                    "latitude": [40.0, 41.0],
                    "longitude": [-75.0, -74.0],
                },
            )

        return factory

    async def test_open_handle_concats_a_multi_granule_bundle_on_time(self):
        import xarray as xr

        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_multi", {
            "granule_20260709.nc4": {None: self._make_granule(9)},
            "granule_20260710.nc4": {None: self._make_granule(10)},
        })

        ds = await open_handle("obs_bundle_multi", self.tools)

        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("no2", ds.data_vars)
        self.assertEqual(ds.sizes["time"], 2)
        self.assertIn("latitude", ds.coords)
        self.assertIn("longitude", ds.coords)

    async def test_open_handle_orders_bundle_by_time_not_filename(self):
        """T44 story #2: the MCP concatenates bundle members in filename order
        ('names sort chronologically'). A provider whose names don't sort
        against their dates would yield a non-monotonic time axis, and a later
        sel(time=slice(...)) would silently return the wrong subset. The open
        must order by the decoded timestamps, not the alphabetics."""
        import numpy as np

        from services.open_handle import open_handle

        # Alphabetical order (a_ < z_) is the OPPOSITE of date order (9 < 10).
        self.volume.add_netcdf_bundle("obs_bundle_misordered", {
            "z_earliest.nc4": {None: self._make_granule(9)},
            "a_latest.nc4": {None: self._make_granule(10)},
        })

        ds = await open_handle("obs_bundle_misordered", self.tools)

        times = ds["time"].values
        self.assertEqual(ds.sizes["time"], 2)
        self.assertTrue(np.all(np.diff(times) > np.timedelta64(0)))  # strictly increasing
        self.assertEqual(str(times[0])[:10], "2026-07-09")

    async def test_open_handle_dedupes_duplicate_bundle_timestamps_keep_first(self):
        """T44 story #2: overlapping orbits and reprocessed granules can carry
        identical timestamps. Left in, they double-count that granule in a mean
        and break later sel(time=...) with an opaque non-unique-index error.
        Keep the first occurrence, disclose the drop in a log event, and never
        crash."""
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_dupes", {
            "granule_a.nc4": {None: self._make_granule(9)},
            "granule_b.nc4": {None: self._make_granule(9)},  # same timestamp
            "granule_c.nc4": {None: self._make_granule(10)},
        })

        with self.assertLogs("services.open_handle", level="INFO") as logs:
            ds = await open_handle("obs_bundle_dupes", self.tools)

        self.assertEqual(ds.sizes["time"], 2)  # the duplicate collapsed to one
        # sel on the previously-duplicated timestamp resolves cleanly.
        picked = ds.sel(time="2026-07-09")
        self.assertEqual(float(picked["no2"].sum()), 18.0)  # 9+2+3+4 — one granule, not doubled to 36
        self.assertTrue(
            any("bundle_duplicate_timestamps" in msg for msg in logs.output),
            f"expected a bundle_duplicate_timestamps event, got: {logs.output}",
        )

    async def test_open_handle_opens_a_single_member_bundle(self):
        """OPeNDAP subsets arrive as a bundle even for one granule
        (subset.nc.zip with a single member)."""
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_single", {
            "subset.nc4": {None: self._make_granule(10)},
        })

        ds = await open_handle("obs_bundle_single", self.tools)

        self.assertIn("no2", ds.data_vars)
        self.assertEqual(ds.sizes["time"], 1)

    async def test_open_handle_merges_groups_and_promotes_latlon_inside_bundle_members(self):
        """Bundle members go through the same _open_netcdf path as bare
        NetCDF exports: grouped products keep bare variable names and get
        their root-group grid coordinates attached."""
        import xarray as xr

        from services.open_handle import open_handle

        def make_root():
            return xr.Dataset(coords={
                "longitude": ("longitude", [-75.0, -74.0]),
                "latitude": ("latitude", [40.0, 41.0]),
                "time": ("time", [0]),
            })

        def make_product_group():
            return xr.Dataset({
                "vertical_column_troposphere": (
                    ("time", "latitude", "longitude"),
                    [[[1.0, 2.0], [3.0, 4.0]]],
                ),
            })

        self.volume.add_netcdf_bundle("obs_bundle_grouped", {
            "TEMPO_NO2_L3_subsetted.nc4": {None: make_root, "product": make_product_group},
        })

        ds = await open_handle("obs_bundle_grouped", self.tools)

        self.assertIn("vertical_column_troposphere", ds.data_vars)
        self.assertIn("latitude", ds.coords)
        self.assertIn("longitude", ds.coords)

    async def test_open_handle_self_heals_a_corrupt_bundle_via_rematerialize(self):
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_corrupt", {
            "granule_20260710.nc4": {None: self._make_granule(10)},
        })
        self.volume.corrupt("obs_bundle_corrupt")  # ready, but body is an HTML error page

        ds = await open_handle("obs_bundle_corrupt", self.tools)

        self.assertIn("no2", ds.data_vars)
        self.assertEqual(self.volume.rematerialize_calls["obs_bundle_corrupt"], 1)

    async def test_open_handle_refuses_a_bundle_over_the_uncompressed_size_gate(self):
        """A bundle whose members would exceed the configured uncompressed
        size must refuse with a deterministic too_large error before any
        member is extracted or opened — the previous behavior was to load
        everything and let the OOM killer take the process down (live
        2026-07-12, full-day TEMPO NO2)."""
        from unittest.mock import patch

        from config.settings import Settings
        from earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_big", {
            "granule_20260709.nc4": {None: self._make_granule(9)},
            "granule_20260710.nc4": {None: self._make_granule(10)},
        })

        tiny_cap = Settings(bundle_open_max_uncompressed_bytes=1)
        with patch("services.open_handle.get_settings", return_value=tiny_cap):
            with self.assertRaises(MCPToolError) as ctx:
                await open_handle("obs_bundle_big", self.tools)

        self.assertEqual(ctx.exception.category, CATEGORY_TOO_LARGE)
        self.assertIn("Narrow", ctx.exception.suggestion or "")

    @staticmethod
    def _make_attr_dated_granule(month: int):
        """HAQ TROPOMI monthly L3 shape (live 2026-07-16): no time dimension
        at all — 2D (Latitude, Longitude) only, the month living solely in
        the RangeBeginningDate/RangeBeginningTime global attrs."""
        import xarray as xr

        def factory():
            return xr.Dataset(
                {"Tropospheric_NO2": (("Latitude", "Longitude"), [[1.0 * month, 2.0], [3.0, 4.0]])},
                coords={"Latitude": [40.0, 41.0], "Longitude": [-75.0, -74.0]},
                attrs={
                    "RangeBeginningDate": f"2024-{month:02d}-01",
                    "RangeBeginningTime": "00:00:00.000000Z",
                },
            )

        return factory

    async def test_bundle_members_with_no_time_dim_get_an_indexed_time_coord(self):
        """Members with no time dim at all (attr-dated monthly L3, e.g. HAQ
        TROPOMI NO2) must gain a synthesized, *indexed* time coordinate
        before concat. Left alone, xr.concat(dim="time") fabricates a bare
        index-less dim, and every downstream time selection dies with
        xarray's "no associated coordinate or index" (live 2026-07-16)."""
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_attr_dated", {
            "tropomi_202406.nc4": {None: self._make_attr_dated_granule(6)},
            "tropomi_202407.nc4": {None: self._make_attr_dated_granule(7)},
        })

        ds = await open_handle("obs_bundle_attr_dated", self.tools)

        self.assertEqual(ds.sizes["time"], 2)
        self.assertIn("time", ds.coords)  # indexed coordinate, not a bare stacking dim
        self.assertIn("time", ds.indexes)
        self.assertEqual(
            [str(t)[:10] for t in ds["time"].values],
            ["2024-06-01", "2024-07-01"],
        )

    async def test_single_member_bundle_with_no_time_dim_also_gains_time(self):
        """The synthesis applies uniformly, so single-month opens of the same
        product carry the same shape (a size-1 indexed time) as multi-month
        opens — downstream squeezing already handles time=1 cleanly."""
        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_attr_dated_single", {
            "tropomi_202406.nc4": {None: self._make_attr_dated_granule(6)},
        })

        ds = await open_handle("obs_bundle_attr_dated_single", self.tools)

        self.assertEqual(ds.sizes["time"], 1)
        self.assertIn("time", ds.coords)
        self.assertEqual(str(ds["time"].values[0])[:10], "2024-06-01")

    def test_synthesis_preserves_a_real_datetime_coord_on_a_cased_time_dim(self):
        """Finding #15: a differently-cased singleton ``Time`` dim can already
        carry a *real* per-granule overpass time. Synthesis must rename it to
        ``time`` and KEEP that coordinate -- overwriting it with the attr
        date's (midnight) timestamp would flatten two same-day granules with
        distinct overpass times to identical stamps, and the bundle dedup would
        then drop one, halving a 'daily average'."""
        import numpy as np
        import xarray as xr
        from services.open_handle import _synthesize_member_time_coord

        ds = xr.Dataset(
            {"no2": (("Time", "lat", "lon"), [[[1.0, 2.0], [3.0, 4.0]]])},
            coords={
                "Time": [np.datetime64("2024-06-01T13:30:00")],  # real overpass time
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            attrs={"RangeBeginningDate": "2024-06-01"},  # date only -> would synth midnight
        )

        out = _synthesize_member_time_coord(ds)

        self.assertIn("time", out.dims)
        self.assertIn("time", out.coords)
        self.assertEqual(out["time"].values[0], np.datetime64("2024-06-01T13:30:00"))

    def test_synthesis_still_fills_a_cased_time_dim_that_carries_no_coordinate(self):
        """The existing OMI_MINDS_NO2d shape -- a differently-cased singleton
        ``Time`` dim with NO coordinate variable -- must still be renamed and
        given the synthesized attr timestamp (Finding #15 preserves real
        coords; it does not stop filling absent ones)."""
        import xarray as xr
        from services.open_handle import _synthesize_member_time_coord

        ds = xr.Dataset(
            {"no2": (("Time", "lat", "lon"), [[[1.0, 2.0], [3.0, 4.0]]])},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},  # Time dim has no coord
            attrs={"RangeBeginningDate": "2024-06-01", "RangeBeginningTime": "00:00:00"},
        )

        out = _synthesize_member_time_coord(ds)

        self.assertIn("time", out.coords)
        self.assertEqual(str(out["time"].values[0])[:10], "2024-06-01")

    async def test_same_day_granules_with_distinct_overpass_times_both_survive(self):
        """Finding #15 end-to-end: two same-day granules whose real overpass
        times live on a differently-cased ``Time`` dim must NOT be flattened to
        one timestamp and deduped down to a single granule -- the daily mean
        must see both observations."""
        import numpy as np
        import xarray as xr
        from services.open_handle import open_handle

        def _make(hour: int, minute: int):
            def factory():
                return xr.Dataset(
                    {"no2": (("Time", "lat", "lon"), [[[1.0, 2.0], [3.0, 4.0]]])},
                    coords={
                        "Time": [np.datetime64(f"2024-06-01T{hour:02d}:{minute:02d}:00")],
                        "lat": [40.0, 41.0],
                        "lon": [-75.0, -74.0],
                    },
                    attrs={"RangeBeginningDate": "2024-06-01"},
                )
            return factory

        self.volume.add_netcdf_bundle("obs_bundle_overpasses", {
            "granule_am.nc4": {None: _make(13, 30)},
            "granule_pm.nc4": {None: _make(18, 45)},
        })

        ds = await open_handle("obs_bundle_overpasses", self.tools)

        self.assertEqual(ds.sizes["time"], 2)

    async def test_bundle_members_open_lazily_as_dask_chunks(self):
        """The memory contract behind the OOM fix: opening a bundle loads no
        member into RAM — each granule stays on disk as one dask chunk, so a
        downstream reduction streams granule-by-granule instead of staging
        the whole day plus a concat copy. The compute assertion doubles as
        the lifetime check: the extracted members must still be readable
        after open_handle has returned."""
        if importlib.util.find_spec("dask") is None:
            self.skipTest("dask is not installed")

        from services.open_handle import open_handle

        self.volume.add_netcdf_bundle("obs_bundle_lazy", {
            "granule_20260709.nc4": {None: self._make_granule(9)},
            "granule_20260710.nc4": {None: self._make_granule(10)},
        })

        ds = await open_handle("obs_bundle_lazy", self.tools)

        self.assertIsNotNone(ds["no2"].chunks)  # dask-backed, not an in-memory numpy array
        # (9+2+3+4) + (10+2+3+4) — full compute still works post-return
        self.assertEqual(float(ds["no2"].sum()), 37.0)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle grouped-netcdf test dependencies are not installed",
)
class OpenNetcdfMislabeledZipTests(unittest.TestCase):
    """Legacy rows materialized before the MCP's content sniffing can carry a
    plain-netCDF media type over zip bytes. _open_netcdf must route by the
    file's own magic instead of failing with the misleading incomplete-
    retrieval message."""

    def test_open_netcdf_detects_zip_bytes_and_opens_them_as_a_bundle(self):
        import zipfile

        import numpy as np
        import xarray as xr

        from services.open_handle import _open_netcdf

        member = xr.Dataset(
            {"no2": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]])},
            coords={
                "time": [np.datetime64("2026-07-10T12:00:00")],
                "latitude": [40.0, 41.0],
                "longitude": [-75.0, -74.0],
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            member_path = os.path.join(tmpdir, "granule.nc4")
            member.to_netcdf(member_path)
            zip_path = os.path.join(tmpdir, "result.nc4")  # netcdf-looking name, zip bytes
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(member_path, arcname="granule.nc4")

            ds = _open_netcdf(zip_path)

            self.assertIn("no2", ds.data_vars)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle grouped-netcdf test dependencies are not installed",
)
class BundleExtractionCacheTests(unittest.TestCase):
    """Bundle members are extracted into a TTL-pruned cache directory that
    outlives the open call — lazily-opened members are read from these files
    well after _open_netcdf_bundle returns, and no per-call cleanup hook can
    know when the last derived Dataset is done with them. Entries are keyed
    by the bundle file's identity, so a repeat open of the same bundle skips
    re-extraction; stale entries are swept on the next extraction."""

    def _write_bundle(self, tmpdir: str, zip_name: str, value: float) -> str:
        import zipfile

        import numpy as np
        import xarray as xr

        member = xr.Dataset(
            {"no2": (("time", "latitude", "longitude"), [[[value, 2.0], [3.0, 4.0]]])},
            coords={
                "time": [np.datetime64("2026-07-10T12:00:00")],
                "latitude": [40.0, 41.0],
                "longitude": [-75.0, -74.0],
            },
        )
        member_path = os.path.join(tmpdir, f"member_{zip_name}.nc4")
        member.to_netcdf(member_path)
        zip_path = os.path.join(tmpdir, zip_name)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(member_path, arcname="granule.nc4")
        return zip_path

    def test_bundle_extraction_cache_reuses_and_prunes_entries(self):
        import gc
        import time
        from unittest.mock import patch

        from services.open_handle import _open_netcdf_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_home = os.path.join(tmpdir, "fake_tmp")
            os.makedirs(cache_home)
            zip_a = self._write_bundle(tmpdir, "bundle_a.zip", 1.0)
            zip_b = self._write_bundle(tmpdir, "bundle_b.zip", 5.0)

            def cache_entries():
                roots = [
                    d for d in os.listdir(cache_home)
                    if os.path.isdir(os.path.join(cache_home, d))
                ]
                return sorted(
                    entry
                    for root in roots
                    for entry in os.listdir(os.path.join(cache_home, root))
                )

            with patch("tempfile.gettempdir", return_value=cache_home):
                ds_a = _open_netcdf_bundle(zip_a)
                self.assertEqual(float(ds_a["no2"].sum()), 10.0)
                first_entries = cache_entries()
                self.assertEqual(len(first_entries), 1)

                ds_a2 = _open_netcdf_bundle(zip_a)  # same bundle → reused, not re-extracted
                self.assertEqual(cache_entries(), first_entries)
                self.assertEqual(float(ds_a2["no2"].sum()), 10.0)

                # Release open member files, then age the entry past the TTL:
                # the next extraction (a different bundle) sweeps it.
                del ds_a, ds_a2
                gc.collect()
                root = os.path.join(cache_home, os.listdir(cache_home)[0])
                stale = time.time() - 100_000
                os.utime(os.path.join(root, first_entries[0]), (stale, stale))

                ds_b = _open_netcdf_bundle(zip_b)
                self.assertEqual(float(ds_b["no2"].sum()), 14.0)
                remaining = cache_entries()
                self.assertEqual(len(remaining), 1)
                self.assertNotEqual(remaining, first_entries)

                # Release ds_b's own open file handle on its extracted member
                # before the tempdir (which contains the patched cache_home)
                # tears down below -- on Windows an open handle makes that
                # cleanup raise PermissionError; POSIX allows unlinking an
                # open file, so this only bites on a Windows host run.
                del ds_b
                gc.collect()


class OpenNativeFormatMediaTypeTests(unittest.TestCase):
    """HDF4 / native-archive exports (e.g. MODIS MAIAC) have no local reader,
    and re-retrieving returns the same bytes — the error must say "pick a
    different product", not the retry-shaped unreadable-file message."""

    def test_open_raises_actionable_error_for_native_format_media_types(self):
        from services.open_handle import OpenHandleError, UnreadableExportError, _open

        for media_type in ("application/x-hdf4", "application/x-native-archive+zip"):
            with self.subTest(media_type=media_type):
                with self.assertRaises(OpenHandleError) as ctx:
                    _open("file:///data/whatever", media_type)
                self.assertNotIsInstance(ctx.exception, UnreadableExportError)
                msg = str(ctx.exception)
                self.assertIn("Retrying the retrieval will not help", msg)
                self.assertIn("different collection", msg)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle grouped-netcdf test dependencies are not installed",
)
class OpenNetcdfUnreadableFileTests(unittest.TestCase):
    """A 'ready' export whose file has no valid NetCDF/HDF5 magic (a zero-
    byte file or an error-response body saved as .nc4) must surface an
    actionable error -- not xarray's misleading "did not find a match in any
    of xarray's currently installed IO backends" message, which sends users
    to install packages that are already installed."""

    def _open_bytes(self, contents: bytes):
        from services.open_handle import _open_netcdf

        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as fh:
            fh.write(contents)
            path = fh.name
        try:
            return _open_netcdf(path)
        finally:
            os.unlink(path)

    def test_open_netcdf_raises_actionable_error_on_a_non_netcdf_file(self):
        from services.open_handle import UnreadableExportError

        for contents in (b"", b"<html><body>503</body></html>", os.urandom(4096)):
            with self.subTest(contents=contents[:16]):
                with self.assertRaises(UnreadableExportError) as ctx:
                    self._open_bytes(contents)
                msg = str(ctx.exception)
                self.assertIn("incomplete or failed retrieval", msg)
                self.assertNotIn("did not find a match in any of xarray", msg)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "open_handle lazy/eager equivalence test dependencies are not installed",
)
class OpenNetcdfLazyVsEagerEquivalenceTests(unittest.TestCase):
    """T45: a bare (non-bundle) export opens via _open_netcdf(path) eagerly,
    while a bundle member opens via _open_netcdf(path, chunks={}) with dask
    chunks. Nothing pinned that the two paths mask/aggregate identically --
    a dask-related regression in one path could silently diverge from the
    other, surfacing as a subtly different mean rather than a test failure."""

    def setUp(self):
        import tempfile

        import numpy as np
        import xarray as xr

        self.np = np
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        ds = xr.Dataset(
            {
                "no2": (
                    ("time", "lat", "lon"),
                    np.array([
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[-999.0, -999.0], [-999.0, -999.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]),
                )
            },
            coords={
                "time": np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            attrs={"cadence": "daily"},
        )
        self.path = os.path.join(self._tmpdir.name, "eager_vs_lazy.nc")
        ds.to_netcdf(self.path)
        self.col_info = {
            "primary_var": "no2",
            "cadence": "daily",
            "fill_value": -999.0,
            "valid_min": 0.0,
            "valid_max": 100.0,
        }

    def test_eager_and_dask_backed_opens_aggregate_identically(self):
        from preprocessing.aggregation_service import AggregationService
        from services.open_handle import _open_netcdf

        eager_ds = _open_netcdf(self.path)
        lazy_ds = _open_netcdf(self.path, chunks={})

        service = AggregationService()
        eager_result = service.aggregate(eager_ds, stat="mean", variable="no2", col_info=self.col_info)
        lazy_result = service.aggregate(lazy_ds, stat="mean", variable="no2", col_info=self.col_info)

        self.assertEqual(eager_result.meta["n_granules"], lazy_result.meta["n_granules"])
        self.assertEqual(eager_result.meta["granule_dates"], lazy_result.meta["granule_dates"])
        self.np.testing.assert_array_equal(
            eager_result.ds["no2"].values, lazy_result.ds["no2"].values,
        )


if __name__ == "__main__":
    unittest.main()
