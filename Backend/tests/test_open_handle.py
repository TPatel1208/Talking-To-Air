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


if __name__ == "__main__":
    unittest.main()
