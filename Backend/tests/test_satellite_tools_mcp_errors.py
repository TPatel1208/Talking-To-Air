"""
tests/test_satellite_tools_mcp_errors.py
==========================================
PRD T18: handle-based satellite tools already catch OpenHandleError at their
open_handle() call sites and surface it verbatim (prior art: open_handle's
verbatim error surfacing, test_open_handle.py). Since parse_tool_result now
classifies more raw shapes into MCPToolError, those same call sites must
also catch MCPToolError — one representative tool per file, not every call
site, mirroring how thoroughly OpenHandleError itself was covered before.
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

REQUIRED_MODULES = ["langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn", "numpy", "xarray", "zarr", "pandas"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite-tool MCP error test dependencies are not installed",
)
class HandleBasedToolsSurfaceClassifiedErrorsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        # A tool-raised error unrecognized by the classifier's known
        # prefixes — pins the contract fallback, not a hand-picked category,
        # so this test doesn't accidentally depend on the prose table.
        async def export_result(handle, workspace_id="default"):
            raise ValueError("harmony: provider GES_DISC rejected retrieval for an unmapped reason")

        self.server = FakeEarthdataMCPServer(build_fake_mcp({"export_result": export_result}))
        self.server.start()
        self.addCleanup(self.server.stop)
        settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def test_plot_singular_surfaces_a_classified_error_as_structured_json(self):
        from tools.satellite_tools.plot_tools import make_plot_singular

        plot_singular = make_plot_singular(self.mcp_tools)
        raw = await plot_singular.ainvoke({"handle": "obs_1", "location": "New Jersey"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "contract")
        self.assertTrue(payload["error"]["message"])

    async def test_compute_statistic_tool_surfaces_a_classified_error_as_structured_json(self):
        from tools.satellite_tools.stat_tools import make_compute_statistic_tool

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({"handle": "obs_1", "location": "New Jersey"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "contract")

    async def test_validate_against_ground_surfaces_a_classified_error_as_structured_json(self):
        from tools.satellite_tools.validation_tools import make_validate_against_ground

        validate_against_ground = make_validate_against_ground(self.mcp_tools)
        raw = await validate_against_ground.ainvoke({"handle": "obs_1", "location": "New Jersey"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "contract")

    async def test_compare_surfaces_a_classified_error_as_structured_json(self):
        from tools.satellite_tools.comparison_tools import make_compare

        compare = make_compare(self.mcp_tools)
        raw = await compare.ainvoke({"handle_a": "obs_1", "handle_b": "obs_2", "mode": "region"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "contract")

    async def test_await_retrieval_tool_surfaces_a_classified_error_as_structured_json(self):
        from tools.satellite_tools.retrieval_tools import make_await_retrieval

        async def get_retrieval_status(job_handle, workspace_id="default"):
            raise ValueError("harmony: provider GES_DISC rejected the status poll for an unmapped reason")

        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp({"get_retrieval_status": get_retrieval_status}))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)

        await_retrieval = make_await_retrieval(tools)
        raw = await await_retrieval.ainvoke({"job_handle": "job_1"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "contract")


if __name__ == "__main__":
    unittest.main()
