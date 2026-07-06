import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class EarthdataToolsetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        self.server = FakeEarthdataMCPServer(build_fake_mcp())
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings

        self.settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)

    async def test_curated_model_tools_matches_the_curated_list_exactly(self):
        from earthdata_mcp.client import CURATED_TOOL_NAMES
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        model_tools = curated_model_tools(tools)

        self.assertEqual({t.name for t in model_tools}, set(CURATED_TOOL_NAMES))

    async def test_curated_model_tools_excludes_internal_and_hidden_tools(self):
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        model_tool_names = {t.name for t in curated_model_tools(tools)}

        # align is internal as of T08 (the compare tool's period mode calls
        # it directly) but still never model-facing, same as the other
        # internal/hidden composite plumbing.
        for hidden in ("retrieve_subset", "estimate_retrieval_size", "retrieve_data", "align", "cancel_retrieval", "convert_format"):
            self.assertNotIn(hidden, model_tool_names)


if __name__ == "__main__":
    unittest.main()
