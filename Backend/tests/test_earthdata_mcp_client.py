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
class EarthdataMCPClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from fake_earthdata_mcp import ALL_RAW_TOOL_NAMES, build_fake_mcp, FakeEarthdataMCPServer

        self.server = FakeEarthdataMCPServer(build_fake_mcp())
        self.server.start()
        self.addCleanup(self.server.stop)
        self.all_names = set(ALL_RAW_TOOL_NAMES)

    def _settings(self, url=None):
        from config.settings import Settings

        return Settings(earthdata_mcp_url=url or self.server.url, earthdata_mcp_token=None)

    async def test_load_raw_mcp_tools_returns_every_tool_by_name(self):
        from earthdata_mcp.client import load_raw_mcp_tools

        tools = await load_raw_mcp_tools(self._settings())

        self.assertEqual(set(tools.keys()), self.all_names)

    async def test_load_raw_mcp_tools_fails_loud_when_unreachable(self):
        from earthdata_mcp.client import EarthdataMCPUnavailableError, load_raw_mcp_tools

        with self.assertRaises(EarthdataMCPUnavailableError):
            await load_raw_mcp_tools(self._settings(url="http://127.0.0.1:1/mcp"))

    async def test_load_raw_mcp_tools_fails_loud_when_required_tool_missing(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import EarthdataMCPUnavailableError, load_raw_mcp_tools

        mcp = build_fake_mcp(exclude=("get_provenance",))
        server = FakeEarthdataMCPServer(mcp)
        server.start()
        self.addCleanup(server.stop)

        with self.assertRaisesRegex(EarthdataMCPUnavailableError, "get_provenance"):
            await load_raw_mcp_tools(self._settings(url=server.url))

    async def test_load_raw_mcp_tools_fails_loud_when_list_workspace_missing(self):
        # T05's jobs panel composes list_workspace + get_retrieval_status;
        # a stack that can't list a workspace's jobs should fail at boot.
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import EarthdataMCPUnavailableError, load_raw_mcp_tools

        mcp = build_fake_mcp(exclude=("list_workspace",))
        server = FakeEarthdataMCPServer(mcp)
        server.start()
        self.addCleanup(server.stop)

        with self.assertRaisesRegex(EarthdataMCPUnavailableError, "list_workspace"):
            await load_raw_mcp_tools(self._settings(url=server.url))

    async def test_load_raw_mcp_tools_fails_loud_when_convert_format_missing(self):
        # T10's data-download endpoints need format conversion (e.g. NetCDF)
        # over the exported handle; a stack that can't convert formats
        # should fail at boot, not mid-download.
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import EarthdataMCPUnavailableError, load_raw_mcp_tools

        mcp = build_fake_mcp(exclude=("convert_format",))
        server = FakeEarthdataMCPServer(mcp)
        server.start()
        self.addCleanup(server.stop)

        with self.assertRaisesRegex(EarthdataMCPUnavailableError, "convert_format"):
            await load_raw_mcp_tools(self._settings(url=server.url))

if __name__ == "__main__":
    unittest.main()
