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
class WorkspaceBindingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        received = {}

        async def search_datasets(query, filters, workspace_id):
            received["workspace_id"] = workspace_id
            return {"datasets": [], "count": 0}

        self.received = received
        self.server = FakeEarthdataMCPServer(build_fake_mcp({"search_datasets": search_datasets}))
        self.server.start()
        self.addCleanup(self.server.stop)

        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_bind_workspace_injects_workspace_id_at_call_time(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertEqual(self.received["workspace_id"], "user-17")

    async def test_bind_workspace_hides_workspace_id_from_the_schema(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        schema = bound["search_datasets"].args_schema
        properties = schema["properties"] if isinstance(schema, dict) else schema.schema()["properties"]

        self.assertNotIn("workspace_id", properties)
        self.assertIn("query", properties)


if __name__ == "__main__":
    unittest.main()
