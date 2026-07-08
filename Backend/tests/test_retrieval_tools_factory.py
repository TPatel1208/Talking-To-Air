import importlib.util
import json
import os
import sys
import unittest
from dataclasses import replace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "retrieval tools factory test dependencies are not installed",
)
class RetrievalToolsFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def _tools_for(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = replace(
            Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None),
            await_retrieval_poll_min_seconds=0,
            await_retrieval_poll_max_seconds=0,
        )
        raw_tools = await load_raw_mcp_tools(settings)
        return raw_tools, settings

    def _built_tool(self, mcp_tools, name):
        from tools.satellite_tools.factory import build_satellite_tools

        tools = {t.name: t for t in build_satellite_tools(mcp_tools)}
        return tools[name]

    async def test_build_satellite_tools_exposes_safe_retrieve_and_await_retrieval(self):
        from tools.satellite_tools.factory import build_satellite_tools

        mcp_tools, _ = await self._tools_for({})
        names = {t.name for t in build_satellite_tools(mcp_tools)}

        self.assertIn("safe_retrieve", names)
        self.assertIn("await_retrieval", names)

    async def test_safe_retrieve_tool_proceeds_under_the_soft_cap(self):
        async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"estimated_bytes": 100}

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            return {"job_handle": "job_1", "obs_handle": "obs_1"}

        mcp_tools, settings = await self._tools_for({
            "estimate_retrieval_size": estimate_retrieval_size,
            "retrieve_subset": retrieve_subset,
        })

        safe_retrieve = self._built_tool(mcp_tools, "safe_retrieve")
        raw = await safe_retrieve.ainvoke({
            "dataset_handle": "dataset_1",
            "aoi_handle": "aoi_1",
            "time_range": "2024-01-01/2024-01-02",
            "variables": ["no2"],
        })
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "submitted")
        self.assertEqual(payload["job_handle"], "job_1")

    async def test_safe_retrieve_tool_asks_for_confirmation_between_caps(self):
        async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"estimated_bytes": 5 * 1024 ** 3}

        mcp_tools, _ = await self._tools_for({
            "estimate_retrieval_size": estimate_retrieval_size,
        })

        safe_retrieve = self._built_tool(mcp_tools, "safe_retrieve")
        raw = await safe_retrieve.ainvoke({
            "dataset_handle": "dataset_1",
            "aoi_handle": "aoi_1",
            "time_range": "2024-01-01/2024-01-02",
            "variables": ["no2"],
        })
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "needs_confirmation")

    async def test_await_retrieval_tool_returns_terminal_status(self):
        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": job_handle, "status": "ready", "obs_handle": "obs_1"}

        mcp_tools, _ = await self._tools_for({
            "get_retrieval_status": get_retrieval_status,
        })

        await_retrieval = self._built_tool(mcp_tools, "await_retrieval")
        raw = await await_retrieval.ainvoke({"job_handle": "job_1"})
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["obs_handle"], "obs_1")


if __name__ == "__main__":
    unittest.main()
