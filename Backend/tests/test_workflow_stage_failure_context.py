"""
T19 Testing Decisions / story #9: a scripted failure at the coverage stage
yields a T18-classified error answer, and the stream's last narrated stage
is coverage — no later stage narrates over a workflow that never got past
checking coverage.
"""
import importlib.util
import json
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
class CoverageStageFailureContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_coverage_stage_failure_is_classified_and_leaves_coverage_as_the_last_stage(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.workspace import bind_workspace
        from config.settings import Settings
        import utils.streaming as streaming

        async def search_datasets(query, filters, workspace_id):
            return {"dataset_handle": "dataset_1", "short_name": query, "title": query}

        async def define_area_of_interest(location, workspace_id):
            return {"aoi_handle": "aoi_1", "location": location}

        async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id):
            raise ValueError("harmony: provider rejected coverage check")

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "search_datasets": search_datasets,
            "define_area_of_interest": define_area_of_interest,
            "check_coverage": check_coverage,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        raw_tools = await load_raw_mcp_tools(settings)
        tools = bind_workspace(raw_tools, lambda: "test-user")

        seen = []
        token = streaming._status_emitter.set(
            lambda message, *, stage=None, detail=None: seen.append({"stage": stage})
        )
        try:
            await tools["search_datasets"].ainvoke({"query": "no2", "filters": None})
            await tools["define_area_of_interest"].ainvoke({"location": "New Jersey"})
            raw = await tools["check_coverage"].ainvoke({
                "dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "2024-01-01/2024-01-02",
            })
        finally:
            streaming._status_emitter.reset(token)

        payload = json.loads(raw)
        self.assertIn("error", payload)
        self.assertEqual(payload["error"]["category"], "contract")

        stage_sequence = [s["stage"] for s in seen if s["stage"]]
        self.assertEqual(stage_sequence, ["search", "aoi", "coverage"])


if __name__ == "__main__":
    unittest.main()
