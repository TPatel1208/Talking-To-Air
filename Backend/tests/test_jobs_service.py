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
class ListJobsTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_list_jobs_fans_out_status_per_handle_from_list_workspace(self):
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "jobs": [
                    {"job_handle": "job_1", "dataset": "TEMPO_NO2", "submitted_at": "2026-07-01T00:00:00Z"},
                    {"job_handle": "job_2", "dataset": "MOD11A1", "submitted_at": "2026-07-02T00:00:00Z"},
                ]
            }

        statuses = {
            "job_1": {"job_handle": "job_1", "status": "ready", "progress": 100, "phase": "done", "obs_handle": "obs_1"},
            "job_2": {"job_handle": "job_2", "status": "processing", "progress": 40, "phase": "materializing"},
        }

        async def get_retrieval_status(job_handle, workspace_id):
            return statuses[job_handle]

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["job_handle"], "job_1")
        self.assertEqual(jobs[0]["dataset"], "TEMPO_NO2")
        self.assertEqual(jobs[0]["status"], "ready")
        self.assertEqual(jobs[0]["obs_handle"], "obs_1")
        self.assertEqual(jobs[1]["status"], "processing")
        self.assertEqual(jobs[1]["progress"], 40)

    async def test_list_jobs_passes_through_the_mcps_failed_status_message_verbatim(self):
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {"jobs": [{"job_handle": "job_3", "dataset": "TEMPO_NO2", "submitted_at": "2026-07-01T00:00:00Z"}]}

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": "job_3",
                "status": "failed",
                "message": "harmony: provider GES_DISC rejected request: invalid bbox",
            }

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        self.assertEqual(jobs[0]["status"], "failed")
        self.assertEqual(jobs[0]["message"], "harmony: provider GES_DISC rejected request: invalid bbox")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class CancelJobTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_cancel_job_proxies_the_mcps_cancel_tool(self):
        from services.jobs_service import cancel_job

        calls = {}

        async def cancel_retrieval(job_handle, workspace_id):
            calls["job_handle"] = job_handle
            return {"job_handle": job_handle, "status": "cancelled"}

        tools = await self._tools({"cancel_retrieval": cancel_retrieval})

        result = await cancel_job("job_1", tools)

        self.assertEqual(result, {"job_handle": "job_1", "status": "cancelled"})
        self.assertEqual(calls["job_handle"], "job_1")


if __name__ == "__main__":
    unittest.main()
