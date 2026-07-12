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

    async def test_list_jobs_reads_handles_filtered_to_type_job_and_maps_real_fields(self):
        """The real list_workspace returns every handle in the workspace
        (jobs, AOIs, datasets, ...) as {handles: [{handle, type, created_at,
        summary}]} — list_jobs must filter to type == "job" and map
        handle -> job_handle rather than reading a "jobs" key the real MCP
        never returns."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {
                        "handle": "job_1", "type": "job", "created_at": "2026-07-01T00:00:00Z",
                        "summary": {"dataset_handle": "TEMPO_NO2"},
                    },
                    {
                        "handle": "aoi_1", "type": "aoi", "created_at": "2026-07-01T00:00:00Z",
                        "summary": {"location": "New Jersey"},
                    },
                    {
                        "handle": "job_2", "type": "job", "created_at": "2026-07-02T00:00:00Z",
                        "summary": {"dataset_handle": "MOD11A1"},
                    },
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

        # The non-job "aoi_1" handle must never surface as a job.
        self.assertEqual({job["job_handle"] for job in jobs}, {"job_1", "job_2"})

        by_handle = {job["job_handle"]: job for job in jobs}
        self.assertEqual(by_handle["job_1"]["dataset_handle"], "TEMPO_NO2")
        self.assertEqual(by_handle["job_1"]["created_at"], "2026-07-01T00:00:00Z")
        self.assertEqual(by_handle["job_1"]["status"], "ready")
        self.assertEqual(by_handle["job_1"]["obs_handle"], "obs_1")
        self.assertEqual(by_handle["job_2"]["status"], "processing")
        self.assertEqual(by_handle["job_2"]["progress"], 40)

    async def test_list_jobs_sorts_active_jobs_first_then_newest_first_within_each_group(self):
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_old_terminal", "type": "job", "created_at": "2026-07-01T00:00:00Z", "summary": {}},
                    {"handle": "job_newest_active", "type": "job", "created_at": "2026-07-03T00:00:00Z", "summary": {}},
                    {"handle": "job_older_active", "type": "job", "created_at": "2026-07-02T00:00:00Z", "summary": {}},
                ]
            }

        statuses = {
            "job_old_terminal": {"job_handle": "job_old_terminal", "status": "ready"},
            "job_newest_active": {"job_handle": "job_newest_active", "status": "processing"},
            "job_older_active": {"job_handle": "job_older_active", "status": "processing"},
        }

        async def get_retrieval_status(job_handle, workspace_id):
            return statuses[job_handle]

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        self.assertEqual(
            [job["job_handle"] for job in jobs],
            ["job_newest_active", "job_older_active", "job_old_terminal"],
        )

    async def test_list_jobs_passes_through_the_mcps_failed_status_message_verbatim(self):
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {
                        "handle": "job_3", "type": "job", "created_at": "2026-07-01T00:00:00Z",
                        "summary": {"dataset_handle": "TEMPO_NO2"},
                    },
                ]
            }

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
