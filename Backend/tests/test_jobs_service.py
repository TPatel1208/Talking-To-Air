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
    def setUp(self):
        # The terminal-status cache is process-global; clear it between tests
        # so one test's cached rows never leak into another's fan-out count.
        from services.jobs_service import clear_terminal_status_cache

        clear_terminal_status_cache()

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

    async def test_list_jobs_surfaces_a_provider_paused_job_as_paused_with_guidance(self):
        """Live 2026-07-16 (job_142cbb2faa6aecc0): a Harmony auto-paused job
        reports status "running" from the MCP forever, with the pause visible
        only in the provider message. The panel row must say paused (with the
        cancel-and-narrow note), not "Processing — 0%" indefinitely."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_paused", "type": "job", "created_at": "2026-07-16T00:00:00Z", "summary": {}},
                ]
            }

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": "job_paused",
                "status": "running",
                "progress": 0,
                "phase": "processing",
                "message": "The job is paused and may be resumed using the provided link",
            }

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        self.assertEqual(jobs[0]["status"], "paused")
        self.assertEqual(jobs[0]["phase"], "paused at provider")
        self.assertIn("cancel", jobs[0]["note"].lower())

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

    async def test_list_jobs_passes_through_prd021s_enriched_status_fields_untouched(self):
        """list_jobs is a thin composite (entry | status merge) — PRD 021's
        curated request_spec slice on get_retrieval_status must reach the
        frontend verbatim, with no backend-side reshaping (T27)."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {
                        "handle": "job_4", "type": "job", "created_at": "2026-07-01T00:00:00Z",
                        "summary": {"dataset_handle": "TEMPO_NO2"},
                    },
                ]
            }

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": "job_4",
                "status": "running",
                "progress": 40,
                "phase": "processing",
                "short_name": "TEMPO_NO2_L3",
                "variables": ["product/vertical_column_troposphere"],
                "aoi_bbox": [-75.5, 39.5, -74.0, 41.0],
                "time_range": "2026-06-01T00:00:00/2026-06-30T23:59:59",
                "provider": "harmony",
                "output_format": "application/netcdf4",
                "granule_count": 30,
            }

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        self.assertEqual(jobs[0]["short_name"], "TEMPO_NO2_L3")
        self.assertEqual(jobs[0]["variables"], ["product/vertical_column_troposphere"])
        self.assertEqual(jobs[0]["aoi_bbox"], [-75.5, 39.5, -74.0, 41.0])
        self.assertEqual(jobs[0]["time_range"], "2026-06-01T00:00:00/2026-06-30T23:59:59")
        self.assertEqual(jobs[0]["provider"], "harmony")
        self.assertEqual(jobs[0]["output_format"], "application/netcdf4")
        self.assertEqual(jobs[0]["granule_count"], 30)

    async def test_list_jobs_degrades_one_unreadable_status_instead_of_failing_the_panel(self):
        """The status fan-out is fault-isolated: a single job whose
        get_retrieval_status returns an error envelope degrades to a
        status:"error" row, and every healthy sibling still lists."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_ok", "type": "job", "created_at": "2026-07-02T00:00:00Z", "summary": {}},
                    {"handle": "job_bad", "type": "job", "created_at": "2026-07-01T00:00:00Z", "summary": {}},
                ]
            }

        async def get_retrieval_status(job_handle, workspace_id):
            if job_handle == "job_bad":
                return {"error": {"category": "provider_unavailable", "message": "status read failed"}}
            return {"job_handle": "job_ok", "status": "running", "progress": 20, "phase": "processing"}

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        jobs = await list_jobs(tools)

        by_handle = {job["job_handle"]: job for job in jobs}
        self.assertEqual(len(jobs), 2)
        self.assertEqual(by_handle["job_ok"]["status"], "running")
        self.assertEqual(by_handle["job_bad"]["status"], "error")

    async def test_list_jobs_caches_terminal_status_and_skips_the_repeat_mcp_call(self):
        """A terminal job's status is immutable, so re-polling the panel must
        not re-issue get_retrieval_status for it — that per-job fan-out, run on
        the frontend's 15s poll, drove hundreds of MCP round-trips/min on a
        large workspace for data that can't have changed. Only non-terminal
        jobs incur a status call on subsequent polls."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_done", "type": "job", "created_at": "2026-07-01T00:00:00Z", "summary": {}},
                    {"handle": "job_running", "type": "job", "created_at": "2026-07-02T00:00:00Z", "summary": {}},
                ]
            }

        calls = {"job_done": 0, "job_running": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls[job_handle] += 1
            if job_handle == "job_done":
                return {"job_handle": "job_done", "status": "ready", "progress": 100, "obs_handle": "obs_done"}
            return {"job_handle": "job_running", "status": "running", "progress": 40, "phase": "processing"}

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        await list_jobs(tools)
        second = await list_jobs(tools)

        # Terminal job: fetched exactly once, then served from the cache.
        self.assertEqual(calls["job_done"], 1)
        # Active job: re-fetched every poll — its status can still change.
        self.assertEqual(calls["job_running"], 2)

        # The cached terminal row is still present and correct on the 2nd poll.
        by_handle = {job["job_handle"]: job for job in second}
        self.assertEqual(set(by_handle), {"job_done", "job_running"})
        self.assertEqual(by_handle["job_done"]["status"], "ready")
        self.assertEqual(by_handle["job_done"]["progress"], 100)
        self.assertEqual(by_handle["job_done"]["obs_handle"], "obs_done")

    async def test_list_jobs_caches_a_not_found_dead_handle_and_stops_re_polling_it(self):
        """A handle list_workspace still lists but get_retrieval_status reports
        as not_found is a dead, evicted job — permanently. On a long-lived
        workspace these dominate (live repro: 134 not_found vs 46 real jobs),
        so re-polling them every 15s was the bulk of the fan-out. They must be
        cached like a terminal status: fetched once, then served from cache."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_dead", "type": "job", "created_at": "2026-01-01T00:00:00Z", "summary": {}},
                ]
            }

        calls = {"job_dead": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["job_dead"] += 1
            # The MCP models a purged handle as a structured passthrough, not
            # an error (results._STRUCTURED_PASSTHROUGH_STATUSES).
            return {"job_handle": "job_dead", "status": "not_found", "message": "Unknown handle."}

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        first = await list_jobs(tools)
        await list_jobs(tools)

        self.assertEqual(first[0]["status"], "not_found")
        self.assertEqual(calls["job_dead"], 1)

    async def test_list_jobs_does_not_cache_a_nonterminal_or_paused_status(self):
        """Paused and still-running jobs are not immutable — they must stay on
        the per-poll fan-out so a resume/completion is picked up, and an
        errored status read must be retried, not frozen."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_paused", "type": "job", "created_at": "2026-07-01T00:00:00Z", "summary": {}},
                ]
            }

        calls = {"job_paused": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["job_paused"] += 1
            return {
                "job_handle": "job_paused", "status": "running", "progress": 0, "phase": "processing",
                "message": "The job is paused and may be resumed using the provided link",
            }

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        first = await list_jobs(tools)
        await list_jobs(tools)

        # annotate_paused surfaces it as "paused" (non-terminal) — never cached.
        self.assertEqual(first[0]["status"], "paused")
        self.assertEqual(calls["job_paused"], 2)

    async def test_list_jobs_degrades_one_raw_exception_instead_of_failing_the_panel(self):
        """Fault isolation must hold for *any* failure mode a status call
        raises, not just ones the MCP adapter classifies as MCPToolError --
        a raw exception (e.g. a transport-level error the adapter doesn't
        wrap, or a bug in parsing the response) must degrade that single row
        instead of blowing up the whole asyncio.gather and blanking every
        healthy sibling."""
        from services.jobs_service import list_jobs

        async def list_workspace(workspace_id):
            return {
                "handles": [
                    {"handle": "job_ok", "type": "job", "created_at": "2026-07-02T00:00:00Z", "summary": {}},
                    {"handle": "job_bad", "type": "job", "created_at": "2026-07-01T00:00:00Z", "summary": {}},
                ]
            }

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_ok", "status": "running", "progress": 20, "phase": "processing"}

        tools = await self._tools({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
        })

        real_status_tool = tools["get_retrieval_status"]

        class RaisesRawExceptionForJobBad:
            async def ainvoke(self, args):
                if args["job_handle"] == "job_bad":
                    raise RuntimeError("connection reset by peer")
                return await real_status_tool.ainvoke(args)

        tools = {**tools, "get_retrieval_status": RaisesRawExceptionForJobBad()}

        jobs = await list_jobs(tools)

        by_handle = {job["job_handle"]: job for job in jobs}
        self.assertEqual(len(jobs), 2)
        self.assertEqual(by_handle["job_ok"]["status"], "running")
        self.assertEqual(by_handle["job_bad"]["status"], "error")


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

    async def test_cancel_job_passes_through_prd021s_upstream_flag_untouched(self):
        """cancel_job is a thin proxy — PRD 021's `upstream` outcome
        ("requested"/"unsupported"/"already_terminal"/"error") must reach the
        frontend verbatim so the cancel UX can render an honest line (T27)."""
        from services.jobs_service import cancel_job

        async def cancel_retrieval(job_handle, workspace_id):
            return {"job_handle": job_handle, "status": "cancelled", "cancelled": True, "upstream": "requested"}

        tools = await self._tools({"cancel_retrieval": cancel_retrieval})

        result = await cancel_job("job_1", tools)

        self.assertEqual(result["upstream"], "requested")


if __name__ == "__main__":
    unittest.main()
