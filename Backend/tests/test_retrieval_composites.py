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
class AwaitRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)
        return tools, settings

    async def test_await_retrieval_polls_until_materialized_and_emits_progress_in_order(self):
        from services.retrieval_composites import await_retrieval

        responses = [
            {"job_handle": "job_1", "status": "queued", "progress": 0, "phase": "submitting", "message": None},
            {"job_handle": "job_1", "status": "processing", "progress": 40, "phase": "materializing", "message": "40%"},
            {"job_handle": "job_1", "status": "materialized", "progress": 100, "phase": "done", "obs_handle": "obs_1"},
        ]
        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            data = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return data

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        seen = []
        import utils.streaming as streaming

        token = streaming._job_progress_emitter.set(lambda data: seen.append(data))
        try:
            result = await await_retrieval("job_1", tools, settings=settings)
        finally:
            streaming._job_progress_emitter.reset(token)

        self.assertEqual(result["status"], "materialized")
        self.assertEqual(result["obs_handle"], "obs_1")
        self.assertEqual([e["status"] for e in seen], ["queued", "processing", "materialized"])

    async def test_await_retrieval_returns_failed_status_verbatim_without_raising(self):
        from services.retrieval_composites import await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": "job_2",
                "status": "failed",
                "message": "harmony: provider GES_DISC rejected request: invalid bbox",
            }

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        result = await await_retrieval("job_2", tools, settings=settings)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["message"], "harmony: provider GES_DISC rejected request: invalid bbox")

    async def test_await_retrieval_times_out_when_job_never_reaches_terminal_state(self):
        from services.retrieval_composites import RetrievalTimeoutError, await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_3", "status": "processing", "progress": 10}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)
        from dataclasses import replace

        settings = replace(settings, await_retrieval_timeout_seconds=0)

        with self.assertRaises(RetrievalTimeoutError):
            await await_retrieval("job_3", tools, settings=settings)

    def _fast_settings(self, settings):
        from dataclasses import replace

        return replace(
            settings,
            await_retrieval_poll_min_seconds=0,
            await_retrieval_poll_max_seconds=0,
            await_retrieval_timeout_seconds=5,
        )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class SafeRetrieveTests(unittest.IsolatedAsyncioTestCase):
    async def _tools_and_settings(self, estimated_bytes, retrieve_subset=None):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings
        from dataclasses import replace

        calls = {"retrieve_subset": 0}

        async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"estimated_bytes": estimated_bytes}

        async def default_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            calls["retrieve_subset"] += 1
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "estimate_retrieval_size": estimate_retrieval_size,
            "retrieve_subset": retrieve_subset or default_retrieve_subset,
        }))
        server.start()
        self.addCleanup(server.stop)

        settings = Settings(
            earthdata_mcp_url=server.url,
            earthdata_mcp_token=None,
            retrieval_soft_cap_bytes=2000,
            retrieval_hard_cap_bytes=10000,
        )
        tools = await load_raw_mcp_tools(settings)
        return tools, settings, calls

    async def test_safe_retrieve_proceeds_automatically_at_or_below_soft_cap(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["job_handle"], "job_new")
        self.assertEqual(calls["retrieve_subset"], 1)

    async def test_safe_retrieve_pauses_for_confirmation_between_caps(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(result["estimated_bytes"], 6000)
        self.assertEqual(calls["retrieve_subset"], 0)

    async def test_safe_retrieve_proceeds_between_caps_once_confirmed(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings, confirmed=True
        )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(calls["retrieve_subset"], 1)

    async def test_safe_retrieve_refuses_above_hard_cap_even_if_confirmed(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=50000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings, confirmed=True
        )

        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["estimated_bytes"], 50000)
        self.assertIn("narrow", result["message"].lower())
        self.assertEqual(calls["retrieve_subset"], 0)


if __name__ == "__main__":
    unittest.main()
