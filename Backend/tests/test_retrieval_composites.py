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

    async def test_await_retrieval_polls_until_ready_and_emits_progress_in_order(self):
        from services.retrieval_composites import await_retrieval

        responses = [
            {"job_handle": "job_1", "status": "queued", "progress": 0, "phase": "submitting", "message": None},
            {"job_handle": "job_1", "status": "processing", "progress": 40, "phase": "materializing", "message": "40%"},
            {"job_handle": "job_1", "status": "ready", "progress": 100, "phase": "done", "obs_handle": "obs_1"},
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

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["obs_handle"], "obs_1")
        self.assertEqual([e["status"] for e in seen], ["queued", "processing", "ready"])

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

    async def test_await_retrieval_forwards_each_polls_progress_as_a_stage_status(self):
        """T19 story #2: retrieval progress narrated as a percentage while
        the job runs, forwarded from the same poll that already drives
        emit_job_progress — one poll, two audiences (job panel + chat
        strip), never two separate polling loops."""
        from services.retrieval_composites import await_retrieval

        responses = [
            {"job_handle": "job_1", "status": "queued", "progress": 0, "phase": "submitting"},
            {"job_handle": "job_1", "status": "processing", "progress": 40, "phase": "materializing"},
            {"job_handle": "job_1", "status": "ready", "progress": 100, "phase": "done", "obs_handle": "obs_1"},
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

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await await_retrieval("job_1", tools, settings=settings)
        finally:
            streaming._status_emitter.reset(token)

        stage_events = [s for s in seen if s["stage"] == "progress"]
        self.assertEqual(len(stage_events), 3)
        self.assertEqual([s["detail"] for s in stage_events], [0, 40, 100])

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

    async def test_safe_retrieve_emits_estimate_and_submit_stage_status(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        seen = []
        import utils.streaming as streaming

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await safe_retrieve("dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings)
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["estimate", "submit"])

    async def test_safe_retrieve_does_not_emit_submit_when_it_pauses_for_confirmation(self):
        from services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        seen = []
        import utils.streaming as streaming

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await safe_retrieve("dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings)
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["estimate"])

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


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class PointTimeseriesTests(unittest.IsolatedAsyncioTestCase):
    """T20: the point-timeseries composite — resolve AOI, gate the
    requested span, submit a point-sampled retrieve_timeseries call, and
    await it to a terminal state. Chart/open concerns live in the tool
    wrapper (tools/satellite_tools/retrieval_tools.py); this only covers
    the retrieval mechanics, mirroring safe_retrieve+await_retrieval."""

    async def _tools_and_settings(self, handlers, **settings_kwargs):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings
        from dataclasses import replace

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = replace(
            Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None),
            await_retrieval_poll_min_seconds=0,
            await_retrieval_poll_max_seconds=0,
            **settings_kwargs,
        )
        tools = await load_raw_mcp_tools(settings)
        return tools, settings

    async def test_point_timeseries_resolves_aoi_submits_point_sampled_retrieval_and_awaits_to_ready(self):
        from services.retrieval_composites import point_timeseries

        aoi_calls = []
        submit_calls = []

        async def define_area_of_interest(location, workspace_id):
            aoi_calls.append(location)
            return {"handle": "aoi_newark", "location": location}

        async def retrieve_timeseries(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
            submit_calls.append({
                "dataset_handle": dataset_handle, "time_range": time_range, "variables": variables,
                "aoi_handle": aoi_handle, "point_sample": point_sample,
            })
            return {"job_handle": "job_ts_1"}

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": job_handle, "status": "ready", "obs_handle": "cube_ts_1"}

        tools, settings = await self._tools_and_settings({
            "define_area_of_interest": define_area_of_interest,
            "retrieve_timeseries": retrieve_timeseries,
            "get_retrieval_status": get_retrieval_status,
        })

        result = await point_timeseries(
            "dataset_1", "Newark, NJ", "2024-01-01/2024-01-31", "no2", tools, settings=settings,
        )

        self.assertEqual(aoi_calls, ["Newark, NJ"])
        self.assertEqual(len(submit_calls), 1)
        self.assertEqual(submit_calls[0]["aoi_handle"], "aoi_newark")
        self.assertEqual(submit_calls[0]["variables"], ["no2"])
        self.assertTrue(submit_calls[0]["point_sample"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["obs_handle"], "cube_ts_1")
        self.assertEqual(result["aoi_handle"], "aoi_newark")

    async def test_point_timeseries_refuses_an_over_span_request_without_any_mcp_calls(self):
        from earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from services.retrieval_composites import point_timeseries

        calls = []

        async def define_area_of_interest(location, workspace_id):
            calls.append("define_area_of_interest")
            return {"handle": "aoi_1", "location": location}

        async def retrieve_timeseries(**kwargs):
            calls.append("retrieve_timeseries")
            return {"job_handle": "job_ts_1"}

        tools, settings = await self._tools_and_settings(
            {
                "define_area_of_interest": define_area_of_interest,
                "retrieve_timeseries": retrieve_timeseries,
            },
            retrieval_max_timeseries_days=30,
        )

        with self.assertRaises(MCPToolError) as ctx:
            await point_timeseries(
                "dataset_1", "Newark, NJ", "2020-01-01/2024-01-31", "no2", tools, settings=settings,
            )

        self.assertEqual(ctx.exception.category, CATEGORY_TOO_LARGE)
        self.assertIsNotNone(ctx.exception.suggestion)
        self.assertEqual(calls, [])

    async def test_point_timeseries_returns_a_failed_job_verbatim_without_raising(self):
        from services.retrieval_composites import point_timeseries

        async def define_area_of_interest(location, workspace_id):
            return {"handle": "aoi_1", "location": location}

        async def retrieve_timeseries(**kwargs):
            return {"job_handle": "job_ts_failed"}

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": job_handle,
                "status": "failed",
                "message": "appeears: provider rejected point-sample request",
            }

        tools, settings = await self._tools_and_settings({
            "define_area_of_interest": define_area_of_interest,
            "retrieve_timeseries": retrieve_timeseries,
            "get_retrieval_status": get_retrieval_status,
        })

        result = await point_timeseries(
            "dataset_1", "Newark, NJ", "2024-01-01/2024-01-31", "no2", tools, settings=settings,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["message"], "appeears: provider rejected point-sample request")


if __name__ == "__main__":
    unittest.main()
