import importlib.util
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


def _harmony_fetch_count() -> float:
    """Observations currently recorded on the harmony_fetch_duration_seconds
    histogram. Process-wide state, so callers assert on a delta."""
    from tta_backend.infrastructure.metrics import HARMONY_FETCH_DURATION_SECONDS

    for metric in HARMONY_FETCH_DURATION_SECONDS.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                return sample.value
    return 0.0


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class AwaitRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.infrastructure.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)
        return tools, settings

    async def test_await_retrieval_polls_until_ready(self):
        from tta_backend.services.retrieval_composites import await_retrieval

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

        result = await await_retrieval("job_1", tools, settings=settings)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["obs_handle"], "obs_1")
        self.assertEqual(calls["n"], 3)

    async def test_await_retrieval_returns_failed_status_verbatim_without_raising(self):
        from tta_backend.services.retrieval_composites import await_retrieval

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
        """Retrieval progress narrated as a percentage while the job runs, one
        status line per poll."""
        from tta_backend.services.retrieval_composites import await_retrieval

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
        import tta_backend.utils.streaming as streaming

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

    async def test_await_retrieval_surfaces_a_provider_paused_job_instead_of_polling_to_timeout(self):
        """Live 2026-07-16 (job_142cbb2faa6aecc0): Harmony auto-paused a
        multi-month retrieval at 0% and the MCP kept reporting status
        "running" (its status mapper folds "paused" into RUNNING) with the
        provider's paused text only in `message`. await_retrieval must stop
        polling and return an honest "paused" status with guidance — not
        narrate an ever-climbing timer until the timeout."""
        from tta_backend.services.retrieval_composites import await_retrieval

        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["n"] += 1
            return {
                "job_handle": "job_paused",
                "status": "running",
                "progress": 0,
                "phase": "processing",
                "message": "The job is paused and may be resumed using the provided link",
            }

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        result = await await_retrieval("job_paused", tools, settings=settings)

        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["phase"], "paused at provider")
        self.assertIn("narrow", result["note"].lower())
        self.assertIn("cancel", result["note"].lower())
        # One poll was enough — no spin until the timeout.
        self.assertEqual(calls["n"], 1)

    async def test_await_retrieval_does_not_mistake_a_terminal_message_mentioning_paused(self):
        """A failed job whose provider message happens to contain the word
        "paused" must stay failed — annotation only rescues non-terminal
        statuses."""
        from tta_backend.services.retrieval_composites import await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {
                "job_handle": "job_f",
                "status": "failed",
                "message": "job was paused and then failed upstream",
            }

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        result = await await_retrieval("job_f", tools, settings=settings)

        self.assertEqual(result["status"], "failed")
        self.assertNotIn("note", result)

    async def test_await_retrieval_survives_transient_status_poll_failures(self):
        """A single network blip during a 15-minute Harmony job used to throw
        away the whole await — the agent reported failure while the job
        completed unobserved at the provider. A transient provider_unavailable
        poll outcome must be retried (bounded), not surfaced."""
        from tta_backend.services.retrieval_composites import await_retrieval

        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["n"] += 1
            if calls["n"] <= 2:
                # The MCP's own upstream call dying mid-poll arrives as prose
                # the classifier maps to provider_unavailable (results.py's
                # _TRANSIENT_NETWORK_PATTERNS).
                raise ValueError("All connection attempts failed")
            return {"job_handle": "job_flaky", "status": "ready", "obs_handle": "obs_flaky"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        result = await await_retrieval("job_flaky", tools, settings=settings)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["obs_handle"], "obs_flaky")
        self.assertEqual(calls["n"], 3)

    async def test_await_retrieval_gives_up_after_persistent_poll_failures(self):
        # Bounded, not infinite: a genuinely down MCP must still surface as
        # the classified provider_unavailable error, not retry forever.
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.services.retrieval_composites import POLL_TRANSIENT_FAILURE_LIMIT, await_retrieval

        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["n"] += 1
            raise ValueError("All connection attempts failed")

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        with self.assertRaises(MCPToolError) as ctx:
            await await_retrieval("job_down", tools, settings=settings)

        self.assertEqual(ctx.exception.category, "provider_unavailable")
        self.assertEqual(calls["n"], POLL_TRANSIENT_FAILURE_LIMIT + 1)

    async def test_await_retrieval_does_not_retry_a_non_transient_poll_error(self):
        # Only transient outages are retried — a contract-class failure (a
        # malformed request, a backend bug) must fail loud on the first poll.
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.services.retrieval_composites import await_retrieval

        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            calls["n"] += 1
            raise ValueError("1 validation error for get_retrieval_status")

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        with self.assertRaises(MCPToolError) as ctx:
            await await_retrieval("job_bug", tools, settings=settings)

        self.assertNotEqual(ctx.exception.category, "provider_unavailable")
        self.assertEqual(calls["n"], 1)

    async def test_await_retrieval_times_out_when_job_never_reaches_terminal_state(self):
        from tta_backend.services.retrieval_composites import RetrievalTimeoutError, await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_3", "status": "processing", "progress": 10}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)
        from dataclasses import replace

        settings = replace(settings, await_retrieval_timeout_seconds=0)

        with self.assertRaises(RetrievalTimeoutError):
            await await_retrieval("job_3", tools, settings=settings)

    async def test_a_completed_await_observes_exactly_one_harmony_fetch_duration(self):
        """``observe_harmony_fetch`` must have a real call site, or the
        metrics module advertises a capability the system doesn't have."""
        from tta_backend.services.retrieval_composites import await_retrieval

        responses = [
            {"job_handle": "job_h", "status": "processing", "progress": 40},
            {"job_handle": "job_h", "status": "ready", "obs_handle": "obs_h"},
        ]
        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            data = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return data

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        before = _harmony_fetch_count()
        await await_retrieval("job_h", tools, settings=settings)

        # Exactly one: the whole submission-through-download span, observed
        # once at completion -- not once per poll.
        self.assertEqual(_harmony_fetch_count() - before, 1)

    async def test_a_failed_await_does_not_observe_a_harmony_fetch_duration(self):
        """The histogram is documented as "job duration ... through download
        completion". Feeding it jobs that never downloaded anything would make
        its percentiles describe a different population than they claim."""
        from tta_backend.services.retrieval_composites import await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_hf", "status": "failed", "message": "boom"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        before = _harmony_fetch_count()
        await await_retrieval("job_hf", tools, settings=settings)

        self.assertEqual(_harmony_fetch_count() - before, 0)

    def test_the_poll_backoff_caps_low_enough_not_to_flatten_a_fast_retrieval(self):
        """The backoff doubles 2 -> 4 -> 8 -> cap, so once it saturates the
        reported status is up to a full cap-length behind reality: a job that
        finished at t=30s is still narrated as running until the next poll.

        At the old 15s cap that staleness was 7.5s on average and 15s at worst.
        On a 3-minute Harmony job that is 4% noise, but on a 30-second
        retrieval it is 25-45% — i.e. the backoff cost the most exactly where
        there was the least to hide it, and the fast retrievals this app can
        actually deliver never got to feel fast. The cap buys that back for a
        handful of extra status calls per job.

        Asserted against the *default*, with any ambient env override removed:
        a developer whose .env still carries the old 15 should not see this as
        a code failure (their .env is the thing that is stale)."""
        import os
        from unittest import mock

        from tta_backend.infrastructure.settings import Settings

        with mock.patch.dict(os.environ):
            os.environ.pop("AWAIT_RETRIEVAL_POLL_MAX_SECONDS", None)
            self.assertLessEqual(Settings().await_retrieval_poll_max_seconds, 5)

    async def test_the_status_line_prefers_the_providers_phase_over_the_bare_status(self):
        """`phase` is harmony-retrieval-mcp's qualitative label for a job and
        reads better mid-flight than the durable `status` it qualifies
        ("queued at provider" vs. "submitted"). The jobs panel already prefers
        it for exactly that reason (Frontend/src/utils/jobCard.js), so the chat
        line was showing the worse of two strings it already had in hand."""
        from tta_backend.services.retrieval_composites import await_retrieval

        responses = [
            {"job_handle": "job_p", "status": "submitted", "phase": "queued at provider"},
            {"job_handle": "job_p", "status": "ready", "obs_handle": "obs_p"},
        ]
        calls = {"n": 0}

        async def get_retrieval_status(job_handle, workspace_id):
            data = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return data

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        seen = self._capture_status()
        await await_retrieval("job_p", tools, settings=settings)

        first = [s["message"] for s in seen if s["stage"] == "progress"][0]
        self.assertIn("queued at provider", first)
        self.assertNotIn("submitted", first)

    async def test_a_terminal_poll_ignores_a_stale_phase(self):
        """Mirrors the jobs panel's rule: a terminal response can carry no
        phase at all (a cancel) or a stale one, and rendering "cancelled" under
        a phase that still says "materializing" reads as a contradiction. Once
        terminal, the durable status is the honest label."""
        from tta_backend.services.retrieval_composites import await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_t", "status": "cancelled", "phase": "materializing"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        seen = self._capture_status()
        await await_retrieval("job_t", tools, settings=settings)

        last = [s["message"] for s in seen if s["stage"] == "progress"][-1]
        self.assertIn("cancelled", last)
        self.assertNotIn("materializing", last)

    async def test_the_status_line_says_data_is_being_retrieved(self):
        from tta_backend.services.retrieval_composites import await_retrieval

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_b", "status": "ready", "obs_handle": "obs_b"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        seen = self._capture_status()
        await await_retrieval("job_b", tools, settings=settings)

        self.assertIn("Retrieving data — ready", [s["message"] for s in seen if s["stage"] == "progress"][-1])

    def _capture_status(self):
        """Collect emit_status calls for the duration of one test."""
        import tta_backend.utils.streaming as streaming

        seen = []

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        self.addCleanup(streaming._status_emitter.reset, token)
        return seen

    def _fast_settings(self, settings):
        """Zero backoff, plus a timeout that is a *hang* guard rather than a race.

        The zeroed poll intervals are the only thing that makes these tests
        fast; the timeout is not a deadline anyone here asserts on (the one
        test that does, test_await_retrieval_times_out_when_job_never_reaches_
        terminal_state, sets its own 0). It exists so a poll loop that stops
        terminating fails instead of hanging the suite.

        It was 5s, which read as generous only because polls look free. They
        are not: ``_tools`` hands back ``load_raw_mcp_tools`` output, whose
        tools carry ``session=None``, so every ``ainvoke`` opens a *fresh*
        streamable-HTTP session -- see earthdata_mcp.client.
        open_earthdata_session on why that path exists and what it costs.
        Measured against this fixture's in-process server, that handshake is
        ~1.2-1.6s while the tool call itself is ~30ms, i.e. ~1.4s per poll.
        (The steady-state path production actually uses holds one session
        open and calls in ~35ms, which is why nothing outside these tests
        pays this.)

        await_retrieval only checks the deadline after a *non-terminal* poll,
        so the binding quantity is the clock at the last non-terminal one --
        for the 3-response tests here, poll 2, measured at 2.2-3.0s. Against
        5s that is under 2x, and it degrades exactly the way the failure did:
        at a 3s budget these tests already fail ~1 run in 5, surfacing as
        RetrievalTimeoutError with the progress list truncated to [0, 40].
        That is why the test survived alone and in-class but died deep into a
        full run (position ~1014 of 1359), where the process is slow enough
        for two handshakes to cross 5s -- no state left behind by another
        test, just a budget sized as if the work were free.

        30s is ~10x the measured worst case, and still a bounded failure
        rather than a hang. Production's own default is 900s.
        """
        from dataclasses import replace

        return replace(
            settings,
            await_retrieval_poll_min_seconds=0,
            await_retrieval_poll_max_seconds=0,
            await_retrieval_timeout_seconds=30,
        )


if __name__ == "__main__":
    unittest.main()
