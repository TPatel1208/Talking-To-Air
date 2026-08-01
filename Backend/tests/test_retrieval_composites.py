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


def _harmony_fetch_count() -> float:
    """Observations currently recorded on the harmony_fetch_duration_seconds
    histogram. Process-wide state, so callers assert on a delta."""
    from tta_backend.utils.metrics import HARMONY_FETCH_DURATION_SECONDS

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
        from tta_backend.config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)
        return tools, settings

    async def test_await_retrieval_finalizes_a_pending_variable_choice_into_the_ready_handle(self):
        """T25: the retrieval composite records the model's chosen science
        variable, keyed by the handle the job resolves to, so a later plot/
        stat/compare call inherits it instead of AggregationService.
        to_dataarray refusing a multi-variable file all over again."""
        from tta_backend.services import variable_choice_registry
        from tta_backend.services.retrieval_composites import await_retrieval

        variable_choice_registry._pending.clear()
        variable_choice_registry._choices.clear()
        self.addCleanup(variable_choice_registry._pending.clear)
        self.addCleanup(variable_choice_registry._choices.clear)
        variable_choice_registry.record_pending("job_choice", "Cloud_Fraction")

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_choice", "status": "ready", "obs_handle": "obs_choice"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        await await_retrieval("job_choice", tools, settings=settings)

        self.assertEqual(variable_choice_registry.get("obs_choice"), "Cloud_Fraction")

    async def test_await_retrieval_finalizes_a_pending_requested_scope_into_the_ready_handle(self):
        """T46: the requested scope safe_retrieve/point_timeseries recorded by
        job_handle is promoted onto the obs_/cube_ handle the job resolves to,
        so a later plot can echo it — the same two-step handoff as the T25
        variable choice above."""
        from tta_backend.services import scope_registry
        from tta_backend.services.retrieval_composites import await_retrieval

        scope_registry._pending.clear()
        scope_registry._scopes.clear()
        self.addCleanup(scope_registry._pending.clear)
        self.addCleanup(scope_registry._scopes.clear)
        scope_registry.record_pending("job_scope", {"location": "California", "time_range": "2024-07-15/2024-07-15"})

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_scope", "status": "ready", "obs_handle": "obs_scope"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        await await_retrieval("job_scope", tools, settings=settings)

        self.assertEqual(scope_registry.get("obs_scope")["location"], "California")

    async def test_await_retrieval_does_not_record_a_choice_for_a_failed_job(self):
        from tta_backend.services import variable_choice_registry
        from tta_backend.services.retrieval_composites import await_retrieval

        variable_choice_registry._pending.clear()
        variable_choice_registry._choices.clear()
        self.addCleanup(variable_choice_registry._pending.clear)
        self.addCleanup(variable_choice_registry._choices.clear)
        variable_choice_registry.record_pending("job_failed", "Cloud_Fraction")

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": "job_failed", "status": "failed", "message": "boom"}

        tools, settings = await self._tools({"get_retrieval_status": get_retrieval_status})
        settings = self._fast_settings(settings)

        await await_retrieval("job_failed", tools, settings=settings)

        self.assertIsNone(variable_choice_registry.get("obs_never_ready"))

    async def test_await_retrieval_polls_until_ready_and_emits_progress_in_order(self):
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

        seen = []
        import tta_backend.utils.streaming as streaming

        token = streaming._job_progress_emitter.set(lambda data: seen.append(data))
        try:
            result = await await_retrieval("job_1", tools, settings=settings)
        finally:
            streaming._job_progress_emitter.reset(token)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["obs_handle"], "obs_1")
        self.assertEqual([e["status"] for e in seen], ["queued", "processing", "ready"])

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
        """T19 story #2: retrieval progress narrated as a percentage while
        the job runs, forwarded from the same poll that already drives
        emit_job_progress — one poll, two audiences (job panel + chat
        strip), never two separate polling loops."""
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

        seen = []
        import tta_backend.utils.streaming as streaming

        token = streaming._job_progress_emitter.set(lambda data: seen.append(data))
        try:
            result = await await_retrieval("job_paused", tools, settings=settings)
        finally:
            streaming._job_progress_emitter.reset(token)

        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["phase"], "paused at provider")
        self.assertIn("narrow", result["note"].lower())
        # One poll was enough — no spin until the timeout.
        self.assertEqual(calls["n"], 1)
        # The jobs panel heard about the pause too, guidance included.
        self.assertEqual(seen[-1]["status"], "paused")
        self.assertIn("cancel", seen[-1]["note"].lower())

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
        """T51 regression guard. ``observe_harmony_fetch`` and its histogram
        shipped with no call site at all and sat dead across three releases --
        the metrics module advertising a capability the system didn't have.
        This is the assertion that notices if it goes dead again."""
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
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

        calls = {"retrieve_subset": 0}

        async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
            # None models an estimator that couldn't price the request — the
            # response simply carries no estimated_bytes field at all.
            if estimated_bytes is None:
                return {}
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
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["job_handle"], "job_new")
        self.assertEqual(calls["retrieve_subset"], 1)

    async def test_safe_retrieve_records_the_requested_time_range_scope_for_the_job(self):
        """T46: safe_retrieve only ever sees an opaque aoi_handle (not a place
        name), but it knows the requested time_range — record it so a later
        plot can disclose a single-day request served by a monthly mean."""
        from tta_backend.services import scope_registry
        from tta_backend.services.retrieval_composites import safe_retrieve

        scope_registry._pending.clear()
        scope_registry._scopes.clear()
        self.addCleanup(scope_registry._pending.clear)
        self.addCleanup(scope_registry._scopes.clear)

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        await safe_retrieve(
            "dataset_1", "aoi_1", "2024-07-15/2024-07-15", ["no2"], tools, settings=settings
        )
        scope_registry.finalize("job_new", "obs_new")

        self.assertIn("2024-07-15", scope_registry.get("obs_new")["time_range"])

    async def test_safe_retrieve_pauses_for_confirmation_between_caps(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(result["estimated_bytes"], 6000)
        self.assertEqual(calls["retrieve_subset"], 0)

    async def test_safe_retrieve_proceeds_between_caps_once_confirmed(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings, confirmed=True
        )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(calls["retrieve_subset"], 1)

    async def test_safe_retrieve_emits_estimate_and_submit_stage_status(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        seen = []
        import tta_backend.utils.streaming as streaming

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await safe_retrieve("dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings)
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["estimate", "submit"])

    async def test_safe_retrieve_does_not_emit_submit_when_it_pauses_for_confirmation(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=6000)

        seen = []
        import tta_backend.utils.streaming as streaming

        def _capture(message, *, stage=None, detail=None):
            seen.append({"message": message, "stage": stage, "detail": detail})

        token = streaming._status_emitter.set(_capture)
        try:
            await safe_retrieve("dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings)
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["estimate"])

    async def test_safe_retrieve_omits_variables_for_a_collection_that_does_not_support_subsetting(self):
        """TROPOMI_NO2 (datasets/collections.yaml) is registered with
        supports_variable_subsetting: false -- without this gate,
        safe_retrieve forwards the model's requested variables to
        retrieve_subset unconditionally, and the MCP attempts a doomed
        variable subset before falling back to a full-file retrieval on
        every single call."""
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen_variables = []

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen_variables.append(variables)
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        result = await safe_retrieve(
            "dataset_tropomi", "aoi_1", "2024-01-01/2024-01-02", ["Tropospheric_NO2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(seen_variables, [[]])

    async def test_safe_retrieve_still_forwards_variables_for_a_collection_that_supports_subsetting(self):
        """TEMPO_NO2 is registered with supports_variable_subsetting: true --
        the gate must not suppress variables for collections that actually
        support it."""
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen_variables = []

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen_variables.append(variables)
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        await safe_retrieve(
            "dataset_tempo", "aoi_1", "2024-01-01/2024-01-02", ["vertical_column_troposphere"], tools,
            settings=settings,
        )

        self.assertEqual(seen_variables, [["vertical_column_troposphere"]])

    async def test_safe_retrieve_forwards_variables_unknown_to_the_registry_unchanged(self):
        """A variable name the registry has never heard of must not be
        silently dropped -- default to today's send-it-and-see behavior."""
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen_variables = []

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen_variables.append(variables)
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        await safe_retrieve(
            "dataset_unknown", "aoi_1", "2024-01-01/2024-01-02", ["some_unregistered_variable"], tools,
            settings=settings,
        )

        self.assertEqual(seen_variables, [["some_unregistered_variable"]])

    async def test_safe_retrieve_widens_a_single_date_time_range_to_the_full_day(self):
        """Live 2026-07-11: "June 15, 2024" reached the MCP as
        '2024-06-15/2024-06-15' and Harmony rejected all six jobs with "The
        temporal range's start must be earlier than its stop datetime",
        leaving retrievals to live or die on the OPeNDAP fallback. A
        degenerate start==end date pair means "that whole day" — widen it
        before any provider sees it."""
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen = {}

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen["time_range"] = time_range
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        await safe_retrieve(
            "dataset_1", "aoi_1", "2024-06-15/2024-06-15", ["no2"], tools, settings=settings
        )

        self.assertEqual(seen["time_range"], "2024-06-15T00:00:00/2024-06-15T23:59:59")

    async def test_safe_retrieve_widens_an_equal_timestamp_range_by_one_second(self):
        # A timestamped instant ("midday") is degenerate the same way; give
        # it one second of width instead of inventing a whole-day window.
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen = {}

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen["time_range"] = time_range
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        await safe_retrieve(
            "dataset_1", "aoi_1", "2024-06-15T12:00:00/2024-06-15T12:00:00", ["no2"], tools, settings=settings
        )

        self.assertEqual(seen["time_range"], "2024-06-15T12:00:00/2024-06-15T12:00:01")

    async def test_safe_retrieve_leaves_a_real_time_range_unchanged(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

        seen = {}

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            seen["time_range"] = time_range
            return {"job_handle": "job_new", "obs_handle": "obs_new"}

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000, retrieve_subset=retrieve_subset)

        await safe_retrieve(
            "dataset_1", "aoi_1", "2024-06-01/2024-06-30", ["no2"], tools, settings=settings
        )

        self.assertEqual(seen["time_range"], "2024-06-01/2024-06-30")

    async def test_safe_retrieve_records_a_pending_choice_for_a_single_requested_variable(self):
        from tta_backend.services import variable_choice_registry
        from tta_backend.services.retrieval_composites import safe_retrieve

        variable_choice_registry._pending.clear()
        self.addCleanup(variable_choice_registry._pending.clear)

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["Cloud_Fraction"], tools, settings=settings,
        )

        self.assertEqual(variable_choice_registry._pending[result["job_handle"]][0], "Cloud_Fraction")

    async def test_safe_retrieve_records_no_pending_choice_for_multiple_requested_variables(self):
        """More than one requested variable is not an unambiguous choice --
        must not poison the registry with a guess when the file later opens
        multi-variable."""
        from tta_backend.services import variable_choice_registry
        from tta_backend.services.retrieval_composites import safe_retrieve

        variable_choice_registry._pending.clear()
        self.addCleanup(variable_choice_registry._pending.clear)

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["Cloud_Fraction", "Aerosol_Optical_Depth"], tools,
            settings=settings,
        )

        self.assertNotIn(result["job_handle"], variable_choice_registry._pending)

    async def test_safe_retrieve_records_the_science_variable_when_a_qa_flag_rides_along(self):
        """T25 review #2: a standard TEMPO retrieval requests the science
        variable *and* main_data_quality_flag together (both group-qualified).
        Counting raw ``variables`` would see 2 and record nothing, so the
        opened 2-var file later refuses. The QA flag isn't a science choice --
        excluding it (by bare leaf) leaves a single science variable still
        worth recording."""
        from tta_backend.services import variable_choice_registry
        from tta_backend.services.retrieval_composites import safe_retrieve

        variable_choice_registry._pending.clear()
        self.addCleanup(variable_choice_registry._pending.clear)

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=1000)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02",
            ["product/vertical_column_troposphere", "product/main_data_quality_flag"],
            tools, settings=settings,
        )

        self.assertEqual(
            variable_choice_registry._pending[result["job_handle"]][0],
            "product/vertical_column_troposphere",
        )

    async def test_safe_retrieve_pauses_for_confirmation_when_size_cannot_be_estimated(self):
        """A missing estimate must not sail past the caps as if it were zero
        bytes — "couldn't price it" is not "free". The old
        ``estimate.get("estimated_bytes", 0)`` submitted unconditionally,
        making the guardrail silently inert exactly when the estimator was
        blind."""
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=None)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings
        )

        self.assertEqual(result["status"], "needs_confirmation")
        self.assertIsNone(result["estimated_bytes"])
        self.assertIn("could not estimate", result["message"].lower())
        self.assertEqual(calls["retrieve_subset"], 0)

    async def test_safe_retrieve_proceeds_without_an_estimate_once_confirmed(self):
        # The researcher can still say "go ahead" — the bundle-open size gate
        # (services/open_handle.py) remains the backstop at open time.
        from tta_backend.services.retrieval_composites import safe_retrieve

        tools, settings, calls = await self._tools_and_settings(estimated_bytes=None)

        result = await safe_retrieve(
            "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], tools, settings=settings, confirmed=True
        )

        self.assertEqual(result["status"], "submitted")
        self.assertIsNone(result["estimated_bytes"])
        self.assertEqual(calls["retrieve_subset"], 1)

    async def test_safe_retrieve_refuses_above_hard_cap_even_if_confirmed(self):
        from tta_backend.services.retrieval_composites import safe_retrieve

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
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings
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
        from tta_backend.services.retrieval_composites import point_timeseries

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

    async def test_point_timeseries_records_the_requested_location_and_time_range_scope(self):
        """T46: point_timeseries has both the place name and the time range, so
        it records the full requested scope — echoed onto the cube handle the
        job resolves to (via await_retrieval), disclosable end-to-end."""
        from tta_backend.services import scope_registry
        from tta_backend.services.retrieval_composites import point_timeseries

        scope_registry._pending.clear()
        scope_registry._scopes.clear()
        self.addCleanup(scope_registry._pending.clear)
        self.addCleanup(scope_registry._scopes.clear)

        async def define_area_of_interest(location, workspace_id):
            return {"handle": "aoi_newark", "location": location}

        async def retrieve_timeseries(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
            return {"job_handle": "job_ts_1"}

        async def get_retrieval_status(job_handle, workspace_id):
            return {"job_handle": job_handle, "status": "ready", "obs_handle": "cube_ts_1"}

        tools, settings = await self._tools_and_settings({
            "define_area_of_interest": define_area_of_interest,
            "retrieve_timeseries": retrieve_timeseries,
            "get_retrieval_status": get_retrieval_status,
        })

        await point_timeseries(
            "dataset_1", "Newark, NJ", "2024-01-01/2024-01-31", "no2", tools, settings=settings,
        )

        recorded = scope_registry.get("cube_ts_1")
        self.assertEqual(recorded["location"], "Newark, NJ")
        self.assertIn("2024-01-01", recorded["time_range"])

    async def test_point_timeseries_refuses_an_over_span_request_without_any_mcp_calls(self):
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from tta_backend.services.retrieval_composites import point_timeseries

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
        from tta_backend.services.retrieval_composites import point_timeseries

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
