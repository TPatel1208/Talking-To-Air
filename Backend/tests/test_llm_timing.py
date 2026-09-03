"""Model-latency and graph-superstep timing.

A 2026-08-07 trace of a 372s plot turn accounted for the Harmony retrieval
wait (171s) and the data pipeline (61s) and left 107s -- 29% of the turn --
attributable to nothing: T51's phase vocabulary is all data work, so time
spent waiting on the model had no series to land in.

These cover the two phases that close that hole: ``llm_call`` (one provider
request, from the LangChain callback pair) and ``agent_step`` (one LangGraph
superstep, from the gap between ``updates`` chunks).

``llm_call`` also carries the provider's token accounting -- input, output,
and the cached-prefix read that says whether the constant sub-agent prefix is
hitting the provider's prompt cache. The rule those tests exist to pin is
that an *absent* count is never recorded as a zero: a working cache and a
missing usage block must not produce the same number.

Like the timer they record through, both sit on the hot path, so the
governing constraint is the same: a telemetry failure must never be the
reason a turn fails.
"""
import importlib.util
import os
import sys
import unittest
from uuid import uuid4


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _phase_samples(phase: str) -> tuple[float, float]:
    """(count, sum) currently recorded for ``phase``. Histograms are
    process-wide state, so every test below asserts on a delta."""
    from tta_backend.utils.metrics import PIPELINE_PHASE_DURATION_SECONDS

    count = total = 0.0
    for metric in PIPELINE_PHASE_DURATION_SECONDS.collect():
        for sample in metric.samples:
            if sample.labels.get("phase") != phase:
                continue
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return count, total


def _token_total(model: str, kind: str, agent_type: str = "unknown") -> float:
    """Tokens currently counted for this model/agent_type/kind. Counters are
    process-wide state, so every test below asserts on a delta."""
    from tta_backend.utils.metrics import LLM_TOKENS_TOTAL

    want = {"model": model, "agent_type": agent_type, "kind": kind}
    for metric in LLM_TOKENS_TOTAL.collect():
        for sample in metric.samples:
            if not sample.name.endswith("_total"):
                continue
            if all(sample.labels.get(k) == v for k, v in want.items()):
                return sample.value
    return 0.0


def _llm_result(usage):
    """An LLMResult shaped the way a chat model delivers one: the usage block
    rides on the AIMessage of the first generation."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    usage = dict(usage)
    # LangChain's UsageMetadata requires total_tokens; the tests below care
    # about the three fields this module reads, so fill the rest in here
    # rather than restating it in every case.
    usage.setdefault("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    message = AIMessage(content="ok", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class RecordPhaseTests(unittest.TestCase):
    """``record_phase`` is the seam for spans whose start and end arrive as
    two separate events, which the context manager cannot express."""

    def test_a_measured_span_reaches_the_histogram(self):
        from tta_backend.utils.phase_timing import record_phase

        before = _phase_samples("llm_call")
        record_phase("llm_call", 2.5)
        after = _phase_samples("llm_call")

        self.assertEqual(after[0] - before[0], 1)
        self.assertAlmostEqual(after[1] - before[1], 2.5, places=5)

    def test_context_reaches_the_log_event(self):
        from tta_backend.utils.phase_timing import record_phase

        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            record_phase("llm_call", 1.0, model="gemma-4-31b-it", outcome="success")

        record = captured.records[-1]
        self.assertEqual(record._event, "phase_timing")
        self.assertEqual(record._phase, "llm_call")
        self.assertEqual(record._model, "gemma-4-31b-it")
        self.assertEqual(record._outcome, "success")

    def test_a_context_key_may_be_named_like_a_parameter(self):
        """phase/duration_seconds are positional-only precisely so a call
        site is never forced to rename its own context field."""
        from tta_backend.utils.phase_timing import record_phase

        record_phase("llm_call", 1.0, phase="model", duration_seconds=3)  # must not raise

    def test_a_broken_metrics_backend_degrades_to_a_no_op(self):
        from unittest.mock import patch

        from tta_backend.utils.phase_timing import record_phase

        with patch(
            "tta_backend.utils.metrics.observe_phase_duration",
            side_effect=RuntimeError("registry down"),
        ):
            record_phase("llm_call", 1.0)  # must not raise

    def test_the_context_manager_still_records_through_this_seam(self):
        """_finish delegates here rather than duplicating the log shape;
        that refactor must not have changed what the timer emits."""
        import time

        from tta_backend.utils.phase_timing import phase_timer

        before = _phase_samples("crop")
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            with phase_timer("crop", cells_in=10) as ctx:
                ctx["cells_out"] = 4
                time.sleep(0.02)
        after = _phase_samples("crop")

        self.assertEqual(after[0] - before[0], 1)
        record = captured.records[-1]
        self.assertEqual(record._phase, "crop")
        self.assertEqual(record._cells_in, 10)
        self.assertEqual(record._cells_out, 4)
        self.assertIsInstance(record._thread_id, int)
        self.assertGreaterEqual(record._duration_seconds, 0.015)


@unittest.skipIf(
    importlib.util.find_spec("langchain_core") is None,
    "langchain_core is not installed",
)
class LlmTimingCallbackTests(unittest.TestCase):
    def _callback(self):
        from tta_backend.utils.llm_timing import LlmTimingCallback

        return LlmTimingCallback()

    def test_a_chat_model_call_records_its_span(self):
        """on_chat_model_start, not on_llm_start, is what a chat model
        raises -- the pair this backend actually exercises."""
        import time

        callback = self._callback()
        run_id = uuid4()

        before = _phase_samples("llm_call")
        callback.on_chat_model_start({}, [], run_id=run_id)
        time.sleep(0.05)
        callback.on_llm_end(None, run_id=run_id)
        after = _phase_samples("llm_call")

        self.assertEqual(after[0] - before[0], 1)
        self.assertGreaterEqual(after[1] - before[1], 0.04)

    def test_a_completion_model_call_records_its_span(self):
        callback = self._callback()
        run_id = uuid4()

        before = _phase_samples("llm_call")
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 1)

    def test_a_failed_call_is_still_timed(self):
        """A call that spent 90s retrying and then raised is exactly what a
        'why was that turn slow' investigation needs."""
        callback = self._callback()
        run_id = uuid4()

        before = _phase_samples("llm_call")
        callback.on_chat_model_start({}, [], run_id=run_id)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_error(ValueError("429 quota"), run_id=run_id)

        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 1)
        record = captured.records[-1]
        self.assertEqual(record._outcome, "error")
        self.assertEqual(record._error_type, "ValueError")

    def test_the_model_name_reaches_the_log(self):
        """Attribution needs to survive the supervisor and its sub-agents
        running different models concurrently."""
        callback = self._callback()
        run_id = uuid4()

        callback.on_chat_model_start(
            {}, [], run_id=run_id, invocation_params={"model": "gemini-3.1-flash-lite"}
        )
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(None, run_id=run_id)

        self.assertEqual(captured.records[-1]._model, "gemini-3.1-flash-lite")

    def test_an_unidentifiable_model_degrades_to_a_usable_log_line(self):
        callback = self._callback()
        run_id = uuid4()

        callback.on_chat_model_start(None, [], run_id=run_id)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(None, run_id=run_id)

        self.assertEqual(captured.records[-1]._model, "unknown")

    def test_concurrent_calls_are_not_attributed_to_each_other(self):
        """The sub-agent dispatch path runs several models at once; a single
        start timestamp would charge one call's span to another."""
        import time

        callback = self._callback()
        slow, fast = uuid4(), uuid4()

        callback.on_chat_model_start({}, [], run_id=slow)
        time.sleep(0.06)
        callback.on_chat_model_start({}, [], run_id=fast)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(None, run_id=fast)
        fast_seconds = captured.records[-1]._duration_seconds

        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(None, run_id=slow)
        slow_seconds = captured.records[-1]._duration_seconds

        self.assertLess(fast_seconds, 0.05)
        self.assertGreaterEqual(slow_seconds, 0.06)

    def test_an_end_without_a_start_records_nothing(self):
        """Recording a span that was never measured would put a fabricated
        number in the histogram."""
        callback = self._callback()

        before = _phase_samples("llm_call")
        callback.on_llm_end(None, run_id=uuid4())

        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 0)

    def test_a_duplicate_end_records_only_once(self):
        callback = self._callback()
        run_id = uuid4()

        before = _phase_samples("llm_call")
        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 1)

    def test_abandoned_runs_do_not_accumulate_without_bound(self):
        """A turn cancelled by the T38 whole-turn deadline never delivers its
        end callback; the in-flight map must not be a leak."""
        from tta_backend.utils.llm_timing import _MAX_TRACKED_RUNS

        callback = self._callback()
        for _ in range(_MAX_TRACKED_RUNS + 50):
            callback.on_chat_model_start({}, [], run_id=uuid4())

        self.assertLessEqual(len(callback._started), _MAX_TRACKED_RUNS)

    def test_a_langchain_driven_retry_records_its_backoff(self):
        """Retries LangChain drives itself are visible here. Their *absence*
        beside a long llm_call is what identifies a provider-SDK retry."""

        class FakeRetryState:
            attempt_number = 3
            idle_for = 46.0

        callback = self._callback()
        run_id = uuid4()

        before = _phase_samples("llm_retry_sleep")
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_retry(FakeRetryState(), run_id=run_id)
        after = _phase_samples("llm_retry_sleep")

        self.assertEqual(after[0] - before[0], 1)
        self.assertAlmostEqual(after[1] - before[1], 46.0, places=3)
        self.assertEqual(captured.records[-1]._attempt, 3)

    def test_a_retried_call_reports_its_attempt_count(self):
        class FakeRetryState:
            attempt_number = 2
            idle_for = 4.0

        callback = self._callback()
        run_id = uuid4()

        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_retry(FakeRetryState(), run_id=run_id)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(None, run_id=run_id)

        record = next(r for r in captured.records if r._phase == "llm_call")
        self.assertEqual(record._retries, 2)

    def test_a_hostile_retry_state_does_not_break_the_call(self):
        class Hostile:
            @property
            def attempt_number(self):
                raise RuntimeError("no")

        callback = self._callback()
        callback.on_retry(Hostile(), run_id=uuid4())  # must not raise


@unittest.skipIf(
    importlib.util.find_spec("langchain_google_genai") is None,
    "langchain_google_genai is not installed",
)
class ModelFactoryWiringTests(unittest.TestCase):
    """The timer is only worth having if it is actually attached -- and
    build_chat_model is the one seam that guarantees every agent gets it."""

    def test_a_constructed_google_model_carries_the_timing_callback(self):
        from tta_backend.config.settings import Settings
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.utils.llm_timing import LlmTimingCallback

        model = build_chat_model("google", "gemini-3.1-flash-lite", Settings(google_api_key="k"))

        self.assertTrue(
            any(isinstance(cb, LlmTimingCallback) for cb in (model.callbacks or [])),
            "build_chat_model did not attach the llm_call timer",
        )


class AgentStepTimingTests(unittest.IsolatedAsyncioTestCase):
    """``agent_step`` covers what ``llm_call`` cannot: checkpointer writes and
    graph overhead. The difference between the two separates "the model was
    slow" from "we were slow around it"."""

    async def asyncSetUp(self):
        if importlib.util.find_spec("langchain_core") is None:
            self.skipTest("langchain_core is not installed")

    def _agent(self, chunks):
        class FakeAgent:
            async def astream(self, _input, config=None, stream_mode=None):
                for chunk in chunks:
                    yield chunk

        return FakeAgent()

    async def _drain(self, agent):
        from tta_backend.utils.streaming import stream_response

        return [event async for event in stream_response(agent, "hi", "thread-1")]

    async def test_a_superstep_is_recorded_per_updates_chunk(self):
        agent = self._agent(
            [("updates", {"model": {"messages": []}}), ("updates", {"tools": {"messages": []}})]
        )

        before = _phase_samples("agent_step")
        await self._drain(agent)

        self.assertEqual(_phase_samples("agent_step")[0] - before[0], 2)

    async def test_the_node_that_produced_the_step_is_logged(self):
        """Which node the time went to is the whole point -- 'the turn was
        slow' is not actionable, 'the model node was slow' is."""
        agent = self._agent([("updates", {"model": {"messages": []}})])

        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            await self._drain(agent)

        record = next(r for r in captured.records if r._phase == "agent_step")
        self.assertEqual(record._node, "model")

    async def test_a_silent_step_is_distinguishable_from_a_streaming_one(self):
        """A node that streamed nothing for 107s and then delivered a finished
        message was blocked on one provider call; a node that streamed
        throughout was working. Identical durations, opposite diagnoses."""
        from langchain_core.messages import AIMessageChunk

        agent = self._agent(
            [
                ("messages", (AIMessageChunk(content="tok"), {})),
                ("messages", (AIMessageChunk(content="en"), {})),
                ("updates", {"model": {"messages": []}}),
                ("updates", {"tools": {"messages": []}}),
            ]
        )

        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            await self._drain(agent)

        steps = [r for r in captured.records if r._phase == "agent_step"]
        self.assertEqual(steps[0]._streamed_chunks, 2)
        # The counter resets per superstep, or every later step inherits the
        # first one's tokens and no step ever reads as silent.
        self.assertEqual(steps[1]._streamed_chunks, 0)

    async def test_a_multi_node_chunk_is_charged_once(self):
        """Charging the gap to each node would inflate the histogram by the
        fan-out, making a parallel superstep look like several slow ones."""
        agent = self._agent(
            [("updates", {"model": {"messages": []}, "tools": {"messages": []}})]
        )

        before = _phase_samples("agent_step")
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            await self._drain(agent)

        self.assertEqual(_phase_samples("agent_step")[0] - before[0], 1)
        record = next(r for r in captured.records if r._phase == "agent_step")
        self.assertIn(",", record._node)


class PhaseVocabularyTests(unittest.TestCase):
    def test_the_new_phases_are_pre_declared_at_zero(self):
        """Prometheus pulls: a phase that has not run yet must still exist as
        a series, or a dashboard cannot tell "never exercised" from "removed"."""
        from tta_backend.utils.metrics import render_prometheus_metrics

        rendered = render_prometheus_metrics().decode()

        for phase in ("llm_call", "llm_retry_sleep", "agent_step"):
            self.assertIn(
                f'pipeline_phase_duration_seconds_count{{phase="{phase}"}}',
                rendered,
                f"phase '{phase}' has no pre-declared series",
            )


@unittest.skipIf(
    importlib.util.find_spec("langchain_core") is None,
    "langchain_core is not installed",
)
class TokenAccountingTests(unittest.TestCase):
    """The token half of the ``llm_call`` callback.

    Model latency and model cost are different questions, and a cache hit is
    invisible in the first one -- a hit and a miss take the same wall clock.
    These pin the second.
    """

    def _callback(self):
        from tta_backend.utils.llm_timing import LlmTimingCallback

        return LlmTimingCallback()

    def _finish(self, callback, response, *, model):
        """Run one start/end pair and return the phase_timing log record."""
        run_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=run_id, invocation_params={"model": model})
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(response, run_id=run_id)
        return captured.records[-1]

    def test_reported_counts_reach_the_counters_and_the_log(self):
        model = "gemini-token-happy-path"
        before = {k: _token_total(model, k) for k in ("input", "output", "cache_read")}

        record = self._finish(
            self._callback(),
            _llm_result(
                {
                    "input_tokens": 12_000,
                    "output_tokens": 300,
                    "total_tokens": 12_300,
                    "input_token_details": {"cache_read": 11_500},
                }
            ),
            model=model,
        )

        self.assertEqual(_token_total(model, "input") - before["input"], 12_000)
        self.assertEqual(_token_total(model, "output") - before["output"], 300)
        self.assertEqual(_token_total(model, "cache_read") - before["cache_read"], 11_500)
        # Same log line as the duration, so one event answers both "how long"
        # and "what did it cost".
        self.assertEqual(record._phase, "llm_call")
        self.assertEqual(record._input_tokens, 12_000)
        self.assertEqual(record._output_tokens, 300)
        self.assertEqual(record._cache_read_tokens, 11_500)

    def test_an_absent_usage_block_records_no_zero(self):
        """The load-bearing rule. A streaming response whose usage never
        arrived must contribute nothing -- recorded as zeros, a cache that is
        working perfectly would read as a 0% hit rate."""
        model = "gemini-token-no-usage"
        before = _token_total(model, "cache_read")

        record = self._finish(self._callback(), None, model=model)

        self.assertEqual(_token_total(model, "cache_read"), before)
        self.assertFalse(hasattr(record, "_cache_read_tokens"))
        self.assertFalse(hasattr(record, "_input_tokens"))

    def test_a_call_that_missed_the_cache_records_a_real_zero(self):
        """The other half of that rule: a *reported* zero is a measurement and
        must be counted, or a cold cache is indistinguishable from an
        unmeasured one."""
        model = "gemini-token-cold-cache"

        record = self._finish(
            self._callback(),
            _llm_result(
                {
                    "input_tokens": 9_000,
                    "output_tokens": 40,
                    "input_token_details": {"cache_read": 0},
                }
            ),
            model=model,
        )

        self.assertEqual(record._cache_read_tokens, 0)
        self.assertEqual(_token_total(model, "input"), 9_000)

    def test_a_usage_block_without_cache_detail_still_records_what_it_has(self):
        """Not every provider reports a cache breakdown; input/output must not
        be lost because the third field is missing."""
        model = "gemini-token-no-detail"

        record = self._finish(
            self._callback(),
            _llm_result({"input_tokens": 500, "output_tokens": 20}),
            model=model,
        )

        self.assertEqual(record._input_tokens, 500)
        self.assertFalse(hasattr(record, "_cache_read_tokens"))
        self.assertEqual(_token_total(model, "cache_read"), 0.0)

    def test_the_legacy_llm_output_shape_is_also_read(self):
        """Integrations disagree on where usage lands, exactly as they do on
        the model name -- see _model_name for the same problem."""
        from langchain_core.outputs import LLMResult

        model = "gemini-token-llm-output"
        response = LLMResult(
            generations=[[]],
            llm_output={"usage_metadata": {"input_tokens": 77, "output_tokens": 7}},
        )

        record = self._finish(self._callback(), response, model=model)

        self.assertEqual(record._input_tokens, 77)
        self.assertEqual(_token_total(model, "output"), 7)

    def test_a_failed_call_is_timed_but_charged_no_tokens(self):
        """on_llm_error carries no response, so there is nothing to read --
        and inventing a count for a call that produced none would corrupt
        exactly the series this exists to make trustworthy."""
        from tta_backend.utils.llm_timing import LlmTimingCallback

        model = "gemini-token-error"
        callback = LlmTimingCallback()
        run_id = uuid4()

        before = _phase_samples("llm_call")
        callback.on_chat_model_start({}, [], run_id=run_id, invocation_params={"model": model})
        callback.on_llm_error(ValueError("429 quota"), run_id=run_id)

        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 1)
        self.assertEqual(_token_total(model, "input"), 0.0)

    def test_a_malformed_response_costs_the_field_not_the_turn(self):
        class Exploding:
            @property
            def generations(self):
                raise RuntimeError("provider changed the payload shape")

        model = "gemini-token-malformed"
        before = _phase_samples("llm_call")

        record = self._finish(self._callback(), Exploding(), model=model)

        # The span is still recorded: losing telemetry must not lose the turn.
        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 1)
        self.assertFalse(hasattr(record, "_input_tokens"))

    def test_tokens_are_attributed_to_the_same_model_as_the_span(self):
        """The sub-agents run different models concurrently; charging a call's
        tokens to another call's model would make cost-per-model a guess."""
        callback = self._callback()
        supervisor, subagent = "gemini-token-supervisor", "gemini-token-subagent"

        run_a, run_b = uuid4(), uuid4()
        callback.on_chat_model_start({}, [], run_id=run_a, invocation_params={"model": supervisor})
        callback.on_chat_model_start({}, [], run_id=run_b, invocation_params={"model": subagent})
        callback.on_llm_end(_llm_result({"input_tokens": 1_000, "output_tokens": 1}), run_id=run_b)
        callback.on_llm_end(_llm_result({"input_tokens": 5, "output_tokens": 1}), run_id=run_a)

        self.assertEqual(_token_total(subagent, "input"), 1_000)
        self.assertEqual(_token_total(supervisor, "input"), 5)

    def test_an_end_without_a_start_charges_no_tokens(self):
        """No start means no measured span and no model to charge it to -- the
        same rule the histogram already applies."""
        model = "gemini-token-orphan"
        callback = self._callback()

        callback.on_llm_end(_llm_result({"input_tokens": 400, "output_tokens": 9}), run_id=uuid4())

        self.assertEqual(_token_total(model, "input"), 0.0)


@unittest.skipIf(
    importlib.util.find_spec("langchain_core") is None,
    "langchain_core is not installed",
)
class AgentAttributionTests(unittest.TestCase):
    """``agent_type`` exists because ``model`` cannot stand in for it."""

    def _finish(self, callback, response, *, model):
        run_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=run_id, invocation_params={"model": model})
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            callback.on_llm_end(response, run_id=run_id)
        return captured.records[-1]

    def test_the_two_subagents_sharing_a_model_id_are_separable(self):
        """The whole reason this label exists: both sub-agents default to
        gemini-3.1-flash-lite, so without agent_type their spend is one
        indivisible series -- and their prefixes differ by ~3x."""
        from tta_backend.utils.llm_timing import LlmTimingCallback

        model = "gemini-shared-model-id"
        satellite = LlmTimingCallback("satellite")
        ground = LlmTimingCallback("ground_sensor")

        self._finish(satellite, _llm_result({"input_tokens": 13_000, "output_tokens": 1}), model=model)
        self._finish(ground, _llm_result({"input_tokens": 4_100, "output_tokens": 1}), model=model)

        self.assertEqual(_token_total(model, "input", "satellite"), 13_000)
        self.assertEqual(_token_total(model, "input", "ground_sensor"), 4_100)

    def test_the_agent_type_reaches_the_log_line_too(self):
        """Not only the counters: per-agent *latency* is the other question
        the shared llm_call histogram cannot answer on its own."""
        from tta_backend.utils.llm_timing import LlmTimingCallback

        record = self._finish(
            LlmTimingCallback("supervisor"),
            _llm_result({"input_tokens": 900, "output_tokens": 5}),
            model="gemini-supervisor-log",
        )

        self.assertEqual(record._agent_type, "supervisor")

    def test_an_undeclared_build_is_visibly_unknown(self):
        """Defaulted rather than required, so anything building a model
        outside the three agents still works -- and says so."""
        from tta_backend.utils.llm_timing import LlmTimingCallback, timing_callbacks

        self.assertEqual(timing_callbacks()[0]._agent_type, "unknown")

        model = "gemini-undeclared"
        self._finish(LlmTimingCallback(), _llm_result({"input_tokens": 11, "output_tokens": 1}), model=model)

        self.assertEqual(_token_total(model, "input", "unknown"), 11)

    def test_each_callback_keeps_its_own_in_flight_state(self):
        """Per-model instances replaced a process-wide singleton; two agents'
        concurrent calls must not pop each other's start entries."""
        import time

        from tta_backend.utils.llm_timing import LlmTimingCallback

        a, b = LlmTimingCallback("satellite"), LlmTimingCallback("ground_sensor")
        run = uuid4()  # deliberately the SAME run_id on both instances

        a.on_chat_model_start({}, [], run_id=run, invocation_params={"model": "m"})
        time.sleep(0.05)
        b.on_chat_model_start({}, [], run_id=run, invocation_params={"model": "m"})

        before = _phase_samples("llm_call")
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            b.on_llm_end(None, run_id=run)
            a.on_llm_end(None, run_id=run)
        spans = [r._duration_seconds for r in captured.records if r._phase == "llm_call"]

        # Asserted as a relationship, not against wall-clock thresholds: a
        # shared map would let b's end pop a's (earlier) start, leaving a's
        # end with nothing to record -- so "two spans, and the one that
        # started first is the longer" is the discriminating claim and does
        # not flake on a slow or coarse clock.
        self.assertEqual(_phase_samples("llm_call")[0] - before[0], 2)
        b_seconds, a_seconds = spans
        self.assertGreater(a_seconds, b_seconds)


@unittest.skipIf(
    importlib.util.find_spec("langchain_google_genai") is None,
    "langchain_google_genai is not installed",
)
class AgentTypeWiringTests(unittest.TestCase):
    """Each agent must declare itself at the one seam that builds its model;
    an agent that forgets is silently attributed to "unknown"."""

    def _agent_type_of(self, model):
        from tta_backend.utils.llm_timing import LlmTimingCallback

        callback = next(
            cb for cb in (model.callbacks or []) if isinstance(cb, LlmTimingCallback)
        )
        return callback._agent_type

    def test_build_chat_model_forwards_the_declared_agent_type(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings

        model = build_chat_model(
            "google", "gemini-3.1-flash-lite", Settings(google_api_key="k"), agent_type="satellite"
        )

        self.assertEqual(self._agent_type_of(model), "satellite")

    def test_every_agent_declares_a_vocabulary_value_that_joins(self):
        """The values must match AGENT_REQUESTS_TOTAL's, or tokens-per-agent-
        call needs a relabel to join. Read off the real build functions rather
        than restated here, so renaming one and not the other fails."""
        import re

        expected = {
            "tta_backend/agents/earthdata_agent.py": "satellite",
            "tta_backend/agents/ground_sensor_agent.py": "ground_sensor",
            "tta_backend/agents/supervisor_agent.py": "supervisor",
        }
        root = os.path.dirname(TESTS_DIR)
        for relative, agent_type in expected.items():
            source = open(os.path.join(root, relative), encoding="utf-8").read()
            found = re.findall(r"build_chat_model\([^)]*agent_type=\"([^\"]+)\"", source)
            self.assertEqual(
                found,
                [agent_type],
                f"{relative} does not declare agent_type={agent_type!r} exactly once",
            )


if __name__ == "__main__":
    unittest.main()
