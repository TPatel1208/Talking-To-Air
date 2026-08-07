"""Model-latency and graph-superstep timing.

A 2026-08-07 trace of a 372s plot turn accounted for the Harmony retrieval
wait (171s) and the data pipeline (61s) and left 107s -- 29% of the turn --
attributable to nothing: T51's phase vocabulary is all data work, so time
spent waiting on the model had no series to land in.

These cover the two phases that close that hole: ``llm_call`` (one provider
request, from the LangChain callback pair) and ``agent_step`` (one LangGraph
superstep, from the gap between ``updates`` chunks).

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


if __name__ == "__main__":
    unittest.main()
