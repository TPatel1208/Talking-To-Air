import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]


class ContainsSubsequenceTests(unittest.TestCase):
    def test_matches_a_non_contiguous_subsequence_in_order(self):
        from eval_harness import contains_subsequence

        trace = ["search_datasets", "describe_dataset", "safe_retrieve", "await_retrieval", "plot_singular"]

        self.assertTrue(contains_subsequence(trace, ["search_datasets", "safe_retrieve", "plot_singular"]))

    def test_rejects_out_of_order_calls(self):
        from eval_harness import contains_subsequence

        trace = ["safe_retrieve", "search_datasets"]

        self.assertFalse(contains_subsequence(trace, ["search_datasets", "safe_retrieve"]))

    def test_rejects_a_missing_call(self):
        from eval_harness import contains_subsequence

        trace = ["search_datasets", "define_area_of_interest"]

        self.assertFalse(contains_subsequence(trace, ["search_datasets", "safe_retrieve"]))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class EvalHarnessStructureTests(unittest.TestCase):
    """Structural checks that don't spend model tokens — the 12-task shape
    itself is verified on every run; only the real agent execution is
    opt-in (see EvalSuiteTests below)."""

    def _tasks(self):
        from eval_harness import DIRECT_AGENT_TASK_COUNT, build_eval_tasks
        from fake_earthdata_mcp import HandleVolume

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        volume = HandleVolume(tmpdir.name)
        tasks = build_eval_tasks(volume)
        self.assertEqual(len(tasks), DIRECT_AGENT_TASK_COUNT)
        return tasks

    def test_builds_exactly_total_tasks(self):
        self._tasks()

    def test_covers_task_categories_including_the_new_point_timeseries_plotting_task(self):
        from collections import Counter

        tasks = self._tasks()
        counts = Counter(task.category for task in tasks)

        self.assertEqual(
            counts,
            {
                "discovery": 2,
                "retrieval": 2,
                "plotting": 3,
                "comparison_setup": 2,
                "failure_recovery": 2,
                "ground_validation": 1,
                "comparison": 1,
                "robustness": 1,
            },
        )

    def test_every_task_declares_expected_tool_calls(self):
        tasks = self._tasks()

        for task in tasks:
            self.assertTrue(task.expected_tool_calls, f"{task.name} has no expected tool calls")

    def test_category_budgets_cover_every_task_category(self):
        from eval_harness import CATEGORY_BUDGETS

        tasks = self._tasks()
        categories = {task.category for task in tasks}

        missing = categories - CATEGORY_BUDGETS.keys()
        self.assertFalse(missing, f"categories with no latency budget: {missing}")
        for category, budget in CATEGORY_BUDGETS.items():
            self.assertGreater(budget, 0, f"{category} budget must be positive")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class RobustnessTaskTests(unittest.IsolatedAsyncioTestCase):
    """T15: the malformed-envelope robustness task is scripted at the
    finalization seam (subagent_dispatch._finalize_sub_agent_result) rather
    than the live model loop — spends no model tokens, runs on every
    invocation of the suite structure tests above."""

    async def test_robustness_task_scores_a_non_error_answer_carrying_the_artifact(self):
        from eval_harness import run_robustness_task

        result = await run_robustness_task()

        self.assertTrue(result.passed)
        self.assertEqual(result.task.category, "robustness")

    async def test_robustness_task_records_elapsed_seconds(self):
        from eval_harness import run_robustness_task

        result = await run_robustness_task()

        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertLess(result.elapsed_seconds, 1.0)  # spends no tokens — must be near-instant


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class E2ETaskStructureTests(unittest.TestCase):
    """T16: three tasks that enter through ChatStreamService.stream_chat_
    events with the real intent_router and real sub-agents — extend, not
    replace, the 13 direct-agent tasks above."""

    def _tasks(self):
        from eval_harness import build_e2e_tasks
        from fake_earthdata_mcp import HandleVolume

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        volume = HandleVolume(tmpdir.name)
        return build_e2e_tasks(volume)

    def test_builds_exactly_three_tasks(self):
        self.assertEqual(len(self._tasks()), 3)

    def test_covers_ground_satellite_and_cross_source(self):
        categories = {task.category for task in self._tasks()}

        self.assertEqual(categories, {"e2e_ground", "e2e_satellite", "e2e_cross_source"})

    def test_ground_and_cross_source_expect_the_ground_agent(self):
        by_category = {task.category: task for task in self._tasks()}

        self.assertTrue(by_category["e2e_ground"].expects_ground)
        self.assertTrue(by_category["e2e_cross_source"].expects_ground)
        self.assertFalse(by_category["e2e_satellite"].expects_ground)

    def test_satellite_and_cross_source_expect_the_satellite_agent(self):
        by_category = {task.category: task for task in self._tasks()}

        self.assertTrue(by_category["e2e_satellite"].expects_satellite)
        self.assertTrue(by_category["e2e_cross_source"].expects_satellite)
        self.assertFalse(by_category["e2e_ground"].expects_satellite)

    def test_only_the_ground_task_carries_a_relative_date_check(self):
        by_category = {task.category: task for task in self._tasks()}

        self.assertIsNotNone(by_category["e2e_ground"].date_check)
        self.assertIsNone(by_category["e2e_satellite"].date_check)

    def test_only_the_satellite_task_carries_a_stage_sequence_check(self):
        """T19: dead-chat as a command-detectable regression — only the
        satellite e2e task exercises the composites/handle tools that
        narrate stages; ground and cross-source don't need this check."""
        by_category = {task.category: task for task in self._tasks()}

        self.assertIsNotNone(by_category["e2e_satellite"].stage_check)
        self.assertIsNone(by_category["e2e_ground"].stage_check)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class RelativeDateCheckTests(unittest.TestCase):
    """T11 Further Notes / T16: a 'yesterday' query was live-caught
    dispatching *today's* date (2026-07-07 live test). Pure logic over
    captured (endpoint, params) tuples — no live model needed."""

    def test_true_when_a_captured_call_used_real_utc_yesterday(self):
        from datetime import datetime, timedelta, timezone

        from eval_harness import _dispatched_correct_relative_date

        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y%m%d")
        calls = [("dailyData/byBox", {"bdate": yesterday, "edate": yesterday})]

        self.assertTrue(_dispatched_correct_relative_date(calls))

    def test_false_when_no_captured_call_used_real_utc_yesterday(self):
        from eval_harness import _dispatched_correct_relative_date

        calls = [("dailyData/byBox", {"bdate": "20260617", "edate": "20260617"})]

        self.assertFalse(_dispatched_correct_relative_date(calls))

    def test_false_for_no_calls_at_all(self):
        from eval_harness import _dispatched_correct_relative_date

        self.assertFalse(_dispatched_correct_relative_date([]))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class StageSequenceCheckTests(unittest.TestCase):
    """T19: pure logic over a captured/fabricated joined SSE string — no
    live model needed to prove the scoring logic itself is correct; the
    live enforcement only runs under -m eval (EvalSuiteTests below)."""

    def test_extracts_stage_keys_from_status_events_in_order(self):
        from eval_harness import _stage_sequence

        joined = (
            'event: status\ndata: {"message": "Searching...", "stage": "search"}\n\n'
            'event: tool_call\ndata: {"name": "search_datasets", "args": {}}\n\n'
            'event: status\ndata: {"message": "Resolving...", "stage": "aoi"}\n\n'
        )

        self.assertEqual(_stage_sequence(joined), ["search", "aoi"])

    def test_covers_full_satellite_workflow_accepts_the_canonical_sequence_with_extra_stages(self):
        from eval_harness import _covers_full_satellite_workflow

        stage_sequence = [
            "search", "aoi", "coverage", "coverage", "estimate", "submit",
            "progress", "progress", "open", "render", "render",
        ]

        self.assertTrue(_covers_full_satellite_workflow(stage_sequence))

    def test_covers_full_satellite_workflow_rejects_a_workflow_missing_a_stage(self):
        from eval_harness import _covers_full_satellite_workflow

        stage_sequence = ["search", "aoi", "estimate", "submit", "progress", "open", "render"]

        self.assertFalse(_covers_full_satellite_workflow(stage_sequence))

    def test_covers_full_satellite_workflow_rejects_out_of_order_stages(self):
        from eval_harness import _covers_full_satellite_workflow

        stage_sequence = ["aoi", "search", "coverage", "estimate", "submit", "progress", "open", "render"]

        self.assertFalse(_covers_full_satellite_workflow(stage_sequence))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class AgentsConsultedParsingTests(unittest.TestCase):
    """The router fast path and the supervisor path emit differently-shaped
    'Agent consulted: ...' headers — a single source vs. one combined
    'GROUND + SATELLITE' line — the e2e cross-source task's scoring must
    handle both."""

    def test_parses_a_single_source_fast_path_header(self):
        from eval_harness import _agents_consulted

        text = "Agent consulted: GROUND\n\nThe monitor is Rutgers University."

        self.assertEqual(_agents_consulted(text), {"GROUND"})

    def test_parses_a_combined_cross_source_supervisor_header(self):
        from eval_harness import _agents_consulted

        text = "Agent consulted: GROUND + SATELLITE\n\nBoth sources agree the level rose."

        self.assertEqual(_agents_consulted(text), {"GROUND", "SATELLITE"})

    def test_returns_an_empty_set_when_no_header_is_present(self):
        from eval_harness import _agents_consulted

        self.assertEqual(_agents_consulted("no header in this text"), set())


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class ResultsTableTests(unittest.TestCase):
    """T16 user story 9: a compact per-task table (pass/fail, trace verdict,
    seconds) so a regression's location is obvious from the output alone —
    hermetic, built from fabricated results rather than a live run."""

    def test_table_reports_name_category_pass_trace_and_seconds(self):
        from eval_harness import EvalTask, EvalTaskResult, format_results_table

        passing_task = EvalTask(
            name="discovery_no2_dataset", category="discovery", prompt="p",
            handlers={}, expected_tool_calls=["search_datasets"],
        )
        failing_task = EvalTask(
            name="plotting_single_map", category="plotting", prompt="p",
            handlers={}, expected_tool_calls=["safe_retrieve", "plot_singular"],
        )
        results = [
            EvalTaskResult(
                task=passing_task, tool_calls=["search_datasets"], raw_text="", envelope=None,
                passed=True, elapsed_seconds=1.234,
            ),
            EvalTaskResult(
                task=failing_task, tool_calls=["safe_retrieve"], raw_text="", envelope=None,
                passed=False, elapsed_seconds=50.5,
            ),
        ]

        table = format_results_table(results)

        self.assertIn("discovery_no2_dataset", table)
        self.assertIn("discovery", table)
        self.assertIn("plotting_single_map", table)
        self.assertIn("1.23", table)
        self.assertIn("50.50", table)
        # The passing task's full trace is present; the failing one's isn't.
        self.assertIn("PASS", table)
        self.assertIn("FAIL", table)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class RateLimitDetectionTests(unittest.TestCase):
    """T16 user story 3: single-user rate-limit pressure — the original
    outage mode — must never silently reappear. Groq's client only logs
    retries (no exception raised on the caller's side), so the harness
    watches the groq/httpx loggers for the duration of a run. The satellite
    agent (and supervisor) now default to google/gemini, whose client logs
    its own retry backoff on "google_genai._api_client" via tenacity's
    before_sleep_log — watched too, now that provider is load-bearing."""

    def test_captures_a_groq_retry_log_record(self):
        import logging

        from eval_harness import capture_rate_limit_evidence

        with capture_rate_limit_evidence() as handler:
            logging.getLogger("groq._base_client").warning(
                "Retrying request to /v1/chat/completions in 5.2 seconds"
            )

        self.assertEqual(len(handler.matches), 1)
        self.assertIn("Retrying", handler.matches[0])

    def test_captures_a_429_mention_on_the_httpx_logger(self):
        import logging

        from eval_harness import capture_rate_limit_evidence

        with capture_rate_limit_evidence() as handler:
            logging.getLogger("httpx").info(
                'HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"'
            )

        self.assertEqual(len(handler.matches), 1)

    def test_ignores_unrelated_log_records(self):
        import logging

        from eval_harness import capture_rate_limit_evidence

        with capture_rate_limit_evidence() as handler:
            logging.getLogger("groq._base_client").info("Request completed successfully")
            logging.getLogger("httpx").info(
                'HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"'
            )

        self.assertEqual(handler.matches, [])

    def test_captures_a_google_genai_retry_backoff_log_record(self):
        """Mirrors tenacity's before_sleep_log wording for a 429 APIError:
        'Retrying <fn> in 1.0 seconds as it raised APIError: 429
        RESOURCE_EXHAUSTED. {...}'."""
        import logging

        from eval_harness import capture_rate_limit_evidence

        with capture_rate_limit_evidence() as handler:
            logging.getLogger("google_genai._api_client").info(
                "Retrying google.genai.models.Models.generate_content in 1.0 seconds as it "
                "raised APIError: 429 RESOURCE_EXHAUSTED. {'message': 'Resource exhausted'}."
            )

        self.assertEqual(len(handler.matches), 1)
        self.assertIn("429", handler.matches[0])


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class RunEvalSuiteRateLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_eval_suite_fails_the_run_if_rate_limit_evidence_is_logged(self):
        import logging

        from eval_harness import EvalTaskResult, RateLimitDetected, run_eval_suite
        from fake_earthdata_mcp import HandleVolume

        async def fake_run_eval_task(task, *, model=None):
            logging.getLogger("groq._base_client").warning(
                "Retrying request to /v1/chat/completions in 3.0 seconds"
            )
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        async def fake_run_e2e_task(task, volume, *, model=None):
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            volume = HandleVolume(tmpdir)
            with patch("eval_harness.run_eval_task", side_effect=fake_run_eval_task), \
                 patch("eval_harness.run_e2e_task", side_effect=fake_run_e2e_task):
                with self.assertRaises(RateLimitDetected):
                    await run_eval_suite(volume)

    async def test_run_eval_suite_succeeds_with_no_rate_limit_evidence(self):
        from eval_harness import EvalTaskResult, TOTAL_TASKS, run_eval_suite
        from fake_earthdata_mcp import HandleVolume

        async def fake_run_eval_task(task, *, model=None):
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        async def fake_run_e2e_task(task, volume, *, model=None):
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            volume = HandleVolume(tmpdir)
            with patch("eval_harness.run_eval_task", side_effect=fake_run_eval_task), \
                 patch("eval_harness.run_e2e_task", side_effect=fake_run_e2e_task):
                results = await run_eval_suite(volume)

        self.assertEqual(len(results), TOTAL_TASKS)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class RunEvalSuiteIncludesE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_run_eval_suite_runs_all_three_e2e_tasks_alongside_the_direct_agent_ones(self):
        from eval_harness import DIRECT_AGENT_TASK_COUNT, EvalTaskResult, TOTAL_TASKS, run_eval_suite
        from fake_earthdata_mcp import HandleVolume

        direct_calls = []
        e2e_calls = []

        async def fake_run_eval_task(task, *, model=None):
            direct_calls.append(task.name)
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        async def fake_run_e2e_task(task, volume, *, model=None):
            e2e_calls.append(task.name)
            return EvalTaskResult(
                task=task, tool_calls=[], raw_text="", envelope=None, passed=True, elapsed_seconds=0.1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            volume = HandleVolume(tmpdir)
            with patch("eval_harness.run_eval_task", side_effect=fake_run_eval_task), \
                 patch("eval_harness.run_e2e_task", side_effect=fake_run_e2e_task):
                results = await run_eval_suite(volume)

        self.assertEqual(TOTAL_TASKS, DIRECT_AGENT_TASK_COUNT + 3)
        self.assertEqual(len(results), TOTAL_TASKS)
        self.assertEqual(len(direct_calls), DIRECT_AGENT_TASK_COUNT - 1)  # robustness skips run_eval_task
        self.assertEqual(len(e2e_calls), 3)


def _real_groq_key_available() -> bool:
    from config.settings import get_settings

    key = get_settings().groq_api_key
    return bool(key) and key not in ("test", "your_groq_key")


def _real_google_key_available() -> bool:
    from config.settings import get_settings

    key = get_settings().google_api_key
    return bool(key) and key not in ("test", "your_google_key")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class EvalSuiteTests(unittest.IsolatedAsyncioTestCase):
    """The actual 14-task scripted eval. Opt-in (pytest -m eval) because it
    calls a real model and spends real tokens; skipped without both a real
    GROQ_API_KEY (ground agent) and a real GOOGLE_API_KEY (satellite agent
    and supervisor both default to google/gemini) even when explicitly
    selected."""

    @pytest.mark.eval
    @unittest.skipUnless(_real_groq_key_available(), "requires a real GROQ_API_KEY")
    @unittest.skipUnless(_real_google_key_available(), "requires a real GOOGLE_API_KEY")
    async def test_earthdata_agent_passes_at_least_eleven_of_thirteen_tasks(self):
        from eval_harness import CATEGORY_BUDGETS, PASS_THRESHOLD, TOTAL_TASKS, format_results_table, run_eval_suite
        from fake_earthdata_mcp import HandleVolume

        # T13: the subagent trim safety net's ceiling is sized so it never
        # fires in a healthy workflow — this run is the proof. If it fires
        # here, either a real compaction gap regressed or the ceiling itself
        # needs raising, not the tasks.
        with self.assertNoLogs("agents.subagent_trim", level="WARNING"):
            with tempfile.TemporaryDirectory() as tmpdir:
                volume = HandleVolume(tmpdir)
                results = await run_eval_suite(volume)

        # T16 user story 9: the table is the deliverable a human reads —
        # a regression's location must be obvious from the output alone.
        print("\n" + format_results_table(results))

        passed = sum(1 for r in results if r.passed)
        failures = [r.task.name for r in results if not r.passed]
        budget_breaches = [
            f"{r.task.name} ({r.elapsed_seconds:.1f}s > {CATEGORY_BUDGETS[r.task.category]:.0f}s budget)"
            for r in results
            if r.elapsed_seconds > CATEGORY_BUDGETS[r.task.category]
        ]

        self.assertEqual(len(results), TOTAL_TASKS)
        self.assertFalse(budget_breaches, f"tasks over their category latency budget: {budget_breaches}")
        self.assertGreaterEqual(passed, PASS_THRESHOLD, f"failed tasks: {failures}")


if __name__ == "__main__":
    unittest.main()
