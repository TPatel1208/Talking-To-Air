import importlib.util
import os
import sys
import tempfile
import unittest

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
        from eval_harness import TOTAL_TASKS, build_eval_tasks
        from fake_earthdata_mcp import HandleVolume

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        volume = HandleVolume(tmpdir.name)
        tasks = build_eval_tasks(volume)
        self.assertEqual(len(tasks), TOTAL_TASKS)
        return tasks

    def test_builds_exactly_total_tasks(self):
        self._tasks()

    def test_covers_five_categories_twice_each_plus_one_ground_validation_and_comparison_task(self):
        from collections import Counter

        tasks = self._tasks()
        counts = Counter(task.category for task in tasks)

        self.assertEqual(
            counts,
            {
                "discovery": 2,
                "retrieval": 2,
                "plotting": 2,
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


def _real_groq_key_available() -> bool:
    from config.settings import get_settings

    key = get_settings().groq_api_key
    return bool(key) and key not in ("test", "your_groq_key")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class EvalSuiteTests(unittest.IsolatedAsyncioTestCase):
    """The actual 13-task scripted eval. Opt-in (pytest -m eval) because it
    calls a real model and spends real tokens; skipped without a real
    GROQ_API_KEY even when explicitly selected."""

    @pytest.mark.eval
    @unittest.skipUnless(_real_groq_key_available(), "requires a real GROQ_API_KEY")
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
