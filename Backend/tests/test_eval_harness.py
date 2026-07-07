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
            },
        )

    def test_every_task_declares_expected_tool_calls(self):
        tasks = self._tasks()

        for task in tasks:
            self.assertTrue(task.expected_tool_calls, f"{task.name} has no expected tool calls")


def _real_groq_key_available() -> bool:
    from config.settings import get_settings

    key = get_settings().groq_api_key
    return bool(key) and key not in ("test", "your_groq_key")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "eval harness test dependencies are not installed",
)
class EvalSuiteTests(unittest.IsolatedAsyncioTestCase):
    """The actual 12-task scripted eval. Opt-in (pytest -m eval) because it
    calls a real model and spends real tokens; skipped without a real
    GROQ_API_KEY even when explicitly selected."""

    @pytest.mark.eval
    @unittest.skipUnless(_real_groq_key_available(), "requires a real GROQ_API_KEY")
    async def test_earthdata_agent_passes_at_least_ten_of_twelve_tasks(self):
        from eval_harness import PASS_THRESHOLD, TOTAL_TASKS, run_eval_suite
        from fake_earthdata_mcp import HandleVolume

        # T13: the subagent trim safety net's ceiling is sized so it never
        # fires in a healthy workflow — this run is the proof. If it fires
        # here, either a real compaction gap regressed or the ceiling itself
        # needs raising, not the tasks.
        with self.assertNoLogs("agents.subagent_trim", level="WARNING"):
            with tempfile.TemporaryDirectory() as tmpdir:
                volume = HandleVolume(tmpdir)
                results = await run_eval_suite(volume)

        passed = sum(1 for r in results if r.passed)
        failures = [r.task.name for r in results if not r.passed]

        self.assertEqual(len(results), TOTAL_TASKS)
        self.assertGreaterEqual(passed, PASS_THRESHOLD, f"failed tasks: {failures}")


if __name__ == "__main__":
    unittest.main()
