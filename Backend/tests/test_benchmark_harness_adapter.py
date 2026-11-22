"""The harness adapter: a case's prompt run through the supervisor on the fake boundary, with scripted models.

No network and no real model: the adapter is given scripted models in place of
the app's, so these pin how a ``Turn`` becomes an ``Outcome`` the gate scores.
"""
import asyncio
import dataclasses
import importlib.util
import tempfile
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.cases import BenchmarkCase, build_cases

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow", "langgraph"]
_MISSING_DEPENDENCIES = any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)

SUPERVISOR_MODEL = "scripted:supervisor"
EXPLORE_MODEL = "scripted:explore"


class _NullVolume:
    """Stands in for the handle volume; case building only registers files on it."""

    def add_zarr(self, *args, **kwargs) -> None: ...

    def add_parquet(self, *args, **kwargs) -> None: ...

    async def export_result(self, *args, **kwargs) -> dict:
        return {}

    async def rematerialize(self, *args, **kwargs) -> dict:
        return {}

    async def get_retrieval_status(self, *args, **kwargs) -> dict:
        return {}


def _cases_by_id() -> dict[str, BenchmarkCase]:
    return {case.id: case for case in build_cases(_NullVolume())}


def _models(supervisor_script, explore_script=()):
    from tta_backend.harness.models import ModelChoice, NamedModel
    from tta_backend.harness.testing import ScriptedChatModel, ScriptedModels

    def choice(script, name):
        return ModelChoice(primary=NamedModel(ScriptedChatModel(script=list(script)), name))

    return ScriptedModels({
        "supervisor": choice(supervisor_script, SUPERVISOR_MODEL),
        "explore": choice(explore_script, EXPLORE_MODEL),
    })


def _run(case_id: str, models, **changes):
    """Run the case on a real handle volume, which the retrieval cases' fake service reads; ``models`` may be a
    function of the case. ``changes`` replace case fields."""
    from benchmark.adapters.harness import HarnessAdapter
    from benchmark.fake_earthdata import HandleVolume

    async def run():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            volume = HandleVolume(tmp)
            case = {c.id: c for c in build_cases(volume)}[case_id]
            case = dataclasses.replace(case, **changes)
            scripted = models(case) if callable(models) else models
            return await HarnessAdapter(models=scripted).run(case, volume)

    return asyncio.run(run())


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class HarnessAdapterRunTests(unittest.TestCase):
    def test_a_workflow_case_passes_its_checks_end_to_end(self):
        from benchmark.scoring import score
        from tta_backend.harness.testing import answer, tool_call

        holder = {}

        def models(case):
            holder["case"] = case
            return _models([
                tool_call("run_workflow", {"name": case.workflow, "inputs": case.inputs_for(date.today())},
                          usage=(500, 50)),
                answer("The NO2 map over New Jersey is ready.", usage=(700, 30)),
            ])

        outcome = _run("plotting_single_map", models)
        case = holder["case"]

        self.assertTrue(outcome.done, outcome.unfinished or outcome.error)
        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.handles)
        self.assertTrue(outcome.charts)
        self.assertIn("The NO2 map over New Jersey is ready.", outcome.answer_text)
        self.assertTrue(outcome.mode.startswith("Draft workflow: map v"))
        self.assertIn("render.map", outcome.tool_calls)
        self.assertEqual(outcome.tokens[(SUPERVISOR_MODEL, "supervisor", "input")], 1200)
        self.assertEqual(outcome.tokens[(SUPERVISOR_MODEL, "supervisor", "output")], 80)
        scored = score(case, outcome)
        self.assertTrue(scored.passed, scored.first_failure and scored.first_failure.reason)

    def test_an_exploration_case_reports_the_explore_tokens_beside_the_supervisors(self):
        from benchmark.checks import BoundaryCalled, Completed
        from tta_backend.harness.specialists import ANSWER_TOOL
        from tta_backend.harness.testing import answer, tool_call

        def models(case):
            task = {"objective": case.prompt, "bundle": case.bundle, "places": list(case.explore_places)}
            return _models(
                [
                    tool_call("explore", {"tasks": [task]}, usage=(400, 40)),
                    answer("Two NO2 products cover it.", usage=(600, 20)),
                ],
                [
                    tool_call("sources__satellite__search", {"query": "NO2 column"}, usage=(100, 20)),
                    tool_call(ANSWER_TOOL, {"summary": "Two.", "findings": [], "followups": []}, usage=(200, 40)),
                ],
            )

        outcome = _run("discovery_no2_dataset", models)

        self.assertTrue(outcome.done, outcome.unfinished or outcome.error)
        self.assertEqual(outcome.mode, "Exploratory")
        self.assertIn("sources.satellite.search", outcome.tool_calls)
        self.assertEqual([call.tool for call in outcome.boundary_calls], ["search_datasets"])
        self.assertTrue(BoundaryCalled("search_datasets").evaluate(outcome).passed)
        self.assertTrue(Completed().evaluate(outcome).passed)
        self.assertEqual(outcome.tokens[(EXPLORE_MODEL, "explore", "input")], 300)
        self.assertEqual(outcome.tokens[(SUPERVISOR_MODEL, "supervisor", "input")], 1000)
        self.assertEqual(outcome.token_total("input"), 1300)

    def test_a_turn_that_gets_no_model_is_an_error_outcome(self):
        from tta_backend.harness.testing import ScriptedModels

        outcome = _run("discovery_no2_dataset", ScriptedModels())

        self.assertFalse(outcome.done)
        self.assertTrue(outcome.error)
        self.assertEqual(outcome.boundary_calls, [])

    def test_a_turn_that_asks_a_question_is_unfinished_unless_the_case_expects_one(self):
        from tta_backend.harness.testing import answer, tool_call

        def models(case):
            call = tool_call("run_workflow", {"name": case.workflow, "inputs": case.inputs_for(date.today())})
            return _models([call, answer("I need a place.")])

        expecting = _run("region_refusal_unresolvable_member", models)
        unexpecting = _run("region_refusal_unresolvable_member", models, expects_question=False)

        self.assertTrue(expecting.done, expecting.unfinished or expecting.error)
        self.assertFalse(unexpecting.done)
        self.assertIn("stopped to ask", unexpecting.unfinished)

    def test_a_case_with_a_scripted_answer_sends_it_through_the_question_control_and_scores_the_last_turn(self):
        from benchmark.cases import EVENT_BASELINE_DAYS, EVENT_DAY, EVENT_DAYS_REMOVED
        from benchmark.scoring import score
        from tta_backend.harness.testing import answer, tool_call

        holder = {}

        def models(case):
            holder["case"] = case
            return _models([
                tool_call("run_workflow", {"name": case.workflow, "inputs": case.inputs_for(date.today())},
                          usage=(500, 50)),
                answer("Please confirm the comparison days.", usage=(600, 20)),
                answer("The event day was higher than the baseline.", usage=(900, 40)),
            ])

        outcome = _run("event_baseline_days_edited", models)
        kept = [day for day in EVENT_BASELINE_DAYS if day not in EVENT_DAYS_REMOVED]

        self.assertTrue(outcome.done, outcome.unfinished or outcome.error)
        [question] = outcome.questions
        self.assertEqual((question["step"], question["kind"]), ("days", "confirm_list"))
        [sent] = outcome.answers
        self.assertEqual({k: sent[k] for k in ("run_id", "step", "value")}, {"run_id": question["run_id"], "step": "days", "value": kept})
        # The event was retrieved in the first turn, before the answer was sent.
        before = [c for c in outcome.boundary_calls[:sent["calls_before"]] if c.tool == "retrieve_subset"]
        self.assertEqual([c.args["time_range"][:10] for c in before], [EVENT_DAY])
        self.assertEqual(outcome.run_statuses[question["run_id"]], "done")
        self.assertIn("The event day was higher than the baseline.", outcome.answer_text)
        self.assertNotIn("Please confirm the comparison days.", outcome.answer_text)
        self.assertEqual({f["name"]: f["value"] for f in outcome.findings}["baseline_days"], len(kept))
        retrieved = sorted(c.args["time_range"][:10] for c in outcome.boundary_calls if c.tool == "retrieve_subset")
        self.assertEqual(retrieved, sorted([EVENT_DAY, *kept]))
        # Both turns' supervisor tokens are counted.
        self.assertEqual(outcome.tokens[(SUPERVISOR_MODEL, "supervisor", "input")], 2000)
        self.assertIn("render.map", outcome.tool_calls)
        scored = score(holder["case"], outcome, harness=True)
        self.assertTrue(scored.passed, scored.first_failure and scored.first_failure.reason)

    def test_answer_message_shows_a_list_as_the_transcript_does(self):
        from benchmark.adapters.harness import answer_message

        self.assertEqual(answer_message(["2024-06-01", "2024-06-02"]), "2024-06-01, 2024-06-02")
        self.assertEqual(answer_message("confirm"), "confirm")

    def test_every_case_is_covered(self):
        from benchmark.adapters.harness import HarnessAdapter

        self.assertTrue(all(HarnessAdapter.covers(case) for case in _cases_by_id().values()))


class SupervisorTokensTests(unittest.TestCase):
    def test_zero_counts_are_left_out_and_cache_kinds_are_their_own(self):
        from benchmark.adapters.harness import supervisor_tokens

        turn = SimpleNamespace(input_tokens=10, output_tokens=0, cache_read_tokens=4, cache_creation_tokens=0)

        self.assertEqual(
            supervisor_tokens(turn, "m"), {("m", "supervisor", "input"): 10, ("m", "supervisor", "cache_read"): 4}
        )


class RunnerTests(unittest.TestCase):
    def test_the_harness_system_is_the_default_and_unknown_systems_are_refused(self):
        from benchmark import runner

        systems = []

        def fake_run(coroutine, **kwargs):
            systems.append(coroutine.cr_frame.f_locals["system"])
            coroutine.close()
            return 0

        with patch.object(runner.asyncio, "run", fake_run):
            self.assertEqual(runner.main(["--system", "harness"]), 0)
            self.assertEqual(runner.main([]), 0)
            for rejected in ("nonsense", "current"):
                with self.assertRaises(SystemExit):
                    runner.main(["--system", rejected])
        self.assertEqual(systems, ["harness", "harness"])

    def test_the_table_says_it_is_the_gate_system(self):
        from benchmark.scoring import HARNESS_TABLE_NOTE, format_results_table

        table = format_results_table([], HARNESS_TABLE_NOTE)

        self.assertTrue(table.startswith("Harness system"))
        self.assertIn("gate system", table)
        self.assertNotIn("gate system", format_results_table([]))
