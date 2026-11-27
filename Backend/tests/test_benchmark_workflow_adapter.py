"""The workflow adapter: each covered case run through the harness on the fake data boundary, no model.

The expected outcomes below are what the harness does today. A failing case
is pinned with the reason of its first failing check, so a fix to the
harness, the fake boundary or a spec shows up here as a diff.
"""
import asyncio
import dataclasses
import importlib.util
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from benchmark.cases import DAILY_UNAVAILABLE_PATTERN, BenchmarkCase, build_cases
from benchmark.checks import Completed
from benchmark.scoring import TOTAL_CASES

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow", "langgraph"]
_MISSING_DEPENDENCIES = any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)

# The place each case's inputs carry, exactly as the prompt words it.
PLACES = {
    "plotting_single_map": "New Jersey",
    "plotting_timeseries": "Newark NJ",
    "plotting_point_timeseries": "Newark, NJ",
    "failure_recovery_no_data": "Death Valley",
    "comparison_period_tempo_no2": "New Jersey",
    "ground_validation_tempo_vs_epa": "Newark NJ",
    "e2e_cross_source_ground_vs_satellite": "Newark NJ",
    "e2e_ground_relative_date": "Newark, New Jersey",
    "e2e_satellite_plot": "New Jersey",
    "e2e_routing_ozone_otc_satellite_only": "the OTC",
    "region_refusal_unresolvable_member": "NY + Wakanda",
    "region_refusal_buffer_too_large": "within 3000 miles of NYC",
    "event_baseline_confirmed": "New Jersey",
    "event_baseline_days_edited": "New Jersey",
    "event_baseline_too_few_days": "New Jersey",
}

# Refusals end the run needing input, so no data service is touched.
REFUSALS = ("region_refusal_unresolvable_member", "region_refusal_buffer_too_large")

# case id -> (workflow, passes, start of the first failing check's reason)
EXPECTED: dict[str, tuple[str, bool | None, str]] = {
    "plotting_single_map": ("map", True, ""),
    "plotting_timeseries": ("timeseries", True, ""),
    "plotting_point_timeseries": ("timeseries", True, None),
    "failure_recovery_no_data": ("map", True, ""),
    "comparison_period_tempo_no2": ("period-comparison", True, ""),
    "ground_validation_tempo_vs_epa": ("satellite-vs-monitor", True, ""),
    "e2e_cross_source_ground_vs_satellite": ("satellite-vs-monitor", True, None),
    "e2e_ground_relative_date": ("monitor-readings", True, ""),
    "e2e_satellite_plot": ("map", True, ""),
    "e2e_routing_ozone_otc_satellite_only": ("map", True, ""),
    "region_refusal_unresolvable_member": ("map", True, ""),
    "region_refusal_buffer_too_large": ("map", True, ""),
    "event_baseline_confirmed": ("event-vs-baseline", True, ""),
    "event_baseline_days_edited": ("event-vs-baseline", True, ""),
    "event_baseline_too_few_days": ("event-vs-baseline", True, ""),
}


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


class CaseFieldTests(unittest.TestCase):
    def test_the_covered_cases_are_the_ones_that_name_a_workflow(self):
        covered = {case.id: case.workflow for case in _cases_by_id().values() if case.workflow}

        self.assertEqual(covered, {case_id: workflow for case_id, (workflow, _, _) in EXPECTED.items()})

    def test_a_callable_gets_the_run_date_and_a_dict_is_used_as_given(self):
        cases = _cases_by_id()

        relative = cases["e2e_ground_relative_date"].inputs_for(date(2030, 1, 2))
        fixed = cases["plotting_single_map"].inputs_for(date(2030, 1, 2))

        self.assertEqual(relative["dates"], {"start": "2030-01-01", "end": "2030-01-01"})
        self.assertEqual(fixed["dates"], {"start": "2024-01-15", "end": "2024-01-15"})
        self.assertEqual(cases["discovery_no2_dataset"].inputs_for(date(2030, 1, 2)), {})

    def test_an_area_is_exactly_the_place_the_prompt_words(self):
        cases = _cases_by_id()

        self.assertEqual(set(PLACES), set(EXPECTED))
        for case_id, place in PLACES.items():
            self.assertIn(place, cases[case_id].prompt, case_id)
            self.assertEqual(cases[case_id].inputs_for(date(2030, 1, 2))["area"], place, case_id)

    def test_a_described_variable_is_the_one_its_file_holds(self):
        import xarray as xr

        class Volume(_NullVolume):
            def __init__(self) -> None:
                self.files: dict[str, object] = {}

            def add_zarr(self, handle, make, **kwargs) -> None:
                self.files[handle] = make

        volume = Volume()
        cases = {case.id: case for case in build_cases(volume)}
        for case_id in ("plotting_single_map", "plotting_timeseries", "comparison_period_tempo_no2",
                        "e2e_satellite_plot", "e2e_routing_ozone_otc_satellite_only"):
            handlers = cases[case_id].handlers
            described = asyncio.run(handlers["describe_dataset"](dataset_handle="d", detail=False, workspace_id="w"))
            retrieved = asyncio.run(handlers["retrieve_subset"](
                dataset_handle="d", aoi_handle="a", time_range="2024-01-01/2024-01-02", variables=[],
                output_format=None, workspace_id="w",
            ))
            dataset = volume.files[retrieved["obs_handle"]]()
            self.assertIsInstance(dataset, xr.Dataset)
            [variable] = described["variables"]
            self.assertEqual([variable["name"]], list(dataset.data_vars), case_id)
            self.assertEqual(variable["units"], dataset[variable["name"]].attrs["units"], case_id)

    def test_the_event_fakes_answer_one_day_at_a_time_in_the_services_shapes(self):
        from benchmark.cases import EVENT_BASELINE_DAYS, EVENT_DAY

        class Volume(_NullVolume):
            def __init__(self) -> None:
                self.files: dict[str, object] = {}

            def add_zarr(self, handle, make, **kwargs) -> None:
                self.files[handle] = make

        volume = Volume()
        handlers = {case.id: case for case in build_cases(volume)}["event_baseline_confirmed"].handlers

        def call(tool: str, day: str) -> dict:
            args = {"dataset_handle": "d", "aoi_handle": "a", "time_range": f"{day}T00:00:00Z/{day}T23:59:59Z",
                    "workspace_id": "w"}
            if tool == "retrieve_subset":
                args.update(variables=[], output_format=None)
            return asyncio.run(handlers[tool](**args))

        covered = call("check_coverage", EVENT_BASELINE_DAYS[0])
        self.assertEqual((covered["granule_count"] > 0, covered["covered"]), (True, True))
        self.assertEqual(set(covered) - {"dataset_handle", "aoi_handle", "time_range"},
                         {"all_time", "granule_count", "covered", "sampled_granules", "sample_granules"})
        empty = call("check_coverage", "2024-07-05")
        self.assertEqual((empty["granule_count"], empty["covered"]), (0, False))
        for day in (EVENT_DAY, *EVENT_BASELINE_DAYS):
            submitted = call("retrieve_subset", day)
            self.assertEqual(submitted["job_handle"], f"job_{submitted['obs_handle']}")
            dataset = volume.files[submitted["obs_handle"]]()
            self.assertEqual(str(dataset["time"].values[0])[:10], day)

    def test_the_case_fields_default_to_no_workflow(self):
        fields = {field.name: field.default for field in dataclasses.fields(BenchmarkCase)}

        self.assertIsNone(fields["workflow"])
        self.assertIsNone(fields["workflow_inputs"])


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class WorkflowAdapterRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from benchmark.adapters.workflow import WorkflowAdapter
        from benchmark.fake_earthdata import HandleVolume
        from benchmark.runner import run_workflow_suite

        async def run():
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                return await run_workflow_suite(WorkflowAdapter(), HandleVolume(tmp))

        cls.suite = asyncio.run(run())
        cls.results = {result.case.id: result for result in cls.suite.results}

    def test_each_covered_case_gets_the_pinned_result(self):
        self.assertEqual(set(self.results), set(EXPECTED))
        for case_id, (workflow, passes, reason) in EXPECTED.items():
            result = self.results[case_id]
            self.assertEqual(result.case.workflow, workflow, case_id)
            if passes is not None:
                self.assertEqual(result.passed, passes, f"{case_id}: {result.first_failure}")
            if passes is False:
                self.assertIsNotNone(result.first_failure, case_id)
                self.assertTrue(result.first_failure.reason.startswith(reason), f"{case_id}: {result.first_failure}")

    def test_the_point_case_asks_for_daily_values_and_the_workflow_refuses_them(self):
        result = self.results["plotting_point_timeseries"]
        outcome = result.outcome

        self.assertEqual(result.case.workflow_inputs["cadence"], "daily")
        self.assertEqual(outcome.mode, "Draft workflow: timeseries v2")
        self.assertEqual([call.tool for call in outcome.boundary_calls if call.tool.startswith("retrieve_")], [])
        self.assertEqual(outcome.tool_calls, ["sources.areas.resolve", "sources.satellite.get_point_series"])
        self.assertIn("a monthly product, so daily values are not available", outcome.answer_text)
        self.assertTrue(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertTrue(result.passed, result.first_failure)

    def test_a_refused_region_reaches_no_data_service_and_is_not_substituted(self):
        for case_id in REFUSALS:
            outcome = self.results[case_id].outcome

            self.assertEqual(outcome.boundary_calls, [], case_id)
            self.assertEqual(outcome.aqs_calls, [], case_id)
            self.assertEqual(outcome.handles, [], case_id)
            self.assertEqual(outcome.charts, [], case_id)
            self.assertEqual(outcome.tool_calls, ["sources.areas.resolve"], case_id)
            self.assertTrue(outcome.done, case_id)
            self.assertIsNone(outcome.error, case_id)

        self.assertIn("'Wakanda' in 'NY + Wakanda' could not be found", self.results[REFUSALS[0]].outcome.answer_text)
        self.assertIn("4,000,000-cell limit", self.results[REFUSALS[1]].outcome.answer_text)

    def test_a_named_product_reaches_the_workflow_and_is_the_one_used(self):
        outcome = self.results["plotting_single_map"].outcome

        self.assertIn("Product: TROPOMI_NO2", outcome.answer_text)
        self.assertEqual(outcome.boundary_calls[0].args["query"], "C3087325222-GES_DISC")

    def test_a_case_without_a_workflow_is_listed_as_not_covered_and_never_run(self):
        expected = [case.id for case in build_cases(_NullVolume()) if case.workflow is None]

        self.assertEqual(self.suite.uncovered, expected)
        self.assertEqual(len(self.suite.uncovered) + len(self.suite.results), TOTAL_CASES)
        self.assertFalse(set(self.suite.uncovered) & set(self.results))

    def test_the_adapter_refuses_a_case_without_a_workflow(self):
        from benchmark.adapters.workflow import NotCovered, WorkflowAdapter

        case = _cases_by_id()["discovery_no2_dataset"]

        self.assertFalse(WorkflowAdapter.covers(case))
        with self.assertRaises(NotCovered):
            asyncio.run(WorkflowAdapter().run(case, _NullVolume()))

    def test_the_relative_date_reaches_the_data_boundary_as_yesterday(self):
        outcome = self.results["e2e_ground_relative_date"].outcome
        used = {params.get(key) for _, params in outcome.aqs_calls for key in ("bdate", "edate")}

        self.assertIn((outcome.clock - timedelta(days=1)).strftime("%Y%m%d"), used)

    def test_an_outcome_records_the_run_the_way_the_checks_read_it(self):
        outcome = self.results["e2e_ground_relative_date"].outcome

        self.assertTrue(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.mode, "Draft workflow: monitor-readings v3")
        self.assertEqual(outcome.tokens, {})
        self.assertEqual(outcome.stages[0], "workflow.monitor-readings")
        self.assertEqual(outcome.tool_calls[:2], ["sources.areas.resolve", "sources.monitors.find"])
        self.assertNotIn("sources.monitors.daily_summary", outcome.tool_calls)
        self.assertEqual(outcome.findings[0]["name"], "monitors_found")
        self.assertIn("monitors_found: 1 monitors", outcome.answer_text)
        self.assertTrue(outcome.answer_text.startswith(outcome.mode))

    def test_a_satellite_run_hands_its_charts_and_artifacts_to_the_checks(self):
        outcome = self.results["plotting_single_map"].outcome

        self.assertTrue(outcome.done)
        self.assertEqual(len(outcome.charts), 1)
        self.assertEqual(len(outcome.artifact_ids), 1)
        self.assertEqual(outcome.handles, ["obs_map_1"])
        self.assertEqual(outcome.tool_calls[-1], "render.map")

    def test_the_ground_validation_pairs_the_monitor_s_cell_with_its_readings_from_one_retrieval(self):
        outcome = self.results["ground_validation_tempo_vs_epa"].outcome

        self.assertEqual(outcome.mode, "Draft workflow: satellite-vs-monitor v1")
        self.assertEqual([call.tool for call in outcome.boundary_calls].count("retrieve_subset"), 1)
        self.assertNotIn("retrieve_timeseries", [call.tool for call in outcome.boundary_calls])
        self.assertIn("operations.pair_with_monitors", outcome.tool_calls)
        findings = {f["name"]: f["value"] for f in outcome.findings}
        # The monitor's cell reads 1 to 7 on the days it has readings of 2 to 14; every other cell reads 10.
        self.assertEqual((findings["paired_days"], findings["pairs"], findings["monitors_paired"]), (7, 7, 1))
        self.assertAlmostEqual(findings["correlation_r"], 1.0)
        self.assertIn("agreement in pattern over time, not in level", outcome.answer_text)
        self.assertEqual(len(outcome.charts), 1)
        self.assertEqual(len(outcome.artifact_ids), 2)

    def test_the_cross_source_case_refuses_daily_pairs_from_the_monthly_tropomi_product(self):
        outcome = self.results["e2e_cross_source_ground_vs_satellite"].outcome

        self.assertNotIn("TEMPO", outcome.answer_text)
        self.assertRegex(outcome.answer_text, DAILY_UNAVAILABLE_PATTERN)
        self.assertNotIn("correlation_r", {f["name"] for f in outcome.findings})

    def test_a_run_that_ends_without_data_counts_as_completed(self):
        outcome = self.results["failure_recovery_no_data"].outcome

        self.assertTrue(Completed().evaluate(outcome).passed)
        self.assertIn("No data", outcome.answer_text)

    def retrieved_days(self, outcome) -> list[str]:
        return sorted(call.args["time_range"][:10] for call in outcome.boundary_calls if call.tool == "retrieve_subset")

    def test_the_confirmed_event_case_answers_the_days_as_offered_in_a_second_turn(self):
        from benchmark.cases import EVENT_BASELINE_DAYS, EVENT_DAY

        outcome = self.results["event_baseline_confirmed"].outcome

        [question] = outcome.questions
        self.assertEqual((question["step"], question["kind"]), ("days", "confirm_list"))
        self.assertEqual(question["items"], list(EVENT_BASELINE_DAYS))
        [sent] = outcome.answers
        self.assertEqual(
            {k: sent[k] for k in ("run_id", "step", "value")},
            {"run_id": question["run_id"], "step": "days", "value": list(EVENT_BASELINE_DAYS)},
        )
        # The event was retrieved before the answer was sent; every comparison day after it.
        calls = outcome.boundary_calls
        self.assertEqual(
            [c.args["time_range"][:10] for c in calls[:sent["calls_before"]] if c.tool == "retrieve_subset"], [EVENT_DAY],
        )
        self.assertEqual(
            sorted(c.args["time_range"][:10] for c in calls[sent["calls_before"]:] if c.tool == "retrieve_subset"),
            sorted(EVENT_BASELINE_DAYS),
        )
        self.assertEqual(outcome.run_statuses, {question["run_id"]: "done"})
        self.assertEqual(self.retrieved_days(outcome), sorted([EVENT_DAY, *EVENT_BASELINE_DAYS]))
        self.assertEqual(outcome.mode, "Draft workflow: event-vs-baseline v1")
        findings = {f["name"]: f["value"] for f in outcome.findings}
        # The event's file reads 1.0 above every comparison day's, cell for cell.
        self.assertAlmostEqual(findings["mean_difference"], 1.0)
        self.assertEqual(findings["baseline_days"], 10)
        self.assertEqual(len(outcome.charts), 1)
        self.assertIn("operations.select_comparison_days", outcome.tool_calls)
        # Two retrievals, the event and the baseline, each saved as waiting and then done.
        self.assertEqual(outcome.tool_calls.count("sources.satellite.get_data"), 4)

    def test_the_edited_event_case_leaves_the_removed_days_out_of_the_retrieval(self):
        from benchmark.cases import EVENT_BASELINE_DAYS, EVENT_DAY, EVENT_DAYS_REMOVED

        outcome = self.results["event_baseline_days_edited"].outcome
        kept = [day for day in EVENT_BASELINE_DAYS if day not in EVENT_DAYS_REMOVED]

        [answer] = outcome.answers
        self.assertEqual(answer["value"], kept)
        self.assertEqual(self.retrieved_days(outcome), sorted([EVENT_DAY, *kept]))
        self.assertEqual({f["name"]: f["value"] for f in outcome.findings}["baseline_days"], 8)

    def test_too_few_days_with_granules_stop_the_event_case_before_any_question(self):
        from benchmark.cases import SPARSE_EVENT_DAY

        outcome = self.results["event_baseline_too_few_days"].outcome

        self.assertEqual(outcome.questions, [])
        self.assertEqual(list(outcome.run_statuses.values()), ["insufficient_evidence"])
        self.assertEqual(self.retrieved_days(outcome), [SPARSE_EVENT_DAY])
        self.assertEqual(len([call for call in outcome.boundary_calls if call.tool == "check_coverage"]), 11)
        self.assertIn("Only 3 of the 10 candidate days have TEMPO_NO2 data", outcome.answer_text)
        self.assertEqual(outcome.charts, [])

    def test_nothing_in_an_outcome_or_its_saved_record_holds_a_credential(self):
        from benchmark.adapters.workflow import CREDENTIAL_STRINGS
        from benchmark.report import case_record

        for case_id, result in self.results.items():
            text = repr(result.outcome) + json.dumps(case_record(result), default=str)
            for secret in CREDENTIAL_STRINGS:
                self.assertNotIn(secret, text, case_id)


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class UnscriptedQuestionTests(unittest.TestCase):
    def test_a_question_no_answer_is_scripted_for_leaves_the_case_unfinished(self):
        from benchmark.adapters.workflow import WorkflowAdapter
        from benchmark.fake_earthdata import HandleVolume

        async def run():
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                volume = HandleVolume(tmp)
                case = {c.id: c for c in build_cases(volume)}["event_baseline_confirmed"]
                return await WorkflowAdapter().run(dataclasses.replace(case, answers=()), volume)

        outcome = asyncio.run(run())

        self.assertFalse(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.unfinished.startswith("the workflow stopped to ask: These days will be used"))
        self.assertEqual((outcome.questions, outcome.answers), ([], []))
        self.assertEqual(list(outcome.run_statuses.values()), ["needs_input"])


class NeedsInputKindTests(unittest.TestCase):
    def _result(self, reason, question=None):
        from types import SimpleNamespace

        return SimpleNamespace(reason=reason, question=question)

    def test_a_missing_connector_or_input_is_the_adapters_fault_and_a_refusal_is_not(self):
        from benchmark.adapters.workflow import _misconfigured

        connector = "This workflow needs the epa_aqs connector. Connect it in Settings -> Connectors, then run it again."

        self.assertTrue(_misconfigured(self._result(connector)))
        self.assertTrue(_misconfigured(self._result("Which area?", {"kind": "inputs", "inputs": []})))
        self.assertFalse(_misconfigured(self._result("'Wakanda' in 'NY + Wakanda' could not be found.")))
        self.assertFalse(_misconfigured(self._result("Pick one.", {"kind": "pick_one"})))


class StoppedReasonTests(unittest.TestCase):
    def _result(self, status, question=None, reason="why"):
        from types import SimpleNamespace

        return SimpleNamespace(status=status, question=question, reason=reason)

    def test_a_question_finishes_only_a_case_that_expects_one(self):
        from benchmark.adapters.workflow import stopped_reason

        question = {"kind": "text", "prompt": "Name the place.", "options": []}
        case = _cases_by_id()["plotting_single_map"]
        asking = dataclasses.replace(case, expects_question=True)

        self.assertEqual(stopped_reason(self._result("needs_input", question), case, "the workflow"),
                         "the workflow stopped to ask: Name the place.")
        self.assertIsNone(stopped_reason(self._result("needs_input", question), asking, "the workflow"))

    def test_waiting_incomplete_and_cancelled_never_finish(self):
        from benchmark.adapters.workflow import stopped_reason

        case = dataclasses.replace(_cases_by_id()["plotting_single_map"], expects_question=True)

        for status in ("waiting", "incomplete", "cancelled"):
            self.assertIsNotNone(stopped_reason(self._result(status), case, "the workflow"), status)
        for status in ("done", "failed", "insufficient_evidence"):
            self.assertIsNone(stopped_reason(self._result(status), case, "the workflow"), status)


class ComposeAnswerTests(unittest.TestCase):
    def _result(self, status, question=None, reason=None):
        from types import SimpleNamespace

        class Mode:
            def label(self):
                return "Draft workflow: map v1"

        return SimpleNamespace(
            mode=Mode(), findings=[], disclosures=[], gaps=[], status=status, reason=reason, question=question,
        )

    def test_a_question_the_user_must_answer_is_part_of_the_answer_with_its_options(self):
        from benchmark.adapters.workflow import compose_answer

        question = {"kind": "pick_one", "prompt": "Which Springfield?", "options": ["Illinois", "Missouri"]}

        text = compose_answer(self._result("needs_input", question, "The map needs an area."))

        self.assertEqual(text.splitlines()[1:], [
            "The map needs an area.", "Which Springfield?", "Options: Illinois, Missouri",
        ])

    def test_a_text_question_has_no_options_line(self):
        from benchmark.adapters.workflow import compose_answer

        text = compose_answer(self._result("needs_input", {"kind": "text", "prompt": "Name the place.", "options": []}))

        self.assertIn("Name the place.", text)
        self.assertNotIn("Options", text)

    def test_a_result_without_a_question_adds_nothing(self):
        from benchmark.adapters.workflow import compose_answer

        self.assertEqual(compose_answer(self._result("needs_input")), "Draft workflow: map v1")


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class AdapterFailureTests(unittest.TestCase):
    def _run(self, case, patches=()):
        from contextlib import ExitStack
        from unittest.mock import patch

        from benchmark.adapters.workflow import WorkflowAdapter
        from benchmark.fake_earthdata import HandleVolume

        async def run():
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, ExitStack() as stack:
                for target, kwargs in patches:
                    stack.enter_context(patch(target, **kwargs))
                return await WorkflowAdapter().run(case, HandleVolume(tmp))

        return asyncio.run(run())

    def test_a_missing_input_is_an_error_not_a_completed_run(self):
        case = dataclasses.replace(_cases_by_id()["plotting_single_map"], workflow_inputs={"area": "New Jersey"})

        outcome = self._run(case)

        self.assertFalse(outcome.done)
        self.assertIn("did not supply", outcome.error or "")

    def test_an_unexpected_exception_is_that_cases_error_by_type_name_only(self):
        case = _cases_by_id()["plotting_single_map"]

        outcome = self._run(case, [("tta_backend.harness.workflows.WorkflowRunner.start", {"side_effect": KeyError("secret-text")})])

        self.assertEqual(outcome.error, "the workflow run raised KeyError")
        self.assertNotIn("secret-text", repr(outcome))


class WorkflowReportTests(unittest.TestCase):
    def _suite(self):
        from benchmark.checks import CheckResult
        from benchmark.outcome import Outcome
        from benchmark.runner import WorkflowSuiteRun
        from benchmark.scoring import CaseResult

        cases = _cases_by_id()
        passing = CaseResult(cases["e2e_ground_relative_date"], Outcome(done=True), [CheckResult(True, "ok")], True, True)
        failing = CaseResult(
            cases["plotting_single_map"], Outcome(), [CheckResult(False, "no map was drawn")], True, False,
        )
        return WorkflowSuiteRun([passing, failing], ["discovery_no2_dataset"])

    def test_the_table_has_a_row_per_workflow_the_failures_and_the_uncovered_ids(self):
        from benchmark.scoring import format_workflow_table

        run = self._suite()
        table = format_workflow_table(run.results, run.uncovered)

        self.assertIn("not compared with the pass threshold", table)
        self.assertRegex(table, r"map\s+1\s+0\s+1")
        self.assertRegex(table, r"monitor-readings\s+1\s+1\s+0")
        self.assertIn("FAIL plotting_single_map: no map was drawn", table)
        self.assertIn("not covered (1): discovery_no2_dataset", table)

    def test_results_are_written_apart_from_the_current_systems(self):
        from benchmark.report import write_workflow_report

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_workflow_report(self._suite(), Path(tmp) / "workflow")
            records = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
            uncovered = json.loads((folder / "uncovered.json").read_text(encoding="utf-8"))
            table = (folder / "table.txt").read_text(encoding="utf-8")

        self.assertEqual(folder.parent, Path(tmp) / "workflow")
        self.assertEqual([record["workflow"] for record in records], ["monitor-readings", "map"])
        self.assertEqual(uncovered, ["discovery_no2_dataset"])
        self.assertIn("not compared", table)


if __name__ == "__main__":
    unittest.main()
