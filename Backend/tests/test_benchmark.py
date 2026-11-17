"""Benchmark: hermetic tests for the checks, scoring, adapter parsing and case
structure, plus the opt-in live run against the current system.

Everything here but the last class spends no model tokens.
"""
import dataclasses
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date

import pytest

from benchmark.adapters.current import outcome_from_sse, parse_sse, sources_in_header
from benchmark.checks import (
    AnswerMatches,
    AnswerMentionsAny,
    AqsDateRelativeToClock,
    AreaRequested,
    BoundaryCalled,
    BoundaryNotCalled,
    ChartProduced,
    Completed,
    GroundSourceNotUsed,
    GroundSourceUsed,
    HasResult,
    SatelliteSourceNotUsed,
    SatelliteSourceUsed,
    StageSubsequence,
    TimeRangeCovers,
    contains_subsequence,
)
from benchmark.outcome import BoundaryCall, Outcome, handles_from_calls

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]
_MISSING_DEPENDENCIES = any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)

# Names that exist only inside the current system's agents; no case may depend on them.
FORBIDDEN_INTERNAL_TOOLS = {
    "safe_retrieve", "await_retrieval", "point_timeseries", "plot_singular",
    "conduct_temporal_statistic", "compare", "validate_against_ground", "_finalize_sub_agent_result",
}


def _outcome(**fields) -> Outcome:
    return Outcome(done=True, **fields)


def _call(tool: str, **args) -> BoundaryCall:
    return BoundaryCall(tool, args, {})


class OutcomeTests(unittest.TestCase):
    def test_handles_come_from_retrieval_and_derived_results_without_repeats(self):
        calls = [
            BoundaryCall("retrieve_subset", {}, {"job_handle": "job_obs_1", "obs_handle": "obs_1"}),
            BoundaryCall("get_retrieval_status", {}, {"status": "ready", "obs_handle": "obs_1"}),
            BoundaryCall("align", {}, {"handle": "cube_1", "status": "ok"}),
            BoundaryCall("search_datasets", {}, {"dataset_handle": "dataset_1"}),
        ]

        self.assertEqual(handles_from_calls(calls), ["obs_1", "cube_1"])

    def test_a_submitted_retrieval_alone_gives_no_handle(self):
        calls = [BoundaryCall("retrieve_subset", {}, {"job_handle": "job_obs_1", "obs_handle": "obs_1"})]

        self.assertEqual(handles_from_calls(calls), [])

    def test_a_failed_retrieval_status_gives_no_handle(self):
        calls = [BoundaryCall("get_retrieval_status", {}, {"status": "failed", "obs_handle": "obs_1"})]

        self.assertEqual(handles_from_calls(calls), [])


class CheckTests(unittest.TestCase):
    def assertPasses(self, check, outcome):
        result = check.evaluate(outcome)
        self.assertTrue(result.passed, result.reason)

    def assertFails(self, check, outcome, reason_part):
        result = check.evaluate(outcome)
        self.assertFalse(result.passed)
        self.assertIn(reason_part, result.reason)

    def test_completed(self):
        self.assertPasses(Completed(), _outcome())
        self.assertFails(Completed(), Outcome(done=False), "never finished")
        self.assertFails(Completed(), Outcome(done=True, error="boom"), "boom")

    def test_boundary_called_by_tool_name(self):
        outcome = _outcome(boundary_calls=[_call("search_datasets", query="no2")])

        self.assertPasses(BoundaryCalled("search_datasets"), outcome)
        self.assertFails(BoundaryCalled("retrieve_subset"), outcome, "retrieve_subset was never called")

    def test_boundary_called_matches_an_argument_case_insensitively(self):
        outcome = _outcome(boundary_calls=[_call("define_area_of_interest", location="Houston, TX")])

        self.assertPasses(BoundaryCalled("define_area_of_interest", (("location", "houston"),)), outcome)
        self.assertFails(
            BoundaryCalled("define_area_of_interest", (("location", "Dallas"),)), outcome, "Houston, TX",
        )

    def test_boundary_called_needs_every_argument_in_one_call(self):
        outcome = _outcome(boundary_calls=[
            _call("define_area_of_interest", location="Newark", workspace_id="a"),
            _call("define_area_of_interest", location="Los Angeles", workspace_id="b"),
        ])

        both = BoundaryCalled("define_area_of_interest", (("location", "Newark"), ("workspace_id", "b")))

        self.assertFails(both, outcome, "never with")

    def test_boundary_not_called(self):
        outcome = _outcome(boundary_calls=[_call("retrieve_subset", time_range="2024-01-01/2024-01-02")])

        self.assertPasses(BoundaryNotCalled("define_area_of_interest"), outcome)
        self.assertFails(BoundaryNotCalled("retrieve_subset"), outcome, "retrieve_subset was called 1 time")

    def test_time_range_covers(self):
        outcome = _outcome(boundary_calls=[_call("retrieve_subset", time_range="2024-06-01/2024-06-08")])

        self.assertPasses(TimeRangeCovers("retrieve_subset", date(2024, 6, 1)), outcome)
        self.assertPasses(TimeRangeCovers("retrieve_subset", date(2024, 6, 8)), outcome)
        self.assertFails(TimeRangeCovers("retrieve_subset", date(2024, 6, 30)), outcome, "2024-06-01/2024-06-08")
        self.assertFails(TimeRangeCovers("retrieve_subset", date(2024, 6, 1)), _outcome(), "never called")

    def test_time_range_covers_reads_timestamps_and_ignores_unparseable_ranges(self):
        stamped = _outcome(boundary_calls=[_call("retrieve_subset", time_range="2024-06-01T00:00:00Z/2024-06-02T00:00:00Z")])
        garbled = _outcome(boundary_calls=[_call("retrieve_subset", time_range="last june")])

        self.assertPasses(TimeRangeCovers("retrieve_subset", date(2024, 6, 2)), stamped)
        self.assertFails(TimeRangeCovers("retrieve_subset", date(2024, 6, 2)), garbled, "last june")

    def test_satellite_source(self):
        used = _outcome(boundary_calls=[_call("search_datasets")])

        self.assertPasses(SatelliteSourceUsed(), used)
        self.assertFails(SatelliteSourceUsed(), _outcome(), "never used")
        self.assertPasses(SatelliteSourceNotUsed(), _outcome())
        self.assertFails(SatelliteSourceNotUsed(), used, "search_datasets called")

    def test_ground_source(self):
        used = _outcome(aqs_calls=[("dailyData/byBox", {})])

        self.assertPasses(GroundSourceUsed(), used)
        self.assertFails(GroundSourceUsed(), _outcome(), "never used")
        self.assertPasses(GroundSourceNotUsed(), _outcome())
        self.assertFails(GroundSourceNotUsed(), used, "dailyData/byBox")

    def test_has_result_accepts_a_handle_a_chart_or_an_artifact(self):
        self.assertPasses(HasResult(), _outcome(handles=["obs_1"]))
        self.assertPasses(HasResult(), _outcome(charts=[{"type": "heatmap"}]))
        self.assertPasses(HasResult(), _outcome(artifact_ids=["map_1"]))
        self.assertFails(HasResult(), _outcome(), "no handle, chart or artifact")

    def test_chart_produced_needs_a_chart_or_artifact_not_just_a_handle(self):
        self.assertPasses(ChartProduced(), _outcome(charts=[{"type": "heatmap"}]))
        self.assertPasses(ChartProduced(), _outcome(artifact_ids=["map_1"]))
        self.assertFails(ChartProduced(), _outcome(handles=["obs_1"]), "no chart or artifact")

    def test_answer_mentions_any_ignores_case_and_curly_apostrophes(self):
        check = AnswerMentionsAny(("no data", "couldn't find"))

        self.assertPasses(check, _outcome(answer_text="There is NO DATA for that day."))
        self.assertPasses(check, _outcome(answer_text="I couldn’t find any granules."))
        self.assertFails(check, _outcome(answer_text="Here is your map."), "none of")

    def test_stage_subsequence_allows_extra_and_repeated_stages(self):
        check = StageSubsequence(("search", "aoi", "render"))

        self.assertPasses(check, _outcome(stages=["search", "coverage", "aoi", "aoi", "render"]))
        self.assertFails(check, _outcome(stages=["aoi", "search", "render"]), "not found in order")
        self.assertFails(check, _outcome(stages=["search", "aoi"]), "['search', 'aoi']")

    def test_contains_subsequence(self):
        self.assertTrue(contains_subsequence(["a", "b", "c"], ["a", "c"]))
        self.assertFalse(contains_subsequence(["c", "a"], ["a", "c"]))


class AqsDateRelativeToClockTests(unittest.TestCase):
    clock = date(2026, 3, 15)

    def _make(self, bdate, edate=None):
        params = {"bdate": bdate, "edate": edate or bdate}
        return Outcome(done=True, clock=self.clock, aqs_calls=[("dailyData/byBox", params)])

    def test_passes_on_the_clock_minus_one_day(self):
        result = AqsDateRelativeToClock().evaluate(self._make("20260314"))

        self.assertTrue(result.passed, result.reason)

    def test_today_fails_since_that_is_the_bug_the_check_exists_for(self):
        result = AqsDateRelativeToClock().evaluate(self._make("20260315"))

        self.assertFalse(result.passed)
        self.assertIn("20260314", result.reason)
        self.assertIn("20260315", result.reason)

    def test_other_days_fail(self):
        self.assertFalse(AqsDateRelativeToClock().evaluate(self._make("20260313")).passed)

    def test_a_run_crossing_midnight_accepts_yesterday_of_either_date(self):
        def crossing(bdate):
            outcome = self._make(bdate)
            outcome.clock_end = date(2026, 3, 16)
            return outcome

        self.assertTrue(AqsDateRelativeToClock().evaluate(crossing("20260314")).passed)
        self.assertTrue(AqsDateRelativeToClock().evaluate(crossing("20260315")).passed)
        self.assertFalse(AqsDateRelativeToClock().evaluate(crossing("20260316")).passed)

    def test_either_bound_may_carry_the_date(self):
        outcome = self._make("20260301", "20260314")

        self.assertTrue(AqsDateRelativeToClock().evaluate(outcome).passed)

    def test_no_aqs_calls_fails(self):
        result = AqsDateRelativeToClock().evaluate(Outcome(done=True, clock=self.clock))

        self.assertFalse(result.passed)
        self.assertIn("saw []", result.reason)


JOINED_STREAM = (
    'event: status\ndata: {"message": "Searching...", "stage": "search"}\n\n'
    'event: tool_call\ndata: {"name": "search_datasets", "args": {}}\n\n'
    'event: status\ndata: {"message": "Resolving...", "stage": "aoi"}\n\n'
    'event: chart\ndata: {"type": "heatmap", "title": "NO2 over NJ", "chart_id": "c1", "lats": [1, 2]}\n\n'
    'event: artifact\ndata: {"id": "map_1", "type": "map", "title": "NO2 over NJ"}\n\n'
    'event: artifact\ndata: {"id": "map_1", "type": "map", "title": "NO2 over NJ"}\n\n'
    'event: text\ndata: {"content": "Data sources: TEMPO_NO2_L3 \\u00b7 EPA AQS\\n\\n"}\n\n'
    'event: text\ndata: {"content": "NO2 peaked near Newark."}\n\n'
    'event: done\ndata: {"thread_id": "t", "response": "Data sources: TEMPO_NO2_L3 \\u00b7 EPA AQS\\n\\nNO2 peaked near Newark.",'
    ' "artifacts": [], "tool_calls": []}\n\n'
)


class SseToOutcomeTests(unittest.TestCase):
    def _make(self, joined=JOINED_STREAM):
        calls = [
            BoundaryCall("retrieve_subset", {"time_range": "2024-01-15/2024-01-16"}, {"job_handle": "job_obs_1"}),
            BoundaryCall("get_retrieval_status", {"job_handle": "job_obs_1"}, {"status": "ready", "obs_handle": "obs_1"}),
        ]
        return outcome_from_sse(
            joined, boundary_calls=calls, aqs_calls=[("monitors/byBox", {})],
            clock=date(2026, 1, 1), elapsed_seconds=2.5,
        )

    def test_parse_sse_decodes_each_event_in_order(self):
        events = parse_sse(JOINED_STREAM)

        self.assertEqual([name for name, _ in events][:3], ["status", "tool_call", "status"])
        self.assertEqual(events[3][1]["type"], "heatmap")

    def test_charts_keep_type_and_title_but_not_the_payload(self):
        outcome = self._make()

        self.assertEqual(outcome.charts, [{"type": "heatmap", "title": "NO2 over NJ", "chart_id": "c1"}])

    def test_artifacts_are_listed_once(self):
        self.assertEqual(self._make().artifact_ids, ["map_1"])

    def test_stages_and_tool_calls_keep_their_order(self):
        outcome = self._make()

        self.assertEqual(outcome.stages, ["search", "aoi"])
        self.assertEqual(outcome.tool_calls, ["search_datasets"])

    def test_answer_comes_from_done_and_names_its_sources(self):
        outcome = self._make()

        self.assertTrue(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertIn("NO2 peaked near Newark.", outcome.answer_text)
        self.assertEqual(outcome.sources_header, {"TEMPO_NO2_L3", "EPA AQS"})

    def test_boundary_facts_and_run_metadata_pass_through(self):
        outcome = self._make()

        self.assertEqual(outcome.handles, ["obs_1"])
        self.assertEqual(len(outcome.boundary_calls), 2)
        self.assertEqual(outcome.aqs_calls, [("monitors/byBox", {})])
        self.assertEqual(outcome.clock, date(2026, 1, 1))
        self.assertEqual(outcome.elapsed_seconds, 2.5)

    def test_answer_falls_back_to_streamed_text_without_a_done_response(self):
        joined = 'event: text\ndata: {"content": "Part one. "}\n\nevent: text\ndata: {"content": "Part two."}\n\n'

        outcome = self._make(joined)

        self.assertEqual(outcome.answer_text, "Part one. Part two.")
        self.assertFalse(outcome.done)
        self.assertIsNone(outcome.sources_header)

    def test_an_error_event_marks_the_run_errored(self):
        joined = 'event: error\ndata: {"detail": "The agent failed."}\n\n'

        outcome = self._make(joined)

        self.assertEqual(outcome.error, "The agent failed.")
        self.assertFalse(Completed().evaluate(outcome).passed)

    def test_sources_header_is_none_without_one(self):
        self.assertIsNone(sources_in_header("No header here."))
        self.assertEqual(sources_in_header("Data sources: EPA AQS\n\nAnswer."), {"EPA AQS"})


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class FakeMcpRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_tool_name_arguments_and_result_of_each_call(self):
        from fastmcp import Client

        from benchmark.fake_earthdata import build_fake_mcp

        async def define_area_of_interest(location, workspace_id):
            return {"aoi_handle": "aoi_9", "location": location}

        recorder: list[BoundaryCall] = []
        mcp = build_fake_mcp({"define_area_of_interest": define_area_of_interest}, recorder=recorder)

        async with Client(mcp) as client:
            await client.call_tool("define_area_of_interest", {"location": "Houston", "workspace_id": "w1"})
            await client.call_tool("search_datasets", {"query": "no2"})

        self.assertEqual([call.tool for call in recorder], ["define_area_of_interest", "search_datasets"])
        self.assertEqual(recorder[0].args, {"location": "Houston", "workspace_id": "w1"})
        self.assertEqual(recorder[0].result, {"aoi_handle": "aoi_9", "location": "Houston"})
        self.assertEqual(recorder[1].args["query"], "no2")
        self.assertEqual(recorder[1].result["tool"], "search_datasets")

    async def test_nothing_is_recorded_without_a_recorder(self):
        from fastmcp import Client

        from benchmark.fake_earthdata import build_fake_mcp

        async with Client(build_fake_mcp()) as client:
            result = await client.call_tool("search_datasets", {"query": "no2"})

        self.assertEqual(result.data["tool"], "search_datasets")


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class CaseStructureTests(unittest.TestCase):
    def setUp(self):
        from benchmark.cases import build_cases
        from benchmark.fake_earthdata import HandleVolume

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.cases = build_cases(HandleVolume(tmpdir.name))

    def test_builds_the_total_number_of_cases_with_unique_ids(self):
        from benchmark.scoring import TOTAL_CASES

        ids = [case.id for case in self.cases]

        self.assertEqual(len(ids), TOTAL_CASES)
        self.assertEqual(len(set(ids)), len(ids))

    def test_category_counts(self):
        self.assertEqual(
            Counter(case.category for case in self.cases),
            {
                "discovery": 2,
                "retrieval": 2,
                "plotting": 3,
                "comparison_setup": 2,
                "failure_recovery": 2,
                "ground_validation": 1,
                "comparison": 1,
                "e2e_ground": 1,
                "e2e_satellite": 1,
                "e2e_cross_source": 1,
                "e2e_routing": 2,
                "region_refusal": 2,
            },
        )

    def test_every_case_has_a_prompt_and_at_least_one_check(self):
        for case in self.cases:
            with self.subTest(case=case.id):
                self.assertTrue(case.prompt)
                self.assertTrue(case.checks)

    def test_every_category_has_a_positive_budget(self):
        from benchmark.scoring import CATEGORY_BUDGETS

        missing = {case.category for case in self.cases} - CATEGORY_BUDGETS.keys()

        self.assertFalse(missing, f"categories with no latency budget: {missing}")
        for category, budget in CATEGORY_BUDGETS.items():
            self.assertGreater(budget, 0, category)
        self.assertEqual(CATEGORY_BUDGETS["region_refusal"], 45.0)

    def test_pass_threshold_allows_three_misses(self):
        from benchmark.scoring import PASS_THRESHOLD, TOTAL_CASES

        self.assertEqual(TOTAL_CASES - PASS_THRESHOLD, 3)

    def test_no_check_names_a_current_system_tool(self):
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from strings(item)

        for case in self.cases:
            for check in case.checks:
                for field in dataclasses.fields(check):
                    for text in strings(getattr(check, field.name)):
                        with self.subTest(case=case.id, check=type(check).__name__, field=field.name):
                            self.assertNotIn(text, FORBIDDEN_INTERNAL_TOOLS)

    def test_region_cases_carry_the_geocoder_fixture(self):
        region = [case for case in self.cases if case.category == "region_refusal"]

        self.assertEqual([case.id for case in region], ["region_refusal_unresolvable_member", "region_refusal_buffer_too_large"])
        for case in region:
            self.assertEqual([spec.target for spec in case.patches], ["tta_backend.utils.plotting.GeocodingService.ageocode"])

    def test_the_relative_date_case_checks_against_the_run_clock(self):
        case = next(case for case in self.cases if case.id == "e2e_ground_relative_date")

        self.assertTrue(any(isinstance(check, AqsDateRelativeToClock) for check in case.checks))

    def test_the_no_data_case_reads_the_answer_not_the_options_offered(self):
        case = next(case for case in self.cases if case.id == "failure_recovery_no_data")
        marker_check = next(check for check in case.checks if isinstance(check, AnswerMatches))

        self.assertTrue(marker_check.evaluate(_outcome(answer_text="TEMPO has no granules for that day.")).passed)
        self.assertTrue(marker_check.evaluate(_outcome(answer_text="I couldn't find observations.")).passed)
        self.assertFalse(marker_check.evaluate(_outcome(answer_text="Here is your map of Death Valley.")).passed)

    def test_a_passing_outcome_for_a_retrieval_case_satisfies_its_checks(self):
        from benchmark.scoring import score

        case = next(case for case in self.cases if case.id == "retrieval_tempo_no2")
        outcome = _outcome(
            handles=["obs_retrieval_1"],
            boundary_calls=[
                _call("search_datasets", query="TEMPO NO2"),
                _call("define_area_of_interest", location="Houston, TX"),
                _call("retrieve_subset", time_range="2024-06-01/2024-06-02"),
            ],
        )

        self.assertTrue(score(case, outcome).passed)

    def test_the_refusal_cases_fail_if_a_substitute_region_was_retrieved(self):
        from benchmark.scoring import score

        case = next(case for case in self.cases if case.id == "region_refusal_buffer_too_large")
        outcome = _outcome(boundary_calls=[_call("define_area_of_interest", location="Northeastern United States")])

        result = score(case, outcome)

        self.assertFalse(result.passed)
        self.assertIn("define_area_of_interest was called", result.first_failure.reason)


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class NewarkAqsFakeTests(unittest.IsolatedAsyncioTestCase):
    """The AQS fakes answer the harness's endpoints as they do the current system's."""

    async def test_site_and_box_endpoints_serve_the_same_readings(self):
        from benchmark.cases import _requested_day_aqs_get, _week_of_january_aqs_get

        site = {"param": "42602", "bdate": "20240102", "edate": "20240102", "state": "34", "county": "017", "site": "0006"}
        box = {"param": "42602", "bdate": "20240102", "edate": "20240102", "minlat": 40.0, "maxlat": 41.0, "minlon": -75.0, "maxlon": -74.0}
        for get in (_week_of_january_aqs_get, _requested_day_aqs_get):
            by_site = (await get("dailyData/bySite", site))["Data"]
            self.assertTrue(by_site)
            self.assertEqual(by_site, (await get("dailyData/byBox", box))["Data"])
        self.assertEqual((await _week_of_january_aqs_get("dailyData/bySite", {**site, "site": "0099"}))["Data"], [])
        self.assertEqual((await _week_of_january_aqs_get("dailyData/bySite", {**site, "param": "44201"}))["Data"], [])

    async def test_the_monitor_is_listed_only_for_a_box_holding_it_and_for_no2(self):
        from benchmark.cases import _week_of_january_aqs_get as get

        holding = {"param": "42602", "minlat": 40.6, "maxlat": 40.9, "minlon": -74.3, "maxlon": -74.0}
        elsewhere = {**holding, "minlat": 30.0, "maxlat": 31.0}
        self.assertEqual(len((await get("monitors/byBox", holding))["Data"]), 1)
        self.assertEqual((await get("monitors/byBox", elsewhere))["Data"], [])
        self.assertEqual((await get("monitors/byBox", {**holding, "param": "44201"}))["Data"], [])
        self.assertEqual((await get("dailyData/byBox", {**elsewhere, "bdate": "20240101", "edate": "20240103"}))["Data"], [])

    async def test_the_daily_max_is_above_the_daily_mean(self):
        from benchmark.cases import _week_of_january_aqs_get as get

        site = {"param": "42602", "bdate": "20240101", "edate": "20240103", "state": "34", "county": "017", "site": "0006"}
        for row in (await get("dailyData/bySite", site))["Data"]:
            self.assertGreater(float(row["first_max_value"]), float(row["arithmetic_mean"]))

    def test_the_monitor_is_inside_the_offline_geocoders_newark_box(self):
        from benchmark.adapters.harness_boundary import _NEWARK as newark
        from benchmark.cases import _NEWARK_MONITOR

        south, north, west, east = newark["bbox"]
        self.assertTrue(south <= float(_NEWARK_MONITOR["latitude"]) <= north)
        self.assertTrue(west <= float(_NEWARK_MONITOR["longitude"]) <= east)


class StarterPromptCoverageTests(unittest.TestCase):
    def test_every_starter_prompt_names_a_benchmark_case(self):
        from benchmark.fake_earthdata import HandleVolume
        from benchmark.runner import starter_prompts_missing_cases

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = starter_prompts_missing_cases(HandleVolume(tmpdir))

        self.assertEqual(missing, [])

    def test_starter_prompts_span_more_than_one_category(self):
        from tta_backend.config.starter_prompts import STARTER_PROMPTS

        self.assertGreater(len({entry["category"] for entry in STARTER_PROMPTS}), 1)

    def test_each_starter_prompt_shows_the_workflow_its_case_runs_on_the_same_question(self):
        from benchmark.cases import build_cases
        from benchmark.fake_earthdata import HandleVolume
        from tta_backend.config.starter_prompts import STARTER_PROMPTS
        from tta_backend.datasets.registry import load_registry
        from tta_backend.harness.capabilities.catalogue import build_registry
        from tta_backend.harness.workflows.loader import load_packs
        from tta_backend.services.harness_chat import PACKS_ROOT

        with tempfile.TemporaryDirectory() as tmpdir:
            cases = {case.id: case for case in build_cases(HandleVolume(tmpdir))}
        workflows = load_packs(PACKS_ROOT, build_registry(), load_registry()).names()
        self.assertEqual(len({entry["category"] for entry in STARTER_PROMPTS}), len(STARTER_PROMPTS))
        for entry in STARTER_PROMPTS:
            with self.subTest(entry["id"]):
                case = cases[entry["id"]]
                self.assertIn(entry["category"], workflows)
                self.assertEqual(case.workflow, entry["category"])
                self.assertEqual(case.prompt, entry["prompt"])


class _FakeAdapter:
    def __init__(self, on_run=None):
        self.ran: list[str] = []
        self.on_run = on_run

    async def run(self, case, volume):
        self.ran.append(case.id)
        if self.on_run is not None:
            self.on_run()
        return Outcome(done=True, elapsed_seconds=0.1)


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class RunSuiteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        from benchmark.fake_earthdata import HandleVolume

        self.volume = HandleVolume(tmpdir.name)

    async def test_runs_every_case_and_scores_each(self):
        from benchmark.runner import run_suite
        from benchmark.scoring import TOTAL_CASES

        adapter = _FakeAdapter()

        results = await run_suite(adapter, self.volume)

        self.assertEqual(len(results), TOTAL_CASES)
        self.assertEqual(adapter.ran, [result.case.id for result in results])
        self.assertTrue(all(result.within_budget for result in results))

    async def test_a_harness_suite_scores_with_the_harness_checks_where_a_case_has_them(self):
        from benchmark.runner import run_suite

        current = await run_suite(_FakeAdapter(), self.volume)
        harness = await run_suite(_FakeAdapter(), self.volume, harness=True)

        self.assertEqual({result.check_set for result in current}, {"default"})
        with_harness_checks = {result.case.id for result in harness if result.case.harness_checks is not None}
        self.assertTrue(with_harness_checks)
        for result in harness:
            expected = "harness" if result.case.id in with_harness_checks else "default"
            self.assertEqual(result.check_set, expected, result.case.id)

    async def test_raises_when_a_provider_retry_was_logged(self):
        from benchmark.runner import RateLimitDetected, run_suite

        def log_retry():
            logging.getLogger("groq._base_client").warning("Retrying request to /v1/chat/completions in 3.0 seconds")

        with self.assertRaises(RateLimitDetected):
            await run_suite(_FakeAdapter(on_run=log_retry), self.volume)

    async def test_raises_when_the_openai_client_behind_portkey_logs_a_retry(self):
        from benchmark.runner import RateLimitDetected, run_suite

        def log_retry():
            logging.getLogger("openai._base_client").info("Retrying request to /chat/completions in 0.4 seconds")

        with self.assertRaises(RateLimitDetected):
            await run_suite(_FakeAdapter(on_run=log_retry), self.volume)

    async def test_ignores_unrelated_log_records(self):
        from benchmark.runner import run_suite

        def log_success():
            logging.getLogger("httpx").info('HTTP Request: POST https://api.groq.com/v1 "HTTP/1.1 200 OK"')

        results = await run_suite(_FakeAdapter(on_run=log_success), self.volume)

        self.assertEqual(len(results), 20)

    async def test_a_case_over_its_category_budget_fails_even_when_every_check_passes(self):
        from benchmark.cases import BenchmarkCase
        from benchmark.scoring import CATEGORY_BUDGETS, score

        case = BenchmarkCase(id="c", category="discovery", prompt="p", handlers={}, checks=(Completed(),))
        slow = Outcome(done=True, elapsed_seconds=CATEGORY_BUDGETS["discovery"] + 1)

        result = score(case, slow)

        self.assertFalse(result.within_budget)
        self.assertFalse(result.passed)
        self.assertIn(f"over the {CATEGORY_BUDGETS['discovery']:.0f}s budget", _table_row(result))


def _table_row(result) -> str:
    from benchmark.scoring import format_results_table

    # Header, rule, the case's row, rule, total.
    return format_results_table([result]).splitlines()[2]


class ResultsTableTests(unittest.TestCase):
    def test_table_lists_case_category_verdict_seconds_and_first_failure(self):
        from benchmark.cases import BenchmarkCase
        from benchmark.scoring import format_results_table, score

        passing = BenchmarkCase(id="discovery_no2_dataset", category="discovery", prompt="p", handlers={}, checks=(Completed(),))
        failing = BenchmarkCase(
            id="plotting_single_map", category="plotting", prompt="p", handlers={},
            checks=(Completed(), BoundaryCalled("retrieve_subset"), HasResult()),
        )

        table = format_results_table([
            score(passing, Outcome(done=True, elapsed_seconds=1.234)),
            score(failing, Outcome(done=True, elapsed_seconds=50.5)),
        ])

        self.assertIn("discovery_no2_dataset", table)
        self.assertIn("plotting", table)
        self.assertIn("PASS", table)
        self.assertIn("FAIL", table)
        self.assertIn("1.23", table)
        self.assertIn("50.50", table)
        self.assertIn("retrieve_subset was never called", table)
        # The passing row carries no reason, and the failure shows only its first.
        self.assertNotIn("no handle, chart or artifact", table)

    def test_table_shows_tokens_per_case_and_totals_with_a_dash_when_none_were_reported(self):
        from benchmark.cases import BenchmarkCase
        from benchmark.scoring import format_results_table, score

        case = BenchmarkCase(id="discovery_no2_dataset", category="discovery", prompt="p", handlers={}, checks=(Completed(),))
        counted = Outcome(done=True, tokens={
            ("sonnet", "supervisor", "input"): 1000,
            ("haiku", "satellite", "input"): 2500,
            ("haiku", "satellite", "output"): 300,
            ("haiku", "satellite", "cache_read"): 2000,
        })
        uncounted = Outcome(done=True)

        lines = format_results_table([score(case, counted), score(case, uncounted)]).splitlines()

        header, counted_row, uncounted_row = lines[0], lines[2], lines[3]
        total_row = next(line for line in lines if line.startswith("total"))
        for label in ("in tok", "out tok", "cached"):
            self.assertIn(label, header)
        self.assertEqual(counted_row.split()[4:7], ["3,500", "300", "2,000"])
        self.assertEqual(uncounted_row.split()[4:7], ["-", "-", "-"])
        self.assertEqual(total_row.split()[0], "total")
        self.assertEqual(total_row.split()[1], "2/2")
        # The total row has no category, so its tokens start one field earlier.
        self.assertEqual(total_row.split()[3:6], ["3,500", "300", "2,000"])


class PricingTests(unittest.TestCase):
    HAIKU = "@gcp-internal-dataset-integration/anthropic.claude-haiku-4-5@20251001"
    SONNET = "@gcp-internal-dataset-integration/anthropic.claude-sonnet-5"

    def test_configured_model_ids_resolve_to_their_prices(self):
        from benchmark.pricing import price_for

        self.assertEqual(price_for(self.HAIKU).input, 1.00)
        self.assertEqual(price_for(self.SONNET).output, 10.00)
        self.assertIsNone(price_for("gemini-3.7-flash"))

    def test_a_longer_model_name_is_not_priced_as_one_it_contains(self):
        from benchmark.pricing import price_for

        self.assertEqual(price_for("anthropic.claude-sonnet-5-5").cache_read, 0.10)
        self.assertEqual(price_for("anthropic.claude-sonnet-5").cache_read, 0.20)

    def test_cache_reads_are_charged_at_the_cache_price_not_as_input(self):
        from benchmark.pricing import estimate_cost

        outcome = Outcome(tokens={
            (self.HAIKU, "satellite", "input"): 1_000_000,
            (self.HAIKU, "satellite", "cache_read"): 400_000,
            (self.HAIKU, "satellite", "output"): 100_000,
        })

        # 600k uncached at $1, 400k cached at $0.10, 100k out at $5.
        self.assertAlmostEqual(estimate_cost(outcome).dollars, 0.60 + 0.04 + 0.50)

    def test_models_are_priced_separately_and_unpriced_ones_are_named(self):
        from benchmark.pricing import estimate_cost

        outcome = Outcome(tokens={
            (self.SONNET, "supervisor", "input"): 500_000,
            (self.HAIKU, "ground_sensor", "output"): 200_000,
            ("gemini-3.7-flash", "satellite", "input"): 9_000,
        })

        cost = estimate_cost(outcome)

        self.assertAlmostEqual(cost.dollars, 1.00 + 1.00)
        self.assertEqual(cost.unpriced, ("gemini-3.7-flash",))

    def test_no_tokens_means_no_estimate(self):
        from benchmark.pricing import estimate_cost

        self.assertIsNone(estimate_cost(Outcome()))

    def test_table_shows_cost_per_case_a_total_and_names_unpriced_models(self):
        from benchmark.cases import BenchmarkCase
        from benchmark.scoring import format_results_table, score

        case = BenchmarkCase(id="c", category="discovery", prompt="p", handlers={}, checks=(Completed(),))
        priced = Outcome(done=True, tokens={(self.HAIKU, "satellite", "output"): 2_000})
        partly = Outcome(done=True, tokens={
            (self.HAIKU, "satellite", "output"): 2_000,
            ("gemini-3.7-flash", "supervisor", "input"): 10,
        })

        lines = format_results_table([score(case, priced), score(case, partly), score(case, Outcome(done=True))]).splitlines()
        total_row = next(line for line in lines if line.startswith("total"))

        self.assertIn("est cost", lines[0])
        self.assertIn("$0.0100", lines[2])
        self.assertIn("$0.0100?", lines[3])
        self.assertEqual(lines[4].split()[-1], "-")
        self.assertIn("$0.0200?", total_row)
        self.assertIn("no price for: gemini-3.7-flash", lines[-1])


# The explore run's answer for the 1990 Death Valley case, which the old pattern missed.
LIVE_NO_DATA_ANSWER = (
    "Exploratory\nI could not plot TEMPO NO2 over Death Valley for 1990-01-01 because the request falls outside "
    "the TEMPO record. The check_availability call returned status \"done\" but available=false with reason "
    "\"outside_record\". TEMPO_NO2's record begins 2023-08-02 (per the product description) and has no end date "
    "(ongoing); the TEMPO instrument did not exist or operate in 1990. No grid was retrieved and no chart was made."
)


class NoDataAnswerTests(unittest.TestCase):
    def _check(self):
        from benchmark.cases import NO_DATA_PATTERN

        return AnswerMatches(NO_DATA_PATTERN, "says no data was found")

    def test_recognises_no_data_answers_that_name_the_product(self):
        for answer in (
            "No TEMPO NO2 data available for Death Valley on 1990-01-01.",
            "There were no granules for that day.",
            "TEMPO observations began in 2023; data from 1990 does not exist in this product.",
            "I couldn't find any coverage.",
            "That product is unavailable before 2018.",
            LIVE_NO_DATA_ANSWER,
            "1990-01-01 falls outside the TEMPO record.",
            "The date is before the product's record begins.",
            "The instrument didn't exist then.",
            "The satellite did not operate in 1990.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(self._check().evaluate(_outcome(answer_text=answer)).passed)

    def test_does_not_mistake_a_normal_answer_for_no_data(self):
        for answer in (
            "Here is the TEMPO NO2 map over Houston; values range from 1 to 4 mol/m^2.",
            "The record shows NO2 rose through the afternoon.",
            "Data was found for Death Valley on 2024-01-01 and the map is ready.",
            "The grid was retrieved; it covers the outside of the city and the instrument operated normally.",
            "The observations did not exist in the cache, so I retrieved them and here is the map.",
            "Before the record high of 5, values were normal; data found for Houston.",
            "outside hours were recorded",
            "The peak occurred before the end of the record.",
            "Outside the city; the record shows 3 days.",
        ):
            with self.subTest(answer=answer):
                result = self._check().evaluate(_outcome(answer_text=answer))

                self.assertFalse(result.passed)
                self.assertIn("never says no data was found", result.reason)


LIVE_MONTHLY_REFUSAL_ANSWER = (
    "Exploratory\nA daily point-series for TROPOMI NO2 at 40.735,-74.172 (Newark, NJ) for January 2024 could not "
    "be produced. The only product serving the no2_tropospheric_column quantity is TROPOMI_NO2, whose native "
    "cadence is monthly, not daily — the request was refused with reason \"cadence_too_coarse\"."
)


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class HarnessChecksTests(unittest.TestCase):
    def _case(self, case_id):
        from benchmark.cases import build_cases
        from benchmark.fake_earthdata import HandleVolume

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            return next(case for case in build_cases(HandleVolume(tmp)) if case.id == case_id)

    def test_the_point_case_passes_on_the_live_refusal_and_fails_when_data_was_retrieved(self):
        from benchmark.scoring import score

        case = self._case("plotting_point_timeseries")
        refused = _outcome(answer_text=LIVE_MONTHLY_REFUSAL_ANSWER)
        retrieved = _outcome(
            answer_text="Draft workflow: timeseries v1\npoint_mean: 2.0 molecules/cm^2\nProduct: TROPOMI_NO2",
            boundary_calls=[_call("retrieve_timeseries", time_range="2024-01-01/2024-01-31")],
        )

        self.assertTrue(score(case, refused, harness=True).passed)
        scored = score(case, retrieved, harness=True)
        self.assertFalse(scored.passed)
        self.assertEqual(
            [result.passed for result in scored.check_results], [True, False, False, True],
        )
        self.assertFalse(score(case, _outcome(answer_text=LIVE_MONTHLY_REFUSAL_ANSWER, boundary_calls=[
            _call("retrieve_subset"),
        ]), harness=True).passed)

    def test_the_daily_pattern_needs_a_negation_or_refusal_tied_to_daily(self):
        from benchmark.cases import DAILY_UNAVAILABLE_PATTERN

        check = AnswerMatches(DAILY_UNAVAILABLE_PATTERN, "says daily values are not available")
        for answer in (
            "Monthly and daily views are available.",
            "Daily mean computed. The cadence is daily.",
            "Daily values were retrieved; the product is monthly overall.",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(check.evaluate(_outcome(answer_text=answer)).passed)
        for answer in (
            LIVE_MONTHLY_REFUSAL_ANSWER,
            "Daily values are not available from this product.",
            "I cannot give a daily series.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(check.evaluate(_outcome(answer_text=answer)).passed)

    def test_the_result_records_which_check_set_scored_it(self):
        from benchmark.report import case_record
        from benchmark.scoring import score

        case = self._case("plotting_point_timeseries")
        other = self._case("discovery_no2_dataset")

        self.assertEqual(score(case, _outcome(), harness=True).check_set, "harness")
        self.assertEqual(score(case, _outcome()).check_set, "default")
        self.assertEqual(score(other, _outcome(), harness=True).check_set, "default")
        self.assertEqual(case_record(score(case, _outcome(), harness=True))["check_set"], "harness")
        self.assertEqual(case_record(score(case, _outcome()))["check_set"], "default")

    def test_the_current_system_keeps_the_original_checks(self):
        from benchmark.scoring import score

        case = self._case("plotting_point_timeseries")
        refused = _outcome(answer_text=LIVE_MONTHLY_REFUSAL_ANSWER)

        self.assertFalse(score(case, refused).passed)
        self.assertEqual(len(score(case, refused).check_results), len(case.checks))
        self.assertEqual(len(score(case, refused, harness=True).check_results), len(case.harness_checks))

    def test_a_case_without_harness_checks_is_scored_on_its_checks_either_way(self):
        from benchmark.scoring import score

        case = self._case("discovery_no2_dataset")

        self.assertIsNone(case.harness_checks)
        self.assertEqual(
            [r.reason for r in score(case, _outcome(), harness=True).check_results],
            [r.reason for r in score(case, _outcome()).check_results],
        )

    def test_the_harness_suite_scores_with_harness_checks_and_the_current_suite_does_not(self):
        import asyncio

        from benchmark.runner import run_suite, run_workflow_suite

        class Adapter:
            covers = staticmethod(lambda case: case.id == "plotting_point_timeseries")

            async def run(self, case, volume):
                return Outcome(done=True, elapsed_seconds=0.1, answer_text=LIVE_MONTHLY_REFUSAL_ANSWER)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            from benchmark.fake_earthdata import HandleVolume

            volume = HandleVolume(tmp)
            harness = asyncio.run(run_workflow_suite(Adapter(), volume))
            current = {r.case.id: r for r in asyncio.run(run_suite(Adapter(), volume))}

        [result] = harness.results
        self.assertTrue(result.passed)
        self.assertFalse(current["plotting_point_timeseries"].passed)


class AreaRequestedTests(unittest.TestCase):
    NEWARK = (40.7357, -74.1724)

    def test_passes_on_the_place_name_in_any_case(self):
        outcome = _outcome(boundary_calls=[_call("define_area_of_interest", location="newark, NJ")])

        self.assertTrue(AreaRequested("Newark", self.NEWARK).evaluate(outcome).passed)

    def test_passes_on_a_bounding_box_that_contains_the_place(self):
        outcome = _outcome(boundary_calls=[
            _call("define_area_of_interest", location="-124.371654,32.533365,-114.125068,42.000768"),
        ])

        self.assertTrue(AreaRequested("California", (36.7783, -119.4179)).evaluate(outcome).passed)

    def test_fails_on_a_box_elsewhere_and_says_what_was_requested(self):
        outcome = _outcome(boundary_calls=[_call("define_area_of_interest", location="-80,25,-79,26")])

        result = AreaRequested("Newark", self.NEWARK).evaluate(outcome)

        self.assertFalse(result.passed)
        self.assertIn("-80,25,-79,26", result.reason)

    def test_fails_when_no_area_was_requested(self):
        result = AreaRequested("Newark", self.NEWARK).evaluate(_outcome())

        self.assertFalse(result.passed)
        self.assertIn("never called", result.reason)


class FixtureLocationTests(unittest.TestCase):
    """A grid outside the area a prompt names narrows to nothing, and the case
    then fails for a reason that has nothing to do with the system."""

    def setUp(self):
        from benchmark.cases import build_cases
        from benchmark.fake_earthdata import HandleVolume

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.volume = HandleVolume(tmpdir.name)
        build_cases(self.volume)

    def _open(self, handle):
        _media_type, make_dataset = self.volume._factories[handle]
        return make_dataset()

    def test_the_map_grid_covers_new_jersey(self):
        ds = self._open("obs_map_1")

        self.assertLessEqual(float(ds.lat.min()), 39.0)
        self.assertGreaterEqual(float(ds.lat.max()), 41.3)
        self.assertLessEqual(float(ds.lon.min()), -75.5)
        self.assertGreaterEqual(float(ds.lon.max()), -74.0)

    def test_the_timeseries_grid_has_cells_inside_newark_for_every_day_of_january(self):
        ds = self._open("obs_ts_1")
        inside = ((ds.lat > 40.67) & (ds.lat < 40.79)).sum() * ((ds.lon > -74.25) & (ds.lon < -74.11)).sum()

        self.assertGreater(int(inside), 4)
        self.assertEqual(ds.sizes["time"], 31)


class ReportTests(unittest.TestCase):
    def test_writes_the_table_and_each_cases_answer_calls_checks_and_tokens(self):
        import json
        from datetime import datetime, timezone

        from benchmark.cases import BenchmarkCase
        from benchmark.report import write_report
        from benchmark.scoring import score

        case = BenchmarkCase(
            id="retrieval_tempo_no2", category="retrieval", prompt="Retrieve TEMPO NO2 over Houston.",
            handlers={}, checks=(Completed(), BoundaryCalled("retrieve_subset")),
        )
        outcome = Outcome(
            done=True, elapsed_seconds=3.0, answer_text="Here is the retrieval.",
            boundary_calls=[BoundaryCall("search_datasets", {"query": "TEMPO NO2"}, {"dataset_handle": "d1"})],
            tokens={("anthropic.claude-haiku-4-5", "satellite", "output"): 1000},
        )

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_report([score(case, outcome)], tmp, now=datetime(2026, 11, 13, 20, 0, tzinfo=timezone.utc))
            table = (folder / "table.txt").read_text(encoding="utf-8")
            records = json.loads((folder / "cases.json").read_text(encoding="utf-8"))

        self.assertEqual(folder.name, "20261113T200000Z")
        self.assertIn("retrieval_tempo_no2", table)
        record = records[0]
        self.assertFalse(record["passed"])
        self.assertEqual(record["answer_text"], "Here is the retrieval.")
        self.assertEqual(record["boundary_calls"][0]["tool"], "search_datasets")
        self.assertIn("retrieve_subset was never called", [check["reason"] for check in record["checks"]])
        self.assertEqual(record["tokens"][0]["tokens"], 1000)
        self.assertAlmostEqual(record["est_cost_dollars"], 0.005)


class TokenCountingTests(unittest.TestCase):
    def test_token_total_sums_one_kind_across_models_and_agents(self):
        outcome = Outcome(tokens={("a", "supervisor", "input"): 10, ("b", "satellite", "input"): 5, ("b", "satellite", "output"): 2})

        self.assertEqual(outcome.token_total("input"), 15)
        self.assertEqual(outcome.token_total("output"), 2)
        self.assertIsNone(outcome.token_total("cache_read"))

    def test_counting_tokens_adds_to_the_sink_and_still_reaches_the_app_metrics(self):
        from benchmark.adapters.current import counting_tokens

        forwarded = []
        sink = {}
        counted = counting_tokens(sink, lambda *args: forwarded.append(args))

        counted("haiku", "satellite", "input", 100)
        counted("haiku", "satellite", "input", 50)
        counted("haiku", "satellite", "output", 7)

        self.assertEqual(sink, {("haiku", "satellite", "input"): 150, ("haiku", "satellite", "output"): 7})
        self.assertEqual(len(forwarded), 3)
        self.assertEqual(forwarded[0], ("haiku", "satellite", "input", 100))

    def test_the_patched_recorder_is_the_one_the_app_calls(self):
        """The adapter patches record_llm_tokens on the metrics module, so the
        app must resolve it there at call time, not hold its own reference."""
        import inspect

        from benchmark.adapters.current import _TOKEN_RECORDER_TARGET
        from tta_backend.utils import llm_timing

        module, name = _TOKEN_RECORDER_TARGET.rsplit(".", 1)
        source = inspect.getsource(llm_timing._record_token_counters)

        self.assertIn(f"from {module} import {name}", source)


class CoreImportsTests(unittest.TestCase):
    def test_the_core_modules_do_not_import_the_application(self):
        code = (
            "import sys\n"
            "import benchmark.outcome, benchmark.checks, benchmark.cases, benchmark.scoring, benchmark.runner\n"
            "import benchmark.report, benchmark.fake_earthdata, benchmark.adapters.current\n"
            "import benchmark.adapters.workflow\n"
            "assert 'tta_backend' not in sys.modules, 'tta_backend was imported'\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


_PROVIDER_KEYS = {"google": "google_api_key", "groq": "groq_api_key", "portkey": "portkey_api_key"}
_PLACEHOLDER_KEYS = {"test", "your_google_key", "your_groq_key", "your_portkey_key"}


def _missing_model_keys() -> list[str]:
    """API keys missing for the providers the three agents are configured to use."""
    from tta_backend.infrastructure.settings import get_settings

    settings = get_settings()
    providers = {
        settings.supervisor_model_provider,
        settings.earthdata_agent_provider,
        settings.ground_agent_provider,
    }
    missing = []
    for provider in sorted(providers):
        attribute = _PROVIDER_KEYS.get(provider)
        key = getattr(settings, attribute, None) if attribute else None
        if not key or key in _PLACEHOLDER_KEYS:
            missing.append(attribute.upper() if attribute else f"a key for provider {provider!r}")
    return missing


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class BenchmarkSuiteTests(unittest.IsolatedAsyncioTestCase):
    """The whole benchmark against the current system. Opt-in (pytest -m eval)
    because it calls real models and spends real tokens. The models come from
    the environment's provider settings; skipped when a configured provider
    has no real key, even when selected."""

    @pytest.mark.eval
    @unittest.skipIf(_missing_model_keys(), f"model keys missing: {_missing_model_keys()}")
    async def test_current_system_meets_the_pass_threshold_within_budget(self):
        from benchmark.adapters.current import CurrentSystemAdapter
        from benchmark.fake_earthdata import HandleVolume
        from benchmark.runner import run_suite
        from benchmark.report import write_report
        from benchmark.scoring import CATEGORY_BUDGETS, PASS_THRESHOLD, TOTAL_CASES, format_results_table

        # The sub-agent trim ceiling is sized never to fire in a healthy
        # workflow; if it does here, a compaction gap regressed.
        with self.assertNoLogs("tta_backend.agents.subagent_trim", level="WARNING"):
            with tempfile.TemporaryDirectory() as tmpdir:
                results = await run_suite(CurrentSystemAdapter(), HandleVolume(tmpdir))

        print("\n" + format_results_table(results))
        # Saved before any assertion, so a failing run still leaves its evidence.
        folder = write_report(results, os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark-results"))
        print(f"results saved to {folder}")

        passed = sum(1 for result in results if result.passed)
        failures = [result.case.id for result in results if not result.passed]
        budget_breaches = [
            f"{result.case.id} ({result.outcome.elapsed_seconds:.1f}s > {CATEGORY_BUDGETS[result.case.category]:.0f}s budget)"
            for result in results
            if not result.within_budget
        ]

        self.assertEqual(len(results), TOTAL_CASES)
        self.assertFalse(budget_breaches, f"cases over their category latency budget: {budget_breaches}")
        self.assertGreaterEqual(passed, PASS_THRESHOLD, f"failed cases: {failures}")


if __name__ == "__main__":
    unittest.main()
