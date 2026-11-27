"""The explore adapter: a case's prompt run as an exploration task on the fake boundary, with a scripted model.

No network and no real model: the adapter is given scripted models in place of
the app's, so these pin how a ``Result`` becomes an ``Outcome``.
"""
import asyncio
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from benchmark.cases import BenchmarkCase, build_cases
from benchmark.pricing import estimate_cost

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow", "langgraph"]
_MISSING_DEPENDENCIES = any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)

MODEL_NAME = "scripted:explore"


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
    def test_every_case_names_a_known_bundle_and_places_to_resolve(self):
        from tta_backend.harness.specialists import BUNDLES

        for case in _cases_by_id().values():
            # An exploration cannot be answered into, so a case that spans turns names no bundle.
            if case.answers:
                self.assertIsNone(case.bundle, case.id)
                continue
            self.assertIn(case.bundle, BUNDLES, case.id)
            self.assertIsInstance(case.explore_places, tuple, case.id)

    def test_a_case_is_covered_when_it_names_a_bundle(self):
        from benchmark.adapters.explore import ExploreAdapter

        case = _cases_by_id()["discovery_no2_dataset"]

        self.assertTrue(ExploreAdapter.covers(case))
        self.assertFalse(ExploreAdapter.covers(_replace(case, bundle=None)))


def _replace(case: BenchmarkCase, **changes) -> BenchmarkCase:
    import dataclasses

    return dataclasses.replace(case, **changes)


def _scripted(*turns):
    from tta_backend.harness.models import ModelChoice, NamedModel
    from tta_backend.harness.testing import ScriptedChatModel, ScriptedModels

    model = ScriptedChatModel(script=list(turns))
    return model, ScriptedModels.every_role(ModelChoice(primary=NamedModel(model, MODEL_NAME)))


def _answer(summary: str, usage=(200, 40)):
    from tta_backend.harness.specialists import ANSWER_TOOL
    from tta_backend.harness.testing import tool_call

    return tool_call(ANSWER_TOOL, {"summary": summary, "findings": [], "followups": []}, usage=usage)


def _run(case: BenchmarkCase, models, limits=None):
    from benchmark.adapters.explore import ExploreAdapter
    from benchmark.fake_earthdata import HandleVolume

    async def run():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            return await ExploreAdapter(models=models, limits=limits).run(case, HandleVolume(tmp))

    return asyncio.run(run())


@unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
class ExploreAdapterRunTests(unittest.TestCase):
    def test_a_discovery_case_reaches_the_boundary_and_reports_what_the_model_used(self):
        from benchmark.checks import BoundaryCalled, Completed
        from tta_backend.harness.testing import tool_call

        case = _cases_by_id()["discovery_no2_dataset"]
        model, models = _scripted(
            tool_call("sources__satellite__search", {"query": "NO2 column"}, usage=(100, 20)),
            _answer("Two NO2 products cover New Jersey."),
        )

        outcome = _run(case, models)

        self.assertTrue(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.mode, "Exploratory")
        self.assertEqual(outcome.tool_calls, ["sources.areas.resolve", "sources.satellite.search"])
        self.assertEqual([call.tool for call in outcome.boundary_calls], ["search_datasets"])
        self.assertIn("Two NO2 products cover New Jersey.", outcome.answer_text)
        self.assertTrue(outcome.answer_text.startswith("Exploratory\nTwo NO2"))
        self.assertTrue(Completed().evaluate(outcome).passed)
        self.assertTrue(BoundaryCalled("search_datasets").evaluate(outcome).passed)
        self.assertEqual(outcome.tokens, {
            (MODEL_NAME, "explore", "input"): 300,
            (MODEL_NAME, "explore", "output"): 60,
        })
        self.assertEqual(outcome.token_total("input"), 300)
        self.assertEqual(estimate_cost(outcome).unpriced, (MODEL_NAME,))
        # The case's prompt is the objective, word for word.
        self.assertIn(case.prompt, str(model.received[0]))

    def test_an_analysis_case_resolves_its_place_and_a_model_that_answers_at_once_retrieves_nothing(self):
        case = _cases_by_id()["retrieval_tempo_no2"]
        _, models = _scripted(_answer("Nothing was retrieved.", usage=(50, 5)))

        outcome = _run(case, models)

        self.assertTrue(outcome.done)
        self.assertEqual(outcome.tool_calls, ["sources.areas.resolve"])
        self.assertEqual(outcome.boundary_calls, [])
        self.assertEqual(outcome.handles, [])
        self.assertEqual(outcome.token_total("input"), 50)

    def test_a_refusal_puts_the_question_in_the_answer_so_the_case_can_check_it(self):
        from benchmark.scoring import score

        case = _cases_by_id()["region_refusal_unresolvable_member"]
        _, models = _scripted(_answer("unused"))

        outcome = _run(case, models)

        self.assertTrue(outcome.done)
        self.assertIn("could not be found", outcome.answer_text)
        self.assertIn("Wakanda", outcome.answer_text)
        self.assertEqual(outcome.boundary_calls, [])
        self.assertTrue(score(case, outcome).passed)

    def test_a_question_ends_an_unexpecting_case_unfinished_and_an_expecting_one_finished(self):
        from benchmark.checks import Completed

        case = _cases_by_id()["region_refusal_unresolvable_member"]
        self.assertTrue(case.expects_question)
        expecting = _run(case, _scripted(_answer("unused"))[1])
        unexpecting = _run(_replace(case, expects_question=False), _scripted(_answer("unused"))[1])

        self.assertTrue(expecting.done)
        self.assertIsNone(expecting.unfinished)
        self.assertFalse(unexpecting.done)
        self.assertIn("stopped to ask: 'Wakanda'", unexpecting.unfinished)
        self.assertIn("stopped to ask", Completed().evaluate(unexpecting).reason)

    def test_only_the_cases_whose_point_is_a_question_expect_one(self):
        self.assertEqual(
            sorted(case.id for case in _cases_by_id().values() if case.expects_question),
            ["region_refusal_buffer_too_large", "region_refusal_unresolvable_member"],
        )

    def test_a_run_that_spends_its_budget_does_not_complete_and_says_why(self):
        from benchmark.checks import Completed
        from benchmark.scoring import score
        from tta_backend.harness.specialists import ExploreLimits
        from tta_backend.harness.testing import tool_call

        case = _cases_by_id()["discovery_no2_dataset"]
        _, models = _scripted(
            tool_call("sources__satellite__search", {"query": "NO2"}),
            tool_call("sources__satellite__search", {"query": "NO2 again"}),
            _answer("never reached"),
        )

        outcome = _run(case, models, limits=ExploreLimits(model_calls=1))

        self.assertFalse(outcome.done)
        self.assertIsNone(outcome.error)
        self.assertIn("ran out of its budget", outcome.unfinished)
        result = Completed().evaluate(outcome)
        self.assertFalse(result.passed)
        self.assertIn("ran out of its budget", result.reason)
        scored = score(case, outcome)
        self.assertFalse(scored.passed)
        self.assertIn("ran out of its budget", scored.first_failure.reason)

    def test_a_model_that_cannot_answer_fails_the_run_and_the_outcome_says_so(self):
        from tta_backend.harness.models import NO_BACKOFF
        from tta_backend.harness.testing import ScriptedModels

        case = _cases_by_id()["discovery_no2_dataset"]

        with patch("tta_backend.harness.models.DEFAULT_BACKOFF", NO_BACKOFF):
            outcome = _run(case, ScriptedModels())

        self.assertFalse(outcome.done)
        self.assertTrue(outcome.error)
        self.assertEqual(outcome.tokens, {})
        self.assertIsNone(estimate_cost(outcome))

    def test_the_adapter_refuses_a_case_without_a_bundle(self):
        from benchmark.adapters.explore import ExploreAdapter, NotCovered

        case = _replace(_cases_by_id()["discovery_no2_dataset"], bundle=None)

        with self.assertRaises(NotCovered):
            _run(case, None)
        self.assertFalse(ExploreAdapter.covers(case))


class TokensFromTraceTests(unittest.TestCase):
    def _entry(self, kind, result):
        from types import SimpleNamespace

        return SimpleNamespace(kind=kind, result=result)

    def test_only_explore_entries_count_and_zero_counts_are_left_out(self):
        from benchmark.adapters.explore import tokens_from_trace

        trace = [
            self._entry("use", {"input_tokens": 999}),
            self._entry("explore", {"model": "portkey:claude-sonnet-5", "input_tokens": 10, "output_tokens": 0}),
            self._entry("explore", {"model": "portkey:claude-sonnet-5", "input_tokens": 5, "output_tokens": 2}),
        ]

        self.assertEqual(tokens_from_trace(trace), {
            ("portkey:claude-sonnet-5", "explore", "input"): 15,
            ("portkey:claude-sonnet-5", "explore", "output"): 2,
        })


    def test_cache_tokens_are_their_own_kinds_and_absent_means_zero(self):
        from benchmark.adapters.explore import tokens_from_trace

        model = "portkey:claude-sonnet-5"
        trace = [
            self._entry("explore", {
                "model": model, "input_tokens": 1000, "output_tokens": 50,
                "cache_read_tokens": 600, "cache_creation_tokens": 200,
            }),
            self._entry("explore", {"model": model, "input_tokens": 10, "output_tokens": 5}),
        ]

        self.assertEqual(tokens_from_trace(trace), {
            (model, "explore", "input"): 1010,
            (model, "explore", "output"): 55,
            (model, "explore", "cache_read"): 600,
            (model, "explore", "cache_creation"): 200,
        })


class CachePricingTests(unittest.TestCase):
    SONNET = "portkey:claude-sonnet-5-5"

    def test_cache_reads_and_writes_are_priced_at_their_own_rates_within_input(self):
        from benchmark.outcome import Outcome

        outcome = Outcome(tokens={
            (self.SONNET, "explore", "input"): 1_000_000,
            (self.SONNET, "explore", "cache_read"): 600_000,
            (self.SONNET, "explore", "cache_creation"): 100_000,
            (self.SONNET, "explore", "output"): 100_000,
        })

        # 300k plain input at $2, 600k read at $0.10, 100k written at $2.50, 100k out at $10.
        self.assertAlmostEqual(estimate_cost(outcome).dollars, 0.60 + 0.06 + 0.25 + 1.00)

    def test_cached_tokens_are_never_counted_twice_when_they_exceed_the_input(self):
        from benchmark.outcome import Outcome

        outcome = Outcome(tokens={
            (self.SONNET, "explore", "input"): 100,
            (self.SONNET, "explore", "cache_read"): 100,
            (self.SONNET, "explore", "cache_creation"): 100,
        })

        self.assertAlmostEqual(estimate_cost(outcome).dollars, (100 * 0.10 + 100 * 2.50) / 1_000_000)

    def test_cached_tokens_beyond_the_input_are_flagged_on_the_estimate_and_the_table(self):
        from benchmark.outcome import Outcome
        from benchmark.scoring import CaseResult, format_explore_table

        outcome = Outcome(tokens={
            (self.SONNET, "explore", "input"): 100,
            (self.SONNET, "explore", "cache_read"): 80,
            (self.SONNET, "explore", "cache_creation"): 80,
        })
        consistent = Outcome(tokens={(self.SONNET, "explore", "input"): 100, (self.SONNET, "explore", "cache_read"): 80})

        self.assertEqual(estimate_cost(outcome).inconsistent_models, (self.SONNET,))
        self.assertEqual(estimate_cost(consistent).inconsistent_models, ())
        case = _cases_by_id()["discovery_no2_dataset"]
        table = format_explore_table([CaseResult(case, outcome)], [], 0.01)
        self.assertIn(f"cached tokens exceed input for: {self.SONNET}", table)

    def test_a_write_costs_one_and_a_quarter_times_the_input_price(self):
        from benchmark.pricing import PRICES_PER_MILLION

        for name, price in PRICES_PER_MILLION:
            self.assertAlmostEqual(price.cache_creation, price.input * 1.25, msg=name)


from benchmark.outcome import Outcome  # noqa: E402


class SatelliteStageCheckTests(unittest.TestCase):
    def _check(self):
        from benchmark.checks import StageSequenceAny

        [stage_check] = [c for c in _cases_by_id()["e2e_satellite_plot"].checks if isinstance(c, StageSequenceAny)]
        return stage_check

    def _with_stages(self, stages, produced=True):
        from benchmark.outcome import Outcome

        if not produced:
            return Outcome(done=True, stages=stages)
        return Outcome(done=True, stages=stages, handles=["obs_1"], charts=[{"type": "map"}])

    def test_the_current_systems_narration_passes(self):
        from benchmark.cases import SATELLITE_STAGES

        self.assertTrue(self._check().evaluate(self._with_stages(list(SATELLITE_STAGES))).passed)

    def test_the_harness_narration_passes_with_extra_and_repeated_stages(self):
        stages = ["sources.areas.resolve", "sources.satellite.get_data", "sources.satellite.get_data", "render.map"]

        self.assertTrue(self._check().evaluate(self._with_stages(stages)).passed)

    def test_the_harness_narration_needs_a_data_handle_and_a_chart_behind_it(self):
        stages = ["sources.areas.resolve", "sources.satellite.get_data", "render.map"]

        no_chart = self._check().evaluate(Outcome(done=True, stages=stages, handles=["obs_1"]))
        no_handle = self._check().evaluate(Outcome(done=True, stages=stages, charts=[{"type": "map"}]))
        neither = self._check().evaluate(self._with_stages(stages, produced=False))

        for result in (no_chart, no_handle, neither):
            self.assertFalse(result.passed)
            self.assertIn("no data handle and chart", result.reason)

    def test_the_current_systems_narration_needs_no_handle(self):
        from benchmark.cases import SATELLITE_STAGES

        self.assertTrue(self._check().evaluate(self._with_stages(list(SATELLITE_STAGES), produced=False)).passed)

    def test_the_harness_stages_in_the_wrong_order_fail(self):
        stages = ["render.map", "sources.satellite.get_data", "sources.areas.resolve"]

        result = self._check().evaluate(self._with_stages(stages))

        self.assertFalse(result.passed)
        self.assertIn("render.map", result.reason)

    def test_a_partial_or_empty_narration_fails_so_the_check_is_not_weakened(self):
        check = self._check()

        self.assertFalse(check.evaluate(self._with_stages(["search", "aoi", "render"])).passed)
        self.assertFalse(check.evaluate(self._with_stages(["sources.areas.resolve", "render.map"])).passed)
        self.assertFalse(check.evaluate(self._with_stages([])).passed)


class OfflineGeocoderTests(unittest.TestCase):
    def test_both_cities_of_the_two_city_case_resolve_in_nominatims_shape(self):
        from benchmark.adapters.harness_boundary import _geocode

        for place in _cases_by_id()["comparison_setup_two_cities"].explore_places:
            hit = _geocode(place)
            self.assertIsNotNone(hit, place)
            self.assertEqual(sorted(hit), ["bbox", "display_name", "latitude", "longitude", "polygon"])
            south, north, west, east = hit["bbox"]
            self.assertTrue(south < hit["latitude"] < north and west < hit["longitude"] < east, place)

    def test_los_angeles_lies_where_the_area_check_looks_for_it(self):
        from benchmark.adapters.harness_boundary import _geocode
        from benchmark.checks import AreaRequested
        from benchmark.outcome import BoundaryCall, Outcome

        south, north, west, east = _geocode("Los Angeles CA")["bbox"]
        box = f"{west},{south},{east},{north}"
        outcome = Outcome(boundary_calls=[BoundaryCall("define_area_of_interest", {"location": box}, {})])

        self.assertTrue(AreaRequested("Los Angeles", (34.0522, -118.2437)).evaluate(outcome).passed)

    @unittest.skipIf(_MISSING_DEPENDENCIES, "benchmark test dependencies are not installed")
    def test_the_two_city_case_resolves_both_places(self):
        _, models = _scripted(_answer("Both cities were resolved."))

        outcome = _run(_cases_by_id()["comparison_setup_two_cities"], models)

        self.assertEqual(outcome.tool_calls, ["sources.areas.resolve", "sources.areas.resolve"])
        self.assertIn("Newark, New Jersey", outcome.answer_text)
        self.assertIn("Los Angeles, California", outcome.answer_text)


class ExploreReportTests(unittest.TestCase):
    def test_the_table_says_it_is_exploratory_and_states_coverage_and_cost(self):
        from benchmark.scoring import format_explore_table

        table = format_explore_table([], ["a", "b"], 1.25)

        self.assertIn("exploratory", table)
        self.assertIn("not covered (2): a, b", table)
        self.assertIn("estimated cost: $1.2500", table)
        self.assertIn("not available", format_explore_table([], [], None))

    def test_a_saved_run_holds_the_table_the_cases_and_the_cost(self):
        import json
        from pathlib import Path

        from benchmark.report import write_explore_report
        from benchmark.runner import WorkflowSuiteRun

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_explore_report(WorkflowSuiteRun(results=[], uncovered=["x"]), Path(tmp) / "explore")

            self.assertEqual(sorted(path.name for path in folder.iterdir()),
                             ["cases.json", "cost.json", "table.txt", "uncovered.json"])
            self.assertEqual(json.loads((folder / "cost.json").read_text())["covered"], 0)
