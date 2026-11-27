"""sources.satellite on fakes: product choice, coverage by date role, size gate, waiting, recording."""
import datetime
import json
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tests.test_harness_grids import LATS, LONS, NJ, granule
from tta_backend.datasets.qa_flags import QA_VERIFIED
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities import CapabilityRegistry, CapabilityResult, JobStillRunning, begin, complete, invoke
from tta_backend.harness.capabilities.failures import JobFailed
from tta_backend.harness.capabilities.sources import areas, satellite
from tta_backend.harness.handles import Descriptor, HandleRecord
from tta_backend.harness.run_context import Limits, RunEnv
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.testing import (
    FakeEarthdata,
    FakePlaces,
    FakeToolError,
    FixedClock,
    StaticCredentials,
    box_region,
    make_run_context,
    make_run_env,
)
from tta_backend.harness.trace import MAX_REASON_CHARS, FailureCategory, StepFailure

TODAY = datetime.datetime(2024, 7, 1, 12, tzinfo=datetime.UTC)
SECRET = "edl-secret-token-value"
GET = satellite.get_data
NO2 = "no2_tropospheric_column"
TEMPO_COLLECTION = load_registry()["TEMPO_NO2"].collection_id
FAR = box_region("Lisbon", -9.3, 38.6, -9.0, 38.8, region_type="polygon")


def _concept_id(hit: Any) -> Any:
    if not isinstance(hit, dict):
        return None
    return hit.get("concept_id") or (hit.get("summary") or {}).get("concept_id")


class FakeService:
    """A scripted earthdata service: coverage per day, sizes, jobs and their statuses."""

    def __init__(self) -> None:
        self.days_with_data: set[str] | None = None  # None: every day has data
        self.estimated_bytes: int | None = 1000
        # Per day a size request starts on, overriding estimated_bytes; None is a day the provider cannot size.
        self.sizes_by_day: dict[str, int | None] = {}
        self.granules_per_day = 1
        self.statuses: list[dict[str, Any]] = []  # consumed per poll before a job is ready
        self.files: dict[str, xr.Dataset | pd.DataFrame] = {}
        self.jobs: dict[str, str] = {}
        self.make_file: Any = lambda day: granule(day=day)
        self.series: pd.DataFrame | None = None
        self.cancelled: list[str] = []
        self.search_hits: Any = None  # None: one hit for the queried collection
        self.searches: list[dict[str, Any]] = []
        self.honour_filters = True

    def _days(self, time_range: str) -> list[str]:
        start, end = (part[:10] for part in time_range.split("/"))
        days = pd.date_range(start, end, freq="D")
        return [d.date().isoformat() for d in days]

    def coverage(self, args: dict[str, Any]) -> dict[str, Any]:
        days = self._days(args["time_range"])
        count = len(days) if self.days_with_data is None else sum(d in self.days_with_data for d in days)
        return {"granule_count": count * self.granules_per_day}

    def estimate(self, args: dict[str, Any]) -> dict[str, Any]:
        day = args["time_range"][:10]
        size = self.sizes_by_day[day] if day in self.sizes_by_day else self.estimated_bytes
        # The service's own shape; no granule size reported reads as 0 MB over the granules.
        return {"total_granules": 1, "total_size_mb": 0.0 if size is None else size / 1024 ** 2}

    def submit(self, args: dict[str, Any]) -> dict[str, Any]:
        index = len(self.jobs) + 1
        job, obs = f"job_{index}", f"obs_{index}"
        day = args["time_range"][:10]
        self.files[obs] = self.make_file(day)
        self.jobs[job] = obs
        return {"job_handle": job}

    def submit_series(self, args: dict[str, Any]) -> dict[str, Any]:
        job, obs = "job_series", "obs_series"
        self.files[obs] = self.series if self.series is not None else pd.DataFrame()
        self.jobs[job] = obs
        return {"job_handle": job}

    def status(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.statuses:
            return {"job_handle": args["job_handle"], **self.statuses.pop(0)}
        return {"job_handle": args["job_handle"], "status": "ready", "obs_handle": self.jobs[args["job_handle"]]}

    def search(self, args: dict[str, Any]) -> dict[str, Any]:
        self.searches.append(args)
        # A concept_id filter keeps only that collection's hits, as the service's search does.
        concept_id = (args.get("filters") or {}).get("concept_id") if self.honour_filters else None
        if self.search_hits is not None:
            hits = self.search_hits
            if concept_id is not None and isinstance(hits, list):
                hits = [h for h in hits if _concept_id(h) == concept_id]
            return {"datasets": hits}
        # The service's own shape: each hit's fields sit under its summary.
        return {"datasets": [{"handle": "dataset_1", "summary": {
            "concept_id": concept_id or args["query"], "short_name": "TEMPO_NO2_L3", "version": "V04",
            "entry_title": "TEMPO NO2",
        }}], "count": 1}

    def cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        self.cancelled.append(args["job_handle"])
        return {"job_handle": args["job_handle"], "status": "cancelled"}

    def earthdata(self) -> FakeEarthdata:
        return FakeEarthdata({
            "search_datasets": self.search,
            "cancel_retrieval": self.cancel,
            "define_area_of_interest": lambda args: {"handle": "aoi_1", "location": args["location"]},
            "check_coverage": self.coverage,
            "estimate_retrieval_size": self.estimate,
            "retrieve_subset": self.submit,
            "retrieve_timeseries": self.submit_series,
            "get_retrieval_status": self.status,
        })


class SatelliteCase(unittest.IsolatedAsyncioTestCase):
    stores: Stores | None = None

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service = FakeService()
        self.earthdata = self.service.earthdata()
        self.env = await self.make_env()
        self.area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle

    async def make_env(self, limits: Limits | None = None, stores: Stores | None = None) -> RunEnv:
        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.service.files[record.id]

        context = make_run_context(
            credentials=StaticCredentials({"earthdata": SECRET}), clock=FixedClock(TODAY),
            stores=stores or self.stores or memory_stores(),
        )
        return await make_run_env(
            self._tmp.name, context=context, earthdata=self.earthdata, open_retrieval=open_retrieval,
            places=FakePlaces({"new jersey": NJ, "lisbon": FAR}), limits=limits,
        )

    def params(self, **over: Any) -> dict[str, Any]:
        return {"quantity": NO2, "area": self.area, "dates": "2024-06-01", "date_role": "subject", **over}

    def tools(self) -> list[str]:
        return [tool for tool, _ in self.earthdata.calls]

    async def entries(self) -> list[Any]:
        """This run's entries after the area was resolved."""
        entries = await self.env.context.storage.traces.entries("user-a", self.env.run_id)
        return [e for e in entries if e.capability != "sources.areas.resolve"]

    async def failure(self, params: dict[str, Any], capability: Any = GET) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, capability, params)
        return caught.exception


class GetDataTests(SatelliteCase):
    async def test_a_subject_date_is_retrieved_recorded_and_quality_measured(self):
        result = await invoke(self.env, GET, self.params(), step="event")

        self.assertEqual((result.status, result.kind, result.handle, result.product), ("done", "grid", "obs_1", "TEMPO_NO2"))
        self.assertEqual(result.unit, "molecules/cm^2")
        self.assertEqual(result.disclosures[0], (
            "Product: TEMPO_NO2 (TEMPO_NO2_L3), chosen because it is the registry's preferred product "
            "for this quantity that covers the area and dates."
        ))
        assert result.quality is not None
        self.assertEqual(result.quality["valid_fraction"], 1.0)
        self.assertEqual(result.quality["qa_pass_rate"], 1.0)
        self.assertEqual(result.quality["qa_status"], QA_VERIFIED)
        self.assertEqual(result.read("rule"), "highest_rank")
        self.assertEqual(result.method, (
            "TEMPO_NO2 was retrieved over New Jersey for 2024-06-01, chosen by rule highest_rank, and masked "
            f"to the area with the product's quality rules ({QA_VERIFIED})."
        ))

        record = await self.env.resolver.get("user-a", "obs_1", "grid")
        self.assertEqual(record.origin, "retrieval")
        descriptor = record.descriptor
        self.assertEqual((descriptor.product, descriptor.quantity, descriptor.unit), ("TEMPO_NO2", NO2, "molecules/cm^2"))
        self.assertEqual(descriptor.region, {
            "name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4], "area": self.area, "wording": "New Jersey",
        })
        self.assertEqual((descriptor.period.start, descriptor.period.end), ("2024-06-01", "2024-06-01"))
        self.assertEqual(descriptor.masking, {
            "variable": "vertical_column_troposphere", "quality_flag": "main_data_quality_flag", "geometry": "area",
        })

        submit = next(args for tool, args in self.earthdata.calls if tool == "retrieve_subset")
        self.assertEqual(submit["variables"], ["vertical_column_troposphere", "main_data_quality_flag"])
        self.assertEqual(submit["time_range"], "2024-06-01T00:00:00/2024-06-01T23:59:59")
        aoi = next(args for tool, args in self.earthdata.calls if tool == "define_area_of_interest")
        self.assertEqual(aoi["location"], "-75.600000,38.900000,-73.900000,41.400000")
        key = self.earthdata.keys[self.tools().index("retrieve_subset")]
        self.assertRegex(key, r"^tta-[0-9a-f]{32}$")

        waiting, done = await self.entries()
        self.assertEqual((waiting.status, done.status), ("waiting", "done"))
        self.assertEqual(waiting.result["job"]["jobs"], [{"job": "job_1", "day": "2024-06-01"}])
        self.assertEqual(done.params["_job"]["product"], "TEMPO_NO2")

    async def test_the_rule_and_a_move_to_a_later_approved_product_are_disclosed(self):
        # OMI is published two days late, so 2024-06-30 is not out yet on 2024-07-01.
        result = await invoke(self.env, GET, self.params(dates="2024-06-30", approved=["OMI_NO2", "TEMPO_NO2"]))
        self.assertEqual(result.product, "TEMPO_NO2")
        self.assertEqual(result.read("rule"), "only_candidate")
        self.assertTrue(result.disclosures[1].startswith(
            "Moved from OMI_NO2 to TEMPO_NO2: OMI_NO2 was ruled out (not_published_yet: published through 2024-06-29"
        ))

    async def test_a_named_product_is_used_and_supplies_the_quantity(self):
        result = await invoke(self.env, GET, {
            "named": "TEMPO_NO2_NRT", "area": self.area, "dates": "2025-10-01", "date_role": "subject",
        })
        # The NRT record starts 2025-09-17, after this run's today, so nothing fits and nothing is substituted.
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(self.tools(), [])

        result = await invoke(self.env, GET, {"named": "TEMPO", "quantity": NO2, "area": self.area,
                                              "dates": "2024-06-01", "date_role": "subject"})
        self.assertEqual(result.product, "TEMPO_NO2")

    async def test_a_pinned_product_is_kept(self):
        result = await invoke(self.env, GET, self.params(product="TEMPO_NO2", approved=["TEMPO_NO2_NRT", "TEMPO_NO2"]))
        self.assertEqual((result.product, result.read("rule")), ("TEMPO_NO2", "pinned"))
        self.assertIn("this run already uses it", result.disclosures[0])

    async def test_a_name_that_matches_nothing_is_bad_input(self):
        failure = await self.failure(self.params(named="NotAnInstrument"))
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(self.tools(), [])

    async def test_parameters_are_checked(self):
        for over in (
            {"quantity": None},
            {"date_role": "range"},
            {"date_role": "subject", "dates": ["2024-06-01", "2024-06-02"]},
            {"dates": {"start": "2024-06-05", "end": "2024-06-01"}, "date_role": "range"},
            {"dates": [f"2024-05-{d:02d}" for d in range(1, 32)] + ["2024-06-01"], "date_role": "comparison"},
        ):
            with self.subTest(over):
                failure = await self.failure(self.params(**over))
                self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class SizeGateTests(SatelliteCase):
    async def test_between_the_caps_asks_and_submits_nothing(self):
        self.service.estimated_bytes = 100_000_000
        result = await invoke(self.env, GET, self.params())
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertEqual((result.question.kind, result.question.options), ("pick_one", ("confirm", "narrow")))
        self.assertIn("about 100 MB, above the 50 MB soft limit", result.question.prompt)
        self.assertEqual(result.read("estimated_bytes"), 100_000_000)
        self.assertNotIn("retrieve_subset", self.tools())
        [entry] = await self.entries()
        self.assertEqual(entry.status, "waiting")

    async def test_begin_settles_a_size_question_without_a_job(self):
        self.service.estimated_bytes = 100_000_000
        outcome = await begin(self.env, GET, self.params(), step="event")
        self.assertIsInstance(outcome, CapabilityResult)
        self.assertEqual(outcome.status, "needs_input")

    async def test_a_confirmed_request_between_the_caps_is_submitted(self):
        self.service.estimated_bytes = 100_000_000
        result = await invoke(self.env, GET, self.params(confirmed=True))
        self.assertEqual(result.status, "done")
        self.assertIn("retrieve_subset", self.tools())

    async def test_above_the_hard_cap_is_refused_even_when_confirmed(self):
        self.service.estimated_bytes = 300_000_000
        failure = await self.failure(self.params(confirmed=True))
        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertIn("about 300 MB, above the 200 MB limit", failure.reason)
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_an_absent_estimate_over_many_granules_is_treated_as_between_the_caps(self):
        self.service.estimated_bytes = None
        # The fake counts one granule a day, so a month is past the unsized limit.
        month = {"start": "2024-06-01", "end": "2024-06-30"}
        result = await invoke(self.env, GET, self.params(dates=month, date_role="range"))
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertIn("of unknown size", result.question.prompt)
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_an_absent_estimate_over_few_granules_goes_ahead_and_says_so(self):
        self.service.estimated_bytes = None
        result = await invoke(self.env, GET, self.params())
        self.assertEqual(result.status, "done")
        self.assertIn("retrieve_subset", self.tools())
        self.assertIn(
            "The data provider reports no size for TEMPO_NO2's files, so this retrieval of 1 granule was not "
            "checked against the size limits.",
            result.disclosures,
        )

    async def test_a_sized_part_above_the_hard_cap_is_refused_even_when_another_part_is_unsized(self):
        self.service.sizes_by_day = {"2024-06-01": 300_000_000, "2024-06-02": None}
        failure = await self.failure(self.params(dates=["2024-06-01", "2024-06-02"], date_role="comparison", confirmed=True))
        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertIn("at least about 300 MB, above the 200 MB limit", failure.reason)
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_the_unsized_allowance_counts_only_the_unsized_parts_granules(self):
        # 40 granules in all, but only the 20 of the unsized day go unchecked: within the allowance.
        self.service.granules_per_day = 20
        self.service.sizes_by_day = {"2024-06-01": 1000, "2024-06-02": None}
        result = await invoke(self.env, GET, self.params(dates=["2024-06-01", "2024-06-02"], date_role="comparison"))
        self.assertEqual(result.status, "done")
        self.assertIn(
            "The data provider reports no size for TEMPO_NO2's files, so this retrieval of 20 granules was not "
            "checked against the size limits.",
            result.disclosures,
        )

    async def test_an_unsized_part_past_the_allowance_asks_and_names_the_sized_part(self):
        self.service.granules_per_day = 30
        self.service.sizes_by_day = {"2024-06-01": 1000, "2024-06-02": None}
        result = await invoke(self.env, GET, self.params(dates=["2024-06-01", "2024-06-02"], date_role="comparison"))
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertIn("plus a part of unknown size", result.question.prompt)
        self.assertNotIn("soft limit", result.question.prompt)
        self.assertNotIn("retrieve_subset", self.tools())

    def test_the_size_gate_lets_only_small_unsized_retrievals_through(self):
        from tta_backend.domain.earthdata.retrieval import MAX_UNSIZED_GRANULES, size_gate

        gate = {"soft": 100, "hard": 200, "confirmed": False, "max_unsized_granules": MAX_UNSIZED_GRANULES}
        self.assertEqual(size_gate(None, granules=MAX_UNSIZED_GRANULES, **gate), "proceed")
        self.assertEqual(size_gate(None, granules=MAX_UNSIZED_GRANULES + 1, **gate), "confirm")
        self.assertEqual(size_gate(None, granules=None, **gate), "confirm")
        self.assertEqual(size_gate(None, granules=0, **gate), "confirm")
        self.assertEqual(size_gate(150, granules=1, **gate), "confirm")


class NoDataTests(SatelliteCase):
    async def test_outside_the_record_names_the_situation_and_calls_nothing(self):
        result = await invoke(self.env, GET, self.params(dates="2023-01-15", approved=["TEMPO_NO2", "TEMPO_NO2_NRT"]))
        self.assertEqual((result.status, result.read("situation")), ("insufficient_evidence", "outside_record"))
        self.assertEqual(result.reason, (
            "No data: 2023-01-15 is outside the record of every product that could serve "
            f"{NO2} over New Jersey."
        ))
        self.assertEqual(self.tools(), [])
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.NO_DATA))

    async def test_not_published_yet_names_the_situation(self):
        result = await invoke(self.env, GET, self.params(dates="2024-07-03", approved=["TEMPO_NO2"]))
        self.assertEqual(result.read("situation"), "not_published_yet")
        self.assertTrue(result.reason.startswith("No data yet:"))
        self.assertEqual(self.tools(), [])

    async def test_a_gap_on_a_subject_date_is_never_substituted_and_names_nearby_dates(self):
        self.service.days_with_data = {"2024-05-30", "2024-06-02", "2024-06-04"}
        result = await invoke(self.env, GET, self.params())
        self.assertEqual((result.status, result.read("situation")), ("insufficient_evidence", "gap"))
        self.assertEqual(result.reason, (
            "No data: TEMPO_NO2 has no observations over New Jersey for 2024-06-01; this is a gap in its record."
        ))
        self.assertEqual(result.read("nearest_dates"), ["2024-06-02", "2024-05-30"])
        self.assertEqual(result.product, "TEMPO_NO2")
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2 (TEMPO_NO2_L3), chosen because"))
        self.assertNotIn("retrieve_subset", self.tools())
        self.assertNotIn("dropped_dates", result.values)

    async def test_too_little_valid_data_after_masking(self):
        self.service.make_file = lambda day: granule(day=day, flags=3)
        result = await invoke(self.env, GET, self.params())
        self.assertEqual((result.status, result.read("situation")), ("insufficient_evidence", "too_little_valid"))
        self.assertEqual(result.reason, (
            "Too little valid data: after quality masking, TEMPO_NO2 kept 0.0% of New Jersey for 2024-06-01."
        ))
        self.assertEqual((result.handle, result.kind), ("obs_1", "grid"))
        assert result.quality is not None
        self.assertEqual(result.quality["qa_pass_rate"], 0.0)

    async def test_a_subject_date_with_too_little_valid_data_offers_the_nearest_dates_with_data(self):
        self.service.days_with_data = {"2024-06-01", "2024-06-03"}
        self.service.make_file = lambda day: granule(day=day, flags=3)
        result = await invoke(self.env, GET, self.params())
        self.assertEqual(result.read("situation"), "too_little_valid")
        self.assertEqual(result.read("nearest_dates"), ["2024-06-03"])
        self.assertIn(
            "The nearest dates with data over the area are 2024-06-03; whether enough of it is valid there "
            "is known only once retrieved.",
            result.disclosures,
        )
        self.assertEqual(self.tools().count("retrieve_subset"), 1)

    async def test_a_coverage_answer_without_a_granule_count_is_a_contract_error(self):
        self.earthdata.tools["check_coverage"] = lambda args: {"has_data": True}
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(failure.reason, (
            "sources.satellite.get_data: the data service returned no granule count for the coverage check."
        ))
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_a_minimum_valid_fraction_applies(self):
        flags = np.zeros((1, LATS.size, LONS.size), dtype="int16")
        flags[0, 2:, :] = 1
        self.service.make_file = lambda day: granule(day=day, flags=flags)
        result = await invoke(self.env, GET, self.params(min_valid_fraction=0.5))
        self.assertEqual(result.read("situation"), "too_little_valid")

    async def test_an_area_outside_every_product_is_insufficient_with_the_reasons(self):
        lisbon = (await invoke(self.env, areas.resolve, {"place": "Lisbon"})).handle
        result = await invoke(self.env, GET, self.params(area=lisbon, approved=["TEMPO_NO2"]))
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("TEMPO_NO2: the area is outside its coverage", result.reason)
        self.assertNotIn("area_outside_coverage", result.reason)
        self.assertNotIn("situation", result.values)


class DateRoleTests(SatelliteCase):
    async def test_comparison_days_without_data_are_dropped_and_reported(self):
        self.service.days_with_data = {"2024-06-03", "2024-06-10"}
        days = ["2024-06-03", "2024-06-10", "2024-06-17"]
        result = await invoke(self.env, GET, self.params(dates=days, date_role="comparison"))

        self.assertEqual(result.status, "done")
        self.assertTrue(result.handle.startswith("grid_"))
        self.assertEqual(result.read("dropped_dates"), ["2024-06-17"])
        self.assertIn("1 of the requested days had no data and were left out: 2024-06-17.", result.disclosures)
        self.assertEqual(self.tools().count("retrieve_subset"), 2)
        assert result.quality is not None
        self.assertEqual((result.quality["days_requested"], result.quality["days_with_data"]), (2, 2))

        record = await self.env.resolver.get("user-a", result.handle, "grid")
        self.assertEqual(record.parents, (self.area, "obs_1", "obs_2"))
        self.assertEqual(record.descriptor.masking["parts"], 2)
        self.assertEqual(record.descriptor.masking["qa_status"], QA_VERIFIED)
        stored = await self.env.resolver.open("user-a", result.handle, "grid")
        self.assertEqual(stored.sizes["time"], 2)

    async def test_comparison_days_all_without_data_is_a_gap(self):
        self.service.days_with_data = set()
        result = await invoke(self.env, GET, self.params(dates=["2024-06-03", "2024-06-10"], date_role="comparison"))
        self.assertEqual(result.read("situation"), "gap")
        self.assertEqual(result.read("dropped_dates"), ["2024-06-03", "2024-06-10"])
        self.assertNotIn("nearest_dates", result.values)

    async def test_comparison_days_with_nothing_valid_after_masking_are_dropped_and_reported(self):
        self.service.days_with_data = {"2024-06-03", "2024-06-10", "2024-06-17"}
        self.service.make_file = lambda day: granule(day=day, flags=3 if day == "2024-06-10" else 0)
        days = ["2024-06-03", "2024-06-10", "2024-06-17", "2024-06-24"]
        result = await invoke(self.env, GET, self.params(dates=days, date_role="comparison"))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.read("dropped_dates"), ["2024-06-10", "2024-06-24"])
        self.assertEqual(result.read("dropped_after_masking"), ["2024-06-10"])
        self.assertEqual(result.read("days_without_valid_data"), [])
        self.assertIn(
            "1 of the requested days had no valid data after quality masking and were left out: 2024-06-10.",
            result.disclosures,
        )
        assert result.quality is not None
        self.assertEqual((result.quality["days_requested"], result.quality["days_with_data"]), (2, 2))
        record = await self.env.resolver.get("user-a", result.handle, "grid")
        self.assertEqual(record.parents, (self.area, "obs_1", "obs_3"))
        stored = await self.env.resolver.open("user-a", result.handle, "grid")
        self.assertEqual([str(t)[:10] for t in stored["time"].values], ["2024-06-03", "2024-06-17"])

    async def test_comparison_days_all_empty_after_masking_are_too_little_valid(self):
        self.service.make_file = lambda day: granule(day=day, flags=3)
        result = await invoke(self.env, GET, self.params(dates=["2024-06-03", "2024-06-10"], date_role="comparison"))
        self.assertEqual(result.read("situation"), "too_little_valid")
        self.assertNotIn("nearest_dates", result.values)

    async def test_a_range_allows_gaps_and_counts_the_days(self):
        def week(day: str) -> xr.Dataset:
            frames = [granule(day=f"2024-06-0{d}") for d in (1, 2, 4)]
            return xr.concat(frames, dim="time")

        self.service.make_file = week
        result = await invoke(self.env, GET, self.params(
            dates={"start": "2024-06-01", "end": "2024-06-05"}, date_role="range",
        ))
        self.assertEqual(result.status, "done")
        submit = next(args for tool, args in self.earthdata.calls if tool == "retrieve_subset")
        self.assertEqual(submit["time_range"], "2024-06-01T00:00:00/2024-06-05T23:59:59")
        assert result.quality is not None
        self.assertEqual((result.quality["days_with_data"], result.quality["days_requested"]), (3, 5))
        self.assertEqual(result.quality["missing_days"], ["2024-06-03", "2024-06-05"])
        self.assertIn(
            "3 of 5 days had valid data after quality masking; the others are gaps and are not filled.",
            result.disclosures,
        )
        record = await self.env.resolver.get("user-a", result.handle, "grid")
        self.assertEqual((record.descriptor.period.start, record.descriptor.period.end), ("2024-06-01", "2024-06-05"))

    async def test_a_range_with_no_coverage_is_a_gap(self):
        self.service.days_with_data = set()
        result = await invoke(self.env, GET, self.params(
            dates={"start": "2024-06-01", "end": "2024-06-05"}, date_role="range",
        ))
        self.assertEqual(result.read("situation"), "gap")
        self.assertNotIn("nearest_dates", result.values)
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_a_subject_range_with_no_granules_offers_the_nearest_dates_with_data(self):
        self.service.days_with_data = {"2024-05-29", "2024-06-07"}
        result = await invoke(self.env, GET, self.params(dates={"start": "2024-06-01", "end": "2024-06-05"}))
        self.assertEqual((result.status, result.read("situation")), ("insufficient_evidence", "gap"))
        self.assertEqual(result.read("nearest_dates"), ["2024-06-07", "2024-05-29"])
        self.assertNotIn("retrieve_subset", self.tools())


class DatasetChoiceTests(SatelliteCase):
    async def test_only_the_chosen_products_collection_is_retrieved(self):
        self.service.search_hits = [
            {"handle": "dataset_other", "concept_id": "C0000000001-OTHER", "short_name": "OTHER"},
            {"handle": "dataset_1", "concept_id": TEMPO_COLLECTION, "short_name": "TEMPO_NO2_L3"},
        ]
        result = await invoke(self.env, GET, self.params())
        self.assertEqual(result.status, "done")
        submit = next(args for tool, args in self.earthdata.calls if tool == "retrieve_subset")
        self.assertEqual(submit["dataset_handle"], "dataset_1")

    async def test_the_dataset_search_is_filtered_to_the_collections_concept_id(self):
        await invoke(self.env, GET, self.params())
        [search] = self.service.searches
        self.assertEqual(search, {"query": TEMPO_COLLECTION, "filters": {"concept_id": TEMPO_COLLECTION}})

    async def test_a_reply_that_ignores_the_filter_is_still_checked_for_the_exact_collection(self):
        self.service.honour_filters = False
        self.service.search_hits = [{"handle": "dataset_other", "concept_id": "C0000000001-OTHER"}]
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
        self.assertNotIn("retrieve_subset", self.tools())

    async def test_a_search_without_the_chosen_collection_retrieves_nothing(self):
        self.service.search_hits = [{"handle": "dataset_other", "concept_id": "C0000000001-OTHER"}]
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(failure.reason, (
            "sources.satellite.get_data: the data service's search did not list the collection of TEMPO_NO2 "
            f"({TEMPO_COLLECTION}), so nothing was retrieved."
        ))
        self.assertEqual(self.tools(), ["search_datasets"])


class AreaInputTests(SatelliteCase):
    async def test_a_polygons_handle_that_is_not_an_area_is_bad_input(self):
        smoke = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="polygons",
            data={"type": "FeatureCollection", "features": []},
            descriptor=Descriptor(summary="Smoke", region="full product"), operation="test.seed",
            created_at=TODAY,
        )
        failure = await self.failure(self.params(area=smoke.id))
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(failure.reason, (
            "sources.satellite.get_data: area is not an area handle; resolve the place with sources.areas.resolve."
        ))
        self.assertEqual(self.tools(), [])


class IdempotencyTests(SatelliteCase):
    async def test_the_same_step_with_other_dates_retrieves_those_dates(self):
        self.service.make_file = lambda day: granule(day=day, value=1.0e15 if day == "2024-06-01" else 3.0e15)
        first = await invoke(self.env, GET, self.params(), step="event")
        second = await invoke(self.env, GET, self.params(dates="2024-06-02"), step="event")

        self.assertEqual((first.handle, second.handle), ("obs_1", "obs_2"))
        self.assertEqual(self.tools().count("retrieve_subset"), 2)
        record = await self.env.resolver.get("user-a", "obs_2", "grid")
        self.assertEqual((record.descriptor.period.start, record.descriptor.period.end), ("2024-06-02", "2024-06-02"))
        self.assertEqual(float(self.service.files["obs_2"]["vertical_column_troposphere"].max()), 3.0e15)

    async def test_a_retry_after_a_failed_job_submits_again(self):
        # A failed job stays failed: the retry's reuse check is told so too.
        failed = {"status": "failed", "message": "provider says no"}
        self.service.statuses = [failed, failed]
        params = self.params()
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, GET, params, step="event")
        self.assertEqual(caught.exception.result["failed"], ["job_1"])

        again = await begin(self.env, GET, params, step="event")
        self.assertEqual(again["jobs"], [{"job": "job_1", "day": "2024-06-01"}])
        retried = await invoke(self.env, GET, params, step="event", attempt=1)
        self.assertEqual((retried.status, retried.handle), ("done", "obs_2"))
        self.assertEqual(self.tools().count("retrieve_subset"), 2)


class WaitingTests(SatelliteCase):
    async def test_a_paused_job_is_cancelled_and_ends_with_that_reason(self):
        self.service.statuses = [{"status": "running", "message": "The job is paused and may be resumed"}]
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertEqual(failure.reason, (
            "The data provider paused the retrieval of TEMPO_NO2, usually because the request is too large; "
            "it will not finish on its own, so it was cancelled. Narrow the area or the dates."
        ))
        self.assertEqual((failure.result["paused"], failure.result["cancelled"]), (["job_1"], True))
        self.assertEqual(self.service.cancelled, ["job_1"])
        self.assertEqual(self.tools().count("get_retrieval_status"), 1)

    async def test_a_paused_job_the_service_will_not_cancel_says_so(self):
        def refuse(args: dict[str, Any]) -> dict[str, Any]:
            raise FakeToolError("not_found")

        self.earthdata.tools["cancel_retrieval"] = refuse
        self.service.statuses = [{"status": "running", "message": "paused by provider"}]
        failure = await self.failure(self.params())
        self.assertIn("and it could not be cancelled.", failure.reason)
        self.assertFalse(failure.result["cancelled"])

    async def test_a_failed_job_is_named(self):
        self.service.statuses = [{"status": "failed", "message": "provider says no"}]
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertNotIn("provider says no", failure.reason)
        self.assertIsInstance(failure, JobFailed)
        self.assertEqual(failure.result["failed"], ["job_1"])

    async def test_a_job_the_provider_cancelled_ends_the_step_and_is_not_retried(self):
        self.service.statuses = [{"status": "cancelled"}]
        failure = await self.failure(self.params())
        self.assertNotIsInstance(failure, JobFailed)
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(failure.reason, "The retrieval of TEMPO_NO2 was cancelled at the data provider.")
        self.assertEqual(failure.result["cancelled"], ["job_1"])

    async def test_a_job_the_service_no_longer_holds_is_a_dead_job_to_begin_again(self):
        self.service.statuses = [{"status": "expired"}]
        failure = await self.failure(self.params())
        self.assertIsInstance(failure, JobFailed)
        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertEqual(failure.result["expired"], ["job_1"])
        self.assertIn("no longer held by the data service", failure.reason)

    async def test_transient_poll_failures_are_retried_a_bounded_number_of_times(self):
        calls = {"n": 0}
        status = self.service.status

        def flaky(args: dict[str, Any]) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise FakeToolError("provider_unavailable")
            return status(args)

        self.earthdata.tools["get_retrieval_status"] = flaky
        result = await invoke(self.env, GET, self.params())
        self.assertEqual(result.status, "done")

        def down(args: dict[str, Any]) -> dict[str, Any]:
            raise FakeToolError("provider_unavailable")

        self.earthdata.tools["get_retrieval_status"] = down
        failure = await self.failure(self.params())
        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)

    async def test_a_timeout_keeps_the_job_and_complete_resumes_without_resubmitting(self):
        limits = Limits(
            retrieval_soft_bytes=50_000_000, retrieval_hard_bytes=200_000_000, max_point_timeseries_days=366,
            retrieval_timeout_seconds=0.05, poll_min_seconds=0.01, poll_max_seconds=0.02,
        )
        env = await self.make_env(limits=limits, stores=self.env.context.storage)
        self.service.statuses = [{"status": "running", "phase": "processing"}] * 1000
        params = self.params()
        job = await begin(env, GET, params, step="event")
        with self.assertRaises(JobStillRunning) as caught:
            await complete(env, GET, params, job, step="event")
        self.assertEqual(caught.exception.category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertEqual(caught.exception.job, job)
        self.assertEqual(caught.exception.still_running, ("job_1",))

        # Later, in another process: the saved job is completed, never resubmitted.
        self.service.statuses = []
        later = await self.make_env(stores=self.env.context.storage)
        result = await complete(later, GET, params, job, step="event")
        self.assertEqual((result.status, result.handle), ("done", "obs_1"))
        self.assertEqual(self.tools().count("retrieve_subset"), 1)
        entries = await env.context.storage.traces.entries("user-a", env.run_id)
        self.assertEqual([e.status for e in entries], ["waiting", "failed"])
        self.assertEqual(entries[1].result["still_running"], ["job_1"])

    async def test_beginning_the_same_step_again_does_not_submit_again(self):
        params = self.params()
        first = await begin(self.env, GET, params, step="event")
        again = await begin(self.env, GET, params, step="event")
        self.assertEqual(first, again)
        self.assertEqual(self.tools().count("retrieve_subset"), 1)

    async def test_the_secret_is_never_recorded(self):
        await invoke(self.env, GET, self.params(), step="event")
        entries = await self.entries()
        durable = json.dumps([[e.params, e.result, e.reason] for e in entries])
        for handle in ("obs_1", self.area):
            durable += (await self.env.resolver.get("user-a", handle)).model_dump_json()
        self.assertNotIn(SECRET, durable)
        self.assertNotIn(SECRET, json.dumps(self.earthdata.calls))


class PointSeriesTests(SatelliteCase):
    async def test_a_point_series_becomes_a_series_handle(self):
        self.service.series = pd.DataFrame({
            "time": ["2024-06-02T15:00:00Z", "2024-06-01T15:00:00Z", "2024-06-03T15:00:00Z"],
            "vertical_column_troposphere": [2.0, 1.0, None],
        })
        params = {"quantity": NO2, "area": self.area, "dates": {"start": "2024-06-01", "end": "2024-06-03"}}
        result = await invoke(self.env, satellite.get_point_series, params)

        self.assertEqual((result.status, result.kind, result.product), ("done", "series", "TEMPO_NO2"))
        self.assertEqual(result.summary["points"], 2)
        frame = await self.env.resolver.open("user-a", result.handle, "series")
        self.assertEqual(list(frame.columns[:3]), ["time", "value", "series"])
        self.assertEqual(frame["value"].tolist(), [1.0, 2.0])
        record = await self.env.resolver.get("user-a", result.handle)
        self.assertEqual(record.parents, (self.area, "obs_series"))
        submit = next(args for tool, args in self.earthdata.calls if tool == "retrieve_timeseries")
        self.assertTrue(submit["point_sample"])
        self.assertEqual(submit["variables"], ["vertical_column_troposphere", "main_data_quality_flag"])

    async def test_a_span_over_the_limit_is_refused_before_any_call(self):
        params = {"quantity": NO2, "area": self.area, "dates": {"start": "2023-09-01", "end": "2024-06-30"}}
        limits = Limits(
            retrieval_soft_bytes=1, retrieval_hard_bytes=2, max_point_timeseries_days=30,
            retrieval_timeout_seconds=1.0, poll_min_seconds=0.01, poll_max_seconds=0.1,
        )
        env = await self.make_env(limits=limits, stores=self.env.context.storage)
        with self.assertRaises(StepFailure) as caught:
            await invoke(env, satellite.get_point_series, params)
        self.assertEqual(caught.exception.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertIn("304 days", caught.exception.reason)
        self.assertEqual(self.tools(), [])

    async def test_an_empty_series_is_too_little_valid(self):
        self.service.series = pd.DataFrame({"time": ["2024-06-01T15:00:00Z"], "vertical_column_troposphere": [None]})
        params = {"quantity": NO2, "area": self.area, "dates": {"start": "2024-06-01", "end": "2024-06-02"}}
        result = await invoke(self.env, satellite.get_point_series, params)
        self.assertEqual(result.read("situation"), "too_little_valid")


class DiscoveryTests(SatelliteCase):
    async def test_search_lists_datasets_with_their_registry_product(self):
        result = await invoke(self.env, satellite.search, {"query": "C2930763263-LARC_CLOUD"})
        [found] = result.read("datasets")
        self.assertEqual(found["dataset"], "dataset_1")
        self.assertEqual(found["product"], "TEMPO_NO2")
        self.assertEqual(result.method, "The catalogue was searched for 'C2930763263-LARC_CLOUD' and returned 1 datasets.")

    async def test_describe_reads_registry_facts(self):
        result = await invoke(self.env, satellite.describe, {"product": "TEMPO_NO2"})
        self.assertEqual((result.read("quantity"), result.read("cadence"), result.read("record_start")),
                         (NO2, "hourly", "2023-08-02"))
        self.assertEqual(self.tools(), [])

    async def test_an_unknown_product_fails_as_bad_input_for_both_capabilities(self):
        keys = ", ".join(sorted(load_registry()))
        cases = [
            (satellite.describe, {"product": "NOPE"}, "sources.satellite.describe"),
            (satellite.check_availability,
             {"product": "NOPE", "area": self.area, "dates": "2024-06-03"}, "sources.satellite.check_availability"),
        ]
        for capability, params, name in cases:
            with self.subTest(name):
                with self.assertRaises(StepFailure) as caught:
                    await invoke(self.env, capability, params)
                self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)
                self.assertEqual(caught.exception.reason, f"{name} needs a registered product key; call it again with one of: {keys}.")
                self.assertLessEqual(len(caught.exception.reason), MAX_REASON_CHARS)
        self.assertEqual(self.tools(), [])

    def test_the_product_description_lists_every_key_within_the_cap(self):
        for params in (satellite.DescribeParams, satellite.AvailabilityParams):
            description = params.model_fields["product"].description or ""
            for key in load_registry():
                self.assertIn(key, description)
            self.assertLess(len(description), 600)

    async def test_an_empty_or_unregistered_search_carries_the_note(self):
        note = "No registered product matched; only registered products can be retrieved."
        for hits in ([], [{"handle": "dataset_other", "concept_id": "C0000000001-OTHER"}]):
            with self.subTest(hits=len(hits)):
                self.service.search_hits = hits
                result = await invoke(self.env, satellite.search, {"query": "ozone"})
                self.assertEqual(result.read("note"), note)
                self.assertEqual(result.read("registered_quantities"), sorted(satellite.QUANTITIES))

    async def test_a_registered_hit_past_the_limit_means_no_note(self):
        other = [{"handle": f"dataset_{n}", "concept_id": f"C{n}-OTHER"} for n in range(6)]
        match = {"handle": "dataset_1", "concept_id": load_registry()["TEMPO_NO2"].collection_id}
        self.service.search_hits = [*other, match, {"handle": "dataset_x", "concept_id": "C9-OTHER"}]
        result = await invoke(self.env, satellite.search, {"query": "no2", "limit": 5})
        self.assertEqual(result.read("count"), 5)
        self.assertTrue(all(item["product"] is None for item in result.read("datasets")))
        self.assertNotIn("note", result.values)

    async def test_a_search_with_a_registered_match_has_no_note(self):
        result = await invoke(self.env, satellite.search, {"query": "C2930763263-LARC_CLOUD"})
        self.assertNotIn("note", result.values)
        self.assertNotIn("registered_quantities", result.values)

    async def test_availability_by_day_and_by_registry_facts(self):
        self.service.days_with_data = {"2024-06-03"}
        result = await invoke(self.env, satellite.check_availability, {
            "product": "TEMPO_NO2", "area": self.area, "dates": ["2024-06-03", "2024-06-04"],
        })
        self.assertEqual((result.read("available"), result.read("days_with_data"), result.read("days_without_data")),
                         (True, ["2024-06-03"], ["2024-06-04"]))
        self.earthdata.calls.clear()
        early = await invoke(self.env, satellite.check_availability, {
            "product": "TEMPO_NO2", "area": self.area, "dates": "2022-01-01",
        })
        self.assertEqual((early.read("available"), early.read("situation")), (False, "outside_record"))
        self.assertEqual(self.tools(), [])

    async def test_availability_with_no_data_carries_the_fixed_no_data_wording(self):
        early = await invoke(self.env, satellite.check_availability, {
            "product": "TEMPO_NO2", "area": self.area, "dates": "2022-01-01",
        })
        self.assertEqual(len(early.disclosures), 1)
        self.assertTrue(early.disclosures[0].startswith("No data: 2022-01-01 is outside the record"), early.disclosures)

        self.service.days_with_data = set()
        gap = await invoke(self.env, satellite.check_availability, {
            "product": "TEMPO_NO2", "area": self.area, "dates": ["2024-06-03", "2024-06-04"],
        })
        self.assertEqual(gap.read("situation"), "gap")
        self.assertTrue(gap.disclosures and gap.disclosures[0].startswith("No data: TEMPO_NO2 has no observations"))

        self.service.days_with_data = {"2024-06-03"}
        found = await invoke(self.env, satellite.check_availability, {
            "product": "TEMPO_NO2", "area": self.area, "dates": ["2024-06-03", "2024-06-04"],
        })
        self.assertEqual(found.disclosures, ())

    def test_groups_register_with_notes(self):
        registry = CapabilityRegistry()
        registry.add_group(satellite.GROUP)
        for capability in satellite.CAPABILITIES:
            registry.register(capability)
        self.assertEqual([c.name for c in registry.in_group("sources.satellite")], [
            "sources.satellite.check_availability", "sources.satellite.describe", "sources.satellite.get_data",
            "sources.satellite.get_point_series", "sources.satellite.search",
        ])
        self.assertIn("insufficient_evidence", satellite.GROUP.notes)
        self.assertEqual(registry.schema("sources.satellite.get_data")["properties"]["area"]["x-handle-kinds"], ["polygons"])

    def test_the_quantity_description_names_every_registry_quantity(self):
        registry = CapabilityRegistry()
        registry.add_group(satellite.GROUP)
        for capability in satellite.CAPABILITIES:
            registry.register(capability)
        description = registry.schema("sources.satellite.get_data")["properties"]["quantity"]["description"]
        quantities = sorted({cfg.quantity for cfg in load_registry().values() if cfg.quantity})
        for quantity in quantities:
            self.assertIn(quantity, description)
        self.assertIn("no2_tropospheric_column", description)
        self.assertIn("o3_total_column", description)
        self.assertLess(len(description), 400)


if __name__ == "__main__":
    unittest.main()
