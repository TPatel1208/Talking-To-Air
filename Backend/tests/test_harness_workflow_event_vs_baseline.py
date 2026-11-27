"""The event-vs-baseline workflow run end to end in memory on a fake earthdata service, paused and answered."""
import unittest
from typing import Any

import numpy as np
import xarray as xr

from tests.test_harness_grids import granule
from tests.test_harness_sources_satellite import NO2
from tests.test_harness_workflow_packs_run import GET_DATA, MAP, PackRun, step
from tests.test_harness_workflow_runner import AREA_ENTRY
from tta_backend.harness.results import Mode, Result
from tta_backend.harness.workflows import BadAnswer

NAME = "event-vs-baseline"
# A Wednesday; the fixture's clock reads 2024-07-01.
EVENT = "2024-06-12"
# The ten weekdays nearest it within four weeks, leaving out 06-11 and 06-13.
OFFERED = [
    "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-10",
    "2024-06-14", "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20",
]
EVENT_VALUE = 3.0e15
BASELINE_VALUE = 2.0e15
AVAILABILITY = "sources.satellite.check_availability"
SELECT = "operations.select_comparison_days"
AGGREGATE = "operations.aggregate_time"
DIFFERENCE = "operations.difference"


class EventVsBaselineRuns(PackRun):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.retrieved: list[str] = []
        self.empty_days: set[str] = set()
        self.service.make_file = self.file_for
        self.inputs: dict[str, Any] = {"quantity": NO2, "area": "New Jersey", "event_date": EVENT}

    def file_for(self, day: str) -> xr.Dataset:
        self.retrieved.append(day)
        if day in self.empty_days:
            return granule(day=day, value=float("nan"))
        return granule(day=day, value=EVENT_VALUE if day == EVENT else BASELINE_VALUE)

    async def paused(self, inputs: dict[str, Any] | None = None) -> Result:
        result = await self.f.start(inputs or self.inputs, name=NAME)
        self.assertEqual(result.status, "needs_input", result.reason)
        return result

    async def answer(self, run: Result, value: Any) -> Result:
        return await self.f.runner.resume(await self.f.env(run.trace_id), value, step="days")

    def finding(self, result: Result, name: str) -> Any:
        return next(f.value for f in result.findings if f.name == name)

    async def test_it_pauses_on_the_days_with_granules_after_retrieving_only_the_event(self):
        result = await self.paused()

        self.assertEqual(result.mode, Mode.draft(NAME, 1))
        question = result.question or {}
        self.assertEqual(question["kind"], "confirm_list")
        self.assertEqual(question["step"], "days")
        self.assertEqual(question["run_id"], result.trace_id)
        self.assertEqual(question["items"], OFFERED)
        self.assertTrue(question["question"].startswith("These days will be used as the baseline."))
        self.assertEqual(self.retrieved, [EVENT])
        self.assertEqual(await self.f.run_status(result.trace_id), "needs_input")
        entries = await self.f.entries(result.trace_id)
        self.assertEqual(entries[0], AREA_ENTRY)
        self.assertIn(step(SELECT, "candidates"), entries)
        self.assertIn(step(AVAILABILITY, "available"), entries)
        self.assertEqual(entries[-1], ("ask", None, "days", "waiting"))

    async def test_confirming_the_days_as_offered_retrieves_exactly_those_days(self):
        result = await self.answer(await self.paused(), OFFERED)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(sorted(self.retrieved), sorted([EVENT, *OFFERED]))
        self.assertEqual(self.finding(result, "baseline_days"), 10)
        self.assert_charts_stored(result)
        products = [d for d in result.disclosures if d.startswith("Product:")]
        self.assertTrue(all("TEMPO_NO2" in d for d in products), products)
        self.assertFalse(any("weekdays within" in d for d in result.disclosures))
        entries = await self.f.entries(result.trace_id)
        for expected in (step(GET_DATA, "baseline"), step(AGGREGATE, "baseline_mean"), step(DIFFERENCE, "diff"),
                         step(MAP, "figure"), ("ask", None, "days", "done")):
            self.assertIn(expected, entries)
        self.assertEqual(await self.f.run_status(result.trace_id), "done")

    async def test_an_edited_list_is_the_baseline(self):
        edited = [day for day in OFFERED if day not in ("2024-06-05", "2024-06-18")]

        result = await self.answer(await self.paused(), edited)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(sorted(self.retrieved), sorted([EVENT, *edited]))
        self.assertEqual(self.finding(result, "baseline_days"), 8)

    async def test_a_day_the_user_adds_is_retrieved_with_the_rest(self):
        added = [*OFFERED[:5], "2024-06-21"]

        result = await self.answer(await self.paused(), added)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(sorted(self.retrieved), sorted([EVENT, *added]))
        self.assertEqual(self.finding(result, "baseline_days"), 6)

    async def test_an_added_event_day_neighbour_or_future_day_is_left_out_and_disclosed(self):
        # The fixture's clock reads 2024-07-01, and TEMPO_NO2 publishes the same day.
        added = {
            EVENT: "it is the event day.",
            "2024-06-11": "it is next to the event day, 2024-06-12.",
            "2024-06-13": "it is next to the event day, 2024-06-12.",
            "2024-07-02": "it is after 2024-07-01, the last day TEMPO_NO2 has published.",
        }

        result = await self.answer(await self.paused(), [*OFFERED, *added])

        self.assertEqual(result.status, "done", result.reason)
        # The event is retrieved once, as the subject; the baseline only the offered days.
        self.assertEqual(sorted(self.retrieved), sorted([EVENT, *OFFERED]))
        self.assertEqual(self.finding(result, "baseline_days"), 10)
        for day, why in added.items():
            self.assertIn(f"{day} was left out of the baseline: {why}", result.disclosures)

    async def test_left_out_days_count_against_the_minimum(self):
        result = await self.answer(await self.paused(), [*OFFERED[:4], EVENT, "2024-06-13"])

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.reason,
            "Only 4 of the 6 confirmed days can be comparison days; at least 5 are needed for a baseline.",
        )
        self.assertEqual(self.retrieved, [EVENT])

    async def test_the_question_limits_the_answer_to_what_a_comparison_retrieval_takes(self):
        result = await self.paused()

        self.assertEqual(self.f.catalogue.get(NAME, 1).step("days").ask.max_items, 31)
        self.assertEqual(len((result.question or {})["items"]), 10)
        self.assertEqual((result.question or {})["max_items"], 31)

    async def test_a_list_longer_than_a_comparison_retrieval_takes_is_refused_and_the_run_waits(self):
        run = await self.paused()
        too_many = [f"2024-05-{day:02d}" for day in range(1, 32)] + ["2024-06-03"]

        with self.assertRaises(BadAnswer) as refused:
            await self.answer(run, too_many)

        self.assertIn("1 to 31", refused.exception.reason)
        self.assertEqual(await self.f.run_status(run.trace_id), "needs_input")
        self.assertEqual(self.retrieved, [EVENT])

    async def test_the_findings_are_the_two_means_their_difference_and_the_days_used(self):
        result = await self.answer(await self.paused(), OFFERED)

        self.assertEqual(
            [f.name for f in result.findings],
            ["event_mean", "baseline_mean", "mean_difference", "percent_change", "baseline_days"],
        )
        self.assertAlmostEqual(self.finding(result, "event_mean") / EVENT_VALUE, 1.0)
        self.assertAlmostEqual(self.finding(result, "baseline_mean") / BASELINE_VALUE, 1.0)
        self.assertAlmostEqual(self.finding(result, "mean_difference") / (EVENT_VALUE - BASELINE_VALUE), 1.0)
        self.assertAlmostEqual(self.finding(result, "percent_change"), 50.0)
        units = {f.name: f.unit for f in result.findings}
        self.assertEqual(units["mean_difference"], units["event_mean"])
        self.assertEqual(units["percent_change"], "%")
        self.assertEqual(units["baseline_days"], "days")

    async def test_the_method_states_the_rule_and_the_cap(self):
        result = await self.answer(await self.paused(), OFFERED)

        steps = {s.step: s.method or "" for s in await self.f.runner.steps(await self.f.env(result.trace_id))}
        self.assertIn("weekdays within 28 days either side of 2024-06-12", steps["candidates"])
        self.assertIn("the cap of 10 days bounds the retrieval", steps["candidates"])
        self.assertIn("10 of the 10 usable confirmed days had valid data", steps["baseline"])
        self.assertTrue(steps["checked"].startswith("10 of the 10 confirmed days were kept"))
        self.assertIn("event-day value minus its baseline value", steps["diff"])

    async def test_too_few_days_with_granules_stops_before_asking(self):
        self.service.days_with_data = {EVENT, *OFFERED[:3]}

        result = await self.f.start(self.inputs, name=NAME)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.reason,
            "Only 3 of the 10 candidate days have TEMPO_NO2 data over New Jersey; at least 5 are needed for a baseline.",
        )
        self.assertIsNone(result.question)
        self.assertEqual(self.retrieved, [EVENT])
        entries = await self.f.entries(result.trace_id)
        self.assertNotIn("ask", [entry[0] for entry in entries])

    async def test_too_few_confirmed_days_stop_without_retrieving_them(self):
        result = await self.answer(await self.paused(), OFFERED[:4])

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.reason,
            "Only 4 of the 4 confirmed days can be comparison days; at least 5 are needed for a baseline.",
        )
        self.assertEqual(self.retrieved, [EVENT])

    async def test_too_few_days_left_after_masking_stops_after_the_retrieval(self):
        self.empty_days = set(OFFERED[:6])

        result = await self.answer(await self.paused(), OFFERED)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.reason,
            "Only 4 of the 10 usable confirmed days had valid TEMPO_NO2 data over New Jersey; at least 5 are needed "
            "for a baseline.",
        )
        self.assertEqual(sorted(self.retrieved), sorted([EVENT, *OFFERED]))
        self.assertEqual(result.charts, ())
        self.assertTrue(any("no valid data after quality masking" in d for d in result.disclosures))

    async def test_a_few_days_empty_after_masking_are_dropped_and_disclosed(self):
        self.empty_days = {OFFERED[0], OFFERED[1]}

        result = await self.answer(await self.paused(), OFFERED)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(self.finding(result, "baseline_days"), 8)
        self.assertTrue(any(
            "2 of the requested days had no valid data after quality masking" in d and OFFERED[0] in d
            for d in result.disclosures
        ))

    async def test_an_answer_that_is_not_a_list_of_dates_is_refused_and_the_run_keeps_waiting(self):
        run = await self.paused()

        with self.assertRaises(BadAnswer):
            await self.answer(run, ["not a date"])
        self.assertEqual(await self.f.run_status(run.trace_id), "needs_input")
        self.assertEqual(self.retrieved, [EVENT])

    async def test_a_weekend_event_is_compared_with_weekend_days(self):
        result = await self.paused({**self.inputs, "event_date": "2024-06-15"})

        items = (result.question or {})["items"]
        self.assertEqual(len(items), 10)
        self.assertTrue(all(np.datetime64(day).astype("datetime64[D]").item().weekday() >= 5 for day in items))
        self.assertNotIn("2024-06-16", items)


if __name__ == "__main__":
    unittest.main()
