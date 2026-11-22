"""Choosing an event day's comparison days: the date rule, and the capability that applies it to a product's record."""
import datetime
import tempfile
import unittest
from datetime import date
from typing import Any

from tta_backend.domain.analysis.dates import MAX_DAYS, comparison_days, day_type, usable_days
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.operations.days import check_comparison_days, select_comparison_days
from tta_backend.harness.testing import FixedClock, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

FAR = date(2030, 1, 1)


def iso(days: list[date]) -> list[str]:
    return [day.isoformat() for day in days]


class DayRuleTests(unittest.TestCase):
    def test_a_weekday_event_gets_the_nearest_weekdays_in_date_order(self):
        # Wednesday 2024-06-12.
        days = comparison_days(date(2024, 6, 12), latest=FAR)

        self.assertEqual(iso(days), [
            "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-10",
            "2024-06-14", "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20",
        ])
        self.assertTrue(all(day_type(day) == "weekday" for day in days))

    def test_a_weekend_event_gets_weekend_days(self):
        # Saturday 2024-06-15.
        days = comparison_days(date(2024, 6, 15), latest=FAR)

        self.assertEqual(iso(days), [
            "2024-05-25", "2024-05-26", "2024-06-01", "2024-06-02", "2024-06-08",
            "2024-06-09", "2024-06-22", "2024-06-23", "2024-06-29", "2024-06-30",
        ])
        self.assertEqual(day_type(date(2024, 6, 16)), "weekend")
        self.assertEqual(day_type(date(2024, 6, 17)), "weekday")

    def test_the_event_and_the_day_either_side_are_left_out(self):
        # Sunday 2024-06-16: the Saturday before it is the excluded neighbour.
        days = comparison_days(date(2024, 6, 16), latest=FAR, limit=28)

        for day in (date(2024, 6, 15), date(2024, 6, 16), date(2024, 6, 17)):
            self.assertNotIn(day, days)
        self.assertIn(date(2024, 6, 22), days)

    def test_no_day_is_after_the_latest(self):
        days = comparison_days(date(2024, 6, 12), latest=date(2024, 6, 14))

        self.assertEqual(max(days), date(2024, 6, 14))
        self.assertEqual(len(days), MAX_DAYS)
        self.assertEqual(min(days), date(2024, 5, 29))

    def test_no_day_is_before_the_earliest(self):
        days = comparison_days(date(2024, 6, 12), latest=FAR, earliest=date(2024, 6, 10))

        self.assertEqual(min(days), date(2024, 6, 10))
        self.assertEqual(len(days), MAX_DAYS)

    def test_at_most_the_limit_and_nearest_first_with_ties_taken_earlier(self):
        days = comparison_days(date(2024, 6, 12), latest=FAR, limit=3)

        # 06-10 and 06-14 are 2 days away; 06-07 and 06-17 are 5, and the earlier is taken.
        self.assertEqual(iso(days), ["2024-06-07", "2024-06-10", "2024-06-14"])
        self.assertEqual(len(comparison_days(date(2024, 6, 12), latest=FAR, limit=40)), 38)

    def test_an_event_at_a_year_end_reaches_into_the_next_year(self):
        # Tuesday 2024-12-31.
        days = comparison_days(date(2024, 12, 31), latest=FAR)

        self.assertIn(date(2025, 1, 2), days)
        self.assertIn(date(2024, 12, 27), days)
        self.assertNotIn(date(2025, 1, 1), days)
        self.assertNotIn(date(2024, 12, 30), days)
        self.assertEqual(days, sorted(days))

    def test_an_event_at_a_month_end_reaches_into_the_next_month(self):
        # Friday 2024-05-31; 06-01 is left out and 06-02 is a Sunday.
        days = comparison_days(date(2024, 5, 31), latest=FAR, limit=4)

        self.assertEqual(iso(days), ["2024-05-27", "2024-05-28", "2024-05-29", "2024-06-03"])

    def test_an_event_on_the_latest_day_looks_only_back(self):
        days = comparison_days(date(2024, 6, 12), latest=date(2024, 6, 12))

        self.assertTrue(all(day < date(2024, 6, 11) for day in days))
        self.assertEqual(len(days), MAX_DAYS)

    def test_nothing_in_range_is_an_empty_list(self):
        self.assertEqual(comparison_days(date(2024, 6, 12), latest=date(2024, 6, 1), earliest=date(2024, 6, 1)), [])

    def test_a_window_or_limit_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            comparison_days(date(2024, 6, 12), latest=FAR, limit=0)


class UsableDaysTests(unittest.TestCase):
    def test_the_event_its_neighbours_and_days_outside_the_record_are_left_out_with_why(self):
        event = date(2024, 6, 12)
        days = [date(2024, 6, d) for d in (20, 4, 11, 12, 13, 4)] + [date(2024, 7, 2), date(2023, 1, 2)]

        kept, dropped = usable_days(event, days, latest=date(2024, 7, 1), earliest=date(2023, 8, 2))

        self.assertEqual(kept, [date(2024, 6, 4), date(2024, 6, 20)])
        self.assertEqual(dropped, [
            (date(2023, 1, 2), "before_earliest"), (date(2024, 6, 11), "next_to_event"),
            (date(2024, 6, 12), "event_day"), (date(2024, 6, 13), "next_to_event"), (date(2024, 7, 2), "after_latest"),
        ])

    def test_a_day_of_another_type_or_further_away_is_the_users_choice(self):
        # Saturday 2024-06-15 for a Wednesday event, and a day six weeks before it.
        kept, dropped = usable_days(date(2024, 6, 12), [date(2024, 6, 15), date(2024, 5, 1)], latest=FAR)

        self.assertEqual(kept, [date(2024, 5, 1), date(2024, 6, 15)])
        self.assertEqual(dropped, [])


class _CapabilityCase(unittest.IsolatedAsyncioTestCase):
    capability: Any = select_comparison_days

    async def run_it(self, params: dict, *, today: datetime.datetime = datetime.datetime(2024, 7, 1, 12, tzinfo=datetime.UTC)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = await make_run_env(tmp.name, context=make_run_context(clock=FixedClock(today)))
        return await invoke(env, self.capability, params)


class SelectComparisonDaysTests(_CapabilityCase):
    async def test_it_lists_the_days_with_the_rule_and_retrieves_nothing(self):
        result = await self.run_it({"event_date": "2024-06-12", "product": "TEMPO_NO2"})

        self.assertEqual(result.status, "done")
        self.assertIsNone(result.handle)
        self.assertIsNone(result.product)
        self.assertEqual(result.values["selection"], "weekdays within 28 days either side of 2024-06-12, nearest first")
        self.assertEqual(len(result.values["days"]), 10)
        self.assertEqual(result.values["day_type"], "weekday")
        self.assertEqual(result.values["latest"], "2024-07-01")
        self.assertIn("the cap of 10 days bounds the retrieval", result.method or "")

    async def test_never_after_today_without_a_product(self):
        result = await self.run_it({"event_date": "2024-06-26"})

        self.assertEqual(result.values["latest"], "2024-07-01")
        self.assertLessEqual(max(result.values["days"]), "2024-07-01")

    async def test_a_product_with_latency_stops_at_its_last_published_day(self):
        # OMI_NO2 is published 2 days after observation.
        result = await self.run_it({"event_date": "2024-06-26", "product": "OMI_NO2"})

        self.assertEqual(result.values["latest"], "2024-06-29")
        self.assertLessEqual(max(result.values["days"]), "2024-06-29")

    async def test_a_product_record_start_bounds_the_days(self):
        # TEMPO_NO2's record starts on 2023-08-02, a Wednesday.
        result = await self.run_it({"event_date": "2023-08-04", "product": "TEMPO_NO2"})

        self.assertGreaterEqual(min(result.values["days"]), "2023-08-02")

    async def test_no_day_in_range_is_insufficient_evidence_with_fixed_wording(self):
        result = await self.run_it(
            {"event_date": "2024-07-15", "product": "TEMPO_NO2"},
            today=datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, (
            "No comparison days: no weekday within 28 days either side of 2024-07-15, other than the event day "
            "and the day either side of it, falls between 2023-08-02 and 2024-06-01."
        ))
        self.assertEqual(result.values["situation"], "no_comparison_days")

    async def test_an_unknown_product_is_refused(self):
        with self.assertRaises(StepFailure) as raised:
            await self.run_it({"event_date": "2024-06-12", "product": "NOT_A_PRODUCT"})

        self.assertEqual(raised.exception.category, FailureCategory.BAD_INPUT)
        self.assertIn("needs a registered product key", raised.exception.reason)


class CheckComparisonDaysTests(_CapabilityCase):
    capability = check_comparison_days

    async def test_left_out_days_are_disclosed_one_by_one_in_fixed_wording(self):
        result = await self.run_it({
            "event_date": "2024-06-26", "product": "OMI_NO2",
            "days": ["2024-06-19", "2024-06-26", "2024-06-27", "2024-06-30", "2024-06-19"],
        })

        self.assertEqual(result.status, "done")
        self.assertIsNone(result.product)
        self.assertEqual(result.values["days"], ["2024-06-19"])
        self.assertEqual((result.values["kept"], result.values["confirmed"]), (1, 4))
        self.assertEqual(result.values["dropped"], [
            {"day": "2024-06-26", "reason": "event_day"}, {"day": "2024-06-27", "reason": "next_to_event"},
            {"day": "2024-06-30", "reason": "after_latest"},
        ])
        self.assertEqual(result.disclosures, (
            "2024-06-26 was left out of the baseline: it is the event day.",
            "2024-06-27 was left out of the baseline: it is next to the event day, 2024-06-26.",
            "2024-06-30 was left out of the baseline: it is after 2024-06-29, the last day OMI_NO2 has published.",
        ))
        self.assertTrue((result.method or "").startswith("1 of the 4 confirmed days were kept"))

    async def test_a_day_before_the_record_is_left_out(self):
        result = await self.run_it({"event_date": "2023-08-10", "product": "TEMPO_NO2", "days": ["2023-08-01", "2023-08-03"]})

        self.assertEqual(result.values["days"], ["2023-08-03"])
        self.assertEqual(
            result.disclosures,
            ("2023-08-01 was left out of the baseline: it is before 2023-08-02, the first day TEMPO_NO2 holds.",),
        )

    async def test_every_day_left_out_is_an_empty_list_not_a_failure(self):
        result = await self.run_it({"event_date": "2024-06-12", "days": ["2024-06-12"]})

        self.assertEqual(result.status, "done")
        self.assertEqual(result.values["days"], [])


if __name__ == "__main__":
    unittest.main()
