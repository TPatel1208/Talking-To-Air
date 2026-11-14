"""The in-memory run-context parts behave as the harness will rely on."""
import dataclasses
import datetime
import unittest

from tta_backend.harness.testing import (
    FixedClock,
    MemoryArtifacts,
    RecordingEvents,
    StaticCredentials,
    make_run_context,
)


class HarnessFakesTests(unittest.IsolatedAsyncioTestCase):
    def test_events_are_recorded_in_order(self):
        events = RecordingEvents()
        events.progress("one", stage="a")
        events.progress("two", detail={"k": 1})
        events.chart({"title": "c"})
        events.question({"q": "?"})
        self.assertEqual([e["message"] for e in events.progress_events], ["one", "two"])
        self.assertEqual(events.progress_events[0]["stage"], "a")
        self.assertEqual(events.progress_events[1]["detail"], {"k": 1})
        self.assertEqual(events.charts, [{"title": "c"}])
        self.assertEqual(events.questions, [{"q": "?"}])

    def test_fixed_clock_is_stable_and_utc(self):
        now = datetime.datetime(2026, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
        clock = FixedClock(now)
        self.assertEqual(clock.now(), now)
        self.assertEqual(clock.today(), datetime.date(2026, 3, 4))
        with self.assertRaises(ValueError):
            FixedClock(datetime.datetime(2026, 1, 1))

    async def test_credentials_miss_returns_none(self):
        credentials = StaticCredentials({"epa": "s3cret"})
        self.assertEqual(await credentials.secret("epa"), "s3cret")
        self.assertIsNone(await credentials.secret("other"))

    async def test_memory_artifacts_mint_distinct_ids(self):
        artifacts = MemoryArtifacts()
        chart = await artifacts.store_chart({"title": "C"})
        table = await artifacts.store_table("T", ["a"], [{"a": 1}])
        self.assertNotEqual(chart.id, table.id)
        self.assertEqual((chart.type, chart.title), ("chart", "C"))
        self.assertEqual(table.type, "table")
        self.assertEqual(artifacts.stored[table.id]["rows"], [{"a": 1}])

    def test_run_context_is_frozen(self):
        ctx = make_run_context(user_id="u", conversation_id="c")
        self.assertEqual((ctx.identity.user_id, ctx.identity.conversation_id), ("u", "c"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.identity = ctx.identity  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.identity.user_id = "x"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
