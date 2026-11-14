"""The transcript store against a real Postgres."""
import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from tta_backend.harness.results import Mode
from tta_backend.models.artifact import ArtifactReference
from tta_backend.models.transcript import TranscriptTurn
from tta_backend.repositories import transcript_repository as repo
from tests.postgres_helpers import PostgresTestCase, isolated_schema


def _turn(thread_id="t1", user_id="u1", text="q", **overrides):
    return TranscriptTurn(
        thread_id=thread_id, user_id=user_id, user_text=text, answer_text=f"a:{text}", **overrides
    )


class TranscriptRepositoryPostgresTests(PostgresTestCase):
    async def asyncSetUp(self):
        manager = isolated_schema()
        connect = await manager.__aenter__()
        self.addAsyncCleanup(manager.__aexit__, None, None, None)
        patcher = patch.object(repo, "pg_connection", connect)
        patcher.start()
        self.addCleanup(patcher.stop)
        await repo.ensure_transcript_table()

    async def test_round_trip_keeps_json_fields(self):
        turn = _turn(
            charts=[{"chart_id": "c1", "title": "NO2"}],
            artifacts=[ArtifactReference(id="tbl_1", type="table", title="T", row_count=2, metadata={"k": [1, 2]})],
            sources=["EPA AQS", "TEMPO"],
            label=Mode.validated("trend", 2),
            status="stopped",
        )
        stored = await repo.append_turn(turn)
        self.assertEqual(stored.turn_index, 0)
        self.assertIsNotNone(stored.created_at)

        [fetched] = await repo.list_turns("t1", "u1")
        self.assertEqual(fetched, stored)
        self.assertEqual(fetched.charts, [{"chart_id": "c1", "title": "NO2"}])
        self.assertEqual(fetched.artifacts[0].metadata, {"k": [1, 2]})
        self.assertEqual(fetched.label, Mode.validated("trend", 2))
        self.assertEqual(fetched.sources, ["EPA AQS", "TEMPO"])
        self.assertEqual(fetched.status, "stopped")

    async def test_missing_label_round_trips_as_none(self):
        await repo.append_turn(_turn())
        [fetched] = await repo.list_turns("t1", "u1")
        self.assertIsNone(fetched.label)

    async def test_turns_come_back_in_order_per_thread(self):
        for text in ("one", "two", "three"):
            await repo.append_turn(_turn(text=text))
        await repo.append_turn(_turn(thread_id="t2", text="other"))

        turns = await repo.list_turns("t1", "u1")
        self.assertEqual([t.turn_index for t in turns], [0, 1, 2])
        self.assertEqual([t.user_text for t in turns], ["one", "two", "three"])
        other = await repo.list_turns("t2", "u1")
        self.assertEqual([t.turn_index for t in other], [0])

    async def test_racing_appends_get_distinct_consecutive_indexes(self):
        stored = await asyncio.gather(*(repo.append_turn(_turn(text=str(i))) for i in range(12)))
        self.assertEqual(sorted(t.turn_index for t in stored), list(range(12)))
        turns = await repo.list_turns("t1", "u1")
        self.assertEqual([t.turn_index for t in turns], list(range(12)))

    async def test_reads_and_deletes_are_scoped_to_the_owner(self):
        await repo.append_turn(_turn())
        self.assertEqual(await repo.list_turns("t1", "intruder"), [])

        await repo.delete_transcript("t1", "intruder")
        self.assertEqual(len(await repo.list_turns("t1", "u1")), 1)

        await repo.delete_transcript("t1", "u1")
        self.assertEqual(await repo.list_turns("t1", "u1"), [])

    async def test_users_sharing_a_thread_id_have_independent_transcripts(self):
        for text in ("a", "b", "c"):
            await repo.append_turn(_turn(text=text))
        stranger = await repo.append_turn(_turn(user_id="u2", text="x"))
        self.assertEqual(stranger.turn_index, 0)
        owner_next = await repo.append_turn(_turn(text="d"))
        self.assertEqual(owner_next.turn_index, 3)

        self.assertEqual([t.user_text for t in await repo.list_turns("t1", "u1")], ["a", "b", "c", "d"])
        self.assertEqual([t.user_text for t in await repo.list_turns("t1", "u2")], ["x"])

    async def test_supplied_created_at_is_kept_and_default_uses_the_database_clock(self):
        when = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
        kept = await repo.append_turn(_turn(created_at=when))
        self.assertEqual(kept.created_at, when)
        defaulted = await repo.append_turn(_turn())
        self.assertLess(abs(datetime.now(UTC) - defaulted.created_at), timedelta(minutes=1))

    async def test_ensure_is_safe_to_call_twice(self):
        await repo.append_turn(_turn())
        await repo.ensure_transcript_table()
        await repo.ensure_transcript_table()
        self.assertEqual(len(await repo.list_turns("t1", "u1")), 1)

    async def test_non_finite_floats_are_stored_as_null(self):
        await repo.append_turn(_turn(
            charts=[{"chart_id": "c1", "valid_max": float("inf"), "vals": [1.0, float("nan")]}],
            artifacts=[ArtifactReference(id="a", type="map", metadata={"x": float("-inf")})],
        ))
        [fetched] = await repo.list_turns("t1", "u1")
        self.assertIsNone(fetched.charts[0]["valid_max"])
        self.assertEqual(fetched.charts[0]["vals"], [1.0, None])
        self.assertIsNone(fetched.artifacts[0].metadata["x"])


if __name__ == "__main__":
    unittest.main()
