"""What "Recent analyses" is allowed to show.

The ``session_metadata`` row is written the moment the message is posted,
because it is the only record of who owns the thread and three endpoints --
history, stop and the T63 reattach stream -- refuse without it. It is not
evidence that the conversation has anything in it: on the fast path the
transcript is checkpointed once, at the end of the turn, so a turn that is
stopped, times out, errors or dies with its replica leaves a titled row over
an empty conversation, listed forever.

``first_frame_at`` is the second fact, stamped when the turn first produces
narration. The listing reads it *or* the presence of a checkpoint: the fast
path narrates long before it checkpoints, and a stopped supervisor turn
checkpoints without ever narrating, so neither fact alone covers both.

These tests drive the repository against a recording connection, so they pin
the SQL each function issues rather than what Postgres does with it. That is
enough for the two guards that carry the risk -- the listing's filter, and
the backfill that must not run twice -- and it is the same shape as
test_session_repository.py.
"""
from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch


class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class FakeConnection:
    """Records every statement, answers reads from ``rows_for``."""

    def __init__(self, rows_for=None):
        self.statements: list[tuple[str, tuple | None]] = []
        self.commits = 0
        self._rows_for = rows_for or (lambda sql: ())

    async def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))
        return FakeCursor(self._rows_for(sql))

    async def commit(self):
        self.commits += 1

    def sql_matching(self, *needles: str) -> list[str]:
        return [sql for sql, _ in self.statements if all(n in sql for n in needles)]


def connected(conn):
    @asynccontextmanager
    async def _pg_connection(*args, **kwargs):
        yield conn

    return patch(
        "tta_backend.repositories.session_metadata_repository.pg_connection",
        _pg_connection,
    )


class SessionListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_listing_leaves_out_a_thread_that_never_produced_a_frame(self):
        from tta_backend.repositories.session_metadata_repository import list_session_metadata

        conn = FakeConnection(lambda sql: [("th-1", "How is the air", None)])
        with connected(conn):
            await list_session_metadata("user-1")

        selects = conn.sql_matching("FROM session_metadata", "WHERE user_id")
        self.assertEqual(len(selects), 1, conn.statements)
        self.assertIn(
            "first_frame_at IS NOT NULL",
            selects[0],
            "an unstamped thread is one whose turn never said anything -- listing it "
            "is the empty-conversation row this column exists to hide",
        )

    async def test_a_thread_with_a_transcript_lists_even_with_no_stamp(self):
        """"Has something in it" is narration OR a transcript, not narration
        alone -- and the difference is a live bug, not a hypothetical.

        The stamp only sees frames from the turn's own generator. On the
        supervisor route LangGraph checkpoints the human message at the first
        superstep, well before the first frame is yielded, and a Stop in that
        window is written by the registry -- which the stamp never sees. The
        thread then holds the user's question and nothing lists it. Observed
        on the deployed stack: thread 70dc275f, stopped 26s after it started,
        two checkpoint rows, no stamp.

        The fast path is the other half and is why the stamp exists at all:
        it checkpoints once, at the end, so a turn still running has a
        transcript of nothing.
        """
        from tta_backend.repositories.session_metadata_repository import list_session_metadata

        conn = FakeConnection(lambda sql: [("th-1", "How is the air", None)])
        with connected(conn):
            await list_session_metadata("user-1")

        listing = conn.sql_matching("FROM session_metadata", "WHERE user_id")[0]
        self.assertIn("EXISTS", listing)
        self.assertIn("FROM checkpoints", listing)
        self.assertIn(
            "first_frame_at IS NOT NULL OR",
            listing,
            "the two conditions are alternatives: a running turn has narrated "
            "without checkpointing, a stopped one checkpointed without narrating",
        )

    async def test_a_listed_row_carries_no_first_frame_column_into_the_response(self):
        """The column decides membership; it is not part of the contract the
        frontend reads, which still shapes a row as id/title/created_at."""
        from tta_backend.repositories.session_metadata_repository import list_session_metadata

        conn = FakeConnection(lambda sql: [("th-1", "How is the air", None)])
        with connected(conn):
            rows = await list_session_metadata("user-1")

        self.assertEqual([set(row) for row in rows], [{"id", "title", "created_at"}])


class SessionOrderTests(unittest.IsolatedAsyncioTestCase):
    """Most recently used first, where "used" is the last turn that spoke.

    Creation order is what the sidebar had before, and it is wrong the
    moment a thread is returned to: a conversation carried on all week sank
    below whatever was started after it and never touched again.
    """

    async def test_threads_are_ordered_by_their_last_turn_not_their_first(self):
        from tta_backend.repositories.session_metadata_repository import list_session_metadata

        conn = FakeConnection(lambda sql: [("th-1", "How is the air", None)])
        with connected(conn):
            await list_session_metadata("user-1")

        listing = conn.sql_matching("FROM session_metadata", "WHERE user_id")[0]
        self.assertIn("ORDER BY COALESCE(last_event_at,", listing)
        self.assertIn("DESC", listing.split("ORDER BY")[1])

    async def test_a_thread_with_no_stamps_falls_back_to_when_it_was_created(self):
        """Two kinds of row have no ``last_event_at`` and both are real:
        every thread that predates the column (it is added without a
        backfill), and a thread listed by the checkpoint branch, which was
        stopped before it narrated and so has no ``first_frame_at`` either.
        Without the fallback they sort as NULL -- last, together, forever."""
        from tta_backend.repositories.session_metadata_repository import list_session_metadata

        conn = FakeConnection(lambda sql: [("th-1", "How is the air", None)])
        with connected(conn):
            await list_session_metadata("user-1")

        listing = conn.sql_matching("FROM session_metadata", "WHERE user_id")[0]
        order = listing.split("ORDER BY")[1]
        self.assertIn("first_frame_at", order)
        self.assertIn("created_at", order)
        self.assertIn("thread_id", order, "a stable tiebreak, or equal stamps shuffle")


class ActivityStampTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_stamp_records_when_the_thread_first_narrated(self):
        from tta_backend.repositories.session_metadata_repository import mark_session_activity

        conn = FakeConnection()
        with connected(conn):
            await mark_session_activity("th-1")

        updates = conn.sql_matching("UPDATE session_metadata", "first_frame_at")
        self.assertEqual(len(updates), 1, conn.statements)
        self.assertEqual(conn.statements[0][1], ("th-1",))
        self.assertEqual(conn.commits, 1)

    async def test_a_later_turn_on_the_same_thread_does_not_move_the_first_stamp(self):
        """Every turn's first frame calls this, not just the thread's first.

        ``first_frame_at`` answers "did this conversation ever begin", and
        the listing filter reads it. Letting a later turn move it would make
        a thread read as though it had just been created every time it was
        used -- which is what ``last_event_at`` is for, and why the two
        cannot be one column.
        """
        from tta_backend.repositories.session_metadata_repository import mark_session_activity

        conn = FakeConnection()
        with connected(conn):
            await mark_session_activity("th-1")

        sql = conn.statements[0][0]
        self.assertIn("first_frame_at = COALESCE(first_frame_at, now())", sql)
        self.assertNotIn(
            "first_frame_at IS NULL",
            sql,
            "the guard moved into the SET on purpose: a WHERE that skips an "
            "already-stamped thread would skip its last_event_at too, and every "
            "turn after the first would stop reordering the sidebar",
        )

    async def test_every_turn_moves_the_thread_up_the_list(self):
        """The half a NULL guard would silently swallow."""
        from tta_backend.repositories.session_metadata_repository import mark_session_activity

        conn = FakeConnection()
        with connected(conn):
            await mark_session_activity("th-1")

        self.assertIn("last_event_at = now()", conn.statements[0][0])

    async def test_both_stamps_are_one_write(self):
        """This runs on the path that carries every answer, so the two facts
        share the event that produced them and the round trip."""
        from tta_backend.repositories.session_metadata_repository import mark_session_activity

        conn = FakeConnection()
        with connected(conn):
            await mark_session_activity("th-1")

        self.assertEqual(len(conn.sql_matching("UPDATE session_metadata")), 1, conn.statements)


class FirstFrameMigrationTests(unittest.IsolatedAsyncioTestCase):
    """The backfill is the dangerous half of this change.

    Existing threads have no stamp and every one of them is real, so the
    column is filled from ``created_at`` when it is added. But
    ``ensure_session_metadata_table`` runs on every startup, and rows written
    by the new code are *deliberately* NULL until their turn produces --
    backfilling them on the next restart would list exactly the empty threads
    this change hides, and would do it silently.
    """

    def _rows_for(self, column_exists: bool):
        def rows(sql: str):
            if "information_schema.columns" in sql:
                return [(1,)] if column_exists else []
            return []

        return rows

    async def test_adding_the_column_backfills_the_threads_that_predate_it(self):
        from tta_backend.repositories.session_metadata_repository import (
            ensure_session_metadata_table,
        )

        conn = FakeConnection(self._rows_for(column_exists=False))
        with connected(conn):
            await ensure_session_metadata_table()

        self.assertEqual(
            len(conn.sql_matching("UPDATE session_metadata", "SET first_frame_at = created_at")),
            1,
            conn.statements,
        )

    async def test_a_restart_does_not_backfill_a_turn_that_is_still_running(self):
        from tta_backend.repositories.session_metadata_repository import (
            ensure_session_metadata_table,
        )

        conn = FakeConnection(self._rows_for(column_exists=True))
        with connected(conn):
            await ensure_session_metadata_table()

        self.assertEqual(
            conn.sql_matching("UPDATE session_metadata", "SET first_frame_at = created_at"),
            [],
            "the column was already there, so every NULL left in it is a thread "
            "whose turn has not produced yet -- not a thread that predates the column",
        )

    async def test_the_column_is_added_before_it_is_read(self):
        """The probe reads information_schema, not the column itself, so the
        order is: ask, add, backfill. Adding first would make the probe
        always true and the backfill never run."""
        from tta_backend.repositories.session_metadata_repository import (
            ensure_session_metadata_table,
        )

        conn = FakeConnection(self._rows_for(column_exists=False))
        with connected(conn):
            await ensure_session_metadata_table()

        order = [sql for sql, _ in conn.statements if "first_frame_at" in sql]
        self.assertEqual(len(order), 3, order)
        self.assertIn("information_schema.columns", order[0])
        self.assertIn("ADD COLUMN IF NOT EXISTS first_frame_at", order[1])
        self.assertIn("UPDATE session_metadata", order[2])


class LastEventMigrationTests(unittest.IsolatedAsyncioTestCase):
    """``last_event_at`` takes the other road: added, never filled.

    It can, because the listing COALESCEs down to ``created_at``, so an
    existing thread sorts sensibly with the column empty. That is worth more
    than a backfill: filling it on startup would need the same probe as
    above to avoid running twice, and getting that wrong would reset the
    stamp of every thread whose turn was running across the restart --
    dropping live conversations down the sidebar.
    """

    def _rows_for(self, column_exists: bool):
        def rows(sql: str):
            if "information_schema.columns" in sql:
                return [(1,)] if column_exists else []
            return []

        return rows

    async def test_the_column_is_added(self):
        from tta_backend.repositories.session_metadata_repository import (
            ensure_session_metadata_table,
        )

        conn = FakeConnection(self._rows_for(column_exists=False))
        with connected(conn):
            await ensure_session_metadata_table()

        self.assertEqual(
            len(conn.sql_matching("ADD COLUMN IF NOT EXISTS last_event_at")), 1, conn.statements
        )

    async def test_nothing_backfills_it(self):
        from tta_backend.repositories.session_metadata_repository import (
            ensure_session_metadata_table,
        )

        for exists in (True, False):
            with self.subTest(column_exists=exists):
                conn = FakeConnection(self._rows_for(column_exists=exists))
                with connected(conn):
                    await ensure_session_metadata_table()

                self.assertEqual(
                    conn.sql_matching("UPDATE session_metadata", "last_event_at"),
                    [],
                    "startup must not write this column -- a running turn's stamp "
                    "would be reset to something older than itself",
                )


if __name__ == "__main__":
    unittest.main()
