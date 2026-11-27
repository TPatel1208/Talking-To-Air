import json
import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from tta_backend.harness.results import Mode
from tta_backend.models.artifact import ArtifactReference
from tta_backend.models.transcript import TranscriptTurn
from tta_backend.repositories import transcript_repository

_PATCH = "tta_backend.repositories.transcript_repository.pg_connection"


def _fake_pg_connection(conn):
    @asynccontextmanager
    async def _factory(*args, **kwargs):
        yield conn

    return _factory


def _turn(**overrides):
    fields = {"thread_id": "t1", "user_id": "u1", "user_text": "hi", "answer_text": "hello"}
    fields.update(overrides)
    return TranscriptTurn(**fields)


class TranscriptTurnValidationTests(unittest.TestCase):
    def test_status_is_restricted(self):
        with self.assertRaises(ValidationError):
            _turn(status="pending")

    def test_label_must_be_a_consistent_mode(self):
        _turn(label=Mode.validated("trend", 1))
        with self.assertRaises(ValidationError):
            _turn(label={"kind": "validated"})

    def test_nul_is_rejected_in_every_stored_field(self):
        bad = "a\x00b"
        for overrides in (
            {"thread_id": bad},
            {"user_id": bad},
            {"user_text": bad},
            {"answer_text": bad},
            {"sources": [bad]},
            {"charts": [{"title": bad}]},
            {"charts": [{bad: 1}]},
            {"artifacts": [ArtifactReference(id="a", type="map", metadata={"k": [bad]})]},
        ):
            with self.subTest(overrides=list(overrides)):
                with self.assertRaises(ValidationError):
                    _turn(**overrides)

    def test_created_at_must_be_timezone_aware(self):
        _turn(created_at=datetime.now(UTC))
        with self.assertRaises(ValidationError):
            _turn(created_at=datetime(2026, 1, 1))


class HistoryMessagesTests(unittest.TestCase):
    def test_pair_matches_the_history_view_shape(self):
        turn = _turn(
            charts=[{"chart_id": "c1"}],
            artifacts=[ArtifactReference(id="tbl_1", type="table", title="T", row_count=3)],
        )
        user, assistant = turn.to_history_messages()
        self.assertEqual(user, {"role": "user", "content": "hi", "toolCalls": []})
        self.assertEqual(assistant, {
            "role": "assistant",
            "content": "hello",
            "toolCalls": [],
            "charts": [{"chart_id": "c1"}],
            "artifacts": [{"id": "tbl_1", "type": "table", "title": "T", "row_count": 3, "metadata": {}}],
        })


class AppendTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_finite_floats_persist_as_null(self):
        row = ("t1", "u1", 0, "hi", "hello", [], [], [], None, "done", datetime.now(UTC), [], [])
        cursor = MagicMock()
        cursor.fetchone = AsyncMock(return_value=row)
        conn = MagicMock()
        conn.execute = AsyncMock(return_value=cursor)
        conn.commit = AsyncMock()

        turn = _turn(
            charts=[{"chart_id": "c1", "valid_max": float("inf")}],
            artifacts=[ArtifactReference(id="a", type="map", metadata={"v": float("nan")})],
        )
        with patch(_PATCH, _fake_pg_connection(conn)):
            await transcript_repository.append_turn(turn)

        params = conn.execute.await_args.args[1]
        charts_arg, artifacts_arg = params[4], params[5]
        json.dumps(charts_arg.obj, allow_nan=False)
        json.dumps(artifacts_arg.obj, allow_nan=False)
        self.assertIsNone(charts_arg.obj[0]["valid_max"])
        self.assertIsNone(artifacts_arg.obj[0]["metadata"]["v"])


if __name__ == "__main__":
    unittest.main()
