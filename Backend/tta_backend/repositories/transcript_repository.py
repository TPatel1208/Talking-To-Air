from __future__ import annotations

import math
from typing import Any

from psycopg.types.json import Jsonb

from tta_backend.harness.results import Mode
from tta_backend.models.transcript import TranscriptResult, TranscriptTurn
from tta_backend.utils.db import pg_connection

_COLUMNS = (
    "thread_id, user_id, turn_index, user_text, answer_text, "
    "charts, artifacts, sources, label, status, created_at, results, suggested_followups"
)


def _sanitize_non_finite(value: Any) -> Any:
    """Clamp non-finite floats to None, recursively; jsonb rejects them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(v) for v in value]
    return value


def _from_row(row: Any) -> TranscriptTurn:
    return TranscriptTurn(
        thread_id=row[0],
        user_id=row[1],
        turn_index=row[2],
        user_text=row[3],
        answer_text=row[4],
        charts=row[5],
        artifacts=row[6],
        sources=row[7],
        label=Mode.model_validate(row[8]) if row[8] is not None else None,
        status=row[9],
        created_at=row[10],
        results=[TranscriptResult.model_validate(r) for r in row[11]],
        suggested_followups=row[12],
    )


async def ensure_transcript_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                user_text TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                charts JSONB NOT NULL DEFAULT '[]'::jsonb,
                artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                label JSONB,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                results JSONB NOT NULL DEFAULT '[]'::jsonb,
                suggested_followups JSONB NOT NULL DEFAULT '[]'::jsonb,
                PRIMARY KEY (user_id, thread_id, turn_index)
            )
            """
        )
        # Tables made before these columns existed gain them empty.
        for column in ("results", "suggested_followups"):
            await conn.execute(
                f"ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS {column} JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_user_id ON conversation_turns (user_id)"
        )
        await conn.commit()


async def append_turn(turn: TranscriptTurn) -> TranscriptTurn:
    """Store a turn at the thread's next index and return it as stored.

    Indexes are per user and thread. The advisory lock serialises appends to
    one user's thread, so racing writers get distinct consecutive indexes
    instead of a primary-key conflict.
    """
    charts = _sanitize_non_finite(turn.charts)
    artifacts = _sanitize_non_finite([a.model_dump(exclude_none=True) for a in turn.artifacts])
    label = _sanitize_non_finite(turn.label.model_dump(mode="json")) if turn.label else None
    results = [r.model_dump(mode="json") for r in turn.results]
    async with pg_connection() as conn:
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{len(turn.user_id)}:{turn.user_id}:{turn.thread_id}",))
        cursor = await conn.execute(
            f"""
            INSERT INTO conversation_turns
                (thread_id, turn_index, user_id, user_text, answer_text, charts, artifacts, sources, label, status,
                 created_at, results, suggested_followups)
            SELECT %s, COALESCE(MAX(turn_index) + 1, 0), %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()),
                %s, %s
            FROM conversation_turns WHERE user_id = %s AND thread_id = %s
            RETURNING {_COLUMNS}
            """,
            (
                turn.thread_id, turn.user_id, turn.user_text, turn.answer_text,
                Jsonb(charts), Jsonb(artifacts), Jsonb(list(turn.sources)),
                Jsonb(label) if label is not None else None,
                turn.status, turn.created_at, Jsonb(results), Jsonb(list(turn.suggested_followups)),
                turn.user_id, turn.thread_id,
            ),
        )
        row = await cursor.fetchone()
        await conn.commit()
    if row is None:
        raise RuntimeError("transcript insert returned no row")
    return _from_row(row)


async def list_turns(thread_id: str, user_id: str) -> list[TranscriptTurn]:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            f"SELECT {_COLUMNS} FROM conversation_turns "
            "WHERE thread_id = %s AND user_id = %s ORDER BY turn_index",
            (thread_id, user_id),
        )
        rows = await cursor.fetchall()
    return [_from_row(row) for row in rows]


async def delete_transcript(thread_id: str, user_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            "DELETE FROM conversation_turns WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
