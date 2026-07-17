from __future__ import annotations

import math
from typing import Any

from psycopg.types.json import Jsonb

from utils.db import pg_connection


def _sanitize_non_finite(value: Any) -> Any:
    """Clamp non-finite floats (inf/-inf/nan) to None, recursively.

    Postgres `jsonb` only accepts RFC-compliant JSON, but Python's serialiser
    happily emits the literal tokens ``Infinity``/``-Infinity``/``NaN`` -- the
    same class of crash chart_repository.save_chart already guards against
    (the HCHO ``valid_max: .inf`` regression). EPA/ground stat rows can carry
    the same kind of value (e.g. a divide-by-zero in a coverage/percentage
    column), so this is applied to rows and metadata before every INSERT.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(v) for v in value]
    return value


async def ensure_artifact_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_artifacts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                title TEXT NOT NULL,
                columns JSONB NOT NULL,
                rows JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                claimed_at TIMESTAMPTZ
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_artifacts_thread_created ON agent_artifacts (thread_id, created_at)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_artifacts_user_id ON agent_artifacts (user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_artifacts_unclaimed ON agent_artifacts (created_at) WHERE claimed_at IS NULL"
        )
        await conn.commit()


async def save_artifact(
    artifact_id: str,
    user_id: str,
    thread_id: str,
    title: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    sanitized_rows = _sanitize_non_finite(rows)
    sanitized_metadata = _sanitize_non_finite(metadata)
    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_artifacts (id, user_id, thread_id, title, columns, rows, metadata, claimed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (id) DO UPDATE
            SET user_id = EXCLUDED.user_id,
                thread_id = EXCLUDED.thread_id,
                title = EXCLUDED.title,
                columns = EXCLUDED.columns,
                rows = EXCLUDED.rows,
                metadata = EXCLUDED.metadata,
                claimed_at = COALESCE(agent_artifacts.claimed_at, EXCLUDED.claimed_at)
            """,
            (
                artifact_id, user_id, thread_id, title,
                Jsonb(columns), Jsonb(sanitized_rows), Jsonb(sanitized_metadata),
            ),
        )
        await conn.commit()


async def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT user_id, thread_id, title, columns, rows, metadata FROM agent_artifacts WHERE id = %s",
            (artifact_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "thread_id": row[1],
        "title": row[2],
        "columns": row[3],
        "rows": row[4],
        "metadata": row[5],
    }


async def delete_artifacts_for_session(thread_id: str, user_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            "DELETE FROM agent_artifacts WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()


async def delete_expired_unclaimed(ttl_seconds: int) -> None:
    """Defensive sweep: rows are only ever written at claim time (mint stays
    in-memory-only), so claimed_at is always set today. Guards against a
    future write path inserting an unclaimed row and never coming back."""
    async with pg_connection() as conn:
        await conn.execute(
            "DELETE FROM agent_artifacts WHERE claimed_at IS NULL AND created_at < now() - make_interval(secs => %s)",
            (ttl_seconds,),
        )
        await conn.commit()
