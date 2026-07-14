from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from utils.db import pg_connection

MAX_TITLE_LENGTH = 60


def generate_session_title(message: str) -> str:
    title = re.sub(r"\s+", " ", message or "").strip()
    if not title:
        return "Untitled session"
    if len(title) <= MAX_TITLE_LENGTH:
        return title
    return title[: MAX_TITLE_LENGTH - 3].rstrip() + "..."


def _serialize_created_at(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else None


async def ensure_session_metadata_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_id TEXT NOT NULL DEFAULT '__legacy__'
            )
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '__legacy__'
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_metadata_user_id
            ON session_metadata(user_id)
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS ground_monitor_context JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS satellite_context JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        )
        await conn.commit()


async def save_session_metadata_once(thread_id: str, first_message: str, user_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO session_metadata (thread_id, title, created_at, user_id)
            VALUES (%s, %s, now(), %s)
            ON CONFLICT (thread_id) DO NOTHING
            """,
            (thread_id, generate_session_title(first_message), user_id),
        )
        await conn.commit()


async def get_session_metadata(thread_id: str) -> dict[str, Any] | None:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT thread_id, title, created_at, user_id
            FROM session_metadata
            WHERE thread_id = %s
            """,
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "created_at": _serialize_created_at(row[2]),
        "user_id": row[3],
    }


async def session_belongs_to_user(thread_id: str, user_id: str) -> bool:
    metadata = await get_session_metadata(thread_id)
    return metadata is not None and metadata["user_id"] == user_id


async def list_session_metadata(user_id: str) -> list[dict[str, Any]]:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT thread_id, title, created_at
            FROM session_metadata
            WHERE user_id = %s
            ORDER BY created_at DESC NULLS LAST, thread_id
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": row[0],
            "title": row[1],
            "created_at": _serialize_created_at(row[2]),
        }
        for row in rows
    ]


async def get_ground_monitor_context(thread_id: str) -> dict[str, str]:
    """The ground path's cross-turn monitor context (last monitor discussed
    on this thread) — per-thread, not process-wide, so concurrent
    conversations never bleed into each other (T14)."""
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT ground_monitor_context FROM session_metadata WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    return dict(row[0])


async def save_ground_monitor_context(thread_id: str, context: dict[str, str]) -> None:
    """Best-effort — a thread with no session_metadata row yet (should not
    happen on the normal chat flow, which always saves metadata first)
    simply does not persist the context."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE session_metadata SET ground_monitor_context = %s WHERE thread_id = %s",
            (json.dumps(context), thread_id),
        )
        await conn.commit()


async def get_satellite_context(thread_id: str) -> dict[str, str]:
    """The satellite path's cross-turn retrieval context for this thread — the
    dataset/AOI last worked with and the handles minted for them. Per-thread,
    so concurrent conversations never bleed into each other (mirrors
    ``get_ground_monitor_context``). Injected into a follow-up earthdata task
    so a continuation ("pick a date in that range") arrives with the concrete
    dataset/location/handles instead of only the prose answer the fast path
    wrote back — it is never treated as an availability verdict (the earthdata
    agent re-checks coverage; see its Availability-must-be-tool-grounded rule)."""
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT satellite_context FROM session_metadata WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    return dict(row[0])


async def save_satellite_context(thread_id: str, context: dict[str, str]) -> None:
    """Best-effort — a thread with no session_metadata row yet (should not
    happen on the normal chat flow, which always saves metadata first)
    simply does not persist the context."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE session_metadata SET satellite_context = %s WHERE thread_id = %s",
            (json.dumps(context), thread_id),
        )
        await conn.commit()


async def delete_session_metadata(thread_id: str, user_id: str) -> bool:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM session_metadata WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
    return cursor.rowcount > 0
