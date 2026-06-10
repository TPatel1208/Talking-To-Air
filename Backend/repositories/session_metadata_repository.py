from __future__ import annotations

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
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.commit()


async def save_session_metadata_once(thread_id: str, first_message: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO session_metadata (thread_id, title, created_at)
            VALUES (%s, %s, now())
            ON CONFLICT (thread_id) DO NOTHING
            """,
            (thread_id, generate_session_title(first_message)),
        )
        await conn.commit()


async def list_session_metadata() -> list[dict[str, Any]]:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            """
            WITH session_ids AS (
                SELECT DISTINCT thread_id FROM checkpoints
                UNION
                SELECT thread_id FROM session_metadata
            )
            SELECT s.thread_id, m.title, m.created_at
            FROM session_ids s
            LEFT JOIN session_metadata m ON m.thread_id = s.thread_id
            ORDER BY m.created_at DESC NULLS LAST, s.thread_id
            """
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


async def delete_session_metadata(thread_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute("DELETE FROM session_metadata WHERE thread_id = %s", (thread_id,))
        await conn.commit()
