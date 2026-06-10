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


async def delete_session_metadata(thread_id: str, user_id: str) -> bool:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM session_metadata WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
    return cursor.rowcount > 0
