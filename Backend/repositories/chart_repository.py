from __future__ import annotations

import uuid
import json
from typing import Any

from psycopg.types.json import Jsonb

from utils.db import pg_connection


async def ensure_chart_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_charts (
                id UUID PRIMARY KEY,
                thread_id TEXT NOT NULL,
                payload JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_charts_thread_created
            ON agent_charts (thread_id, created_at)
            """
        )
        await conn.commit()


async def save_chart(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    await ensure_chart_table()
    stored_payload = dict(payload)
    chart_id = stored_payload.get("chart_id")
    if not chart_id:
        stable_payload = json.dumps(stored_payload, sort_keys=True, separators=(",", ":"))
        chart_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{thread_id}:{stable_payload}"))
    stored_payload["chart_id"] = chart_id
    metadata = stored_payload.get("metadata") or {}

    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_charts (id, thread_id, payload, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET payload = EXCLUDED.payload,
                metadata = EXCLUDED.metadata
            """,
            (chart_id, thread_id, Jsonb(stored_payload), Jsonb(metadata)),
        )
        await conn.commit()

    return stored_payload


async def get_chart(chart_id: str) -> dict[str, Any] | None:
    await ensure_chart_table()
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT payload FROM agent_charts WHERE id = %s",
            (chart_id,),
        )
        row = await cursor.fetchone()
    return row[0] if row else None


async def delete_charts_for_session(thread_id: str) -> None:
    await ensure_chart_table()
    async with pg_connection() as conn:
        await conn.execute("DELETE FROM agent_charts WHERE thread_id = %s", (thread_id,))
        await conn.commit()
