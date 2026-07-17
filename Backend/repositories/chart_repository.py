from __future__ import annotations

import math
import uuid
import json
from typing import Any

from psycopg.types.json import Jsonb

from utils.db import pg_connection


def _sanitize_non_finite(value: Any) -> Any:
    """Clamp non-finite floats (inf/-inf/nan) to None, recursively.

    Postgres `json`/`jsonb` only accepts RFC-compliant JSON, but Python's
    serialiser happily emits the literal tokens ``Infinity``/``-Infinity``/
    ``NaN`` — so any payload carrying a non-finite float (e.g. the
    ``valid_max: .inf`` "no upper bound" sentinel collections.yaml pins for
    the HCHO products) crashed the INSERT with ``invalid input syntax for
    type json``. This is the single persistence boundary for chart payloads,
    so sanitising here protects every current and future payload shape.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(v) for v in value]
    return value


async def save_chart(thread_id: str, payload: dict[str, Any], user_id: str) -> dict[str, Any]:
    # Sanitise before the id hash below too, so a payload that differs only
    # by inf-vs-null dedupes to the same chart_id.
    stored_payload = _sanitize_non_finite(dict(payload))
    # Callers that already minted an id up front (T06 artifact-typed plot
    # payloads, so the id is stable and visible to the LLM before this ever
    # persists) win over the content-hash id generic chart payloads get.
    chart_id = stored_payload.get("chart_id")
    if not chart_id:
        stable_payload = json.dumps(
            {k: v for k, v in stored_payload.items() if k not in {"chart_id", "thread_id", "user_id"}},
            sort_keys=True,
            separators=(",", ":"),
        )
        chart_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{thread_id}:{stable_payload}"))
    stored_payload["chart_id"] = chart_id
    stored_payload["thread_id"] = thread_id
    stored_payload["user_id"] = user_id
    metadata = stored_payload.get("metadata") or {}

    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_charts (id, thread_id, user_id, payload, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET payload = EXCLUDED.payload,
                metadata = EXCLUDED.metadata,
                user_id = EXCLUDED.user_id
            """,
            (chart_id, thread_id, user_id, Jsonb(stored_payload), Jsonb(metadata)),
        )
        await conn.commit()

    return stored_payload


async def get_chart(chart_id: str) -> dict[str, Any] | None:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT payload, thread_id, user_id FROM agent_charts WHERE id = %s",
            (chart_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    payload = dict(row[0])
    payload.setdefault("chart_id", chart_id)
    payload["thread_id"] = row[1]
    payload["user_id"] = row[2]
    return payload


async def delete_charts_for_session(thread_id: str, user_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            "DELETE FROM agent_charts WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
