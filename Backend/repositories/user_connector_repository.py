from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from utils.db import pg_connection

_STATUS_VIEW_COLUMNS = "connector_type, auth_method, expires_at, status, connected_at, last_used_at"


async def ensure_user_connector_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_connectors (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                connector_type TEXT NOT NULL,
                auth_method TEXT NOT NULL,
                encrypted_secret TEXT NOT NULL,
                expires_at TIMESTAMPTZ,
                status TEXT NOT NULL DEFAULT 'connected',
                connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_used_at TIMESTAMPTZ,
                UNIQUE (user_id, connector_type)
            )
            """
        )


async def upsert_connector(
    user_id: str,
    connector_type: str,
    auth_method: str,
    encrypted_secret: str,
    expires_at: datetime,
    status: str = "connected",
) -> dict[str, Any]:
    """Insert or replace the caller's row for this connector type -- re-paste
    is one action (upsert), not disconnect-then-reconnect. Never selects or
    returns encrypted_secret; the response callers build from this can't leak
    what it never fetched."""
    async with pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                INSERT INTO user_connectors
                    (id, user_id, connector_type, auth_method, encrypted_secret, expires_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, connector_type) DO UPDATE
                SET auth_method = EXCLUDED.auth_method,
                    encrypted_secret = EXCLUDED.encrypted_secret,
                    expires_at = EXCLUDED.expires_at,
                    status = EXCLUDED.status
                RETURNING {_STATUS_VIEW_COLUMNS}
                """,
                (str(uuid.uuid4()), user_id, connector_type, auth_method, encrypted_secret, expires_at, status),
            )
            return await cur.fetchone()


async def list_connectors_for_user(user_id: str) -> list[dict[str, Any]]:
    async with pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"SELECT {_STATUS_VIEW_COLUMNS} FROM user_connectors WHERE user_id = %s",
                (user_id,),
            )
            return await cur.fetchall()


async def delete_connector(user_id: str, connector_type: str) -> bool:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM user_connectors WHERE user_id = %s AND connector_type = %s",
            (user_id, connector_type),
        )
        return cursor.rowcount > 0


async def get_connector_secret_row(user_id: str, connector_type: str) -> dict[str, Any] | None:
    """T31: the one read that includes ``encrypted_secret`` -- used only by
    services/connector_credential_service.py to resolve a per-call injection
    candidate (earthdata_mcp/workspace.py). Never used to build an API
    response; callers decrypt just-in-time and never persist the plaintext."""
    async with pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT encrypted_secret, expires_at, status FROM user_connectors "
                "WHERE user_id = %s AND connector_type = %s",
                (user_id, connector_type),
            )
            return await cur.fetchone()


async def touch_last_used_at(user_id: str, connector_type: str) -> None:
    """T31: bumped fire-and-forget by services/connector_credential_service.py
    after a successful call that actually injected this connector's token."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE user_connectors SET last_used_at = NOW() WHERE user_id = %s AND connector_type = %s",
            (user_id, connector_type),
        )


async def set_connector_status(user_id: str, connector_type: str, status: str) -> None:
    """T31: flips status (e.g. to ``error`` on a classified TOKEN_INVALID
    against an injected call) so the Connectors tab agrees with a failure
    the researcher just saw in chat."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE user_connectors SET status = %s WHERE user_id = %s AND connector_type = %s",
            (status, user_id, connector_type),
        )
