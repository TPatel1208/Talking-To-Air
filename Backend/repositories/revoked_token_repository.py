from __future__ import annotations

from datetime import datetime, timezone

from utils.db import pg_connection


async def ensure_revoked_token_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                revoked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        await conn.execute("DELETE FROM revoked_tokens WHERE expires_at < now()")
        await conn.commit()


async def revoke_token(jti: str, expires_at: datetime) -> None:
    async with pg_connection() as conn:
        await conn.execute("DELETE FROM revoked_tokens WHERE expires_at < now()")
        await conn.execute(
            """
            INSERT INTO revoked_tokens (jti, expires_at)
            VALUES (%s, %s)
            ON CONFLICT (jti) DO NOTHING
            """,
            (jti, expires_at.astimezone(timezone.utc)),
        )
        await conn.commit()


async def is_token_revoked(jti: str) -> bool:
    async with pg_connection() as conn:
        await conn.execute("DELETE FROM revoked_tokens WHERE expires_at < now()")
        cursor = await conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = %s",
            (jti,),
        )
        row = await cursor.fetchone()
        await conn.commit()
    return row is not None
