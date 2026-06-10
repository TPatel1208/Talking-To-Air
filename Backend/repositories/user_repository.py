from __future__ import annotations

import uuid

import psycopg
from psycopg.rows import dict_row

from models.user import User
from utils.db import pg_connection


async def ensure_user_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            )
            """
        )


def _row_to_user(row) -> User | None:
    return User(**row) if row else None


async def create_user(username: str, password_hash: str) -> User:
    async with pg_connection() as conn:
        try:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO users (id, username, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING id, username, password_hash, created_at, is_active
                    """,
                    (str(uuid.uuid4()), username, password_hash),
                )
                return User(**await cur.fetchone())
        except psycopg.errors.UniqueViolation:
            raise


async def get_user_by_username(username: str) -> User | None:
    async with pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, username, password_hash, created_at, is_active
                FROM users
                WHERE username = %s
                """,
                (username,),
            )
            return _row_to_user(await cur.fetchone())


async def get_user_by_id(user_id: str) -> User | None:
    async with pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, username, password_hash, created_at, is_active
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            return _row_to_user(await cur.fetchone())

