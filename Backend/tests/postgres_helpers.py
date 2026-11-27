"""The real Postgres the integration tests run against, and when they may skip without it.

Each test gets its own throwaway schema in the test database, so tests never
see each other's tables and never touch ``public``.

They skip when no Postgres is reachable so a host-side ``pytest`` still runs.
Where one is provisioned -- CI's service container, the compose ``backend-test``
profile -- ``TTA_REQUIRE_POSTGRES=1`` turns that skip into a collection error.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import psycopg
from psycopg import sql

from tta_backend.harness.database import Connect

DEFAULT_TEST_DATABASE_URL = "postgresql://postgres:test@127.0.0.1:55432/tta_test"
TEST_DATABASE_URL = os.environ.get("TTA_TEST_DATABASE_URL") or DEFAULT_TEST_DATABASE_URL

#: Set where a Postgres is provisioned, so its absence fails rather than skips.
REQUIRE_POSTGRES_ENV = "TTA_REQUIRE_POSTGRES"


def _postgres_is_reachable(url: str) -> bool:
    parsed = urlsplit(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 5432), timeout=1.0):
            return True
    except OSError:
        return False


POSTGRES_REACHABLE = _postgres_is_reachable(TEST_DATABASE_URL)

if not POSTGRES_REACHABLE and os.environ.get(REQUIRE_POSTGRES_ENV) == "1":
    raise RuntimeError(
        f"{REQUIRE_POSTGRES_ENV}=1 but no Postgres is reachable at {TEST_DATABASE_URL}; "
        "refusing to skip the Postgres-backed tests."
    )

requires_postgres = unittest.skipUnless(POSTGRES_REACHABLE, f"no Postgres reachable at {TEST_DATABASE_URL}")


@requires_postgres
class PostgresTestCase(unittest.IsolatedAsyncioTestCase):
    """Base for Postgres-backed tests.

    psycopg's async mode cannot run on Windows' default Proactor loop, so the
    selector loop is used for this class only; the rest of the suite keeps the
    default loop, whose timers the timing tests depend on.
    """

    _previous_policy: asyncio.AbstractEventLoopPolicy | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if sys.platform == "win32":
            cls._previous_policy = asyncio.get_event_loop_policy()
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_policy is not None:
            asyncio.set_event_loop_policy(cls._previous_policy)
            cls._previous_policy = None
        super().tearDownClass()


@asynccontextmanager
async def isolated_schema() -> AsyncIterator[Connect]:
    """Yield a ``Connect`` whose connections are confined to a fresh schema.

    Connections commit on clean exit and roll back on error. The schema is
    dropped on the way out.
    """
    name = f"t_{uuid.uuid4().hex}"
    ident = sql.Identifier(name)

    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(ident))

    @asynccontextmanager
    async def connect() -> AsyncIterator[psycopg.AsyncConnection[Any]]:
        async with await psycopg.AsyncConnection.connect(
            TEST_DATABASE_URL, options=f"-c search_path={name}"
        ) as conn:
            yield conn

    try:
        yield connect
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(ident))
