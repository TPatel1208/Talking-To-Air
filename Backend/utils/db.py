"""
utils/db.py
-----------
Shared asynchronous database connection helpers used by agents and request handlers.

Import from here instead of duplicating these functions:

    from utils.db import pg_connection, get_checkpointer
"""

import contextlib
import logging
from collections.abc import AsyncIterator

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer_conn: psycopg.AsyncConnection | None = None


def _db_config() -> dict:
    return get_settings().db_kwargs


def validate_config() -> None:
    """Fail early for configuration that would make startup unreliable."""
    get_settings().validate_startup()


async def init_db_pool() -> AsyncConnectionPool:
    """Create the shared asynchronous PostgreSQL connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        settings = get_settings()
        min_size = settings.db_pool_min_size
        max_size = settings.db_pool_max_size
        _pool = AsyncConnectionPool(
            kwargs=_db_config(),
            min_size=min_size,
            max_size=max_size,
            open=False,
        )
        await _pool.open()
        logger.info("database_reconnect", extra={"_min_size": min_size, "_max_size": max_size})
    return _pool


async def close_db_pool() -> None:
    """Close shared database resources on application shutdown."""
    global _pool, _checkpointer_conn
    if _checkpointer_conn is not None and not _checkpointer_conn.closed:
        try:
            if _pool is not None and not _pool.closed:
                await _pool.putconn(_checkpointer_conn)
            else:
                await _checkpointer_conn.close()
        finally:
            _checkpointer_conn = None
    if _pool is not None and not _pool.closed:
        await _pool.close()
    _pool = None


async def pg_connect(autocommit: bool = False) -> psycopg.AsyncConnection:
    """Return an open async psycopg connection built from environment variables."""
    config = _db_config()
    return await psycopg.AsyncConnection.connect(**config, autocommit=autocommit)


async def _rollback_if_in_transaction(conn: psycopg.AsyncConnection) -> None:
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        await conn.rollback()


@contextlib.asynccontextmanager
async def pg_connection(autocommit: bool = False) -> AsyncIterator[psycopg.AsyncConnection]:
    """
    Acquire an async connection from the shared pool when available.

    Command-line scripts can still use this before FastAPI startup; in that case
    it falls back to a short-lived direct async connection.
    """
    if _pool is None or _pool.closed:
        async with await pg_connect(autocommit=autocommit) as conn:
            yield conn
        return

    async with _pool.connection() as conn:
        await _rollback_if_in_transaction(conn)
        previous = conn.autocommit
        await conn.set_autocommit(autocommit)
        try:
            yield conn
            if not conn.autocommit:
                await conn.commit()
        except Exception:
            if not conn.autocommit:
                await conn.rollback()
            raise
        finally:
            await conn.set_autocommit(previous)


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Build and return an AsyncPostgresSaver for use as a LangGraph checkpointer.

    Opens a dedicated autocommit=True connection, calls async setup() to create
    checkpoint tables on first run, and returns the ready-to-use saver.

    The supervisor should call this once and share the returned instance with
    both sub-agents to avoid multiple connections racing on the same tables.
    """
    global _checkpointer_conn
    if _pool is None or _pool.closed:
        await init_db_pool()
    if _checkpointer_conn is None or _checkpointer_conn.closed:
        _checkpointer_conn = await _pool.getconn()
        await _checkpointer_conn.set_autocommit(True)
    conn = _checkpointer_conn
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return checkpointer
