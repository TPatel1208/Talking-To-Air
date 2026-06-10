"""
utils/db.py
-----------
Shared asynchronous database connection helpers used by agents and request handlers.

Import from here instead of duplicating these functions:

    from utils.db import pg_connection, get_checkpointer
"""

import contextlib
import asyncio
import logging
from collections.abc import AsyncIterator

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


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
    global _pool, _checkpointer_pool, _checkpointer
    _checkpointer = None
    if _checkpointer_pool is not None and not _checkpointer_pool.closed:
        await _checkpointer_pool.close()
        logger.info("checkpointer_pool_closed")
    _checkpointer_pool = None
    if _pool is not None and not _pool.closed:
        await _pool.close()
    _pool = None


async def check_db_pool(timeout_seconds: float = 2.0) -> tuple[bool, str | None]:
    """Return whether the shared pool can execute a trivial query."""
    if _pool is None or _pool.closed:
        return False, "database pool is not initialized"
    try:
        async def _probe() -> None:
            async with pg_connection(autocommit=True) as conn:
                await conn.execute("SELECT 1")

        await asyncio.wait_for(_probe(), timeout=timeout_seconds)
        return True, None
    except Exception as exc:
        return False, str(exc)


def active_pool_connections() -> int | None:
    """Best-effort active connection count for the shared request pool."""
    if _pool is None or _pool.closed:
        return 0
    try:
        stats = _pool.get_stats()
    except Exception:
        return None
    pool_size = stats.get("pool_size")
    available = stats.get("pool_available")
    if pool_size is None or available is None:
        return None
    return max(0, int(pool_size) - int(available))


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

    Uses a dedicated one-connection async pool instead of retaining a raw
    PostgreSQL connection. The pool owns reconnection behavior after transient
    database failures and prevents a stale module-level connection from
    permanently breaking checkpoint persistence.

    The supervisor should call this once and share the returned instance with
    both sub-agents to avoid multiple connections racing on the same tables.
    """
    global _checkpointer_pool, _checkpointer
    if _checkpointer is not None and _checkpointer_pool is not None and not _checkpointer_pool.closed:
        return _checkpointer

    if _checkpointer_pool is not None and _checkpointer_pool.closed:
        logger.warning("checkpointer_pool_closed_reinitializing")
        _checkpointer_pool = None
        _checkpointer = None

    if _checkpointer_pool is None:
        _checkpointer_pool = AsyncConnectionPool(
            kwargs={**_db_config(), "autocommit": True, "row_factory": dict_row},
            min_size=1,
            max_size=1,
            open=False,
            check=AsyncConnectionPool.check_connection,
        )
        await _checkpointer_pool.open()
        logger.info("checkpointer_pool_initialized", extra={"_min_size": 1, "_max_size": 1})

    # Use a local variable so a failed setup() never caches a broken instance.
    # Assigning to _checkpointer only after setup() succeeds means a subsequent
    # call will retry pool creation rather than returning a partially-initialized
    # saver.
    checkpointer = AsyncPostgresSaver(conn=_checkpointer_pool)
    try:
        await checkpointer.setup()
    except Exception:
        logger.exception("checkpointer_setup_failed")
        raise
    _checkpointer = checkpointer
    logger.info("checkpointer_initialized")
    return _checkpointer
