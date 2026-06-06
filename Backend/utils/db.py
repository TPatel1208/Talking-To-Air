"""
utils/db.py
-----------
Shared database connection helpers used by every agent and the cache layer.

Import from here instead of duplicating these functions:

    from utils.db import pg_connect, get_checkpointer
"""

import contextlib
import logging
from collections.abc import Iterator

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_checkpointer_conn: psycopg.Connection | None = None


def _db_config() -> dict:
    return get_settings().db_kwargs


def validate_config() -> None:
    """Fail early for configuration that would make startup unreliable."""
    get_settings().validate_startup()


def init_db_pool() -> ConnectionPool:
    """Create the shared PostgreSQL connection pool used by request handlers."""
    global _pool
    if _pool is None or _pool.closed:
        settings = get_settings()
        min_size = settings.db_pool_min_size
        max_size = settings.db_pool_max_size
        _pool = ConnectionPool(
            kwargs=_db_config(),
            min_size=min_size,
            max_size=max_size,
            open=True,
        )
        logger.info("database_reconnect", extra={"_min_size": min_size, "_max_size": max_size})
    return _pool


def close_db_pool() -> None:
    """Close shared database resources on application shutdown."""
    global _pool, _checkpointer_conn
    if _checkpointer_conn is not None and not _checkpointer_conn.closed:
        try:
            if _pool is not None and not _pool.closed:
                _pool.putconn(_checkpointer_conn)
            else:
                _checkpointer_conn.close()
        finally:
            _checkpointer_conn = None
    if _pool is not None and not _pool.closed:
        _pool.close()
    _pool = None


def pg_connect(autocommit: bool = False) -> psycopg.Connection:
    """
    Return an open psycopg (v3) connection built from environment variables.

    Reads:  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    Falls back to sensible defaults for all except DB_PASSWORD.

    Parameters
    ----------
    autocommit : bool
        When True the connection operates in autocommit mode, which is
        required by PostgresSaver (LangGraph checkpoint backend).
    """
    config = _db_config()
    return psycopg.connect(**config, autocommit=autocommit)


def _rollback_if_in_transaction(conn: psycopg.Connection) -> None:
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        conn.rollback()


@contextlib.contextmanager
def pg_connection(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """
    Acquire a connection from the shared pool when available.

    Command-line scripts can still use this before FastAPI startup; in that
    case it falls back to a short-lived direct connection.
    """
    if _pool is None or _pool.closed:
        with pg_connect(autocommit=autocommit) as conn:
            yield conn
        return

    with _pool.connection() as conn:
        _rollback_if_in_transaction(conn)
        previous = conn.autocommit
        conn.autocommit = autocommit
        try:
            yield conn
            if not conn.autocommit:
                conn.commit()
        except Exception:
            if not conn.autocommit:
                conn.rollback()
            raise
        finally:
            conn.autocommit = previous


def get_checkpointer() -> PostgresSaver:
    """
    Build and return a PostgresSaver for use as a LangGraph checkpointer.

    Opens a dedicated autocommit=True connection (required by PostgresSaver),
    calls setup() to create checkpoint tables on first run (no-op thereafter),
    and returns the ready-to-use saver.

    The supervisor should call this once and share the returned instance with
    both sub-agents to avoid multiple connections racing on the same tables.
    """
    global _checkpointer_conn
    if _pool is None or _pool.closed:
        init_db_pool()
    if _checkpointer_conn is None or _checkpointer_conn.closed:
        _checkpointer_conn = _pool.getconn()
        _checkpointer_conn.autocommit = True
    conn = _checkpointer_conn
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer
