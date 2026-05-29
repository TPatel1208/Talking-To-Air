"""
utils/db.py
-----------
Shared database connection helpers used by every agent and the cache layer.

Import from here instead of duplicating these functions:

    from utils.db import pg_connect, get_checkpointer
"""

import logging
import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


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
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "talking_to_air_memory"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        autocommit=autocommit,
    )


def get_checkpointer() -> PostgresSaver:
    """
    Build and return a PostgresSaver for use as a LangGraph checkpointer.

    Opens a dedicated autocommit=True connection (required by PostgresSaver),
    calls setup() to create checkpoint tables on first run (no-op thereafter),
    and returns the ready-to-use saver.

    The supervisor should call this once and share the returned instance with
    both sub-agents to avoid multiple connections racing on the same tables.
    """
    conn = pg_connect(autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer
