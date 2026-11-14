"""Database access the host hands the harness."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import psycopg

# Yields a connection that commits on clean exit and rolls back on error.
Connect = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection[Any]]]
