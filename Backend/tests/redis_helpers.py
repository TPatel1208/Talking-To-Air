"""The real Redis the turn tests run against, and when they may skip without it.

The event log, the turn registry and the routes built on them are tested
against a real Redis: what they pin is Redis's own behaviour (cursor
placement, MAXLEN trimming, SETNX with a TTL), which a fake would not
reproduce.

They skip when none is reachable so a host-side ``pytest`` still runs. Where a
Redis is provisioned -- CI's service container, the compose ``backend-test``
profile -- ``TTA_REQUIRE_REDIS=1`` turns that skip into a collection error, so
an unreachable Redis fails the run instead of dropping ~85 tests from it while
it still reports green.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import unittest
from urllib.parse import urlsplit

#: Database 15 by default, never 0 -- the live stack's turns are on 0.
DEFAULT_TEST_REDIS_URL = "redis://127.0.0.1:6379/15"
REDIS_URL = os.environ.get("REDIS_URL") or DEFAULT_TEST_REDIS_URL

#: Set where a Redis is provisioned, so its absence fails rather than skips.
REQUIRE_REDIS_ENV = "TTA_REQUIRE_REDIS"


def _redis_is_reachable(url: str) -> bool:
    if importlib.util.find_spec("redis") is None:
        return False
    parsed = urlsplit(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 6379), timeout=1.0):
            return True
    except OSError:
        return False


REDIS_REACHABLE = _redis_is_reachable(REDIS_URL)

if not REDIS_REACHABLE and os.environ.get(REQUIRE_REDIS_ENV) == "1":
    raise RuntimeError(
        f"{REQUIRE_REDIS_ENV}=1 but no Redis is reachable at {REDIS_URL} (or the "
        "redis package is not installed); refusing to skip the Redis-backed tests."
    )

requires_redis = unittest.skipUnless(REDIS_REACHABLE, f"no Redis reachable at {REDIS_URL}")
