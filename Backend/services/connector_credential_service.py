"""T31: resolves a connected user's decrypted Earthdata token for per-call
injection into the workspace-binding wrapper (earthdata_mcp/workspace.py) --
the single seam every model-facing and composite MCP call already passes
through to acquire caller context (T19/T26 prior art).

A short-TTL in-process cache holds the *encrypted* row per user; decryption
stays just-in-time per call, so no decrypted material is cached anywhere.
Cache entries are invalidated explicitly on disconnect/re-paste (api.py's
connector endpoints) so a revoked or renewed token is never injected stale
within the TTL window.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from repositories.user_connector_repository import (
    get_connector_secret_row,
    set_connector_status,
    touch_last_used_at,
)
from utils.connector_crypto import decrypt_secret, get_connector_cipher
from utils.streaming import get_call_budget

logger = logging.getLogger(__name__)

CONNECTOR_TYPE_EARTHDATA = "earthdata"
DEFAULT_CACHE_TTL_SECONDS = 30.0


class EdlCredentialInjector:
    """The concrete implementation of earthdata_mcp.workspace's
    EdlCredentialInjector protocol. One instance is constructed alongside
    the earthdata MCP connection manager and shared by every request --
    the in-process cache and per-turn last_used_at coalescing are only
    meaningful as shared, process-lifetime state."""

    def __init__(self, settings: Settings, *, cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS):
        self._settings = settings
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

    def invalidate(self, user_id: str) -> None:
        """Called by api.py's set-token/disconnect endpoints so a re-paste
        or disconnect is never shadowed by a stale cached row within the
        TTL window."""
        self._cache.pop(user_id, None)

    async def _get_row(self, user_id: str) -> dict[str, Any] | None:
        cached = self._cache.get(user_id)
        now = time.monotonic()
        if cached is not None and now - cached[0] < self._cache_ttl:
            return cached[1]
        row = await get_connector_secret_row(user_id, CONNECTOR_TYPE_EARTHDATA)
        self._cache[user_id] = (now, row)
        return row

    async def resolve(self, user_id: str) -> str | None:
        """Injection policy: connected ∧ unexpired ∧ (advertising is the
        caller's concern, checked in bind_workspace before this is even
        called). Anything else returns None so bind_workspace sends
        nothing and the MCP falls back to its shared env credential."""
        cipher = get_connector_cipher(self._settings)
        if cipher is None:
            return None
        row = await self._get_row(user_id)
        if row is None or row.get("status") != "connected":
            return None
        expires_at = row.get("expires_at")
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            return None
        try:
            return decrypt_secret(cipher, row["encrypted_secret"])
        except Exception:
            logger.warning("edl_token_decrypt_failed", extra={"_event": "edl_token_decrypt_failed"})
            return None

    def mark_used(self, user_id: str) -> None:
        """Fire-and-forget, coalesced per agent turn via the shared
        call-budget dict (utils.streaming.get_call_budget) -- several
        injected calls in one turn produce at most one last_used_at write,
        and a failing write never fails the tool call that triggered it."""
        budget = get_call_budget()
        touched = budget.setdefault("connector_last_used_touched", set())
        if user_id in touched:
            return
        touched.add(user_id)

        async def _write() -> None:
            try:
                await touch_last_used_at(user_id, CONNECTOR_TYPE_EARTHDATA)
            except Exception:
                logger.warning(
                    "connector_last_used_write_failed",
                    extra={"_event": "connector_last_used_write_failed"},
                )

        asyncio.create_task(_write())

    async def mark_invalid(self, user_id: str) -> None:
        self.invalidate(user_id)
        try:
            await set_connector_status(user_id, CONNECTOR_TYPE_EARTHDATA, "error")
        except Exception:
            logger.warning("connector_status_flip_failed", extra={"_event": "connector_status_flip_failed"})
