"""T31: resolves a connected user's decrypted connector secret for per-call
injection -- the Earthdata token into the workspace-binding wrapper
(earthdata_mcp/workspace.py), the EPA AQS email + key into the AQS client
(tools/ground_sensor_tools/epa_aqs_tools.py).

A short-TTL in-process cache holds the *encrypted* row per user; decryption
stays just-in-time per call, so no decrypted material is cached anywhere.
Cache entries are invalidated explicitly on disconnect/re-paste (api.py's
connector endpoints) so a revoked or renewed secret is never injected stale
within the TTL window.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from typing import Any

from tta_backend.config.settings import Settings, get_settings
from tta_backend.repositories.user_connector_repository import (
    get_connector_secret_row,
    set_connector_status,
    touch_last_used_at,
)
from tta_backend.utils.connector_crypto import decrypt_secret, get_connector_cipher
from tta_backend.utils.streaming import get_call_budget

logger = logging.getLogger(__name__)

CONNECTOR_TYPE_EARTHDATA = "earthdata"
CONNECTOR_TYPE_EPA_AQS = "epa_aqs"
DEFAULT_CACHE_TTL_SECONDS = 30.0


class _ConnectorSecretInjector:
    """Resolves one connector type's secret per user. One instance per
    connector type is shared by every request -- the in-process cache and
    per-turn last_used_at coalescing are only meaningful as shared,
    process-lifetime state."""

    connector_type: str
    # Earthdata tokens carry an expiry and are unusable without one; AQS
    # keys never expire and are stored with expires_at NULL.
    expiry_required: bool
    decrypt_failed_event: str

    def __init__(self, settings: Settings, *, cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS):
        self._settings = settings
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        # The event loop holds only weak references to tasks, so an
        # unreferenced fire-and-forget write can be collected mid-flight.
        self._background_writes: set[asyncio.Task[None]] = set()

    def invalidate(self, user_id: str) -> None:
        """Called by api.py's set/disconnect endpoints so a re-paste or
        disconnect is never shadowed by a stale cached row within the TTL
        window."""
        self._cache.pop(user_id, None)

    async def _get_row(self, user_id: str) -> dict[str, Any] | None:
        cached = self._cache.get(user_id)
        now = time.monotonic()
        if cached is not None and now - cached[0] < self._cache_ttl:
            return cached[1]
        row = await get_connector_secret_row(user_id, self.connector_type)
        self._cache[user_id] = (now, row)
        return row

    async def _resolve_secret(self, user_id: str) -> str | None:
        """Injection policy: connected ∧ unexpired. Anything else returns
        None so the caller falls back to the shared app credential."""
        cipher = get_connector_cipher(self._settings)
        if cipher is None:
            return None
        row = await self._get_row(user_id)
        if row is None or row.get("status") != "connected":
            return None
        expires_at = row.get("expires_at")
        if expires_at is None:
            if self.expiry_required:
                return None
        elif expires_at <= datetime.now(timezone.utc):
            return None
        try:
            return decrypt_secret(cipher, row["encrypted_secret"])
        except Exception:
            logger.warning(self.decrypt_failed_event, extra={"_event": self.decrypt_failed_event})
            return None

    def mark_used(self, user_id: str) -> None:
        """Fire-and-forget, coalesced per agent turn and connector via the
        shared call-budget dict (utils.streaming.get_call_budget) -- several
        injected calls in one turn produce at most one last_used_at write
        per connector, and a failing write never fails the tool call that
        triggered it."""
        budget = get_call_budget()
        touched = budget.setdefault("connector_last_used_touched", set())
        touch_key = (self.connector_type, user_id)
        if touch_key in touched:
            return
        touched.add(touch_key)

        async def _write() -> None:
            try:
                await touch_last_used_at(user_id, self.connector_type)
            except Exception:
                logger.warning(
                    "connector_last_used_write_failed",
                    extra={"_event": "connector_last_used_write_failed"},
                )

        task = asyncio.create_task(_write())
        self._background_writes.add(task)
        task.add_done_callback(self._background_writes.discard)

    async def mark_invalid(self, user_id: str) -> None:
        self.invalidate(user_id)
        try:
            await set_connector_status(user_id, self.connector_type, "error")
        except Exception:
            logger.warning("connector_status_flip_failed", extra={"_event": "connector_status_flip_failed"})


class EdlCredentialInjector(_ConnectorSecretInjector):
    """The concrete implementation of earthdata_mcp.workspace's
    EdlCredentialInjector protocol. One instance is constructed alongside
    the earthdata MCP connection manager."""

    connector_type = CONNECTOR_TYPE_EARTHDATA
    expiry_required = True
    decrypt_failed_event = "edl_token_decrypt_failed"

    async def resolve(self, user_id: str) -> str | None:
        """Advertising is the caller's concern, checked in bind_workspace
        before this is even called; None means bind_workspace sends nothing
        and the MCP falls back to its shared env credential."""
        return await self._resolve_secret(user_id)


@dataclass(frozen=True)
class AqsCredentials:
    email: str
    key: str


class AqsCredentialInjector(_ConnectorSecretInjector):
    """Resolves a researcher's own EPA AQS email + key, stored as one JSON
    secret by api.py's PUT /connectors/epa_aqs/credentials."""

    connector_type = CONNECTOR_TYPE_EPA_AQS
    expiry_required = False
    decrypt_failed_event = "aqs_credentials_decrypt_failed"

    async def resolve_credentials(self, user_id: str) -> AqsCredentials | None:
        secret = await self._resolve_secret(user_id)
        if secret is None:
            return None
        try:
            parsed = json.loads(secret)
            email, key = parsed["email"], parsed["key"]
        except (ValueError, TypeError, KeyError):
            logger.warning("aqs_credentials_malformed", extra={"_event": "aqs_credentials_malformed"})
            return None
        if not isinstance(email, str) or not isinstance(key, str) or not email or not key:
            return None
        return AqsCredentials(email=email, key=key)


@cache
def get_aqs_credential_injector() -> AqsCredentialInjector:
    """The one process-wide instance, shared by the AQS client (which
    resolves through it) and api.py (which invalidates it)."""
    return AqsCredentialInjector(get_settings())
