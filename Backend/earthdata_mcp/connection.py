"""
earthdata_mcp/connection.py
=============================
Owns the earthdata-retrieval MCP connection as an attached service, not a
boot dependency (PRD T17): a background task connects (and reconnects, with
capped exponential backoff) while the rest of the backend boots and serves
ground/EPA traffic immediately. Every consumer that needs MCP tools reads
them through this manager instead of touching earthdata_mcp.client directly,
so "is the data layer usable right now" has exactly one answer.

States
------
connecting    -- boot or a reconnect attempt is in progress; no tools yet.
ready         -- tools loaded, present, and contract-checked; ``.tools``
                 returns them.
unavailable   -- the MCP could not be reached, or was missing a required
                 tool; a retry is scheduled.
incompatible  -- the MCP answered and every required tool is present, but at
                 least one tool's advertised schema is missing a parameter
                 this backend sends. A retry is still scheduled (a redeploy
                 may fix it), and the mismatch is logged at CRITICAL.

The loop never stops watching once ready: it re-verifies on a heartbeat
interval, so a mid-session MCP outage (not just a down-at-boot one) flips
the state away from ready — otherwise ``/health`` would keep lying "ready"
and every consumer gated on ``.state`` (discovery/jobs/provenance endpoints,
run_satellite) would let a bare connection failure through instead of the
structured unavailable answer. ``on_ready`` fires only on a genuine
transition into ready (boot or recovery), never on a heartbeat that finds
the MCP still healthy, so a steady-state loop doesn't rebuild the satellite
agent every interval for nothing.

Boot-time misconfiguration (a malformed URL, a missing required setting) is
not this module's concern — that still fails loudly from
config.settings.Settings.validate_startup before the manager ever starts.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool

from config.settings import Settings
from earthdata_mcp.client import REQUIRED_TOOL_PARAMS, EarthdataMCPUnavailableError, load_raw_mcp_tools
from earthdata_mcp.workspace import bind_workspace

logger = logging.getLogger(__name__)

STATE_CONNECTING = "connecting"
STATE_READY = "ready"
STATE_UNAVAILABLE = "unavailable"
STATE_INCOMPATIBLE = "incompatible"

Loader = Callable[[Settings], Awaitable[dict[str, BaseTool]]]
OnReady = Callable[[dict[str, BaseTool]], Awaitable[None]]
Sleeper = Callable[[float], Awaitable[None]]


class EarthdataMCPNotReadyError(RuntimeError):
    """Raised by ``.tools`` when the manager is not in the ready state."""


def check_tool_schemas(tools: dict[str, BaseTool]) -> dict[str, list[str]]:
    """Return ``{tool_name: [missing_param, ...]}`` for every required tool
    whose advertised input schema (``tool.args``) is missing a parameter this
    backend sends (``REQUIRED_TOOL_PARAMS``). Empty when every present
    required tool's contract holds. A tool absent from ``tools`` entirely is
    not reported here — presence is the separate missing-tools verdict
    ``load_raw_mcp_tools`` already raises on.
    """
    mismatches: dict[str, list[str]] = {}
    for name, sent_params in REQUIRED_TOOL_PARAMS.items():
        tool = tools.get(name)
        if tool is None:
            continue
        schema_params = set(tool.args.keys())
        missing = [param for param in sent_params if param not in schema_params]
        if missing:
            mismatches[name] = missing
    return mismatches


class EarthdataMCPConnectionManager:
    """Background connect/reconnect loop for the earthdata-retrieval MCP.

    ``loader``/``sleep`` are injectable seams for tests (a fake client that
    fails N times then succeeds, and a no-op sleep so retry tests don't
    actually wait out the backoff) — production callers rely on the
    defaults (``load_raw_mcp_tools``, ``asyncio.sleep``).
    """

    def __init__(
        self,
        settings: Settings,
        user_id_getter: Callable[[], str],
        *,
        on_ready: OnReady | None = None,
        loader: Loader = load_raw_mcp_tools,
        sleep: Sleeper = asyncio.sleep,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 30.0,
    ):
        self._settings = settings
        self._user_id_getter = user_id_getter
        self._on_ready = on_ready
        self._loader = loader
        self._sleep = sleep
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._state = STATE_CONNECTING
        self._tools: dict[str, BaseTool] | None = None
        self._task: asyncio.Task | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def tools(self) -> dict[str, BaseTool]:
        """Workspace-bound tools, once ready. Raises otherwise — callers are
        expected to gate on ``.state`` first; this exists so a programming
        error surfaces loudly instead of silently handing out a stale or
        empty dict."""
        if self._state != STATE_READY or self._tools is None:
            raise EarthdataMCPNotReadyError(f"earthdata-retrieval MCP is not ready (state={self._state})")
        return self._tools

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._connect_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _connect_loop(self) -> None:
        """Runs for the manager's whole lifetime (until ``stop()`` cancels
        it) — connect/reconnect with backoff while not ready, then heartbeat
        on ``_heartbeat_interval`` while ready so a later outage is detected
        too, not just a down-at-boot one."""
        backoff = self._initial_backoff
        while True:
            try:
                raw = await self._loader(self._settings)
            except EarthdataMCPUnavailableError as exc:
                self._transition(STATE_UNAVAILABLE, detail=str(exc))
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue

            mismatches = check_tool_schemas(raw)
            if mismatches:
                self._transition(STATE_INCOMPATIBLE, detail=mismatches)
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue

            tools = bind_workspace(raw, self._user_id_getter)
            # on_ready runs BEFORE the state flips to ready, so no caller can
            # ever observe state == ready with a stale/empty consumer (e.g.
            # api.py's satellite agent) still in place. It fires only on a
            # genuine transition into ready (boot or recovery) — not on a
            # heartbeat that finds an already-ready MCP still healthy.
            if self._state != STATE_READY and self._on_ready is not None:
                await self._on_ready(tools)
            self._tools = tools
            self._transition(STATE_READY)
            backoff = self._initial_backoff
            await self._sleep(self._heartbeat_interval)

    def _transition(self, new_state: str, *, detail: Any = None) -> None:
        old_state = self._state
        self._state = new_state
        if old_state == new_state:
            return
        logger.warning(
            "earthdata_mcp_state_transition",
            extra={"_event": "earthdata_mcp_state_transition", "_from": old_state, "_to": new_state},
        )
        if new_state == STATE_INCOMPATIBLE:
            logger.critical(
                "earthdata_mcp_schema_mismatch",
                extra={"_event": "earthdata_mcp_schema_mismatch", "_mismatches": detail},
            )
