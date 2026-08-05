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

Steady-state heartbeats are dampened (PRD T40): a single probe miss logs
and re-probes shortly after instead of demoting, so one transient blip
doesn't flap every consumer to unavailable for 30 seconds. Only
``heartbeat_failure_threshold`` *consecutive* misses demotes to
unavailable and falls back to the full connect loop above. The steady-state
probe itself (``earthdata_mcp.client.probe_mcp_tools`` by default) is a
tool listing, not a full load-bind-verify — no rebind, no ``on_ready`` — so
a healthy connection isn't rebuilt from scratch every 30 seconds for
nothing.

Boot-time misconfiguration (a malformed URL, a missing required setting) is
not this module's concern — that still fails loudly from
config.settings.Settings.validate_startup before the manager ever starts.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from langchain_core.tools import BaseTool

from tta_backend.config.settings import Settings
from tta_backend.earthdata_mcp.client import (
    REQUIRED_TOOL_PARAMS,
    EarthdataMCPUnavailableError,
    HeldSession,
    open_earthdata_session,
    probe_mcp_tools,
)
from tta_backend.earthdata_mcp.workspace import EdlCredentialInjector, bind_workspace

logger = logging.getLogger(__name__)

STATE_CONNECTING = "connecting"
STATE_READY = "ready"
STATE_UNAVAILABLE = "unavailable"
STATE_INCOMPATIBLE = "incompatible"

# A plain, non-session-held loader (the legacy test seam): connect, list, and
# return the tools, holding nothing open. Wrapped into a SessionScope below.
Loader = Callable[[Settings], Awaitable[dict[str, BaseTool]]]
# The production seam (default ``open_earthdata_session``): an async context
# manager that opens ONE MCP session, yields a ``HeldSession`` (its bound tools
# + a liveness probe on that session), and holds the session open for the whole
# ``async with`` body — so the tools the manager hands out reuse that one
# connection instead of reconnecting per call.
SessionScope = Callable[[Settings], AbstractAsyncContextManager[HeldSession]]
Probe = Callable[[Settings], Awaitable[Any]]
OnReady = Callable[[dict[str, BaseTool]], Awaitable[None]]
Sleeper = Callable[[float], Awaitable[None]]


def _loader_as_scope(loader: Loader) -> SessionScope:
    """Adapt a plain ``loader`` (legacy test seam) into a ``SessionScope`` that
    holds nothing open — it just loads once and yields a ``HeldSession`` with no
    liveness probe (``probe=None``), so the manager falls back to its injected
    fresh-session probe. Lets every existing hermetic connection test keep
    injecting a plain ``async def loader`` while the production path uses a real
    session-holding scope with a held-session probe."""

    @asynccontextmanager
    async def _scope(settings: Settings) -> AsyncIterator[HeldSession]:
        yield HeldSession(await loader(settings), probe=None)

    return _scope


class EarthdataMCPNotReadyError(RuntimeError):
    """Raised by ``.tools`` when the manager is not in the ready state."""


def check_tool_schemas(tools: dict[str, BaseTool]) -> dict[str, list[str]]:
    """Return ``{tool_name: [missing_param, ...]}`` for every required tool
    whose advertised input schema (``tool.args``) is missing a parameter this
    backend sends (``REQUIRED_TOOL_PARAMS``). Empty when every present
    required tool's contract holds. A tool absent from ``tools`` entirely is
    not reported here — presence is the separate missing-tools verdict the
    session scope (``open_earthdata_session``, via ``_require_all_present``)
    already raises on.
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

    ``session_scope``/``loader``/``probe``/``sleep`` are injectable seams for
    tests (a fake client that fails N times then succeeds, a fake steady-state
    probe, and a no-op sleep so retry tests don't actually wait out the
    backoff) — production callers rely on the defaults
    (``open_earthdata_session``, ``probe_mcp_tools``, ``asyncio.sleep``).

    ``session_scope`` is the session-holding seam the production path uses; a
    plain ``loader`` (legacy) is accepted too and wrapped into a scope that
    holds nothing open, so every existing hermetic test keeps working. Passing
    neither uses the real ``open_earthdata_session`` — the whole point of this
    module now is to keep one MCP session alive so the tools it hands out
    reuse it instead of reconnecting on every ``ainvoke``.
    """

    def __init__(
        self,
        settings: Settings,
        user_id_getter: Callable[[], str],
        *,
        on_ready: OnReady | None = None,
        session_scope: SessionScope | None = None,
        loader: Loader | None = None,
        probe: Probe = probe_mcp_tools,
        sleep: Sleeper = asyncio.sleep,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 30.0,
        heartbeat_failure_threshold: int = 2,
        heartbeat_retry_seconds: float = 5.0,
        edl_injector: EdlCredentialInjector | None = None,
    ):
        self._settings = settings
        self._user_id_getter = user_id_getter
        self._on_ready = on_ready
        if session_scope is not None:
            self._session_scope = session_scope
        elif loader is not None:
            self._session_scope = _loader_as_scope(loader)
        else:
            self._session_scope = open_earthdata_session
        self._probe = probe
        self._sleep = sleep
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._heartbeat_failure_threshold = heartbeat_failure_threshold
        self._heartbeat_retry_seconds = heartbeat_retry_seconds
        # T31: passed straight through to bind_workspace below -- None
        # reproduces pre-T31 behavior exactly (no edl_token ever injected).
        self._edl_injector = edl_injector
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
        # Belt and suspenders: the loop body classifies every exception, so
        # this should never fire — but if the loop ever does die, the failure
        # must be loud. A silently-dead watchdog freezes the state forever:
        # /health keeps reporting the last state and no reconnect ever runs.
        self._task.add_done_callback(self._log_if_loop_died)

    def _log_if_loop_died(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.critical(
                "earthdata_mcp_connect_loop_died",
                exc_info=exc,
                extra={"_event": "earthdata_mcp_connect_loop_died", "_state": self._state},
            )

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
                reached_ready = await self._connect_once()
            except EarthdataMCPUnavailableError as exc:
                self._transition(STATE_UNAVAILABLE, detail=str(exc))
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue
            except Exception:
                # Anything the specific branch above didn't classify — a
                # bind_workspace/schema-check bug, an on_ready (satellite
                # agent rebuild) failure, a transport error the client didn't
                # wrap. Before this catch, any such exception killed the task
                # silently and froze the state forever (a dead watchdog:
                # /health lies, no reconnect ever runs). CancelledError is a
                # BaseException, so stop() still cancels cleanly through here
                # — and unwinds ``_connect_once``'s ``async with`` on the way
                # out, closing the live session (clean teardown on stop()).
                logger.exception(
                    "earthdata_mcp_connect_loop_error",
                    extra={"_event": "earthdata_mcp_connect_loop_error", "_state": self._state},
                )
                self._transition(STATE_UNAVAILABLE, detail="unexpected connect-loop error")
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue

            if reached_ready:
                # We held a live session, reached ready, and the heartbeat has
                # now demoted us (the session closed as ``_connect_once``'s
                # ``async with`` exited). Reconnect from the initial backoff —
                # a working connection a moment ago means no long wait is
                # warranted before re-establishing one.
                backoff = self._initial_backoff
            else:
                # Incompatible schema (session already closed). Retry with
                # backoff — a redeploy may fix the mismatch.
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    async def _connect_once(self) -> bool:
        """Open the session scope, verify the tool contract, bind the
        workspace, and — on success — stay inside the scope heartbeating until
        demotion. Holding the scope open for the whole ready phase is what
        keeps the one MCP session alive so the bound tools reuse it.

        Returns ``True`` if it reached ready (and the heartbeat later demoted
        it), ``False`` if the advertised schema was incompatible. Raises
        ``EarthdataMCPUnavailableError`` (unreachable / missing tool) or any
        other exception up to ``_connect_loop`` to classify, exactly as the
        pre-session loop did — but now the ``async with`` guarantees the
        session is closed on every exit path, including cancellation."""
        async with self._session_scope(self._settings) as held:
            raw = held.tools
            mismatches = check_tool_schemas(raw)
            if mismatches:
                self._transition(STATE_INCOMPATIBLE, detail=mismatches)
                return False

            tools = bind_workspace(raw, self._user_id_getter, edl_injector=self._edl_injector)
            # on_ready runs BEFORE the state flips to ready, so no caller can
            # ever observe state == ready with a stale/empty consumer (e.g.
            # api.py's satellite agent) still in place. It fires only on a
            # genuine transition into ready (boot or recovery) — not on a
            # heartbeat that finds an already-ready MCP still healthy.
            if self._state != STATE_READY and self._on_ready is not None:
                await self._on_ready(tools)
            self._tools = tools
            self._transition(STATE_READY)
            # Stays here (probing, not reconnecting) until N consecutive
            # misses -- see _heartbeat_while_ready's docstring. When it
            # returns, the state has already been demoted; falling out of this
            # ``async with`` closes the session, and _connect_loop re-runs the
            # full connect path (on_ready fires again on recovery).
            await self._heartbeat_while_ready(held.probe)
            return True

    async def _heartbeat_while_ready(self, held_probe: Callable[[], Awaitable[Any]] | None = None) -> None:
        """PRD T40: while ready, probe on the heartbeat interval. A single
        miss doesn't demote -- it logs and re-probes after a short retry
        delay instead of transitioning, so a transient blip stays invisible
        to every consumer gated on ``.state``. Only
        ``heartbeat_failure_threshold`` *consecutive* misses transitions to
        unavailable, at which point this returns so the outer connect loop
        re-runs the full connect path. A probe that succeeds at any point
        (including between failures) resets the count.

        ``held_probe`` (2026-07-21) pings the HELD session — the one the bound
        tools actually use — so a 404-zombie session (alive-looking after a
        server restart, but every request on its lost session id 404s) is
        detected and healed by a reconnect, instead of masked by a fresh
        throwaway probe that reconnects cleanly while the tool surface stays
        wedged. When absent (the legacy plain-loader test seam) it falls back to
        the injected fresh-session ``self._probe``."""
        consecutive_failures = 0
        await self._sleep(self._heartbeat_interval)
        while True:
            try:
                if held_probe is not None:
                    await held_probe()
                else:
                    await self._probe(self._settings)
            except Exception as exc:
                consecutive_failures += 1
                logger.warning(
                    "earthdata_mcp_heartbeat_miss",
                    extra={
                        "_event": "earthdata_mcp_heartbeat_miss",
                        "_consecutive_failures": consecutive_failures,
                        "_threshold": self._heartbeat_failure_threshold,
                    },
                )
                if consecutive_failures >= self._heartbeat_failure_threshold:
                    self._transition(STATE_UNAVAILABLE, detail=str(exc))
                    return
                await self._sleep(self._heartbeat_retry_seconds)
                continue
            consecutive_failures = 0
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
