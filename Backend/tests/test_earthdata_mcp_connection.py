"""
tests/test_earthdata_mcp_connection.py
=========================================
PRD T17: the connection manager owns the earthdata-retrieval MCP relationship
as an attached service, not a boot dependency. These tests fake the manager's
own seam (a loader callable standing in for load_raw_mcp_tools) rather than
spinning up a real MCP server — the manager's job is state/retry/schema-diff
bookkeeping, which the fake-client pattern documented in the PRD's Testing
Decisions is enough to prove.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["langchain_core"]


async def _yielding_sleep(_seconds) -> None:
    """A fake ``sleep`` that ignores the requested duration but still hands
    control back to the event loop (unlike a bare ``AsyncMock()``, whose
    awaited call resolves without ever suspending). The connect loop now runs
    for the manager's whole lifetime — heartbeating after reaching ready, not
    just returning — so every test that starts it needs a sleep that
    genuinely yields, or the loop live-locks the event loop and the test's
    own poll task never gets a turn to observe a state change."""
    await asyncio.sleep(0)


def _fake_tool(name: str, params: tuple[str, ...]):
    # ``.args`` is what check_tool_schemas compares against; ``.args_schema``/
    # ``.description`` are what earthdata_mcp.workspace.bind_workspace needs
    # to wrap the tool once the manager reaches ready — a fake used all the
    # way through the manager's connect loop needs both.
    return SimpleNamespace(
        name=name,
        description=f"fake {name}",
        args={p: {"type": "string"} for p in params},
        args_schema={"properties": {p: {"type": "string"} for p in params}, "required": list(params)},
    )


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "connection manager test dependencies are not installed",
)
class CheckToolSchemasTests(unittest.TestCase):
    """Hermetic — a fake tool dict, no network, no fake MCP server."""

    def test_every_required_tool_present_and_complete_yields_no_mismatch(self):
        from earthdata_mcp.connection import check_tool_schemas
        from earthdata_mcp.client import REQUIRED_TOOL_PARAMS

        tools = {name: _fake_tool(name, params) for name, params in REQUIRED_TOOL_PARAMS.items()}

        self.assertEqual(check_tool_schemas(tools), {})

    def test_a_tool_missing_a_sent_param_is_named_in_the_diff(self):
        from earthdata_mcp.connection import check_tool_schemas
        from earthdata_mcp.client import REQUIRED_TOOL_PARAMS

        tools = {name: _fake_tool(name, params) for name, params in REQUIRED_TOOL_PARAMS.items()}
        # Real-world shape of the bug this check exists for (T11's
        # aoi_handle-vs-handle mismatch): the MCP's advertised schema drops a
        # parameter this backend actually sends.
        tools["define_area_of_interest"] = _fake_tool("define_area_of_interest", ("workspace_id",))

        mismatches = check_tool_schemas(tools)

        self.assertEqual(mismatches, {"define_area_of_interest": ["location"]})

    def test_a_missing_tool_is_not_reported_as_a_schema_mismatch(self):
        # Tool presence is a separate (missing-tools) verdict, handled by
        # load_raw_mcp_tools raising before the schema check ever runs.
        from earthdata_mcp.connection import check_tool_schemas

        self.assertEqual(check_tool_schemas({}), {})


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "connection manager test dependencies are not installed",
)
class ConnectionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_starts_connecting_and_reaches_ready_on_first_success(self):
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, STATE_CONNECTING, STATE_READY

        tools = {"search_datasets": _fake_tool("search_datasets", ("query", "filters", "workspace_id"))}

        async def loader(settings):
            return tools

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "17", loader=loader, sleep=_yielding_sleep,
        )
        self.assertEqual(manager.state, STATE_CONNECTING)

        manager.start()
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)

        self.assertEqual(manager.state, STATE_READY)
        self.assertEqual(set(manager.tools.keys()), {"search_datasets"})
        await manager.stop()

    async def test_tools_raises_when_not_ready(self):
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, EarthdataMCPNotReadyError

        async def loader(settings):
            raise AssertionError("loader should never be called — manager.start() was not called")

        manager = EarthdataMCPConnectionManager(settings=object(), user_id_getter=lambda: "1", loader=loader)

        with self.assertRaises(EarthdataMCPNotReadyError):
            manager.tools

    async def test_retries_with_backoff_and_recovers_after_n_failures(self):
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, STATE_READY, STATE_UNAVAILABLE
        from earthdata_mcp.client import EarthdataMCPUnavailableError

        attempts = {"count": 0}
        tools = {"search_datasets": _fake_tool("search_datasets", ("query", "filters", "workspace_id"))}

        async def flaky_loader(settings):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise EarthdataMCPUnavailableError("connection refused")
            return tools

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            await asyncio.sleep(0)

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "1", loader=flaky_loader, sleep=fake_sleep,
        )

        manager.start()
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)

        # At least the 3 attempts needed to reach ready — the loop keeps
        # heartbeating (and thus keeps calling the loader) after that, and
        # the fake sleep's immediate yield lets several such cycles race
        # ahead before this test observes ready, so an exact count here
        # would be asserting on that race rather than the retry behavior.
        self.assertGreaterEqual(attempts["count"], 3)
        # First two sleeps are the two failed attempts' backoff (1.0, then
        # 2.0 after doubling) — this is what proves the retry/backoff
        # behavior, regardless of how many heartbeat sleeps follow.
        self.assertEqual(sleeps[:2], [1.0, 2.0])
        await manager.stop()

    async def test_a_schema_mismatch_lands_the_manager_in_incompatible(self):
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, STATE_INCOMPATIBLE

        broken_tools = {"define_area_of_interest": _fake_tool("define_area_of_interest", ("workspace_id",))}

        async def loader(settings):
            return broken_tools

        async def fake_sleep(_seconds):
            # A real yield (unlike a bare AsyncMock) — this state never
            # resolves on its own, so the retry loop must actually hand
            # control back to the event loop each pass for the test's own
            # poll task (and the eventual manager.stop()) to ever run.
            await asyncio.sleep(0)

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "1", loader=loader, sleep=fake_sleep,
        )

        with self.assertLogs("earthdata_mcp.connection", level="CRITICAL") as captured:
            manager.start()
            await asyncio.wait_for(_wait_for_state(manager, STATE_INCOMPATIBLE), timeout=1)
            await manager.stop()

        self.assertEqual(manager.state, STATE_INCOMPATIBLE)
        self.assertTrue(any("earthdata_mcp_schema_mismatch" in line for line in captured.output))

    async def test_on_ready_is_awaited_exactly_once_before_state_flips_to_ready(self):
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, STATE_READY

        tools = {"search_datasets": _fake_tool("search_datasets", ("query", "filters", "workspace_id"))}
        observed_state_at_callback = []

        async def loader(settings):
            return tools

        async def on_ready(ready_tools):
            # The callback must see the tools before external observers can
            # see state == ready (T17: rebuilding the satellite agent must
            # never race a caller that just checked manager.state).
            observed_state_at_callback.append(manager.state)
            self.assertEqual(set(ready_tools.keys()), {"search_datasets"})

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "1", loader=loader, on_ready=on_ready, sleep=_yielding_sleep,
        )

        manager.start()
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)
        await manager.stop()

        self.assertNotEqual(observed_state_at_callback[0], STATE_READY)

    async def test_on_ready_does_not_re_fire_on_a_healthy_heartbeat(self):
        # T17: a steady-state loop must not rebuild the satellite agent every
        # heartbeat — only a genuine transition into ready calls on_ready.
        from earthdata_mcp.connection import EarthdataMCPConnectionManager, STATE_READY

        tools = {"search_datasets": _fake_tool("search_datasets", ("query", "filters", "workspace_id"))}
        on_ready_calls = []

        async def loader(settings):
            return tools

        async def on_ready(ready_tools):
            on_ready_calls.append(ready_tools)

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "1", loader=loader, on_ready=on_ready, sleep=_yielding_sleep,
        )

        manager.start()
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)
        # The fake sleep ignores the heartbeat interval and yields
        # immediately, so several heartbeat cycles pass in these few ticks.
        for _ in range(10):
            await asyncio.sleep(0)
        await manager.stop()

        self.assertEqual(len(on_ready_calls), 1)

    async def test_detects_a_later_outage_and_recovers_without_a_restart(self):
        # T17 story #5/#6: the manager must keep watching after boot — a
        # mid-session MCP outage (not just a down-at-boot one) has to flip
        # /health away from "ready", or every consumer gated on manager.state
        # would let a bare connection failure through during that window
        # (the exact "/health lies" bug this PRD's Problem Statement names).
        from earthdata_mcp.connection import (
            EarthdataMCPConnectionManager,
            STATE_READY,
            STATE_UNAVAILABLE,
        )
        from earthdata_mcp.client import EarthdataMCPUnavailableError

        tools = {"search_datasets": _fake_tool("search_datasets", ("query", "filters", "workspace_id"))}
        mcp_up = {"value": True}
        on_ready_calls = []

        async def flaky_loader(settings):
            if mcp_up["value"]:
                return tools
            raise EarthdataMCPUnavailableError("connection refused")

        async def on_ready(ready_tools):
            on_ready_calls.append(ready_tools)

        manager = EarthdataMCPConnectionManager(
            settings=object(), user_id_getter=lambda: "1", loader=flaky_loader, on_ready=on_ready,
            sleep=_yielding_sleep,
        )

        manager.start()
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)
        self.assertEqual(len(on_ready_calls), 1)

        mcp_up["value"] = False
        await asyncio.wait_for(_wait_for_state(manager, STATE_UNAVAILABLE), timeout=1)
        with self.assertRaises(Exception):
            manager.tools  # gated consumers must stop seeing the stale tools

        mcp_up["value"] = True
        await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=1)
        await manager.stop()

        # Recovery re-runs on_ready — a fresh connection may hold new tool
        # objects, so consumers (e.g. the satellite agent) must rebuild too.
        self.assertEqual(len(on_ready_calls), 2)


async def _wait_for_state(manager, target_state) -> None:
    while manager.state != target_state:
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
