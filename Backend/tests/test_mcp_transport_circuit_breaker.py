"""
tests/test_mcp_transport_circuit_breaker.py
============================================
The per-turn MCP transport-failure circuit breaker (storm containment).

Diagnosis (2026-07-20): every earthdata-retrieval MCP tool is a
``MultiServerMCPClient.get_tools()`` tool with no persistent session, so each
``ainvoke`` opens a fresh streamable-HTTP session. When the MCP (or a specific
query's SSE stream) misbehaves, a single wedged call can churn reconnect
requests for its whole ``mcp_call_timeout_seconds`` window, and the ReAct loop
retries it up to the graph recursion budget — observed as 600+ MCP requests per
15s for 20+ minutes without ever reaching the plot step.

``call_tool`` is the one seam every model-facing MCP call passes through. The
breaker counts *consecutive* transport failures within a turn and, once they
cross ``mcp_transport_failure_ceiling``, short-circuits every remaining MCP
call this turn deterministically — turning an unbounded storm into a fast fail.
One success resets the count (a transient blip never trips it), and a caller
with no turn context bound (the jobs/discovery endpoints) is never broken by a
chat turn's storm.
"""
from __future__ import annotations

import unittest
from unittest import mock

import httpx

import tta_backend.earthdata_mcp.results as results
from tta_backend.earthdata_mcp.results import MCPToolError, call_tool
from tta_backend.utils.streaming import _mcp_failure_state, get_mcp_failure_state


class _CountingTransportFailureTool:
    """A tool whose ``ainvoke`` always dies with a transport error (the shape
    ``call_tool`` classifies as provider_unavailable), counting invocations so
    a test can prove a short-circuited call never reached the transport."""

    name = "search_datasets"

    def __init__(self):
        self.invocations = 0

    async def ainvoke(self, kwargs):
        self.invocations += 1
        raise httpx.ConnectError("connection reset")


class _CountingWorkingTool:
    name = "search_datasets"

    def __init__(self):
        self.invocations = 0

    async def ainvoke(self, kwargs):
        self.invocations += 1
        return '{"results": []}'


class _SwitchableTool:
    """Fails with a transport error while ``down`` is True, succeeds otherwise —
    models the MCP server going down (crash-restart) then coming back."""

    name = "search_datasets"

    def __init__(self, down: bool = True):
        self.down = down
        self.invocations = 0

    async def ainvoke(self, kwargs):
        self.invocations += 1
        if self.down:
            raise httpx.ConnectError("connection reset")
        return '{"results": []}'


class MCPTransportCircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # A fresh per-turn state dict, exactly as stream_response binds it.
        self._token = _mcp_failure_state.set({"consecutive": 0, "tripped": False})

    def tearDown(self):
        _mcp_failure_state.reset(self._token)

    async def test_breaker_trips_after_ceiling_consecutive_transport_failures(self):
        tool = _CountingTransportFailureTool()

        # Ceiling defaults to 3: three consecutive transport failures each
        # still hit the transport (and classify as provider_unavailable)...
        for _ in range(3):
            with self.assertRaises(MCPToolError) as ctx:
                await call_tool(tool, {"query": "no2"})
            self.assertEqual(ctx.exception.category, "provider_unavailable")

        self.assertEqual(tool.invocations, 3)
        self.assertTrue(get_mcp_failure_state()["tripped"])

    async def test_tripped_turn_short_circuits_without_touching_the_transport(self):
        tool = _CountingTransportFailureTool()
        for _ in range(3):
            with self.assertRaises(MCPToolError):
                await call_tool(tool, {"query": "no2"})

        # The 4th call must fail instantly, WITHOUT another ainvoke — this is
        # what converts a 20-minute reconnect storm into a bounded fast fail.
        before = tool.invocations
        with self.assertRaises(MCPToolError) as ctx:
            await call_tool(tool, {"query": "no2"})
        self.assertEqual(ctx.exception.category, "provider_unavailable")
        self.assertEqual(tool.invocations, before, "tripped breaker still invoked the tool")

    async def test_one_success_resets_the_consecutive_count(self):
        failing = _CountingTransportFailureTool()
        working = _CountingWorkingTool()

        # Two failures (below the ceiling of 3)...
        for _ in range(2):
            with self.assertRaises(MCPToolError):
                await call_tool(failing, {"query": "no2"})
        # ...then a success resets the streak...
        result = await call_tool(working, {"query": "no2"})
        self.assertEqual(result, '{"results": []}')
        self.assertEqual(get_mcp_failure_state()["consecutive"], 0)

        # ...so two more failures still do not trip (would have, at 4 total).
        for _ in range(2):
            with self.assertRaises(MCPToolError):
                await call_tool(failing, {"query": "no2"})
        self.assertFalse(get_mcp_failure_state()["tripped"])

    async def _trip(self, tool):
        """Trip the breaker with `ceiling` consecutive failures at t=0."""
        for _ in range(3):
            with self.assertRaises(MCPToolError):
                await call_tool(tool, {"query": "no2"})
        self.assertTrue(get_mcp_failure_state()["tripped"])

    async def test_tripped_breaker_stays_fast_fail_until_the_cooldown_elapses(self):
        # A tripped breaker must keep failing instantly (no transport touch)
        # for the whole recovery cooldown — this is the storm-containment
        # behavior the breaker was built for, unchanged within the window.
        tool = _CountingTransportFailureTool()
        with mock.patch.object(results, "_monotonic", return_value=100.0):
            await self._trip(tool)
            tripped_count = tool.invocations
            # Still inside the cooldown (default 5s): 4.9s later, no probe.
            with mock.patch.object(results, "_monotonic", return_value=104.9):
                with self.assertRaises(MCPToolError):
                    await call_tool(tool, {"query": "no2"})
        self.assertEqual(tool.invocations, tripped_count, "probed before cooldown elapsed")

    async def test_half_open_probe_after_cooldown_recovers_when_the_server_is_back(self):
        # The MCP restarted and is healthy again. After the cooldown, the
        # breaker lets ONE probe through; it succeeds and the breaker closes so
        # the rest of the turn proceeds instead of staying poisoned.
        tool = _SwitchableTool(down=True)
        with mock.patch.object(results, "_monotonic", return_value=100.0):
            await self._trip(tool)
        tripped_count = tool.invocations
        tool.down = False  # server recovered
        # Cooldown elapsed (default 5s) -> half-open probe is admitted.
        with mock.patch.object(results, "_monotonic", return_value=106.0):
            result = await call_tool(tool, {"query": "no2"})
        self.assertEqual(result, '{"results": []}')
        self.assertEqual(tool.invocations, tripped_count + 1, "probe did not reach the transport")
        state = get_mcp_failure_state()
        self.assertFalse(state["tripped"], "successful probe did not close the breaker")
        self.assertEqual(state["consecutive"], 0)

    async def test_half_open_probe_failure_rearms_the_cooldown(self):
        # Server still down when the probe fires: the probe fails, the breaker
        # stays open, and the cooldown re-arms — the very next call fast-fails
        # again rather than probing on every call (no storm while still down).
        tool = _CountingTransportFailureTool()
        with mock.patch.object(results, "_monotonic", return_value=100.0):
            await self._trip(tool)
        with mock.patch.object(results, "_monotonic", return_value=106.0):
            with self.assertRaises(MCPToolError):
                await call_tool(tool, {"query": "no2"})  # probe, still down
            probed = tool.invocations
            # Immediately after (same instant, within the re-armed cooldown):
            with self.assertRaises(MCPToolError):
                await call_tool(tool, {"query": "no2"})
        self.assertEqual(tool.invocations, probed, "probed again within the re-armed cooldown")
        self.assertTrue(get_mcp_failure_state()["tripped"])

    async def test_no_turn_context_is_never_circuit_broken(self):
        # The jobs/discovery endpoints call MCP tools with a user_id bound but
        # no stream_response turn — get_mcp_failure_state() is None there, and
        # those callers must never be short-circuited by a chat turn's storm.
        _mcp_failure_state.reset(self._token)
        self._token = _mcp_failure_state.set(None)  # restored shape for tearDown
        self.assertIsNone(get_mcp_failure_state())

        tool = _CountingTransportFailureTool()
        for _ in range(5):
            with self.assertRaises(MCPToolError):
                await call_tool(tool, {"query": "no2"})
        # Every one hit the transport — no breaker, no short-circuit.
        self.assertEqual(tool.invocations, 5)


if __name__ == "__main__":
    unittest.main()
