"""
tests/test_earthdata_mcp_session_reuse.py
===========================================
Root-cause regression for the 2026-07-20 MCP request storm.

The per-turn circuit breaker (earthdata_mcp/results.py) *bounded* the storm;
this proves the source is gone. Tools loaded via
``MultiServerMCPClient.get_tools()`` carry ``session=None``, so
langchain-mcp-adapters 0.3.0 opens a fresh streamable-HTTP session (POST
initialize + notifications/initialized + GET SSE + POST tools/call) on every
single ``ainvoke`` — an N-fold HTTP fan-out and a reconnect-storm surface.
Holding one long-lived session (``open_earthdata_session``, owned by the
connection manager) makes bound tools reuse it: N logical tool calls must
produce exactly ONE session ``initialize``.

These tests run against the real FastMCP streamable-HTTP fake server (no
adapter internals mocked), counting ``initialize`` requests at the ASGI layer
via ``InitializeCountingMiddleware``.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


async def _probe_ok(_settings) -> set[str]:
    """A no-network heartbeat probe: the real ``probe_mcp_tools`` would open
    its own extra session on every heartbeat and pollute the initialize count.
    These tests care only about the steady-state tool surface reusing one
    session, so the probe is stubbed and the heartbeat interval set high
    enough that it never fires during the test."""
    return set()


async def _wait_for_state(manager, target_state) -> None:
    while manager.state != target_state:
        await asyncio.sleep(0.01)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP session-reuse test dependencies are not installed",
)
class SessionReuseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from fake_earthdata_mcp import (
            FakeEarthdataMCPServer,
            InitializeCountingMiddleware,
            build_fake_mcp,
        )

        self.middleware: InitializeCountingMiddleware | None = None

        def _wrap(app):
            self.middleware = InitializeCountingMiddleware(app)
            return self.middleware

        self.server = FakeEarthdataMCPServer(build_fake_mcp(), app_wrapper=_wrap)
        self.server.start()
        self.addCleanup(self.server.stop)

    def _settings(self):
        from config.settings import Settings

        return Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)

    async def test_middleware_counts_initializes(self):
        # Control: the legacy per-call path (get_tools -> session=None) opens a
        # fresh session on every ainvoke, so the same N calls storm the server
        # with initializes. This both documents the bug and proves the counter
        # is actually observing session opens (so a 1 in the reuse test below
        # is meaningful, not a counter that never increments).
        from earthdata_mcp.toolset import load_earthdata_tools

        n = 5
        tools = await load_earthdata_tools(self._settings(), lambda: "17")
        # Loading the surface already opened at least one session to list.
        count_after_load = self.middleware.initialize_count
        self.assertGreaterEqual(count_after_load, 1)

        for _ in range(n):
            await tools["list_workspace"].ainvoke({})

        # Each ainvoke reconnected -> one more initialize apiece.
        self.assertGreaterEqual(self.middleware.initialize_count, count_after_load + n)

    async def test_manager_holds_one_session_for_many_tool_calls(self):
        from earthdata_mcp.connection import (
            STATE_READY,
            EarthdataMCPConnectionManager,
        )

        manager = EarthdataMCPConnectionManager(
            settings=self._settings(),
            user_id_getter=lambda: "17",
            probe=_probe_ok,
            # High enough that the heartbeat never fires during the test, so
            # the only session open is the one the ready phase holds.
            heartbeat_interval_seconds=3600.0,
        )

        manager.start()
        try:
            await asyncio.wait_for(_wait_for_state(manager, STATE_READY), timeout=10)

            # Reaching ready opened exactly one session.
            self.assertEqual(self.middleware.initialize_count, 1)

            tool = manager.tools["list_workspace"]

            # N sequential logical tool calls...
            for _ in range(5):
                result = await tool.ainvoke({})
                self.assertIn("list_workspace", str(result))

            # ...and N concurrent ones (the ReAct ToolNode's asyncio.gather
            # fan-out) — the shared session multiplexes by request id, so they
            # all succeed without opening a new connection.
            results = await asyncio.gather(*(tool.ainvoke({}) for _ in range(5)))
            self.assertEqual(len(results), 5)
            for result in results:
                self.assertIn("list_workspace", str(result))

            # The whole point: 10 logical calls, still one initialize.
            self.assertEqual(self.middleware.initialize_count, 1)
        finally:
            await manager.stop()


if __name__ == "__main__":
    unittest.main()
