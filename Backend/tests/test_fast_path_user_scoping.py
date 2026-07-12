"""
tests/test_fast_path_user_scoping.py
======================================
T26 regression: the satellite fast path (services/chat_stream_service.py::
_fast_path_events) used to call run_satellite directly, bypassing
stream_response's own current_user_id binding entirely — every fast-pathed
retrieval landed in a shared "user-None" workspace (113 orphaned rows found
live). Fixed by wrapping stream_chat_events in user_id_context(user_id)
*before* the fast path's asyncio.create_task(run()) spawns, so that task
(and every context it copies further down, including the sub-agent's own
tool calls) sees the real user id.

These tests drive the real bind_workspace wrapper against a fake MCP server
through the actual ChatStreamService seam, rather than asserting on
internals, so a regression in the context-propagation chain (chat_stream_
service -> subagent_dispatch -> utils.streaming -> earthdata_mcp.workspace)
fails here regardless of which layer breaks.
"""
import asyncio
import importlib.util
import json
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


class UntouchedAgent:
    """Fails the test loudly if the fast path (wrongly) invokes it."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to untouched agent: {name}")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class FastPathUserScopingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.workspace import bind_workspace
        from config.settings import Settings
        from utils.streaming import current_user_id

        self.received_workspace_ids = []

        async def search_datasets(query, filters, workspace_id):
            self.received_workspace_ids.append(workspace_id)
            return {"datasets": [], "count": 0}

        self.server = FakeEarthdataMCPServer(build_fake_mcp({"search_datasets": search_datasets}))
        self.server.start()
        self.addCleanup(self.server.stop)

        raw_tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        )
        # Bound against current_user_id itself (the real ContextVar getter
        # api.py wires up in production, T18) — not a stand-in lambda — so
        # this test actually exercises the getter the fix threads context to.
        self.bound_tools = bind_workspace(raw_tools, current_user_id)

    async def test_satellite_fast_path_binds_the_real_users_workspace_not_user_none(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        bound_tools = self.bound_tools

        class ToolCallingSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await bound_tools["search_datasets"].ainvoke({"query": "no2", "filters": None})
                yield "updates", {
                    "agent": {"messages": [
                        SimpleNamespace(tool_calls=[{"id": "tc1", "name": "search_datasets", "args": {}}], content=""),
                    ]},
                }
                await asyncio.sleep(0)
                envelope = json.dumps({"summary": "No datasets found.", "artifact_ids": [], "handles": []})
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        [
            event
            async for event in service.stream_chat_events(
                UntouchedAgent(), UntouchedAgent(), ToolCallingSatelliteAgent(),
                "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", "researcher-42", "req-1",
            )
        ]

        self.assertEqual(self.received_workspace_ids, ["user-researcher-42"])

    async def test_a_missing_user_id_never_reaches_the_mcp_as_user_none(self):
        """A None user id must not silently pool into a shared "user-None"
        workspace (T26). bind_workspace's own guard (asserted directly and
        more precisely in test_earthdata_mcp_workspace.py's
        WorkspaceMissingUserContextTests) raises before the MCP is ever
        called; run_satellite's existing generic-failure handling folds that
        into the turn's answer text rather than reaching the fake MCP — this
        test's job is just to confirm no "user-None" call ever lands here."""
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        bound_tools = self.bound_tools

        class ToolCallingSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await bound_tools["search_datasets"].ainvoke({"query": "no2", "filters": None})
                yield "messages", (SimpleNamespace(content="unreachable", type="ai", tool_calls=None), {})

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        [
            event
            async for event in service.stream_chat_events(
                UntouchedAgent(), UntouchedAgent(), ToolCallingSatelliteAgent(),
                "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", None, "req-1",
            )
        ]

        self.assertEqual(self.received_workspace_ids, [])


if __name__ == "__main__":
    unittest.main()
