"""
tests/test_earthdata_agent_tools_refresh.py
============================================
refresh_live_tools — the in-place tools-dict refresh that makes an in-flight
chat turn survive a mid-turn MCP reconnect (2026-07-21).

The earthdata MCP server crash-restarts, killing the long-lived client session;
the connection manager reconnects with a brand-new session's bound tools and
re-fires on_ready. Because every satellite tool indexes its mcp_tools dict at
call time, refreshing that dict IN PLACE (identity preserved) lets tool closures
already compiled into an in-progress turn read the reconnected session's tools
on their next call — instead of retrying the dead session forever.
"""
from __future__ import annotations

import unittest

from agents.earthdata_agent import refresh_live_tools


class RefreshLiveToolsTests(unittest.TestCase):
    def test_preserves_dict_identity_so_captured_closures_see_fresh_tools(self):
        live = {"align": "old_align", "export_result": "old_export"}
        captured = live  # a tool closure's captured reference to the dict
        refresh_live_tools(live, {"align": "new_align", "export_result": "new_export"})
        # Same object — the closure never re-binds, it just re-reads.
        self.assertIs(captured, live)
        self.assertEqual(captured["align"], "new_align")
        self.assertEqual(captured["export_result"], "new_export")

    def test_populates_an_initially_empty_live_dict(self):
        live: dict = {}
        refresh_live_tools(live, {"align": "a", "export_result": "e"})
        self.assertEqual(live, {"align": "a", "export_result": "e"})

    def test_drops_keys_absent_from_the_fresh_set(self):
        live = {"align": 1, "export_result": 2, "removed": 3}
        refresh_live_tools(live, {"align": 10, "export_result": 20})
        self.assertEqual(live, {"align": 10, "export_result": 20})


if __name__ == "__main__":
    unittest.main()
