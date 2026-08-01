"""
tests/test_compare_transient_retry.py
======================================
compare's transient-retry helper (2026-07-20).

The earthdata-mcp *server* crash-restarts in ~10-15s windows. compare is the
heaviest MCP caller (open A + open B + align + open aligned), so it lands in
those windows far more than any single-map tool and used to fail the whole
comparison on the first ``provider_unavailable`` blip — surfaced to the user as
a "temporary service interruption". ``_retry_transient`` rides through a
transient outage (only ``provider_unavailable`` is retried) while re-raising any
real, non-retryable error at once.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tta_backend.earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_USER_INPUT,
    MCPToolError,
)
from tta_backend.tools.satellite_tools.comparison_tools import _TRANSIENT_RETRY_ATTEMPTS, _retry_transient


class CompareTransientRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # No real backoff waits — the retry logic is what's under test, not the clock.
        self._sleep = mock.patch("asyncio.sleep", new=mock.AsyncMock())
        self._sleep.start()

    async def asyncTearDown(self):
        self._sleep.stop()

    async def test_recovers_after_a_few_transient_failures(self):
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            if calls["n"] < 3:  # first two attempts hit the restart window
                raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, "temporarily unavailable")
            return {"handle": "cube_ok"}

        result = await _retry_transient(op, label="align")
        self.assertEqual(result, {"handle": "cube_ok"})
        self.assertEqual(calls["n"], 3, "did not retry through the transient failures")

    async def test_non_transient_error_is_not_retried(self):
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise MCPToolError(CATEGORY_USER_INPUT, "unknown variable 'foo'")

        with self.assertRaises(MCPToolError) as ctx:
            await _retry_transient(op, label="open A")
        self.assertEqual(ctx.exception.category, CATEGORY_USER_INPUT)
        self.assertEqual(calls["n"], 1, "a non-transient error must fail on the first try")

    async def test_a_contract_error_mid_retry_stops_immediately(self):
        # A transient blip that then turns into a real contract error must
        # surface the contract error at once, not keep retrying it.
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            if calls["n"] == 1:
                raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, "temporarily unavailable")
            raise MCPToolError(CATEGORY_CONTRACT, "the data service returned an unrecognized error")

        with self.assertRaises(MCPToolError) as ctx:
            await _retry_transient(op, label="align")
        self.assertEqual(ctx.exception.category, CATEGORY_CONTRACT)
        self.assertEqual(calls["n"], 2)

    async def test_retry_picks_up_a_refreshed_tool_after_a_mid_flight_reconnect(self):
        # The in-flight-recovery mechanism end to end: compare indexes its
        # mcp_tools dict at call time, so when on_ready refreshes that dict IN
        # PLACE (refresh_live_tools) after the MCP reconnects, the retry re-reads
        # the LIVE tool instead of the dead-session one and the turn recovers.
        live: dict = {}

        class LiveTool:
            async def ainvoke(self, kwargs):
                return {"handle": "cube_aligned"}

        class DeadSessionTool:
            async def ainvoke(self, kwargs):
                # The reconnect lands between attempts: swap in the live tool
                # (as _on_earthdata_mcp_ready would), then fail this dead call.
                live["align"] = LiveTool()
                raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, "dead session after restart")

        live["align"] = DeadSessionTool()
        # Exactly how compare invokes align: index the dict at call time.
        result = await _retry_transient(lambda: live["align"].ainvoke({}), label="align")
        self.assertEqual(result, {"handle": "cube_aligned"})

    async def test_persistent_outage_exhausts_attempts_then_raises(self):
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, "still down")

        with self.assertRaises(MCPToolError) as ctx:
            await _retry_transient(op, label="open aligned")
        self.assertEqual(ctx.exception.category, CATEGORY_PROVIDER_UNAVAILABLE)
        self.assertEqual(calls["n"], _TRANSIENT_RETRY_ATTEMPTS, "should try exactly the attempt budget")


if __name__ == "__main__":
    unittest.main()
