import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class HeartbeatTests(unittest.IsolatedAsyncioTestCase):
    """Hermetic per Testing Decisions: stalls a scripted tool and asserts a
    working-status event appears within the (patched-low) threshold, then
    stops once real events resume — no timing internals asserted, only
    presence/absence of the stage="working" status event."""

    async def test_heartbeat_fires_during_a_stalled_tool_and_stops_once_real_events_resume(self):
        import utils.streaming as streaming
        from utils.streaming import stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                # Stalls well past the patched heartbeat threshold before
                # ever yielding anything — the honest silence the PRD's
                # Problem Statement describes (a minutes-long provider gap).
                await asyncio.sleep(0.25)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.05), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.02):
            events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        working_events = [
            data for event_type, data in events
            if event_type == "status" and data.get("stage") == "working"
        ]
        self.assertGreaterEqual(len(working_events), 1)
        self.assertIn("elapsed", working_events[0]["message"])
        self.assertIsInstance(working_events[0]["detail"], int)

    async def test_heartbeat_does_not_fire_while_real_events_keep_flowing(self):
        import utils.streaming as streaming
        from utils.streaming import emit_status, stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                for _ in range(5):
                    emit_status("Still working on it (a real stage)...", stage="render")
                    await asyncio.sleep(0.03)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.5), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.02):
            events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        working_events = [
            data for event_type, data in events
            if event_type == "status" and data.get("stage") == "working"
        ]
        self.assertEqual(working_events, [])


if __name__ == "__main__":
    unittest.main()
