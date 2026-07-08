import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class StageStatusEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_emit_status_carries_stage_and_detail_through_the_status_event(self):
        from utils.streaming import emit_status, stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Checking coverage...", stage="coverage", detail="14 granules")
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        status_events = [data for event_type, data in events if event_type == "status"]
        self.assertEqual(len(status_events), 1)
        self.assertEqual(status_events[0]["message"], "Checking coverage...")
        self.assertEqual(status_events[0]["stage"], "coverage")
        self.assertEqual(status_events[0]["detail"], "14 granules")

    async def test_emit_status_without_stage_stays_a_bare_message(self):
        """Additive: an existing bare emit_status(message) call keeps
        producing a status event with no stage/detail keys at all, so
        frontend handling that only ever read `.message` degrades
        gracefully (Implementation Decisions: additive fields)."""
        from utils.streaming import emit_status, stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Working on it...")
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        status_events = [data for event_type, data in events if event_type == "status"]
        self.assertEqual(status_events[0], {"message": "Working on it..."})

    async def test_nested_stream_response_bubbles_stage_status_to_the_outer_context(self):
        """Mirrors emit_chart/emit_job_progress's bubbling pattern for stage
        status events emitted deep inside a nested stream_response call."""
        from utils.streaming import emit_status, stream_response

        class InnerAgent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Searching datasets...", stage="search")
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        class OuterAgent:
            async def astream(self, input_, config, stream_mode):
                async for _ in stream_response(InnerAgent(), "nested", "inner-thread"):
                    pass
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="outer done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(OuterAgent(), "do it", "outer-thread")]

        status_events = [data for event_type, data in events if event_type == "status"]
        self.assertEqual([s.get("stage") for s in status_events], ["search"])


if __name__ == "__main__":
    unittest.main()
