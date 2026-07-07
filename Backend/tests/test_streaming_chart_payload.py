import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class ChartPayloadStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_response_forwards_chart_payload_events(self):
        from utils.streaming import emit_chart, stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                emit_chart({"type": "heatmap", "chart_id": "map_1", "values": [[1.0]]})
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        chart_events = [data for event_type, data in events if event_type == "chart_payload"]

        self.assertEqual(len(chart_events), 1)
        self.assertEqual(chart_events[0]["chart_id"], "map_1")
        self.assertEqual(chart_events[0]["values"], [[1.0]])

    async def test_emit_chart_is_a_no_op_outside_a_stream_response_context(self):
        from utils.streaming import emit_chart

        # Should not raise even though no stream_response is active.
        emit_chart({"type": "heatmap"})

    async def test_nested_stream_response_bubbles_chart_payload_to_the_outer_context(self):
        """Mirrors emit_job_progress's bubbling: a chart emitted deep inside a
        nested stream_response call (the pattern _run_satellite uses to consume
        the satellite sub-agent's own stream) must reach the outer stream_response
        call's queue too, exactly as job_progress events do."""
        from utils.streaming import emit_chart, stream_response

        class InnerAgent:
            async def astream(self, input_, config, stream_mode):
                emit_chart({"type": "heatmap", "chart_id": "map_inner"})
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        class OuterAgent:
            async def astream(self, input_, config, stream_mode):
                async for event_type, data in stream_response(InnerAgent(), "nested", "inner-thread"):
                    pass
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="outer done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(OuterAgent(), "do it", "outer-thread")]

        chart_events = [data for event_type, data in events if event_type == "chart_payload"]
        self.assertEqual(len(chart_events), 1)
        self.assertEqual(chart_events[0]["chart_id"], "map_inner")


if __name__ == "__main__":
    unittest.main()
