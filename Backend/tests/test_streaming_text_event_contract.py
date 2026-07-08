import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class TextEventContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_updates_fallback_flattens_list_shaped_ai_message_content(self):
        """stream_response's docstring promises ("text", str) for every
        "text" event. When the "messages" stream mode stays silent for a
        turn (the model returned one complete, non-streamed message) the
        "updates" fallback used to publish msg.content verbatim — a list of
        content blocks for providers that shape AIMessage content that way —
        breaking that contract and crashing any consumer (e.g.
        subagent_dispatch.run_satellite's ``data.get("response", "")``) that
        trusted it."""
        from utils.streaming import stream_response

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                await asyncio.sleep(0)
                ai_message = SimpleNamespace(
                    content=[{"type": "text", "text": "final answer"}],
                    type="ai",
                    tool_calls=None,
                    name=None,
                )
                yield "updates", {"agent": {"messages": [ai_message]}}

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        text_events = [data for event_type, data in events if event_type == "text"]
        self.assertEqual(len(text_events), 1)
        self.assertIsInstance(text_events[0], str)
        self.assertEqual(text_events[0], "final answer")


if __name__ == "__main__":
    unittest.main()
