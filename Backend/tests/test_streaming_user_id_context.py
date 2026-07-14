import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class UserIdContextGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_stream_response_never_clobbers_the_outer_bound_user_id(self):
        """Mirrors the outermost-wins pattern stream_response already uses for
        _call_budget/_turn_started_at: a nested stream_response call must not
        override an already-bound _current_user_id, even if (unlike today's
        real callers, which never pass user_id on the nested call) some
        future nested call passes one of its own."""
        from utils.streaming import current_user_id, stream_response

        seen_during_inner = []

        class InnerAgent:
            async def astream(self, input_, config, stream_mode):
                seen_during_inner.append(current_user_id())
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="inner done", type="ai", tool_calls=None), {})

        class OuterAgent:
            async def astream(self, input_, config, stream_mode):
                async for _ in stream_response(InnerAgent(), "nested", "inner-thread", user_id="inner-user"):
                    pass
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="outer done", type="ai", tool_calls=None), {})

        [
            event
            async for event in stream_response(OuterAgent(), "do it", "outer-thread", user_id="outer-user")
        ]

        self.assertEqual(seen_during_inner, ["outer-user"])
        self.assertIsNone(current_user_id())


if __name__ == "__main__":
    unittest.main()
