import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["langchain", "langchain_core"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "langchain is not installed",
)
class SubagentTrimMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """T13's high-ceiling safety net on the sub-agents: a last-resort trim
    that must never fire in a healthy workflow, but converts an unforeseen
    bloat source into a degraded-but-alive turn instead of a provider 413,
    logging subagent_trim_activated whenever it actually removes messages.

    Exercises the middleware's own wrap_model_call hook directly against a
    hand-built ModelRequest/handler — not a full create_agent/LangGraph
    invocation, which pulls in a real chat model, a real graph run, and (as
    observed) can misbehave badly under concurrent test-suite load.
    """

    async def _invoke(self, *, max_tokens: int, messages: list):
        from langchain.agents.middleware import ModelRequest

        from agents.subagent_trim import build_subagent_trim_middleware

        middleware = build_subagent_trim_middleware("earthdata", max_tokens=max_tokens)
        request = ModelRequest(model=object(), messages=messages, state={"messages": messages})
        received: dict = {}

        async def handler(req):
            received["messages"] = req.messages
            return "handled"

        result = await middleware.awrap_model_call(request, handler)
        return result, received["messages"]

    async def test_trims_an_oversized_history_and_logs_the_activation_event(self):
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=f"message {i}") for i in range(200)]

        with self.assertLogs("agents.subagent_trim", level="WARNING") as captured:
            result, passed_messages = await self._invoke(max_tokens=50, messages=messages)

        # The turn completes — degraded, not dropped/errored.
        self.assertEqual(result, "handled")
        self.assertLess(len(passed_messages), len(messages))
        self.assertIn("subagent_trim_activated", captured.output[0])

    async def test_never_fires_for_a_history_that_already_fits(self):
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=f"message {i}") for i in range(3)]

        with self.assertNoLogs("agents.subagent_trim", level="WARNING"):
            result, passed_messages = await self._invoke(max_tokens=20000, messages=messages)

        self.assertEqual(result, "handled")
        self.assertEqual(len(passed_messages), len(messages))


if __name__ == "__main__":
    unittest.main()
