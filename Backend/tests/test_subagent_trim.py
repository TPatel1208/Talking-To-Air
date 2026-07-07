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
    logging subagent_trim_activated whenever it actually removes messages."""

    async def _run_agent(self, *, max_tokens: int, n_messages: int):
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        from agents.subagent_trim import build_subagent_trim_middleware

        trim_middleware = build_subagent_trim_middleware("earthdata", max_tokens=max_tokens)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
        agent = create_agent(
            model=model, tools=[], system_prompt="sys", checkpointer=None, middleware=[trim_middleware],
        )
        messages = [HumanMessage(content=f"message {i}") for i in range(n_messages)]
        result = await agent.ainvoke({"messages": messages})
        return result

    async def test_trims_an_oversized_history_and_logs_the_activation_event(self):
        with self.assertLogs("agents.subagent_trim", level="WARNING") as captured:
            result = await self._run_agent(max_tokens=50, n_messages=200)

        # The turn completes — degraded, not dropped/errored.
        self.assertEqual(result["messages"][-1].content, "done")
        self.assertIn("subagent_trim_activated", captured.output[0])

    async def test_never_fires_for_a_history_that_already_fits(self):
        from agents.subagent_trim import build_subagent_trim_middleware

        with self.assertNoLogs("agents.subagent_trim", level="WARNING"):
            result = await self._run_agent(max_tokens=20000, n_messages=3)

        self.assertEqual(result["messages"][-1].content, "done")


if __name__ == "__main__":
    unittest.main()
