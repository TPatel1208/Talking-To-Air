"""Sub-agents answer with a SubAgentEnvelope tool call, so narration and
fenced JSON (live 2026-11-05, Claude via Portkey) never become the answer."""
import asyncio
import importlib.util
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

_HAS_LANGCHAIN = importlib.util.find_spec("langchain") is not None

# eval_harness lives beside the tests, not in the package.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _envelope(**overrides):
    from tta_backend.models import SubAgentEnvelope

    fields = {"summary": "Plotted TEMPO NO2 over NJ.", "artifact_ids": ["map_1"], "handles": ["obs_1"]}
    fields.update(overrides)
    return SubAgentEnvelope(**fields)


@unittest.skipUnless(_HAS_LANGCHAIN, "langchain is not installed")
class ResponseFormatTests(unittest.TestCase):
    def test_the_envelope_tool_is_named_after_the_schema(self):
        """run_satellite hides calls by this name, so it must match ToolStrategy's."""
        from tta_backend.agents.subagent_output import subagent_response_format
        from tta_backend.models import SUB_AGENT_ENVELOPE_TOOL, SubAgentEnvelope

        strategy = subagent_response_format()

        self.assertIs(strategy.schema, SubAgentEnvelope)
        self.assertEqual([spec.name for spec in strategy.schema_specs], [SUB_AGENT_ENVELOPE_TOOL])

    def test_both_sub_agents_are_built_with_the_structured_answer(self):
        from tta_backend.agents import earthdata_agent, ground_sensor_agent
        from tta_backend.models import SubAgentEnvelope

        for module, build in (
            (earthdata_agent, lambda: earthdata_agent.build_earthdata_agent(mcp_tools={})),
            (ground_sensor_agent, ground_sensor_agent.build_ground_agent),
        ):
            with self.subTest(module=module.__name__):
                created = SimpleNamespace()
                create_agent = Mock(return_value=created)
                with patch.object(module, "build_chat_model", Mock(return_value=object())), \
                     patch.object(module, "create_agent", create_agent), \
                     patch.object(earthdata_agent, "build_satellite_tools", Mock(return_value=[])):
                    build()

                response_format = create_agent.call_args.kwargs["response_format"]
                self.assertIs(response_format.schema, SubAgentEnvelope)


@unittest.skipUnless(_HAS_LANGCHAIN, "langchain is not installed")
class StreamResponseStructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_publishes_the_structured_answer_alongside_its_tool_call(self):
        """The shared stream publishes the envelope call; run_satellite hides it."""
        from tta_backend.models import SUB_AGENT_ENVELOPE_TOOL
        from tta_backend.utils.streaming import stream_response

        envelope = _envelope()

        class FakeAgent:
            async def astream(self, input_, config, stream_mode):
                await asyncio.sleep(0)
                yield "updates", {"model": {"messages": [
                    SimpleNamespace(tool_calls=[{"id": "t1", "name": "plot_singular", "args": {"x": 1}}], content=""),
                ]}}
                yield "updates", {"model": {
                    "messages": [
                        SimpleNamespace(
                            tool_calls=[{"id": "t2", "name": SUB_AGENT_ENVELOPE_TOOL, "args": envelope.model_dump()}],
                            content="",
                        ),
                        SimpleNamespace(name=SUB_AGENT_ENVELOPE_TOOL, content="Answer recorded.", tool_calls=None),
                    ],
                    "structured_response": envelope,
                }}

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        self.assertIn(("structured_response", envelope), events)
        tool_names = [data["name"] for kind, data in events if kind in ("tool_call", "tool_result")]
        self.assertEqual(tool_names, ["plot_singular", SUB_AGENT_ENVELOPE_TOOL, SUB_AGENT_ENVELOPE_TOOL])

    async def test_a_real_agent_streams_its_structured_answer(self):
        """Uses a real create_agent graph, so the update shape is LangChain's."""
        from langchain.agents import create_agent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        from tta_backend.agents.subagent_output import subagent_response_format
        from tta_backend.models import SUB_AGENT_ENVELOPE_TOOL, SubAgentEnvelope
        from tta_backend.utils.streaming import stream_response

        class EnvelopeCallingModel(BaseChatModel):
            """Always calls the envelope tool. Non-streaming: the stock fakes
            drop tool calls when streaming."""

            calls: int = 0

            @property
            def _llm_type(self) -> str:
                return "envelope-calling-fake"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.calls += 1
                message = AIMessage(
                    content="Now let me record the answer:",
                    tool_calls=[{
                        "id": f"call_{self.calls}",
                        "name": SUB_AGENT_ENVELOPE_TOOL,
                        "args": {"summary": "Plotted it.", "artifact_ids": [], "handles": []},
                    }],
                )
                return ChatResult(generations=[ChatGeneration(message=message)])

        model = EnvelopeCallingModel()
        agent = create_agent(model=model, tools=[], response_format=subagent_response_format())

        events = [event async for event in stream_response(agent, "plot it", "thread-real")]

        answers = [data for kind, data in events if kind == "structured_response"]
        self.assertEqual(len(answers), 1)
        self.assertIsInstance(answers[0], SubAgentEnvelope)
        self.assertEqual(answers[0].summary, "Plotted it.")
        self.assertEqual([data["name"] for kind, data in events if kind == "tool_call"], [SUB_AGENT_ENVELOPE_TOOL])
        self.assertEqual(model.calls, 1)  # the envelope call ended the run


@unittest.skipUnless(_HAS_LANGCHAIN, "langchain is not installed")
class RunSatelliteStructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_answers_from_the_structured_answer_not_the_narration(self):
        """Narration plus fenced JSON (the 2026-11-05 shape): no salvage."""
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import get_call_budget

        envelope = _envelope()

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                yield "messages", (SimpleNamespace(content="I'll search for the dataset.", type="ai", tool_calls=None), {})
                yield "updates", {"model": {"messages": [
                    SimpleNamespace(tool_calls=[{"id": "t1", "name": "search_datasets", "args": {"query": "TEMPO NO2"}}], content=""),
                ]}}
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content='Now let me plot it:```json\n{"summary": "x"}\n```', type="ai", tool_calls=None), {})
                yield "updates", {"model": {"messages": [], "structured_response": envelope}}

        get_call_budget().clear()
        with patch.object(subagent_dispatch, "get_satellite_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_satellite_context", AsyncMock()), \
             self.assertNoLogs("tta_backend.services.subagent_dispatch", level="WARNING"):
            result = await subagent_dispatch.run_satellite(FakeSatelliteAgent(), "Plot TEMPO NO2 over NJ", "thread-1")

        self.assertEqual(result.text, "Plotted TEMPO NO2 over NJ.")
        self.assertEqual(result.handles, ["obs_1"])

    async def test_does_not_forward_the_answer_or_its_tool_call_as_live_events(self):
        """The fast path shows each forwarded tool_call in the workflow strip."""
        from tta_backend.models import SUB_AGENT_ENVELOPE_TOOL
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import get_call_budget

        envelope = _envelope()

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await asyncio.sleep(0)
                yield "updates", {"model": {"messages": [
                    SimpleNamespace(tool_calls=[{"id": "t1", "name": "plot_singular", "args": {}}], content=""),
                ]}}
                yield "updates", {"model": {
                    "messages": [
                        SimpleNamespace(
                            tool_calls=[{"id": "t2", "name": SUB_AGENT_ENVELOPE_TOOL, "args": envelope.model_dump()}],
                            content="",
                        ),
                        SimpleNamespace(name=SUB_AGENT_ENVELOPE_TOOL, content="Answer recorded.", tool_calls=None),
                    ],
                    "structured_response": envelope,
                }}

        forwarded = []

        async def on_event(event_type, data):
            forwarded.append((event_type, data.get("name") if isinstance(data, dict) else None))

        get_call_budget().clear()
        with patch.object(subagent_dispatch, "get_satellite_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_satellite_context", AsyncMock()):
            result = await subagent_dispatch.run_satellite(
                FakeSatelliteAgent(), "Plot NO2", "thread-1", on_event=on_event,
            )

        self.assertEqual(result.text, "Plotted TEMPO NO2 over NJ.")
        self.assertEqual(forwarded, [("tool_call", "plot_singular")])


@unittest.skipUnless(_HAS_LANGCHAIN, "langchain is not installed")
class RunGroundStructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_answers_from_the_structured_answer(self):
        """The last message is now "Answer recorded.", not the answer."""
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import get_call_budget

        envelope = _envelope(summary="Newark monitor 34-013-0003 read 41 ppb.", artifact_ids=[], handles=[])

        class FakeGroundAgent:
            async def ainvoke(self, input_, config):
                return {
                    "messages": [
                        SimpleNamespace(content="Let me look that up.", type="ai"),
                        SimpleNamespace(content="Answer recorded.", type="tool", name="SubAgentEnvelope"),
                    ],
                    "structured_response": envelope,
                }

        get_call_budget().clear()
        with patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()):
            result = await subagent_dispatch.run_ground(FakeGroundAgent(), "readings", "thread-1")

        self.assertEqual(result.text, "Newark monitor 34-013-0003 read 41 ppb.")

    async def test_still_reads_a_text_answer_without_a_structured_one(self):
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import get_call_budget

        text = json.dumps({"summary": "Text answer.", "artifact_ids": [], "handles": []})

        class FakeGroundAgent:
            async def ainvoke(self, input_, config):
                return {"messages": [SimpleNamespace(content=text, type="ai")]}

        get_call_budget().clear()
        with patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()):
            result = await subagent_dispatch.run_ground(FakeGroundAgent(), "readings", "thread-1")

        self.assertEqual(result.text, "Text answer.")


@unittest.skipUnless(_HAS_LANGCHAIN, "langchain is not installed")
class EvalHarnessStructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_the_structured_answer_not_the_narration(self):
        """The eval used to read only streamed text, so every envelope was None."""
        import eval_harness

        from tta_backend.agents import earthdata_agent
        from tta_backend.models import SUB_AGENT_ENVELOPE_TOOL

        envelope = _envelope()

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="Let me search.", type="ai", tool_calls=None), {})
                yield "updates", {"model": {"messages": [
                    SimpleNamespace(tool_calls=[{"id": "t1", "name": "search_datasets", "args": {}}], content=""),
                ]}}
                yield "updates", {"model": {
                    "messages": [SimpleNamespace(
                        tool_calls=[{"id": "t2", "name": SUB_AGENT_ENVELOPE_TOOL, "args": envelope.model_dump()}],
                        content="",
                    )],
                    "structured_response": envelope,
                }}

        task = eval_harness.EvalTask(
            name="structured_answer_offline", category="discovery", prompt="find TEMPO NO2",
            handlers={}, expected_tool_calls=["search_datasets"],
            outcome_check=lambda env, calls: env is not None and env.handles == ["obs_1"],
        )
        with patch("tta_backend.earthdata_mcp.client.load_raw_mcp_tools", AsyncMock(return_value={})), \
             patch.object(earthdata_agent, "build_earthdata_agent", Mock(return_value=FakeSatelliteAgent())):
            result = await eval_harness.run_eval_task(task)

        self.assertEqual(result.envelope, envelope)
        self.assertEqual(result.tool_calls, ["search_datasets"])
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
