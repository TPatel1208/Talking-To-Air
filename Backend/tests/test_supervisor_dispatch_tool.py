"""T64: the supervisor's one sub-agent tool, ``dispatch``, end to end through
real run_ground/run_satellite with fake sub-agent graphs."""
import asyncio
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def _envelope(summary):
    return json.dumps({"summary": summary, "artifact_ids": [], "handles": []})


class FakeGroundAgent:
    def __init__(self):
        self.tasks = []

    async def ainvoke(self, input_, config):
        self.tasks.append(input_["messages"][0].content)
        return {"messages": [SimpleNamespace(
            content=_envelope("Nearest NO2 monitor is Camden Spruce St, station_id 34-007-0002."), type="ai",
        )]}


class FakeSatelliteAgent:
    def __init__(self):
        self.tasks = []

    async def astream(self, input_, config, stream_mode):
        task = input_["messages"][0]["content"]
        self.tasks.append(task)
        region = "the OTC" if "the OTC" in task else "New Jersey"
        yield "updates", {"agent": {"messages": [SimpleNamespace(
            tool_calls=[{"id": "tc1", "name": "define_area_of_interest", "args": {"location": region}}],
            content="",
        )]}}
        await asyncio.sleep(0)
        yield "messages", (SimpleNamespace(content=_envelope(f"Mapped ozone over {region}."), type="ai",
                                           tool_calls=None), {})


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class SupervisorDispatchToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tta_backend.utils.streaming import _call_budget

        self._budget_token = _call_budget.set({})

    async def asyncTearDown(self):
        from tta_backend.utils.streaming import _call_budget

        _call_budget.reset(self._budget_token)

    async def _tools(self, ground, satellite):
        from tta_backend.agents import supervisor_agent

        captured = {}

        def fake_create_agent(*, tools, **kwargs):
            captured["tools"] = {t.name: t for t in tools}
            return object()

        with patch.object(supervisor_agent, "build_chat_model", return_value="llm"), \
             patch.object(supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")), \
             patch.object(supervisor_agent, "create_agent", side_effect=fake_create_agent):
            await supervisor_agent.build_agent(ground_agent=ground, satellite_agent=satellite)
        return captured["tools"]

    async def test_the_supervisor_has_one_tool_dispatch_declared_with_the_gated_schema(self):
        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA
        from tta_backend.models import dispatch_json_schema

        tools = await self._tools(FakeGroundAgent(), FakeSatelliteAgent())

        self.assertEqual(list(tools), ["dispatch"])
        self.assertEqual(tools["dispatch"].metadata[NATIVE_JSON_SCHEMA], dispatch_json_schema())

    async def test_each_request_reaches_its_sub_agent_as_a_rendered_task_under_the_researchers_words(self):
        from tta_backend.models import parse_dispatch_result
        from tta_backend.utils.streaming import user_message_context

        ground, satellite = FakeGroundAgent(), FakeSatelliteAgent()
        tools = await self._tools(ground, satellite)
        message = "Plot ozone over the OTC for 2024-07-15 and find the nearest NO2 monitor to Camden"

        with user_message_context(message):
            raw = await tools["dispatch"].ainvoke({"requests": [
                {"agent": "satellite", "intent": "visualize", "figure_type": "map", "variable": "ozone",
                 "location": "the OTC", "start_date": "2024-07-15"},
                {"agent": "ground", "intent": "nearest_monitor", "pollutant": "NO2", "location": "Camden"},
            ]})

        batch = parse_dispatch_result(raw)
        self.assertEqual([(i.agent, i.status) for i in batch.results], [("satellite", "ok"), ("ground", "ok")])
        self.assertEqual(batch.results[0].result.text, "Mapped ozone over the OTC.")
        (satellite_task,), (ground_task,) = satellite.tasks, ground.tasks
        for task in (satellite_task, ground_task):
            self.assertTrue(task.endswith(f"Researcher's full message (context only): {message}"))
        self.assertIn("Location (use exactly as written): the OTC", satellite_task)
        self.assertIn("Pollutant: NO2", ground_task)

    async def test_a_batch_on_a_thread_saves_its_contexts_once_in_request_order(self):
        from tta_backend.agents import supervisor_agent
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import _current_thread_id

        tools = await self._tools(FakeGroundAgent(), FakeSatelliteAgent())
        token = _current_thread_id.set("thread-1")
        try:
            with patch.object(subagent_dispatch, "get_satellite_context", AsyncMock(return_value={})), \
                 patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
                 patch.object(subagent_dispatch, "save_satellite_context", AsyncMock()) as save_satellite, \
                 patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()) as save_ground, \
                 patch.object(supervisor_agent, "persist_dispatch_contexts", AsyncMock()) as persist:
                await tools["dispatch"].ainvoke({"requests": [
                    {"agent": "satellite", "intent": "visualize", "variable": "ozone", "location": "the OTC"},
                    {"agent": "satellite", "intent": "visualize", "variable": "ozone", "location": "New Jersey"},
                    {"agent": "ground", "intent": "nearest_monitor", "pollutant": "NO2", "location": "Camden"},
                ]})
        finally:
            _current_thread_id.reset(token)

        save_satellite.assert_not_called()
        save_ground.assert_not_called()
        persist.assert_awaited_once()
        self.assertEqual(persist.call_args.args, ("thread-1",))
        self.assertEqual(
            [c["location"] for c in persist.call_args.kwargs["satellite"]], ["the OTC", "New Jersey"],
        )
        self.assertEqual([c["site_id"] for c in persist.call_args.kwargs["ground"]], ["34-007-0002"])

    async def test_only_a_compare_request_lets_the_satellite_agent_use_its_ground_tools(self):
        """A visualize request that wanders into validate_against_ground is
        ground work the supervisor never dispatched (live 2026-10-16)."""
        from tta_backend.utils.streaming import ground_validation_allowed

        class ScopeRecordingSatellite:
            def __init__(self):
                self.allowed = {}

            async def astream(self, input_, config, stream_mode):
                task = input_["messages"][0]["content"]
                intent = next(line.split(": ", 1)[1] for line in task.splitlines() if line.startswith("Intent:"))
                self.allowed[intent] = ground_validation_allowed()
                yield "messages", (SimpleNamespace(content=_envelope("ok"), type="ai", tool_calls=None), {})

        satellite = ScopeRecordingSatellite()
        tools = await self._tools(FakeGroundAgent(), satellite)

        await tools["dispatch"].ainvoke({"requests": [
            {"agent": "satellite", "intent": "visualize", "figure_type": "map", "variable": "NO2", "location": "NJ"},
            {"agent": "satellite", "intent": "compare", "variable": "NO2", "location": "NJ"},
        ]})

        self.assertEqual(satellite.allowed, {"visualize": False, "compare": True})
        self.assertTrue(ground_validation_allowed())

    async def test_sibling_dispatch_calls_save_their_context_one_at_a_time(self):
        from tta_backend.agents import supervisor_agent
        from tta_backend.services import subagent_dispatch
        from tta_backend.utils.streaming import _current_thread_id

        tools = await self._tools(FakeGroundAgent(), FakeSatelliteAgent())
        active, peak = [0], [0]

        async def slow_persist(*args, **kwargs):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.5)
            active[0] -= 1

        token = _current_thread_id.set("thread-1")
        try:
            with patch.object(subagent_dispatch, "get_satellite_context", AsyncMock(return_value={})), \
                 patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
                 patch.object(supervisor_agent, "persist_dispatch_contexts", side_effect=slow_persist):
                await asyncio.gather(
                    tools["dispatch"].ainvoke({"requests": [
                        {"agent": "satellite", "intent": "visualize", "variable": "ozone", "location": "the OTC"},
                    ]}),
                    tools["dispatch"].ainvoke({"requests": [
                        {"agent": "ground", "intent": "nearest_monitor", "pollutant": "NO2", "location": "Camden"},
                    ]}),
                )
        finally:
            _current_thread_id.reset(token)

        self.assertEqual(peak[0], 1)

    async def test_an_invalid_dispatch_goes_back_to_the_model_as_an_error_and_runs_nothing(self):
        """Seven requests, one over the schema cap: the turn must not crash,
        and the model must be told why so it can resend a valid call."""
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        ground, satellite = FakeGroundAgent(), FakeSatelliteAgent()
        tools = await self._tools(ground, satellite)
        requests = [{"agent": "ground", "intent": "readings", "pollutant": "NO2", "location": f"site {i}"}
                    for i in range(7)]
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([tools["dispatch"]]))
        graph.add_edge(START, "tools")

        out = await graph.compile().ainvoke({"messages": [AIMessage(
            content="", tool_calls=[{"name": "dispatch", "args": {"requests": requests}, "id": "call-1"}],
        )]})

        (message,) = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(message.status, "error")
        self.assertIn("at most 6", message.content)
        self.assertEqual((ground.tasks, satellite.tasks), ([], []))


if __name__ == "__main__":
    unittest.main()


class RepeatingDispatchModel:
    """Sends the same dispatch every time it may call tools; answers when the
    supervisor takes tools away (tool_choice "none")."""

    def __init__(self):
        self.tool_choices = []

    def bind_tools(self, tools, **kwargs):
        self.tool_choices.append(kwargs.get("tool_choice"))
        return self

    async def ainvoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage

        if self.tool_choices[-1] == "none":
            return AIMessage(content="Ozone peaked over Philadelphia.")
        return AIMessage(content="", tool_calls=[{"name": "dispatch", "id": f"call-{len(self.tool_choices)}", "args": {
            "requests": [{"agent": "satellite", "intent": "visualize", "variable": "ozone", "location": "the OTC"}],
        }}])


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class DispatchClosesTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_model_repeating_a_dispatch_loses_the_tool_and_answers(self):
        """A repeat runs nothing and changes nothing; without closing the tool
        the model loops until the recursion limit and the turn errors."""
        from langgraph.checkpoint.memory import InMemorySaver

        from tta_backend.agents import supervisor_agent
        from tta_backend.models import AgentResult
        from tta_backend.utils.streaming import stream_response

        model = RepeatingDispatchModel()
        with patch.object(supervisor_agent, "build_chat_model", return_value=model), \
             patch.object(supervisor_agent, "get_checkpointer", AsyncMock(return_value=InMemorySaver())):
            supervisor = await supervisor_agent.build_agent(ground_agent=object(), satellite_agent=object())

        run_satellite = AsyncMock(return_value=AgentResult(text="Mapped ozone."))
        with patch.object(supervisor_agent, "run_satellite", run_satellite), \
             patch.object(supervisor_agent, "persist_dispatch_contexts", AsyncMock()):
            events = [e async for e in stream_response(supervisor, "ozone over the OTC", "thread-1")]

        self.assertEqual(run_satellite.await_count, 1)
        self.assertEqual(model.tool_choices, [None, None, "none"])
        self.assertIn("Ozone peaked over Philadelphia.", "".join(d for t, d in events if t == "text"))
