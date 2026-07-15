import asyncio
import importlib.util
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class FakeGroundAgent:
    """Mirrors the shape services.subagent_dispatch.run_ground expects:
    a stateless LangGraph agent invoked via ainvoke()."""

    def __init__(self, envelope_text=None, raises=None):
        self.envelope_text = envelope_text or json.dumps({
            "summary": "The closest NO2 monitor is Rutgers University.", "artifact_ids": [], "handles": [],
        })
        self.raises = raises
        self.invocations = []

    async def ainvoke(self, input_, config):
        self.invocations.append(input_["messages"][0].content)
        if self.raises is not None:
            raise self.raises
        return {"messages": [SimpleNamespace(content=self.envelope_text, type="ai")]}


class FakeSatelliteAgent:
    """Mirrors the shape services.subagent_dispatch.run_satellite expects:
    a stateless LangGraph agent invoked via stream_response's astream()."""

    def __init__(self, envelope_text=None):
        self.envelope_text = envelope_text or json.dumps({
            "summary": "Plotted NO2 over New Jersey.", "artifact_ids": [], "handles": ["obs_1"],
        })
        self.invocations = 0

    async def astream(self, input_, config, stream_mode):
        self.invocations += 1
        yield "updates", {
            "agent": {"messages": [
                SimpleNamespace(tool_calls=[{"id": "tc1", "name": "plot_singular", "args": {}}], content=""),
            ]},
        }
        await asyncio.sleep(0)
        yield "messages", (SimpleNamespace(content=self.envelope_text, type="ai", tool_calls=None), {})


class UntouchedAgent:
    """Fails the test loudly if the fast path (wrongly) invokes it."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to untouched agent: {name}")


class FakeSupervisorAgent:
    """A minimal stand-in for the checkpointed LangGraph supervisor: records
    aupdate_state calls and, on astream, echoes back what it has accumulated
    so a follow-up turn can prove the fast-pathed exchange is visible."""

    def __init__(self):
        self.state_messages = []
        self.update_state_calls = []

    async def aupdate_state(self, config, values, as_node=None):
        self.update_state_calls.append((config, values, as_node))
        self.state_messages.extend(values["messages"])

    async def astream(self, input_, config, stream_mode):
        history = " | ".join(getattr(m, "content", "") for m in self.state_messages)
        new_message = input_["messages"][0]["content"]
        yield "messages", (
            SimpleNamespace(content=f"Agent consulted: GROUND\n\n[history={history}] {new_message}", type="ai", tool_calls=None),
            {},
        )


def _no_monitor_context():
    from services import subagent_dispatch

    return (
        patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})),
        patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()),
    )


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class RouterFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from services import subagent_dispatch

        subagent_dispatch.get_call_budget().clear()

    async def test_ground_only_message_invokes_only_the_ground_agent(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = FakeGroundAgent()
        satellite_agent = UntouchedAgent()
        supervisor_agent = AsyncMock()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            events = [
                event
                async for event in service.stream_chat_events(
                    supervisor_agent, ground_agent, satellite_agent,
                    "Find the nearest NO2 monitor to Tampa FL", "thread-1", "user-1", "req-1",
                )
            ]

        self.assertEqual(len(ground_agent.invocations), 1)
        joined = "".join(events)
        self.assertIn("event: text", joined)
        self.assertIn("Agent consulted: GROUND", joined)
        self.assertIn("event: done", joined)

    async def test_satellite_only_message_invokes_only_the_satellite_agent(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = UntouchedAgent()
        satellite_agent = FakeSatelliteAgent()
        supervisor_agent = AsyncMock()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        events = [
            event
            async for event in service.stream_chat_events(
                supervisor_agent, ground_agent, satellite_agent,
                "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", "user-1", "req-1",
            )
        ]

        self.assertEqual(satellite_agent.invocations, 1)
        joined = "".join(events)
        self.assertIn("event: tool_call", joined)  # forwarded live from the sub-agent's own stream
        self.assertIn("Agent consulted: SATELLITE", joined)
        self.assertIn("event: done", joined)

    async def test_satellite_fast_path_forwards_the_stage_and_detail_fields_over_sse(self):
        """T19: chat_stream_service must not rebuild the status SSE payload
        as message-only — a stage-tagged emit_status call deep in the
        sub-agent's own stream has to survive all the way to the wire, or
        the frontend's workflow strip never lights up."""
        import json as _json

        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService
        from utils.streaming import emit_status

        class StageEmittingSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Checking coverage...", stage="coverage", detail=14)
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(
                    content=json.dumps({"summary": "Plotted NO2.", "artifact_ids": [], "handles": ["obs_1"]}),
                    type="ai", tool_calls=None,
                ), {})

        ground_agent = UntouchedAgent()
        satellite_agent = StageEmittingSatelliteAgent()
        supervisor_agent = AsyncMock()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        events = [
            event
            async for event in service.stream_chat_events(
                supervisor_agent, ground_agent, satellite_agent,
                "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", "user-1", "req-1",
            )
        ]

        status_lines = [line for line in "".join(events).split("\n\n") if line.startswith("event: status")]
        self.assertTrue(status_lines, "expected at least one status event")
        payload = _json.loads(status_lines[0].split("data: ", 1)[1])
        self.assertEqual(payload["stage"], "coverage")
        self.assertEqual(payload["detail"], 14)

    async def test_ambiguous_message_uses_the_supervisor_and_never_touches_subagents(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "text", "Agent consulted: GROUND + SATELLITE\n\nHere is the synthesis."

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(
                    object(), UntouchedAgent(), UntouchedAgent(),
                    "Compare ground NO2 to TROPOMI over Austin", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        self.assertIn("Here is the synthesis.", joined)

    async def test_fast_path_done_event_carries_suggestions_from_a_well_formed_envelope(self):
        """T22 story #9: the done event is the additive surface for the
        finalized envelope's suggestions on the router fast path."""
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        envelope = json.dumps({
            "summary": "The closest NO2 monitor is Rutgers University.",
            "artifact_ids": [], "handles": [],
            "suggested_followups": ["What about last month?", "Any exceedances nearby?"],
        })
        ground_agent = FakeGroundAgent(envelope_text=envelope)
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            events = [
                event
                async for event in service.stream_chat_events(
                    AsyncMock(), ground_agent, UntouchedAgent(),
                    "Find the nearest NO2 monitor to Tampa FL", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        done_line = next(line for line in joined.split("\n\n") if line.startswith("event: done"))
        payload = json.loads(done_line.split("data: ", 1)[1])
        self.assertEqual(
            payload["suggested_followups"],
            ["What about last month?", "Any exceedances nearby?"],
        )

    async def test_fast_path_done_event_omits_suggestions_when_the_envelope_has_none(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = FakeGroundAgent()  # default envelope has no suggested_followups key
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            events = [
                event
                async for event in service.stream_chat_events(
                    AsyncMock(), ground_agent, UntouchedAgent(),
                    "Find the nearest NO2 monitor to Tampa FL", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        done_line = next(line for line in joined.split("\n\n") if line.startswith("event: done"))
        payload = json.loads(done_line.split("data: ", 1)[1])
        self.assertNotIn("suggested_followups", payload)

    async def test_fast_path_done_event_omits_suggestions_for_a_salvaged_result(self):
        """T22 story #7/#12: a malformed final message is salvaged from raw
        prose (T15) — it must never carry suggestions, even when the
        salvaged prose happens to contain question marks."""
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = FakeGroundAgent(envelope_text="The nearest monitor is Rutgers. What about last month?")
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            events = [
                event
                async for event in service.stream_chat_events(
                    AsyncMock(), ground_agent, UntouchedAgent(),
                    "Find the nearest NO2 monitor to Tampa FL", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        done_line = next(line for line in joined.split("\n\n") if line.startswith("event: done"))
        payload = json.loads(done_line.split("data: ", 1)[1])
        self.assertNotIn("suggested_followups", payload)

    async def test_supervisor_path_done_event_carries_suggestions_from_a_sub_agent_tool_result(self):
        """T22 story #8: the supervisor's synthesis must not strip a
        sub-agent's suggestions — chat_stream_service reads them straight
        off the tool_result envelope, not from the supervisor's own prose."""
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService
        from models import AgentResult, agent_result_to_json

        sub_agent_result = agent_result_to_json(AgentResult(
            text="The ground monitor reads 12 ppb.",
            suggested_followups=["How does that compare to satellite data?"],
        ))

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "tool_result", {"content": sub_agent_result}
            yield "text", "Agent consulted: GROUND + SATELLITE\n\nHere is the synthesis."

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(
                    object(), UntouchedAgent(), UntouchedAgent(),
                    "Compare ground NO2 to TROPOMI over Austin", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        done_line = next(line for line in joined.split("\n\n") if line.startswith("event: done"))
        payload = json.loads(done_line.split("data: ", 1)[1])
        self.assertEqual(payload["suggested_followups"], ["How does that compare to satellite data?"])

    async def test_sub_agent_failure_yields_error_and_does_not_write_back(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = FakeGroundAgent(raises=TimeoutError("AQS timed out"))
        supervisor_agent = AsyncMock()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            events = [
                event
                async for event in service.stream_chat_events(
                    supervisor_agent, ground_agent, UntouchedAgent(),
                    "Find the nearest NO2 monitor to Tampa FL", "thread-1", "user-1", "req-1",
                )
            ]

        joined = "".join(events)
        self.assertIn("event: error", joined)
        self.assertNotIn("event: done", joined)
        supervisor_agent.aupdate_state.assert_not_called()

    async def test_fast_pathed_turn_is_written_back_and_visible_to_the_next_supervisor_turn(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        ground_agent = FakeGroundAgent()
        supervisor = FakeSupervisorAgent()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        get_ctx, save_ctx = _no_monitor_context()
        with get_ctx, save_ctx:
            first_turn_message = "Find the nearest NO2 monitor to Tampa FL"
            [
                event
                async for event in service.stream_chat_events(
                    supervisor, ground_agent, UntouchedAgent(),
                    first_turn_message, "thread-1", "user-1", "req-1",
                )
            ]

        self.assertEqual(len(supervisor.update_state_calls), 1)

        # A genuinely ambiguous follow-up takes the supervisor path — its
        # input must now contain the fast-pathed exchange.
        events = [
            event
            async for event in service.stream_chat_events(
                supervisor, UntouchedAgent(), UntouchedAgent(),
                "How does that compare to last month?", "thread-1", "user-1", "req-2",
            )
        ]

        joined = "".join(events)
        self.assertIn(first_turn_message, joined)
        self.assertIn("Rutgers University", joined)

    async def test_fast_pathed_chart_card_survives_a_reload(self):
        """Regression: a fast-pathed turn's write-back used to append only a
        bare Human/AI pair to the supervisor's checkpointed thread. On
        reload, HistoryService only ever reconstructs a message's chart/
        artifact cards from a role=="tool" message (_attach_tool_output) —
        so the chart the live stream showed vanished after a refresh even
        though it was already durably persisted in agent_charts. The
        write-back must now also carry a ToolMessage with the full
        AgentResult envelope, the same shape the supervisor's own
        ask_earthdata_agent tool call produces."""
        from models import AgentResult, ChartPayload
        from services.chart_service import ChartService
        from services.chat_stream_service import ChatStreamService
        from services.history_service import HistoryService

        chart = ChartPayload(type="heatmap", chart_id="chart_xyz", title="NO2 over NJ")
        satellite_result = AgentResult(text="Plotted NO2 over New Jersey.", charts=[chart])

        supervisor = FakeSupervisorAgent()
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        saved = {}

        async def fake_save_chart(thread_id, payload, user_id):
            saved.setdefault(payload["chart_id"], {**payload, "thread_id": thread_id, "user_id": user_id})
            return saved[payload["chart_id"]]

        with patch("services.chat_stream_service.run_satellite", AsyncMock(return_value=satellite_result)), \
             patch("services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=fake_save_chart)):
            [
                event
                async for event in service.stream_chat_events(
                    supervisor, UntouchedAgent(), AsyncMock(),
                    "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", "user-1", "req-1",
                )
            ]

            class ReloadedAgent:
                async def aget_state(self, config):
                    return SimpleNamespace(values={"messages": supervisor.state_messages})

            history = await HistoryService(ChartService()).build_history(ReloadedAgent(), "thread-1", "user-1")

        assistant = next(m for m in history if m["role"] == "assistant" and m["content"])
        self.assertEqual(len(assistant["charts"]), 1)
        self.assertEqual(assistant["charts"][0]["chart_id"], "chart_xyz")

    async def test_two_consecutive_fast_pathed_turns_on_one_thread_both_persist(self):
        """Regression: LangGraph's aupdate_state only auto-infers as_node
        when a thread's checkpoint has never been manually updated before —
        a second fast-pathed write-back on the same thread, with no as_node,
        hits genuinely ambiguous versions_seen and raises
        InvalidUpdateError, silently dropping that turn (and every later
        one) from history, not just its chart card. A hand-rolled fake
        agent (FakeSupervisorAgent above) can't reproduce this — it's a real
        LangGraph pregel behavior — so this drives the actual supervisor
        graph (agents.supervisor_agent.build_agent) against an in-memory
        checkpointer."""
        from langgraph.checkpoint.memory import InMemorySaver

        from models import AgentResult
        from services.chart_service import ChartService
        from services.chat_stream_service import ChatStreamService

        class FakeModel:
            def bind_tools(self, tools, **kw):
                return self

            async def ainvoke(self, *a, **kw):
                return SimpleNamespace(content="hi", tool_calls=[])

        with patch("agents.supervisor_agent.build_chat_model", return_value=FakeModel()), \
             patch("agents.supervisor_agent.get_checkpointer", return_value=InMemorySaver()):
            from agents.supervisor_agent import build_agent

            supervisor = await build_agent(ground_agent=object(), satellite_agent=object())

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        thread_id = "thread-shared"

        for i in range(2):
            turn_result = AgentResult(text=f"Plotted NO2 over New Jersey, turn {i}.")
            with patch("services.chat_stream_service.run_satellite", AsyncMock(return_value=turn_result)):
                events = [
                    event
                    async for event in service.stream_chat_events(
                        supervisor, UntouchedAgent(), AsyncMock(),
                        f"Plot TROPOMI NO2 over New Jersey for 2024-01-{15 + i}",
                        thread_id, "user-1", f"req-{i}",
                    )
                ]
            joined = "".join(events)
            self.assertNotIn("event: error", joined, f"turn {i} write-back must not fail")

        state = await supervisor.aget_state({"configurable": {"thread_id": thread_id}})
        human_messages = [m for m in state.values["messages"] if getattr(m, "type", None) == "human"]
        self.assertEqual(len(human_messages), 2, "the second turn's write-back must not be silently dropped")


if __name__ == "__main__":
    unittest.main()
