"""The supervisor's checkpoint carries chart references, never chart grids.

Every chart is durably stored in agent_charts; a ToolMessage that also carries
the grid duplicates it into a LangGraph checkpoint blob that is rewritten in
full at every step of the thread, so its size grows with the square of the
turn count. These tests pin that the checkpoint holds only what history needs
to find the stored chart again.
"""

import importlib.util
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

GRID_KEYS = ("lats", "lons", "values", "panels", "frames")


def _grid_chart_payload(**extra):
    return {
        "type": "heatmap",
        "title": "NO2 over NJ",
        "variable": "nitrogendioxide_tropospheric_column",
        "units": "mol/m^2",
        "lats": [40.0, 40.5, 41.0],
        "lons": [-75.0, -74.5],
        "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        **extra,
    }


class FakeChartStore:
    """agent_charts in memory, behind chart_repository's two functions."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.saves = 0

    async def save_chart(self, thread_id, payload, user_id):
        self.saves += 1
        stored = dict(payload)
        stored.setdefault("chart_id", f"hash_{len(self.rows)}")
        stored["thread_id"] = thread_id
        stored["user_id"] = user_id
        self.rows[stored["chart_id"]] = stored
        return dict(stored)

    async def get_chart(self, chart_id):
        row = self.rows.get(chart_id)
        return dict(row) if row is not None else None

    async def get_charts(self, chart_ids, user_id):
        self.batch_reads += 1
        return {
            chart_id: dict(self.rows[chart_id])
            for chart_id in chart_ids
            if chart_id in self.rows and self.rows[chart_id]["user_id"] == user_id
        }

    @contextmanager
    def installed(self):
        self.batch_reads = 0
        with patch("tta_backend.services.chart_service.chart_repository.save_chart",
                   AsyncMock(side_effect=self.save_chart)), \
             patch("tta_backend.services.chart_service.chart_repository.get_chart",
                   AsyncMock(side_effect=self.get_chart)), \
             patch("tta_backend.services.chart_service.chart_repository.get_charts",
                   AsyncMock(side_effect=self.get_charts)):
            yield self


class RecordingSupervisor:
    def __init__(self):
        self.state_messages = []

    async def aupdate_state(self, config, values, as_node=None):
        self.state_messages.extend(values["messages"])

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.state_messages})


class UntouchedAgent:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to untouched agent: {name}")


def _tool_message_contents(messages):
    from langchain_core.messages import ToolMessage

    return [m.content for m in messages if isinstance(m, ToolMessage)]


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class FastPathCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tta_backend.services import subagent_dispatch

        subagent_dispatch.get_call_budget().clear()

    async def _run_fast_path_turn(self, supervisor, result):
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.run_satellite", AsyncMock(return_value=result)):
            return [
                event
                async for event in service.stream_chat_events(
                    supervisor, UntouchedAgent(), AsyncMock(),
                    "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "thread-1", "user-1", "req-1",
                )
            ]

    async def test_fast_path_write_back_checkpoints_no_chart_grid(self):
        from tta_backend.models import AgentResult, ChartPayload

        chart = ChartPayload.model_validate(_grid_chart_payload())
        supervisor = RecordingSupervisor()
        with FakeChartStore().installed():
            await self._run_fast_path_turn(supervisor, AgentResult(text="Plotted.", charts=[chart]))

        contents = _tool_message_contents(supervisor.state_messages)
        self.assertEqual(len(contents), 1)
        envelope = json.loads(contents[0])
        self.assertEqual(len(envelope["charts"]), 1)
        for key in GRID_KEYS:
            self.assertNotIn(key, envelope["charts"][0])

    async def test_a_reload_restores_the_full_chart_from_agent_charts(self):
        from tta_backend.models import AgentResult, ChartPayload
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        chart = ChartPayload.model_validate(_grid_chart_payload())
        supervisor = RecordingSupervisor()
        with FakeChartStore().installed():
            await self._run_fast_path_turn(supervisor, AgentResult(text="Plotted.", charts=[chart]))
            history = await HistoryService(ChartService()).build_history(supervisor, "thread-1", "user-1")

        assistant = next(m for m in history if m["role"] == "assistant" and m["charts"])
        self.assertEqual(len(assistant["charts"]), 1)
        self.assertEqual(assistant["charts"][0]["values"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class ReferenceReadTests(unittest.IsolatedAsyncioTestCase):
    async def _history_for(self, envelope_charts, store):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        supervisor = RecordingSupervisor()
        supervisor.state_messages = [
            HumanMessage(content="Plot it"),
            AIMessage(content="", tool_calls=[{"name": "ask_earthdata_agent", "args": {}, "id": "c1"}]),
            ToolMessage(
                content=json.dumps({"text": "Plotted.", "charts": envelope_charts}),
                tool_call_id="c1", name="ask_earthdata_agent",
            ),
            AIMessage(content="Here it is."),
        ]
        with store.installed():
            return await HistoryService(ChartService()).build_history(supervisor, "thread-1", "user-1")

    async def test_a_reference_to_another_users_chart_is_neither_shown_nor_overwritten(self):
        store = FakeChartStore()
        store.rows["map_1"] = {**_grid_chart_payload(chart_id="map_1"), "thread_id": "t-x", "user_id": "user-2"}
        reference = {"chart_id": "map_1", "type": "heatmap", "reference": True}

        history = await self._history_for([reference], store)

        self.assertEqual(store.saves, 0)
        self.assertEqual(store.rows["map_1"]["user_id"], "user-2")
        self.assertEqual([c for m in history for c in m.get("charts", [])], [])

    async def test_a_reference_to_a_missing_chart_saves_no_stub(self):
        store = FakeChartStore()
        reference = {"chart_id": "map_gone", "type": "heatmap", "reference": True}

        history = await self._history_for([reference], store)

        self.assertEqual(store.rows, {})
        self.assertEqual([c for m in history for c in m.get("charts", [])], [])

    async def test_a_legacy_inline_chart_still_loads(self):
        """Checkpoints written before references carry the whole grid inline."""
        store = FakeChartStore()

        history = await self._history_for([_grid_chart_payload(chart_id="map_old")], store)

        charts = [c for m in history for c in m.get("charts", [])]
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0]["lats"], [40.0, 40.5, 41.0])

    async def test_loading_history_never_writes_a_chart(self):
        """A GET must not recreate a chart whose row is gone, e.g. one a
        session delete already removed."""
        store = FakeChartStore()

        await self._history_for(
            [_grid_chart_payload(chart_id="map_deleted"), _grid_chart_payload(title="no id")],
            store,
        )

        self.assertEqual(store.saves, 0)
        self.assertEqual(store.rows, {})

    async def test_history_resolves_every_chart_in_one_read(self):
        store = FakeChartStore()
        for chart_id in ("map_1", "map_2", "map_3"):
            store.rows[chart_id] = {**_grid_chart_payload(chart_id=chart_id), "thread_id": "thread-1", "user_id": "user-1"}
        references = [{"chart_id": c, "type": "heatmap", "reference": True} for c in ("map_1", "map_2", "map_3")]

        history = await self._history_for(references, store)

        self.assertEqual(store.batch_reads, 1)
        charts = [c for m in history for c in m.get("charts", [])]
        self.assertEqual([c["chart_id"] for c in charts], ["map_1", "map_2", "map_3"])
        self.assertEqual(charts[0]["values"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class ToolThenAnswerModel:
    """Calls ask_earthdata_agent once, then answers."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kw):
        return self

    async def ainvoke(self, *a, **kw):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "ask_earthdata_agent", "args": {"task": "plot it"}, "id": "call-1"},
            ])
        return AIMessage(content="Here is the map.")


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class SupervisorPathCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def _build_supervisor(self, checkpointer):
        with patch("tta_backend.agents.supervisor_agent.build_chat_model", return_value=ToolThenAnswerModel()), \
             patch("tta_backend.agents.supervisor_agent.get_checkpointer", return_value=checkpointer):
            from tta_backend.agents.supervisor_agent import build_agent

            return await build_agent(ground_agent=object(), satellite_agent=object())

    async def _run_supervisor_turn(self, supervisor, result):
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.agents.supervisor_agent.run_satellite", AsyncMock(return_value=result)):
            return [
                event
                async for event in service.stream_chat_events(
                    supervisor, UntouchedAgent(), UntouchedAgent(),
                    "What does that look like?", "thread-1", "user-1", "req-1",
                )
            ]

    async def test_supervisor_tool_result_checkpoints_no_chart_grid(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from tta_backend.models import AgentResult, ChartPayload

        supervisor = await self._build_supervisor(InMemorySaver())
        chart = ChartPayload.model_validate(_grid_chart_payload())
        with FakeChartStore().installed():
            await self._run_supervisor_turn(supervisor, AgentResult(text="Plotted.", charts=[chart]))

        state = await supervisor.aget_state({"configurable": {"thread_id": "thread-1"}})
        contents = _tool_message_contents(state.values["messages"])
        self.assertEqual(len(contents), 1)
        envelope = json.loads(contents[0])
        self.assertEqual(len(envelope["charts"]), 1)
        for key in GRID_KEYS:
            self.assertNotIn(key, envelope["charts"][0])

    async def test_supervisor_turn_still_streams_the_full_chart(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from tta_backend.models import AgentResult, ChartPayload

        supervisor = await self._build_supervisor(InMemorySaver())
        chart = ChartPayload.model_validate(_grid_chart_payload())
        with FakeChartStore().installed():
            events = await self._run_supervisor_turn(supervisor, AgentResult(text="Plotted.", charts=[chart]))

        chart_events = [
            json.loads(e.split("data: ", 1)[1]) for e in events if e.startswith("event: chart\n")
        ]
        self.assertEqual(len(chart_events), 1)
        self.assertEqual(chart_events[0]["values"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
