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


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class AskEarthdataAgentStageForwardingTests(unittest.IsolatedAsyncioTestCase):
    """T19: narration must not be a fast-path-only feature — a status/job_
    progress/chart_payload event the satellite sub-agent emits deep inside
    ask_earthdata_agent's own nested stream_response call must still reach
    the supervisor's own outer stream_response (and therefore the SSE
    stream), even when the tool call passes through a ToolNode-style Task
    boundary (asyncio.gather spawns a fresh Task with a copied context per
    call). This already holds via the same ContextVar-bubbling mechanism
    T13/T14 established for chart_payload/job_progress — see utils/
    streaming.py's publish_status/publish_job_progress/publish_chart_
    payload, each of which forwards to a `parent_emitter` captured once at
    the nested stream_response's own construction, not re-read per call —
    this test pins that stage status events bubble identically now that
    emit_status carries a stage/detail payload (T19), and guards against a
    future change (e.g. an explicit on_event re-forward) double-emitting."""

    async def _build_ask_earthdata_agent(self, fake_satellite_agent):
        from agents import supervisor_agent

        captured = {}

        def fake_create_agent(*, tools, **kwargs):
            captured["tools"] = {t.name: t for t in tools}
            return object()

        with patch.object(supervisor_agent, "build_chat_model", return_value="llm"), \
             patch.object(supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")), \
             patch.object(supervisor_agent, "create_agent", side_effect=fake_create_agent):
            await supervisor_agent.build_agent(ground_agent="ground", satellite_agent=fake_satellite_agent)

        return captured["tools"]["ask_earthdata_agent"]

    async def test_a_sub_agent_stage_status_reaches_the_outer_stream_response(self):
        from utils.streaming import emit_status, stream_response

        envelope = json.dumps({"summary": "Plotted NO2 over NJ.", "artifact_ids": [], "handles": ["obs_1"]})

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Searching datasets...", stage="search")
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        ask_earthdata_agent = await self._build_ask_earthdata_agent(FakeSatelliteAgent())

        class OuterAgent:
            async def astream(self, input_, config, stream_mode):
                # asyncio.gather (not a plain await) — the real ToolNode
                # dispatch shape: a fresh Task per tool call, context copied
                # at Task-creation time (see get_call_budget()'s docstring).
                await asyncio.gather(ask_earthdata_agent.ainvoke({"task": "Plot NO2 over NJ"}))
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="outer done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(OuterAgent(), "do it", "outer-thread")]

        status_events = [data for event_type, data in events if event_type == "status"]
        self.assertEqual([s.get("stage") for s in status_events], ["search"])
        self.assertEqual(status_events[0]["message"], "Searching datasets...")

    async def test_a_sub_agent_chart_and_job_progress_also_reach_the_outer_stream(self):
        from utils.streaming import emit_chart, emit_job_progress, stream_response

        envelope = json.dumps({"summary": "Plotted NO2 over NJ.", "artifact_ids": [], "handles": ["obs_1"]})

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                emit_job_progress("job_1", "processing", 40, "materializing", "40%")
                emit_chart({"type": "heatmap", "chart_id": "map_1", "values": [[1.0]]})
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        ask_earthdata_agent = await self._build_ask_earthdata_agent(FakeSatelliteAgent())

        class OuterAgent:
            async def astream(self, input_, config, stream_mode):
                # asyncio.gather (not a plain await) — the real ToolNode
                # dispatch shape: a fresh Task per tool call, context copied
                # at Task-creation time (see get_call_budget()'s docstring).
                await asyncio.gather(ask_earthdata_agent.ainvoke({"task": "Plot NO2 over NJ"}))
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="outer done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(OuterAgent(), "do it", "outer-thread")]

        job_events = [data for event_type, data in events if event_type == "job_progress"]
        chart_events = [data for event_type, data in events if event_type == "chart_payload"]
        self.assertEqual([e["status"] for e in job_events], ["processing"])
        self.assertEqual([c["chart_id"] for c in chart_events], ["map_1"])


if __name__ == "__main__":
    unittest.main()
