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
class RunSatelliteChartHarvestingTests(unittest.IsolatedAsyncioTestCase):
    """ask_earthdata_agent's _run_satellite now harvests charts from the
    ("chart_payload", dict) events emit_chart produces (T13), not by parsing
    ChartPayload out of tool_result content — plot/comparison tool results
    are compact summaries now and no longer parse as a ChartPayload."""

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

    async def test_harvests_a_chart_from_a_chart_payload_event(self):
        from models import AgentResult

        envelope = json.dumps({"summary": "Plotted NO2 over NJ.", "artifact_ids": [], "handles": ["obs_1"]})

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                from utils.streaming import emit_chart

                emit_chart({"type": "heatmap", "chart_id": "map_1", "values": [[1.0]], "title": "NO2"})
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        ask_earthdata_agent = await self._build_ask_earthdata_agent(FakeSatelliteAgent())

        raw = await ask_earthdata_agent.ainvoke({"task": "Plot NO2 over NJ"})
        result = AgentResult.model_validate_json(raw)

        self.assertEqual(len(result.charts), 1)
        chart = result.charts[0].model_dump(exclude_none=True)
        self.assertEqual(chart["chart_id"], "map_1")
        self.assertEqual(chart["values"], [[1.0]])

    async def test_a_compact_tool_result_does_not_double_count_as_a_chart(self):
        """The compact tool_result content (render_type, not type) must not be
        mis-parsed as a ChartPayload — charts come exclusively from
        chart_payload events now."""
        from models import AgentResult

        compact_tool_result = json.dumps({
            "render_type": "heatmap",
            "chart_id": "map_2",
            "_artifact_refs": [{"id": "map_2", "type": "map"}],
        })
        envelope = json.dumps({
            "summary": "Plotted NO2 over NJ.", "artifact_ids": ["map_2"], "handles": ["obs_1"],
        })

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                yield "updates", {
                    "tools": {"messages": [
                        SimpleNamespace(name="plot_singular", content=compact_tool_result, tool_calls=None),
                    ]},
                }
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        ask_earthdata_agent = await self._build_ask_earthdata_agent(FakeSatelliteAgent())

        raw = await ask_earthdata_agent.ainvoke({"task": "Plot NO2 over NJ"})
        result = AgentResult.model_validate_json(raw)

        self.assertEqual(result.charts, [])
        self.assertEqual([a.id for a in result.artifacts], ["map_2"])


if __name__ == "__main__":
    unittest.main()
