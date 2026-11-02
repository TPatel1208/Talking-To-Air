import importlib.util
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class AgentHelperTests(unittest.TestCase):
    def test_truncate_text_logs_warning_with_lengths(self):
        from tta_backend.utils.message_utils import truncate_text

        with self.assertLogs("tta_backend.utils.message_utils", level="WARNING") as captured:
            result = truncate_text("abcdef", 3, "satellite", "req-1")

        self.assertEqual(result, "abc")
        self.assertIn("response_truncated", captured.output[0])

    def test_extract_last_text_handles_list_content(self):
        from tta_backend.utils.message_utils import extract_last_text

        class Message:
            content = [{"type": "text", "text": "hello"}, {"type": "thinking", "text": "hidden"}]

        text = extract_last_text({"messages": [Message()]}, "fallback", agent_name="ground")

        self.assertEqual(text, "hello")

    def test_compact_model_input_preserves_anonymous_chart(self):
        from tta_backend.agents.supervisor_agent import _compact_model_input_content
        from tta_backend.models import AgentResult, ChartPayload, agent_result_to_json

        raw = agent_result_to_json(
            AgentResult(
                text="Here is the result.",
                charts=[ChartPayload(type="", title="", metadata={})],
            )
        )

        compacted = _compact_model_input_content(raw)

        self.assertEqual(compacted, "Here is the result.\n\nCharts generated: chart")

    def test_compaction_leaves_non_tool_messages_alone_without_parsing_them(self):
        """Only a ToolMessage carries a sub-agent envelope. trim_middleware
        compacts the whole history before every model call, so parsing every
        human/AI message is wasted JSON validation that grows with the thread."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from tta_backend.agents import supervisor_agent
        from tta_backend.models import AgentResult, ChartPayload, agent_result_to_json

        envelope = agent_result_to_json(AgentResult(text="Plotted.", charts=[ChartPayload(type="heatmap")]))
        human, answer = HumanMessage(content="plot NO2"), AIMessage(content="NO2 peaked near Newark.")

        with patch.object(supervisor_agent, "parse_dispatch_result", side_effect=AssertionError("parsed")), \
             patch.object(supervisor_agent, "parse_agent_result", side_effect=AssertionError("parsed")):
            self.assertIs(supervisor_agent._compact_model_input_message(human), human)
            self.assertIs(supervisor_agent._compact_model_input_message(answer), answer)

        tool = ToolMessage(content=envelope, tool_call_id="c1", name="dispatch")
        self.assertEqual(
            supervisor_agent._compact_model_input_message(tool).content,
            "Plotted.\n\nCharts generated: heatmap",
        )

    def test_compact_model_input_reduces_a_dispatch_batch_to_per_item_text_without_grids(self):
        """T64 D12: a batch is not an AgentResult, so without its own branch
        it would reach the supervisor model as raw JSON, chart grids and all."""
        from tta_backend.agents.supervisor_agent import _compact_model_input_content
        from tta_backend.models import AgentResult, ChartPayload, DispatchItemResult, DispatchResult

        grid = ChartPayload.model_validate({
            "type": "heatmap", "title": "NO2 over NJ", "variable": "NO2",
            "lats": [40.0, 40.5], "lons": [-75.0, -74.5], "values": [[1.5e15, 2.5e15], [3.5e15, 4.5e15]],
        })
        raw = DispatchResult(results=[
            DispatchItemResult(index=0, agent="satellite", intent="visualize", status="ok",
                               result=AgentResult(text="NO2 peaks near Newark.", charts=[grid])),
            DispatchItemResult(index=1, agent="ground", intent="readings", status="rejected",
                               reason="per-turn budget exhausted"),
            DispatchItemResult(index=2, agent="ground", intent="readings", status="duplicate",
                               result=AgentResult(text="Max 41 ppb on 07-02.")),
        ]).model_dump_json(exclude_none=True)

        compacted = _compact_model_input_content(raw)

        self.assertEqual(
            compacted,
            "[1] satellite/visualize: ok\n"
            "NO2 peaks near Newark.\n"
            "Charts generated: heatmap 'NO2 over NJ', variable=NO2, grid=2x2\n"
            "\n"
            "[2] ground/readings: rejected (per-turn budget exhausted)\n"
            "\n"
            "[3] ground/readings: duplicate\n"
            "Max 41 ppb on 07-02.",
        )

    def test_build_agent_builds_the_supervisor_model_via_the_factory(self):
        from tta_backend.agents import supervisor_agent

        created = object()
        with patch.object(
            supervisor_agent, "build_chat_model", return_value="llm"
        ) as factory, patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            result = asyncio.run(
                supervisor_agent.build_agent(
                    model="configured-model", provider="groq",
                    ground_agent="ground", satellite_agent="satellite",
                )
            )

        self.assertIs(result, created)
        self.assertEqual(factory.call_args.args[0], "groq")
        self.assertEqual(factory.call_args.args[1], "configured-model")

    def test_build_agent_defaults_the_supervisor_provider_from_settings(self):
        from tta_backend.agents import supervisor_agent
        from tta_backend.config.settings import get_settings

        created = object()
        with patch.object(
            supervisor_agent, "build_chat_model", return_value="llm"
        ) as factory, patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            asyncio.run(supervisor_agent.build_agent(ground_agent="ground", satellite_agent="satellite"))

        self.assertEqual(factory.call_args.args[0], get_settings().supervisor_model_provider)


if __name__ == "__main__":
    unittest.main()
