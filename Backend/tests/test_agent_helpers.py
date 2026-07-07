import importlib.util
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class AgentHelperTests(unittest.TestCase):
    def test_truncate_text_logs_warning_with_lengths(self):
        from utils.message_utils import truncate_text

        with self.assertLogs("utils.message_utils", level="WARNING") as captured:
            result = truncate_text("abcdef", 3, "satellite", "req-1")

        self.assertEqual(result, "abc")
        self.assertIn("response_truncated", captured.output[0])

    def test_extract_last_text_handles_list_content(self):
        from utils.message_utils import extract_last_text

        class Message:
            content = [{"type": "text", "text": "hello"}, {"type": "thinking", "text": "hidden"}]

        text = extract_last_text({"messages": [Message()]}, "fallback", agent_name="ground")

        self.assertEqual(text, "hello")

    def test_compact_model_input_preserves_anonymous_chart(self):
        from agents.supervisor_agent import _compact_model_input_content
        from models import AgentResult, ChartPayload, agent_result_to_json

        raw = agent_result_to_json(
            AgentResult(
                text="Here is the result.",
                charts=[ChartPayload(type="", title="", metadata={})],
            )
        )

        compacted = _compact_model_input_content(raw)

        self.assertEqual(compacted, "Here is the result.\n\nCharts generated: chart")

    def test_build_agent_builds_the_supervisor_model_via_the_factory(self):
        from agents import supervisor_agent

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
        from agents import supervisor_agent
        from config.settings import get_settings

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

    def test_build_agent_wires_the_passed_in_subagents_into_the_dispatch_tools(self):
        from agents import supervisor_agent

        captured = {}

        def fake_create_agent(*, tools, **kwargs):
            captured["tools"] = {t.name: t for t in tools}
            return object()

        with patch.object(
            supervisor_agent, "build_chat_model", return_value="llm"
        ), patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", side_effect=fake_create_agent
        ), patch.object(
            supervisor_agent, "run_ground", AsyncMock(return_value=Mock(model_dump_json=lambda **_: "{}"))
        ) as run_ground, patch.object(
            supervisor_agent, "run_satellite", AsyncMock(return_value=Mock(model_dump_json=lambda **_: "{}"))
        ) as run_satellite:
            asyncio.run(
                supervisor_agent.build_agent(ground_agent="the-ground-agent", satellite_agent="the-satellite-agent")
            )
            asyncio.run(captured["tools"]["ask_ground_sensor_agent"].ainvoke({"task": "find nearest monitor"}))
            asyncio.run(captured["tools"]["ask_earthdata_agent"].ainvoke({"task": "plot NO2"}))

        self.assertEqual(run_ground.call_args.args[0], "the-ground-agent")
        self.assertEqual(run_satellite.call_args.args[0], "the-satellite-agent")


if __name__ == "__main__":
    unittest.main()
