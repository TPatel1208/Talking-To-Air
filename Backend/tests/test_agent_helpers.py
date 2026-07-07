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

    def test_ground_tool_failure_detects_provider_tool_call_errors(self):
        from agents.supervisor_agent import _is_ground_tool_failure

        self.assertTrue(_is_ground_tool_failure("Error: Failed to call a function. failed_generation"))
        self.assertTrue(_is_ground_tool_failure("tool call validation failed: parameters for tool did not match schema"))
        self.assertFalse(_is_ground_tool_failure("The monitor is Rutgers University."))

    def test_ground_context_extracts_and_injects_monitor_facts(self):
        from agents.supervisor_agent import _extract_ground_monitor_context, _inject_ground_context

        context = _extract_ground_monitor_context(
            "The closest NO2 monitor is Rutgers University with station_id 34-023-0011 "
            "located at coordinates (40.462182, -74.429439)."
        )

        self.assertEqual(context["name"], "Rutgers University")
        self.assertEqual(context["site_id"], "34-023-0011")
        self.assertEqual(context["latitude"], "40.462182")
        self.assertEqual(context["longitude"], "-74.429439")
        # Pollutant is intentionally NOT extracted — the user's request is always
        # authoritative and injecting a stale pollutant caused param_code confusion.
        self.assertNotIn("pollutant", context)
        enriched = _inject_ground_context("Give quarterly summary for Q1 2024.", context)
        self.assertIn("station_id=34-023-0011", enriched)
        self.assertIn("coordinates=(40.462182, -74.429439)", enriched)
        self.assertNotIn("pollutant=", enriched)

    def test_ground_context_does_not_bleed_pollutant_across_requests(self):
        from agents.supervisor_agent import _extract_ground_monitor_context, _inject_ground_context

        # Simulate a failure response that mentions multiple pollutants — the first
        # match used to be SO2, which then poisoned a subsequent NO2 request.
        failure_text = (
            "The Chester, NJ monitor has no daily SO2 records for the requested period. "
            "No NO2 or PM2.5 data was found either."
        )
        context = _extract_ground_monitor_context(failure_text)
        self.assertNotIn("pollutant", context)

        enriched = _inject_ground_context("Give daily NO2 summary for NJ in Jan 2024.", context)
        self.assertNotIn("pollutant=", enriched)

    def test_finalize_sub_agent_result_resolves_matching_artifact_ids_and_handles(self):
        from agents.supervisor_agent import _finalize_sub_agent_result
        from models import AgentResult
        from models.artifact import ArtifactReference

        raw = AgentResult(
            text='{"summary": "Found the closest monitor.", "artifact_ids": ["art_1"], "handles": ["obs_1"]}',
            artifacts=[
                ArtifactReference(id="art_1", type="table"),
                ArtifactReference(id="art_2", type="table"),
            ],
        )

        finalized = _finalize_sub_agent_result(raw, "ground sensor")

        self.assertEqual(finalized.text, "Found the closest monitor.")
        self.assertEqual([a.id for a in finalized.artifacts], ["art_1"])
        self.assertEqual(finalized.handles, ["obs_1"])

    def test_finalize_sub_agent_result_drops_unknown_artifact_ids(self):
        from agents.supervisor_agent import _finalize_sub_agent_result
        from models import AgentResult

        raw = AgentResult(text='{"summary": "ok", "artifact_ids": ["missing"], "handles": []}')

        finalized = _finalize_sub_agent_result(raw, "ground sensor")

        self.assertEqual(finalized.artifacts, [])

    def test_finalize_sub_agent_result_is_a_structured_failure_on_invalid_envelope(self):
        from agents.supervisor_agent import _finalize_sub_agent_result
        from models import AgentResult, ChartPayload

        raw = AgentResult(
            text="I found 3 monitors near Newark, NJ.",
            charts=[ChartPayload(type="heatmap", title="NO2")],
        )

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        self.assertNotEqual(finalized.text, raw.text)
        self.assertEqual(finalized.metadata.get("error"), "invalid_envelope")
        self.assertIn("earthdata", finalized.text.lower())
        # Charts already produced this turn are not silently discarded.
        self.assertEqual(len(finalized.charts), 1)

    def test_finalize_sub_agent_result_parses_an_envelope_longer_than_the_display_limit(self):
        import json

        from agents.supervisor_agent import _finalize_sub_agent_result
        from models import AgentResult
        from models.artifact import ArtifactReference

        long_summary = "A" * 2500
        raw_text = json.dumps({
            "summary": long_summary,
            "artifact_ids": ["art_1"],
            "handles": ["obs_1"],
        })
        self.assertGreater(len(raw_text), 2000)

        raw = AgentResult(
            text=raw_text,
            artifacts=[ArtifactReference(id="art_1", type="table")],
        )

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        self.assertNotEqual(finalized.metadata.get("error"), "invalid_envelope")
        self.assertEqual([a.id for a in finalized.artifacts], ["art_1"])
        self.assertEqual(finalized.handles, ["obs_1"])
        # The extracted summary is truncated for display only after parsing.
        self.assertEqual(len(finalized.text), 2000)

    def test_satellite_retry_task_names_only_the_sanctioned_toolset(self):
        from agents.supervisor_agent import _satellite_retry_task
        from tools.satellite_tools.factory import sanctioned_tool_names

        retry_task = _satellite_retry_task("Plot TROPOMI NO2 over New Jersey")

        for name in sanctioned_tool_names():
            self.assertIn(name, retry_task)
        # Demoted/removed tools must never appear in retry guidance.
        for stale in ("geocode_location", "get_retrieval_status", "summarize_dataset"):
            self.assertNotIn(stale, retry_task)
        self.assertIn("Task: Plot TROPOMI NO2 over New Jersey", retry_task)

    def test_ground_retry_guidance_names_only_ground_tools(self):
        from agents.supervisor_agent import _GROUND_RETRY_TOOL_GUIDANCE
        from tools import GROUND_TOOLS

        for tool in GROUND_TOOLS:
            self.assertIn(tool.name, _GROUND_RETRY_TOOL_GUIDANCE)
        self.assertNotIn("geocode_location", _GROUND_RETRY_TOOL_GUIDANCE)

    def test_build_agent_builds_the_supervisor_model_via_the_factory(self):
        from agents import supervisor_agent

        created = object()
        with patch.object(
            supervisor_agent, "build_chat_model", return_value="llm"
        ) as factory, patch.object(
            supervisor_agent, "build_ground_agent", return_value="ground"
        ), patch.object(
            supervisor_agent, "build_earthdata_agent", return_value="satellite"
        ), patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            result = asyncio.run(
                supervisor_agent.build_agent(model="configured-model", provider="groq")
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
            supervisor_agent, "build_ground_agent", return_value="ground"
        ), patch.object(
            supervisor_agent, "build_earthdata_agent", return_value="satellite"
        ), patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            asyncio.run(supervisor_agent.build_agent())

        self.assertEqual(factory.call_args.args[0], get_settings().supervisor_model_provider)

    def test_build_agent_passes_per_agent_provider_overrides_to_each_subagent(self):
        from agents import supervisor_agent

        created = object()
        with patch.object(
            supervisor_agent, "build_chat_model", return_value="llm"
        ), patch.object(
            supervisor_agent, "build_ground_agent", return_value="ground"
        ) as ground, patch.object(
            supervisor_agent, "build_earthdata_agent", return_value="satellite"
        ) as earthdata, patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            asyncio.run(
                supervisor_agent.build_agent(
                    ground_agent_provider="google",
                    earthdata_agent_provider="google",
                )
            )

        self.assertEqual(ground.call_args.kwargs["provider"], "google")
        self.assertEqual(earthdata.call_args.kwargs["provider"], "google")


if __name__ == "__main__":
    unittest.main()
