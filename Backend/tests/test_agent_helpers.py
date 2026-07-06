import importlib.util
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _load_query_parser_module():
    path = os.path.join(BACKEND_DIR, "tools", "satellite_tools", "query_parser.py")
    spec = importlib.util.spec_from_file_location("satellite_query_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SatelliteQueryParserTests(unittest.TestCase):
    def test_parser_removes_for_the_month_from_location(self):
        query_parser = _load_query_parser_module()

        parsed = query_parser.parse_satellite_plot_query(
            "Plot TROPOMI NO2 over New Jersey for the month of February 2024"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.variable, "TROPOMI_NO2")
        self.assertEqual(parsed.location, "New Jersey")
        self.assertEqual(parsed.temporal.start, "2024-02-01T00:00:00Z")
        self.assertEqual(parsed.temporal.end, "2024-02-29T23:59:59Z")

    def test_parser_handles_iso_range(self):
        query_parser = _load_query_parser_module()

        parsed = query_parser.parse_satellite_plot_query(
            "plot OMI NO2 over Texas from 2024-01-01 to 2024-01-31"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.variable, "OMI_NO2")
        self.assertEqual(parsed.location, "Texas")
        self.assertEqual(parsed.temporal.start, "2024-01-01T00:00:00Z")
        self.assertEqual(parsed.temporal.end, "2024-01-31T23:59:59Z")

    def test_parser_handles_generic_no2_month_of_year(self):
        query_parser = _load_query_parser_module()

        parsed = query_parser.parse_satellite_plot_query(
            "show NO2 over New Jersey during February of 2024"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.variable, "TROPOMI_NO2")
        self.assertEqual(parsed.location, "New Jersey")
        self.assertEqual(parsed.temporal.start, "2024-02-01T00:00:00Z")
        self.assertEqual(parsed.temporal.end, "2024-02-29T23:59:59Z")

    def test_location_validation_rejects_temporal_pollution(self):
        query_parser = _load_query_parser_module()

        self.assertFalse(query_parser.is_valid_location_candidate("New Jersey during February 2024"))
        self.assertTrue(query_parser.is_valid_location_candidate("New Jersey"))


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class AgentHelperTests(unittest.TestCase):
    def test_simple_satellite_plot_parser_handles_iso_day(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task("Plot TROPOMI NO2 over Newark NJ for 2024-01-15")

        self.assertEqual(
            parsed,
            ("TROPOMI_NO2", "Newark NJ", "2024-01-15T00:00:00Z", "2024-01-15T23:59:59Z"),
        )

    def test_simple_satellite_plot_parser_handles_over_location_on_day(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task("plot TROPOMI NO2 over New York on 2024-06-01")

        self.assertEqual(
            parsed,
            ("TROPOMI_NO2", "New York", "2024-06-01T00:00:00Z", "2024-06-01T23:59:59Z"),
        )

    def test_simple_satellite_plot_parser_handles_month_range(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task("Show OMI ozone in Tampa FL during February 2024")

        self.assertEqual(
            parsed,
            ("OMI_O3", "Tampa FL", "2024-02-01T00:00:00Z", "2024-02-29T23:59:59Z"),
        )

    def test_simple_satellite_plot_parser_removes_for_the_month_from_location(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task(
            "Plot TROPOMI NO2 over New Jersey for the month of February 2024"
        )

        self.assertEqual(
            parsed,
            ("TROPOMI_NO2", "New Jersey", "2024-02-01T00:00:00Z", "2024-02-29T23:59:59Z"),
        )

    def test_simple_satellite_plot_parser_handles_iso_range(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task(
            "plot OMI NO2 over Texas from 2024-01-01 to 2024-01-31"
        )

        self.assertEqual(
            parsed,
            ("OMI_NO2", "Texas", "2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"),
        )

    def test_location_validation_rejects_temporal_pollution(self):
        from tools.satellite_tools.query_parser import is_valid_location_candidate

        self.assertFalse(is_valid_location_candidate("New Jersey during February 2024"))
        self.assertTrue(is_valid_location_candidate("New Jersey"))

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

    def test_build_agent_uses_configured_supervisor_model(self):
        from agents import supervisor_agent

        created = object()
        with patch.object(supervisor_agent, "ChatGroq", return_value="llm") as chat, patch.object(
            supervisor_agent, "build_ground_agent", return_value="ground"
        ), patch.object(
            supervisor_agent, "build_earthdata_agent", return_value="satellite"
        ), patch.object(
            supervisor_agent, "get_checkpointer", AsyncMock(return_value="checkpointer")
        ), patch.object(
            supervisor_agent, "create_agent", Mock(return_value=created)
        ):
            result = asyncio.run(supervisor_agent.build_agent(model="configured-model"))

        self.assertIs(result, created)
        self.assertEqual(chat.call_args.kwargs["model"], "configured-model")


if __name__ == "__main__":
    unittest.main()
