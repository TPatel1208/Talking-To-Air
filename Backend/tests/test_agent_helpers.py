import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


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


if __name__ == "__main__":
    unittest.main()
