import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class AgentHelperTests(unittest.TestCase):
    def test_simple_satellite_plot_parser_handles_iso_day(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task("Plot TROPOMI NO2 over Newark NJ for 2024-01-15")

        self.assertEqual(
            parsed,
            ("TROPOMI_NO2", "Newark NJ", "2024-01-15T00:00:00Z", "2024-01-15T23:59:59Z"),
        )

    def test_simple_satellite_plot_parser_handles_month_range(self):
        from agents.supervisor_agent import _parse_simple_satellite_plot_task

        parsed = _parse_simple_satellite_plot_task("Show OMI ozone in Tampa FL during February 2024")

        self.assertEqual(
            parsed,
            ("OMI_O3", "Tampa FL", "2024-02-01T00:00:00Z", "2024-02-29T23:59:59Z"),
        )

    def test_truncate_text_logs_warning_with_lengths(self):
        from agents.supervisor_agent import _truncate_text

        with self.assertLogs("agents.supervisor_agent", level="WARNING") as captured:
            result = _truncate_text("abcdef", 3, "satellite", "req-1")

        self.assertEqual(result, "abc")
        self.assertIn("response_truncated", captured.output[0])

    def test_extract_last_text_handles_list_content(self):
        from agents.supervisor_agent import _extract_last_text

        class Message:
            content = [{"type": "text", "text": "hello"}, {"type": "thinking", "text": "hidden"}]

        text = _extract_last_text({"messages": [Message()]}, "fallback", agent_name="ground")

        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()
