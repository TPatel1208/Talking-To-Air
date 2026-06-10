import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class ExtractedServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_export_service_safe_name_and_missing_export(self):
        from services.export_service import ExportService

        service = ExportService(csv_export_max_granules=2)

        self.assertEqual(service.safe_export_name({"title": "TEMPO over Texas!"}, "csv"), "tempo-over-texas.csv")
        with self.assertRaisesRegex(ValueError, "full-resolution export metadata"):
            list(service.iter_chart_csv_rows({}))

    def test_chart_service_parses_agent_result_and_persists(self):
        from models import AgentResult, ChartPayload, agent_result_to_json
        from services.chart_service import ChartService

        service = ChartService()
        raw = agent_result_to_json(AgentResult(text="done", charts=[ChartPayload(type="heatmap", title="Map")]))

        text, charts = service.parse_charts(raw)

        self.assertEqual(text, "done")
        self.assertEqual(charts[0].type, "heatmap")

    async def test_chart_service_reuses_owned_stored_chart(self):
        from models import ChartPayload
        from services.chart_service import ChartService

        stored = {"chart_id": "chart-1", "user_id": "user-1"}
        with patch("services.chart_service.chart_repository.get_chart", AsyncMock(return_value=stored)):
            result = await ChartService().persist_chart_payload(
                "thread-1",
                ChartPayload(type="heatmap", chart_id="chart-1"),
                "user-1",
            )

        self.assertEqual(result, stored)

    async def test_history_service_builds_plain_history(self):
        from services.chart_service import ChartService
        from services.history_service import HistoryService

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [
                    SimpleNamespace(type="human", content="hi"),
                    SimpleNamespace(type="ai", content="hello", tool_calls=[]),
                ]})

        messages = await HistoryService(ChartService()).build_history(FakeAgent(), "thread-1", "user-1")

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["content"], "hello")

    async def test_chat_stream_service_emits_done_event(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        async def fake_stream_response(agent, message, thread_id):
            yield "text", "hello"

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(object(), "hi", "thread-1", "user-1", "req-1")
            ]

        self.assertIn("event: done", events[-1])
        self.assertIn('"response": "hello"', events[-1])

    async def test_chat_stream_service_does_not_warn_for_plain_tool_result(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with self.assertNoLogs("services.chat_stream_service", level="WARNING"):
            events = [
                event
                async for event in service._tool_result_events(
                    "plain tool text",
                    "thread-1",
                    "user-1",
                    [],
                )
            ]

        self.assertEqual(events, [])

    async def test_chat_stream_service_warns_for_malformed_chart_payload(self):
        from services.chat_stream_service import ChatStreamService
        from services.chart_service import ChartService

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with self.assertLogs("services.chat_stream_service", level="WARNING") as captured:
            events = [
                event
                async for event in service._tool_result_events(
                    '{"type":',
                    "thread-1",
                    "user-1",
                    [],
                )
            ]

        self.assertEqual(events, [])
        self.assertIn("chart_payload_parse_failure", captured.output[0])


if __name__ == "__main__":
    unittest.main()
