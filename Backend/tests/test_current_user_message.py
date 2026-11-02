"""T64 D6: the researcher's verbatim message is bound server-side for the
whole chat turn, so the dispatch layer can hand it to sub-agents without the
supervisor model ever writing (or paraphrasing) it."""
import unittest
from types import SimpleNamespace


class UntouchedAgent:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to untouched agent: {name}")


class CurrentUserMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_supervisor_route_sees_the_researchers_message_verbatim_for_the_turn_only(self):
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.utils.streaming import current_user_message

        seen = []

        class Supervisor:
            async def astream(self, input_, config, stream_mode):
                seen.append(current_user_message())
                yield "messages", (SimpleNamespace(content="Agent consulted: SATELLITE\n\nDone.", type="ai", tool_calls=None), {})

        message = "  Compare ozone over the OTC  with NJ + NY, July 2024 "
        service = ChatStreamService(ChartService(), long_request_seconds=999)
        [
            event
            async for event in service.stream_chat_events(
                Supervisor(), UntouchedAgent(), UntouchedAgent(), message, "thread-1", "researcher-1", "req-1",
            )
        ]

        self.assertEqual(seen, [message])
        self.assertIsNone(current_user_message())


if __name__ == "__main__":
    unittest.main()
