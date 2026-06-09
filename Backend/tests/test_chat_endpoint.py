import os
import sys
import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

_REQUIRED = ["fastapi", "httpx", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import api

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()

    async def test_chat_streams_done_event(self):
        async def fake_stream_response(agent, message, thread_id):
            yield "text", "hello"

        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "stream_response", fake_stream_response):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertIn('"response": "hello"', response.text)

    async def test_session_flow_lists_history_and_deletes(self):
        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    values={
                        "messages": [
                            SimpleNamespace(type="human", content="hi"),
                            SimpleNamespace(type="ai", content="hello", tool_calls=[]),
                        ]
                    }
                )

        self.api.app.state.agent = FakeAgent()
        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_list_sessions():
            return ["thread-1"]

        async def fake_delete_session(thread_id):
            fake_delete_session.called_with = thread_id

        fake_delete_session.called_with = None

        with patch.object(self.api, "list_sessions", fake_list_sessions), \
             patch.object(self.api, "delete_session", fake_delete_session):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                sessions = await client.get("/sessions")
                history = await client.get("/session/thread-1/history")
                deleted = await client.delete("/session/thread-1")

        self.assertEqual(sessions.json(), {"sessions": ["thread-1"]})
        self.assertEqual(history.status_code, 200)
        self.assertEqual(
            history.json()["messages"],
            [
                {"role": "user", "content": "hi", "toolCalls": [], "imageUrls": []},
                {"role": "assistant", "content": "hello", "toolCalls": [], "imageUrls": [], "charts": []},
            ],
        )
        self.assertEqual(deleted.json(), {"deleted": "thread-1"})
        self.assertEqual(fake_delete_session.called_with, "thread-1")

    async def test_chart_export_endpoints_return_downloads(self):
        payload = {"chart_id": "chart-1", "title": "TEMPO over Texas", "export": {"type": "heatmap"}}

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        with patch.object(self.api, "get_chart", fake_get_chart), \
             patch.object(self.api, "_iter_chart_csv_chunks", return_value=iter([b"variable,latitude,longitude,value,units\n"])), \
             patch.object(self.api, "_build_chart_png", return_value=b"\x89PNG\r\n\x1a\n"):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                csv_response = await client.get("/chart/chart-1/export.csv")
                png_response = await client.get("/chart/chart-1/export.png")

        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("tempo-over-texas.csv", csv_response.headers["content-disposition"])
        self.assertEqual(csv_response.headers["x-accel-buffering"], "no")
        self.assertEqual(csv_response.content, b"variable,latitude,longitude,value,units\n")
        self.assertEqual(png_response.status_code, 200)
        self.assertEqual(png_response.headers["content-type"], "image/png")
        self.assertEqual(png_response.content, b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
