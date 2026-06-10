import os
import sys
import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime, timezone

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_REQUIRED = ["fastapi", "httpx", "jwt", "bcrypt", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import api
        from models.user import User

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()
        self.user = User(
            id="user-1",
            username="tester",
            password_hash="hash",
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )
        token, _ = self.api.create_access_token(self.user)
        self.auth_headers = {"Authorization": f"Bearer {token}"}

    def _auth_patch(self):
        async def fake_get_user_by_id(user_id):
            return self.user if user_id == self.user.id else None

        async def fake_is_token_revoked(jti):
            return False

        return patch("services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_chat_streams_done_event(self):
        async def fake_stream_response(agent, message, thread_id):
            yield "status", {"message": "Downloading satellite granules..."}
            yield "text", "hello"

        async def fake_save_session_metadata_once(thread_id, first_message, user_id):
            fake_save_session_metadata_once.called_with = (thread_id, first_message, user_id)

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "save_session_metadata_once", fake_save_session_metadata_once), \
             patch.object(self.api, "stream_response", fake_stream_response):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post("/chat", json={"message": "hi"}, headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: status", response.text)
        self.assertIn('"message": "Downloading satellite granules..."', response.text)
        self.assertIn("event: done", response.text)
        self.assertIn('"response": "hello"', response.text)
        self.assertEqual(fake_save_session_metadata_once.called_with[2], self.user.id)

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
        async def fake_list_sessions(user_id):
            fake_list_sessions.called_with = user_id
            return [{"id": "thread-1", "title": "hi", "created_at": "2026-06-09T00:00:00+00:00"}]

        async def fake_delete_session(thread_id, user_id):
            fake_delete_session.called_with = (thread_id, user_id)
            return True

        async def fake_session_belongs_to_user(thread_id, user_id):
            return thread_id == "thread-1" and user_id == self.user.id

        fake_delete_session.called_with = None

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "list_sessions", fake_list_sessions), \
             patch.object(self.api, "delete_session", fake_delete_session), \
             patch.object(self.api, "session_belongs_to_user", fake_session_belongs_to_user):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                sessions = await client.get("/sessions", headers=self.auth_headers)
                history = await client.get("/session/thread-1/history", headers=self.auth_headers)
                deleted = await client.delete("/session/thread-1", headers=self.auth_headers)

        self.assertEqual(
            sessions.json(),
            {"sessions": [{"id": "thread-1", "title": "hi", "created_at": "2026-06-09T00:00:00+00:00"}]},
        )
        self.assertEqual(fake_list_sessions.called_with, self.user.id)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(
            history.json()["messages"],
            [
                {"role": "user", "content": "hi", "toolCalls": [], "imageUrls": []},
                {"role": "assistant", "content": "hello", "toolCalls": [], "imageUrls": [], "charts": []},
            ],
        )
        self.assertEqual(deleted.json(), {"deleted": "thread-1"})
        self.assertEqual(fake_delete_session.called_with, ("thread-1", self.user.id))

    async def test_chart_export_endpoints_return_downloads(self):
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "export": {"type": "heatmap"},
            "user_id": self.user.id,
        }

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "get_chart", fake_get_chart), \
             patch.object(self.api, "_iter_chart_csv_chunks", return_value=iter([b"variable,latitude,longitude,value,units\n"])), \
             patch.object(self.api, "_build_chart_png", return_value=b"\x89PNG\r\n\x1a\n"):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                csv_response = await client.get("/chart/chart-1/export.csv", headers=self.auth_headers)
                png_response = await client.get("/chart/chart-1/export.png", headers=self.auth_headers)

        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("tempo-over-texas.csv", csv_response.headers["content-disposition"])
        self.assertEqual(csv_response.headers["x-accel-buffering"], "no")
        self.assertEqual(csv_response.content, b"variable,latitude,longitude,value,units\n")
        self.assertEqual(png_response.status_code, 200)
        self.assertEqual(png_response.headers["content-type"], "image/png")
        self.assertEqual(png_response.content, b"\x89PNG\r\n\x1a\n")

    async def test_protected_endpoints_require_authentication(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            response = await client.get("/sessions")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(response.status_code, 401)

    async def test_chat_validation_happens_before_streaming(self):
        async def fake_stream_response(agent, message, thread_id):
            fake_stream_response.called = True
            yield "text", "should not run"

        fake_stream_response.called = False
        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_save_session_metadata_once(thread_id, first_message, user_id):
            pass

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "save_session_metadata_once", fake_save_session_metadata_once), \
             patch.object(self.api, "stream_response", fake_stream_response):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                empty = await client.post("/chat", json={"message": ""}, headers=self.auth_headers)
                long_message = await client.post(
                    "/chat",
                    json={"message": "x" * 10001},
                    headers=self.auth_headers,
                )
                bad_thread = await client.post(
                    "/chat",
                    json={"message": "hi", "thread_id": "../bad"},
                    headers=self.auth_headers,
                )

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(long_message.status_code, 422)
        self.assertEqual(bad_thread.status_code, 422)
        self.assertFalse(fake_stream_response.called)

    async def test_login_issues_bearer_token(self):
        password_hash = self.api.hash_password("correct-password")
        user = self.user.model_copy(update={"password_hash": password_hash})

        async def fake_get_user_by_username(username):
            return user if username == "tester" else None

        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "get_user_by_username", fake_get_user_by_username):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/auth/login",
                    json={"username": "tester", "password": "correct-password"},
                )
                invalid = await client.post(
                    "/auth/login",
                    json={"username": "tester", "password": "wrong-password"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["token_type"], "bearer")
        self.assertEqual(response.json()["expires_in"], 3600)
        self.assertTrue(response.json()["access_token"])
        self.assertEqual(invalid.status_code, 401)


if __name__ == "__main__":
    unittest.main()
