"""T45: /chat latency in request metrics must reflect the full SSE stream
duration, not just the time to send response headers — record_request_metrics
used to observe duration at call_next() return, which for a StreamingResponse
happens the instant the generator object is constructed (~0ms), so the p95
dashboard could never see the slowest thing in the app.
"""
import asyncio
import importlib.util
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_REQUIRED = ["fastapi", "httpx", "jwt", "bcrypt", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "request metrics test dependencies are not installed",
)
class StreamingRequestMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        from tta_backend.models.user import User

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()
        self.api.app.state.earthdata_mcp_tools = {}
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

        return patch("tta_backend.services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("tta_backend.services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_chat_stream_duration_covers_the_full_stream_not_just_headers(self):
        sleep_seconds = 0.3

        async def fake_stream_chat_events(
            agent, ground_agent, satellite_agent, message, thread_id, user_id, request_id,
        ):
            # An early chunk (like a real status event) arrives almost
            # immediately — GZipMiddleware only buffers headers until the
            # *first* body chunk, so a fixture that sleeps before its only
            # yield would pass by accident (buffering, not the fix, would
            # explain the measured duration). The slow work has to happen
            # *between* chunks, matching a real stream's shape, so only a
            # fix that measures at stream *close* can pass this.
            yield 'event: status\ndata: {"message": "starting"}\n\n'
            await asyncio.sleep(sleep_seconds)
            yield 'event: done\ndata: {"response": "hi"}\n\n'

        async def fake_save_session_metadata_once(thread_id, first_message, user_id):
            pass

        observed = []

        def fake_observe_http_request(method, path, status_code, duration_seconds):
            observed.append((method, path, status_code, duration_seconds))

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "save_session_metadata_once", fake_save_session_metadata_once), \
             patch.object(self.api.chat_stream_service, "stream_chat_events", fake_stream_chat_events), \
             patch.object(self.api, "observe_http_request", fake_observe_http_request):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post("/chat", json={"message": "hi"}, headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        chat_observations = [o for o in observed if o[1] == "/chat"]
        self.assertEqual(len(chat_observations), 1)
        self.assertEqual(chat_observations[0][0], "POST")
        self.assertEqual(chat_observations[0][2], 200)
        self.assertGreaterEqual(chat_observations[0][3], sleep_seconds)

    async def test_non_streaming_route_still_observes_duration_immediately(self):
        """Non-streaming routes must be unchanged — /metrics is measured the
        same way it always was."""
        observed = []

        def fake_observe_http_request(method, path, status_code, duration_seconds):
            observed.append((method, path, status_code, duration_seconds))

        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "observe_http_request", fake_observe_http_request):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        metrics_observations = [o for o in observed if o[1] == "/metrics"]
        self.assertEqual(len(metrics_observations), 1)
        self.assertEqual(metrics_observations[0][0], "GET")
        self.assertEqual(metrics_observations[0][2], 200)


if __name__ == "__main__":
    unittest.main()
