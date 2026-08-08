import os
import importlib.util
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from datetime import datetime, timezone


os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_REQUIRED = ["fastapi", "httpx", "jwt", "bcrypt", "langchain", "langgraph"]


async def _aiter(items):
    for item in items:
        yield item


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_chat_streams_done_event(self):
        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "status", {"message": "Downloading satellite granules...", "stage": "progress", "detail": 40}
            yield "text", "hello"

        async def fake_save_session_metadata_once(thread_id, first_message, user_id):
            fake_save_session_metadata_once.called_with = (thread_id, first_message, user_id)

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "save_session_metadata_once", fake_save_session_metadata_once), \
             patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post("/chat", json={"message": "hi"}, headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: status", response.text)
        self.assertIn('"message": "Downloading satellite granules..."', response.text)
        # T19: the supervisor path's own status forwarding must not rebuild
        # a message-only payload — stage/detail have to survive to the wire.
        self.assertIn('"stage": "progress"', response.text)
        self.assertIn('"detail": 40', response.text)
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
             patch.object(self.api.session_repository, "list_sessions", fake_list_sessions), \
             patch.object(self.api.session_repository, "delete_session", fake_delete_session), \
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
                {"role": "assistant", "content": "hello", "toolCalls": [], "imageUrls": [], "charts": [], "artifacts": []},
            ],
        )
        self.assertEqual(deleted.json(), {"deleted": "thread-1"})
        self.assertEqual(fake_delete_session.called_with, ("thread-1", self.user.id))

    async def test_session_endpoint_500s_never_leak_the_exception_text(self):
        # T37: internal error text (driver messages, paths) must go to the
        # logs, not the client — every 500 body carries the generic detail.
        class FakeAgent:
            async def aget_state(self, config):
                raise RuntimeError("secret driver path C:\\db\\creds")

        self.api.app.state.agent = FakeAgent()

        async def raising_list_sessions(user_id):
            raise RuntimeError("secret driver path C:\\db\\creds")

        async def raising_delete_session(thread_id, user_id):
            raise RuntimeError("secret driver path C:\\db\\creds")

        async def fake_session_belongs_to_user(thread_id, user_id):
            return True

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.session_repository, "list_sessions", raising_list_sessions), \
             patch.object(self.api.session_repository, "delete_session", raising_delete_session), \
             patch.object(self.api, "session_belongs_to_user", fake_session_belongs_to_user):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                sessions = await client.get("/sessions", headers=self.auth_headers)
                history = await client.get("/session/thread-1/history", headers=self.auth_headers)
                deleted = await client.delete("/session/thread-1", headers=self.auth_headers)

        for response in (sessions, history, deleted):
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json()["detail"], "Internal server error")
            self.assertNotIn("secret driver path", response.text)

    async def test_chart_export_endpoints_return_downloads(self):
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "export": {"type": "heatmap"},
            "user_id": self.user.id,
        }

        # T37: export endpoints read tools through the readiness gate.
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api.export_service, "iter_chart_csv_chunks", return_value=_aiter([b"variable,latitude,longitude,value,units\n"])), \
             patch.object(self.api.export_service, "build_chart_png", return_value=b"\x89PNG\r\n\x1a\n"):
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

    async def test_chart_csv_export_503s_with_the_taxonomy_body_when_the_mcp_is_not_ready(self):
        # T37: export.csv must go through the same T17 readiness gate as
        # every other MCP-backed endpoint — never fail inside the
        # StreamingResponse generator after a 200 has been committed.
        # No earthdata_mcp_manager on app.state (lifespan never ran here),
        # which reads as "connecting" — not ready.
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.csv", headers=self.auth_headers)

        self.assertEqual(response.status_code, 503)
        body = response.json()["error"]
        self.assertEqual(body["category"], "provider_unavailable")
        self.assertIn("temporarily unavailable", body["message"])

    async def test_chart_csv_export_fails_clean_when_the_stream_dies_before_the_first_chunk(self):
        # T37: the common failures (missing handle, evicted export) surface
        # inside the CSV generator. The endpoint must materialize the first
        # chunk before committing to a 200, so these become a clean 422 —
        # never a 0-byte file that looks like a successful download.
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "export": {"type": "heatmap"},
            "user_id": self.user.id,
        }
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        async def broken_chunks(payload_, tools, chunk_size=64 * 1024):
            raise ValueError("This chart does not include a source handle for full-resolution export.")
            yield  # pragma: no cover — makes this an async generator

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api.export_service, "iter_chart_csv_chunks", broken_chunks):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.csv", headers=self.auth_headers)

        self.assertEqual(response.status_code, 422)
        self.assertIn("source handle", response.json()["detail"])

    async def test_chart_csv_export_marks_a_mid_stream_failure_with_an_incomplete_trailer(self):
        # T37: once streaming has begun the 200 is committed — a failure
        # after the first chunk must append a clearly marked trailer so a
        # truncated file is never mistaken for the full dataset.
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "export": {"type": "heatmap"},
            "user_id": self.user.id,
        }
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        async def dying_chunks(payload_, tools, chunk_size=64 * 1024):
            yield b"variable,latitude,longitude,value,units\n"
            raise RuntimeError("secret internal path leak")

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api.export_service, "iter_chart_csv_chunks", dying_chunks):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.csv", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        text = response.content.decode("utf-8")
        self.assertTrue(text.startswith("variable,latitude,longitude,value,units\n"))
        self.assertIn("# EXPORT INCOMPLETE — contract", text)
        self.assertNotIn("secret internal path leak", text)

    async def test_chart_csv_export_trailer_names_the_taxonomy_category_of_a_classified_failure(self):
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "export": {"type": "heatmap"},
            "user_id": self.user.id,
        }
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        async def dying_chunks(payload_, tools, chunk_size=64 * 1024):
            from tta_backend.earthdata_mcp.results import CATEGORY_PROVIDER_UNAVAILABLE, MCPToolError

            yield b"variable,latitude,longitude,value,units\n"
            raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, "The satellite data layer is temporarily unavailable.")

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api.export_service, "iter_chart_csv_chunks", dying_chunks):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.csv", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertIn("# EXPORT INCOMPLETE — provider_unavailable", response.content.decode("utf-8"))

    async def test_chart_netcdf_export_503s_with_the_taxonomy_body_when_the_mcp_is_not_ready(self):
        # T37: export.nc answered a bespoke bare-detail 503; it must speak
        # the shared T18 taxonomy like every other MCP-backed endpoint.
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.nc", headers=self.auth_headers)

        self.assertEqual(response.status_code, 503)
        body = response.json()["error"]
        self.assertEqual(body["category"], "provider_unavailable")
        self.assertIn("temporarily unavailable", body["message"])

    async def test_chart_netcdf_export_fails_clean_when_the_converted_file_is_gone(self):
        # T37: an evicted/vanished converted file must fail before the 200
        # commits — never stream a 0-byte "successful" NetCDF download.
        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "metadata": {"source_handles": ["obs_1"]},
            "user_id": self.user.id,
        }
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        async def fake_export_converted(handle, target_format, tools):
            return {"status": "ready", "storage_uri": "file:///does/not/exist.nc"}

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api, "export_converted", fake_export_converted):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/export.nc", headers=self.auth_headers)

        self.assertEqual(response.status_code, 422)
        self.assertIn("no longer available", response.json()["detail"])

    async def test_chart_overlay_endpoint_streams_the_stored_png(self):
        import os
        import tempfile

        fd, overlay_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(overlay_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nOVERLAYBYTES")

        payload = {
            "chart_id": "chart-1",
            "title": "TEMPO over Texas",
            "overlay": {"bounds": [0, 0, 1, 1], "_path": overlay_path},
            "user_id": self.user.id,
        }

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        try:
            auth_patches = self._auth_patch()
            with auth_patches[0], auth_patches[1], \
                 patch.object(self.api.chart_service, "get_chart", fake_get_chart):
                async with self.httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.get("/chart/chart-1/overlay.png", headers=self.auth_headers)
        finally:
            os.remove(overlay_path)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, b"\x89PNG\r\n\x1a\nOVERLAYBYTES")

    async def test_chart_overlay_endpoint_serves_the_requested_panel(self):
        import os
        import tempfile

        fd, path_a = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path_a, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nPANEL_A")
        fd, path_b = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path_b, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nPANEL_B")

        payload = {
            "chart_id": "chart-1",
            "type": "heatmap_multi",
            "panels": [
                {"overlay": {"bounds": [0, 0, 1, 1], "_path": path_a}},
                {"overlay": {"bounds": [0, 0, 1, 1], "_path": path_b}},
            ],
            "user_id": self.user.id,
        }

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        try:
            auth_patches = self._auth_patch()
            with auth_patches[0], auth_patches[1], \
                 patch.object(self.api.chart_service, "get_chart", fake_get_chart):
                async with self.httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    resp_a = await client.get("/chart/chart-1/overlay.png?panel=0", headers=self.auth_headers)
                    resp_b = await client.get("/chart/chart-1/overlay.png?panel=1", headers=self.auth_headers)
                    resp_missing = await client.get("/chart/chart-1/overlay.png?panel=5", headers=self.auth_headers)
        finally:
            os.remove(path_a)
            os.remove(path_b)

        self.assertEqual(resp_a.content, b"\x89PNG\r\n\x1a\nPANEL_A")
        self.assertEqual(resp_b.content, b"\x89PNG\r\n\x1a\nPANEL_B")
        self.assertEqual(resp_missing.status_code, 404)

    async def test_chart_overlay_endpoint_serves_the_difference_panel(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nDIFF")

        payload = {
            "chart_id": "chart-1",
            "type": "heatmap_multi",
            "mode": "difference",
            "difference": {"overlay": {"bounds": [0, 0, 1, 1], "_path": path}},
            "user_id": self.user.id,
        }

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        try:
            auth_patches = self._auth_patch()
            with auth_patches[0], auth_patches[1], \
                 patch.object(self.api.chart_service, "get_chart", fake_get_chart):
                async with self.httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.get("/chart/chart-1/overlay.png", headers=self.auth_headers)
        finally:
            os.remove(path)

        self.assertEqual(response.content, b"\x89PNG\r\n\x1a\nDIFF")

    async def test_chart_overlay_endpoint_404s_when_no_overlay_was_rendered(self):
        payload = {"chart_id": "chart-1", "title": "TEMPO over Texas", "user_id": self.user.id}

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/overlay.png", headers=self.auth_headers)

        self.assertEqual(response.status_code, 404)

    async def test_chart_overlay_endpoint_404s_for_another_users_chart(self):
        payload = {
            "chart_id": "chart-1",
            "overlay": {"bounds": [0, 0, 1, 1], "_path": "/does/not/matter.png"},
            "user_id": "someone-else",
        }

        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_get_chart(chart_id):
            return payload

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/chart/chart-1/overlay.png", headers=self.auth_headers)

        self.assertEqual(response.status_code, 404)

    async def test_chart_overlay_endpoint_does_not_block_the_event_loop(self):
        """T45: chart_overlay_png used to read the overlay PNG with a
        synchronous open()/read() straight on the event loop — a slow disk
        (or a large overlay) would stall every other concurrent stream for
        the read's duration. Hermetic per the OpenHandleEventLoopOffloadTests
        pattern: only that a concurrent coroutine keeps making progress while
        a (patched-slow) read is in flight."""
        import asyncio
        import time

        payload = {
            "chart_id": "chart-1",
            "overlay": {"bounds": [0, 0, 1, 1], "_path": "/overlay/does-not-need-to-exist.png"},
            "user_id": self.user.id,
        }

        transport = self.httpx.ASGITransport(app=self.api.app)

        async def fake_get_chart(chart_id):
            return payload

        def slow_read_overlay_bytes(path):
            time.sleep(0.5)
            return b"\x89PNG\r\n\x1a\nSLOW"

        tick_count = 0

        async def ticker():
            nonlocal tick_count
            for _ in range(15):
                await asyncio.sleep(0.03)
                tick_count += 1

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch("os.path.isfile", return_value=True), \
             patch.object(self.api, "_read_overlay_bytes", slow_read_overlay_bytes):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response, _ = await asyncio.gather(
                    client.get("/chart/chart-1/overlay.png", headers=self.auth_headers),
                    ticker(),
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"\x89PNG\r\n\x1a\nSLOW")
        # ticker() needs ~0.45s (15 * 0.03s) of its own. If the 0.5s read ran
        # on the event loop it would starve the ticker down to a couple of
        # ticks in that window instead of the full 15.
        self.assertEqual(tick_count, 15)

    async def test_artifact_endpoints_return_paginated_rows_and_csv(self):
        from tta_backend.services.artifact_store import artifact_store

        ref = artifact_store.put_table(
            "EPA Summary",
            ["date", "value"],
            [{"date": "2024-01-01", "value": 10}, {"date": "2024-01-02", "value": 20}],
        )

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch("tta_backend.services.artifact_store.artifact_repository.save_artifact", AsyncMock()), \
             patch("tta_backend.services.artifact_store.artifact_repository.delete_expired_unclaimed", AsyncMock()):
            await artifact_store.claim(ref.id, self.user.id, "thread-1")
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                page_response = await client.get(
                    f"/artifacts/{ref.id}?offset=1&limit=1",
                    headers=self.auth_headers,
                )
                csv_response = await client.get(f"/artifacts/{ref.id}/csv", headers=self.auth_headers)

        self.assertEqual(page_response.status_code, 200)
        self.assertEqual(page_response.json()["total_rows"], 2)
        self.assertEqual(page_response.json()["rows"], [{"date": "2024-01-02", "value": 20}])
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("epa-summary.csv", csv_response.headers["content-disposition"])
        self.assertIn(b"2024-01-01,10", csv_response.content)

    async def test_protected_endpoints_require_authentication(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            metrics = await client.get("/metrics")
            response = await client.get("/sessions")

        self.assertNotEqual(health.status_code, 401)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("http_requests_total", metrics.text)
        self.assertEqual(response.status_code, 401)

    async def test_map_tiles_config_is_public_and_reflects_settings(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/config/map-tiles")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["basemap_light_url"], self.api.settings.map_basemap_light_url)
        self.assertEqual(body["basemap_dark_url"], self.api.settings.map_basemap_dark_url)
        self.assertEqual(body["terrain_dem_url"], self.api.settings.map_terrain_dem_url)
        self.assertEqual(body["basemap_attribution"], self.api.settings.map_basemap_attribution)
        self.assertEqual(body["terrain_attribution"], self.api.settings.map_terrain_attribution)

    async def test_health_reports_ok_when_dependencies_are_ready(self):
        transport = self.httpx.ASGITransport(app=self.api.app)

        async def healthy_db(timeout_seconds=2.0):
            return True, None

        with patch.object(self.api, "check_db_pool", healthy_db):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/health")

        self.assertEqual(response.status_code, 200)
        # earthdata_mcp_manager is never started in this test (lifespan
        # never runs), so it reports its initial "connecting" state (T17).
        self.assertEqual(
            response.json(), {"status": "ok", "db": True, "agent": True, "earthdata_mcp": "connecting"},
        )

    async def test_health_reports_the_earthdata_mcp_connection_state(self):
        from types import SimpleNamespace

        transport = self.httpx.ASGITransport(app=self.api.app)

        async def healthy_db(timeout_seconds=2.0):
            return True, None

        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="unavailable")
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        with patch.object(self.api, "check_db_pool", healthy_db):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/health")

        self.assertEqual(response.json()["earthdata_mcp"], "unavailable")

    async def test_health_reports_degraded_when_database_fails(self):
        transport = self.httpx.ASGITransport(app=self.api.app)

        async def unhealthy_db(timeout_seconds=2.0):
            return False, "connection refused"

        with patch.object(self.api, "check_db_pool", unhealthy_db):
            async with self.httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["db"])
        self.assertTrue(response.json()["agent"])
        self.assertEqual(response.json()["db_error"], "connection refused")

    async def test_metrics_endpoint_returns_prometheus_text(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        for name in [
            "http_requests_total",
            "http_request_duration_seconds",
            "agent_requests_total",
            "harmony_fetch_duration_seconds",
            "harmony_timeouts_total",
            "cache_hits_total",
            "cache_misses_total",
            "db_pool_connections_active",
        ]:
            self.assertIn(name, response.text)

    async def test_chat_validation_happens_before_streaming(self):
        async def fake_stream_response(agent, message, thread_id, **kwargs):
            fake_stream_response.called = True
            yield "text", "should not run"

        fake_stream_response.called = False
        transport = self.httpx.ASGITransport(app=self.api.app)
        async def fake_save_session_metadata_once(thread_id, first_message, user_id):
            pass

        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "save_session_metadata_once", fake_save_session_metadata_once), \
             patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response):
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
