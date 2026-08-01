"""T45: a one-off heap-profiling seam for chasing a specific memory
incident (the 2026-07-17 QA jump-and-plateau) -- gated behind
DEBUG_HEAP_PROFILING_ENABLED so tracemalloc's per-allocation overhead is
opt-in, never a standing production cost. The endpoint requires auth like
every other route (not listed in PUBLIC_ENDPOINTS).
"""
import dataclasses
import importlib.util
import os
import sys
import tracemalloc
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_REQUIRED = ["fastapi", "httpx", "jwt", "bcrypt"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "heap snapshot endpoint test dependencies are not installed",
)
class HeapSnapshotEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        from tta_backend.models.user import User

        self.httpx = httpx
        self.api = api
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

    async def test_heap_snapshot_404s_when_the_debug_flag_is_disabled(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "settings", dataclasses.replace(self.api.settings, debug_heap_profiling_enabled=False)):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/debug/heap-snapshot", headers=self.auth_headers)

        self.assertEqual(response.status_code, 404)

    async def test_heap_snapshot_requires_authentication_even_when_enabled(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "settings", dataclasses.replace(self.api.settings, debug_heap_profiling_enabled=True)):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/debug/heap-snapshot")

        self.assertEqual(response.status_code, 401)

    async def test_heap_snapshot_returns_top_allocations_when_enabled_and_tracing(self):
        tracemalloc.start()
        self.addCleanup(tracemalloc.stop)

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api, "settings", dataclasses.replace(self.api.settings, debug_heap_profiling_enabled=True)):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/debug/heap-snapshot", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("top", body)
        self.assertGreater(len(body["top"]), 0)
        first = body["top"][0]
        self.assertIn("file", first)
        self.assertIn("line", first)
        self.assertIn("size_bytes", first)
        self.assertIn("count", first)


if __name__ == "__main__":
    unittest.main()
