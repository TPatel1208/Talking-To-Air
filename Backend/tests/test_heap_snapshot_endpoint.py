"""T45: a one-off heap-profiling seam for chasing a specific memory
incident (the 2026-07-17 QA jump-and-plateau) -- gated behind
DEBUG_HEAP_PROFILING_ENABLED so tracemalloc's per-allocation overhead is
opt-in, never a standing production cost. The endpoint requires auth like
every other route (not listed in PUBLIC_ENDPOINTS).
"""
import dataclasses
import importlib.util
import os
import tracemalloc
import sys
import unittest
from unittest.mock import patch



TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above


_REQUIRED = ["fastapi", "httpx", "jwt"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "heap snapshot endpoint test dependencies are not installed",
)
class HeapSnapshotEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api
        self.user = auth_helpers.user("user-1", email="tester@example.com")
        token = auth_helpers.make_token(self.user.id, email=self.user.email)
        self.auth_headers = {"Authorization": f"Bearer {token}"}

    def _auth_patch(self):
        return auth_helpers.patch_verifier()

    async def test_heap_snapshot_404s_when_the_debug_flag_is_disabled(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches, \
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
        with auth_patches, \
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
