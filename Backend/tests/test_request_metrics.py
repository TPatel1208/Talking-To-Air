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
from unittest.mock import patch



TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above


_REQUIRED = ["fastapi", "httpx", "jwt", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "request metrics test dependencies are not installed",
)
class StreamingRequestMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()
        self.api.app.state.earthdata_mcp_tools = {}
        self.user = auth_helpers.user("user-1", email="tester@example.com")
        token = auth_helpers.make_token(self.user.id, email=self.user.email)
        self.auth_headers = {"Authorization": f"Bearer {token}"}

    def _auth_patch(self):
        return auth_helpers.patch_verifier()

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
        with auth_patches, \
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


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "request metrics test dependencies are not installed",
)
class UnmatchedRoutePathLabelTests(unittest.IsolatedAsyncioTestCase):
    """The path recorded by record_request_metrics becomes a Prometheus label
    value. For a matched route that is the route *template*, so the label set
    is bounded by the number of routes. For a request matching no route,
    _route_path fell back to the raw request URL — an attacker-controlled,
    unbounded string, reachable without authentication, and every distinct
    value mints a permanently-retained child of the metric family. Unmatched
    requests must collapse to a single constant label."""

    async def test_unmatched_paths_collapse_to_one_constant_label(self):
        import httpx
        import tta_backend.api as api

        observed = []

        def fake_observe_http_request(method, path, status_code, duration_seconds):
            observed.append((method, path, status_code, duration_seconds))

        transport = httpx.ASGITransport(app=api.app)
        with patch.object(api, "observe_http_request", fake_observe_http_request):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                for nonce in ("alpha", "bravo", "charlie"):
                    response = await client.get(f"/no-such-route-{nonce}")
                    self.assertEqual(response.status_code, 404)

        self.assertEqual(len(observed), 3)
        recorded_paths = {o[1] for o in observed}
        self.assertEqual(
            recorded_paths,
            {"__unmatched__"},
            f"unmatched requests must not each mint their own label: {recorded_paths}",
        )


    async def test_a_matched_route_without_a_path_template_is_also_bounded(self):
        """starlette's Host route matches without carrying a .path, so the
        matched branch needs its own constant too -- falling back to the raw
        URL there would restore the unbounded label this guards against."""
        from starlette.requests import Request
        from starlette.routing import Match

        import tta_backend.api as api

        class PathlessRoute:
            def matches(self, scope):
                return Match.FULL, {}

        def request_for(path):
            return Request({
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [],
                "query_string": b"",
            })

        with patch.object(api.app.router, "routes", [PathlessRoute()]):
            labels = {api._route_path(request_for(f"/vhost-{n}")) for n in ("alpha", "bravo")}

        self.assertEqual(labels, {"__unnamed__"}, f"expected one bounded label, got: {labels}")

if __name__ == "__main__":
    unittest.main()
