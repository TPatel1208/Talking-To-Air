"""When a thread earns its place in "Recent analyses".

The ownership row is written by ``POST /chat`` before anything runs, and it
has to be: ``GET /chat/{thread}/stream`` and ``POST /chat/{thread}/stop`` both
refuse a thread with no row, and the client issues the first of those
milliseconds after the 202. So the row cannot also mean "this conversation has
something in it" -- ``first_frame_at`` does, and it is stamped when the turn
first produces narration.

The stamp is applied to the frame iterator at the one point both protocols
share, *outside* ``ChatStreamService``. That placement is the property worth
pinning: the fast path dispatches sub-agents directly and never enters
``stream_response`` (the same seam T63 Phase 5's heartbeat fell through), so
anything that stamps from inside a route is one new route away from being
wrong.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest
import uuid
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from test_turn_registry import REDIS_URL, requires_redis  # noqa: E402

_REQUIRED = ["fastapi", "httpx", "jwt", "langchain", "langgraph"]


@requires_redis
@unittest.skipIf(
    any(importlib.util.find_spec(module) is None for module in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class FirstFrameStampTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        from tta_backend.config.settings import get_settings
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()
        self.api.app.state.earthdata_mcp_tools = {}

        patcher = patch.dict(os.environ, {"CHAT_DETACHED_TURNS_ENABLED": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        self.log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(self.log.aclose)
        self.registry = TurnRegistry(self.log, url=REDIS_URL)
        self.addAsyncCleanup(self.registry.aclose)
        self.api.app.state.turn_registry = self.registry
        self.api.app.state.turn_event_log = self.log
        self.addCleanup(setattr, self.api.app.state, "turn_registry", None)
        self.addCleanup(setattr, self.api.app.state, "turn_event_log", None)

        self.user = auth_helpers.user("user-1", email="tester@example.com")
        token = auth_helpers.make_token(self.user.id, email=self.user.email)
        self.auth_headers = {"Authorization": f"Bearer {token}"}
        self.thread_id = str(uuid.uuid4())
        self.stamped: list[str] = []

    def client(self):
        return self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app),
            base_url="http://testserver",
        )

    @contextmanager
    def serving(self, produce):
        """Patch the turn's whole frame source with ``produce``.

        Whatever a route does internally, this is what ``POST /chat`` holds:
        an async iterator of rendered frames. Replacing it entirely is how
        these tests speak for every route, including the ones that never go
        near ``stream_response``.
        """

        def fake_stream_chat_events(*args, **kwargs):
            return produce()

        async def fake_stamp(thread_id):
            self.stamped.append(thread_id)

        async def fake_save(thread_id, first_message, user_id):
            return None

        async def fake_owns(thread_id, user_id):
            return user_id == self.user.id

        async def fake_metadata(thread_id):
            return {"user_id": self.user.id}

        patches = (
            auth_helpers.patch_verifier(),
            patch.object(self.api, "save_session_metadata_once", fake_save),
            patch.object(self.api, "mark_session_activity", fake_stamp),
            patch.object(self.api, "get_session_metadata", fake_metadata),
            patch.object(self.api, "session_belongs_to_user", fake_owns),
            patch.object(
                self.api.chat_stream_service,
                "stream_chat_events",
                fake_stream_chat_events,
            ),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            yield

    async def post(self, client, **kwargs):
        return await client.post(
            "/chat",
            json={"message": "hi", "thread_id": self.thread_id},
            headers=self.auth_headers,
            **kwargs,
        )

    async def test_accepting_the_message_does_not_list_the_thread(self):
        """The row exists from here on -- the stream and stop endpoints need
        it -- but nothing has been said on this thread yet."""
        started = asyncio.Event()

        async def produce():
            await started.wait()
            yield "event: done\ndata: {}\n\n"

        with self.serving(produce):
            async with self.client() as client:
                response = await self.post(client)

                self.assertEqual(response.status_code, 202)
                self.assertEqual(
                    self.stamped,
                    [],
                    "the POST was accepted, not answered -- a thread listed here is "
                    "the empty row that shows up when the turn then dies",
                )
                started.set()
                await self.registry.wait(response.json()["turn_id"])

        self.assertEqual(self.stamped, [self.thread_id])

    async def test_a_turn_that_never_enters_stream_response_still_lists(self):
        """``stream_chat_events`` is replaced outright here: no route runs.

        The fast path is exactly this shape from the stamp's point of view --
        it dispatches ground/satellite directly and produces its own frames --
        so a stamp that lived inside the supervisor's streaming layer would
        leave every fast-pathed thread unlisted, which is most of them.
        """

        async def produce():
            yield 'event: status\ndata: {"message": "Working"}\n\n'
            yield "event: done\ndata: {}\n\n"

        with self.serving(produce):
            async with self.client() as client:
                response = await self.post(client)
                await self.registry.wait(response.json()["turn_id"])

        self.assertEqual(self.stamped, [self.thread_id])

    async def test_a_thread_is_listed_once_however_long_the_turn_talks(self):
        """One write per thread, not one per frame: this runs on the path
        that carries every token of every answer."""

        async def produce():
            for index in range(25):
                yield f'event: text\ndata: {{"content": "{index}"}}\n\n'
            yield "event: done\ndata: {}\n\n"

        with self.serving(produce):
            async with self.client() as client:
                response = await self.post(client)
                await self.registry.wait(response.json()["turn_id"])

        self.assertEqual(self.stamped, [self.thread_id])

    async def test_a_turn_that_says_nothing_at_all_leaves_the_thread_unlisted(self):
        """The case the whole change is for: a turn that produces no frames
        -- stopped early, failed before narrating, or killed with its replica
        -- leaves a titled row over an empty conversation. It stays out of
        the sidebar, while remaining a thread its owner can still reach."""

        async def produce():
            return
            yield  # pragma: no cover -- makes this an async generator

        with self.serving(produce):
            async with self.client() as client:
                response = await self.post(client)
                await self.registry.wait(response.json()["turn_id"])

        self.assertEqual(self.stamped, [])

    async def test_a_refused_duplicate_send_lists_nothing_of_its_own(self):
        """A 409 bought no turn, so it consumes no frames and stamps nothing.
        The turn it names did its own stamping."""
        release = asyncio.Event()

        async def produce():
            await release.wait()
            yield "event: done\ndata: {}\n\n"

        with self.serving(produce):
            async with self.client() as client:
                first = await self.post(client)
                second = await self.post(client)

                self.assertEqual(first.status_code, 202)
                self.assertEqual(second.status_code, 409)
                self.assertEqual(self.stamped, [])

                release.set()
                await self.registry.wait(first.json()["turn_id"])

        self.assertEqual(self.stamped, [self.thread_id])

    async def test_the_legacy_streaming_post_lists_the_thread_too(self):
        """With the kill switch off the POST streams the turn itself and the
        registry never sees it. Rolling back must not stop new threads
        appearing in the sidebar -- a regression nobody would connect to the
        flag they changed."""
        from tta_backend.config.settings import get_settings

        async def produce():
            yield 'event: text\ndata: {"content": "hello"}\n\n'
            yield "event: done\ndata: {}\n\n"

        with patch.dict(os.environ, {"CHAT_DETACHED_TURNS_ENABLED": "0"}):
            get_settings.cache_clear()
            with self.serving(produce):
                async with self.client() as client:
                    response = await self.post(client)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertEqual(self.stamped, [self.thread_id])


if __name__ == "__main__":
    unittest.main()
