"""Chat turns that outlive the connection that started them (T63 Phase 2).

``POST /chat`` accepts the message and hands back a turn id; the turn runs on
its own and writes into the event log; ``GET /chat/{thread_id}/stream`` is the
only thing that ever produces SSE. Switching away, sleeping a laptop or being
routed to a different replica costs a reader its cursor, never the turn.

``CHAT_DETACHED_TURNS_ENABLED`` is the switch between this and the old
streaming POST. On by default since Phase 5 taught the frontend the protocol;
setting it to ``0`` is the rollback.

Real Redis, same as the registry and event log tests.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest
import uuid
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from tta_backend.earthdata_mcp.connection import STATE_READY  # noqa: E402
from test_turn_registry import REDIS_URL, requires_redis  # noqa: E402

_REQUIRED = ["fastapi", "httpx", "jwt", "langchain", "langgraph"]


@requires_redis
@unittest.skipIf(
    any(importlib.util.find_spec(module) is None for module in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class DetachedChatTurnTests(unittest.IsolatedAsyncioTestCase):
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
        self.api.app.state.turn_event_log = self.log
        self.api.app.state.turn_registry = self.registry
        self.addCleanup(setattr, self.api.app.state, "turn_registry", None)
        self.addCleanup(setattr, self.api.app.state, "turn_event_log", None)

        self.user = auth_helpers.user("user-1", email="tester@example.com")
        token = auth_helpers.make_token(self.user.id, email=self.user.email)
        self.auth_headers = {"Authorization": f"Bearer {token}"}
        self.thread_id = str(uuid.uuid4())

    def client(self):
        return self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app),
            base_url="http://testserver",
        )

    def serving(self, *events):
        """Patch the agent stream so a turn produces these events and ends.

        An ``asyncio.Event`` among them is a place the turn stops until the
        test lets it go on — which is how a reader gets to leave in the
        middle of a turn that is genuinely still running.
        """

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            for item in events:
                if isinstance(item, asyncio.Event):
                    await item.wait()
                    continue
                yield item

        async def fake_save(thread_id, first_message, user_id):
            return None

        async def fake_metadata(thread_id):
            """This thread is the test user's, without asking Postgres."""
            return {"user_id": self.user.id}

        async def fake_owns(thread_id, user_id):
            return user_id == self.user.id

        return (
            auth_helpers.patch_verifier(),
            patch.object(self.api, "save_session_metadata_once", fake_save),
            patch.object(self.api, "get_session_metadata", fake_metadata),
            patch.object(self.api, "session_belongs_to_user", fake_owns),
            patch(
                "tta_backend.services.chat_stream_service.stream_response",
                fake_stream_response,
            ),
        )

    async def test_the_post_hands_back_a_turn_id_instead_of_a_stream(self):
        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                response = await client.post(
                    "/chat",
                    json={"message": "hi", "thread_id": self.thread_id},
                    headers=self.auth_headers,
                )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["thread_id"], self.thread_id)
        self.assertTrue(body["turn_id"])
        self.assertNotIn("event: done", response.text)

        # The turn is running regardless: nothing about the response carried it.
        await self.registry.wait(body["turn_id"])
        frames = (await self.log.read(body["turn_id"])).frames
        self.assertTrue(
            any('"response": "hello"' in frame for frame in frames),
            f"the turn's answer never reached the log: {frames}",
        )

    async def start_turn(self, client, *, message="hi"):
        """POST a message and return the accepted turn id."""
        response = await client.post(
            "/chat",
            json={"message": message, "thread_id": self.thread_id},
            headers=self.auth_headers,
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["turn_id"]

    async def test_the_stream_replays_the_turn_and_closes_at_its_terminal_entry(self):
        """The GET is the only SSE producer, and it ends on its own.

        Nothing closes this connection from the turn's side — under the old
        architecture the generator returning was what ended the response. The
        terminal entry is what takes over that job.
        """
        verifier, save, metadata, owns, stream = self.serving(
            ("status", {"message": "Searching granules"}), ("text", "hello"),
        )
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                turn_id = await self.start_turn(client)
                await self.registry.wait(turn_id)
                replay = await client.get(
                    f"/chat/{self.thread_id}/stream",
                    # The reader that sent this message, coming back to it.
                    params={"turn": turn_id},
                    headers=self.auth_headers,
                )

        self.assertEqual(replay.status_code, 200)
        self.assertIn("event: status", replay.text)
        self.assertIn("event: text", replay.text)
        self.assertIn("event: done", replay.text)
        self.assertIn('"response": "hello"', replay.text)
        # And no resume point past that ending: a reader handed one would
        # store it, having just rendered the answer, and its next attach
        # would resume onto an empty stream it could only read as a lost
        # connection. Mid-turn cursors are covered in test_turn_registry.
        self.assertNotIn("event: cursor", replay.text)

    async def test_a_reader_that_names_no_turn_is_not_handed_one_that_already_ended(self):
        """Asking "is anything running?" must not replay a finished turn.

        Phase 5 probes this route on every mount and every session switch. A
        thread whose last turn finished ten seconds ago still resolves through
        ``last_turn`` for another ten minutes, so an unqualified probe would
        replay that whole answer into a fresh bubble on top of the history
        that already contains it — a duplicate answer, on an ordinary session
        switch.
        """
        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                turn_id = await self.start_turn(client)
                await self.registry.wait(turn_id)
                probe = await client.get(
                    f"/chat/{self.thread_id}/stream", headers=self.auth_headers,
                )

        self.assertEqual(probe.status_code, 404)

    async def test_a_reader_that_names_the_turn_still_gets_it_after_it_ends(self):
        """The other half, and the reason ``last_turn`` exists at all.

        A reader naming a turn is coming back to one it already knows about —
        the tab that sent the message, or a remount resuming from a stored
        cursor. The answer does not reach history until the turn is written
        back, so the window between "the turn stopped" and "history has it"
        is exactly what that reader would otherwise fall into.

        The id is a statement of intent, not an assertion: whatever this
        thread's turn actually is, is what gets streamed.
        """
        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                turn_id = await self.start_turn(client)
                await self.registry.wait(turn_id)
                replay = await client.get(
                    f"/chat/{self.thread_id}/stream",
                    params={"turn": turn_id},
                    headers=self.auth_headers,
                )

        self.assertEqual(replay.status_code, 200)
        self.assertIn('"response": "hello"', replay.text)

    async def test_a_second_send_while_a_turn_runs_is_refused_with_the_running_turn(self):
        """D12. The second tab is told which turn to join, not forked onto its own.

        Two concurrent turns on one thread interleave checkpoint writes, and
        the only thing preventing that today is the client aborting its own
        previous request — which detached turns deliberately stop doing.
        """
        gate = asyncio.Event()
        verifier, save, metadata, owns, stream = self.serving(
            ("status", {"message": "Working"}), gate, ("text", "hello"),
        )
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                running = await self.start_turn(client)
                refused = await client.post(
                    "/chat",
                    json={"message": "again", "thread_id": self.thread_id},
                    headers=self.auth_headers,
                )
                gate.set()
                await self.registry.wait(running)

        self.assertEqual(refused.status_code, 409)
        self.assertEqual(refused.json()["turn_id"], running)

    async def test_a_retried_send_is_answered_with_the_turn_it_already_bought(self):
        """D13. The 202 handshake makes the retry window real.

        A client that retries a send whose response it never saw must not buy
        a second LLM answer and a second retrieval for one message.
        """
        key = {"Idempotency-Key": str(uuid.uuid4()), **self.auth_headers}
        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                first = await client.post(
                    "/chat", json={"message": "hi", "thread_id": self.thread_id}, headers=key,
                )
                await self.registry.wait(first.json()["turn_id"])
                retry = await client.post(
                    "/chat", json={"message": "hi", "thread_id": self.thread_id}, headers=key,
                )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(retry.json()["turn_id"], first.json()["turn_id"])

    async def test_the_stream_refuses_a_thread_that_is_not_the_callers(self):
        """A turn id is not a capability — the thread's ownership is the check.

        Guard, not a driver: the endpoint has this check from the start, and
        losing it would hand one user another's live narration.
        """
        async def owned_by_somebody_else(thread_id, user_id):
            return False

        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, stream, patch.object(
            self.api, "session_belongs_to_user", owned_by_somebody_else
        ):
            async with self.client() as client:
                response = await client.get(
                    f"/chat/{self.thread_id}/stream", headers=self.auth_headers,
                )

        self.assertEqual(response.status_code, 404)

    async def test_a_send_fails_with_503_when_the_event_log_is_unreachable(self):
        """D15. Redis is the transport for every event, the claim and the stop
        signal, so a backend that cannot reach it is not degraded, it is down.

        The rejected alternatives both hide it: falling back to direct
        streaming keeps a branch alive that only ever runs during an incident,
        and running with no narration is a silent five-minute spinner.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        unreachable = TurnRegistry(self.log, url="redis://127.0.0.1:1/15")
        self.addAsyncCleanup(unreachable.aclose)
        self.api.app.state.turn_registry = unreachable

        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                response = await client.post(
                    "/chat",
                    json={"message": "hi", "thread_id": self.thread_id},
                    headers=self.auth_headers,
                )

        self.assertEqual(response.status_code, 503)

    async def test_a_send_landing_mid_shutdown_is_refused_rather_than_started(self):
        """A deploy drains, and the send that arrives one moment later.

        Starting it would produce the worst of both: the drain has already
        passed over the turns it was going to mark, so this one gets no
        ``interrupted`` entry, keeps the thread's claim until its TTL runs
        out, and leaves its reader watching a stream that simply stops. A 503
        is what sends the retry to a replica that is staying up.
        """
        await self.registry.drain()

        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                response = await client.post(
                    "/chat",
                    json={"message": "hi", "thread_id": self.thread_id},
                    headers=self.auth_headers,
                )

        self.assertEqual(response.status_code, 503)

    async def test_with_the_kill_switch_off_the_post_streams_the_turn_itself(self):
        """Old path or new path, wholesale — never a blend.

        The two disagree about what ``POST /chat`` returns, so this is the one
        switch that decides, and it stays off until the frontend speaks the
        202-then-GET protocol.
        """
        from tta_backend.config.settings import get_settings

        verifier, save, metadata, owns, stream = self.serving(("text", "hello"))
        with patch.dict(os.environ, {"CHAT_DETACHED_TURNS_ENABLED": "0"}):
            get_settings.cache_clear()
            with verifier, save, metadata, owns, stream:
                async with self.client() as client:
                    response = await client.post(
                        "/chat",
                        json={"message": "hi", "thread_id": self.thread_id},
                        headers=self.auth_headers,
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertIn('"response": "hello"', response.text)

    def test_detached_turns_are_on_unless_something_turns_them_off(self):
        """Phase 5 flips the default, and the switch reverses with it.

        While the frontend could not speak the 202-then-GET protocol, an
        unset variable had to mean "off" — reading a 202's JSON body into an
        SSE parser finds no events and spins forever. It speaks it now, so
        the default is the shipped path and the variable is what a rollback
        sets.

        The rollback has to work from the environment alone: the frontend
        branches on the response rather than on a flag of its own, so turning
        this off is a backend restart with no image rebuild behind it.
        """
        from tta_backend.config.settings import get_settings

        for value, expected in [(None, True), ("0", False), ("1", True)]:
            with self.subTest(value=value):
                environment = dict(os.environ)
                environment.pop("CHAT_DETACHED_TURNS_ENABLED", None)
                if value is not None:
                    environment["CHAT_DETACHED_TURNS_ENABLED"] = value
                with patch.dict(os.environ, environment, clear=True):
                    get_settings.cache_clear()
                    self.assertEqual(
                        get_settings().chat_detached_turns_enabled, expected
                    )
        get_settings.cache_clear()

    async def test_the_streams_duration_is_measured_over_the_whole_turn(self):
        """T45's property, on the route that now holds a turn's minutes.

        The POST used to stream for 100–370s and was the slowest thing in the
        app; it now returns in milliseconds, and every second of the turn is
        spent on this GET instead. Measuring it at header time here would
        reproduce exactly the blind spot T45 closed — a p95 dashboard that
        cannot see the slowest thing in the app — just one route over.
        """
        import tta_backend.api as api

        working = asyncio.Event()
        self.addCleanup(working.set)
        turn_seconds = 0.3
        observed = []

        def fake_observe(method, path, status_code, duration_seconds):
            observed.append((method, path, status_code, duration_seconds))

        verifier, save, metadata, owns, stream = self.serving(
            ("status", {"message": "Searching granules"}), working, ("text", "hello"),
        )
        with verifier, save, metadata, owns, stream, \
             patch.object(api, "observe_http_request", fake_observe):
            async with self.client() as client:
                turn_id = await self.start_turn(client)
                # Let the turn go on working while the reader is attached, so
                # the time being measured is time spent streaming rather than
                # time spent replaying something already finished.
                asyncio.get_running_loop().call_later(turn_seconds, working.set)
                await client.get(
                    f"/chat/{self.thread_id}/stream",
                    params={"turn": turn_id},
                    headers=self.auth_headers,
                )

        streams = [o for o in observed if o[1].endswith("/stream")]
        self.assertEqual(len(streams), 1, observed)
        self.assertEqual(streams[0][0], "GET")
        self.assertEqual(streams[0][2], 200)
        self.assertGreaterEqual(streams[0][3], turn_seconds)

    async def test_stop_ends_the_threads_running_turn(self):
        """Stop names a thread, not a turn.

        The button is pressed in a tab that may have joined the turn rather
        than started it (D12) — after a reattach it knows the thread and
        nothing else — so the server resolves which turn that is.
        """
        blocked = asyncio.Event()
        self.addCleanup(blocked.set)
        verifier, save, metadata, owns, stream = self.serving(
            ("status", {"message": "Reducing"}), blocked, ("text", "never"),
        )
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                turn_id = await self.start_turn(client)

                response = await client.post(
                    f"/chat/{self.thread_id}/stop", headers=self.auth_headers,
                )

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["turn_id"], turn_id)
                await asyncio.wait_for(self.registry.wait(turn_id), timeout=5.0)

        self.assertEqual(await self.log.terminal_of(turn_id), "stopped")

    async def test_stop_on_a_thread_with_no_turn_is_a_404(self):
        """There is nothing to stop, and saying so beats a cheerful 200.

        A 200 here would tell a frontend its Stop landed when the turn it
        meant is running somewhere the server could not find.
        """
        verifier, save, metadata, owns, stream = self.serving(("text", "hi"))
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                response = await client.post(
                    f"/chat/{uuid.uuid4()}/stop", headers=self.auth_headers,
                )

        self.assertEqual(response.status_code, 404)

    async def test_stop_on_someone_elses_thread_is_a_404(self):
        """Stop is a write on another user's turn; it gets the same gate the
        stream does, and the same answer that does not confirm the thread
        exists."""
        blocked = asyncio.Event()
        self.addCleanup(blocked.set)
        verifier, save, metadata, owns, stream = self.serving(
            ("status", {"message": "Reducing"}), blocked, ("text", "never"),
        )
        intruder = auth_helpers.make_token("user-2", email="other@example.com")
        with verifier, save, metadata, owns, stream:
            async with self.client() as client:
                turn_id = await self.start_turn(client)

                response = await client.post(
                    f"/chat/{self.thread_id}/stop",
                    headers={"Authorization": f"Bearer {intruder}"},
                )

                self.assertEqual(response.status_code, 404)
                # And the turn it tried to stop is untouched.
                self.assertIsNone(await self.log.terminal_of(turn_id))
                await self.registry.stop(turn_id)
                await asyncio.wait_for(self.registry.wait(turn_id), timeout=5.0)

    async def test_stop_cancels_the_retrievals_the_turn_left_at_the_provider(self):
        """D10: the whole point of moving Stop's job cancellation server-side.

        The client used to do this from the ``job_progress`` events it had
        seen. A reader that reattached mid-turn never saw them, so its Stop
        left every retrieval running — the cost falling on exactly the users
        detached turns were built for.
        """
        cancelled: list[str] = []

        async def fake_cancel_job(job_handle, tools):
            cancelled.append(job_handle)
            return {"job_handle": job_handle, "status": "cancelled"}

        class _ReadyMCP:
            state = STATE_READY
            tools: dict = {}

        self.api.app.state.earthdata_mcp_manager = _ReadyMCP()
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

        blocked = asyncio.Event()
        self.addCleanup(blocked.set)
        verifier, save, metadata, owns, stream = self.serving(
            ("job_progress", {"job_handle": "job-a", "status": "running"}),
            blocked,
            ("text", "never"),
        )
        with verifier, save, metadata, owns, stream, patch.object(
            self.api, "cancel_job", fake_cancel_job
        ):
            async with self.client() as client:
                turn_id = await self.start_turn(client)
                await self.await_frames(turn_id)

                await client.post(f"/chat/{self.thread_id}/stop", headers=self.auth_headers)
                await asyncio.wait_for(self.registry.wait(turn_id), timeout=5.0)

        self.assertEqual(cancelled, ["job-a"])

    async def await_frames(self, turn_id: str, timeout: float = 3.0):
        """Wait until the turn has written something, so it is genuinely mid-flight."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not (await self.log.read(turn_id)).frames:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("the turn never produced a frame")
            await asyncio.sleep(0.02)
