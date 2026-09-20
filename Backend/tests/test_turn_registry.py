"""The turn registry: who owns a running turn, and who is allowed to start one.

A turn outlives the request that started it. The registry is what holds it —
it runs the turn as its own task, feeds every frame the turn produces into the
event log, and keeps one turn per thread so two tabs cannot interleave
checkpoint writes on one LangGraph thread.

Run against a **real** Redis for the same reason the event log tests are: the
claim is a `SETNX` with a TTL, and what those actually do across replicas is
the thing under test. They skip when none is reachable so a host-side
``pytest`` still runs.

Nothing here flushes: each test owns its ids and lets ``EXPIRE`` clean up.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import unittest
import uuid
from urllib.parse import urlsplit

#: Database 15 by default, never 0 — the live stack's turns are on 0.
DEFAULT_TEST_REDIS_URL = "redis://127.0.0.1:6379/15"
REDIS_URL = os.environ.get("REDIS_URL") or DEFAULT_TEST_REDIS_URL


def _redis_is_reachable(url: str) -> bool:
    if importlib.util.find_spec("redis") is None:
        return False
    parsed = urlsplit(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 6379), timeout=1.0):
            return True
    except OSError:
        return False


requires_redis = unittest.skipUnless(
    _redis_is_reachable(REDIS_URL), f"no Redis reachable at {REDIS_URL}"
)


def frame(event: str, payload: str) -> str:
    """An SSE frame shaped exactly as ``ChatStreamService.sse`` renders one.

    Written out rather than imported so the byte-identity assertions below
    pin the bytes themselves, not whatever the renderer currently produces.
    """
    return f"event: {event}\ndata: {payload}\n\n"


async def produces(*frames: str):
    """A turn that emits these frames and ends."""
    for item in frames:
        yield item


async def _buffers_then_blocks(gate, *frames: str):
    """Emits these frames and hangs without ever closing the turn.

    Nothing flushes the text among them -- that is the point: the batcher
    still holds it when the stop arrives.
    """
    for item in frames:
        yield item
    await gate.wait()


async def blocks_until(gate, *frames: str, before: str | None = None):
    """A turn that emits ``before``, then waits for the test to let it end."""
    if before is not None:
        yield before
    await gate.wait()
    for item in frames:
        yield item


async def _until(read, ready, timeout: float = 2.0):
    """Poll ``read`` until ``ready`` accepts its result, or fail the wait.

    Turns run as their own tasks, so "the turn has got as far as X" is not
    something a test can await directly.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await read()
        if ready(result):
            return result
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"condition never held; last saw {result}")
        await asyncio.sleep(0.02)


def cursor_of(items: list[str]) -> str:
    """The resume point the last cursor frame carried."""
    for item in reversed(items):
        if item.startswith("event: cursor"):
            return json.loads(item.split("data: ", 1)[1])["cursor"]
    raise AssertionError(f"no cursor frame among {items}")


@requires_redis
class TurnRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        self.log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(self.log.aclose)
        self.registry = TurnRegistry(self.log, url=REDIS_URL)
        self.addAsyncCleanup(self.registry.aclose)
        self.thread_id = f"test-thread-{uuid.uuid4()}"

    async def entries(self, turn_id: str) -> list[dict]:
        """This turn's stream as Redis holds it, one dict per entry.

        Through an independent client. Which frames share an entry is
        invisible to ``TurnEventLog.read`` by design — it returns a flat list
        — but "the closing frame and the terminal marker are one entry" is
        exactly the property that keeps a reader from collecting the close and
        still believing the turn is live.
        """
        from redis import asyncio as aioredis

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            raw = await client.xrange(f"turn:{turn_id}:events", min="-", max="+")
        finally:
            await client.aclose()
        return [fields for _entry_id, fields in raw]

    async def ttl(self, key: str) -> int:
        """Seconds left on a key, as Redis reports it (-1 = no expiry)."""
        from redis import asyncio as aioredis

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            return await client.ttl(key)
        finally:
            await client.aclose()

    async def test_every_frame_a_turn_produces_reaches_the_log_in_order(self):
        sent = [
            frame("status", '{"message": "Searching granules"}'),
            frame("tool_call", '{"name": "search"}'),
            frame("chart", '{"chart_id": "c-1"}'),
        ]

        claim = await self.registry.begin(self.thread_id, produces(*sent))
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames, sent)

    async def test_the_closing_frame_arrives_with_the_terminal_marker(self):
        closing = frame("done", '{"response": "ready"}')

        claim = await self.registry.begin(
            self.thread_id, produces(frame("status", '{"message": "working"}'), closing)
        )
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames[-1], closing)
        self.assertEqual(page.terminal, "done")
        last = (await self.entries(claim.turn_id))[-1]
        self.assertEqual(last.get("terminal"), "done")
        self.assertEqual(last.get("f0"), closing)

    async def test_a_turn_that_only_reports_an_error_still_ends_its_stream(self):
        """The generic-failure path emits an ``error`` and no ``done``.

        Under the old architecture the connection closing was what told the
        client the turn was over. Nothing closes now, so an errored turn that
        wrote no terminal entry would leave a reader waiting forever.
        """
        failure = frame("error", '{"detail": "Something went wrong"}')

        claim = await self.registry.begin(
            self.thread_id, produces(frame("status", '{"message": "working"}'), failure)
        )
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames[-1], failure)
        self.assertEqual(page.terminal, "error")

    async def test_an_error_that_is_followed_by_a_close_keeps_both(self):
        """A turn that times out or sheds emits ``error`` **then** ``done``.

        Holding closing frames back must not swallow the first of a pair: the
        error carries the explanation the user reads, the done carries the
        payload the client reconciles against.
        """
        failure = frame("error", '{"detail": "This turn ran too long"}')
        closing = frame("done", '{"response": ""}')

        claim = await self.registry.begin(self.thread_id, produces(failure, closing))
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames, [failure, closing])
        self.assertEqual(page.terminal, "done")

    async def test_answer_text_is_batched_while_the_frames_around_it_are_not(self):
        """Answer tokens share an entry; everything else gets its own.

        A 2,000-token answer is otherwise 2,000 round-trips to Redis. The
        frames that drive UI state stay immediate, and — because a structural
        frame flushes the buffer before it writes — none of them can overtake
        text produced before it.
        """
        tokens = [frame("text", '{"content": "word%d"}' % index) for index in range(5)]
        chart = frame("chart", '{"chart_id": "c-9"}')
        closing = frame("done", '{"response": "word0word1word2word3word4"}')

        claim = await self.registry.begin(
            self.thread_id, produces(*tokens, chart, closing)
        )
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames, [*tokens, chart, closing])
        # The five tokens together, the chart alone, the close in the terminal.
        self.assertEqual(len(await self.entries(claim.turn_id)), 3)

    async def test_a_second_send_while_a_turn_runs_is_refused_with_the_running_turn(self):
        """One turn per thread (D12).

        The supervisor is one LangGraph thread on a shared checkpointer, and
        two concurrent ``astream`` calls on it interleave checkpoint writes.
        The second tab is told which turn is running so it can join that one
        rather than forking a rival.
        """
        gate = asyncio.Event()
        dispatched = []

        async def records():
            dispatched.append("ran")
            yield frame("done", "{}")

        first = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", '{"response": "first"}'))
        )
        second = await self.registry.begin(self.thread_id, records())

        self.assertEqual(second.outcome, "already_running")
        self.assertEqual(second.turn_id, first.turn_id)
        self.assertEqual(dispatched, [])

        gate.set()
        await self.registry.wait(first.turn_id)

    async def test_the_thread_is_free_again_once_its_turn_ends(self):
        first = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "first"}'))
        )
        await self.registry.wait(first.turn_id)

        second = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "second"}'))
        )
        self.addAsyncCleanup(self.registry.wait, second.turn_id)

        self.assertEqual(second.outcome, "started")
        self.assertNotEqual(second.turn_id, first.turn_id)

    async def test_a_reader_is_pointed_at_the_turn_running_on_its_thread(self):
        gate = asyncio.Event()
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )

        self.assertEqual(await self.registry.turn_for(self.thread_id), claim.turn_id)

        gate.set()
        await self.registry.wait(claim.turn_id)

    async def test_a_reader_arriving_just_after_the_turn_ended_still_finds_its_stream(self):
        """The claim is gone the instant the turn ends, but the stream is not.

        A tab switched back to at the wrong moment would otherwise be told the
        thread has no turn and fall back to history — which does not yet hold
        the answer the terminal entry is about to deliver.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "finished"}'))
        )
        await self.registry.wait(claim.turn_id)

        self.assertEqual(await self.registry.turn_for(self.thread_id), claim.turn_id)

    async def test_a_retried_send_gets_the_first_turn_back_and_starts_no_second(self):
        """D13. A 202 handshake makes the retry window real.

        The per-thread claim covers a retry that arrives while the turn is
        still running. It does not cover one that arrives after the turn
        finished — and that retry would otherwise buy a second LLM answer and
        a duplicate retrieval for a message the user sent once.
        """
        key = f"idem-{uuid.uuid4()}"
        dispatched = []

        async def records():
            dispatched.append("ran")
            yield frame("done", "{}")

        first = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "once"}')), idempotency_key=key
        )
        await self.registry.wait(first.turn_id)

        retry = await self.registry.begin(self.thread_id, records(), idempotency_key=key)

        self.assertEqual(retry.turn_id, first.turn_id)
        self.assertEqual(retry.outcome, "duplicate")
        self.assertEqual(dispatched, [])

    async def test_a_claim_left_behind_by_a_dead_replica_expires(self):
        """A claim is released by the turn that holds it — unless nothing is
        left to release it.

        A replica killed mid-turn leaves its claim in Redis with no owner, and
        a claim with no expiry would refuse every later message on that thread
        for good. The bound is the longest a turn may legitimately run.
        """
        from tta_backend.config.settings import get_settings

        gate = asyncio.Event()
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )

        remaining = await self.ttl(f"thread:{self.thread_id}:active_turn")
        self.assertGreater(remaining, int(get_settings().chat_turn_timeout_seconds))

        gate.set()
        await self.registry.wait(claim.turn_id)

    async def collect(self, turn_id: str, cursor: str | None = None) -> list[str]:
        """Everything a reader following this turn is handed, until it ends."""
        got = []
        async for item in self.registry.follow(turn_id, cursor):
            got.append(item)
        return got

    async def test_following_a_live_turn_delivers_its_frames_and_then_ends(self):
        """A reader attaches while the turn is still working.

        Nothing closes this from the turn's side — under the old architecture
        the response generator returning was what ended the stream. The
        terminal entry is what takes that job over, and a follower that missed
        it would hold its connection open forever.
        """
        gate = asyncio.Event()
        opening = frame("status", '{"message": "Searching granules"}')
        closing = frame("done", '{"response": "hello"}')

        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, closing, before=opening)
        )
        following = asyncio.create_task(self.collect(claim.turn_id))
        gate.set()

        delivered = await asyncio.wait_for(following, timeout=5)

        self.assertIn(opening, delivered)
        self.assertIn(closing, delivered)

    async def follow_until_cursor(self, turn_id: str, cursor: str | None = None) -> list[str]:
        """Follow until the first resume point, then walk away — a detach."""
        got = []
        stream = self.registry.follow(turn_id, cursor)
        try:
            async for item in stream:
                got.append(item)
                if item.startswith("event: cursor"):
                    return got
        finally:
            await stream.aclose()
        raise AssertionError(f"the turn ended without a cursor: {got}")

    async def test_a_follower_resuming_from_its_cursor_is_not_sent_what_it_saw(self):
        """Leaving mid-turn and coming back costs a reader its place, not the turn."""
        gate = asyncio.Event()
        opening = frame("status", '{"message": "Searching granules"}')
        closing = frame("done", '{"response": "hello"}')
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, closing, before=opening)
        )

        before = await asyncio.wait_for(self.follow_until_cursor(claim.turn_id), timeout=5)
        gate.set()
        await self.registry.wait(claim.turn_id)
        after = await asyncio.wait_for(
            self.collect(claim.turn_id, cursor_of(before)), timeout=5
        )

        self.assertIn(opening, before)
        self.assertNotIn(opening, after)
        self.assertIn(closing, after)

    async def test_a_follower_that_resumes_past_the_end_stops_instead_of_waiting(self):
        """A remount replays from a stored cursor — which may already be the last one.

        The terminal marker only reaches a reader inside the page that holds
        it. A reader resuming from beyond that point sees an empty page
        forever, and would hold its connection open against a turn that ended
        long ago.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "hello"}'))
        )
        await self.registry.wait(claim.turn_id)
        everything = await asyncio.wait_for(self.collect(claim.turn_id), timeout=5)

        again = await asyncio.wait_for(
            self.collect(claim.turn_id, cursor_of(everything)), timeout=5
        )

        self.assertEqual(again, [])

    async def test_a_finished_turn_is_not_kept_after_it_ends(self):
        """The registry holds a turn while it runs, and lets go when it stops.

        A replica serving turns for weeks would otherwise accumulate one task
        per turn ever started, each holding its coroutine frame and everything
        that frame captured.
        """
        gate = asyncio.Event()
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )
        self.assertEqual(self.registry.in_flight, 1)

        gate.set()
        await self.registry.wait(claim.turn_id)

        self.assertEqual(self.registry.in_flight, 0)

    async def test_a_send_refused_for_a_running_turn_does_not_bank_its_key(self):
        """A refused send bought nothing, so its key must not be spent.

        The idempotency record is claimed before the thread is — it has to be,
        or two retries race. When the thread claim then fails, that record
        names a turn that was never started, and the retry would be answered
        with a turn id that has no stream behind it.
        """
        gate = asyncio.Event()
        key = f"idem-{uuid.uuid4()}"
        running = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", '{"response": "first"}'))
        )

        refused = await self.registry.begin(
            self.thread_id, produces(frame("done", "{}")), idempotency_key=key
        )
        self.assertEqual(refused.outcome, "already_running")
        self.assertEqual(refused.turn_id, running.turn_id)

        gate.set()
        await self.registry.wait(running.turn_id)

        retry = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "second"}')), idempotency_key=key
        )
        self.addAsyncCleanup(self.registry.wait, retry.turn_id)
        self.assertEqual(retry.outcome, "started")

    async def test_stopping_a_running_turn_ends_it_and_marks_the_log_stopped(self):
        """The tracer bullet for Stop: the turn stops producing and says so.

        A reader has no other way to learn the turn is over -- a stopped turn
        emits no ``done`` of its own, so without the terminal entry the stream
        just goes quiet and looks like a dead replica (D14).
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", '{"response": "never"}'))
        )

        await self.registry.stop(claim.turn_id)
        await self.registry.wait(claim.turn_id)

        self.assertEqual(await self.log.terminal_of(claim.turn_id), "stopped")

    async def test_a_stopped_turn_keeps_what_it_produced_before_the_stop(self):
        """D11: a stopped turn shows the work done, not a blank cancelled turn.

        Including the answer tokens still sitting in the text batcher when the
        cancel lands. Text is held ~50ms to spare the log an XADD per token,
        so a stop mid-sentence is the ordinary case, not a corner: losing that
        buffer truncates the partial answer the user was reading.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        chart = frame("chart", '{"chart_id": "c-1"}')
        tail = frame("text", '{"content": "the levels were"}')
        claim = await self.registry.begin(
            self.thread_id, _buffers_then_blocks(gate, chart, tail)
        )
        await _until(lambda: self.log.read(claim.turn_id), lambda page: page.frames)

        await self.registry.stop(claim.turn_id)
        await self.registry.wait(claim.turn_id)

        page = await self.log.read(claim.turn_id)
        self.assertEqual(page.frames[:2], [chart, tail])
        self.assertEqual(page.terminal, "stopped")

    async def test_a_stopped_turn_hands_the_thread_back(self):
        """Stop then retype: the next message has to be accepted.

        A claim the stop path forgets to release refuses every later message
        on that thread until its TTL runs out -- and that TTL is the whole
        turn deadline, so the thread reads as broken for half an hour.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        stopped = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )

        await self.registry.stop(stopped.turn_id)
        await self.registry.wait(stopped.turn_id)

        nxt = await self.registry.begin(self.thread_id, produces(frame("done", "{}")))
        self.addAsyncCleanup(self.registry.wait, nxt.turn_id)
        self.assertEqual(nxt.outcome, "started")

    async def test_a_reader_following_a_stopped_turn_is_released(self):
        """The follower loop has to end, not hold a connection on a dead turn.

        Its exit is driven by the terminal entry alone -- a stopped turn never
        emits ``done`` -- so this is what proves the ``stopped`` marker is
        wired to the reader and not just written.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )

        async def read_to_the_end() -> list[str]:
            return [item async for item in self.registry.follow(claim.turn_id)]

        reader = asyncio.create_task(read_to_the_end())
        await self.registry.stop(claim.turn_id)

        delivered = await asyncio.wait_for(reader, timeout=5.0)
        self.assertTrue(any(item.startswith("event: stopped") for item in delivered))

    async def test_a_stop_from_another_replica_reaches_the_turns_owner(self):
        """The case sticky sessions fail silently.

        Behind an ALB the Stop POST lands wherever it lands, which is usually
        not the replica running the turn. The stopping registry has no task to
        cancel -- it has never heard of this turn -- so the signal has to
        cross the process boundary or Stop does nothing for half the users
        who press it.
        """
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        other_log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(other_log.aclose)
        other_replica = TurnRegistry(other_log, url=REDIS_URL)
        self.addAsyncCleanup(other_replica.aclose)

        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )
        await _until(lambda: self.log.terminal_of(claim.turn_id), lambda _: self.registry.in_flight == 1)

        await other_replica.stop(claim.turn_id)

        await asyncio.wait_for(self.registry.wait(claim.turn_id), timeout=5.0)
        self.assertEqual(await self.log.terminal_of(claim.turn_id), "stopped")

    async def test_stopping_a_finished_turn_leaves_its_ending_alone(self):
        """Stop is racy by nature and must be a no-op when it loses.

        The route resolves a thread to its *last* turn, not only a running
        one, so a Stop pressed as the answer lands resolves to a turn that is
        already over. Writing ``stopped`` on top of ``done`` would relabel a
        completed answer as cancelled -- and since ``terminal_of`` reads the
        last entry, the relabelling would win.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "ready"}'))
        )
        await self.registry.wait(claim.turn_id)
        self.assertEqual(await self.log.terminal_of(claim.turn_id), "done")

        await self.registry.stop(claim.turn_id)
        await asyncio.sleep(0.1)

        self.assertEqual(await self.log.terminal_of(claim.turn_id), "done")
        self.assertEqual(len(await self.entries(claim.turn_id)), 1)

    async def test_a_stop_whose_publish_is_lost_still_stops_the_turn(self):
        """Why there is a persisted flag as well as a channel.

        Pub/sub delivers to whoever is subscribed *now*: a replica between
        reconnects, or one whose turn was announced in the instant before it
        finished subscribing, simply never hears. The stopping replica gets no
        acknowledgement either way, so without the flag a Stop can be
        swallowed with the user watching a turn it was told had stopped.

        Staged with a replica whose subscription never delivers, which is what
        a lost message looks like from the owner's side.
        """
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        class _DeafReplica(TurnRegistry):
            async def _listen(self, pubsub):
                await asyncio.Event().wait()

        deaf_log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(deaf_log.aclose)
        owner = _DeafReplica(deaf_log, url=REDIS_URL)
        self.addAsyncCleanup(owner.aclose)

        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await owner.begin(self.thread_id, blocks_until(gate, frame("done", "{}")))

        await self.registry.stop(claim.turn_id)

        await asyncio.wait_for(owner.wait(claim.turn_id), timeout=5.0)
        self.assertEqual(await self.log.terminal_of(claim.turn_id), "stopped")

    async def test_stopping_cancels_the_provider_jobs_the_turn_left_running(self):
        """D10: a hard cancel abandons provider work unless someone says so.

        Cancelling the turn's task does nothing to a retrieval already running
        at the provider -- it outlives the connection that asked for it. The
        server tracks the same last-seen statuses the client used to, because
        a user who reattached mid-turn never saw those ``job_progress`` events
        and their Stop would leak every one of them.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        cancelled: list[str] = []

        async def cancel_jobs(handles):
            cancelled.extend(handles)

        claim = await self.registry.begin(
            self.thread_id,
            blocks_until(
                gate,
                frame("done", "{}"),
                before=frame("job_progress", '{"job_handle": "job-a", "status": "running"}'),
            ),
            cancel_jobs=cancel_jobs,
        )
        await _until(lambda: self.log.read(claim.turn_id), lambda page: page.frames)

        await self.registry.stop(claim.turn_id)
        await self.registry.wait(claim.turn_id)

        self.assertEqual(cancelled, ["job-a"])

    async def test_stopping_leaves_alone_the_jobs_that_already_finished(self):
        """Only what is still running is cancelled.

        A retrieval that reached ``ready`` produced a result the user keeps
        (D11) and may already be cached; cancelling it at the provider throws
        that away and can turn a completed job into a ``cancelled`` row in the
        Jobs panel, which reads as data loss rather than as a stop.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        cancelled: list[str] = []

        async def cancel_jobs(handles):
            cancelled.extend(handles)

        claim = await self.registry.begin(
            self.thread_id,
            _buffers_then_blocks(
                gate,
                frame("job_progress", '{"job_handle": "job-a", "status": "running"}'),
                frame("job_progress", '{"job_handle": "job-b", "status": "running"}'),
                # The same handle again: what matters is its *last* status,
                # not that it was ever seen running.
                frame("job_progress", '{"job_handle": "job-a", "status": "ready"}'),
            ),
            cancel_jobs=cancel_jobs,
        )
        await _until(
            lambda: self.log.read(claim.turn_id), lambda page: len(page.frames) >= 3
        )

        await self.registry.stop(claim.turn_id)
        await self.registry.wait(claim.turn_id)

        self.assertEqual(cancelled, ["job-b"])

    async def test_a_replica_whose_redis_connection_blips_still_hears_stops(self):
        """The subscription has to survive a dropped connection.

        redis-py does not resubscribe on its own: the read raises once and, if
        nothing catches it, the listener task dies and takes the channel with
        it for the rest of the process's life. Correctness survives that --
        the flag watchdog still stops the turn -- but every Stop on that
        replica silently degrades to the poll interval, for hours, with
        nothing in the logs saying why.

        Pinned by giving this registry a watchdog too slow to help, so only
        the channel can satisfy it.
        """
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        owner_log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(owner_log.aclose)
        owner = TurnRegistry(owner_log, url=REDIS_URL, stop_poll_interval=3600.0)
        self.addAsyncCleanup(owner.aclose)

        gate = asyncio.Event()
        self.addCleanup(gate.set)
        warmup = await owner.begin(self.thread_id, produces(frame("done", "{}")))
        await owner.wait(warmup.turn_id)

        # Straight at the socket: this is an environmental fault, not a seam
        # worth carving into the registry.
        await owner._pubsub.connection.disconnect()

        second_thread = f"test-thread-{uuid.uuid4()}"
        claim = await owner.begin(second_thread, blocks_until(gate, frame("done", "{}")))
        await _until(lambda: self.log.terminal_of(claim.turn_id), lambda _: owner.in_flight == 1)

        await self.registry.stop(claim.turn_id)

        await asyncio.wait_for(owner.wait(claim.turn_id), timeout=10.0)
        self.assertEqual(await self.log.terminal_of(claim.turn_id), "stopped")
