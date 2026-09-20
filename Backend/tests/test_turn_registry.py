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
    cursor = cursor_or_none(items)
    if cursor is None:
        raise AssertionError(f"no cursor frame among {items}")
    return cursor


def cursor_or_none(items: list[str]) -> str | None:
    """The last resume point offered, or ``None`` if none was.

    A reader stores what it is given and hands it straight back, so ``None``
    here is what a reader that was offered nothing reattaches with.
    """
    for item in reversed(items):
        if item.startswith("event: cursor"):
            return json.loads(item.split("data: ", 1)[1])["cursor"]
    return None


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

    def terminal_now(self, turn_id: str) -> str | None:
        """How this turn ended, read with a blocking client on purpose.

        Synchronous so that asking the question grants the event loop no
        iterations -- see the shutdown test, where "has it been written yet"
        is the entire assertion.
        """
        import redis

        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            entries = client.xrevrange(f"turn:{turn_id}:events", max="+", min="-", count=1)
        finally:
            client.close()
        return (entries[0][1] or {}).get("terminal") if entries else None

    async def last_entry_id(self, turn_id: str) -> str:
        """The id of this turn's last entry, straight from Redis.

        A cursor naming the end is no longer something a reader can be given,
        so a test that needs one has to take it from the log itself.
        """
        from redis import asyncio as aioredis

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            entries = await client.xrevrange(f"turn:{turn_id}:events", max="+", min="-", count=1)
        finally:
            await client.aclose()
        return str(entries[0][0])

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

    async def test_a_claim_left_behind_by_a_dead_replica_expires_within_the_silence_readers_tolerate(self):
        """A replica killed mid-turn must not wedge its thread for 40 minutes.

        A claim is released by the turn that holds it — unless nothing is left
        to release it, which is exactly what a SIGKILL or an OOM leaves
        behind. Bounding it by the longest a turn may legitimately run
        (`chat_turn_timeout` + margin, ~40 min) makes the sequence a reader
        already walks — stream goes quiet, reader reports `interrupted`, user
        retries — answer 409 naming a turn that is dead, for the rest of that
        window.

        So the bound is the same silence a reader is willing to believe in
        (D14): a turn nobody would still call alive is a turn whose thread is
        free. The owner keeps its own claim fresh (below), so a *live* turn
        is never bounded by this at all.
        """
        from tta_backend.services import turn_registry as registry_module

        gate = asyncio.Event()
        claim = await self.registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )

        remaining = await self.ttl(f"thread:{self.thread_id}:active_turn")
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, registry_module.CLAIM_TTL_SECONDS)

        gate.set()
        await self.registry.wait(claim.turn_id)

    async def test_a_live_turn_keeps_its_own_claim_past_the_ttl_it_was_given(self):
        """The other half: a short claim is only safe if its owner renews it.

        Without this the bound above would start refusing messages on threads
        whose turn is still working — turns run 100–370s. The owner renews on
        the same loop that re-reads the stop flags, so a claim is lost only by
        a replica that has stopped running its own event loop, which is the
        case the bound exists for.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        registry = TurnRegistry(
            self.log, url=REDIS_URL, claim_ttl=2, stop_poll_interval=0.2,
        )
        self.addAsyncCleanup(registry.aclose)
        gate = asyncio.Event()
        claim = await registry.begin(
            self.thread_id, blocks_until(gate, frame("done", "{}"))
        )
        key = f"thread:{self.thread_id}:active_turn"

        # Past the TTL the claim was written with, so only a renewal can
        # explain it still being there.
        await asyncio.sleep(3.0)

        self.assertEqual(await self.registry.turn_for(self.thread_id), claim.turn_id)
        self.assertGreater(await self.ttl(key), 0)

        gate.set()
        await registry.wait(claim.turn_id)
        # And it is still handed back the moment the turn ends, renewals or no.
        self.assertEqual(await self.ttl(key), -2)

    async def test_the_turn_stops_being_renewed_the_moment_it_ends(self):
        """First guard: a finished turn is off the renewal list.

        It leaves that list in the same ``finally`` that hands the thread
        back, before the release is even awaited, so the ordinary case never
        reaches the race below at all.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        registry = TurnRegistry(
            self.log, url=REDIS_URL, claim_ttl=2, stop_poll_interval=0.05,
        )
        self.addAsyncCleanup(registry.aclose)
        claim = await registry.begin(self.thread_id, produces(frame("done", "{}")))
        await registry.wait(claim.turn_id)

        # Several renewal ticks after the turn let go of the thread.
        await asyncio.sleep(0.3)

        self.assertEqual(await self.ttl(f"thread:{self.thread_id}:active_turn"), -2)
        # Left on the list it would renew nothing (EXPIRE on a missing key
        # does nothing), so the cost is invisible until it is one dict entry
        # per turn for the life of the process -- the same leak dropping
        # ``_tasks`` from inside this ``finally`` exists to avoid.
        self.assertEqual(registry._threads, {})

    async def test_a_renewal_never_creates_a_claim_that_is_not_there(self):
        """Second guard, for the window the first one cannot cover.

        ``_refresh_claims`` snapshots which threads to renew and then awaits
        the round trip; a turn ending inside that await has already deleted
        its claim by the time the renewal lands. EXPIRE on a missing key does
        nothing. A renewal written as SET-with-TTL — the obvious alternative,
        and the one that also re-asserts ownership — would put the claim back
        with nobody left to release it, wedging the thread until the TTL runs
        out every single time a turn ends mid-tick.

        Called directly because the race is a scheduling accident: staging it
        through the public path would pin the timing, not the property.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        registry = TurnRegistry(self.log, url=REDIS_URL, claim_ttl=2)
        self.addAsyncCleanup(registry.aclose)
        key = f"thread:{self.thread_id}:active_turn"
        registry._threads[f"turn-{uuid.uuid4()}"] = self.thread_id

        await registry._refresh_claims()

        self.assertEqual(await self.ttl(key), -2)

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

        The cursor comes from the log rather than from a cursor frame: no
        reader is handed one past the end any more (see the test below), so
        the only way to be in this position is an old client or a stale store.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "hello"}'))
        )
        await self.registry.wait(claim.turn_id)
        await asyncio.wait_for(self.collect(claim.turn_id), timeout=5)

        again = await asyncio.wait_for(
            self.collect(claim.turn_id, await self.last_entry_id(claim.turn_id)),
            timeout=5,
        )

        self.assertEqual(again, [])

    async def test_a_reader_that_saw_the_end_is_never_sent_back_to_an_empty_stream(self):
        """The resume point a reader is given must never be the end itself.

        The cursor frame trails the page it accounts for, so a page carrying
        the turn's last frame was being followed by a cursor naming it — which
        the reader stores, having just rendered the ending. Its next attach
        then resumes onto a stream with nothing left in it: 200, no frames, no
        terminal, and no way to tell a finished turn from a dead one. Live,
        that reported "connection lost" on top of the answer already on
        screen, and did it after every completed turn.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "hello"}'))
        )
        await self.registry.wait(claim.turn_id)

        delivered = await asyncio.wait_for(self.collect(claim.turn_id), timeout=5)
        resumed = await asyncio.wait_for(
            self.collect(claim.turn_id, cursor_or_none(delivered)), timeout=5
        )

        self.assertTrue(any(item.startswith("event: done") for item in delivered))
        self.assertTrue(any(item.startswith("event: done") for item in resumed))

    async def test_a_cursor_left_over_from_an_earlier_turn_skips_nothing(self):
        """Phase 5 persists one cursor per thread, and reuses it blind.

        A remount learns which turn is running only from the frames it is
        already being handed, so the cursor it resumes with may belong to the
        *previous* turn on that thread. That is safe for one reason and one
        only: a stream id is a wall-clock millisecond, so an older turn's
        cursor sorts before every entry a newer turn writes and the range
        read returns all of them.

        Pinned because it is load-bearing and invisible. Numbering entries
        per turn instead — the obvious thing to reach for if ids ever get
        tidied up — would make that same cursor land in the middle of the new
        turn and silently drop the beginning of the answer, with no error on
        either side.
        """
        first = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "first"}'))
        )
        await self.registry.wait(first.turn_id)
        # The last id the first turn ever wrote: the furthest forward a
        # leftover cursor can point, and so the most adversarial one the
        # second turn can be resumed with.
        spent = await self.last_entry_id(first.turn_id)

        second = await self.registry.begin(
            self.thread_id,
            produces(
                frame("status", '{"message": "Searching granules"}'),
                frame("done", '{"response": "second"}'),
            ),
        )
        await self.registry.wait(second.turn_id)

        resumed = await asyncio.wait_for(
            self.collect(second.turn_id, spent), timeout=5
        )

        self.assertNotEqual(first.turn_id, second.turn_id)
        joined = "".join(resumed)
        self.assertIn("Searching granules", joined)
        self.assertIn('"response": "second"', joined)

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

    async def test_a_replica_going_down_marks_its_live_turns_interrupted(self):
        """The tracer bullet for shutdown: a deploy tells the reader.

        Without this the stream simply stops mid-sentence and every attached
        reader spins against a turn nobody is running any more.

        Asserted on what is in Redis rather than on the code path reached:
        shutting down is a sequence of cancels and awaits in which an entry
        can be written by a task nobody waited for, well after the call that
        was supposed to produce it returned, so "the write was reached" and
        "the write survived" are different claims.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id,
            blocks_until(gate, frame("done", "{}"), before=frame("status", "{}")),
        )
        await _until(lambda: self.log.read(claim.turn_id), lambda page: page.frames)

        await self.registry.aclose()

        # Read without yielding the event loop: shutdown's contract is that
        # the entry is in Redis by the time ``aclose`` returns, because the
        # next thing the lifespan does is close the pool it was written
        # through and let the process exit. An ``await`` here would hand the
        # loop back and let a write nobody waited for land after the fact,
        # which is how this assertion passes against a drain that does not
        # actually drain.
        self.assertEqual(self.terminal_now(claim.turn_id), "interrupted")

    async def test_shutting_down_waits_for_its_turns_to_finish_saying_so(self):
        """``aclose`` returning has to mean the entries are in Redis.

        What follows it in the lifespan is the close of the pool those
        entries travel through and then the process exiting, so a shutdown
        that merely starts the writes loses whichever ones were not quick
        enough -- and it loses them exactly when Redis is slow, which is when
        a deploy is most likely to be happening.

        The delayed client is what makes this assertable: without it the
        write lands during some later await inside ``aclose`` itself, and a
        shutdown that waits for nothing passes.
        """
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        slow = _DelaysEveryWrite(_client(), delay=0.3)
        self.addAsyncCleanup(slow.aclose)
        slow_log = TurnEventLog(client=slow)
        registry = TurnRegistry(slow_log, url=REDIS_URL)
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await registry.begin(
            self.thread_id,
            blocks_until(gate, frame("done", "{}"), before=frame("status", "{}")),
        )
        await _until(lambda: slow_log.read(claim.turn_id), lambda page: page.frames)

        await registry.aclose()

        self.assertEqual(self.terminal_now(claim.turn_id), "interrupted")

    async def test_a_turn_that_already_answered_is_not_relabelled_by_a_shutdown(self):
        """The hazard a check would be the wrong fix for.

        ``terminal_of`` reads the last entry, so an ``interrupted`` written
        on top of a finished turn turns a delivered answer into a failure --
        and the window is real, because a reader can still be collecting that
        answer when the replica is told to go down. Guarded structurally, the
        way the stop path guards the same thing: only a turn that still has a
        consumer to cancel is marked, and a turn that has answered has none.
        """
        claim = await self.registry.begin(
            self.thread_id, produces(frame("done", '{"response": "here it is"}'))
        )
        await self.registry.wait(claim.turn_id)

        await self.registry.aclose()

        self.assertEqual(self.terminal_now(claim.turn_id), "done")

    async def test_a_shutdown_hands_back_the_threads_it_was_holding(self):
        """A replica that dies owing a claim locks the thread out.

        The claim outlives the process -- it is in Redis, with the whole-turn
        deadline as its TTL -- so a thread whose claim is not released reads
        as broken to the user for half an hour after a deploy that took two
        seconds.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        gate = asyncio.Event()
        self.addCleanup(gate.set)
        await self.registry.begin(
            self.thread_id,
            blocks_until(gate, frame("done", "{}"), before=frame("status", "{}")),
        )
        await _until(
            lambda: self.registry.turn_for(self.thread_id), lambda turn: turn is not None
        )

        await self.registry.aclose()

        after = TurnRegistry(self.log, url=REDIS_URL)
        self.addAsyncCleanup(after.aclose)
        claim = await after.begin(self.thread_id, produces(frame("done", "{}")))
        await after.wait(claim.turn_id)
        self.assertEqual(claim.outcome, "started")

    async def test_a_reader_attached_to_an_interrupted_turn_is_released(self):
        """The drain has to reach the reader, not just the log.

        An attached reader is the one the deploy is visible to, and it has no
        other way to learn: an interrupted turn emits no ``done`` of its own,
        so without this the stream goes quiet and the bubble spins until the
        stale window closes thirty seconds later.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id,
            blocks_until(gate, frame("done", "{}"), before=frame("status", "{}")),
        )
        reader = asyncio.create_task(_collect(self.registry.follow(claim.turn_id)))
        await _until(lambda: self.log.read(claim.turn_id), lambda page: page.frames)

        await self.registry.aclose()

        delivered = await asyncio.wait_for(reader, timeout=5.0)
        self.assertTrue(
            any(item.startswith("event: interrupted") for item in delivered),
            f"the reader was never released; it got {delivered}",
        )

    async def test_an_interrupted_turn_says_which_kind_of_interruption_it_was(self):
        """Two things end a turn with nobody asking, and they read differently.

        A drained replica knows the answer is gone and is not coming back; a
        reader that found the stream quiet only knows it cannot see the turn
        any more. Same event, because the offer to the user is the same
        retry, but a reader cannot tell a restart from a lost connection
        without being told, and the two are not the same sentence.
        """
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        claim = await self.registry.begin(
            self.thread_id,
            blocks_until(gate, frame("done", "{}"), before=frame("status", "{}")),
        )
        await _until(lambda: self.log.read(claim.turn_id), lambda page: page.frames)

        await self.registry.aclose()

        page = await self.log.read(claim.turn_id)
        ending = json.loads(page.frames[-1].split("data: ", 1)[1])
        self.assertEqual(ending["reason"], "shutdown")

    async def test_a_replica_that_is_going_down_refuses_to_start_a_turn(self):
        """A send landing mid-drain must not be started into a dying process.

        The drain has already been past the turns it is going to mark, so one
        accepted after it is a turn nobody will interrupt and nobody will
        finish: it takes the thread's claim with it and the reader sees a
        stream that simply stops. Refusing is what lets the load balancer
        send the retry to a replica that is staying up.
        """
        from tta_backend.services.turn_registry import RegistryClosing

        await self.registry.drain()

        with self.assertRaises(RegistryClosing):
            await self.registry.begin(self.thread_id, produces(frame("done", "{}")))

    async def test_a_turn_refused_by_a_shutdown_leaves_the_thread_free(self):
        """The refusal must not bank the claim it did not use.

        A claim written and never released outlives the process that wrote
        it, so the replica that stays up would refuse every message on that
        thread until the whole-turn TTL ran out.
        """
        from tta_backend.services.turn_registry import RegistryClosing

        await self.registry.drain()

        with self.assertRaises(RegistryClosing):
            await self.registry.begin(self.thread_id, produces(frame("done", "{}")))

        self.assertEqual(await self.ttl(f"thread:{self.thread_id}:active_turn"), -2)

    async def test_a_reader_is_told_when_the_turn_it_watches_has_gone_quiet(self):
        """D14: the tracer bullet for a replica that died without saying so.

        Nothing writes a terminal entry when a process is killed rather than
        drained -- an OOM, a SIGKILL after the stop timeout, a lost node --
        so the stream simply stops and every reader on it polls a turn that
        will never move again. The heartbeat is what makes that detectable:
        a live turn writes at least every ten seconds, so silence past a
        multiple of that, with no terminal entry, is a dead owner.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        reader = TurnRegistry(self.log, url=REDIS_URL, stale_after=0.4)
        self.addAsyncCleanup(reader.aclose)
        turn_id = f"test-turn-{uuid.uuid4()}"
        await self.log.append(turn_id, frame("status", '{"message": "Searching"}'))

        delivered = await asyncio.wait_for(
            _collect(reader.follow(turn_id)), timeout=5.0
        )

        self.assertTrue(
            any(item.startswith("event: interrupted") for item in delivered),
            f"the reader was never told; it got {delivered}",
        )

    async def test_a_turn_that_has_not_written_yet_is_not_already_stale(self):
        """The resume cursor a reader starts from is dated 1970.

        ``START_CURSOR`` is ``0-0``, so measuring silence from the cursor
        ``follow`` carries declares every turn decades dead before it writes
        its first frame -- and a reader attaching the moment the 202 comes
        back is the ordinary case, not a corner. The turn's own last entry is
        the only thing that dates the silence, and until there is one the
        reader can only date it from when it arrived.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        reader = TurnRegistry(self.log, url=REDIS_URL, stale_after=0.6)
        self.addAsyncCleanup(reader.aclose)
        turn_id = f"test-turn-{uuid.uuid4()}"

        following = asyncio.create_task(_collect(reader.follow(turn_id)))
        await asyncio.sleep(0.2)
        await self.log.mark_terminal(turn_id, "done", frame("done", '{"response": "hi"}'))
        delivered = await asyncio.wait_for(following, timeout=5.0)

        self.assertFalse(
            any(item.startswith("event: interrupted") for item in delivered),
            f"a turn that answered was called dead; it got {delivered}",
        )

    async def test_a_reader_resuming_onto_a_turn_that_died_is_told_at_once(self):
        """The silence started before this reader did.

        This is the remount case: a tab comes back with a stored cursor onto
        a turn whose replica died during the deploy that restarted it.
        Counting the silence from when *this* reader attached makes it wait a
        further full window to be told what the stream's own timestamps
        already say, and a reader arriving an hour later waits exactly as
        long as one arriving a second later.
        """
        from tta_backend.services.turn_registry import TurnRegistry

        reader = TurnRegistry(self.log, url=REDIS_URL, stale_after=1.0)
        self.addAsyncCleanup(reader.aclose)
        turn_id = f"test-turn-{uuid.uuid4()}"
        await self.log.append(turn_id, frame("status", "{}"))
        resume_from = (await self.log.read(turn_id)).cursor
        await asyncio.sleep(1.2)

        started = asyncio.get_running_loop().time()
        delivered = await asyncio.wait_for(
            _collect(reader.follow(turn_id, resume_from)), timeout=5.0
        )
        waited = asyncio.get_running_loop().time() - started

        self.assertTrue(any(item.startswith("event: interrupted") for item in delivered))
        self.assertLess(
            waited, 0.5, "the reader served out a second silence it had already missed"
        )


async def _collect(stream) -> list[str]:
    """Everything a follower yields before it lets go."""
    return [item async for item in stream]


def _client():
    """A real Redis client on the test database."""
    from redis import asyncio as aioredis

    return aioredis.from_url(REDIS_URL, decode_responses=True)


class _DelaysEveryWrite:
    """A real Redis client whose every pipeline execution takes its time.

    Wraps rather than fakes, so each command still reaches Redis and behaves
    as Redis does. The only thing added is a window wide enough that a caller
    which does not wait for the write can be told apart from one that does.
    """

    def __init__(self, inner, delay: float):
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def pipeline(self, *args, **kwargs):
        return _DelayedPipeline(self._inner.pipeline(*args, **kwargs), self._delay)


class _DelayedPipeline:
    def __init__(self, inner, delay: float):
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def execute(self, *args, **kwargs):
        await asyncio.sleep(self._delay)
        return await self._inner.execute(*args, **kwargs)
