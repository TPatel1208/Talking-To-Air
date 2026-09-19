"""The chat turn event log.

A turn appends its rendered SSE frames here; a reader streams them out and,
after switching sessions or sleeping a laptop, resumes from a cursor instead
of losing the turn.

Run against a **real** Redis, because most of what they pin is Redis's own
behaviour — where a cursor lands, that a trimmed stream still reads, that two
readers do not consume each other's entries. They skip when none is reachable
so a host-side ``pytest`` still runs; ``test_deployment_contract.py`` asserts
the test profile declares the dependency, so the container run cannot skip
them silently.

Nothing here flushes: each test owns its ``turn_id`` and lets ``EXPIRE`` clean
up, so a misconfigured ``REDIS_URL`` cannot destroy anything.
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

#: Database 15 by default, never 0 — the live stack's event log is on 0.
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


@requires_redis
class TurnEventLogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tta_backend.services.turn_event_log import TurnEventLog

        self.log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(self.log.aclose)
        self.turn_id = f"test-{uuid.uuid4()}"

    async def inspect(self, command: str):
        """Ask Redis directly about this turn's stream.

        Through an independent client, so the answer is the store's state and
        not the log's bookkeeping. Both facts needed here — the ``XADD`` count
        and the key's TTL — are invisible to the reading interface by design.
        """
        from redis import asyncio as aioredis

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            return await getattr(client, command)(f"turn:{self.turn_id}:events")
        finally:
            await client.aclose()

    def text_frame(self, token: str) -> str:
        return f"event: text\ndata: {json.dumps(token)}\n\n"

    async def test_a_frame_that_was_appended_can_be_read_back(self):
        frame = 'event: status\ndata: {"message": "Retrieving granules"}\n\n'

        await self.log.append(self.turn_id, frame)

        page = await self.log.read(self.turn_id)
        self.assertEqual([frame], page.frames)

    async def test_a_reader_resuming_from_a_cursor_gets_only_what_it_missed(self):
        """The whole point of the log: a reader that went away and came back
        collects the frames produced while it was gone, and not the ones it
        already rendered."""
        already_seen = 'event: status\ndata: {"message": "Opening granules"}\n\n'
        missed = 'event: status\ndata: {"message": "Reducing"}\n\n'
        await self.log.append(self.turn_id, already_seen)

        first_visit = await self.log.read(self.turn_id)
        await self.log.append(self.turn_id, missed)

        resumed = await self.log.read(self.turn_id, cursor=first_visit.cursor)
        self.assertEqual([missed], resumed.frames)

    async def test_two_readers_each_get_every_frame(self):
        """Two tabs on one thread, each with its own cursor. This is what
        ruled out a list: ``BRPOP`` is destructive, so one tab would consume
        frames the other needed and each would render half a turn."""
        first_frame = 'event: text\ndata: "Ozone over "\n\n'
        second_frame = 'event: text\ndata: "New Jersey"\n\n'
        await self.log.append(self.turn_id, first_frame)
        await self.log.append(self.turn_id, second_frame)

        one_tab = await self.log.read(self.turn_id)
        other_tab = await self.log.read(self.turn_id)

        self.assertEqual([first_frame, second_frame], one_tab.frames)
        self.assertEqual([first_frame, second_frame], other_tab.frames)

    async def test_a_frame_comes_back_byte_identical(self):
        """The log stores the rendered frame, not a parsed event, so replay is
        byte-identical to the live path by construction.

        Units are the realistic hazard — this domain's answers are full of
        ``µg/m³`` and ``°`` — and so is the ``\\n`` that JSON escapes inside a
        payload but SSE treats as structure.
        """
        frame = (
            'event: chart\ndata: {"units": "\\u00b5g/m\\u00b3", '
            '"title": "PM2.5 \\u2014 line 1\\nline 2", "note": "caf\\u00e9"}\n\n'
        )

        await self.log.append(self.turn_id, frame)

        page = await self.log.read(self.turn_id)
        self.assertEqual([frame], page.frames)
        self.assertEqual(frame, page.frames[0])

    async def test_a_terminal_entry_delivers_its_frame_and_ends_the_turn(self):
        """A reader tells "finished" from "the replica running it died" by
        whether a terminal entry arrived. Frame and marker travel as one
        entry, so it cannot see ``done`` and still believe the turn is live.
        """
        done = 'event: done\ndata: {"tool_calls": []}\n\n'
        while_running = await self.log.read(self.turn_id)
        self.assertIsNone(while_running.terminal)

        await self.log.mark_terminal(self.turn_id, "done", done)

        page = await self.log.read(self.turn_id, cursor=while_running.cursor)
        self.assertEqual([done], page.frames)
        self.assertEqual("done", page.terminal)

    async def test_text_frames_coalesce_into_one_entry(self):
        """A 2,000-token answer is otherwise 2,000 ``XADD``s. Every frame still
        has to survive: batching is a write-side economy, never a reduction in
        what the reader receives."""
        tokens = ["Ozone ", "over ", "New Jersey"]
        for token in tokens:
            await self.log.append_text(self.turn_id, self.text_frame(token))

        await self.log.flush(self.turn_id)

        page = await self.log.read(self.turn_id)
        self.assertEqual([self.text_frame(t) for t in tokens], page.frames)
        self.assertEqual(1, await self.inspect("xlen"))

    async def test_a_structural_frame_does_not_jump_ahead_of_buffered_text(self):
        """Writing structural frames straight through while text is buffered
        lets a chart land before the sentence introducing it, and replay
        carries that order forever. Flushing first makes the log's order the
        order the turn produced.
        """
        sentence = self.text_frame("Here is the chart: ")
        chart = 'event: chart\ndata: {"chart_id": "c1"}\n\n'

        await self.log.append_text(self.turn_id, sentence)
        await self.log.append(self.turn_id, chart)

        page = await self.log.read(self.turn_id)
        self.assertEqual([sentence, chart], page.frames)

    async def test_the_last_tokens_of_an_answer_survive_the_terminal_entry(self):
        """Text still buffered when the turn ends is never written, so a
        reader replaying it gets a sentence stopping mid-clause followed by
        ``done`` — and the log, being the record, agrees with the truncation.
        """
        last_words = self.text_frame("\u2026and rising.")
        done = 'event: done\ndata: {"tool_calls": []}\n\n'

        await self.log.append_text(self.turn_id, last_words)
        await self.log.mark_terminal(self.turn_id, "done", done)

        page = await self.log.read(self.turn_id)
        self.assertEqual([last_words, done], page.frames)
        self.assertEqual("done", page.terminal)

    async def test_buffered_text_reaches_the_log_without_another_event(self):
        """An answer of nothing but tokens has no structural frame to push the
        buffer out, and the turn's end can be 300s away. The interval is the
        only thing that makes a live reader see text at all.
        """
        from tta_backend.services.turn_event_log import TurnEventLog

        log = TurnEventLog(REDIS_URL, flush_interval=0.02)
        self.addAsyncCleanup(log.aclose)
        frame = self.text_frame("Ozone concentrations ")

        await log.append_text(self.turn_id, frame)
        await asyncio.sleep(0.2)

        page = await log.read(self.turn_id)
        self.assertEqual([frame], page.frames)

    async def test_the_stream_stops_growing_once_it_reaches_its_bound(self):
        """The sizing — ~400 KB a turn, 100 turns inside a 0.5 GiB Redis —
        holds only if the bound is real. The cost is acceptable: what falls
        off the back is narration, and HistoryService holds the answer.
        """
        from tta_backend.services.turn_event_log import TurnEventLog

        log = TurnEventLog(REDIS_URL, max_entries=5)
        self.addAsyncCleanup(log.aclose)
        frames = [f'event: status\ndata: {{"n": {n}}}\n\n' for n in range(20)]
        for frame in frames:
            await log.append(self.turn_id, frame)

        self.assertEqual(5, await self.inspect("xlen"))
        page = await log.read(self.turn_id)
        self.assertEqual(frames[-5:], page.frames)

    async def test_a_turns_log_outlives_the_longest_turn_and_then_expires(self):
        """Two failures in one bound. Too short and a turn's own log expires
        underneath it while it is still running; absent and every abandoned
        turn's narration stays resident forever.
        """
        from tta_backend.config.settings import get_settings

        await self.log.append(self.turn_id, 'event: status\ndata: {}\n\n')

        ttl = await self.inspect("ttl")

        self.assertGreater(ttl, 0, "the log never expires")
        self.assertGreater(
            ttl,
            get_settings().chat_turn_timeout_seconds,
            "the log can expire while the turn that owns it is still running",
        )

    async def test_a_frame_cannot_overtake_a_timer_flush_already_in_flight(self):
        """Ordering can also break through the timer rather than the buffer.

        ``flush`` takes the buffered text and then awaits its write, so a
        flush running as the timer's own task can be suspended there while the
        producer appends a structural frame, whose own flush finds an empty
        buffer and goes straight out. Measured 7 inversions in 25 under
        ``asyncio.run`` before the fix.

        The delayed client is required, not incidental: on a plain client this
        sequence reproduces under ``asyncio.run`` but never under this
        runner's event loop, so the test would pass either way. Delaying only
        the *first* write keeps the flush in flight when the append starts —
        unserialized the chart always wins, serialized it never does.
        """
        from redis import asyncio as aioredis

        from tta_backend.services.turn_event_log import TurnEventLog

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.addAsyncCleanup(client.aclose)
        log = TurnEventLog(client=_DelaysItsFirstWrite(client, 0.2), flush_interval=0.001)
        self.addAsyncCleanup(log.aclose)
        sentence = self.text_frame("Here is the chart: ")
        chart = 'event: chart\ndata: {"chart_id": "c1"}\n\n'

        await log.append_text(self.turn_id, sentence)
        # Long enough for the timer to fire and enter its write, far short of
        # the delay that write is now sitting in.
        await asyncio.sleep(0.05)
        await log.append(self.turn_id, chart)

        page = await log.read(self.turn_id)
        self.assertEqual(
            [sentence, chart], page.frames,
            "a structural frame overtook a flush that was already in flight",
        )

    async def test_a_flush_that_fails_in_the_background_is_recorded(self):
        """The timer flush runs as its own task, so a write that raises there
        has no caller to surface to. Without this the tokens are lost and the
        only trace is an unretrieved-task warning naming no turn."""
        from tta_backend.services.turn_event_log import TurnEventLog

        log = TurnEventLog(client=_RefusesEveryWrite(), flush_interval=0.01)
        self.addAsyncCleanup(log.aclose)

        with self.assertLogs("tta_backend.services.turn_event_log", level="WARNING") as captured:
            await log.append_text(self.turn_id, self.text_frame("lost tokens"))
            await asyncio.sleep(0.1)

        failures = [r for r in captured.records if r.getMessage() == "turn_event_flush_failed"]
        self.assertTrue(failures, f"the failed flush was not recorded: {captured.output}")
        self.assertEqual(
            self.turn_id, getattr(failures[0], "_turn_id", None),
            "the record does not name the turn whose tokens were lost",
        )
        self.assertIsNotNone(failures[0].exc_info, "the cause was not recorded")


class _RefusesEveryWrite:
    """Stands in for a Redis that has gone away mid-turn."""

    def pipeline(self, *args, **kwargs):
        return self

    def xadd(self, *args, **kwargs):
        return self

    def expire(self, *args, **kwargs):
        return self

    async def execute(self, *args, **kwargs):
        raise ConnectionError("redis is gone")

    async def aclose(self):
        return None


class _DelaysItsFirstWrite:
    """A real Redis client whose first pipeline execution takes its time.

    Wraps rather than fakes: every command still reaches Redis and behaves as
    Redis does. The only thing added is a window, so "the append waited for
    the flush" is something the test can require instead of hope for.
    """

    def __init__(self, inner, delay: float):
        self._inner = inner
        self._delay = delay
        self._delayed_one = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def pipeline(self, *args, **kwargs):
        delay = 0.0 if self._delayed_one else self._delay
        self._delayed_one = True
        return _DelayedPipeline(self._inner.pipeline(*args, **kwargs), delay)


class _DelayedPipeline:
    def __init__(self, inner, delay: float):
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def execute(self, *args, **kwargs):
        if self._delay:
            await asyncio.sleep(self._delay)
        return await self._inner.execute(*args, **kwargs)
