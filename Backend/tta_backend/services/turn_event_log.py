"""T63 — the chat turn event log.

A turn's narration, stored where any replica can read it, so that a turn
outlives the connection that started it.

What travels through here is the *rendered SSE frame*, verbatim, exactly as
``ChatStreamService.sse()`` produced it. Replay is therefore "write these
bytes", which is byte-identical to the live path by construction rather than a
parallel re-rendering that can drift apart from it.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from redis import asyncio as aioredis

from tta_backend.config.settings import get_settings

logger = logging.getLogger(__name__)


#: Where a reader that has seen nothing starts. Redis entry IDs are
#: ``<millis>-<seq>`` and sort above this, so it needs no special-casing.
START_CURSOR = "0-0"

#: How long an answer token may sit buffered waiting for the ones after it.
#: Invisible to the reader — the frontend already coalesces streamed text
#: through its own flush timer — and it is the difference between one XADD
#: per token and one per batch on a 2,000-token answer.
DEFAULT_FLUSH_INTERVAL_SECONDS = 0.05

#: How many entries a turn's log keeps. A guess, not a measurement — size it
#: against p99 events-per-turn once there is live traffic to measure. Losing
#: the oldest entries is acceptable by design (D8): a reader away long enough
#: to be trimmed loses narration, and HistoryService still holds the answer.
DEFAULT_MAX_ENTRIES = 4000

#: Added to the whole-turn deadline to get a log's lifetime. The margin covers
#: the gap between a turn ending and the last reader collecting its terminal
#: entry; the deadline itself is what guarantees the log cannot expire under a
#: turn still producing into it.
TTL_MARGIN_SECONDS = 600


@dataclass(frozen=True)
class TurnEventPage:
    """What a reader got, and where it should resume from.

    ``cursor`` is an opaque resume point, not a count: an empty page returns
    the cursor it was given rather than rewinding, so polling an idle turn
    does not replay it.
    """

    frames: list[str]
    cursor: str
    #: ``done``/``stopped``/``interrupted`` once the turn ended, else None.
    #: ``None`` on a stream that has gone quiet is what D14 reads as "the
    #: replica that owned this turn died".
    terminal: str | None = None


class TurnEventLog:
    def __init__(
        self,
        url: str | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: int | None = None,
        client: Any = None,
    ):
        """Opens its own connection pool from ``url``, or borrows ``client``.

        A caller that passes a client keeps ownership of it: several logs can
        then share one pool instead of each opening its own, and a test can
        hand in a wrapper to control when a write lands.
        """
        if client is None and url is None:
            raise ValueError("TurnEventLog needs a url or a client")
        self._owns_client = client is None
        self._redis = client if client is not None else aioredis.from_url(
            url, decode_responses=True
        )
        self._flush_interval = flush_interval
        self._max_entries = max_entries
        # Read from settings rather than baked in, so the two numbers cannot
        # drift into a log that expires under a running turn.
        self._ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else int(get_settings().chat_turn_timeout_seconds) + TTL_MARGIN_SECONDS
        )
        self._buffered_text: dict[str, list[str]] = {}
        self._flush_timers: dict[str, asyncio.Task] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}

    async def aclose(self) -> None:
        for turn_id in list(self._buffered_text):
            await self.flush(turn_id)
        for timer in self._flush_timers.values():
            timer.cancel()
        self._flush_timers.clear()
        if self._owns_client:
            await self._redis.aclose()

    def _key(self, turn_id: str) -> str:
        return f"turn:{turn_id}:events"

    def _lock_for(self, turn_id: str) -> asyncio.Lock:
        """Serializes everything that writes on behalf of one turn.

        Per turn, not global: two turns have no ordering relationship and must
        not wait on each other. Entries are dropped when the turn ends; a turn
        that is interrupted instead leaks one lock, which is a few hundred
        bytes against a process that restarts on deploy.
        """
        lock = self._write_locks.get(turn_id)
        if lock is None:
            lock = self._write_locks[turn_id] = asyncio.Lock()
        return lock

    async def append(self, turn_id: str, frame: str) -> None:
        """Write a structural frame — status, chart, job_progress — at once.

        Flushes buffered text first. Without that the frame would overtake
        text produced before it, and because replay reads the log's order,
        that inversion would be permanent rather than a live-only artifact.
        """
        async with self._lock_for(turn_id):
            await self._flush_holding_lock(turn_id)
            await self._write(turn_id, [frame])

    async def append_text(self, turn_id: str, frame: str) -> None:
        """Buffer an answer token, to be written with the ones around it.

        The timer started here is what bounds how long a token can sit
        unwritten: an answer that streams only text has no structural frame to
        push the buffer out, and its own terminal entry may be minutes away.
        """
        self._buffered_text.setdefault(turn_id, []).append(frame)
        if turn_id not in self._flush_timers:
            self._flush_timers[turn_id] = asyncio.create_task(self._flush_later(turn_id))

    async def _flush_later(self, turn_id: str) -> None:
        await asyncio.sleep(self._flush_interval)
        try:
            await self.flush(turn_id)
        except Exception:
            # This runs as its own task, so there is no caller to raise to and
            # the buffered tokens are already gone. Logged with the turn it
            # belonged to; surfacing the failure into the turn itself is the
            # job of whatever owns the turn, not of the timer.
            logger.warning(
                "turn_event_flush_failed",
                exc_info=True,
                extra={"_event": "turn_event_flush_failed", "_turn_id": turn_id},
            )

    async def flush(self, turn_id: str) -> None:
        """Write whatever text is buffered for this turn, if any."""
        async with self._lock_for(turn_id):
            await self._flush_holding_lock(turn_id)

    async def _flush_holding_lock(self, turn_id: str) -> None:
        timer = self._flush_timers.pop(turn_id, None)
        # A timer firing reaches here as its own task; cancelling it would
        # abort this call before the write.
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        buffered = self._buffered_text.pop(turn_id, None)
        if buffered:
            await self._write(turn_id, buffered)

    async def mark_terminal(self, turn_id: str, kind: str, frame: str) -> None:
        """End the turn, delivering its closing frame in the same entry.

        One entry rather than two so a reader can never observe the closing
        event while still believing the turn is live. Buffered text is flushed
        first: the tail of an answer is usually still in the buffer when the
        turn ends, and losing it truncates the replayed answer permanently.
        """
        async with self._lock_for(turn_id):
            await self._flush_holding_lock(turn_id)
            await self._write(turn_id, [frame], terminal=kind)
        # The turn is over; nothing may write under this id again.
        self._write_locks.pop(turn_id, None)

    async def _write(
        self, turn_id: str, frames: list[str], terminal: str | None = None
    ) -> None:
        """One entry, carrying one or more frames.

        Frames occupy numbered fields rather than a single concatenated value:
        a reader then rebuilds the exact list that was written, with no
        splitting on ``\\n\\n`` to re-derive boundaries that were already
        known. Numbered rather than positional because Redis field order is
        not something to rely on for correctness.
        """
        fields: dict[str, str] = {f"f{i}": frame for i, frame in enumerate(frames)}
        if terminal is not None:
            fields["terminal"] = terminal
        key = self._key(turn_id)
        # Pipelined so bounding the log costs no extra round trip.
        pipe = self._redis.pipeline(transaction=False)
        # Exact trimming rather than MAXLEN ~. Approximate trimming keeps an
        # unspecified number of extra entries, which turns the per-turn memory
        # ceiling into an estimate and leaves nothing assertable. The write
        # rate that would justify `~` is far above one turn's event rate.
        pipe.xadd(key, fields, maxlen=self._max_entries, approximate=False)
        # Refreshed on every write, so the lifetime is measured from the
        # turn's last activity rather than from its first event.
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def read(self, turn_id: str, cursor: str | None = None) -> TurnEventPage:
        """Everything written after ``cursor``, plus where to resume next.

        The cursor is exclusive: a reader that hands back what it last saw is
        not sent it a second time.
        """
        resume_from = cursor or START_CURSOR
        entries = (
            await self._redis.xrange(self._key(turn_id), min=f"({resume_from}", max="+")
        ) or []
        frames: list[str] = []
        terminal: str | None = None
        for _entry_id, entry_fields in entries:
            fields = entry_fields or {}
            frames.extend(_frames_of(fields))
            terminal = terminal or fields.get("terminal")
        return TurnEventPage(
            frames=frames,
            cursor=str(entries[-1][0]) if entries else resume_from,
            terminal=terminal,
        )


def _frames_of(fields: dict[str, str]) -> list[str]:
    numbered = sorted(
        ((int(name[1:]), value) for name, value in fields.items() if name.startswith("f")),
        key=lambda pair: pair[0],
    )
    return [value for _index, value in numbered]
