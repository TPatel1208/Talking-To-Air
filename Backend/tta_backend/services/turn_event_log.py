"""The chat turn event log: a turn's narration, readable from any replica.

Entries hold the rendered SSE frame exactly as ``ChatStreamService.sse()``
produced it, so replay writes those bytes back and cannot drift from the live
path.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from redis import asyncio as aioredis

from tta_backend.config.settings import get_settings

logger = logging.getLogger(__name__)


#: Where a reader that has seen nothing starts. Entry IDs are
#: ``<millis>-<seq>`` and sort above this, so it needs no special-casing.
START_CURSOR = "0-0"

#: How long an answer token may sit buffered waiting for the ones after it.
#: One XADD per batch rather than per token on a long answer.
DEFAULT_FLUSH_INTERVAL_SECONDS = 0.05

#: How many entries a turn's log keeps. A guess — size it against p99
#: events-per-turn once there is traffic to measure. A reader away long enough
#: to be trimmed loses narration; HistoryService still holds the answer.
DEFAULT_MAX_ENTRIES = 4000

#: Added to the whole-turn deadline to get a log's lifetime, covering the gap
#: between a turn ending and the last reader collecting its terminal entry.
TTL_MARGIN_SECONDS = 600


@dataclass(frozen=True)
class TurnEventPage:
    """What a reader got, and where it should resume from."""

    frames: list[str]
    #: An opaque resume point, not a count. An empty page returns the cursor it
    #: was given rather than rewinding, so polling an idle turn does not replay
    #: it.
    cursor: str
    #: ``done``/``stopped``/``interrupted`` once the turn ended, else None.
    #: None on a stream that has gone quiet means the replica that owned the
    #: turn died.
    terminal: str | None = None


@dataclass(frozen=True)
class TurnTail:
    """The last thing written under a turn id, and when."""

    #: Milliseconds since the epoch, off the entry's own id — Redis's clock,
    #: assigned at the write. None when nothing has been written yet.
    written_ms: int | None
    #: How the turn ended, or None while it is still going.
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

        A caller passing a client keeps ownership of it, so several logs can
        share one pool.
        """
        if client is None and url is None:
            raise ValueError("TurnEventLog needs a url or a client")
        self._owns_client = client is None
        self._redis = client if client is not None else aioredis.from_url(
            url, decode_responses=True
        )
        self._flush_interval = flush_interval
        self._max_entries = max_entries
        # From settings, not baked in: a log must not expire under a turn
        # still producing into it.
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

        Per turn, not global: two turns have no ordering relationship. Dropped
        when the turn ends, so only an interrupted turn leaks one.
        """
        lock = self._write_locks.get(turn_id)
        if lock is None:
            lock = self._write_locks[turn_id] = asyncio.Lock()
        return lock

    async def append(self, turn_id: str, frame: str) -> None:
        """Write a structural frame — status, chart, job_progress — at once.

        Flushes buffered text first, or the frame overtakes text produced
        before it and replay carries that inversion permanently.
        """
        async with self._lock_for(turn_id):
            await self._flush_holding_lock(turn_id)
            await self._write(turn_id, [frame])

    async def append_text(self, turn_id: str, frame: str) -> None:
        """Buffer an answer token, to be written with the ones around it.

        The timer bounds how long a token sits unwritten: an answer that
        streams only text has no structural frame to push the buffer out.
        """
        self._buffered_text.setdefault(turn_id, []).append(frame)
        if turn_id not in self._flush_timers:
            self._flush_timers[turn_id] = asyncio.create_task(self._flush_later(turn_id))

    async def _flush_later(self, turn_id: str) -> None:
        await asyncio.sleep(self._flush_interval)
        try:
            await self.flush(turn_id)
        except Exception:
            # Runs as its own task, so there is no caller to raise to and the
            # buffered tokens are already gone. Surfacing this into the turn
            # belongs to whatever owns the turn, not to the timer.
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

        One entry rather than two, so a reader cannot see the closing event
        and still believe the turn is live. Flushes first: the tail of an
        answer is usually still buffered, and losing it truncates the replay.
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

        Numbered fields rather than one concatenated value, so a reader
        rebuilds the exact list without splitting on ``\\n\\n`` to re-derive
        boundaries already known — and numbered rather than positional,
        because field order is not worth relying on.
        """
        fields: dict[str, str] = {f"f{i}": frame for i, frame in enumerate(frames)}
        if terminal is not None:
            fields["terminal"] = terminal
        key = self._key(turn_id)
        # Pipelined so bounding the log costs no extra round trip.
        pipe = self._redis.pipeline(transaction=False)
        # Exact, not MAXLEN ~: approximate trimming keeps an unspecified
        # number of extra entries, turning the per-turn memory ceiling into an
        # estimate. One turn's write rate does not justify `~`.
        pipe.xadd(key, fields, maxlen=self._max_entries, approximate=False)
        # Refreshed per write, so lifetime runs from the turn's last activity.
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def terminal_of(self, turn_id: str) -> str | None:
        """How this turn ended, or None while it is still going.

        ``read`` reports a terminal only when the entry carrying it falls
        inside the page it returned, so a reader resuming from a cursor at or
        past the end never learns the turn is over. Read off the last entry,
        which is what the terminal entry always is — nothing writes under a
        turn id once it is marked.
        """
        return (await self.tail(turn_id)).terminal

    async def tail(self, turn_id: str) -> TurnTail:
        """The last entry's timestamp and terminal, in one round trip.

        Both answers come off the same entry, and the caller that wants one
        usually wants the other: "has it ended, and if not, when did it last
        say anything" is the whole of liveness (D14). Asking separately would
        double the cost of the one path that polls.
        """
        entries = await self._redis.xrevrange(self._key(turn_id), max="+", min="-", count=1)
        if not entries:
            return TurnTail(written_ms=None)
        entry_id, fields = entries[0]
        return TurnTail(
            written_ms=_millis_of(str(entry_id)),
            terminal=(fields or {}).get("terminal"),
        )

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


def _millis_of(entry_id: str) -> int | None:
    """When Redis wrote this entry, off the ``<millis>-<seq>`` id itself.

    No second key and no heartbeat table: the stream already carries the one
    timestamp liveness needs, stamped by the server that stored it.
    """
    try:
        return int(entry_id.split("-", 1)[0])
    except ValueError:
        return None


def _frames_of(fields: dict[str, str]) -> list[str]:
    numbered = sorted(
        ((int(name[1:]), value) for name, value in fields.items() if name.startswith("f")),
        key=lambda pair: pair[0],
    )
    return [value for _index, value in numbered]
