"""Who owns a running chat turn.

A turn is started by the request that posted the message and outlives it: the
registry runs it as its own task and feeds every frame it produces into the
event log, where any replica's reader can pick it up.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from redis import asyncio as aioredis

from tta_backend.config.settings import get_settings
from tta_backend.services.turn_event_log import (
    START_CURSOR,
    TTL_MARGIN_SECONDS,
    TurnEventLog,
)

#: Events that can be the last thing a turn says. A turn timing out or
#: shedding emits ``error`` then ``done``; the generic failure path emits
#: only ``error``, so neither name alone identifies the close.
_CLOSING_EVENTS = frozenset({"done", "error"})

#: How long a follower waits before asking the log for more. The text
#: batcher already holds tokens ~50ms, so this is the second half of a
#: delay measured in tenths of a second against turns that run minutes.
DEFAULT_POLL_INTERVAL_SECONDS = 0.1


#: This caller's message started the turn it is being handed.
STARTED = "started"
#: The thread already had a turn in flight; the id is that one's.
ALREADY_RUNNING = "already_running"
#: This exact send was already accepted; the id is what it got the first time.
DUPLICATE = "duplicate"


@dataclass(frozen=True)
class TurnClaim:
    """The turn a caller ended up attached to, and how it got there."""

    turn_id: str
    outcome: str = STARTED


class TurnRegistry:
    def __init__(
        self,
        log: TurnEventLog,
        url: str | None = None,
        client: Any = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        """Opens its own connection pool from ``url``, or borrows ``client``.

        A caller passing a client keeps ownership of it, so the registry and
        the event log can share one pool.
        """
        if client is None and url is None:
            raise ValueError("TurnRegistry needs a url or a client")
        self._log = log
        self._owns_client = client is None
        self._redis = client if client is not None else aioredis.from_url(
            url, decode_responses=True
        )
        self._tasks: dict[str, asyncio.Task] = {}
        self._poll_interval = poll_interval
        # An idempotency record has to outlive the turn it names, or a
        # retry arriving late re-buys the answer it already has.
        self._ttl_seconds = int(get_settings().chat_turn_timeout_seconds) + TTL_MARGIN_SECONDS

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        if self._owns_client:
            await self._redis.aclose()

    def _active_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:active_turn"

    def _last_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:last_turn"

    def _idempotency_key(self, key: str) -> str:
        return f"send:{key}:turn"

    async def begin(
        self,
        thread_id: str,
        frames: AsyncIterator[str],
        idempotency_key: str | None = None,
    ) -> TurnClaim:
        """Start a turn on this thread, unless one is already running.

        The claim is in Redis rather than in this process because the second
        tab's message can land on a different replica, which is the case
        sticky sessions fail silently.
        """
        turn_id = str(uuid.uuid4())
        sent = None if idempotency_key is None else self._idempotency_key(idempotency_key)
        if sent is not None:
            # Claimed before the thread is — it has to be, or two retries race
            # past each other — so a retry is answered with the turn it
            # already bought even after that turn finished and gave the thread
            # back.
            if not await self._redis.set(sent, turn_id, nx=True, ex=self._ttl_seconds):
                already = await self._redis.get(sent)
                return TurnClaim(turn_id=str(already), outcome=DUPLICATE)
        key = self._active_key(thread_id)
        # Expiring, because the turn that would release it may die with its
        # replica. Bounded by the whole-turn deadline: a claim that outlives
        # the longest legitimate turn is a thread nobody can message again.
        if not await self._redis.set(key, turn_id, nx=True, ex=self._ttl_seconds):
            if sent is not None:
                # This send bought nothing, so its key is not spent. Only the
                # record written a moment ago on this path is dropped.
                await self._redis.delete(sent)
            running = await self._redis.get(key)
            return TurnClaim(turn_id=str(running), outcome=ALREADY_RUNNING)
        self._tasks[turn_id] = asyncio.create_task(self._run(turn_id, thread_id, frames))
        return TurnClaim(turn_id=turn_id)

    async def turn_for(self, thread_id: str) -> str | None:
        """Which turn a reader attaching to this thread should stream.

        The turn that just ended still counts. Its claim is dropped the
        instant it stops producing, but its terminal entry is what tells a
        returning reader the answer is ready — and history does not hold
        that answer until the turn is written back.
        """
        running = await self._redis.get(self._active_key(thread_id))
        if not running:
            running = await self._redis.get(self._last_key(thread_id))
        return str(running) if running else None

    async def follow(self, turn_id: str, cursor: str | None = None):
        """Replay this turn from ``cursor``, then follow it until it ends.

        Yields the turn's own frames byte-for-byte, each page followed by a
        ``cursor`` frame naming where to resume. The cursor comes *after* the
        frames it accounts for, so a reader that stores it has already
        rendered everything it covers.
        """
        # Resolved up front so an empty first page does not look like
        # progress and send a reader a resume point it already had.
        cursor = cursor or START_CURSOR
        while True:
            page = await self._log.read(turn_id, cursor)
            for frame in page.frames:
                yield frame
            if page.cursor != cursor:
                cursor = page.cursor
                yield _render("cursor", {"turn_id": turn_id, "cursor": cursor})
            if page.terminal is not None:
                return
            if not page.frames and await self._log.terminal_of(turn_id) is not None:
                # Resumed from at or past the terminal entry — a remount
                # replaying a stored cursor. Only asked when there was nothing
                # to deliver, so a working turn pays no extra round trip.
                return
            await asyncio.sleep(self._poll_interval)

    async def wait(self, turn_id: str) -> None:
        """Block until this turn has finished producing."""
        task = self._tasks.get(turn_id)
        if task is not None:
            await task

    @property
    def in_flight(self) -> int:
        """How many turns this replica is running right now."""
        return len(self._tasks)

    async def _run(self, turn_id: str, thread_id: str, frames: AsyncIterator[str]) -> None:
        try:
            await self._produce(turn_id, frames)
        finally:
            # Dropped from inside the task, so anyone awaiting it sees the
            # registry already let go. Holding every turn a replica ever ran
            # keeps its coroutine frame, and everything that frame captured,
            # for the life of the process.
            self._tasks.pop(turn_id, None)
            # Released however the turn ended, including a raise: a claim the
            # turn never gives back wedges the thread for every later message.
            # The pointer outlives the claim so a reader arriving in the gap
            # still reaches the stream, for as long as the stream itself lasts.
            pipe = self._redis.pipeline(transaction=False)
            pipe.set(self._last_key(thread_id), turn_id, ex=TTL_MARGIN_SECONDS)
            pipe.delete(self._active_key(thread_id))
            await pipe.execute()

    async def _produce(self, turn_id: str, frames: AsyncIterator[str]) -> None:
        held: str | None = None
        async for frame in frames:
            if held is not None:
                await self._log.append(turn_id, held)
                held = None
            if _event_name(frame) in _CLOSING_EVENTS:
                # Held rather than appended: whichever closing frame is still
                # in hand when the turn stops producing is the one that
                # belongs in the terminal entry. Holding costs no latency —
                # both names are only ever emitted at the end of a turn, and a
                # held frame is released the moment another arrives.
                held = frame
            elif _event_name(frame) == "text":
                await self._log.append_text(turn_id, frame)
            else:
                await self._log.append(turn_id, frame)
        if held is not None:
            await self._log.mark_terminal(turn_id, _event_name(held), held)


def _render(event: str, data: dict) -> str:
    """An SSE frame in the shape ``ChatStreamService.sse`` renders one.

    Rendered here rather than imported from the chat service: the cursor
    is the follower's own protocol with its reader, not something the turn
    said, and the log never sees it.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_name(frame: str) -> str:
    """The event a rendered SSE frame carries.

    Read back off the frame rather than passed alongside it, because D5 puts
    the rendered bytes in the log verbatim — the name is already in them, and
    a parallel channel could disagree with what was stored.
    """
    first_line, _, _ = frame.partition("\n")
    prefix = "event: "
    return first_line[len(prefix):] if first_line.startswith(prefix) else ""
