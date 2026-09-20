"""Who owns a running chat turn.

A turn is started by the request that posted the message and outlives it: the
registry runs it as its own task and feeds every frame it produces into the
event log, where any replica's reader can pick it up.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from redis import asyncio as aioredis

from tta_backend.config.settings import get_settings
from tta_backend.services.retrieval_composites import TERMINAL_STATUSES
from tta_backend.services.turn_event_log import (
    START_CURSOR,
    TTL_MARGIN_SECONDS,
    TurnEventLog,
)
from tta_backend.utils.streaming import HEARTBEAT_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

#: Events that can be the last thing a turn says. A turn timing out or
#: shedding emits ``error`` then ``done``; the generic failure path emits
#: only ``error``, so neither name alone identifies the close.
_CLOSING_EVENTS = frozenset({"done", "error"})

#: How long a follower waits before asking the log for more. The text
#: batcher already holds tokens ~50ms, so this is the second half of a
#: delay measured in tenths of a second against turns that run minutes.
DEFAULT_POLL_INTERVAL_SECONDS = 0.1

#: How often a replica re-reads the stop flags of the turns it is running.
#: The channel is what makes Stop feel instant; this is what makes it certain.
#: Redis pub/sub is fire-and-forget -- a message published while a subscriber
#: is between connections is gone, with no error on either side -- so without
#: a second look the button silently does nothing for that turn's whole life.
#: One MGET per replica per interval, skipped entirely while it is idle.
DEFAULT_STOP_POLL_SECONDS = 2.0

#: How long a turn's stream may stay silent before a reader gives up on it
#: (D14). A multiple of the heartbeat rather than a number of its own: the
#: watchdog in ``utils.streaming`` is what guarantees a live turn writes at
#: all during a slow retrieval, so the threshold is only meaningful relative
#: to it. Three intervals leaves room for two missed heartbeats before a turn
#: that is merely slow gets called dead.
STALE_AFTER_SECONDS = 3 * HEARTBEAT_INTERVAL_SECONDS

#: How long a thread's claim survives without its owner renewing it.
#:
#: The same number a reader uses to give up on a silent stream, and for the
#: same reason: a turn nobody would still call alive is a turn whose thread
#: should be free. Bounding it by the whole-turn deadline instead (~40 min)
#: wedges a thread for that long every time a replica is SIGKILLed or OOMed,
#: because the reader is then correctly told ``interrupted`` and the retry it
#: offers is answered with a 409 naming a turn that is already dead.
#:
#: Safe only because the owner renews it on ``DEFAULT_STOP_POLL_SECONDS``,
#: fifteen times over before it could lapse. Losing a claim here means a
#: replica that stopped running its own event loop — a turn that is not
#: writing frames either, so its reader has already given up on it.
CLAIM_TTL_SECONDS = int(STALE_AFTER_SECONDS)

#: How long a replica waits before resubscribing after its stop channel drops.
#: Short: the window it covers is one where Stop falls back to the poll.
DEFAULT_RESUBSCRIBE_DELAY_SECONDS = 0.5


#: Where stops are announced. One channel for the whole fleet rather than one
#: per turn: a replica subscribes once at startup instead of on every turn,
#: and the ids it does not recognise cost it a dict lookup.
STOP_CHANNEL = "chat:turn:stop"

#: The terminal kind a stopped turn is marked with. Not ``done``: the user
#: got the work that finished, not the answer they asked for (D11).
STOPPED = "stopped"

#: The terminal kind a turn gets when nobody asked it to end: its replica went
#: down under it, or a reader found its stream abandoned. Distinct from
#: ``stopped`` because the user did not choose it and the retry is theirs to
#: make (D16).
INTERRUPTED = "interrupted"

#: How long the drain waits for a turn to write its ``interrupted`` entry and
#: hand its thread back. The work is two Redis round trips per turn, so this
#: is a bound on a hung connection, not a budget -- a shutdown is already
#: capped by whatever the orchestrator allows before SIGKILL.
DEFAULT_DRAIN_TIMEOUT_SECONDS = 5.0


#: This caller's message started the turn it is being handed.
STARTED = "started"
#: The thread already had a turn in flight; the id is that one's.
ALREADY_RUNNING = "already_running"
#: This exact send was already accepted; the id is what it got the first time.
DUPLICATE = "duplicate"


#: Told the provider-job handles a stop orphaned, so they do not outlive the
#: turn that asked for them.
CancelJobs = Callable[[list[str]], Awaitable[None]]


class RegistryClosing(RuntimeError):
    """This replica is shutting down and will not start another turn.

    Not a ``TurnClaim`` outcome, because there is no turn to name: the caller
    has to be told to try again somewhere else, and a claim carrying an empty
    id would make that the route's job to notice.
    """


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
        stop_poll_interval: float = DEFAULT_STOP_POLL_SECONDS,
        resubscribe_delay: float = DEFAULT_RESUBSCRIBE_DELAY_SECONDS,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT_SECONDS,
        stale_after: float = STALE_AFTER_SECONDS,
        claim_ttl: int = CLAIM_TTL_SECONDS,
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
        # The inner task doing the agent's work, separately cancellable. Stop
        # takes this one and leaves ``_run`` alive to write the terminal entry
        # and hand the thread back -- cancelling the whole turn would take the
        # bookkeeping with it.
        self._consumers: dict[str, asyncio.Task] = {}
        #: Turns this replica cancelled deliberately, and how each should be
        #: marked. A turn missing from here was cancelled by something else
        #: going down on top of it, and writes no terminal entry at all.
        self._cancelling: dict[str, str] = {}
        self._listener: asyncio.Task | None = None
        self._pubsub: Any = None
        self._listening = asyncio.Lock()
        self._watchdog: asyncio.Task | None = None
        # Last-seen provider status per job_handle, per turn -- the same map
        # ``_LiveTurn`` keeps for the timeout answer, rebuilt here from the
        # frames that already pass through. Read only when a turn is stopped.
        self._job_statuses: dict[str, dict[str, str]] = {}
        self._cancel_jobs: dict[str, CancelJobs] = {}
        #: The thread each running turn claimed, so the renewal loop knows
        #: which claims are this replica's to keep alive.
        self._threads: dict[str, str] = {}
        self._poll_interval = poll_interval
        self._stop_poll_interval = stop_poll_interval
        self._resubscribe_delay = resubscribe_delay
        self._drain_timeout = drain_timeout
        self._stale_after = stale_after
        self._claim_ttl = claim_ttl
        #: Set the moment shutdown begins, and never cleared: this replica is
        #: on its way out and must not be handed work it cannot finish.
        self._draining = False
        # An idempotency record has to outlive the turn it names, or a
        # retry arriving late re-buys the answer it already has.
        self._ttl_seconds = int(get_settings().chat_turn_timeout_seconds) + TTL_MARGIN_SECONDS

    async def aclose(self) -> None:
        await self.drain()
        if self._listener is not None:
            self._listener.cancel()
            self._listener = None
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self._pubsub is not None:
            await self._pubsub.aclose()
            self._pubsub = None
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        if self._owns_client:
            await self._redis.aclose()

    async def drain(self) -> None:
        """Close down every turn this replica is running, and say so.

        Waiting is the point. Shutting down used to cancel each turn's task
        and return, which starts the bookkeeping and then races it against
        the pool being closed and the process exiting -- and loses that race
        precisely when Redis is slow, which is when a deploy is most likely
        to be under way.

        It cancels the inner consumer rather than the turn's own task, for
        the same reason a stop does: ``_run`` is then still a live task and
        can write. Cancelling the outer task happens to leave it able to
        write too (a cancel delivered through an awaited future does not mark
        the task itself), but that is an asyncio detail rather than something
        to build on, and it would give this path a second shape to maintain.
        """
        self._draining = True
        live = list(self._consumers)
        for turn_id in live:
            # Only a turn still producing has a consumer to cancel, which is
            # what keeps ``interrupted`` off a turn that already ended -- the
            # last entry is what ``terminal_of`` reads, so marking a finished
            # turn would relabel a delivered answer as a failure.
            self._cancel_local(turn_id, INTERRUPTED)
        tasks = [task for task in (self._tasks.get(t) for t in live) if task is not None]
        if tasks:
            # Bounded: a replica that will not come down is killed by the
            # orchestrator, and holding the loop here past that point buys
            # nothing and delays every turn behind it.
            _, pending = await asyncio.wait(tasks, timeout=self._drain_timeout)
            if pending:
                logger.warning(
                    "turn_drain_incomplete",
                    extra={"_event": "turn_drain_incomplete", "_turns": len(pending)},
                )

    def _active_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:active_turn"

    def _last_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:last_turn"

    def _idempotency_key(self, key: str) -> str:
        return f"send:{key}:turn"

    def _stop_key(self, turn_id: str) -> str:
        return f"turn:{turn_id}:stop"

    async def begin(
        self,
        thread_id: str,
        frames: AsyncIterator[str],
        idempotency_key: str | None = None,
        cancel_jobs: CancelJobs | None = None,
    ) -> TurnClaim:
        """Start a turn on this thread, unless one is already running.

        The claim is in Redis rather than in this process because the second
        tab's message can land on a different replica, which is the case
        sticky sessions fail silently.

        ``cancel_jobs`` is awaited with the handles a stop orphaned. Injected
        rather than imported so the registry stays a Redis-and-tasks module:
        cancelling a retrieval needs MCP tools and the turn's user, neither of
        which has anything to do with owning a turn.
        """
        if self._draining:
            # Before anything is minted or claimed: a turn refused here has
            # spent nothing, so the retry that follows it finds the thread
            # free and its idempotency key unused.
            raise RegistryClosing("this replica is shutting down")
        await self._ensure_listening()
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
        # replica -- and expiring *soon*, because until it does, that thread
        # refuses every message. Short enough to be wrong about a live turn,
        # which is why the owner renews it (``_refresh_claims``).
        if not await self._redis.set(key, turn_id, nx=True, ex=self._claim_ttl):
            if sent is not None:
                # This send bought nothing, so its key is not spent. Only the
                # record written a moment ago on this path is dropped.
                await self._redis.delete(sent)
            running = await self._redis.get(key)
            return TurnClaim(turn_id=str(running), outcome=ALREADY_RUNNING)
        if cancel_jobs is not None:
            self._cancel_jobs[turn_id] = cancel_jobs
        # Recorded before the task is spawned, so the first renewal tick
        # cannot land on a turn this replica owns but has not yet listed.
        self._threads[turn_id] = thread_id
        self._tasks[turn_id] = asyncio.create_task(self._run(turn_id, thread_id, frames))
        return TurnClaim(turn_id=turn_id)

    async def stop(self, turn_id: str) -> None:
        """Cancel this turn now, wherever it is running.

        Hard, not cooperative (D9): waiting for a superstep boundary makes
        Stop's latency the slowest step's, and a button that takes 40s to
        visibly stop reads as broken.
        """
        # Persisted first, then cancelled, and that order is the whole point.
        # A turn's owner registers its consumer before it reads this flag, so
        # a stop arriving in that window is caught by one side or the other:
        # write-then-cancel means we cannot both miss the consumer and be
        # missed by the flag read. Cancel-then-write could do exactly that.
        await self._redis.set(self._stop_key(turn_id), "1", ex=self._ttl_seconds)
        # Both, not either: the channel is what makes Stop feel instant, the
        # flag is what makes it certain. A publish reaches only whoever is
        # subscribed at that moment and is acknowledged by nobody, so on its
        # own it can be swallowed silently; the flag is re-read on a timer
        # and cannot be.
        await self._redis.publish(STOP_CHANNEL, turn_id)
        self._cancel_local(turn_id)

    async def _ensure_listening(self) -> None:
        """Subscribe this replica to the stop channel, once.

        Awaited before a turn id is minted, so by the time anyone could ask
        for this turn to stop, the replica that will own it is already
        listening.
        """
        if self._listener is not None:
            return
        async with self._listening:
            if self._listener is not None:
                return
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(STOP_CHANNEL)
            self._pubsub = pubsub
            self._listener = asyncio.create_task(self._listen(pubsub))
            self._watchdog = asyncio.create_task(self._watch_turns())

    async def _listen(self, pubsub: Any) -> None:
        """Take stops off the channel, and keep taking them across a drop.

        redis-py surfaces a dropped connection as an exception out of the read
        and does not put the subscription back. Left uncaught, the task dies
        and this replica never hears another stop -- Stop still works, via the
        watchdog, but at poll latency forever and with nothing saying why.
        """
        while True:
            try:
                async for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    # Most of these name turns other replicas own. Cancelling
                    # a turn we do not run is a no-op, so nothing is filtered.
                    self._cancel_local(str(message["data"]))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "turn_stop_listener_reconnecting",
                    exc_info=True,
                    extra={"_event": "turn_stop_listener_reconnecting"},
                )
            await asyncio.sleep(self._resubscribe_delay)
            try:
                await pubsub.subscribe(STOP_CHANNEL)
            except Exception:
                continue
            # Whatever was published while this replica was away is gone --
            # pub/sub keeps nothing for an absent subscriber -- so the flags
            # are the only record of a stop that fell in the gap.
            await self._recheck_stops()

    async def _watch_turns(self) -> None:
        """Both things a replica must keep doing for the turns it runs.

        One loop, because both are the same question asked of the same set --
        is this turn still mine, and does anyone still want it -- and two
        loops on the same cadence would only give the two answers a chance to
        disagree.
        """
        while True:
            await asyncio.sleep(self._stop_poll_interval)
            await self._recheck_stops()
            await self._refresh_claims()

    async def _refresh_claims(self) -> None:
        """Keep this replica's thread claims from lapsing under its turns.

        The claim's TTL is deliberately shorter than a turn may run
        (``CLAIM_TTL_SECONDS``), so a claim is only ever held by a replica
        still running its own event loop. This is what makes that safe for a
        turn that is merely slow.

        A finished turn leaves ``_threads`` before its release is awaited, so
        it is normally off this list first. ``EXPIRE`` rather than a write
        covers the window that leaves: the list is snapshotted and *then* the
        round trip is awaited, so a turn ending inside it has already deleted
        its claim -- EXPIRE on a missing key does nothing, where a SET would
        put the claim back with nobody left to release it.

        Failures are logged and swallowed: this shares a loop with the stop
        poll, whose death would take Stop's only guarantee with it.
        """
        live = list(self._threads.items())
        if not live:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            for _, thread_id in live:
                pipe.expire(self._active_key(thread_id), self._claim_ttl)
            await pipe.execute()
        except Exception:
            logger.warning(
                "turn_claim_refresh_failed",
                exc_info=True,
                extra={"_event": "turn_claim_refresh_failed", "_turns": len(live)},
            )

    async def _recheck_stops(self) -> None:
        """Cancel any turn here whose stop flag is set.

        One round trip for the whole replica however many turns it holds, so
        the cost does not scale with load. A failure is logged and swallowed
        rather than raised: this runs from a loop whose death would take
        Stop's only guarantee with it.
        """
        live = list(self._consumers)
        if not live:
            return
        try:
            flags = await self._redis.mget([self._stop_key(t) for t in live])
        except Exception:
            logger.warning(
                "turn_stop_watch_failed",
                exc_info=True,
                extra={"_event": "turn_stop_watch_failed"},
            )
            return
        for turn_id, flag in zip(live, flags):
            if flag:
                self._cancel_local(turn_id)

    def _cancel_local(self, turn_id: str, kind: str = STOPPED) -> None:
        """Cancel this turn if this replica is the one running it.

        ``kind`` is the terminal entry the turn will end up carrying, so a
        reader can tell a user who pressed Stop from a deploy that took the
        answer away.
        """
        consumer = self._consumers.get(turn_id)
        if consumer is not None:
            self._cancelling[turn_id] = kind
            consumer.cancel()

    async def turn_for(self, thread_id: str, *, include_ended: bool = True) -> str | None:
        """Which turn a reader attaching to this thread should stream.

        The turn that just ended still counts, for a reader that knows about
        it. Its claim is dropped the instant it stops producing, but its
        terminal entry is what tells a returning reader the answer is ready —
        and history does not hold that answer until the turn is written back.

        ``include_ended=False`` is for the other kind of reader: one that is
        only asking whether anything is running here. The pointer to the last
        turn outlives that turn by ten minutes, so handing it to a reader that
        just loaded history would replay an answer it is already showing.
        """
        running = await self._redis.get(self._active_key(thread_id))
        if not running and include_ended:
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
        # What a turn with nothing in its stream yet is measured against. A
        # reader attaching the instant the 202 comes back is ahead of the
        # turn's first write, and has no entry to date it from.
        attached_ms = _now_ms()
        while True:
            page = await self._log.read(turn_id, cursor)
            for frame in page.frames:
                yield frame
            if page.cursor != cursor:
                cursor = page.cursor
                yield _render("cursor", {"turn_id": turn_id, "cursor": cursor})
            if page.terminal is not None:
                return
            if not page.frames:
                # Only asked when there was nothing to deliver, so a working
                # turn pays no extra round trip for either question.
                tail = await self._log.tail(turn_id)
                if tail.terminal is not None:
                    # Resumed from at or past the terminal entry — a remount
                    # replaying a stored cursor.
                    return
                quiet_since = attached_ms if tail.written_ms is None else tail.written_ms
                if _now_ms() - quiet_since > self._stale_after * 1000:
                    # D14: nobody is writing and nobody marked it finished, so
                    # the replica that owned it went down without draining —
                    # killed, OOMed, or taken with its node. Reported rather
                    # than written: a turn merely paused longer than the
                    # heartbeat allows is still alive, and recording that
                    # guess would relabel an answer that is on its way.
                    yield _render(INTERRUPTED, {"turn_id": turn_id, "reason": "stale"})
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
        consumer = asyncio.create_task(self._produce(turn_id, frames))
        # Registered before the first await, so a stop racing the turn's own
        # startup finds it here even though ``begin`` returned before this
        # coroutine ever ran.
        self._consumers[turn_id] = consumer
        if await self._redis.exists(self._stop_key(turn_id)):
            # Stopped before the owner was listening -- the case the flag
            # exists for, and the one pub/sub alone cannot cover.
            self._cancel_local(turn_id)
        try:
            await consumer
        except asyncio.CancelledError:
            kind = self._cancelling.get(turn_id)
            if kind is None:
                # Nobody here asked for this -- the cancel came from outside
                # the registry, and there is nothing this task can still do
                # about it. A deliberate shutdown goes through ``drain``,
                # which cancels the consumer and leaves this task alive to
                # write; a cancel landing here instead leaves the turn
                # unmarked, and a reader learns it is dead from the stale
                # stream (D14).
                raise
            if kind == STOPPED:
                # Only a user's Stop drops the retrievals. An interrupted turn
                # is one the user will retry, and its cached results are worth
                # more to that retry than the provider capacity is -- and by
                # the time a shutdown reaches here the MCP connection is
                # already closed, so the call would only fail slowly.
                await self._cancel_orphaned_jobs(turn_id)
            ending: dict[str, str] = {"thread_id": thread_id}
            if kind == INTERRUPTED:
                # Which interruption, because a reader gets ``interrupted``
                # two ways -- written here by a replica on its way down, or
                # inferred by a follower that found the stream abandoned --
                # and only one of them knows the answer is really gone.
                ending["reason"] = "shutdown"
            await self._log.mark_terminal(turn_id, kind, _render(kind, ending))
        finally:
            self._consumers.pop(turn_id, None)
            self._cancelling.pop(turn_id, None)
            self._job_statuses.pop(turn_id, None)
            self._cancel_jobs.pop(turn_id, None)
            self._threads.pop(turn_id, None)
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

    async def _cancel_orphaned_jobs(self, turn_id: str) -> None:
        """Tell the provider to drop whatever this turn left running.

        Best effort, and deliberately not fatal: the turn is already stopping,
        and a provider that will not take the cancel must not also cost the
        user the ``stopped`` entry that ends their spinner.
        """
        cancel = self._cancel_jobs.get(turn_id)
        if cancel is None:
            return
        orphaned = [
            handle
            for handle, status in self._job_statuses.get(turn_id, {}).items()
            if status not in TERMINAL_STATUSES
        ]
        if not orphaned:
            return
        try:
            await cancel(orphaned)
        except Exception:
            logger.warning(
                "turn_stop_job_cancel_failed",
                exc_info=True,
                extra={"_event": "turn_stop_job_cancel_failed", "_turn_id": turn_id},
            )

    def _note_job_status(self, turn_id: str, frame: str) -> None:
        """Remember what this turn last heard about a retrieval job."""
        _, _, data = frame.partition("data: ")
        try:
            payload = json.loads(data)
        except ValueError:
            return
        handle = payload.get("job_handle")
        if handle:
            self._job_statuses.setdefault(turn_id, {})[handle] = payload.get("status", "")

    async def _produce(self, turn_id: str, frames: AsyncIterator[str]) -> None:
        held: str | None = None
        async for frame in frames:
            if held is not None:
                await self._log.append(turn_id, held)
                held = None
            event = _event_name(frame)
            if event == "job_progress":
                self._note_job_status(turn_id, frame)
            if event in _CLOSING_EVENTS:
                # Held rather than appended: whichever closing frame is still
                # in hand when the turn stops producing is the one that
                # belongs in the terminal entry. Holding costs no latency —
                # both names are only ever emitted at the end of a turn, and a
                # held frame is released the moment another arrives.
                held = frame
            elif event == "text":
                await self._log.append_text(turn_id, frame)
            else:
                await self._log.append(turn_id, frame)
        if held is not None:
            await self._log.mark_terminal(turn_id, _event_name(held), held)


def _now_ms() -> int:
    """Wall clock, to compare against a stream entry's own timestamp.

    Wall clock and not a monotonic one, because the other side of the
    comparison is Redis's clock. That makes the threshold sensitive to skew
    between this process and Redis — which is why it is thirty seconds and
    not one: an NTP-synced pair in the same deployment is out by
    milliseconds, and the cost of being wrong is telling a reader to retry a
    turn that was going to answer.
    """
    return int(time.time() * 1000)


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
