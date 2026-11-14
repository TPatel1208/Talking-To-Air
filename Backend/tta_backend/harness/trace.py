"""The durable trace log: one run per request, one entry per step that ran.

Entries are audit records. They hold handle ids and small values, never data
or secrets, so what goes into ``params`` and ``result`` is checked here rather
than trusted to callers.
"""
from __future__ import annotations

import asyncio
import datetime
import enum
import json
import math
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, get_args

import psycopg
from psycopg.types.json import Jsonb

from tta_backend.harness.database import Connect
from tta_backend.harness.release import HARNESS_RELEASE
from tta_backend.harness.results import RUN_ID_PATTERN, Mode

RunStatus = Literal[
    "running", "done", "needs_input", "insufficient_evidence", "failed", "cancelled"
]
EntryKind = Literal["use", "ask", "agent", "workflow", "stop"]
EntryStatus = Literal["done", "failed", "skipped", "waiting"]

_RUN_STATUSES = frozenset(get_args(RunStatus))
# A run in one of these is over and its record is final; needs_input is a pause.
_TERMINAL = frozenset({"done", "failed", "cancelled", "insufficient_evidence"})
_KINDS = frozenset(get_args(EntryKind))
_ENTRY_STATUSES = frozenset(get_args(EntryStatus))

# A serialized params or result above this is almost certainly data, not a handle.
MAX_JSON_BYTES = 4096
MAX_REASON_CHARS = 1000


class FailureCategory(enum.StrEnum):
    BAD_INPUT = "bad_input"
    # An input handle's data was evicted; re-running the step that made it recovers it.
    INPUT_EVICTED = "input_evicted"
    NO_DATA = "no_data"
    REQUEST_TOO_LARGE = "request_too_large"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_DOWN = "source_down"
    CREDENTIAL_INVALID = "credential_invalid"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    INTERNAL_ERROR = "internal_error"
    MODEL_FAILURE = "model_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    # A step stopped because its task was cancelled; nothing went wrong in it.
    CANCELLED = "cancelled"


class TraceError(Exception):
    pass


class InvalidTrace(TraceError, ValueError):
    pass


class RunNotFound(TraceError):
    """Raised for a missing run and for another user's run alike."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"no such run: {run_id}")


class RunClosed(TraceError):
    """Raised for a write the run's current status does not allow."""

    def __init__(self, run_id: str, status: str) -> None:
        super().__init__(f"run {run_id} is {status}")
        self.status = status


class StepFailure(Exception):
    """A step failure that already knows its category and reason.

    The reason is stored as given, so callers write it as fixed wording.
    ``result``, plain JSON, is stored on the failed entry, for what the step
    still reports, such as the values behind an insufficient_evidence answer.
    """

    def __init__(self, category: FailureCategory, reason: str, *, result: Any = None) -> None:
        super().__init__(reason)
        self.category = category
        self.reason = reason
        self.result = result


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def _check_text(value: str, label: str) -> None:
    # Postgres cannot store NUL in TEXT or JSONB.
    if "\x00" in value:
        raise InvalidTrace(f"{label} holds a NUL character")


def _too_large(label: str, size: str) -> InvalidTrace:
    return InvalidTrace(
        f"{label} is {size} bytes serialized; the limit is {MAX_JSON_BYTES}. "
        "Store data as a handle and record the handle id."
    )


def _check_json(value: Any, label: str) -> None:
    # Walked without recursion under a lower bound of the serialized size, so a
    # deep or cyclic value fails as InvalidTrace before anything recurses.
    budget = MAX_JSON_BYTES
    pending = [value]
    while pending:
        item = pending.pop()
        budget -= 2 if isinstance(item, (list, dict)) else 1
        if budget < 0:
            raise _too_large(label, f"over {MAX_JSON_BYTES}")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, str):
            _check_text(item, label)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise InvalidTrace(f"{label} holds a non-finite number")
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise InvalidTrace(f"{label} has a non-string key")
                _check_text(key, label)
                pending.append(child)
        else:
            raise InvalidTrace(f"{label} must be plain JSON, got {type(item).__name__}")


def serialize_json(value: Any, label: str) -> str:
    """The JSON text of a small plain-JSON value; raises InvalidTrace otherwise."""
    _check_json(value, label)
    try:
        text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    except RecursionError:
        raise InvalidTrace(f"{label} is nested too deeply") from None
    size = len(text.encode())
    if size > MAX_JSON_BYTES:
        raise _too_large(label, str(size))
    return text


def _require_aware(value: datetime.datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidTrace(f"{label} must be timezone-aware")


def _require_run_id(value: str, label: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise InvalidTrace(f"{label} must be 'run_' and 32 lowercase hex digits: {value!r}")


@dataclass(frozen=True, kw_only=True)
class Run:
    id: str
    user_id: str
    conversation_id: str
    # The workflow and child versions the run used, kept for the result's label.
    mode: Mode
    started: datetime.datetime
    status: RunStatus = "running"
    harness_release: str = HARNESS_RELEASE
    ended: datetime.datetime | None = None
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        _require_run_id(self.id, "run id")
        if self.parent_run_id is not None:
            _require_run_id(self.parent_run_id, "parent_run_id")
        if not self.user_id or not self.conversation_id:
            raise InvalidTrace("a run needs a user and a conversation")
        _check_text(self.user_id, "user_id")
        _check_text(self.conversation_id, "conversation_id")
        if not isinstance(self.mode, Mode):
            raise InvalidTrace(f"mode must be a Mode, got {type(self.mode).__name__}")
        serialize_json(self.mode.model_dump(mode="json"), "mode")
        if self.status not in _RUN_STATUSES:
            raise InvalidTrace(f"unknown run status: {self.status!r}")
        _require_aware(self.started, "started")
        if self.ended is not None:
            _require_aware(self.ended, "ended")


@dataclass(frozen=True, kw_only=True)
class NewEntry:
    """An entry before the store has numbered it."""

    run_id: str
    kind: EntryKind
    status: EntryStatus
    started: datetime.datetime
    seconds: float
    capability: str | None = None
    # The spec's ``with``; handle ids and small values only.
    params: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    step: str | None = None
    parent_seq: int | None = None
    failure_category: FailureCategory | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise InvalidTrace(f"unknown entry kind: {self.kind!r}")
        if self.status not in _ENTRY_STATUSES:
            raise InvalidTrace(f"unknown entry status: {self.status!r}")
        if self.kind in ("use", "workflow") and not self.capability:
            raise InvalidTrace(f"a {self.kind} entry needs a capability")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise InvalidTrace("seconds must be a non-negative number")
        if self.failure_category is not None:
            FailureCategory(self.failure_category)
        if self.status == "failed" and self.failure_category is None:
            raise InvalidTrace("a failed entry needs a failure category")
        if self.status in ("done", "skipped") and self.failure_category is not None:
            raise InvalidTrace(f"a {self.status} entry cannot carry a failure category")
        if self.reason is not None and len(self.reason) > MAX_REASON_CHARS:
            raise InvalidTrace(f"reason is longer than {MAX_REASON_CHARS} characters")
        for label, text in (
            ("capability", self.capability), ("step", self.step), ("reason", self.reason)
        ):
            if text is not None:
                _check_text(text, label)
        if self.parent_seq is not None and self.parent_seq < 1:
            raise InvalidTrace("parent_seq must be a positive sequence number")
        _require_aware(self.started, "started")
        serialize_json(self.params, "params")
        serialize_json(self.result, "result")


@dataclass(frozen=True, kw_only=True)
class TraceEntry(NewEntry):
    seq: int


class TraceStore(Protocol):
    """Every read and write names the requesting user; a run that is someone
    else's raises RunNotFound exactly as a missing one does. Runs are keyed by
    user and id, so two users' runs never collide.

    Entries are written only while a run is ``running``. A run paused in
    ``needs_input`` takes no entries until ``resume_run``; a finished run's
    record is final, so every write to it raises RunClosed.
    """

    async def start_run(self, run: Run) -> None: ...

    async def finish_run(
        self, user_id: str, run_id: str, status: RunStatus, ended: datetime.datetime
    ) -> Run: ...

    async def resume_run(self, user_id: str, run_id: str) -> Run:
        """Return a run paused in ``needs_input`` to ``running``."""
        ...

    async def reserve_seq(self, user_id: str, run_id: str) -> int:
        """Claim the next sequence number so children can nest under a step
        that has not finished yet."""
        ...

    async def append(
        self, user_id: str, entry: NewEntry, *, seq: int | None = None
    ) -> TraceEntry:
        """Store an entry under ``seq`` (reserved earlier) or the next number."""
        ...

    async def get_run(self, user_id: str, run_id: str) -> Run: ...

    async def entries(self, user_id: str, run_id: str) -> list[TraceEntry]: ...

    async def list_runs(self, user_id: str, conversation_id: str) -> list[Run]: ...


def _check_start(run: Run) -> None:
    if run.status != "running" or run.ended is not None:
        raise InvalidTrace("a run starts as running")


def _check_finish(status: RunStatus, ended: datetime.datetime) -> None:
    if status == "running" or status not in _RUN_STATUSES:
        raise InvalidTrace(f"cannot finish a run as {status!r}")
    _require_aware(ended, "ended")


def _require_running(run_id: str, status: str) -> None:
    if status != "running":
        raise RunClosed(run_id, status)


def _require_unfinished(run_id: str, status: str) -> None:
    if status in _TERMINAL:
        raise RunClosed(run_id, status)


def _require_paused(run_id: str, status: str) -> None:
    if status != "needs_input":
        raise RunClosed(run_id, status)


def _check_numbering(entry: NewEntry, seq: int | None, last_seq: int) -> int:
    """The seq the entry takes: ``seq`` if it was reserved, else the next one."""
    if seq is not None and not 1 <= seq <= last_seq:
        raise InvalidTrace(f"seq {seq} was never reserved")
    target = last_seq + 1 if seq is None else seq
    if entry.parent_seq is not None and entry.parent_seq >= target:
        raise InvalidTrace(f"parent_seq {entry.parent_seq} must come before seq {target}")
    return target


class InMemoryTraceStore:
    def __init__(self) -> None:
        # Keyed by (user_id, run_id).
        self._runs: dict[tuple[str, str], Run] = {}
        self._last_seq: dict[tuple[str, str], int] = {}
        # Entries are kept as JSON text so callers never alias stored data.
        self._entries: dict[tuple[str, str], dict[int, tuple[TraceEntry, str, str]]] = {}

    def _own(self, user_id: str, run_id: str) -> Run:
        run = self._runs.get((user_id, run_id))
        if run is None:
            raise RunNotFound(run_id)
        return run

    async def start_run(self, run: Run) -> None:
        _check_start(run)
        key = (run.user_id, run.id)
        if key in self._runs:
            raise TraceError(f"run already exists: {run.id}")
        if run.parent_run_id is not None:
            self._own(run.user_id, run.parent_run_id)
        self._runs[key] = run
        self._last_seq[key] = 0
        self._entries[key] = {}

    async def finish_run(
        self, user_id: str, run_id: str, status: RunStatus, ended: datetime.datetime
    ) -> Run:
        run = self._own(user_id, run_id)
        _check_finish(status, ended)
        _require_unfinished(run_id, run.status)
        run = replace(run, status=status, ended=ended)
        self._runs[(user_id, run_id)] = run
        return run

    async def resume_run(self, user_id: str, run_id: str) -> Run:
        run = self._own(user_id, run_id)
        _require_paused(run_id, run.status)
        run = replace(run, status="running", ended=None)
        self._runs[(user_id, run_id)] = run
        return run

    async def reserve_seq(self, user_id: str, run_id: str) -> int:
        _require_running(run_id, self._own(user_id, run_id).status)
        self._last_seq[(user_id, run_id)] += 1
        return self._last_seq[(user_id, run_id)]

    async def append(
        self, user_id: str, entry: NewEntry, *, seq: int | None = None
    ) -> TraceEntry:
        _require_running(entry.run_id, self._own(user_id, entry.run_id).status)
        key = (user_id, entry.run_id)
        params = serialize_json(entry.params, "params")
        result = serialize_json(entry.result, "result")
        target = _check_numbering(entry, seq, self._last_seq[key])
        stored = self._entries[key]
        if target in stored:
            raise TraceError(f"seq {target} is already used in {entry.run_id}")
        if seq is None:
            self._last_seq[key] = target
        saved = _numbered(entry, target, json.loads(params), json.loads(result))
        stored[target] = (saved, params, result)
        return _copy_entry(saved, params, result)

    async def get_run(self, user_id: str, run_id: str) -> Run:
        return self._own(user_id, run_id)

    async def entries(self, user_id: str, run_id: str) -> list[TraceEntry]:
        self._own(user_id, run_id)
        stored = self._entries[(user_id, run_id)]
        return [_copy_entry(*stored[seq]) for seq in sorted(stored)]

    async def list_runs(self, user_id: str, conversation_id: str) -> list[Run]:
        runs = [
            r
            for r in self._runs.values()
            if r.user_id == user_id and r.conversation_id == conversation_id
        ]
        return sorted(runs, key=lambda r: (r.started, r.id))


def _numbered(entry: NewEntry, seq: int, params: Any, result: Any) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        run_id=entry.run_id,
        kind=entry.kind,
        status=entry.status,
        started=entry.started,
        seconds=entry.seconds,
        capability=entry.capability,
        params=params,
        result=result,
        step=entry.step,
        parent_seq=entry.parent_seq,
        failure_category=entry.failure_category,
        reason=entry.reason,
    )


def _copy_entry(entry: TraceEntry, params: str, result: str) -> TraceEntry:
    return _numbered(entry, entry.seq, json.loads(params), json.loads(result))


_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS harness_runs (
        user_id TEXT NOT NULL,
        id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        mode JSONB NOT NULL,
        status TEXT NOT NULL,
        harness_release TEXT NOT NULL,
        parent_run_id TEXT,
        started TIMESTAMPTZ NOT NULL,
        ended TIMESTAMPTZ,
        last_seq INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_harness_runs_conversation
        ON harness_runs (user_id, conversation_id, started)
    """,
    """
    CREATE TABLE IF NOT EXISTS harness_trace_entries (
        user_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        parent_seq INTEGER,
        step TEXT,
        kind TEXT NOT NULL,
        capability TEXT,
        params JSONB NOT NULL,
        result JSONB,
        status TEXT NOT NULL,
        failure_category TEXT,
        reason TEXT,
        started TIMESTAMPTZ NOT NULL,
        seconds DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (user_id, run_id, seq),
        FOREIGN KEY (user_id, run_id) REFERENCES harness_runs (user_id, id) ON DELETE CASCADE
    )
    """,
)


async def ensure_tables(connect: Connect) -> None:
    async with connect() as conn:
        for statement in _TABLES:
            await conn.execute(statement)


_RUN_COLUMNS = (
    "id, user_id, conversation_id, mode, status, harness_release, parent_run_id, started, ended"
)
_ENTRY_COLUMNS = (
    "seq, run_id, kind, status, started, seconds, capability, params, result, "
    "step, parent_seq, failure_category, reason"
)


def _run_from_row(row: tuple[Any, ...]) -> Run:
    return Run(
        id=row[0],
        user_id=row[1],
        conversation_id=row[2],
        mode=Mode.model_validate(row[3]),
        status=row[4],
        harness_release=row[5],
        parent_run_id=row[6],
        started=row[7],
        ended=row[8],
    )


def _entry_from_row(row: tuple[Any, ...]) -> TraceEntry:
    return TraceEntry(
        seq=row[0],
        run_id=row[1],
        kind=row[2],
        status=row[3],
        started=row[4],
        seconds=row[5],
        capability=row[6],
        params=row[7],
        result=row[8],
        step=row[9],
        parent_seq=row[10],
        failure_category=FailureCategory(row[11]) if row[11] is not None else None,
        reason=row[12],
    )


class PostgresTraceStore:
    def __init__(self, connect: Connect) -> None:
        self._connect = connect

    async def _lock(
        self,
        conn: psycopg.AsyncConnection[Any],
        user_id: str,
        run_id: str,
        *,
        exclusive: bool = True,
    ) -> tuple[str, int]:
        """The run's status and last seq, with its row locked until commit."""
        lock = "FOR UPDATE" if exclusive else "FOR SHARE"
        cur = await conn.execute(
            f"SELECT status, last_seq FROM harness_runs WHERE user_id = %s AND id = %s {lock}",
            (user_id, run_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return str(row[0]), int(row[1])

    async def _set_status(
        self,
        conn: psycopg.AsyncConnection[Any],
        user_id: str,
        run_id: str,
        status: RunStatus,
        ended: datetime.datetime | None,
    ) -> Run:
        cur = await conn.execute(
            "UPDATE harness_runs SET status = %s, ended = %s "
            f"WHERE user_id = %s AND id = %s RETURNING {_RUN_COLUMNS}",
            (status, ended, user_id, run_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return _run_from_row(row)

    async def start_run(self, run: Run) -> None:
        _check_start(run)
        async with self._connect() as conn:
            if run.parent_run_id is not None:
                cur = await conn.execute(
                    "SELECT 1 FROM harness_runs WHERE user_id = %s AND id = %s",
                    (run.user_id, run.parent_run_id),
                )
                if await cur.fetchone() is None:
                    raise RunNotFound(run.parent_run_id)
            try:
                await conn.execute(
                    f"INSERT INTO harness_runs ({_RUN_COLUMNS}) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        run.id,
                        run.user_id,
                        run.conversation_id,
                        Jsonb(run.mode.model_dump(mode="json")),
                        run.status,
                        run.harness_release,
                        run.parent_run_id,
                        run.started,
                        run.ended,
                    ),
                )
            except psycopg.errors.UniqueViolation:
                raise TraceError(f"run already exists: {run.id}") from None

    async def finish_run(
        self, user_id: str, run_id: str, status: RunStatus, ended: datetime.datetime
    ) -> Run:
        _check_finish(status, ended)
        async with self._connect() as conn:
            current, _ = await self._lock(conn, user_id, run_id)
            _require_unfinished(run_id, current)
            return await self._set_status(conn, user_id, run_id, status, ended)

    async def resume_run(self, user_id: str, run_id: str) -> Run:
        async with self._connect() as conn:
            current, _ = await self._lock(conn, user_id, run_id)
            _require_paused(run_id, current)
            return await self._set_status(conn, user_id, run_id, "running", None)

    async def reserve_seq(self, user_id: str, run_id: str) -> int:
        async with self._connect() as conn:
            status, last_seq = await self._lock(conn, user_id, run_id)
            _require_running(run_id, status)
            await conn.execute(
                "UPDATE harness_runs SET last_seq = %s WHERE user_id = %s AND id = %s",
                (last_seq + 1, user_id, run_id),
            )
            return last_seq + 1

    async def append(
        self, user_id: str, entry: NewEntry, *, seq: int | None = None
    ) -> TraceEntry:
        params = serialize_json(entry.params, "params")
        result = None if entry.result is None else serialize_json(entry.result, "result")
        async with self._connect() as conn:
            # An append that takes a new number serialises on the row lock; one
            # filling a reserved number only needs the run to stay put.
            status, last_seq = await self._lock(
                conn, user_id, entry.run_id, exclusive=seq is None
            )
            _require_running(entry.run_id, status)
            target = _check_numbering(entry, seq, last_seq)
            if seq is None:
                await conn.execute(
                    "UPDATE harness_runs SET last_seq = %s WHERE user_id = %s AND id = %s",
                    (target, user_id, entry.run_id),
                )
            try:
                await conn.execute(
                    f"INSERT INTO harness_trace_entries (user_id, {_ENTRY_COLUMNS}) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        user_id,
                        target,
                        entry.run_id,
                        entry.kind,
                        entry.status,
                        entry.started,
                        entry.seconds,
                        entry.capability,
                        Jsonb(json.loads(params)),
                        None if result is None else Jsonb(json.loads(result)),
                        entry.step,
                        entry.parent_seq,
                        entry.failure_category.value if entry.failure_category else None,
                        entry.reason,
                    ),
                )
            except psycopg.errors.UniqueViolation:
                raise TraceError(f"seq {target} is already used in {entry.run_id}") from None
        return _numbered(
            entry, target, json.loads(params), None if result is None else json.loads(result)
        )

    async def get_run(self, user_id: str, run_id: str) -> Run:
        async with self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM harness_runs WHERE user_id = %s AND id = %s",
                (user_id, run_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return _run_from_row(row)

    async def entries(self, user_id: str, run_id: str) -> list[TraceEntry]:
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM harness_runs WHERE user_id = %s AND id = %s", (user_id, run_id)
            )
            if await cur.fetchone() is None:
                raise RunNotFound(run_id)
            cur = await conn.execute(
                f"SELECT {_ENTRY_COLUMNS} FROM harness_trace_entries "
                "WHERE user_id = %s AND run_id = %s ORDER BY seq",
                (user_id, run_id),
            )
            rows = await cur.fetchall()
        return [_entry_from_row(r) for r in rows]

    async def list_runs(self, user_id: str, conversation_id: str) -> list[Run]:
        async with self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM harness_runs "
                "WHERE user_id = %s AND conversation_id = %s ORDER BY started, id",
                (user_id, conversation_id),
            )
            rows = await cur.fetchall()
        return [_run_from_row(r) for r in rows]


# Fixed wording, so the trace never carries text from an exception or a value.
UNRECORDABLE_REASON = "the step's outcome could not be recorded"
_CANCELLED_REASON = "the step was cancelled"


class StepScope:
    """What a recorded step can set before it ends."""

    def __init__(self, seq: int) -> None:
        # Children of this step pass it as their parent_seq.
        self.seq = seq
        self.result: Any = None
        self.status: EntryStatus = "done"
        self.reason: str | None = None


class StepRecorder:
    """Writes one entry around a step: start time, duration, outcome.

    Time comes from the injected callables so a benchmark run repeats exactly.
    """

    def __init__(
        self,
        store: TraceStore,
        user_id: str,
        run_id: str,
        *,
        now: Callable[[], datetime.datetime],
        monotonic: Callable[[], float],
    ) -> None:
        self._store = store
        self._user_id = user_id
        self._run_id = run_id
        self._now = now
        self._monotonic = monotonic

    @asynccontextmanager
    async def step(
        self,
        *,
        kind: EntryKind,
        capability: str | None = None,
        params: dict[str, Any] | None = None,
        step: str | None = None,
        parent_seq: int | None = None,
    ) -> AsyncIterator[StepScope]:
        """Record the step on exit; every way out of the step leaves an entry.

        An exception or cancellation becomes a failed entry and is re-raised. A
        plain exception is recorded by its type name only, since its message can
        carry URLs, headers or query strings. An outcome that cannot be stored
        is replaced by a failed ``internal_error`` entry and InvalidTrace raised;
        for a StepFailure whose result cannot be stored, an ``internal_error``
        StepFailure is raised instead, matching the entry.
        """
        common: dict[str, Any] = {
            "run_id": self._run_id,
            "kind": kind,
            "capability": capability,
            "params": params or {},
            "step": step,
            "parent_seq": parent_seq,
            "started": self._now(),
        }
        # Checked before the body runs, so bad arguments fail before any work.
        NewEntry(**common, status="done", seconds=0.0)
        seq = await self._store.reserve_seq(self._user_id, self._run_id)
        scope = StepScope(seq)
        began = self._monotonic()

        def build(**outcome: Any) -> NewEntry:
            seconds = max(0.0, self._monotonic() - began)
            return NewEntry(**common, seconds=seconds, **outcome)

        async def write(**outcome: Any) -> InvalidTrace | None:
            refused: InvalidTrace | None = None
            try:
                entry = build(**outcome)
            except InvalidTrace as exc:
                refused = exc
                entry = build(
                    status="failed",
                    failure_category=FailureCategory.INTERNAL_ERROR,
                    reason=UNRECORDABLE_REASON,
                )
            await self._store.append(self._user_id, entry, seq=seq)
            return refused

        try:
            yield scope
        except asyncio.CancelledError:
            await write(
                status="failed",
                failure_category=FailureCategory.CANCELLED,
                reason=_CANCELLED_REASON,
            )
            raise
        except StepFailure as exc:
            refused = await write(
                status="failed",
                failure_category=exc.category,
                reason=exc.reason[:MAX_REASON_CHARS],
                result=exc.result,
            )
            if refused is not None:
                raise StepFailure(FailureCategory.INTERNAL_ERROR, UNRECORDABLE_REASON) from None
            raise
        except Exception as exc:
            await write(
                status="failed",
                failure_category=FailureCategory.INTERNAL_ERROR,
                reason=type(exc).__name__,
            )
            raise
        refused = await write(status=scope.status, result=scope.result, reason=scope.reason)
        if refused is not None:
            raise refused
