"""In-memory run-context parts for tests and benchmarks."""
from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from typing import Any

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.results import Mode
from tta_backend.harness.run_context import (
    ArtifactRef,
    Artifacts,
    Clock,
    Compute,
    Credentials,
    EarthdataSession,
    Events,
    Identity,
    Limits,
    RunContext,
    RunEnv,
    run_env,
)
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.trace import Run, new_run_id


class RecordingEvents:
    def __init__(self) -> None:
        self.progress_events: list[dict[str, Any]] = []
        self.charts: list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []

    def progress(self, message: str, *, stage: str | None = None, detail: Any = None) -> None:
        self.progress_events.append({"message": message, "stage": stage, "detail": detail})

    def chart(self, payload: dict[str, Any]) -> None:
        self.charts.append(payload)

    def question(self, payload: dict[str, Any]) -> None:
        self.questions.append(payload)


class MemoryArtifacts:
    def __init__(self) -> None:
        self.stored: dict[str, dict[str, Any]] = {}

    def _mint(self, kind: str) -> str:
        return f"{kind}-{len(self.stored) + 1}"

    async def store_chart(self, payload: dict[str, Any]) -> ArtifactRef:
        ref = ArtifactRef(id=self._mint("chart"), type="chart", title=payload.get("title"))
        self.stored[ref.id] = payload
        return ref

    async def store_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        ref = ArtifactRef(id=self._mint("table"), type="table", title=title)
        self.stored[ref.id] = {
            "title": title,
            "columns": columns,
            "rows": rows,
            "metadata": metadata,
        }
        return ref


class StaticCredentials(dict[str, str]):
    async def secret(self, connector: str) -> str | None:
        return self.get(connector)


class FixedClock:
    def __init__(self, now: datetime.datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware datetime")
        self._now = now

    def today(self) -> datetime.date:
        return self._now.date()

    def now(self) -> datetime.datetime:
        return self._now


class FakeToolError(Exception):
    """A classified source failure, shaped like the earthdata errors: a string ``category``."""

    def __init__(self, category: str, message: str = "") -> None:
        super().__init__(message)
        self.category = category


class FakeEarthdata:
    """Scripted earthdata tools: each maps arguments to a result, or raises."""

    def __init__(self, tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None) -> None:
        self.tools = dict(tools or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, dict(arguments)))
        if tool not in self.tools:
            raise FakeToolError("contract", f"unscripted tool {tool}")
        return self.tools[tool](arguments)


class InlineCompute:
    """Runs the work on the calling thread, so tests stay deterministic."""

    def __init__(self) -> None:
        self.runs = 0

    async def run[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T:
        self.runs += 1
        return fn(*args, **kwargs)


TEST_LIMITS = Limits(
    retrieval_soft_bytes=50_000_000,
    retrieval_hard_bytes=200_000_000,
    max_point_timeseries_days=366,
    retrieval_timeout_seconds=60.0,
    poll_min_seconds=0.01,
    poll_max_seconds=0.1,
)


async def _no_retrievals(user_id: str, record: HandleRecord) -> Any:
    raise HandleError("not_found", record.id)


async def make_run_env(
    payload_root: str | os.PathLike[str],
    *,
    context: RunContext | None = None,
    earthdata: EarthdataSession | None = None,
    compute: Compute | None = None,
    limits: Limits | None = None,
    open_retrieval: OpenRetrieval | None = None,
    mode: Mode | None = None,
) -> RunEnv:
    """A run started in the context's trace store, with an environment on fakes."""
    context = context or make_run_context()
    run = Run(
        id=new_run_id(),
        user_id=context.identity.user_id,
        conversation_id=context.identity.conversation_id,
        mode=mode or Mode.exploratory(),
        started=context.clock.now(),
    )
    await context.storage.traces.start_run(run)
    return run_env(
        context,
        run_id=run.id,
        payloads=PayloadStore(payload_root),
        open_retrieval=open_retrieval or _no_retrievals,
        earthdata=earthdata or FakeEarthdata(),
        compute=compute or InlineCompute(),
        limits=limits or TEST_LIMITS,
    )


def make_run_context(
    *,
    user_id: str = "user-a",
    conversation_id: str = "conv-1",
    events: Events | None = None,
    artifacts: Artifacts | None = None,
    credentials: Credentials | None = None,
    clock: Clock | None = None,
    stores: Stores | None = None,
) -> RunContext:
    """A run context on fakes. Pass one ``stores`` to several contexts to test users against each other."""
    return RunContext(
        identity=Identity(user_id=user_id, conversation_id=conversation_id),
        events=events or RecordingEvents(),
        artifacts=artifacts or MemoryArtifacts(),
        credentials=credentials or StaticCredentials(),
        storage=stores or memory_stores(),
        clock=clock or FixedClock(datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)),
    )
