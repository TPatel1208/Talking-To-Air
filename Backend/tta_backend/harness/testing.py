"""In-memory run-context parts for tests and benchmarks."""
from __future__ import annotations

import datetime
from typing import Any

from tta_backend.harness.run_context import (
    ArtifactRef,
    Artifacts,
    Clock,
    Credentials,
    Events,
    Identity,
    RunContext,
)
from tta_backend.harness.storage import Stores, memory_stores


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
