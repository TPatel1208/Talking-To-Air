"""The one object a run is handed; the host application implements it."""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Protocol

from tta_backend.harness.storage import Stores


@dataclass(frozen=True)
class Identity:
    user_id: str
    conversation_id: str


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    type: str
    title: str | None = None


class Events(Protocol):
    def progress(self, message: str, *, stage: str | None = None, detail: Any = None) -> None: ...

    def chart(self, payload: dict[str, Any]) -> None: ...

    def question(self, payload: dict[str, Any]) -> None: ...


class Artifacts(Protocol):
    async def store_chart(self, payload: dict[str, Any]) -> ArtifactRef: ...

    async def store_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef: ...


class Credentials(Protocol):
    async def secret(self, connector: str) -> str | None:
        """The user's secret for a connector, resolved now; None when not connected."""
        ...


class Clock(Protocol):
    def today(self) -> datetime.date: ...

    def now(self) -> datetime.datetime:
        """Timezone-aware UTC."""
        ...


@dataclass(frozen=True)
class RunContext:
    identity: Identity
    events: Events
    artifacts: Artifacts
    credentials: Credentials
    storage: Stores
    clock: Clock
