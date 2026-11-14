"""The one object a run is handed; the host application implements it."""
from __future__ import annotations

import datetime
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import HandleResolver, OpenRetrieval
from tta_backend.harness.storage import Stores
from tta_backend.harness.trace import StepRecorder


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


class EarthdataSession(Protocol):
    """The earthdata service, already bound to the run's user and their credential."""

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's parsed result. A classified failure raises an exception
        carrying a string ``category``."""
        ...


class Compute(Protocol):
    """Runs CPU-bound work off the event loop, under the host's concurrency bound."""

    async def run[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T: ...


@dataclass(frozen=True, kw_only=True)
class Limits:
    """Size and time bounds the host sets for retrievals."""

    retrieval_soft_bytes: int
    retrieval_hard_bytes: int
    max_point_timeseries_days: int
    retrieval_timeout_seconds: float
    poll_min_seconds: float
    poll_max_seconds: float

    def __post_init__(self) -> None:
        if not 0 < self.retrieval_soft_bytes <= self.retrieval_hard_bytes:
            raise ValueError("retrieval byte caps are positive and soft <= hard")
        if self.max_point_timeseries_days < 1:
            raise ValueError("max_point_timeseries_days is at least 1")
        if not 0 < self.poll_min_seconds <= self.poll_max_seconds <= self.retrieval_timeout_seconds:
            raise ValueError("poll bounds are positive and poll_min <= poll_max <= timeout")


@dataclass(frozen=True, kw_only=True)
class RunEnv:
    """What every step of one run is handed: the run context, the host's ports,
    the run id, the handle resolver and the trace recorder. Build it with ``run_env``."""

    context: RunContext
    run_id: str
    resolver: HandleResolver
    recorder: StepRecorder
    earthdata: EarthdataSession
    compute: Compute
    limits: Limits


def run_env(
    context: RunContext,
    *,
    run_id: str,
    payloads: PayloadStore,
    open_retrieval: OpenRetrieval,
    earthdata: EarthdataSession,
    compute: Compute,
    limits: Limits,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunEnv:
    """The environment for a run already started in ``context.storage.traces``."""
    user_id = context.identity.user_id
    return RunEnv(
        context=context,
        run_id=run_id,
        # Built on the context's own registry, so handles minted and resolved
        # here are the ones the rest of the run sees.
        resolver=HandleResolver(context.storage.handles, payloads, open_retrieval),
        recorder=StepRecorder(
            context.storage.traces, user_id, run_id, now=context.clock.now, monotonic=monotonic
        ),
        earthdata=earthdata,
        compute=compute,
        limits=limits,
    )
