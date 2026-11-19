"""The one object a run is handed; the host application implements it."""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import HandleResolver, OpenRetrieval
from tta_backend.harness.storage import Stores
from tta_backend.harness.trace import FailureCategory, StepFailure, StepRecorder

if TYPE_CHECKING:
    from tta_backend.harness.models import ModelChoice, ModelRole, Models


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
    def progress(
        self, message: str, *, stage: str | None = None, detail: Any = None, run_id: str | None = None
    ) -> None:
        """A progress line. ``run_id`` names the run the step belongs to; None
        for progress that belongs to no run."""
        ...

    def chart(self, payload: dict[str, Any]) -> None: ...

    def question(self, payload: dict[str, Any]) -> None: ...

    def text(self, content: str) -> None:
        """The next piece of the answer's text. Optional: a host without it
        gets no streamed text, and callers check for it before calling."""
        ...


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


# Whether a connector's service can be reached right now.
Reachable = Callable[[str], bool]


def all_reachable(connector: str) -> bool:
    return True


@dataclass(frozen=True)
class RunContext:
    """``reachable`` says whether a connector's service is up; work that needs
    one that is not is refused before any run starts."""

    identity: Identity
    events: Events
    artifacts: Artifacts
    credentials: Credentials
    storage: Stores
    clock: Clock
    reachable: Reachable = all_reachable


class EarthdataSession(Protocol):
    """The earthdata service, already bound to the run's user and their credential."""

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """The tool's parsed result. A classified failure raises an exception
        carrying a string ``category``.

        A call given an ``idempotency_key`` it has already completed with the
        same tool and arguments returns that call's result again instead of
        repeating it at the provider. The same key with a different tool or
        arguments is refused with category ``contract``. The session may
        forget old keys, so a key protects repeats within a run."""
        ...


def request_fingerprint(tool: str, arguments: dict[str, Any]) -> str:
    """A digest of the canonical JSON of ``tool`` and ``arguments``; equal for equal requests."""
    canonical = json.dumps(
        {"tool": tool, "arguments": arguments}, sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyKeyReused(Exception):
    """A key already answered was presented with a different request; a contract failure."""

    category = "contract"

    def __init__(self) -> None:
        super().__init__("An idempotency key was reused for a different request.")


class AnsweredRequests:
    """What a session answered for each idempotency key, for at most ``limit``
    keys, the least recently used forgotten first."""

    def __init__(self, limit: int = 256) -> None:
        if limit < 1:
            raise ValueError("limit is at least 1")
        self._limit = limit
        self._answers: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._answers)

    def get(self, key: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """The answer already given for ``key``, or None. Raises
        IdempotencyKeyReused when ``key`` was answered for another request."""
        held = self._answers.get(key)
        if held is None:
            return None
        fingerprint, answer = held
        if fingerprint != request_fingerprint(tool, arguments):
            raise IdempotencyKeyReused()
        self._answers.move_to_end(key)
        return answer

    def put(self, key: str, tool: str, arguments: dict[str, Any], answer: dict[str, Any]) -> None:
        self._answers[key] = (request_fingerprint(tool, arguments), answer)
        self._answers.move_to_end(key)
        while len(self._answers) > self._limit:
            self._answers.popitem(last=False)


@dataclass(frozen=True)
class Region:
    """A place resolved to a footprint: its name, (west, south, east, north)
    bounds and GeoJSON geometry. ``region_type`` says what kind of footprint
    it is, such as polygon, bounding_box or point_buffer."""

    name: str
    bounds: tuple[float, float, float, float]
    geometry: dict[str, Any]
    region_type: str | None = None


PlaceRefusalCode = Literal[
    # A combined area, such as "NY + NJ".
    "member_not_found",  # facts: member
    "member_ambiguous",  # facts: member, state_form, country_form
    "member_too_large",  # facts: member, name, cells, max_cells
    "combined_too_large",  # facts: cells, max_cells
    "empty_member",
    # A distance around a place, such as "within 50 km of Boston".
    "distance_unreadable",
    "distance_unit_missing",  # facts: amount
    "distance_unit_unknown",  # facts: unit
    "centre_is_region",  # facts: centre, radius
    "centre_not_found",  # facts: centre, radius
    "reaches_pole",  # facts: radius, centre, pole, pole_km
    "crosses_antimeridian",  # facts: radius, centre
    "distance_too_large",  # facts: radius, centre, cells, max_cells
]
PLACE_REFUSAL_CODES: frozenset[str] = frozenset(get_args(PlaceRefusalCode))
MAX_FACT_CHARS = 120


class PlaceRefused(Exception):
    """A place request the host understood but refuses, such as a combined area
    with a member it cannot find or a distance too large to process.

    ``facts`` are short strings the host computed or cut from the user's own
    wording, never exception text; each is cut to ``MAX_FACT_CHARS``. An
    unexpected refusal is a plain ``user_input`` failure, so ``category`` is
    kept for that path.
    """

    category = "user_input"

    def __init__(self, code: PlaceRefusalCode, facts: Mapping[str, str] | None = None) -> None:
        if code not in PLACE_REFUSAL_CODES:
            raise ValueError(f"unknown place refusal code {code!r}")
        super().__init__(f"the place request was refused: {code}")
        self.code = code
        self.facts = {
            str(key): " ".join(str(value).split())[:MAX_FACT_CHARS] for key, value in (facts or {}).items()
        }


class Places(Protocol):
    """Resolves a place in the user's words to a region."""

    async def resolve(self, wording: str) -> Region | None:
        """The region ``wording`` names, or None when it names none.

        A request the host understood but refuses raises PlaceRefused with
        its code and facts. Any other malformed request raises an exception
        carrying a string ``category`` of ``user_input``.
        """
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


NO_MODEL_REASON = "No language model is configured for {role} in this run."


@dataclass(frozen=True, kw_only=True)
class RunEnv:
    """What every step of one run is handed: the run context, the host's ports,
    the run id, the handle resolver and the trace recorder. Build it with ``run_env``.

    ``models`` is None for a run whose steps need no language model.
    """

    context: RunContext
    run_id: str
    resolver: HandleResolver
    recorder: StepRecorder
    earthdata: EarthdataSession
    compute: Compute
    limits: Limits
    places: Places
    models: Models | None = None

    def model_for(self, role: ModelRole) -> ModelChoice:
        """The models for ``role``. Raises an internal-error StepFailure when the
        host gave this run no models, or none for that role."""
        choice = None if self.models is None else self.models.for_role(role)
        if choice is None:
            raise StepFailure(FailureCategory.INTERNAL_ERROR, NO_MODEL_REASON.format(role=role))
        return choice


def run_env(
    context: RunContext,
    *,
    run_id: str,
    payloads: PayloadStore,
    open_retrieval: OpenRetrieval,
    earthdata: EarthdataSession,
    compute: Compute,
    limits: Limits,
    places: Places,
    models: Models | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunEnv:
    """The environment for a run already started in ``context.storage.traces``."""
    # Imported here because the jobs module builds on this one.
    from tta_backend.harness.jobs import ReusingEarthdata

    user_id = context.identity.user_id
    # Every run reuses a job the same user already submitted for the same
    # request. A session already wrapped is not wrapped again: both layers
    # would take the same lock for one submit and wait on each other.
    if not isinstance(earthdata, ReusingEarthdata):
        earthdata = ReusingEarthdata(earthdata, context.storage.jobs, user_id, context.clock)
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
        places=places,
        models=models,
    )
