"""What a capability's code is handed for one call."""
from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, JsonValue

from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.result import CapabilityResult, needs_choice
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import Descriptor, HandleKind, HandleRecord, ServicePayload
from tta_backend.harness.trace import StepFailure
from tta_backend.harness.run_context import EarthdataSession, Limits, RunEnv, request_fingerprint

if TYPE_CHECKING:
    from tta_backend.harness.capabilities.definition import Capability

P = TypeVar("P", bound=BaseModel)

# A handle parameter after resolution: one record, several, or None when unset.
Resolved = HandleRecord | tuple[HandleRecord, ...] | None


def contains_secret(value: Any, secret: str | None) -> bool:
    """Whether ``value``'s JSON text holds ``secret``, raw or JSON-escaped."""
    if not secret:
        return False
    text = json.dumps(value, default=str)
    return secret in text or json.dumps(secret)[1:-1] in text


class Call(Generic[P]):
    """Validated parameters, the resolved input handles and the run's ports.

    Every input handle was checked for this user and its declared kinds before
    the capability's code runs.
    """

    def __init__(
        self,
        *,
        capability: Capability[P],
        params: P,
        recorded: dict[str, JsonValue],
        env: RunEnv,
        resolved: Mapping[str, Resolved],
        secret: str | None,
        seq: int,
        step: str | None = None,
        attempt: int = 0,
    ) -> None:
        self.capability = capability
        self.params = params
        # What the trace records as this call's parameters.
        self.recorded = recorded
        self.env = env
        self._resolved = dict(resolved)
        self._secret = secret
        self._seq = seq
        self._step = step
        self._attempt = attempt

    def idempotency_key(self, tool: str, arguments: dict[str, Any]) -> str:
        """The key a source passes with one submission, so a repeat of the
        same request is recognised. It covers the run, the step id (or the
        trace seq when the step has no id), the call's attempt, and a digest
        of ``tool`` and ``arguments``: a different request never shares a key."""
        where = f"step:{self._step}" if self._step is not None else f"seq:{self._seq}"
        request = request_fingerprint(tool, arguments)
        text = f"{self.env.run_id}/{where}/attempt:{self._attempt}/{request}"
        return f"tta-{hashlib.sha256(text.encode()).hexdigest()[:32]}"

    @property
    def user_id(self) -> str:
        return self.env.context.identity.user_id

    @property
    def today(self) -> datetime.date:
        return self.env.context.clock.today()

    def _not_connected(self) -> StepFailure:
        connector = str(self.capability.connector)
        return failures.failure(failures.NOT_CONNECTED, name=self.capability.name, connector=connector)

    @property
    def earthdata(self) -> EarthdataSession:
        """The earthdata session. Raises the not-connected failure when the
        capability's connector is declared but not connected."""
        if self.capability.connector is not None and self._secret is None:
            raise self._not_connected()
        return self.env.earthdata

    @property
    def limits(self) -> Limits:
        return self.env.limits

    @property
    def secret(self) -> str:
        """The declared connector's secret, resolved for this call; never store it."""
        if self.capability.connector is None:
            raise RuntimeError(f"{self.capability.name} declares no connector")
        if self._secret is None:
            raise self._not_connected()
        return self._secret

    def handle(self, name: str) -> HandleRecord:
        """The record behind a single-handle parameter."""
        value = self._resolved[name]
        if not isinstance(value, HandleRecord):
            raise TypeError(f"{name} is not a single handle")
        return value

    def handles(self, name: str) -> tuple[HandleRecord, ...]:
        """The records behind a handle parameter, whether one, several or unset."""
        value = self._resolved[name]
        if value is None:
            return ()
        return (value,) if isinstance(value, HandleRecord) else value

    def input_ids(self) -> tuple[str, ...]:
        """Every input handle id, in parameter order, once each."""
        ids = (record.id for name in self._resolved for record in self.handles(name))
        return tuple(dict.fromkeys(ids))

    async def open(self, name: str) -> Any:
        """The data behind a single-handle parameter. Raises InputEvicted when
        its data is no longer stored. A grid is refused: read it through
        ``grids.open_grid``, which masks it."""
        [data] = await self._open_records((self.handle(name),))
        return data

    async def open_each(self, name: str) -> list[Any]:
        """The data behind each handle of a parameter; InputEvicted names every evicted one. A grid is refused."""
        return await self._open_records(self.handles(name))

    async def _open_records(self, records: Iterable[HandleRecord]) -> list[Any]:
        records = tuple(records)
        if any(record.kind == "grid" for record in records):
            raise failures.failure(failures.GRID_OUTSIDE_OPENER, name=self.capability.name)
        opened: list[Any] = []
        evicted: list[str] = []
        for record in records:
            try:
                opened.append(await self.env.resolver.open(self.user_id, record.id, record.kind))
            except HandleError as error:
                if error.reason != "evicted":
                    raise
                evicted.append(record.id)
        if evicted:
            raise failures.InputEvicted(self.capability.name, evicted)
        return opened

    async def compute[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T:
        """Run CPU-bound work off the event loop."""
        return await self.env.compute.run(fn, *args, **kwargs)

    def progress(self, message: str, *, detail: Any = None) -> None:
        """A progress line for this call, under its run; a workflow step's carries ``detail.step``."""
        if self._step is not None and (detail is None or isinstance(detail, dict)):
            detail = {**(detail or {}), "step": self._step}
        self.env.context.events.progress(message, stage=self.capability.name, detail=detail, run_id=self.env.run_id)

    async def mint(
        self,
        kind: HandleKind,
        data: Any,
        descriptor: Descriptor,
        *,
        parents: tuple[str, ...] | None = None,
        parameters: dict[str, JsonValue] | None = None,
    ) -> HandleRecord:
        """A derived handle made by this capability in this run. Its parents
        default to every input handle, its parameters to the recorded ones."""
        parameters = self.recorded if parameters is None else parameters
        self.refuse_secret(parameters, descriptor.model_dump(mode="json"))
        context = self.env.context
        return await self.env.resolver.mint(
            user_id=self.user_id,
            conversation_id=context.identity.conversation_id,
            kind=kind,
            data=data,
            descriptor=descriptor,
            operation=self.capability.name,
            created_at=context.clock.now(),
            parents=self.input_ids() if parents is None else parents,
            parameters=parameters,
            run_id=self.env.run_id,
        )

    async def record_retrieval(self, handle_id: str, kind: HandleKind, descriptor: Descriptor) -> HandleRecord:
        """Record a handle the earthdata service returned, so this user can resolve it."""
        self.refuse_secret(handle_id, descriptor.model_dump(mode="json"))
        context = self.env.context
        record = HandleRecord(
            id=handle_id,
            kind=kind,
            origin="retrieval",
            user_id=self.user_id,
            conversation_id=context.identity.conversation_id,
            descriptor=descriptor,
            created_at=context.clock.now(),
            payload=ServicePayload(),
        )
        return await context.storage.handles.record_retrieval(record)

    def refuse_secret(self, *values: Any) -> None:
        """Refuse, as an internal error, any value whose JSON text holds the connector secret."""
        if contains_secret(values, self._secret):
            raise failures.failure(failures.SECRET_IN_OUTPUT, name=self.capability.name)

    def ask(self, prompt: str, options: Iterable[JsonValue], *, many: bool = False) -> CapabilityResult:
        """A ``needs_input`` result offering ``options``, for example the
        candidates of a variable or dimension the source needs chosen."""
        return needs_choice(prompt, options, many=many)

    def done(self, record: HandleRecord | None = None, **fields: Any) -> CapabilityResult:
        """A ``done`` result for ``record``, taking its unit and product from the descriptor."""
        if record is not None:
            fields = {
                "handle": record.id,
                "kind": record.kind,
                "unit": record.descriptor.unit,
                "product": record.descriptor.product,
                **fields,
            }
        return CapabilityResult(status="done", **fields)
