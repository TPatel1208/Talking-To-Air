"""The one path every capability call takes: validate, check, run, record.

Each call writes one ``use`` trace entry, except a two-phase capability,
which writes two: ``begin`` writes a ``waiting`` entry holding the job
reference, then ``complete`` writes the outcome, its params carrying the job
under ``_job`` so the two entries pair up. ``invoke`` on a two-phase
capability runs ``begin`` then ``complete``, so the job reference is durable
before the provider is asked for the result. A ``begin`` that settles the call
without submitting anything (a question for the user, or no data) returns a
result instead of a job; that is its only entry and nothing is completed.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.call import Call, Resolved
from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.result import CapabilityResult, JobRef
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.trace import (
    UNRECORDABLE_REASON,
    FailureCategory,
    InvalidTrace,
    StepFailure,
    StepScope,
    serialize_json,
)

P = TypeVar("P", bound=BaseModel)
T = TypeVar("T")

logger = logging.getLogger(__name__)

# The key a complete entry's params hold the job reference under; a pydantic
# field cannot start with an underscore, so it never collides with a parameter.
JOB_PARAM = "_job"


class _Insufficient(StepFailure):
    """Carries an insufficient_evidence result past the recorder, which writes
    it as a no_data failure holding the result."""

    def __init__(self, final: CapabilityResult) -> None:
        super().__init__(FailureCategory.NO_DATA, final.reason or "", result=final.recorded())
        self.final = final


async def invoke(
    env: RunEnv,
    capability: Capability[P],
    params: Mapping[str, Any],
    *,
    step: str | None = None,
    parent_seq: int | None = None,
    attempt: int = 0,
) -> CapabilityResult:
    """Run a capability to its result; a two-phase one goes through ``begin`` then ``complete``.

    ``attempt`` is as for ``begin``. Raises StepFailure, already recorded, when the call fails.
    """
    if capability.two_phase:
        job = await begin(env, capability, params, step=step, parent_seq=parent_seq, attempt=attempt)
        if isinstance(job, CapabilityResult):
            return job
        return await complete(env, capability, params, job, step=step, parent_seq=parent_seq)
    run = capability.run
    assert run is not None

    async def body(call: Call[P], scope: StepScope) -> CapabilityResult:
        return await _finish(call, scope, await run(call))

    try:
        return await _step(env, capability, params, step, parent_seq, body, attempt=attempt)
    except _Insufficient as stop:
        return stop.final


async def begin(
    env: RunEnv,
    capability: Capability[P],
    params: Mapping[str, Any],
    *,
    step: str | None = None,
    parent_seq: int | None = None,
    attempt: int = 0,
) -> JobRef | CapabilityResult:
    """Submit a two-phase capability's work and return the job reference, already saved,
    or the result when the capability settled the call without submitting.

    Beginning the same step again with the same parameters and ``attempt``
    submits nothing new: the session answers the repeated request. A larger
    ``attempt`` is a deliberate resubmission, for a job that failed or that
    the service no longer holds.

    Raises StepFailure, already recorded, when the call fails.
    """
    if capability.begin is None:
        raise TypeError(f"{capability.name} is not two-phase")
    submit = capability.begin

    async def body(call: Call[P], scope: StepScope) -> JobRef | CapabilityResult:
        job = await submit(call)
        if isinstance(job, CapabilityResult):
            return await _finish(call, scope, job)
        refusal = _check_job(capability, job, call.recorded)
        if refusal is not None:
            raise refusal
        call.refuse_secret(job)
        scope.status = "waiting"
        scope.result = {"job": job}
        return job

    try:
        return await _step(env, capability, params, step, parent_seq, body, attempt=attempt)
    except _Insufficient as stop:
        return stop.final


async def complete(
    env: RunEnv,
    capability: Capability[P],
    params: Mapping[str, Any],
    job: JobRef,
    *,
    step: str | None = None,
    parent_seq: int | None = None,
) -> CapabilityResult:
    """Finish a two-phase capability from a saved job reference; never submits again.

    Raises StepFailure, already recorded, when the call fails.
    """
    if capability.complete is None:
        raise TypeError(f"{capability.name} is not two-phase")
    finish = capability.complete

    async def body(call: Call[P], scope: StepScope) -> CapabilityResult:
        return await _finish(call, scope, await finish(call, job))

    try:
        return await _step(env, capability, params, step, parent_seq, body, job=job)
    except _Insufficient as stop:
        return stop.final


def _recordable(value: Any) -> bool:
    try:
        serialize_json(value, "params")
    except InvalidTrace:
        return False
    return True


def _check_job(capability: Capability[Any], job: Any, recorded: dict[str, JsonValue]) -> StepFailure | None:
    """A refusal unless both entries that hold ``job`` can store it."""
    if not isinstance(job, dict):
        return failures.failure(failures.BAD_RESULT, name=capability.name)
    if not (_recordable({"job": job}) and _recordable({**recorded, JOB_PARAM: job})):
        return failures.failure(failures.UNRECORDABLE_JOB, name=capability.name)
    return None


def _invalid_fields(error: ValidationError, model: type[BaseModel]) -> str:
    # Only declared names reach the wording; an unknown key is never echoed.
    names = []
    for item in error.errors():
        head = item["loc"][0] if item["loc"] else None
        names.append(head if isinstance(head, str) and head in model.model_fields else "an unexpected parameter")
    return ", ".join(dict.fromkeys(names))


def _validate(capability: Capability[P], raw: Any) -> tuple[P | None, dict[str, JsonValue], StepFailure | None]:
    """The parameters, what the trace records for them, and any refusal."""
    name = capability.name
    if not isinstance(raw, Mapping):
        return None, {}, failures.failure(failures.INVALID_PARAMS, name=name, fields="parameters must be an object")
    try:
        params = capability.params.model_validate(dict(raw))
    except ValidationError as error:
        # Only declared fields are recorded; an unknown key can be anything, a secret included.
        declared = {k: v for k, v in raw.items() if k in capability.params.model_fields}
        fields = _invalid_fields(error, capability.params)
        refusal = failures.failure(failures.INVALID_PARAMS, name=name, fields=fields)
        return None, declared if _recordable(declared) else {}, refusal
    dumped = params.model_dump(mode="json")
    if not _recordable(dumped):
        return None, {}, failures.failure(failures.UNRECORDABLE_PARAMS, name=name)
    return params, dumped, None


async def _resolve(env: RunEnv, capability: Capability[P], params: P) -> dict[str, Resolved]:
    user_id = env.context.identity.user_id
    resolved: dict[str, Resolved] = {}
    for name, kinds in capability.handle_params.items():
        value = getattr(params, name)
        try:
            if value is None:
                resolved[name] = None
            elif isinstance(value, str):
                resolved[name] = await env.resolver.get(user_id, value, *kinds)
            else:
                resolved[name] = tuple([await env.resolver.get(user_id, item, *kinds) for item in value])
        except HandleError as error:
            raise failures.from_param_handle(error, name=capability.name, param=name, kinds=kinds) from None
    return resolved


async def _prepare(
    env: RunEnv,
    capability: Capability[P],
    params: P,
    recorded: dict[str, JsonValue],
    scope: StepScope,
    step: str | None,
    attempt: int,
) -> Call[P]:
    resolved = await _resolve(env, capability, params)
    secret = None
    if capability.connector is not None:
        secret = await env.context.credentials.secret(capability.connector) or None
        if secret is None and capability.connector_required:
            raise failures.failure(failures.NOT_CONNECTED, name=capability.name, connector=capability.connector)
    return Call(
        capability=capability, params=params, recorded=recorded, env=env, resolved=resolved,
        secret=secret, seq=scope.seq, step=step, attempt=attempt,
    )


async def _finish(call: Call[Any], scope: StepScope, result: Any) -> CapabilityResult:
    """Check the result against the declaration, fill the method and record it."""
    capability = call.capability
    bad = failures.failure(failures.BAD_RESULT, name=capability.name)
    if not isinstance(result, CapabilityResult):
        raise bad
    if result.kind != (capability.output_kind if result.handle else None):
        raise bad
    if result.status == "done" and capability.output_kind is not None and result.handle is None:
        raise bad
    if result.handle is not None and result.kind is not None:
        # The handle must be one this user can resolve, minted or recorded in this call or earlier.
        try:
            await call.env.resolver.get(call.user_id, result.handle, result.kind)
        except HandleError:
            raise bad from None
    method = capability.method_sentence(call.recorded, result) if result.status == "done" else None
    final = CapabilityResult.model_validate({**result.model_dump(), "method": method})
    call.refuse_secret(final.recorded())
    if final.status == "insufficient_evidence":
        raise _Insufficient(final)
    scope.status = "waiting" if final.status == "needs_input" else "done"
    scope.result = final.recorded()
    return final


def _log_internal(env: RunEnv, capability: Capability[Any], failure: StepFailure, cause: str) -> None:
    # The type name only: an exception's message or arguments can carry URLs or secrets.
    if failure.category is FailureCategory.INTERNAL_ERROR:
        logger.error(
            "capability %s hit an internal error in run %s: %s", capability.name, env.run_id, cause
        )


def _entry_params(
    capability: Capability[Any], recorded: dict[str, JsonValue], job: JobRef | None
) -> tuple[dict[str, JsonValue], StepFailure | None]:
    """What a step's entry records as params, and a refusal when the job cannot be recorded."""
    if job is None:
        return recorded, None
    refusal = _check_job(capability, job, recorded)
    if refusal is not None:
        alone: dict[str, JsonValue] = {JOB_PARAM: job} if isinstance(job, dict) else {}
        return (alone if _recordable(alone) else {}), refusal
    return {**recorded, JOB_PARAM: job}, None


async def _step(
    env: RunEnv,
    capability: Capability[P],
    raw: Mapping[str, Any],
    step: str | None,
    parent_seq: int | None,
    body: Callable[[Call[P], StepScope], Awaitable[T]],
    *,
    job: JobRef | None = None,
    attempt: int = 0,
) -> T:
    params, recorded, refusal = _validate(capability, raw)
    entry_params, job_refusal = _entry_params(capability, recorded, job)
    refusal = refusal or job_refusal
    entered = False
    try:
        async with env.recorder.step(
            kind="use", capability=capability.name, params=entry_params, step=step, parent_seq=parent_seq
        ) as scope:
            entered = True
            if refusal is not None:
                _log_internal(env, capability, refusal, type(refusal).__name__)
                raise refusal
            assert params is not None
            failure: StepFailure | None = None
            try:
                call = await _prepare(env, capability, params, recorded, scope, step, attempt)
                env.context.events.progress(capability.label, stage=capability.name)
                outcome = await body(call, scope)
            except StepFailure as own:
                _log_internal(env, capability, own, type(own).__name__)
                raise
            except Exception as exc:
                failure = failures.from_exception(exc, name=capability.name, connector=capability.connector)
                _log_internal(env, capability, failure, type(exc).__name__)
            # Raised outside the handler, so the original exception travels no further.
            if failure is not None:
                raise failure from None
    except InvalidTrace:
        if not entered:
            raise
        # The recorder stored an internal_error entry in place of the outcome.
        _log_internal(env, capability, StepFailure(FailureCategory.INTERNAL_ERROR, ""), "InvalidTrace")
        raise StepFailure(FailureCategory.INTERNAL_ERROR, UNRECORDABLE_REASON) from None
    return outcome
