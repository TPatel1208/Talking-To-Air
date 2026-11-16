"""How anything that goes wrong in a capability call becomes a StepFailure.

Every reason is fixed wording filled only from declared facts: the capability's
name, its connector, a parameter's name and its accepted kinds. Exception text
is never used, since it can carry URLs, queries or secrets.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.trace import FailureCategory, StepFailure

_F = FailureCategory

# Keyed by the ``category`` string science code raises with (the earthdata
# error taxonomy). A category not listed here is an internal error.
SOURCE_CATEGORIES: dict[str, tuple[FailureCategory, str]] = {
    "user_input": (_F.BAD_INPUT, "{name} could not use the request as given."),
    "no_data": (_F.NO_DATA, "{name} found no data for the requested area and dates."),
    # The service no longer has something the request named, such as an expired job or granule.
    "not_found": (_F.BAD_INPUT, "{name} referred to something the data service no longer has."),
    "too_large": (_F.REQUEST_TOO_LARGE, "{name} was too large to complete; narrow the area or the dates."),
    "provider_unavailable": (_F.SOURCE_UNAVAILABLE, "{name} could not reach its data source, which is temporarily unavailable."),
    "contract": (_F.INTERNAL_ERROR, "{name} hit an internal error."),
    "unsupported_grid": (_F.UNSUPPORTED_SHAPE, "{name} supports only regular latitude-longitude grids, and this data is on another grid."),
    # Backstop only: a capability is expected to catch these and return needs_input with a Question.
    "variable_choice_required": (_F.BAD_INPUT, "{name} needs a variable chosen from this product."),
    "dimension_choice_required": (_F.BAD_INPUT, "{name} needs a level or other dimension chosen for this product."),
    "token_invalid": (_F.CREDENTIAL_INVALID, "The {connector} credential is invalid. Reconnect it in Settings -> Connectors."),
    "token_expired": (_F.CREDENTIAL_INVALID, "The {connector} credential has expired. Reconnect it in Settings -> Connectors."),
    "eula_not_accepted": (_F.CREDENTIAL_INVALID, "This data requires accepting its license with {connector} before it can be retrieved."),
}

# Used for the credential categories when the capability declares no connector.
NO_CONNECTOR_CATEGORIES: dict[str, tuple[FailureCategory, str]] = {
    "token_invalid": (_F.CREDENTIAL_INVALID, "{name}'s data source refused its credential as invalid."),
    "token_expired": (_F.CREDENTIAL_INVALID, "{name}'s data source reported that its credential has expired."),
    "eula_not_accepted": (_F.CREDENTIAL_INVALID, "This data requires accepting its provider's license before it can be retrieved."),
}

# Keyed by HandleError.reason, for a handle given as a parameter. Resolution
# reads only the record, so eviction shows up when the data is opened.
PARAM_HANDLE_REASONS: dict[str, tuple[FailureCategory, str]] = {
    "not_found": (_F.BAD_INPUT, "{name}: there is no handle for {param}."),
    "wrong_kind": (_F.BAD_INPUT, "{name}: {param} must be a handle of kind {kinds}."),
}

# Keyed by HandleError.reason, for a handle error raised while the capability ran.
BODY_HANDLE_REASONS: dict[str, tuple[FailureCategory, str]] = {
    "not_found": (_F.BAD_INPUT, "{name} needed a handle that does not exist."),
}

EVICTED = (_F.INPUT_EVICTED, "{name} needed data that is no longer stored; re-run the step that made it.")

INTERNAL = (_F.INTERNAL_ERROR, "{name} hit an internal error.")
INVALID_PARAMS = (_F.BAD_INPUT, "The parameters for {name} are not valid: {fields}.")
UNRECORDABLE_PARAMS = (_F.BAD_INPUT, "The parameters for {name} are too large to record; pass data as a handle.")
NOT_CONNECTED = (_F.CREDENTIAL_INVALID, "{name} needs the {connector} connector, which is not connected.")
# The registry keys are registry facts; a user never names one, so this is the caller's mistake.
UNKNOWN_PRODUCT = (_F.BAD_INPUT, "{name} needs a registered product key; call it again with one of: {keys}.")
BAD_RESULT = (_F.INTERNAL_ERROR, "{name} returned a result that does not match its declaration.")
SECRET_IN_OUTPUT = (_F.INTERNAL_ERROR, "{name} produced output holding its connector secret, so it was not stored.")
GRID_OUTSIDE_OPENER = (
    _F.INTERNAL_ERROR,
    "{name} tried to read a grid directly; a grid is read only through the grid opener (grids.open_grid).",
)
UNRECORDABLE_JOB = (_F.INTERNAL_ERROR, "{name} returned a job reference too large or not plain enough to record.")
# Every model for the step failed, after its retries and fallbacks.
MODEL_FAILED = (_F.MODEL_FAILURE, "The language model for {name} did not answer.")
# The model answered twice with something that does not fit the step's declared shape.
MODEL_ANSWER_INVALID = (_F.MODEL_FAILURE, "The language model's answer for {name} did not fit its declared shape.")


class InputEvicted(StepFailure):
    """Data behind ``handle_ids`` was evicted; a runner re-runs the steps that made them.

    The ids are also stored on the failed entry as ``{"evicted": [...]}``.
    """

    def __init__(self, name: str, handle_ids: Iterable[str]) -> None:
        ids = tuple(dict.fromkeys(handle_ids))
        category, wording = EVICTED
        super().__init__(category, wording.format(name=name), result={"evicted": list(ids)})
        self.handle_ids = ids


class JobStillRunning(StepFailure):
    """A two-phase capability's job was still running when ``complete`` stopped waiting.

    Nothing was lost: the job keeps running at the provider. The runner keeps
    the job reference it saved from ``begin`` and calls ``complete`` again
    later with it; ``complete`` never submits. The entry is recorded as a
    retryable ``source_unavailable`` failure holding ``{"job": ..., "still_running": [...]}``.
    """

    def __init__(self, name: str, job: Mapping[str, Any], running: Iterable[str]) -> None:
        pending = list(dict.fromkeys(running))
        super().__init__(
            FailureCategory.SOURCE_UNAVAILABLE,
            f"{name} is still waiting for its retrieval; complete it again later with the saved job.",
            result={"job": dict(job), "still_running": pending},
        )
        self.job = dict(job)
        self.still_running = tuple(pending)


class JobFailed(StepFailure):
    """A two-phase capability's job ended at the provider without a result: it
    failed, or the service no longer holds it.

    Completing the same job again cannot succeed, so a retry begins the step
    again with a new ``attempt``, which submits a new job. The entry is
    recorded as a retryable ``source_unavailable`` failure holding
    ``{"job": ..., <state>: [job ids]}``.
    """

    def __init__(self, reason: str, *, job: Mapping[str, Any], state: str, job_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(job_ids))
        super().__init__(FailureCategory.SOURCE_UNAVAILABLE, reason, result={"job": dict(job), state: ids})
        self.job = dict(job)
        self.job_ids = tuple(ids)


def failure(rule: tuple[FailureCategory, str], **facts: str) -> StepFailure:
    category, wording = rule
    return StepFailure(category, wording.format(**facts))


def from_param_handle(error: HandleError, *, name: str, param: str, kinds: Iterable[str]) -> StepFailure:
    rule = PARAM_HANDLE_REASONS.get(error.reason, INTERNAL)
    return failure(rule, name=name, param=param, kinds=" or ".join(kinds))


def from_exception(exc: BaseException, *, name: str, connector: str | None) -> StepFailure:
    """The failure for an exception raised while a capability ran."""
    if isinstance(exc, StepFailure):
        return exc
    if isinstance(exc, HandleError):
        if exc.reason == "evicted":
            return InputEvicted(name, () if exc.handle_id is None else (exc.handle_id,))
        return failure(BODY_HANDLE_REASONS.get(exc.reason, INTERNAL), name=name)
    category = getattr(exc, "category", None)
    if isinstance(category, str):
        if connector is None and category in NO_CONNECTOR_CATEGORIES:
            return failure(NO_CONNECTOR_CATEGORIES[category], name=name)
        return failure(SOURCE_CATEGORIES.get(category, INTERNAL), name=name, connector=connector or "source")
    return failure(INTERNAL, name=name)
