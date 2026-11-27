"""One langchain tool per capability, each call going through ``invoke`` under the task's explore entry.

The model sees small JSON: a result's status, handle and kind, summary,
values and disclosures, or a failure's category, fixed reason and whether a
retry can help. It never sees data or exception text. Two outcomes end the
task instead of returning to the model: a question for the user
(``needs_input``) and a retrieval still running when the wait ran out; both
raise ``TaskStop``.

Some calls are refused before ``invoke`` with a fixed result: one naming a
handle outside the task, one setting a workflow-only parameter, and one
repeating a call that already failed, came back insufficient_evidence or was
refused for a foreign handle in the task, until a success of that call. A
refused call writes no trace entry.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.failures import JobStillRunning
from tta_backend.harness.capabilities.instruments import explore_refusal
from tta_backend.harness.capabilities.invoke import invoke
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.handles.model import is_handle_id
from tta_backend.harness.knowledge import Note, NoteCatalogue
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.specialists.prompt import note_text, notes_in_play
from tta_backend.harness.specialists.state import TaskState
from tta_backend.harness.trace import FailureCategory, StepFailure

# What providers accept as a tool name.
TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# Argument names langchain keeps for itself and would not pass to the tool.
RESERVED_PARAMS = frozenset({"config", "run_manager"})
# Parameters only the workflow runner sets; specialists never see or set them.
WORKFLOW_ONLY_PARAMS = frozenset({"approved"})

_F = FailureCategory
# Whether retrying can help after a failure of each category. For
# source_unavailable the same call may succeed later, if the capability is
# retry_safe; for input_evicted the step that made the input is re-run, which
# only a call of this task can be.
RETRY_HELPS: Mapping[FailureCategory, bool] = {
    _F.SOURCE_UNAVAILABLE: True,
    _F.INPUT_EVICTED: True,
    _F.BAD_INPUT: False,
    _F.NO_DATA: False,
    _F.REQUEST_TOO_LARGE: False,
    _F.SOURCE_DOWN: False,
    _F.CREDENTIAL_INVALID: False,
    _F.UNSUPPORTED_SHAPE: False,
    _F.INTERNAL_ERROR: False,
    _F.MODEL_FAILURE: False,
    _F.BUDGET_EXHAUSTED: False,
    _F.CANCELLED: False,
}
# Categories after which the identical call may succeed when repeated; it runs at most twice.
REPEATABLE = frozenset({_F.SOURCE_UNAVAILABLE})
# The category a repeat names when the earlier call was refused for a foreign handle.
FOREIGN_CATEGORY = "foreign_handle"

FOREIGN_HANDLE = "Handle {handle} isn't part of this task; use a handle the task gave you or one a call returned."
FOREIGN_VALUE = "A value given for {param} isn't a handle of this task; use a handle the task gave you or one a call returned."
WORKFLOW_PARAM = "{param} is set only by validated workflows; call {tool} again without it."
REPEATED_CALL = (
    "This exact call already failed in this task with {category}; it will fail the same way. "
    "Change the request or answer with what you have."
)
QUESTION_REASON = "{name} needs an answer from the user before the exploration can continue."
JOB_WAITING = (
    "{name} is still waiting for its retrieval, which outlasted the wait; the job reference is saved in the "
    "trace. Ask again later to collect it."
)


class TaskStop(Exception):
    """A call that ends the task: a question for the user, or a retrieval still running."""

    def __init__(
        self, status: Literal["needs_input", "waiting"], reason: str, question: dict[str, Any] | None = None
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.question = question


class HandleRefused(Exception):
    """A call naming a handle outside the task's allowlist, refused before ``invoke``."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def call_key(capability: Capability[Any], params: Mapping[str, Any]) -> str:
    """The capability's name and its parameters as canonical JSON with sorted keys.

    Parameters are validated and dumped as ``invoke`` records them, so defaults,
    ``1`` and ``1.0``, or two spellings of a date make the same key; parameters
    that do not validate are keyed as given.
    """
    try:
        values: Any = capability.params.model_validate(dict(params)).model_dump(mode="json")
    except (ValidationError, ValueError, TypeError):
        values = params
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{capability.name} {canonical}"


def retry_helps(capability: Capability[Any], category: FailureCategory, *, repeated: bool = False) -> bool:
    """Whether retrying can help after ``capability`` failed with ``category``.

    A repeatable category helps only for a ``retry_safe`` capability, and not
    once the identical call has already been repeated.
    """
    if category in REPEATABLE:
        return capability.retry_safe and not repeated
    return RETRY_HELPS.get(category, False)


def tool_name(capability: Capability[Any]) -> str:
    """The capability's name with each dot written as two underscores."""
    name = capability.name.replace(".", "__")
    if not TOOL_NAME.fullmatch(name):
        raise ValueError(f"{capability.name} does not make a valid tool name")
    return name


def tool_schema(capability: Capability[Any]) -> dict[str, Any]:
    """The capability's JSON schema without its workflow-only parameters, with
    each handle parameter's kinds said in its description.

    ``x-handle-kinds`` is moved into the description, since some providers
    refuse keywords they do not know.
    """
    schema = copy.deepcopy(capability.schema())
    properties = schema.get("properties", {})
    for name in WORKFLOW_ONLY_PARAMS & set(properties):
        del properties[name]
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name not in WORKFLOW_ONLY_PARAMS]
    for name, kinds in capability.handle_params.items():
        prop = properties[name]
        prop.pop("x-handle-kinds", None)
        hint = f"A handle id of kind {' or '.join(kinds)}."
        prop["description"] = f"{prop['description']} {hint}" if prop.get("description") else hint
    return schema


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def result_view(result: CapabilityResult, notes: Sequence[Note] = ()) -> dict[str, Any]:
    """What the model is shown of a result: everything recorded but the method and question, then new notes."""
    view = result.recorded()
    view.pop("method", None)
    view.pop("question", None)
    if notes:
        view["notes"] = [note_text(note) for note in notes]
    return view


def failure_view(failure: StepFailure, retry: bool) -> dict[str, Any]:
    return {"status": "failed", "category": failure.category.value, "reason": failure.reason, "retry": retry}


@dataclass(frozen=True)
class Called:
    result: CapabilityResult
    notes: tuple[Note, ...]


class Caller:
    """Runs capability calls for one task: the allowlist check, ``invoke``, and what the task keeps."""

    def __init__(
        self, env: RunEnv, state: TaskState, notes: NoteCatalogue, parent_seq: int, *, instruments: tuple[str, ...] = (),
    ) -> None:
        self._env = env
        self._instruments = instruments
        self._state = state
        self._notes = notes
        self._parent_seq = parent_seq
        # Each call that failed or was refused for a foreign handle: its category and how often.
        self._failed: dict[str, tuple[str, int]] = {}

    def _refuse_foreign(self, capability: Capability[Any], params: Mapping[str, Any]) -> None:
        foreign = self._state.foreign(capability, params)
        if not foreign:
            return
        self._state.refusals += 1
        handle = foreign[0]
        if is_handle_id(handle):
            raise HandleRefused(FOREIGN_HANDLE.format(handle=handle))
        param = next(name for name in capability.handle_params if _names(params.get(name), handle))
        raise HandleRefused(FOREIGN_VALUE.format(param=param))

    async def call(self, capability: Capability[Any], params: Mapping[str, Any]) -> Called:
        """Run one call. Raises HandleRefused, TaskStop, or StepFailure (already recorded)."""
        self._refuse_foreign(capability, params)
        self._state.calls += 1
        try:
            result = await invoke(self._env, capability, params, parent_seq=self._parent_seq)
        except JobStillRunning:
            # Calls made in the same turn may still finish, and their handles are kept: wasted work, not wrong data.
            raise TaskStop("waiting", JOB_WAITING.format(name=capability.name)) from None
        self._state.take(result)
        record = None
        if result.handle is not None:
            record = await self._env.resolver.get(self._env.context.identity.user_id, result.handle)
            self._state.add_minted(record)
        if result.status == "needs_input":
            # As for a job still running, calls made in the same turn may still finish and keep their handles.
            assert result.question is not None
            question = result.question.model_dump(mode="json")
            raise TaskStop("needs_input", QUESTION_REASON.format(name=capability.name), question)
        pollutant = params.get("pollutant") if "pollutant" in capability.params.model_fields else None
        found = notes_in_play(
            self._notes,
            [record] if record is not None else [],
            datasets=[result.product] if result.product else [],
            pollutants=[pollutant] if isinstance(pollutant, str) else [],
        )
        fresh = tuple(note for note in found if note.id not in self._state.shown_notes)
        self._state.shown_notes.update(note.id for note in fresh)
        return Called(result, fresh)

    def _refused(self, message: str) -> str:
        self._state.refusals += 1
        return _dumps({"status": "refused", "reason": message})

    def _repeat(self, capability: Capability[Any], key: str) -> str | None:
        """The fixed reason for a call that already failed in this task, or None when it may run."""
        earlier = self._failed.get(key)
        if earlier is None:
            return None
        category, times = earlier
        if category in REPEATABLE and capability.retry_safe and times < 2:
            return None
        return REPEATED_CALL.format(category=category)

    def _failed_again(self, key: str, category: FailureCategory) -> int:
        """Record a failure of the call ``key``; how often it has now failed since its last success."""
        times = self._failed.get(key, ("", 0))[1] + 1
        self._failed[key] = (category.value, times)
        return times

    def _retry(self, capability: Capability[Any], failure: StepFailure, times: int) -> bool:
        helps = retry_helps(capability, failure.category, repeated=times > 1)
        if helps and failure.category is FailureCategory.INPUT_EVICTED:
            evicted = failure.result.get("evicted") if isinstance(failure.result, dict) else None
            if not isinstance(evicted, list) or not evicted:
                return False
            # Re-running the step that made an input helps only when this task made it.
            return all(self._state.made_here(handle) for handle in evicted)
        return helps

    async def tool_result(self, capability: Capability[Any], params: Mapping[str, Any]) -> str:
        """The JSON the model sees for one call; ``TaskStop`` ends the task instead."""
        workflow_only = sorted(WORKFLOW_ONLY_PARAMS & set(params))
        if workflow_only:
            return self._refused(WORKFLOW_PARAM.format(param=workflow_only[0], tool=tool_name(capability)))
        other = explore_refusal(capability.name, params, self._instruments)
        if other is not None:
            return self._refused(other)
        key = call_key(capability, params)
        repeat = self._repeat(capability, key)
        if repeat is not None:
            return self._refused(repeat)
        try:
            called = await self.call(capability, params)
        except HandleRefused as refused:
            self._failed[key] = (FOREIGN_CATEGORY, 1)
            return _dumps({"status": "refused", "reason": refused.message})
        except StepFailure as failure:
            times = self._failed_again(key, failure.category)
            return _dumps(failure_view(failure, self._retry(capability, failure, times)))
        view = result_view(called.result, called.notes)
        if called.result.status == "insufficient_evidence":
            # No data for this request; the same request finds none again.
            self._failed_again(key, FailureCategory.NO_DATA)
            view["retry"] = False
        else:
            self._failed.pop(key, None)
        return _dumps(view)


def _names(value: Any, handle: str) -> bool:
    return value == handle or (isinstance(value, list | tuple) and handle in value)


def capability_tool(capability: Capability[Any], caller: Caller) -> BaseTool:
    """The tool for ``capability``. A ``workflow_only`` capability is never wrapped."""
    if capability.workflow_only:
        raise ValueError(f"{capability.name} runs only in validated workflows")
    fields = capability.params.model_fields
    reserved = RESERVED_PARAMS & set(fields)
    if reserved:
        raise ValueError(f"{capability.name} has parameters a tool cannot receive: {sorted(reserved)}")
    hidden = sorted(name for name in WORKFLOW_ONLY_PARAMS & set(fields) if fields[name].is_required())
    if hidden:
        raise ValueError(f"{capability.name} requires workflow-only parameters: {hidden}")

    async def run(**params: Any) -> str:
        return await caller.tool_result(capability, params)

    return StructuredTool.from_function(
        coroutine=run,
        name=tool_name(capability),
        description=capability.description,
        args_schema=tool_schema(capability),
    )


def capability_tools(capabilities: Sequence[Capability[Any]], caller: Caller) -> list[BaseTool]:
    tools = [capability_tool(capability, caller) for capability in capabilities]
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("two capabilities make the same tool name")
    return tools
