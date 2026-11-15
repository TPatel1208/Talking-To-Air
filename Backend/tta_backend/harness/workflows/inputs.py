"""Checking a workflow's inputs before any step runs, and resolving area wording to a handle.

``resolve_inputs`` returns the values the steps read, or the problem that ends
the run before it starts:

- a required input is missing: ``needs_input``, listing the ``ask`` of every
  missing input;
- an input has the wrong type or is out of range: ``failed``, category
  ``bad_input``, with wording that names the input and never repeats its value;
- area wording that ``sources.areas.resolve`` cannot resolve: the run ends
  with that capability's own status and wording.

An ``area`` input is a polygons handle id, or the place in the user's words
(any other text, or ``{"place": "..."}``). Wording is resolved through the
normal invoke path as step ``input.<name>``, so it is a traced ``use`` entry;
the steps read the resulting handle id.

An input given as None, or not given, takes its default. One declared
``default: null`` is therefore None and is never asked for; an area input
that is None is not resolved.
"""
from __future__ import annotations

import datetime
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import JsonValue

from tta_backend.harness.capabilities.invoke import invoke
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.sources.areas import area_facts
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleKind, is_handle_id
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.trace import FailureCategory, StepFailure
from tta_backend.harness.workflows.expressions import NAME
from tta_backend.harness.workflows.spec import QUANTITY_NAME, RESERVED_STEP_PREFIX, Input, WorkflowSpec

AREA_CAPABILITY = "sources.areas.resolve"
MAX_TEXT_CHARS = 500
MAX_MONITOR_CHARS = 100
MAX_DATES = 366
_NAMED_CHARS = 40

ProblemStatus = Literal["needs_input", "failed", "insufficient_evidence"]


@dataclass(frozen=True)
class ResolvedInputs:
    """Every input's value as the steps read it, and what resolving them disclosed.

    ``names`` holds an area input's resolved name, which text such as a
    method sentence shows in place of the handle id.
    """

    values: dict[str, JsonValue]
    disclosures: tuple[str, ...] = ()
    names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MissingInput:
    name: str
    ask: str


@dataclass(frozen=True)
class InputProblem:
    """Why the run ends before any step. ``question`` is plain JSON for a needs_input result."""

    status: ProblemStatus
    reason: str
    category: FailureCategory | None = None
    question: dict[str, Any] | None = None
    missing: tuple[MissingInput, ...] = field(default_factory=tuple)


class _Bad(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def input_step_id(name: str) -> str:
    """The trace step id under which an input is resolved."""
    return f"{RESERVED_STEP_PREFIX}.{name}"


def _missing(spec: WorkflowSpec, raw: Mapping[str, Any]) -> tuple[MissingInput, ...]:
    return tuple(
        MissingInput(name, declared.ask or name)
        for name, declared in spec.inputs.items()
        if not declared.optional and raw.get(name) is None
    )


def _named(name: str) -> str:
    return name if NAME.fullmatch(name) and len(name) <= _NAMED_CHARS else "an unexpected input"


def _iso_date(value: Any) -> str | None:
    if isinstance(value, datetime.datetime):
        return None
    if isinstance(value, datetime.date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _number(name: str, declared: Input, value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _Bad(f"Input {name} must be a number.")
    if declared.min is not None and value < declared.min:
        raise _Bad(f"Input {name} must be at least {declared.min:g}.")
    if declared.max is not None and value > declared.max:
        raise _Bad(f"Input {name} must be at most {declared.max:g}.")
    return value


def _text(name: str, declared: Input, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _Bad(f"Input {name} must be text.")
    low = 1 if declared.min is None else declared.min
    high = MAX_TEXT_CHARS if declared.max is None else min(declared.max, MAX_TEXT_CHARS)
    if not low <= len(value) <= high:
        raise _Bad(f"Input {name} must be {low:g} to {high:g} characters long.")
    return value


def _date_range(name: str, value: Any) -> dict[str, JsonValue]:
    wording = f"Input {name} must be a date range with an ISO start date and end date."
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise _Bad(wording)
    start, end = _iso_date(value["start"]), _iso_date(value["end"])
    if start is None or end is None:
        raise _Bad(wording)
    if start > end:
        raise _Bad(f"Input {name} ends before it starts.")
    return {"start": start, "end": end}


def _date_list(name: str, value: Any) -> list[JsonValue]:
    wording = f"Input {name} must be a list of ISO dates."
    if not isinstance(value, list | tuple) or not value:
        raise _Bad(wording)
    if len(value) > MAX_DATES:
        raise _Bad(f"Input {name} lists more than {MAX_DATES} dates.")
    days: list[JsonValue] = []
    for item in value:
        day = _iso_date(item)
        if day is None:
            raise _Bad(wording)
        if day not in days:
            days.append(day)
    return days


async def _handle(env: RunEnv, name: str, value: Any, kind: HandleKind) -> str:
    wording = f"Input {name} must be a {kind} handle from this conversation's results."
    if not isinstance(value, str) or not is_handle_id(value):
        raise _Bad(wording)
    try:
        await env.resolver.get(env.context.identity.user_id, value, kind)
    except HandleError:
        raise _Bad(wording) from None
    return value


def _plain(
    name: str, declared: Input, value: Any, quantities: Collection[str] | None
) -> JsonValue:
    """A checked value for every type except area and handle, which need the run."""
    kind = declared.type
    if kind == "quantity":
        if not isinstance(value, str) or not QUANTITY_NAME.fullmatch(value) or (
            quantities is not None and value not in quantities
        ):
            raise _Bad(f"Input {name} must be a quantity the dataset registry knows, like no2_tropospheric_column.")
        return value
    if kind == "date":
        day = _iso_date(value)
        if day is None:
            raise _Bad(f"Input {name} must be an ISO date, like 2024-06-01.")
        return day
    if kind == "date_range":
        return _date_range(name, value)
    if kind == "date_list":
        return _date_list(name, value)
    if kind == "choice":
        options = declared.options or ()
        if not isinstance(value, str) or value not in options:
            raise _Bad(f"Input {name} must be one of: {', '.join(options)}.")
        return value
    if kind == "number":
        return _number(name, declared, value)
    if kind == "text":
        return _text(name, declared, value)
    if kind == "monitor":
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_MONITOR_CHARS or any(
            c.isspace() for c in value
        ):
            raise _Bad(f"Input {name} must be a monitor id.")
        return value
    raise AssertionError(f"unhandled input type {kind}")


def _area_wording(name: str, value: Any) -> str | None:
    """The place in the user's words, or None when the value is a handle id."""
    if isinstance(value, Mapping):
        if set(value) != {"place"} or not isinstance(value["place"], str):
            raise _Bad(f"Input {name} must be an area handle or a place, like {{place: New Jersey}}.")
        value = value["place"]
    elif not isinstance(value, str):
        raise _Bad(f"Input {name} must be an area handle or a place, like {{place: New Jersey}}.")
    if not value.strip():
        raise _Bad(f"Input {name} must name a place.")
    if len(value) > MAX_TEXT_CHARS:
        raise _Bad(f"Input {name} is longer than {MAX_TEXT_CHARS} characters.")
    return None if is_handle_id(value) else value


async def _area_name(env: RunEnv, name: str, handle_id: str) -> str:
    """The resolved name of an area handle given as input ``name``."""
    wording = f"Input {name} must be an area handle from this conversation's results, or a place."
    try:
        record = await env.resolver.get(env.context.identity.user_id, handle_id, "polygons")
        return area_facts(record).name
    except (HandleError, StepFailure):
        raise _Bad(wording) from None


async def resolve_inputs(
    env: RunEnv,
    spec: WorkflowSpec,
    raw: Mapping[str, Any],
    capabilities: CapabilityRegistry,
    *,
    quantities: Collection[str] | None = None,
) -> ResolvedInputs | InputProblem:
    """The inputs the steps read, or the problem that ends the run before it starts.

    ``quantities`` is the set of canonical quantities the dataset registry
    knows; when None, a quantity is checked for its form only. Area wording is
    resolved last, so no entry is written when another input is wrong.
    """
    missing = _missing(spec, raw)
    if missing:
        asks = " ".join(item.ask for item in missing)
        return InputProblem(
            status="needs_input",
            reason=f"The workflow needs more before it can run: {asks}",
            question={"kind": "inputs", "inputs": [{"name": m.name, "ask": m.ask} for m in missing]},
            missing=missing,
        )

    values: dict[str, JsonValue] = {}
    names: dict[str, str] = {}
    places: dict[str, str] = {}
    try:
        for given in raw:
            if given not in spec.inputs:
                raise _Bad(f"Input {_named(str(given))} is not an input of this workflow.")
        for name, declared in spec.inputs.items():
            value = raw.get(name)
            if value is None:
                values[name] = declared.default
                continue
            if declared.type == "area":
                wording = _area_wording(name, value)
                if wording is None:
                    names[name] = await _area_name(env, name, str(value))
                    values[name] = str(value)
                else:
                    places[name] = wording
            elif declared.type == "handle":
                assert declared.kind is not None
                values[name] = await _handle(env, name, value, declared.kind)
            else:
                values[name] = _plain(name, declared, value, quantities)
    except _Bad as bad:
        return InputProblem(status="failed", reason=bad.reason, category=FailureCategory.BAD_INPUT)

    disclosures: list[str] = []
    capability = capabilities.get(AREA_CAPABILITY)
    for name, wording in places.items():
        try:
            result = await invoke(env, capability, {"place": wording}, step=input_step_id(name))
        except StepFailure as failure:
            return InputProblem(status="failed", reason=failure.reason, category=failure.category)
        if result.status == "needs_input":
            # The place was not found or was refused: the user is asked again, and nothing runs.
            assert result.question is not None
            return InputProblem(
                status="needs_input",
                reason=result.question.prompt,
                question={**result.question.model_dump(mode="json"), "input": name},
            )
        if result.status == "insufficient_evidence":
            return InputProblem(status="insufficient_evidence", reason=result.reason or "", category=FailureCategory.NO_DATA)
        assert result.handle is not None
        values[name] = result.handle
        resolved_name = (result.summary or {}).get("name")
        if isinstance(resolved_name, str):
            names[name] = resolved_name
        disclosures.extend(result.disclosures)
    return ResolvedInputs(
        values={name: values[name] for name in spec.inputs}, disclosures=tuple(disclosures), names=names,
    )
