"""What every capability returns: a handle id and small values, never data."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tta_backend.harness.handles.model import HandleKind, is_derived_id, is_handle_id
from tta_backend.harness.trace import MAX_REASON_CHARS, serialize_json

CapabilityStatus = Literal["done", "needs_input", "insufficient_evidence"]
QuestionKind = Literal["pick_one", "pick_many", "confirm_list", "text"]

# What a two-phase capability saves between submitting and completing.
JobRef = dict[str, JsonValue]

_VALUE_NAME = re.compile(r"[a-z][a-z0-9_]*")


class FieldNotFound(LookupError):
    def __init__(self, path: str) -> None:
        super().__init__(f"no field {path!r}")
        self.path = path


def walk_path(source: Any, path: str) -> Any:
    """The value at a dotted ``path`` through dicts and, by index, lists."""
    value = source
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list | tuple) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise FieldNotFound(path)
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Question(_Model):
    """A typed question for the user; pick kinds offer ``options``, confirm_list edits ``items``."""

    kind: QuestionKind
    prompt: str = Field(min_length=1, max_length=500)
    options: tuple[JsonValue, ...] = ()
    items: tuple[JsonValue, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.kind in ("pick_one", "pick_many"):
            if not self.options or self.items:
                raise ValueError(f"a {self.kind} question has options and no items")
        elif self.kind == "confirm_list":
            if self.options:
                raise ValueError("a confirm_list question has items, not options")
        elif self.options or self.items:
            raise ValueError("a text question has no options or items")
        return self


class CapabilityResult(_Model):
    """One capability call's outcome. ``values`` holds named extras a spec reads
    as ``$step.name``; ``method`` is filled by the invoke path, never by the capability."""

    status: CapabilityStatus
    handle: str | None = None
    kind: HandleKind | None = None
    unit: str | None = None
    product: str | None = None
    quality: dict[str, JsonValue] | None = None
    summary: dict[str, JsonValue] | None = None
    values: dict[str, JsonValue] = Field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    charts: tuple[str, ...] = ()
    question: Question | None = None
    reason: str | None = Field(default=None, max_length=MAX_REASON_CHARS)
    method: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.handle is None) != (self.kind is None):
            raise ValueError("a result names a handle and its kind together")
        if self.handle is not None:
            if not is_handle_id(self.handle):
                raise ValueError("handle is not a handle id")
            if is_derived_id(self.handle) and not self.handle.startswith(f"{self.kind}_"):
                raise ValueError("a derived handle's prefix is its kind")
        if (self.status == "needs_input") != (self.question is not None):
            raise ValueError("a question comes with needs_input and only then")
        if (self.status == "insufficient_evidence") != bool(self.reason):
            raise ValueError("a reason comes with insufficient_evidence and only then")
        for name in self.values:
            if not _VALUE_NAME.fullmatch(name) or name in type(self).model_fields:
                raise ValueError(f"value name {name!r} is not a lower-case name distinct from the result's fields")
        # Bounded by the trace's own limit, so a result always fits its entry.
        serialize_json(self.recorded(), "result")
        return self

    def recorded(self) -> dict[str, Any]:
        """The result as plain JSON, leaving out fields at their defaults."""
        return self.model_dump(mode="json", exclude_defaults=True)

    def read(self, path: str) -> Any:
        """A dotted field, as in ``$step.summary.mean``; a first part that is not
        a field names an entry of ``values``, as in ``$step.days``."""
        data = self.model_dump(mode="json")
        head = path.split(".", 1)[0]
        if head not in data:
            data = data["values"]
        return walk_path(data, path)


def needs_choice(prompt: str, options: Iterable[JsonValue], *, many: bool = False) -> CapabilityResult:
    """A ``needs_input`` result asking the user to pick one of ``options``, or several when ``many``."""
    question = Question(kind="pick_many" if many else "pick_one", prompt=prompt, options=tuple(options))
    return CapabilityResult(status="needs_input", question=question)
