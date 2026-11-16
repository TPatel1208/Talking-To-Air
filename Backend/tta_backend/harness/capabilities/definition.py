"""What a capability declares: its name, parameters, output, connector and how it runs."""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args

from pydantic import BaseModel, ConfigDict

from tta_backend.harness.capabilities.result import CapabilityResult, JobRef, walk_path
from tta_backend.harness.handles.model import KINDS, HandleKind

if TYPE_CHECKING:
    from tta_backend.harness.capabilities.call import Call

P = TypeVar("P", bound=BaseModel)

_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|[0-9]+))*")
MAX_LABEL_CHARS = 80


class Params(BaseModel):
    """Base for a capability's parameters: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class HandleParam:
    """Marks a parameter as a handle id (or a list of them) of one of ``kinds``."""

    kinds: tuple[HandleKind, ...]

    def __post_init__(self) -> None:
        if not self.kinds or any(kind not in KINDS for kind in self.kinds):
            raise ValueError(f"a handle parameter accepts one or more of {KINDS}")


def handle_param(*kinds: HandleKind) -> HandleParam:
    """For ``Annotated[str, handle_param("grid")]``, ``Annotated[list[str], ...]`` or ``Annotated[str | None, ...]``."""
    return HandleParam(kinds)


def _contains_marker(annotation: Any) -> bool:
    if isinstance(annotation, HandleParam):
        return True
    return any(_contains_marker(arg) for arg in get_args(annotation))


def _handle_params(model: type[BaseModel]) -> dict[str, tuple[HandleKind, ...]]:
    found: dict[str, tuple[HandleKind, ...]] = {}
    for name, info in model.model_fields.items():
        markers = [m for m in info.metadata if isinstance(m, HandleParam)]
        if len(markers) > 1:
            raise ValueError(f"parameter {name} has more than one handle_param")
        if markers:
            found[name] = markers[0].kinds
        elif _contains_marker(info.annotation):
            # Nested, e.g. Annotated[str, ...] | None, which the invoke path would not see.
            raise ValueError(f"parameter {name}: put handle_param on the outermost Annotated")
    return found


def _render(value: Any) -> str:
    if value is None:
        raise LookupError("a method placeholder has no value")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list | tuple):
        return ", ".join(_render(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(", ", ": "))
    return str(value)


def _check_template(template: str) -> None:
    if not template.strip() or "\n" in template:
        raise ValueError("method is one non-empty line")
    if not all(_PATH.fullmatch(p) for p in _PLACEHOLDER.findall(template)):
        raise ValueError("method placeholders are dotted lower-case names, like {product} or {summary.mean}")
    if "{" in _PLACEHOLDER.sub("", template) or "}" in _PLACEHOLDER.sub("", template):
        raise ValueError("method has an unmatched brace")


@dataclass(frozen=True, kw_only=True)
class Capability(Generic[P]):
    """A named tool: handles and small values in, a handle and a small summary out.

    Give ``run`` for a single call, or ``begin`` and ``complete`` for a
    retrieval whose job reference is saved between submitting and finishing.
    ``method`` is one sentence filled from the parameters and the result.

    A declared ``connector`` must be connected before the call runs. With
    ``connector_required=False`` the call runs without it, and only reaching
    for it (``call.earthdata`` or ``call.secret``) fails as not connected; for
    a capability that needs the connector for some inputs only.

    ``workflow_only`` marks a capability that only validated workflows may
    run, never an exploratory specialist.
    """

    name: str
    description: str
    params: type[P]
    output_kind: HandleKind | None
    label: str
    method: str
    retrieves: bool
    retry_safe: bool
    connector: str | None = None
    connector_required: bool = True
    workflow_only: bool = False
    run: Callable[[Call[P]], Awaitable[CapabilityResult]] | None = None
    # Returns the job reference, or a result when it settles the call without submitting.
    begin: Callable[[Call[P]], Awaitable[JobRef | CapabilityResult]] | None = None
    complete: Callable[[Call[P], JobRef], Awaitable[CapabilityResult]] | None = None
    handle_params: Mapping[str, tuple[HandleKind, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("a capability name is dotted lower case with a group, like operations.difference")
        if not self.description.strip() or "\n" in self.description:
            raise ValueError("description is one non-empty line")
        if not 0 < len(self.label.strip()) <= MAX_LABEL_CHARS:
            raise ValueError(f"label is 1 to {MAX_LABEL_CHARS} characters")
        if self.output_kind is not None and self.output_kind not in KINDS:
            raise ValueError(f"output_kind is one of {KINDS} or None")
        if self.connector is not None and not self.connector.strip():
            raise ValueError("connector is a name or None")
        if self.connector is None and not self.connector_required:
            raise ValueError("connector_required=False needs a connector")
        if not (isinstance(self.params, type) and issubclass(self.params, BaseModel)):
            raise ValueError("params is a pydantic model")
        _check_template(self.method)
        phases = (self.begin is not None, self.complete is not None)
        if phases not in ((False, False), (True, True)) or (self.run is not None) == (phases == (True, True)):
            raise ValueError("give run, or begin and complete, not both")
        object.__setattr__(self, "handle_params", _handle_params(self.params))

    @property
    def group(self) -> str:
        return self.name.rsplit(".", 1)[0]

    @property
    def two_phase(self) -> bool:
        return self.begin is not None

    def accepts(self, param: str, kind: HandleKind) -> bool:
        """Whether a handle of ``kind`` may be passed as ``param``."""
        return kind in self.handle_params.get(param, ())

    def schema(self) -> dict[str, Any]:
        """The parameters' JSON schema; handle parameters carry ``x-handle-kinds``."""
        schema = self.params.model_json_schema()
        for name, kinds in self.handle_params.items():
            schema["properties"][name]["x-handle-kinds"] = list(kinds)
        return schema

    def method_sentence(self, params: Mapping[str, Any], result: CapabilityResult) -> str:
        """The method template filled from the recorded parameters and the
        result; result fields and values win over a parameter of the same name."""
        recorded = result.recorded()
        values = recorded.pop("values", {})
        sources = {**params, **recorded, **values}
        return _PLACEHOLDER.sub(lambda m: _render(walk_path(sources, m[1])), self.method)
