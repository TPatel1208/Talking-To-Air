"""The workflow spec as data: pydantic models for the YAML, and facts derived from it.

The models check shape only: field names and types, name patterns, and the
syntax of every reference, condition and placeholder. Checks that need the
capability or dataset registries, or the other specs, are in ``loader``;
obtain a checked spec through it.

Derived facts the compiler and runner use:

- ``compiled(step_id)``: the step's references, conditions, values and text
  parsed and bound (each reference's ``root`` says input, step or own result).
- ``dependencies(step_id)`` and ``order()``: the steps a step reads or names in
  ``after``; order follows references, not position in the file.
- ``always_runs(step_id)``: no ``when``, ``required``, and depending only on
  steps that always run.
- ``runnable`` / ``not_runnable_reason``: ``for_each`` and the ``workflow``
  kind are parsed and checked but the runner refuses them.
- ``approved_for(step_id)``: the products the runner passes as ``approved``.
- ``connectors(capabilities)`` and ``mode()``.

Output entries in ``figures``, ``data`` and ``interpretation`` are written as a
reference (``$figure``) or as ``{from: $figure, optional: true}`` for one that
may be absent; a finding carries ``optional: true`` itself.

``products`` is ``registry_default``, or a mapping from a quantity input to
an ordered list of dataset-registry keys or to ``registry_default``.
"""
from __future__ import annotations

import datetime
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    PrivateAttr,
    StrictFloat,
    StrictInt,
    Tag,
    field_validator,
    model_validator,
)

from tta_backend.harness.handles.model import HandleKind
from tta_backend.harness.results import Mode
from tta_backend.harness.workflows.answers import MAX_ITEMS as MAX_ANSWER_ITEMS
from tta_backend.harness.workflows.expressions import (
    MAX_DEPTH,
    NAME,
    RESERVED,
    And,
    Compare,
    Expr,
    Func,
    LimitRef,
    Not,
    Or,
    Ref,
    Reference,
    RootKind,
    Template,
    Value,
    bind,
    parse_condition,
    parse_reference,
    parse_template,
    parse_value,
    references,
)
from tta_backend.harness.workflows.expressions import (
    MapValue as MapValueNode,
)

if TYPE_CHECKING:
    from tta_backend.harness.capabilities.registry import CapabilityRegistry

InputType = Literal["quantity", "area", "date", "date_range", "date_list", "monitor", "choice", "number", "text", "handle"]
StepKind = Literal["use", "ask", "agent", "workflow", "stop"]
StopStatus = Literal["insufficient_evidence", "failed"]
SpecStatus = Literal["draft", "validated"]
AskKind = Literal["pick_one", "pick_many", "confirm_list", "text"]
ReturnType = Literal["string", "number", "boolean", "list"]
RegistryDefault = Literal["registry_default"]

REGISTRY_DEFAULT: RegistryDefault = "registry_default"
STEP_KINDS: tuple[StepKind, ...] = ("use", "ask", "agent", "workflow", "stop")
# The compiler names its own nodes with this prefix.
RESERVED_STEP_PREFIX = "input"
MAX_LABEL_CHARS = 80
MAX_LINE_CHARS = 500
# An agent step's bounds; what its reads resolve to is bounded in ``agent``.
MAX_INSTRUCTION_CHARS = 2000
MAX_READS = 12
MAX_RETURNS = 8

WORKFLOW_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
CAPABILITY_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
QUANTITY_NAME = re.compile(r"[a-z][a-z0-9_]*")
# The input types whose value is a handle, and the kind of that handle.
INPUT_HANDLE_KIND: dict[str, HandleKind] = {"area": "polygons"}


def _one_line(value: str, what: str, limit: int = MAX_LINE_CHARS) -> str:
    value = value.strip()
    if not value or "\n" in value:
        raise ValueError(f"{what} is one non-empty line")
    if len(value) > limit:
        raise ValueError(f"{what} is at most {limit} characters")
    return value


def _name(value: str, what: str) -> str:
    if not NAME.fullmatch(value):
        raise ValueError(f"{what} is a lower-case name of letters, digits and underscores")
    if value in RESERVED:
        raise ValueError(f"{what} may not be {value}, which the language reserves")
    return value


# A spec's own fields add a few levels above a value, which nests at most MAX_DEPTH.
_MAX_SPEC_DEPTH = MAX_DEPTH + 10


def _plain_dates(raw: Any, depth: int = 0) -> Any:
    """YAML reads unquoted dates as date objects; the spec keeps them as ISO text."""
    if depth > _MAX_SPEC_DEPTH:
        raise ValueError(f"a value nests more than {MAX_DEPTH} deep")
    if isinstance(raw, datetime.datetime | datetime.date):
        return raw.isoformat()
    if isinstance(raw, Mapping):
        return {key: _plain_dates(value, depth + 1) for key, value in raw.items()}
    if isinstance(raw, list | tuple):
        return [_plain_dates(value, depth + 1) for value in raw]
    return raw


def _check_value(raw: JsonValue) -> JsonValue:
    parse_value(raw)
    return raw


def _check_reference(raw: str) -> str:
    parse_reference(raw)
    return raw.strip()


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


# Inputs


def _is_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


class Input(_Model):
    """One input. A ``default`` makes it optional; a required one has an ``ask``.

    ``default: null`` is allowed for every type: the input is absent, and
    reads as None, unless a value is given.
    """

    type: InputType
    ask: str | None = None
    default: JsonValue = None
    options: tuple[str, ...] | None = None
    min: StrictInt | StrictFloat | None = None
    max: StrictInt | StrictFloat | None = None
    kind: HandleKind | None = None

    @field_validator("ask")
    @classmethod
    def _ask_line(cls, value: str | None) -> str | None:
        return None if value is None else _one_line(value, "ask")

    @property
    def optional(self) -> bool:
        return "default" in self.model_fields_set

    @property
    def nullable(self) -> bool:
        """Declared ``default: null``: optional, and None unless given."""
        return self.optional and self.default is None

    @property
    def handle_kind(self) -> HandleKind | None:
        """The handle kind this input's value is, if it is a handle."""
        if self.type == "handle":
            return self.kind
        return INPUT_HANDLE_KIND.get(self.type)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.type == "choice") != (self.options is not None):
            raise ValueError("options are given for a choice input and only then")
        if self.options is not None and (not self.options or len(set(self.options)) != len(self.options)):
            raise ValueError("options are a non-empty list without repeats")
        if (self.type == "handle") != (self.kind is not None):
            raise ValueError("kind is given for a handle input and only then")
        if (self.min is not None or self.max is not None) and self.type not in ("number", "text"):
            raise ValueError("min and max apply to number and text inputs")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is above max")
        if not self.optional and self.ask is None:
            raise ValueError("a required input has an ask")
        if self.optional:
            self._check_default()
        return self

    def _check_default(self) -> None:
        value = self.default
        if value is None:
            return
        if self.type in ("area", "handle"):
            article = "an" if self.type == "area" else "a"
            raise ValueError(f"{article} {self.type} input has no default other than null")
        if self.type == "number":
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError("the default of a number input is a number")
            if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
                raise ValueError("the default is outside min and max")
        elif self.type == "text":
            if not isinstance(value, str):
                raise ValueError("the default of a text input is text")
            if (self.min is not None and len(value) < self.min) or (self.max is not None and len(value) > self.max):
                raise ValueError("the default's length is outside min and max")
        elif self.type == "choice":
            if value not in (self.options or ()):
                raise ValueError("the default of a choice input is one of its options")
        elif self.type == "quantity":
            if not isinstance(value, str) or not QUANTITY_NAME.fullmatch(value):
                raise ValueError("the default of a quantity input is a lower-case quantity name")
        elif self.type == "date":
            if not _is_date(value):
                raise ValueError("the default of a date input is an ISO date")
        elif self.type == "date_range":
            if not (isinstance(value, dict) and set(value) == {"start", "end"} and all(_is_date(v) for v in value.values())):
                raise ValueError("the default of a date_range input has an ISO start and end")
            if str(value["start"]) > str(value["end"]):
                raise ValueError("the default's end is before its start")
        elif self.type == "date_list":
            if not (isinstance(value, list) and value and all(_is_date(v) for v in value)):
                raise ValueError("the default of a date_list input is a non-empty list of ISO dates")
        elif self.type == "monitor" and not isinstance(value, str):
            raise ValueError("the default of a monitor input is a monitor id")


# Steps


class _StepBase(_Model):
    kind: ClassVar[StepKind]

    id: str
    when: str | None = None
    required: bool = True
    for_each: str | None = None
    after: tuple[str, ...] = ()
    method: str | None = None
    label: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        _name(value, "a step id")
        if value.startswith(RESERVED_STEP_PREFIX):
            raise ValueError(f"a step id may not start with {RESERVED_STEP_PREFIX}")
        return value

    @field_validator("when")
    @classmethod
    def _when(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_condition(value)
        return value.strip()

    @field_validator("for_each")
    @classmethod
    def _for_each(cls, value: str | None) -> str | None:
        return None if value is None else _check_reference(value)

    @field_validator("after")
    @classmethod
    def _after(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not NAME.fullmatch(item):
                raise ValueError("after lists step ids")
        if len(set(value)) != len(value):
            raise ValueError("after lists each step once")
        return value

    @field_validator("method")
    @classmethod
    def _method(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("method is not empty")
        parse_template(value)
        return value

    @field_validator("label")
    @classmethod
    def _label(cls, value: str | None) -> str | None:
        return None if value is None else _one_line(value, "label", MAX_LABEL_CHARS)


def _check_with(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    for key, item in value.items():
        if not NAME.fullmatch(key):
            raise ValueError("with keys are lower-case parameter names")
        parse_value(item)
    return value


class UseStep(_StepBase):
    """Calls a capability. ``with`` uses the capability's own parameter names."""

    kind: ClassVar[StepKind] = "use"
    use: str
    with_: dict[str, JsonValue] = Field(default_factory=dict, alias="with")

    @field_validator("use")
    @classmethod
    def _use(cls, value: str) -> str:
        if not CAPABILITY_NAME.fullmatch(value):
            raise ValueError("use names a capability, like operations.difference")
        return value

    @field_validator("with_")
    @classmethod
    def _with(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _check_with(value)


class AskSpec(_Model):
    kind: AskKind
    question: str
    options: JsonValue = None
    items: JsonValue = None
    # The most items a confirm_list answer may hold, so a list too long for the step reading it is refused as an answer.
    max_items: int | None = None

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        return _one_line(value, "question")

    @field_validator("options", "items")
    @classmethod
    def _values(cls, value: JsonValue) -> JsonValue:
        return _check_value(value)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.kind in ("pick_one", "pick_many"):
            if self.options is None or self.items is not None:
                raise ValueError(f"a {self.kind} question has options and no items")
        elif self.kind == "confirm_list":
            if self.items is None or self.options is not None:
                raise ValueError("a confirm_list question has items and no options")
        elif self.options is not None or self.items is not None:
            raise ValueError("a text question has no options or items")
        if self.max_items is not None:
            if self.kind != "confirm_list":
                raise ValueError("only a confirm_list question has max_items")
            if isinstance(self.max_items, bool) or not 1 <= self.max_items <= MAX_ANSWER_ITEMS:
                raise ValueError(f"max_items is a whole number from 1 to {MAX_ANSWER_ITEMS}")
        return self


class AskStep(_StepBase):
    """Pauses for the user; the result is the answer."""

    kind: ClassVar[StepKind] = "ask"
    ask: AskSpec


class AgentSpec(_Model):
    """What an agent step is told and what it answers with. A ``list`` field is
    a list of short texts."""

    instruction: str
    reads: tuple[str, ...] = ()
    returns: dict[str, ReturnType]

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction is not empty")
        if len(value) > MAX_INSTRUCTION_CHARS:
            raise ValueError(f"instruction is at most {MAX_INSTRUCTION_CHARS} characters")
        return value

    @field_validator("reads")
    @classmethod
    def _reads(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        found = tuple(_check_reference(item) for item in value)
        if len(found) > MAX_READS:
            raise ValueError(f"reads lists at most {MAX_READS} references")
        if len(set(found)) != len(found):
            raise ValueError("reads lists each reference once")
        return found

    @field_validator("returns")
    @classmethod
    def _returns(cls, value: dict[str, ReturnType]) -> dict[str, ReturnType]:
        if not value:
            raise ValueError("returns names at least one field")
        if len(value) > MAX_RETURNS:
            raise ValueError(f"returns names at most {MAX_RETURNS} fields")
        for key in value:
            if not NAME.fullmatch(key):
                raise ValueError("returns keys are lower-case names")
        return value


class AgentStep(_StepBase):
    """A bounded judgement or piece of writing with a typed result."""

    kind: ClassVar[StepKind] = "agent"
    agent: AgentSpec


class WorkflowStep(_StepBase):
    """Runs another workflow at an exact version; ``answers`` fills the child's ``ask`` steps."""

    kind: ClassVar[StepKind] = "workflow"
    workflow: str
    version: int = Field(ge=1)
    with_: dict[str, JsonValue] = Field(default_factory=dict, alias="with")
    answers: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("workflow")
    @classmethod
    def _workflow(cls, value: str) -> str:
        if not WORKFLOW_NAME.fullmatch(value):
            raise ValueError("workflow names a workflow in lower case with hyphens")
        return value

    @field_validator("with_")
    @classmethod
    def _with(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _check_with(value)

    @field_validator("answers")
    @classmethod
    def _answers(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        for key, item in value.items():
            if not NAME.fullmatch(key):
                raise ValueError("answers keys are the child's ask step ids")
            parse_value(item)
        return value


class StopStep(_StepBase):
    """Ends the run early. The loader requires its ``when`` and ``reason``."""

    kind: ClassVar[StepKind] = "stop"
    stop: StopStatus
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("reason is not empty")
        parse_template(value)
        return value


def _step_kind(raw: Any) -> str | None:
    if isinstance(raw, _StepBase):
        return raw.kind
    if isinstance(raw, Mapping):
        present = [kind for kind in STEP_KINDS if kind in raw]
        return present[0] if len(present) == 1 else None
    return None


Step = Annotated[
    Annotated[UseStep, Tag("use")]
    | Annotated[AskStep, Tag("ask")]
    | Annotated[AgentStep, Tag("agent")]
    | Annotated[WorkflowStep, Tag("workflow")]
    | Annotated[StopStep, Tag("stop")],
    Discriminator(
        _step_kind,
        custom_error_type="step_kind",
        custom_error_message="a step has exactly one of use, ask, agent, workflow or stop",
    ),
]
AnyStep = UseStep | AskStep | AgentStep | WorkflowStep | StopStep


# Outputs


class OutputRef(_Model):
    """An output entry: ``$step`` written plainly, or ``{from: $step, optional: true}``."""

    from_: str = Field(alias="from")
    optional: bool = False

    @model_validator(mode="before")
    @classmethod
    def _plain(cls, raw: Any) -> Any:
        return {"from": raw} if isinstance(raw, str) else raw

    @field_validator("from_")
    @classmethod
    def _from(cls, value: str) -> str:
        return _check_reference(value)


class StepOutputRef(OutputRef):
    """A figure or data entry: a bare step reference such as ``$figure``."""

    @field_validator("from_")
    @classmethod
    def _bare(cls, value: str) -> str:
        node = parse_reference(value)
        if not (isinstance(node, Ref) and node.bare):
            raise ValueError("a figure or data entry is a bare step reference like $figure")
        return value


class FindingSpec(_Model):
    """A named value; ``unit``, ``where`` and ``when`` are references or literal text."""

    name: str
    value: str
    unit: str | None = None
    where: str | None = None
    when: str | None = None
    optional: bool = False

    @field_validator("name")
    @classmethod
    def _finding_name(cls, value: str) -> str:
        if not NAME.fullmatch(value):
            raise ValueError("a finding name is a lower-case name")
        return value

    @field_validator("value")
    @classmethod
    def _value(cls, value: str) -> str:
        return _check_reference(value)

    @field_validator("unit", "where", "when")
    @classmethod
    def _qualifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_value(value)
        return value


class Outputs(_Model):
    findings: tuple[FindingSpec, ...] = ()
    figures: tuple[StepOutputRef, ...] = ()
    data: tuple[StepOutputRef, ...] = ()
    interpretation: OutputRef | None = None

    @model_validator(mode="after")
    def _something(self) -> Self:
        if not (self.findings or self.figures or self.data or self.interpretation):
            raise ValueError("outputs declare at least one finding, figure, data entry or interpretation")
        names = [finding.name for finding in self.findings]
        if len(set(names)) != len(names):
            raise ValueError("finding names are unique")
        return self


# Compiled forms


@dataclass(frozen=True)
class CompiledStep:
    """A step's expressions parsed and bound. Unused fields are None or empty."""

    id: str
    kind: StepKind
    step: AnyStep
    when: Expr | None
    for_each: Reference | None
    with_: MapValueNode
    method: Template | None
    reason: Template | None
    options: Value | None
    items: Value | None
    reads: tuple[Reference, ...]
    answers: MapValueNode

    def references(self) -> tuple[Reference, ...]:
        """Every reference the step reads, including its own result in ``method``."""
        found: list[Reference] = []
        for node in (self.when, self.for_each, self.with_, self.method, self.reason, self.options, self.items, self.answers):
            if node is not None:
                found.extend(references(node))
        found.extend(self.reads)
        return tuple(found)

    def reading_references(self) -> tuple[Reference, ...]:
        """The references whose step being skipped skips this one: every one
        except those only inside ``exists(...)``, which a skipped step makes false."""
        found: list[Reference] = []
        for node in (self.when, self.method, self.reason):
            if node is not None:
                found.extend(_outside_exists(node))
        for value in (self.for_each, self.with_, self.options, self.items, self.answers):
            if value is not None:
                found.extend(references(value))
        found.extend(self.reads)
        return tuple(found)


def _outside_exists(node: Any) -> list[Reference]:
    if isinstance(node, Func):
        return [] if node.name == "exists" else [node.arg]
    if isinstance(node, Ref | LimitRef):
        return [node]
    if isinstance(node, Compare | And | Or):
        return _outside_exists(node.left) + _outside_exists(node.right)
    if isinstance(node, Not):
        return _outside_exists(node.operand)
    if isinstance(node, Template):
        return [ref for slot in node.slots for ref in _outside_exists(slot)]
    return list(references(node))


@dataclass(frozen=True)
class CompiledFinding:
    name: str
    value: Reference
    unit: Value | None
    where: Value | None
    when: Value | None
    optional: bool

    def references(self) -> tuple[Reference, ...]:
        found: list[Reference] = [self.value]
        for node in (self.unit, self.where, self.when):
            if node is not None:
                found.extend(references(node))
        return tuple(found)


@dataclass(frozen=True)
class CompiledOutputRef:
    ref: Reference
    optional: bool


@dataclass(frozen=True)
class CompiledOutputs:
    findings: tuple[CompiledFinding, ...]
    figures: tuple[CompiledOutputRef, ...]
    data: tuple[CompiledOutputRef, ...]
    interpretation: CompiledOutputRef | None


class CycleError(ValueError):
    def __init__(self, steps: tuple[str, ...]) -> None:
        super().__init__("steps " + ", ".join(steps) + " form a cycle")
        self.steps = steps


_EMPTY = MapValueNode(())


def _map(raw: Mapping[str, Any], roots: Mapping[str, RootKind]) -> MapValueNode:
    node = parse_value(dict(raw))
    assert isinstance(node, MapValueNode)
    bound: MapValueNode = bind(node, roots)
    return bound


def _ref(raw: str, roots: Mapping[str, RootKind]) -> Reference:
    bound: Reference = bind(parse_reference(raw), roots)
    return bound


def _optional_value(raw: Any, roots: Mapping[str, RootKind]) -> Value | None:
    if raw is None:
        return None
    bound: Value = bind(parse_value(raw), roots)
    return bound


def _compile(step: AnyStep, roots: Mapping[str, RootKind]) -> CompiledStep:
    with_ = _map(step.with_, roots) if isinstance(step, UseStep | WorkflowStep) else _EMPTY
    return CompiledStep(
        id=step.id,
        kind=step.kind,
        step=step,
        when=None if step.when is None else bind(parse_condition(step.when), roots),
        for_each=None if step.for_each is None else _ref(step.for_each, roots),
        with_=with_,
        method=None if step.method is None else bind(parse_template(step.method), roots, own=step.kind == "use"),
        reason=bind(parse_template(step.reason), roots) if isinstance(step, StopStep) and step.reason else None,
        options=_optional_value(step.ask.options, roots) if isinstance(step, AskStep) else None,
        items=_optional_value(step.ask.items, roots) if isinstance(step, AskStep) else None,
        reads=tuple(_ref(r, roots) for r in step.agent.reads) if isinstance(step, AgentStep) else (),
        answers=_map(step.answers, roots) if isinstance(step, WorkflowStep) else _EMPTY,
    )


# The spec


class WorkflowSpec(_Model):
    """A workflow spec as written. Build it through ``loader`` for a checked one."""

    name: str
    version: int = Field(ge=1)
    status: SpecStatus
    description: str
    examples: tuple[str, ...] = Field(min_length=1)
    inputs: dict[str, Input]
    products: RegistryDefault | dict[str, RegistryDefault | tuple[str, ...]] | None = None
    limits: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict)
    steps: tuple[Step, ...]
    outputs: Outputs
    benchmark: tuple[str, ...] = ()
    validated_by: str | None = None
    validated_on: datetime.date | None = None

    _roots: dict[str, RootKind] = PrivateAttr(default_factory=dict)
    _by_id: dict[str, AnyStep] = PrivateAttr(default_factory=dict)
    _compiled: dict[str, CompiledStep] = PrivateAttr(default_factory=dict)
    _outputs: CompiledOutputs | None = PrivateAttr(default=None)
    _deps: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _dates_as_text(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping):
            out = dict(raw)
            for key in ("inputs", "steps", "outputs", "products", "limits"):
                if key in out:
                    out[key] = _plain_dates(out[key])
            return out
        return raw

    @field_validator("name")
    @classmethod
    def _workflow_name(cls, value: str) -> str:
        if not WORKFLOW_NAME.fullmatch(value):
            raise ValueError("a workflow name is lower case with hyphens, like event-vs-baseline")
        return value

    @field_validator("description")
    @classmethod
    def _description(cls, value: str) -> str:
        return _one_line(value, "description")

    @field_validator("examples")
    @classmethod
    def _examples(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_one_line(item, "an example") for item in value)

    @field_validator("steps")
    @classmethod
    def _some_steps(cls, value: tuple[AnyStep, ...]) -> tuple[AnyStep, ...]:
        if not value:
            raise ValueError("a spec has at least one step")
        return value

    @field_validator("inputs")
    @classmethod
    def _input_names(cls, value: dict[str, Input]) -> dict[str, Input]:
        for key in value:
            _name(key, "an input name")
        return value

    @field_validator("limits")
    @classmethod
    def _limits(cls, value: dict[str, int | float]) -> dict[str, int | float]:
        for key, number in value.items():
            _name(key, "a limit name")
            if not math.isfinite(number):
                raise ValueError("a limit is a finite number")
        return value

    @field_validator("products")
    @classmethod
    def _products(
        cls, value: RegistryDefault | dict[str, RegistryDefault | tuple[str, ...]] | None
    ) -> RegistryDefault | dict[str, RegistryDefault | tuple[str, ...]] | None:
        if isinstance(value, dict):
            for key, listed in value.items():
                _name(key, "a products key")
                if isinstance(listed, tuple) and (not listed or len(set(listed)) != len(listed)):
                    raise ValueError("a products list is non-empty without repeats")
        return value

    @field_validator("benchmark")
    @classmethod
    def _benchmark(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("benchmark lists case ids, each once")
        return value

    @field_validator("validated_by")
    @classmethod
    def _validated_by(cls, value: str | None) -> str | None:
        return None if value is None else _one_line(value, "validated_by", 200)

    def model_post_init(self, context: Any, /) -> None:
        roots: dict[str, RootKind] = {}
        for step in self.steps:
            roots.setdefault(step.id, "step")
            self._by_id.setdefault(step.id, step)
        for name in self.inputs:
            roots[name] = "input"
        self._roots = roots
        for step_id, step in self._by_id.items():
            compiled = _compile(step, roots)
            self._compiled[step_id] = compiled
            deps = {ref.name for ref in compiled.references() if isinstance(ref, Ref) and ref.root == "step"}
            deps.update(item for item in step.after if item in self._by_id)
            self._deps[step_id] = frozenset(deps)
        out = self.outputs
        self._outputs = CompiledOutputs(
            findings=tuple(
                CompiledFinding(
                    name=f.name,
                    value=_ref(f.value, roots),
                    unit=_optional_value(f.unit, roots),
                    where=_optional_value(f.where, roots),
                    when=_optional_value(f.when, roots),
                    optional=f.optional,
                )
                for f in out.findings
            ),
            figures=tuple(CompiledOutputRef(_ref(e.from_, roots), e.optional) for e in out.figures),
            data=tuple(CompiledOutputRef(_ref(e.from_, roots), e.optional) for e in out.data),
            interpretation=(
                None
                if out.interpretation is None
                else CompiledOutputRef(_ref(out.interpretation.from_, roots), out.interpretation.optional)
            ),
        )

    # Lookups

    @property
    def step_ids(self) -> tuple[str, ...]:
        """Step ids in file order."""
        return tuple(step.id for step in self.steps)

    def step(self, step_id: str) -> AnyStep:
        """The step with this id; KeyError if there is none."""
        return self._by_id[step_id]

    def compiled(self, step_id: str) -> CompiledStep:
        """The step's parsed, bound expressions; KeyError if there is none."""
        return self._compiled[step_id]

    @property
    def compiled_outputs(self) -> CompiledOutputs:
        assert self._outputs is not None
        return self._outputs

    def root_of(self, name: str) -> RootKind | None:
        """Whether a name is an input or a step id; None if neither."""
        return self._roots.get(name)

    def children(self) -> tuple[tuple[str, int], ...]:
        """(name, version) of each child workflow, in file order."""
        return tuple((s.workflow, s.version) for s in self.steps if isinstance(s, WorkflowStep))

    # Order

    def dependencies(self, step_id: str) -> frozenset[str]:
        """Step ids this step reads (in any of its fields) or names in ``after``."""
        return self._deps[step_id]

    def reads(self, step_id: str) -> frozenset[str]:
        """Step ids whose being skipped skips this step: its dependencies other
        than those it names only in ``after`` or only inside ``exists(...)``."""
        compiled = self._compiled[step_id]
        return frozenset(
            ref.name for ref in compiled.reading_references() if isinstance(ref, Ref) and ref.root == "step"
        )

    def cycles(self) -> tuple[tuple[str, ...], ...]:
        """Each set of steps that depend on one another in a loop, in file order."""
        index = {step_id: i for i, step_id in enumerate(self._deps)}
        found: list[tuple[str, ...]] = []
        for component in strongly_connected(self._deps):
            if len(component) > 1 or next(iter(component)) in self._deps[next(iter(component))]:
                found.append(tuple(sorted(component, key=index.__getitem__)))
        return tuple(sorted(found, key=lambda c: index[c[0]]))

    def order(self) -> tuple[str, ...]:
        """Step ids in an order that runs each after its dependencies, earliest in the file first among the ready ones."""
        cycles = self.cycles()
        if cycles:
            raise CycleError(cycles[0])
        done: list[str] = []
        placed: set[str] = set()
        remaining = list(self._deps)
        while remaining:
            ready = next(s for s in remaining if self._deps[s] <= placed)
            done.append(ready)
            placed.add(ready)
            remaining.remove(ready)
        return tuple(done)

    def always_runs(self, step_id: str) -> bool:
        """No ``when``, ``required``, and every dependency always runs."""
        if self.cycles():
            raise CycleError(self.cycles()[0])
        return self._always(step_id, {})

    def _always(self, step_id: str, memo: dict[str, bool]) -> bool:
        if step_id not in memo:
            step = self._by_id[step_id]
            memo[step_id] = (
                step.when is None and step.required and all(self._always(dep, memo) for dep in self._deps[step_id])
            )
        return memo[step_id]

    # What the runner needs

    @property
    def not_runnable_reason(self) -> str | None:
        """Why the runner refuses this spec, naming the first step that uses an unsupported feature."""
        for step in self.steps:
            if step.kind == "workflow":
                return f"step {step.id} is a workflow step, which the runner does not run"
            if step.for_each is not None:
                return f"step {step.id} uses for_each, which the runner does not run"
        return None

    @property
    def runnable(self) -> bool:
        return self.not_runnable_reason is None

    def quantity_input_of(self, step_id: str) -> str | None:
        """The quantity input a ``use`` step's ``with.quantity`` is, as in ``quantity: $quantity``."""
        node = self._compiled[step_id].with_.get("quantity")
        if isinstance(node, Ref) and node.bare and node.root == "input" and self.inputs[node.name].type == "quantity":
            return node.name
        return None

    def approved_for(self, step_id: str) -> list[str] | RegistryDefault | None:
        """The approved products for the quantity a ``use`` step retrieves.

        The runner passes this as the capability's ``approved`` parameter when
        the capability's params declare one. None when the step's quantity is
        not a quantity input or ``products`` has no entry for it.
        """
        quantity = self.quantity_input_of(step_id)
        if quantity is None or self.products is None:
            return None
        listed = self.products if isinstance(self.products, str) else self.products.get(quantity)
        if listed is None:
            return None
        return REGISTRY_DEFAULT if isinstance(listed, str) else list(listed)

    def connectors(self, capabilities: CapabilityRegistry) -> frozenset[str]:
        """The connectors the ``use`` steps' capabilities require."""
        found: set[str] = set()
        for step in self.steps:
            if isinstance(step, UseStep):
                capability = capabilities.get(step.use)
                if capability.connector is not None and capability.connector_required:
                    found.add(capability.connector)
        return frozenset(found)

    def mode(self) -> Mode:
        """How a result of this spec is labelled."""
        if self.status == "validated":
            return Mode.validated(self.name, self.version)
        return Mode.draft(self.name, self.version)


def strongly_connected(graph: Mapping[str, frozenset[str]]) -> list[frozenset[str]]:
    """Tarjan's algorithm, iterative; edges to unknown nodes are ignored."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    out: list[frozenset[str]] = []
    counter = 0
    for start in graph:
        if start in index:
            continue
        work: list[tuple[str, list[str]]] = [(start, sorted(d for d in graph[start] if d in graph))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, pending = work[-1]
            if pending:
                nxt = pending.pop()
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, sorted(d for d in graph[nxt] if d in graph)))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                out.append(frozenset(component))
    return out
