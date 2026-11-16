"""Loading workflow specs and checking them against the capabilities, the datasets and each other.

On disk a pack holds ``<root>/<pack>/workflows/<name>/<version>.yaml``; the
directory name and the file stem must match the spec's ``name`` and
``version``. Every problem found is a ``SpecError``; all of them for a spec,
or for every file under a root, are raised together as ``SpecErrors``.

Entry points:

- ``load_packs(root, capabilities, datasets)``: every pack under a root.
- ``load_specs(sources, capabilities, datasets)``: a set of in-memory specs
  (YAML text or mappings, keyed by a label), checked together.
- ``load_spec(source, capabilities, datasets, catalogue=...)``: one spec,
  with its child workflows looked up in ``catalogue``.
- ``parse_spec`` and ``check_spec`` for the two halves of ``load_spec``.

``approved_parameter(spec, step_id, capabilities)`` is what the runner passes
as a ``use`` step's ``approved`` parameter: the spec's products for the step's
quantity input when the capability declares ``approved``, else None.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_args

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from tta_backend.datasets.registry import CollectionConfig
from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.registry import CapabilityRegistry, UnknownCapability
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.workflows.expressions import (
    And,
    Compare,
    Const,
    Expr,
    LimitRef,
    ListValue,
    MapValue,
    Ref,
    Reference,
    Value,
    ValueType,
    references,
    type_problems,
)
from tta_backend.harness.workflows.spec import (
    STEP_KINDS,
    AgentStep,
    AskStep,
    CompiledOutputRef,
    RegistryDefault,
    StopStep,
    UseStep,
    WorkflowSpec,
    WorkflowStep,
    strongly_connected,
)

INLINE = "<inline>"
APPROVED = "approved"
PRODUCT = "product"
QUANTITY = "quantity"
RENDER_GROUP = "render"
_STRING_INPUTS = frozenset({"quantity", "choice", "text", "date", "monitor"})
_RESULT_FIELDS = frozenset(CapabilityResult.model_fields)
_ERROR_PREFIX = "Value error, "
_TOO_DEEP = "the spec nests too deeply to read"


class SpecError(Exception):
    """One problem with a spec, in fixed wording filled from the spec's own names."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        workflow: str | None = None,
        version: int | None = None,
        step: str | None = None,
    ) -> None:
        self.message = message
        self.source = source
        self.workflow = workflow
        self.version = version
        self.step = step
        super().__init__(str(self))

    def __str__(self) -> str:
        where = [self.source]
        if self.workflow is not None:
            where.append(self.workflow if self.version is None else f"{self.workflow} v{self.version}")
        if self.step is not None:
            where.append(f"step {self.step}")
        return ": ".join([*where, self.message])


class SpecErrors(Exception):
    """Every problem found, raised together so an author sees them all at once."""

    def __init__(self, errors: Sequence[SpecError]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(str(error) for error in self.errors))


class UnknownWorkflow(LookupError):
    def __init__(self, name: str, version: int | None = None) -> None:
        label = name if version is None else f"{name} v{version}"
        super().__init__(f"no workflow {label}")
        self.name = name
        self.version = version


class WorkflowCatalogue:
    """Checked specs by name and version."""

    def __init__(self, specs: Iterable[WorkflowSpec] = ()) -> None:
        self._specs: dict[str, dict[int, WorkflowSpec]] = {}
        for spec in specs:
            versions = self._specs.setdefault(spec.name, {})
            if spec.version in versions:
                raise ValueError(f"workflow {spec.name} v{spec.version} is given more than once")
            versions[spec.version] = spec

    def get(self, name: str, version: int) -> WorkflowSpec:
        try:
            return self._specs[name][version]
        except KeyError:
            raise UnknownWorkflow(name, version) from None

    def versions(self, name: str) -> tuple[int, ...]:
        if name not in self._specs:
            raise UnknownWorkflow(name)
        return tuple(sorted(self._specs[name]))

    def latest(self, name: str) -> WorkflowSpec:
        return self.get(name, self.versions(name)[-1])

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return key[1] in self._specs.get(key[0], {})
        return key in self._specs

    def __iter__(self) -> Iterator[WorkflowSpec]:
        for name in self.names():
            for version in sorted(self._specs[name]):
                yield self._specs[name][version]

    def __len__(self) -> int:
        return sum(len(versions) for versions in self._specs.values())

    def index(self) -> str:
        """One line per workflow at its latest version, for the supervisor's prompt."""
        lines = []
        for name in self.names():
            spec = self.latest(name)
            lines.append(f"{spec.name} v{spec.version} ({spec.status}): {spec.description}")
        return "\n".join(lines)


# Parsing


def _field_path(loc: Sequence[int | str]) -> str:
    return ".".join(str(part) for part in loc)


def _message(kind: str, msg: str, path: str) -> str:
    if kind == "missing":
        return f"{path} is required" if path else "a field is required"
    if kind == "extra_forbidden":
        return f"{path} is not a known field"
    if msg.startswith(_ERROR_PREFIX):
        msg = msg[len(_ERROR_PREFIX) :]
    return f"{path}: {msg}" if path else msg


def _validation_errors(error: ValidationError, raw: Mapping[str, Any], label: str) -> list[SpecError]:
    name = raw.get("name") if isinstance(raw.get("name"), str) else None
    version = raw.get("version") if isinstance(raw.get("version"), int) else None
    listed = raw.get("steps")
    steps: list[Any] = listed if isinstance(listed, list) else []
    found: list[SpecError] = []
    for item in error.errors():
        loc = list(item["loc"])
        step: str | None = None
        if len(loc) >= 2 and loc[0] == "steps" and isinstance(loc[1], int) and loc[1] < len(steps):
            entry = steps[loc[1]]
            step_id = entry.get("id") if isinstance(entry, Mapping) else None
            if isinstance(step_id, str):
                step = step_id
                loc = loc[2:]
                if loc and loc[0] in STEP_KINDS:
                    loc = loc[1:]
        found.append(
            SpecError(_message(item["type"], item["msg"], _field_path(loc)), source=label, workflow=name, version=version, step=step)
        )
    return found


def parse_spec(source: str | Mapping[str, Any], *, label: str = INLINE) -> WorkflowSpec:
    """A spec from YAML text or a mapping, checked for shape only; raises SpecErrors."""
    raw: Any = source
    if isinstance(source, str):
        try:
            raw = yaml.safe_load(source)
        except yaml.YAMLError as error:
            mark = getattr(error, "problem_mark", None)
            where = f" at line {mark.line + 1}" if mark is not None else ""
            raise SpecErrors([SpecError(f"the file is not valid YAML{where}", source=label)]) from None
        except RecursionError:
            raise SpecErrors([SpecError(_TOO_DEEP, source=label)]) from None
    if not isinstance(raw, Mapping):
        raise SpecErrors([SpecError("a spec is a mapping of fields", source=label)])
    try:
        return WorkflowSpec.model_validate(raw)
    except ValidationError as error:
        raise SpecErrors(_validation_errors(error, raw, label)) from None
    except RecursionError:
        # Bounded in the spec's own validation; this catches what a library nests further.
        raise SpecErrors([SpecError(_TOO_DEEP, source=label)]) from None


# Checks on one spec


@dataclass
class _Report:
    spec: WorkflowSpec
    label: str
    errors: list[SpecError]

    def add(self, message: str, step: str | None = None) -> None:
        self.errors.append(
            SpecError(message, source=self.label, workflow=self.spec.name, version=self.spec.version, step=step)
        )


def check_spec(
    spec: WorkflowSpec,
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
    *,
    label: str = INLINE,
) -> list[SpecError]:
    """Every problem with one spec that does not need the other specs."""
    report = _Report(spec, label, [])
    _check_names(report)
    _check_references(report)
    cyclic = _check_cycles(report)
    _check_stops(report)
    for step in spec.steps:
        if isinstance(step, UseStep):
            _check_use(report, step, capabilities)
        elif isinstance(step, AgentStep):
            _check_agent(report, step)
    _check_products(report, capabilities, datasets)
    _check_outputs(report, capabilities, cyclic)
    _check_validated(report)
    return report.errors


def _check_names(report: _Report) -> None:
    spec = report.spec
    seen: set[str] = set()
    for step in spec.steps:
        if step.id in seen:
            report.add(f"step id {step.id} is used more than once", step.id)
        seen.add(step.id)
        if step.id in spec.inputs:
            report.add(f"step id {step.id} is also an input name", step.id)
        for other in step.after:
            if other == step.id:
                report.add("after names the step itself", step.id)
            elif spec.root_of(other) != "step":
                report.add(f"after names {other}, which is not a step", step.id)


def _check_reference(report: _Report, ref: Reference, step: str | None, *, in_condition: bool = False) -> None:
    spec = report.spec
    if isinstance(ref, LimitRef):
        if ref.name not in spec.limits:
            report.add(f"{ref.text()} is not a limit", step)
        return
    if ref.root is None:
        report.add(f"{ref.text()} names no input or step", step)
    elif ref.root == "own":
        if ref.name not in _RESULT_FIELDS:
            report.add(f"{{{ref.text()}}} names no input, step or result field", step)
    elif in_condition and not ref.dollar and ref.root != "input":
        report.add(f"{ref.name} in when is not an input; a step is written ${ref.name}", step)
    elif ref.root == "step" and ref.path:
        _check_answer_field(report, ref, step)


def _check_no_answer(report: _Report, field: str, refs: Iterable[Reference], step: str) -> None:
    """An agent step's answer is read only in ``when`` and ``outputs.interpretation``,
    so nothing a capability makes, and no fixed wording, comes from a model."""
    for ref in refs:
        if isinstance(ref, Ref) and ref.root == "step" and isinstance(report.spec.step(ref.name), AgentStep):
            report.add(
                f"{field} reads agent step {ref.name}; a model's answer is read only in when and in outputs.interpretation",
                step,
            )
            return


def _check_answer_field(report: _Report, ref: Ref, step: str | None) -> None:
    """A field read from an agent step is one its ``returns`` declares; only a list goes deeper, by index."""
    target = report.spec.step(ref.name)
    if not isinstance(target, AgentStep):
        return
    returns = target.agent.returns
    head, rest = ref.path[0], ref.path[1:]
    if head not in returns:
        report.add(f"{ref.text()}: agent step {ref.name} returns no field {head}", step)
    elif rest and (returns[head] != "list" or len(rest) > 1 or not rest[0].isdigit()):
        report.add(f"{ref.text()}: field {head} of agent step {ref.name} is a {returns[head]}, with nothing inside it", step)


_ANSWER_TYPES: dict[str, ValueType] = {"string": "string", "number": "number", "boolean": "boolean"}


def _type_of(spec: WorkflowSpec, ref: Reference) -> ValueType | None:
    if isinstance(ref, LimitRef):
        return "number" if ref.name in spec.limits else None
    if ref.root == "step" and len(ref.path) == 1:
        target = spec.step(ref.name)
        if isinstance(target, AgentStep):
            return _ANSWER_TYPES.get(target.agent.returns.get(ref.path[0], ""))
        return None
    if ref.root != "input" or ref.path:
        return None
    kind = spec.inputs[ref.name].type
    if kind == "number":
        return "number"
    return "string" if kind in _STRING_INPUTS else None


def _check_condition(report: _Report, node: Expr, step: str) -> None:
    spec = report.spec
    for ref in references(node):
        _check_reference(report, ref, step, in_condition=True)
    for problem in type_problems(node, lambda ref: _type_of(spec, ref)):
        report.add(f"when: {problem}", step)
    for compare in _comparisons(node):
        for side, other in ((compare.left, compare.right), (compare.right, compare.left)):
            if isinstance(side, Ref) and side.root == "input" and side.bare and isinstance(other, Const):
                options = spec.inputs[side.name].options
                if options is not None and isinstance(other.value, str) and other.value not in options:
                    report.add(f"when compares {side.name} with text that is not one of its options", step)


def _comparisons(node: Expr) -> Iterator[Compare]:
    if isinstance(node, Compare):
        yield node
    for child in (getattr(node, "left", None), getattr(node, "right", None), getattr(node, "operand", None)):
        if child is not None:
            yield from _comparisons(child)


def _check_references(report: _Report) -> None:
    spec = report.spec
    for step_id in dict.fromkeys(spec.step_ids):
        compiled = spec.compiled(step_id)
        if compiled.when is not None:
            _check_condition(report, compiled.when, step_id)
        fields = (
            ("for_each", compiled.for_each), ("with", compiled.with_), ("method", compiled.method),
            ("reason", compiled.reason), ("options", compiled.options), ("items", compiled.items),
            ("answers", compiled.answers),
        )
        for field, node in fields:
            if node is not None:
                for ref in references(node):
                    _check_reference(report, ref, step_id)
                _check_no_answer(report, field, references(node), step_id)
        for ref in compiled.reads:
            _check_reference(report, ref, step_id)
        _check_no_answer(report, "reads", compiled.reads, step_id)
    outputs = spec.compiled_outputs
    for finding in outputs.findings:
        for ref in finding.references():
            _check_reference(report, ref, None)
    entries = [*outputs.figures, *outputs.data]
    if outputs.interpretation is not None:
        entries.append(outputs.interpretation)
    for entry in entries:
        _check_reference(report, entry.ref, None)


def _check_cycles(report: _Report) -> bool:
    cycles = report.spec.cycles()
    for cycle in cycles:
        if len(cycle) == 1:
            report.add("the step depends on itself", cycle[0])
        else:
            report.add("steps " + ", ".join(cycle) + " depend on one another in a cycle")
    return bool(cycles)


def _check_stops(report: _Report) -> None:
    for step in report.spec.steps:
        if isinstance(step, StopStep) and (step.when is None or step.reason is None):
            report.add("a stop step has a when and a reason", step.id)


# agent steps


def _check_agent(report: _Report, step: AgentStep) -> None:
    """``reads`` takes summaries and small values: never a handle, which stands for data."""
    spec = report.spec
    for ref in spec.compiled(step.id).reads:
        if not isinstance(ref, Ref) or not ref.bare:
            continue
        if ref.root == "input" and spec.inputs[ref.name].handle_kind is not None:
            report.add(f"reads {ref.text()}, a handle; an agent step reads summaries and small values", step.id)
        elif ref.root == "step" and isinstance(spec.step(ref.name), UseStep):
            report.add(f"reads {ref.text()}, a handle; read a field of it, like {ref.text()}.summary", step.id)


# use steps


def _shape(annotation: Any) -> tuple[bool, bool]:
    """Whether a parameter type accepts one string, and whether it accepts a list."""
    args = getattr(annotation, "__args__", None)
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, tuple):
        return False, True
    if annotation is str:
        return True, False
    if args and origin is not Annotated:
        single = listed = False
        for arg in args:
            one, many = _shape(arg)
            single, listed = single or one, listed or many
        return single, listed
    return False, False


_ADAPTERS: dict[tuple[type[BaseModel], str], TypeAdapter[Any]] = {}


def _adapter(params: type[BaseModel], name: str, info: FieldInfo) -> TypeAdapter[Any]:
    key = (params, name)
    if key not in _ADAPTERS:
        annotation: Any = info.annotation
        if info.metadata:
            annotation = Annotated[(annotation, *info.metadata)]
        _ADAPTERS[key] = TypeAdapter(annotation)
    return _ADAPTERS[key]


def _is_literal(node: Value) -> bool:
    return not references(node)


def _plain(node: Value) -> Any:
    if isinstance(node, Const):
        return node.value
    if isinstance(node, ListValue):
        return [_plain(item) for item in node.items]
    assert isinstance(node, MapValue)
    return {key: _plain(item) for key, item in node.items}


def _accepts_none(info: FieldInfo) -> bool:
    """Whether a parameter takes None: a None default, or None in its type."""
    if not info.is_required() and info.default is None:
        return True

    def has_none(annotation: Any) -> bool:
        return annotation is None or annotation is type(None) or any(has_none(a) for a in get_args(annotation))

    return has_none(info.annotation)


def _nested_inputs(node: Value) -> list[str]:
    """Inputs passed whole inside a list or mapping, as in ``[$a, $b]`` or ``{start: $s}``."""
    if not isinstance(node, ListValue | MapValue):
        return []
    found: list[str] = []
    items = node.items if isinstance(node, ListValue) else tuple(item for _, item in node.items)
    for item in items:
        if isinstance(item, Ref) and item.bare and item.root == "input":
            found.append(item.name)
        found.extend(_nested_inputs(item))
    return list(dict.fromkeys(found))


def _check_use(report: _Report, step: UseStep, capabilities: CapabilityRegistry) -> None:
    spec = report.spec
    try:
        capability = capabilities.get(step.use)
    except UnknownCapability:
        report.add(f"{step.use} is not a registered capability", step.id)
        return
    fields = capability.params.model_fields
    with_ = spec.compiled(step.id).with_
    for key, node in with_.items:
        if key == APPROVED and key in fields:
            report.add("approved comes from products, not from with", step.id)
            continue
        if key not in fields:
            report.add(f"{capability.name} has no parameter {key}", step.id)
            continue
        info = fields[key]
        if _is_literal(node):
            try:
                _adapter(capability.params, key, info).validate_python(_plain(node))
            except ValidationError as error:
                first = error.errors()[0]["msg"]
                if first.startswith(_ERROR_PREFIX):
                    first = first[len(_ERROR_PREFIX) :]
                report.add(f"with.{key} is not a valid value for {capability.name}: {first}", step.id)
        if isinstance(node, Ref) and node.bare and node.root == "input" and spec.inputs[node.name].nullable:
            if not _accepts_none(info):
                report.add(f"with.{key} gets input {node.name}, which may be null, but {capability.name} needs a value", step.id)
        for nested in _nested_inputs(node):
            if spec.inputs[nested].nullable:
                report.add(
                    f"with.{key} puts input {nested}, which may be null, inside a list or mapping; "
                    "a null there is not a valid value",
                    step.id,
                )
        if key in capability.handle_params:
            _check_handle_value(report, step.id, key, node, capability, capabilities, info)
        else:
            _check_plain_value(report, step.id, key, node, capabilities)
    for key, info in fields.items():
        if info.is_required() and with_.get(key) is None:
            report.add(f"with does not give {key}, which {capability.name} requires", step.id)
    if APPROVED in fields:
        quantity = with_.get(QUANTITY)
        if quantity is not None and spec.quantity_input_of(step.id) is None:
            report.add(f"with.quantity for {capability.name} is a quantity input, like $quantity", step.id)


def _step_kind(spec: WorkflowSpec, capabilities: CapabilityRegistry, step_id: str) -> tuple[bool, Any]:
    """(is a use step of a known capability, its output kind)."""
    step = spec.step(step_id)
    if not isinstance(step, UseStep):
        return False, None
    try:
        return True, capabilities.get(step.use).output_kind
    except UnknownCapability:
        return True, Ellipsis


def _check_handle_value(
    report: _Report,
    step_id: str,
    key: str,
    node: Value,
    capability: Capability[Any],
    capabilities: CapabilityRegistry,
    info: FieldInfo,
) -> None:
    spec = report.spec
    kinds = capability.handle_params[key]
    single_ok, list_ok = _shape(info.annotation)
    wanted = " or ".join(kinds)

    def one(item: Value) -> None:
        if isinstance(item, LimitRef):
            report.add(f"with.{key} takes a handle; a limit is a number", step_id)
            return
        if isinstance(item, ListValue | MapValue):
            report.add(f"with.{key} takes one handle in each place", step_id)
            return
        if not isinstance(item, Ref) or item.root is None:
            return
        if item.root == "input":
            if item.path:
                return
            source = spec.inputs[item.name]
            kind = source.handle_kind
            if kind is None:
                report.add(f"with.{key} takes a handle; input {item.name} is a {source.type}", step_id)
            elif kind not in kinds:
                report.add(f"with.{key} takes {wanted}; input {item.name} is {kind}", step_id)
            return
        if item.root != "step" or item.path not in ((), ("handle",)):
            return
        is_use, output = _step_kind(spec, capabilities, item.name)
        if not is_use:
            report.add(f"with.{key} takes a handle; only a use step's result is a handle, and {item.name} is not one", step_id)
        elif output is Ellipsis:
            return
        elif output is None:
            report.add(f"with.{key} takes a handle; step {item.name} returns none", step_id)
        elif output not in kinds:
            report.add(f"with.{key} takes {wanted}; step {item.name} returns {output}", step_id)

    if isinstance(node, ListValue):
        if not list_ok:
            report.add(f"with.{key} takes one handle, not a list", step_id)
        for item in node.items:
            one(item)
        return
    if isinstance(node, Ref) and not single_ok and list_ok:
        report.add(f"with.{key} takes a list of handles, like [${node.name}]", step_id)
    one(node)


def _check_plain_value(report: _Report, step_id: str, key: str, node: Value, capabilities: CapabilityRegistry) -> None:
    spec = report.spec
    for ref in references(node):
        if not isinstance(ref, Ref) or not ref.bare:
            continue
        if ref.root == "step":
            report.add(
                f"with.{key} takes a value, not a handle; read a field of step {ref.name}, like ${ref.name}.summary",
                step_id,
            )
        elif ref.root == "input" and spec.inputs[ref.name].handle_kind is not None:
            report.add(f"with.{key} takes a value, not a handle; input {ref.name} is a handle", step_id)


# Products


def _injecting_steps(spec: WorkflowSpec, capabilities: CapabilityRegistry) -> list[tuple[UseStep, Capability[Any], str]]:
    found = []
    for step in spec.steps:
        if not isinstance(step, UseStep):
            continue
        try:
            capability = capabilities.get(step.use)
        except UnknownCapability:
            continue
        quantity = spec.quantity_input_of(step.id)
        if APPROVED in capability.params.model_fields and quantity is not None:
            found.append((step, capability, quantity))
    return found


def approved_parameter(
    spec: WorkflowSpec, step_id: str, capabilities: CapabilityRegistry
) -> list[str] | RegistryDefault | None:
    """The ``approved`` value the runner passes to a ``use`` step, or None to pass nothing."""
    step = spec.step(step_id)
    if not isinstance(step, UseStep) or APPROVED not in capabilities.get(step.use).params.model_fields:
        return None
    return spec.approved_for(step_id)


def _check_products(report: _Report, capabilities: CapabilityRegistry, datasets: Mapping[str, CollectionConfig]) -> None:
    spec = report.spec
    products = spec.products
    if isinstance(products, dict):
        for key, listed in products.items():
            if key not in spec.inputs or spec.inputs[key].type != "quantity":
                report.add(f"products key {key} is not a quantity input")
            if isinstance(listed, tuple):
                for product in listed:
                    if product not in datasets:
                        report.add(f"product {product} is not in the dataset registry")
                    elif datasets[product].quantity is None:
                        report.add(f"product {product} records no quantity in the dataset registry")
    injecting = _injecting_steps(spec, capabilities)
    if injecting and products is None:
        report.add("products is required: steps " + ", ".join(s.id for s, _, _ in injecting) + " retrieve by quantity")
    if isinstance(products, dict):
        for step, _, quantity in injecting:
            if quantity not in products:
                report.add(f"products has no entry for {quantity}, which this step retrieves", step.id)
    _check_product_pins(report, injecting)


def _branch(spec: WorkflowSpec, step_id: str) -> dict[str, frozenset[str]]:
    """The options each choice input must equal for a step to run.

    Taken from the top-level ``and`` terms of the step's own ``when`` and of
    every step whose being skipped skips it (``spec.reads``), directly or
    through others; only ``input == "option"`` terms count, so any other
    condition says nothing here. A step named only in ``after`` or inside
    ``exists(...)`` can be skipped while this one runs, so its condition is
    not inherited.
    """
    found: dict[str, set[str]] = {}
    seen: set[str] = set()
    pending = [step_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        compiled = spec.compiled(current)
        if compiled.when is not None:
            for term in _conjuncts(compiled.when):
                pinned = _choice_equality(spec, term)
                if pinned is not None:
                    found.setdefault(pinned[0], set()).add(pinned[1])
        pending.extend(spec.reads(current))
    return {name: frozenset(options) for name, options in found.items()}


def _conjuncts(node: Expr) -> Iterator[Expr]:
    if isinstance(node, And):
        yield from _conjuncts(node.left)
        yield from _conjuncts(node.right)
    else:
        yield node


def _choice_equality(spec: WorkflowSpec, node: Expr) -> tuple[str, str] | None:
    if not isinstance(node, Compare) or node.op != "==":
        return None
    for side, other in ((node.left, node.right), (node.right, node.left)):
        if (
            isinstance(side, Ref) and side.root == "input" and side.bare and spec.inputs[side.name].type == "choice"
            and isinstance(other, Const) and isinstance(other.value, str)
        ):
            return side.name, other.value
    return None


def _exclusive(first: Mapping[str, frozenset[str]], second: Mapping[str, frozenset[str]]) -> bool:
    """Whether two steps can never both run: one choice input must equal different options for each."""
    return any(not first[name] & second[name] for name in first.keys() & second.keys())


def _check_product_pins(report: _Report, injecting: list[tuple[UseStep, Capability[Any], str]]) -> None:
    spec = report.spec
    groups: dict[str, list[str]] = {}
    for step, capability, quantity in injecting:
        if PRODUCT in capability.params.model_fields:
            groups.setdefault(quantity, []).append(step.id)
    for quantity, ids in groups.items():
        members = set(ids)
        choosers = [s for s in ids if spec.compiled(s).with_.get(PRODUCT) is None]
        branches = {s: _branch(spec, s) for s in choosers}
        for i, first in enumerate(choosers):
            for second in choosers[i + 1 :]:
                if not _exclusive(branches[first], branches[second]):
                    report.add(
                        f"steps {first}, {second} each choose the product for {quantity}; pin one with "
                        f"product: ${first}.product, or put them in exclusive branches on one choice input, "
                        f'like when: mode == "region" and when: mode == "point"'
                    )
        pins: dict[str, str] = {}
        for step_id in ids:
            node = spec.compiled(step_id).with_.get(PRODUCT)
            if node is None:
                continue
            if isinstance(node, Ref) and node.root == "step" and node.path == (PRODUCT,) and node.name in members:
                pins[step_id] = node.name
            else:
                report.add(f"with.product pins to another step retrieving {quantity}, like $<step>.product", step_id)
        if not choosers:
            report.add(f"no step chooses the product for {quantity}; steps {', '.join(ids)} pin to one another")
            continue
        looping: list[str] = []
        for step_id in pins:
            seen = {step_id}
            current = pins[step_id]
            while current in pins and current not in seen:
                seen.add(current)
                current = pins[current]
            if current in pins:
                looping.append(step_id)
        if looping:
            report.add(
                f"steps {', '.join(looping)} pin the product for {quantity} to one another, not to {choosers[0]}"
            )


# Outputs


def _check_outputs(report: _Report, capabilities: CapabilityRegistry, cyclic: bool) -> None:
    spec = report.spec
    outputs = spec.compiled_outputs

    def produced(label: str, refs: Iterable[Reference], optional: bool) -> None:
        if cyclic or optional:
            return
        for ref in refs:
            if isinstance(ref, Ref) and ref.root == "step" and not spec.always_runs(ref.name):
                report.add(f"{label} comes from step {ref.name}, which may not run; mark it optional")
                return

    for finding in outputs.findings:
        produced(f"finding {finding.name}", finding.references(), finding.optional)
        for ref in finding.references():
            if isinstance(ref, Ref) and ref.root == "step" and isinstance(spec.step(ref.name), AgentStep):
                report.add(f"finding {finding.name} reads agent step {ref.name}; a finding comes from capabilities, not a model")
                break

    def step_entries(section: str, entries: Sequence[CompiledOutputRef]) -> None:
        for entry in entries:
            ref = entry.ref
            if not isinstance(ref, Ref) or ref.root is None:
                continue
            if ref.root != "step":
                report.add(f"{section} entry {ref.text()} is not a step")
                continue
            produced(f"{section} entry {ref.text()}", (ref,), entry.optional)
            step = spec.step(ref.name)
            if isinstance(step, AgentStep):
                report.add(f"{section} entry {ref.text()} is an agent step, which returns no {'chart' if section == 'figures' else 'handle'}")
                continue
            if not isinstance(step, UseStep):
                continue
            try:
                capability = capabilities.get(step.use)
            except UnknownCapability:
                continue
            if section == "figures" and capability.group.split(".")[0] != RENDER_GROUP:
                report.add(f"figures entry {ref.text()} is not a render step")
            if section == "data" and capability.output_kind is None:
                report.add(f"data entry {ref.text()} returns no handle")

    step_entries("figures", outputs.figures)
    step_entries("data", outputs.data)
    interpretation = outputs.interpretation
    if interpretation is not None and isinstance(interpretation.ref, Ref) and interpretation.ref.root is not None:
        ref = interpretation.ref
        target = spec.step(ref.name) if ref.root == "step" else None
        if not isinstance(target, AgentStep):
            report.add("interpretation comes from an agent step")
        elif len(ref.path) != 1 or target.agent.returns.get(ref.path[0]) != "string":
            report.add(f"interpretation reads a string field of agent step {ref.name}, like ${ref.name}.text")
        else:
            produced("interpretation", (ref,), interpretation.optional)


def _check_validated(report: _Report) -> None:
    spec = report.spec
    if spec.status != "validated":
        return
    if not spec.benchmark:
        report.add("a validated spec lists its benchmark cases")
    if spec.validated_by is None:
        report.add("a validated spec names who signed it off in validated_by")
    if spec.validated_on is None:
        report.add("a validated spec gives the sign-off date in validated_on")


# Checks across specs


def check_set(specs: Sequence[tuple[str, WorkflowSpec]], known: WorkflowCatalogue | None = None) -> list[SpecError]:
    """Problems between specs: repeats, child workflows, and cycles through them.

    ``specs`` are (label, spec) pairs; ``known`` holds already-checked specs a
    child may also come from.
    """
    errors: list[SpecError] = []
    by_key: dict[tuple[str, int], tuple[str, WorkflowSpec]] = {}
    if known is not None:
        for spec in known:
            by_key[(spec.name, spec.version)] = (INLINE, spec)
    for label, spec in specs:
        key = (spec.name, spec.version)
        if key in by_key and (known is None or key not in known):
            errors.append(SpecError(f"{spec.name} v{spec.version} is also defined in {by_key[key][0]}", source=label, workflow=spec.name, version=spec.version))
            continue
        by_key[key] = (label, spec)

    for label, spec in specs:
        for step in spec.steps:
            if isinstance(step, WorkflowStep):
                _check_child(errors, label, spec, step, by_key)

    graph = {f"{n} v{v}": frozenset(f"{c} v{cv}" for c, cv in s.children()) for (n, v), (_, s) in by_key.items()}
    labels = {f"{n} v{v}": (label, s) for (n, v), (label, s) in by_key.items()}
    for component in strongly_connected(graph):
        member = sorted(component)[0]
        if len(component) > 1 or member in graph[member]:
            label, spec = labels[member]
            errors.append(
                SpecError(
                    "workflows " + ", ".join(sorted(component)) + " run one another in a cycle",
                    source=label,
                    workflow=spec.name,
                    version=spec.version,
                )
            )
    return errors


def _check_child(
    errors: list[SpecError],
    label: str,
    spec: WorkflowSpec,
    step: WorkflowStep,
    by_key: Mapping[tuple[str, int], tuple[str, WorkflowSpec]],
) -> None:
    def add(message: str) -> None:
        errors.append(SpecError(message, source=label, workflow=spec.name, version=spec.version, step=step.id))

    found = by_key.get((step.workflow, step.version))
    if found is None:
        add(f"runs {step.workflow} v{step.version}, which is not loaded")
        return
    child = found[1]
    if spec.status == "validated" and child.status != "validated":
        add(f"a validated spec runs only validated workflows; {child.name} v{child.version} is a draft")
    for given in step.with_:
        if given not in child.inputs:
            add(f"with gives {given}, which is not an input of {child.name}")
    for name, declared in child.inputs.items():
        if not declared.optional and name not in step.with_:
            add(f"with does not give {name}, which {child.name} requires")
    for answered in step.answers:
        if child.root_of(answered) != "step" or not isinstance(child.step(answered), AskStep):
            add(f"answers {answered}, which is not an ask step of {child.name}")


# Entry points


def build_catalogue(
    specs: Sequence[tuple[str, WorkflowSpec]],
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
) -> WorkflowCatalogue:
    """Check (label, spec) pairs together and catalogue them; raises SpecErrors."""
    errors: list[SpecError] = []
    for label, spec in specs:
        errors.extend(check_spec(spec, capabilities, datasets, label=label))
    errors.extend(check_set(specs))
    if errors:
        raise SpecErrors(errors)
    return WorkflowCatalogue(spec for _, spec in specs)


def load_spec(
    source: str | Mapping[str, Any],
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
    *,
    label: str = INLINE,
    catalogue: WorkflowCatalogue | None = None,
) -> WorkflowSpec:
    """One checked spec; its child workflows come from ``catalogue``. Raises SpecErrors."""
    spec = parse_spec(source, label=label)
    errors = check_spec(spec, capabilities, datasets, label=label)
    errors.extend(check_set([(label, spec)], catalogue))
    if errors:
        raise SpecErrors(errors)
    return spec


def load_specs(
    sources: Mapping[str, str | Mapping[str, Any]],
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
) -> WorkflowCatalogue:
    """Specs given as {label: YAML text or mapping}, checked together. Raises SpecErrors."""
    errors: list[SpecError] = []
    parsed: list[tuple[str, WorkflowSpec]] = []
    for label, source in sources.items():
        try:
            parsed.append((label, parse_spec(source, label=label)))
        except SpecErrors as failure:
            errors.extend(failure.errors)
    return _finish(parsed, errors, capabilities, datasets)


def _finish(
    parsed: list[tuple[str, WorkflowSpec]],
    errors: list[SpecError],
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
) -> WorkflowCatalogue:
    try:
        catalogue = build_catalogue(parsed, capabilities, datasets)
    except SpecErrors as failure:
        errors.extend(failure.errors)
    if errors:
        raise SpecErrors(errors)
    return catalogue


def _version_stem(stem: str) -> int | None:
    return int(stem) if stem.isdigit() and not stem.startswith("0") else None


def load_packs(
    root: Path,
    capabilities: CapabilityRegistry,
    datasets: Mapping[str, CollectionConfig],
) -> WorkflowCatalogue:
    """Every spec at ``<root>/<pack>/workflows/<name>/<version>.yaml``. Raises SpecErrors."""
    if not root.is_dir():
        raise SpecErrors([SpecError("the pack root is not a directory", source=str(root))])
    errors: list[SpecError] = []
    parsed: list[tuple[str, WorkflowSpec]] = []
    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        workflows = pack / "workflows"
        if not workflows.is_dir():
            continue
        for entry in sorted(workflows.iterdir()):
            label = entry.relative_to(root).as_posix()
            if not entry.is_dir():
                errors.append(SpecError("is not a workflow directory", source=label))
                continue
            for path in sorted(entry.iterdir()):
                label = path.relative_to(root).as_posix()
                version = _version_stem(path.stem) if path.is_file() and path.suffix == ".yaml" else None
                if version is None:
                    errors.append(SpecError("a spec file is named by its version, like 1.yaml", source=label))
                    continue
                try:
                    spec = parse_spec(path.read_text(encoding="utf-8"), label=label)
                except SpecErrors as failure:
                    errors.extend(failure.errors)
                    continue
                mismatch = False
                if spec.name != entry.name:
                    errors.append(SpecError(f"the spec's name is {spec.name} but its directory is {entry.name}", source=label, workflow=spec.name, version=spec.version))
                    mismatch = True
                if spec.version != version:
                    errors.append(SpecError(f"the spec's version is {spec.version} but its file is {path.name}", source=label, workflow=spec.name, version=spec.version))
                    mismatch = True
                if not mismatch:
                    parsed.append((label, spec))
    return _finish(parsed, errors, capabilities, datasets)
