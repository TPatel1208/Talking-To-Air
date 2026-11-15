"""The reference and condition language of a workflow spec.

A deliberately small language: references, comparisons, ``and``/``or``/``not``,
literals and two functions. It is tokenized and parsed by hand into frozen
nodes; nothing is ever passed to ``eval``.

References
    ``$name`` is an input or a step's result; ``$name.a.b.0`` walks its fields
    (dotted lower-case names and list indexes, as ``result.walk_path``).
    ``limits.name`` reads a limit and is always written without ``$``.

Conditions (``when``)
    Comparisons ``== != < <= > >=`` (they do not chain), ``and``, ``or``,
    ``not``, parentheses, literals (numbers, double-quoted strings with ``\\"``
    and ``\\\\`` escapes, ``true``, ``false``, ``null``), references, and the
    functions ``count(ref)`` and ``exists(ref)``. Precedence from loosest:
    ``or``, ``and``, ``not``, comparison. A name written without ``$`` parses
    as a reference with ``dollar=False``; the loader accepts it in a condition
    only when it names an input.

Text placeholders (``method``, ``reason``)
    Each ``{...}`` holds exactly one reference or one ``count``/``exists``
    call, written as in a condition, with the ``$`` optional: ``{$days}`` and
    ``{days}`` are the same. A name resolves to an input or a step; in a
    ``use`` step's ``method`` a name that is neither reads the step's own
    result, as in ``{product}`` or ``{summary.mean}``. A brace outside a
    placeholder is an error.

Values (``with`` and similar)
    JSON-like: scalars, lists and mappings. A string starting with ``$`` or
    ``limits.`` is a reference and must be one exactly; any other leaf is a
    literal.

Binding
    Parsing does not know which names are inputs and which are steps.
    ``bind`` sets each reference's ``root`` from the spec's names; evaluation
    needs bound nodes.

Bare step references
    ``$step`` with no path reads ``StepView.value()``. What that is depends on
    where it appears and is the consumer's decision: in ``with`` and
    ``outputs.data`` it means the step's handle, in ``outputs.figures`` its
    charts, and for an ``ask`` step its answer.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

from tta_backend.harness.capabilities.result import walk_path

RootKind = Literal["input", "step", "own"]
CompareOp = Literal["==", "!=", "<", "<=", ">", ">="]
FunctionName = Literal["count", "exists"]
ValueType = Literal["number", "string", "boolean", "null"]
Scalar = str | int | float | bool | None

LIMITS = "limits"
KEYWORDS = frozenset({"and", "or", "not", "true", "false", "null"})
FUNCTIONS: frozenset[str] = frozenset({"count", "exists"})
# Names an input or step may not take, since the language gives them a meaning.
RESERVED = KEYWORDS | FUNCTIONS | {LIMITS}

NAME = re.compile(r"[a-z][a-z0-9_]*")
_PART = r"(?:[a-z][a-z0-9_]*|[0-9]+)"
_REF = re.compile(r"\$?[a-z][a-z0-9_]*(?:\." + _PART + r")*")
_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_COMPARISONS = ("==", "!=", "<=", ">=", "<", ">")
# How deeply parentheses, ``not`` and nested lists or mappings may go.
MAX_DEPTH = 50
_OPERATORS = (*_COMPARISONS, "(", ")")


# Nodes


@dataclass(frozen=True)
class Ref:
    """``$name.path`` (or ``name.path`` where ``$`` is optional); ``root`` is set by ``bind``."""

    name: str
    path: tuple[str, ...] = ()
    dollar: bool = True
    root: RootKind | None = None

    @property
    def bare(self) -> bool:
        """No path: the whole input or step result."""
        return not self.path

    def dotted_path(self) -> str:
        return ".".join(self.path)

    def text(self) -> str:
        return ("$" if self.dollar else "") + ".".join((self.name, *self.path))


@dataclass(frozen=True)
class LimitRef:
    """``limits.name``."""

    name: str

    def text(self) -> str:
        return f"{LIMITS}.{self.name}"


@dataclass(frozen=True)
class Const:
    value: Any


@dataclass(frozen=True)
class Compare:
    op: CompareOp
    left: Expr
    right: Expr


@dataclass(frozen=True)
class And:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Or:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Not:
    operand: Expr


@dataclass(frozen=True)
class Func:
    name: FunctionName
    arg: Ref | LimitRef


Reference = Ref | LimitRef
Expr = Ref | LimitRef | Const | Compare | And | Or | Not | Func
Slot = Ref | LimitRef | Func


@dataclass(frozen=True)
class ListValue:
    items: tuple[Value, ...]


@dataclass(frozen=True)
class MapValue:
    items: tuple[tuple[str, Value], ...]

    def get(self, key: str) -> Value | None:
        for name, value in self.items:
            if name == key:
                return value
        return None

    def keys(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.items)


Value = Const | Ref | LimitRef | ListValue | MapValue


@dataclass(frozen=True)
class Template:
    """Text with placeholders; ``parts`` alternates literal text and slots."""

    parts: tuple[str | Slot, ...]

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(part for part in self.parts if not isinstance(part, str))


Node = Expr | Value | Template


# Errors


class ExpressionSyntaxError(ValueError):
    """Text that is not in the language; ``position`` is a 0-based character offset."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} at position {position}")
        self.message = message
        self.position = position


class EvaluationError(Exception):
    """A bound node could not be evaluated against the run's values."""


class StepUnavailable(EvaluationError):
    """A reference reads a step that was skipped or has not produced a result."""

    def __init__(self, step_id: str) -> None:
        super().__init__(f"step {step_id} has no result")
        self.step_id = step_id


class MissingValue(EvaluationError):
    """A reference's path names no field of an available value."""

    def __init__(self, reference: str) -> None:
        super().__init__(f"{reference} has no value")
        self.reference = reference


class ValueTypeError(EvaluationError):
    """A value has the wrong type for where it is used."""


class IncomparableValues(ValueTypeError):
    def __init__(self, op: str, left: str, right: str) -> None:
        super().__init__(f"cannot compare {left} with {right} using {op}")
        self.op = op
        self.left = left
        self.right = right


# Tokenizer


TokenKind = Literal["ref", "number", "string", "op", "end"]


@dataclass(frozen=True)
class _Token:
    kind: TokenKind
    text: str
    position: int
    value: Any = None


def _tokens(text: str) -> list[_Token]:
    out: list[_Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            start = i
            i += 1
            chars: list[str] = []
            while True:
                if i >= len(text):
                    raise ExpressionSyntaxError("unterminated string", start)
                c = text[i]
                if c == "\\":
                    if i + 1 < len(text) and text[i + 1] in '"\\':
                        chars.append(text[i + 1])
                        i += 2
                        continue
                    raise ExpressionSyntaxError("unknown escape in string", i)
                if c == '"':
                    i += 1
                    break
                chars.append(c)
                i += 1
            out.append(_Token("string", text[start:i], start, "".join(chars)))
            continue
        number = _NUMBER.match(text, i)
        if number and (ch != "-" or number.end() > i + 1):
            raw = number.group()
            value: int | float = float(raw) if any(c in raw for c in ".eE") else int(raw)
            if isinstance(value, float) and not math.isfinite(value):
                raise ExpressionSyntaxError("a number is finite", i)
            out.append(_Token("number", raw, i, value))
            i = number.end()
            continue
        ref = _REF.match(text, i)
        if ref:
            if ref.end() < len(text) and (text[ref.end()] == "." or text[ref.end()].isalnum()):
                raise ExpressionSyntaxError("a name is lower-case letters, digits and underscores, with dotted parts", ref.end())
            out.append(_Token("ref", ref.group(), i))
            i = ref.end()
            continue
        for op in _OPERATORS:
            if text.startswith(op, i):
                out.append(_Token("op", op, i))
                i += len(op)
                break
        else:
            if ch == "=":
                raise ExpressionSyntaxError("use == to compare", i)
            if ch == "$":
                raise ExpressionSyntaxError("$ is followed by a lower-case name", i)
            raise ExpressionSyntaxError("unexpected character", i)
    out.append(_Token("end", "", len(text)))
    return out


# Parser


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.i = 0
        self.depth = 0

    def deeper(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExpressionSyntaxError(f"an expression nests more than {MAX_DEPTH} deep", self.token.position)

    @property
    def token(self) -> _Token:
        return self.tokens[self.i]

    def advance(self) -> _Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def at_word(self, word: str) -> bool:
        return self.token.kind == "ref" and self.token.text == word

    def at_op(self, op: str) -> bool:
        return self.token.kind == "op" and self.token.text == op

    def expect_op(self, op: str) -> None:
        if not self.at_op(op):
            raise ExpressionSyntaxError(f"expected {op}", self.token.position)
        self.advance()

    def finish(self) -> None:
        if self.token.kind != "end":
            raise ExpressionSyntaxError("unexpected text after the expression", self.token.position)

    def expression(self) -> Expr:
        left = self.conjunction()
        while self.at_word("or"):
            self.advance()
            left = Or(left, self.conjunction())
        return left

    def conjunction(self) -> Expr:
        left = self.negation()
        while self.at_word("and"):
            self.advance()
            left = And(left, self.negation())
        return left

    def negation(self) -> Expr:
        if self.at_word("not"):
            self.advance()
            self.deeper()
            inner = self.negation()
            self.depth -= 1
            return Not(inner)
        return self.comparison()

    def comparison(self) -> Expr:
        left = self.operand()
        if self.at_comparison():
            op = cast(CompareOp, self.advance().text)
            right = self.operand()
            if self.at_comparison():
                raise ExpressionSyntaxError("comparisons do not chain; join them with and", self.token.position)
            return Compare(op, left, right)
        return left

    def at_comparison(self) -> bool:
        return self.token.kind == "op" and self.token.text in _COMPARISONS

    def operand(self) -> Expr:
        token = self.token
        if token.kind == "op" and token.text == "(":
            self.advance()
            self.deeper()
            inner = self.expression()
            self.expect_op(")")
            self.depth -= 1
            return inner
        if token.kind in ("number", "string"):
            self.advance()
            return Const(token.value)
        if token.kind == "ref":
            word = token.text
            if word in ("true", "false"):
                self.advance()
                return Const(word == "true")
            if word == "null":
                self.advance()
                return Const(None)
            if word in KEYWORDS:
                raise ExpressionSyntaxError(f"expected a value before {word}", token.position)
            if word in FUNCTIONS:
                self.advance()
                self.expect_op("(")
                arg_token = self.token
                if arg_token.kind != "ref":
                    raise ExpressionSyntaxError(f"{word} takes one reference", arg_token.position)
                arg = self.reference()
                self.expect_op(")")
                return Func(cast(FunctionName, word), arg)
            return self.reference()
        raise ExpressionSyntaxError("expected a value", token.position)

    def reference(self) -> Ref | LimitRef:
        token = self.advance()
        return _reference(token.text, token.position)


def _reference(text: str, position: int) -> Ref | LimitRef:
    dollar = text.startswith("$")
    parts = (text[1:] if dollar else text).split(".")
    head = parts[0]
    if head == LIMITS:
        if dollar:
            raise ExpressionSyntaxError("limits are written without $", position)
        if len(parts) != 2 or not NAME.fullmatch(parts[1]):
            raise ExpressionSyntaxError("a limit is written limits.name", position)
        return LimitRef(parts[1])
    if head in KEYWORDS or head in FUNCTIONS:
        raise ExpressionSyntaxError(f"{head} is not a name", position)
    return Ref(head, tuple(parts[1:]), dollar)


def parse_condition(text: str) -> Expr:
    """Parse a ``when`` condition."""
    if not text.strip():
        raise ExpressionSyntaxError("a condition is empty", 0)
    parser = _Parser(text)
    node = parser.expression()
    parser.finish()
    return node


def parse_reference(text: str) -> Ref | LimitRef:
    """Parse exactly one reference: ``$name.path`` or ``limits.name``."""
    stripped = text.strip()
    match = _REF.fullmatch(stripped)
    if match is None:
        raise ExpressionSyntaxError("a reference is $name, $name.field or limits.name", 0)
    node = _reference(stripped, 0)
    if isinstance(node, Ref) and not node.dollar:
        raise ExpressionSyntaxError("a reference to an input or step starts with $", 0)
    return node


def is_reference_text(value: str) -> bool:
    """Whether a string in a value position is read as a reference."""
    return value.startswith("$") or value.startswith(f"{LIMITS}.")


def parse_value(raw: Any, _depth: int = 0) -> Value:
    """Parse a JSON-like value whose string leaves may be references."""
    if _depth > MAX_DEPTH:
        raise ExpressionSyntaxError(f"a value nests more than {MAX_DEPTH} deep", 0)
    if isinstance(raw, str):
        if is_reference_text(raw):
            return parse_reference(raw)
        return Const(raw)
    if raw is None or isinstance(raw, bool | int):
        return Const(raw)
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ExpressionSyntaxError("a number is finite", 0)
        return Const(raw)
    if isinstance(raw, list | tuple):
        return ListValue(tuple(parse_value(item, _depth + 1) for item in raw))
    if isinstance(raw, Mapping):
        items: list[tuple[str, Value]] = []
        for key, item in raw.items():
            if not isinstance(key, str):
                raise ExpressionSyntaxError("a mapping key is text", 0)
            items.append((key, parse_value(item, _depth + 1)))
        return MapValue(tuple(items))
    raise ExpressionSyntaxError("a value is text, a number, true, false, null, a list or a mapping", 0)


def parse_template(text: str) -> Template:
    """Parse text with ``{...}`` placeholders."""
    parts: list[str | Slot] = []
    literal: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "}":
            raise ExpressionSyntaxError("unmatched }", i)
        if ch != "{":
            literal.append(ch)
            i += 1
            continue
        end = text.find("}", i + 1)
        nested = text.find("{", i + 1)
        if end < 0 or (0 <= nested < end):
            raise ExpressionSyntaxError("unmatched {", i)
        inner = text[i + 1 : end]
        if not inner.strip():
            raise ExpressionSyntaxError("a placeholder is empty", i)
        try:
            node = parse_condition(inner)
        except ExpressionSyntaxError as error:
            raise ExpressionSyntaxError(error.message, i + 1 + error.position) from None
        if not isinstance(node, Ref | LimitRef | Func):
            raise ExpressionSyntaxError("a placeholder holds one reference, count(...) or exists(...)", i)
        if literal:
            parts.append("".join(literal))
            literal = []
        parts.append(node)
        i = end + 1
    if literal:
        parts.append("".join(literal))
    return Template(tuple(parts))


# Walking


def _children(node: Node) -> Iterator[Node]:
    if isinstance(node, Compare | And | Or):
        yield node.left
        yield node.right
    elif isinstance(node, Not):
        yield node.operand
    elif isinstance(node, Func):
        yield node.arg
    elif isinstance(node, ListValue):
        yield from node.items
    elif isinstance(node, MapValue):
        for _, item in node.items:
            yield item
    elif isinstance(node, Template):
        yield from node.slots


def references(node: Node) -> tuple[Reference, ...]:
    """Every reference a node reads, in order of appearance."""
    if isinstance(node, Ref | LimitRef):
        return (node,)
    found: list[Reference] = []
    for child in _children(node):
        found.extend(references(child))
    return tuple(found)


def bind(node: Any, roots: Mapping[str, RootKind], *, own: bool = False) -> Any:
    """``node`` with each reference's ``root`` set from ``roots``.

    With ``own``, a name not in ``roots`` reads the step's own result; otherwise
    it is left unbound (``root`` None) for the loader to report.
    """
    if isinstance(node, Ref):
        root = roots.get(node.name)
        if root is None and own:
            root = "own"
        return replace(node, root=root)
    if isinstance(node, Compare):
        return Compare(node.op, bind(node.left, roots, own=own), bind(node.right, roots, own=own))
    if isinstance(node, And):
        return And(bind(node.left, roots, own=own), bind(node.right, roots, own=own))
    if isinstance(node, Or):
        return Or(bind(node.left, roots, own=own), bind(node.right, roots, own=own))
    if isinstance(node, Not):
        return Not(bind(node.operand, roots, own=own))
    if isinstance(node, Func):
        return Func(node.name, bind(node.arg, roots, own=own))
    if isinstance(node, ListValue):
        return ListValue(tuple(bind(item, roots, own=own) for item in node.items))
    if isinstance(node, MapValue):
        return MapValue(tuple((key, bind(item, roots, own=own)) for key, item in node.items))
    if isinstance(node, Template):
        return Template(tuple(part if isinstance(part, str) else bind(part, roots, own=own) for part in node.parts))
    return node


# Static type checks


def _const_type(value: Any) -> ValueType | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def type_problems(node: Expr, type_of: Callable[[Reference], ValueType | None]) -> list[str]:
    """Type mistakes visible before running, as fixed sentences.

    ``type_of`` gives a reference's type where the spec states it, else None.
    """
    problems: list[str] = []

    def kind(expr: Expr) -> ValueType | None:
        if isinstance(expr, Const):
            return _const_type(expr.value)
        if isinstance(expr, Ref | LimitRef):
            return type_of(expr)
        if isinstance(expr, Func):
            return "number" if expr.name == "count" else "boolean"
        return "boolean"

    def visit(expr: Expr) -> None:
        if isinstance(expr, Compare):
            visit(expr.left)
            visit(expr.right)
            left, right = kind(expr.left), kind(expr.right)
            if left is not None and right is not None and not _comparable(expr.op, left, right):
                problems.append(f"cannot compare {left} with {right} using {expr.op}")
        elif isinstance(expr, And | Or):
            word = "and" if isinstance(expr, And) else "or"
            for side in (expr.left, expr.right):
                visit(side)
                side_type = kind(side)
                if side_type is not None and side_type != "boolean":
                    problems.append(f"{word} joins true or false values, not {side_type}")
        elif isinstance(expr, Not):
            visit(expr.operand)
            side_type = kind(expr.operand)
            if side_type is not None and side_type != "boolean":
                problems.append(f"not takes a true or false value, not {side_type}")

    visit(node)
    top = kind(node)
    if not isinstance(node, Compare | And | Or | Not) and top is not None and top != "boolean":
        problems.append(f"a condition is true or false, not {top}")
    return problems


def _comparable(op: str, left: ValueType, right: ValueType) -> bool:
    if op in ("==", "!="):
        return left == right or "null" in (left, right)
    return left == right and left in ("number", "string")


# Evaluation


class StepView(Protocol):
    """A finished step's result as a reference reads it."""

    def read(self, path: str) -> Any:
        """The value at a dotted path; raises LookupError when there is none."""
        ...

    def value(self) -> Any:
        """What a bare ``$step`` means where it appears; the consumer decides."""
        ...


class Scope(Protocol):
    """The run's values, supplied by the runner."""

    def input(self, name: str) -> Any: ...

    def step(self, step_id: str) -> StepView | None:
        """The step's result, or None when it was skipped or has none."""
        ...

    def limit(self, name: str) -> Any: ...


def _read(ref: Reference, scope: Scope, own: StepView | None) -> Any:
    if isinstance(ref, LimitRef):
        return scope.limit(ref.name)
    if ref.root is None:
        raise ValueError(f"reference {ref.text()} is not bound")
    if ref.root == "input":
        value = scope.input(ref.name)
        if not ref.path:
            return value
        try:
            return walk_path(value, ref.dotted_path())
        except LookupError:
            raise MissingValue(ref.text()) from None
    if ref.root == "own":
        if own is None:
            raise ValueError(f"reference {ref.text()} reads the step's own result, which was not given")
        try:
            return own.read(".".join((ref.name, *ref.path)))
        except LookupError:
            raise MissingValue(ref.text()) from None
    view = scope.step(ref.name)
    if view is None:
        raise StepUnavailable(ref.name)
    if not ref.path:
        return view.value()
    try:
        return view.read(ref.dotted_path())
    except LookupError:
        raise MissingValue(ref.text()) from None


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "list"
    if isinstance(value, dict):
        return "mapping"
    return "value"


def _compare(op: CompareOp, left: Any, right: Any) -> bool:
    lt, rt = _type_name(left), _type_name(right)
    if op in ("==", "!="):
        if lt != rt and "null" not in (lt, rt):
            raise IncomparableValues(op, lt, rt)
        equal = left == right and lt == rt
        return equal if op == "==" else not equal
    if lt != rt or lt not in ("number", "string"):
        raise IncomparableValues(op, lt, rt)
    if op == "<":
        return bool(left < right)
    if op == "<=":
        return bool(left <= right)
    if op == ">":
        return bool(left > right)
    return bool(left >= right)


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueTypeError(f"{where} needs true or false, not {_type_name(value)}")
    return value


def evaluate(node: Expr, scope: Scope, *, own: StepView | None = None) -> Any:
    """Evaluate a bound expression.

    Raises StepUnavailable for a reference to a skipped or absent step, so
    the runner can skip what depends on it; MissingValue for a path with no
    value; IncomparableValues or ValueTypeError for a type mismatch.
    ``exists`` returns False instead of raising for the first two, and for
    a null value.
    """
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Ref | LimitRef):
        return _read(node, scope, own)
    if isinstance(node, Func):
        if node.name == "exists":
            try:
                return _read(node.arg, scope, own) is not None
            except (StepUnavailable, MissingValue):
                return False
        value = _read(node.arg, scope, own)
        if not isinstance(value, list | tuple):
            raise ValueTypeError(f"count takes a list, not {_type_name(value)}")
        return len(value)
    if isinstance(node, Compare):
        return _compare(node.op, evaluate(node.left, scope, own=own), evaluate(node.right, scope, own=own))
    if isinstance(node, Not):
        return not _boolean(evaluate(node.operand, scope, own=own), "not")
    if isinstance(node, And):
        if not _boolean(evaluate(node.left, scope, own=own), "and"):
            return False
        return _boolean(evaluate(node.right, scope, own=own), "and")
    if not _boolean(evaluate(node.left, scope, own=own), "or"):
        return _boolean(evaluate(node.right, scope, own=own), "or")
    return True


def evaluate_condition(node: Expr, scope: Scope) -> bool:
    """Evaluate a ``when``; a result that is not true or false raises ValueTypeError."""
    return _boolean(evaluate(node, scope), "a condition")


def evaluate_value(node: Value, scope: Scope) -> Any:
    """The plain value of a bound ``with``-style value."""
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Ref | LimitRef):
        return _read(node, scope, None)
    if isinstance(node, ListValue):
        return [evaluate_value(item, scope) for item in node.items]
    return {key: evaluate_value(item, scope) for key, item in node.items}


def render_value(value: Any) -> str:
    """A value as it appears in filled text."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list | tuple):
        return ", ".join(render_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(", ", ": "), default=str)
    return str(value)


def render_template(template: Template, scope: Scope, *, own: StepView | None = None) -> str:
    """Fill a bound template's placeholders."""
    return "".join(
        part if isinstance(part, str) else render_value(evaluate(part, scope, own=own)) for part in template.parts
    )
