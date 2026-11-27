"""The spec expression language: tokenizer, parser, binding, static types, evaluation and text."""
import unittest
from typing import Any

from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.workflows.expressions import (
    MAX_DEPTH,
    And,
    Compare,
    Const,
    ExpressionSyntaxError,
    Func,
    IncomparableValues,
    LimitRef,
    ListValue,
    MapValue,
    MissingValue,
    Not,
    Or,
    Ref,
    StepUnavailable,
    Template,
    ValueTypeError,
    bind,
    evaluate,
    evaluate_condition,
    evaluate_value,
    parse_condition,
    parse_reference,
    parse_template,
    parse_value,
    references,
    render_template,
    type_problems,
)

ROOTS: dict[str, Any] = {"quantity": "input", "days": "step", "diff": "step", "skipped": "step", "mode": "input"}


class _View:
    def __init__(self, whole: Any, result: CapabilityResult | None = None) -> None:
        self.whole = whole
        self.result = result

    def read(self, path: str) -> Any:
        if self.result is None:
            raise LookupError(path)
        return self.result.read(path)

    def value(self) -> Any:
        return self.whole


class _Scope:
    def __init__(self) -> None:
        self.inputs = {"quantity": "no2_tropospheric_column", "mode": "region"}
        diff = CapabilityResult(
            status="done", handle="grid_" + "a" * 32, kind="grid", unit="ppb", product="TEMPO_NO2",
            summary={"mean": 1.5, "values": [1, 2]}, values={"rule": "same weekday"},
        )
        self.steps: dict[str, _View | None] = {
            "days": _View(["2024-06-01", "2024-06-02", "2024-06-03"]),
            "diff": _View("grid_" + "a" * 32, diff),
            "skipped": None,
        }
        self.limits = {"min_days": 2, "min_coverage": 0.3}

    def input(self, name: str) -> Any:
        return self.inputs[name]

    def step(self, step_id: str) -> _View | None:
        return self.steps[step_id]

    def limit(self, name: str) -> Any:
        return self.limits[name]


def _cond(text: str) -> Any:
    return bind(parse_condition(text), ROOTS)


class ParserTests(unittest.TestCase):
    def test_structure(self):
        a, b, c = Ref("a", dollar=False), Ref("b", dollar=False), Ref("c", dollar=False)
        cases = {
            "a == 1": Compare("==", a, Const(1)),
            "a != 1.5": Compare("!=", a, Const(1.5)),
            "a < -2": Compare("<", a, Const(-2)),
            "a <= 2": Compare("<=", a, Const(2)),
            "a > 2": Compare(">", a, Const(2)),
            "a >= 2": Compare(">=", a, Const(2)),
            'a == "x\\"y\\\\"': Compare("==", a, Const('x"y\\')),
            "a == true": Compare("==", a, Const(True)),
            "a == false": Compare("==", a, Const(False)),
            "a == null": Compare("==", a, Const(None)),
            "a or b and c": Or(a, And(b, c)),
            "(a or b) and c": And(Or(a, b), c),
            "not a and b": And(Not(a), b),
            "not not a": Not(Not(a)),
            "not a == 1": Not(Compare("==", a, Const(1))),
            "a and b and c": And(And(a, b), c),
            "count($days) < limits.min_days": Compare("<", Func("count", Ref("days")), LimitRef("min_days")),
            "exists($diff.summary.values.0)": Func("exists", Ref("diff", ("summary", "values", "0"))),
            "$diff.summary.mean": Ref("diff", ("summary", "mean")),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_condition(text), expected)

    def test_bad_syntax_names_the_position(self):
        cases = {
            "a = 1": ("use == to compare", 2),
            "a < b < c": ("comparisons do not chain; join them with and", 6),
            "(a": ("expected )", 2),
            "a)": ("unexpected text after the expression", 1),
            "count(1)": ("count takes one reference", 6),
            "count $days": ("expected (", 6),
            "a and": ("expected a value", 5),
            "and a": ("expected a value before and", 0),
            '"abc': ("unterminated string", 0),
            '"a\\n"': ("unknown escape in string", 2),
            "limits.x.y": ("a limit is written limits.name", 0),
            "$limits.x": ("limits are written without $", 0),
            "$": ("$ is followed by a lower-case name", 0),
            "Mode == 1": ("unexpected character", 0),
            "a.": ("a name is lower-case letters, digits and underscores, with dotted parts", 1),
            "a # b": ("unexpected character", 2),
            "": ("a condition is empty", 0),
        }
        for text, (message, position) in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ExpressionSyntaxError) as caught:
                    parse_condition(text)
                self.assertEqual((caught.exception.message, caught.exception.position), (message, position))

    def test_reference(self):
        self.assertEqual(parse_reference("$a.b.0"), Ref("a", ("b", "0")))
        self.assertEqual(parse_reference("limits.x"), LimitRef("x"))
        for bad in ("a.b", "$a b", "$A", "$a.", "count($a)"):
            with self.subTest(bad=bad), self.assertRaises(ExpressionSyntaxError):
                parse_reference(bad)


class LimitTests(unittest.TestCase):
    def test_deep_nesting_is_a_syntax_error_not_a_crash(self):
        for text in ("(" * 3000 + "$x" + ")" * 3000, "not " * 3000 + "$x", "(" * (MAX_DEPTH + 1) + "$x" + ")" * (MAX_DEPTH + 1)):
            with self.subTest(text=text[:12]), self.assertRaises(ExpressionSyntaxError) as caught:
                parse_condition(text)
            self.assertIn(f"more than {MAX_DEPTH} deep", caught.exception.message)
        parse_condition("(" * MAX_DEPTH + "$x" + ")" * MAX_DEPTH)
        parse_condition("not " * MAX_DEPTH + "$x")

    def test_deeply_nested_values_are_refused(self):
        deep: Any = 1
        for _ in range(3000):
            deep = [deep]
        with self.assertRaises(ExpressionSyntaxError):
            parse_value(deep)
        mapping: Any = 1
        for _ in range(MAX_DEPTH + 1):
            mapping = {"a": mapping}
        with self.assertRaises(ExpressionSyntaxError):
            parse_value(mapping)

    def test_a_number_too_large_to_be_finite_is_refused(self):
        for text in ("$x < 1e999", "$x > -1e999"):
            with self.subTest(text=text), self.assertRaises(ExpressionSyntaxError) as caught:
                parse_condition(text)
            self.assertEqual(caught.exception.message, "a number is finite")


class ValueTests(unittest.TestCase):
    def test_references_and_literals(self):
        node = parse_value({"data": "$diff", "dates": ["$days.0", "2024-01-01"], "n": 3, "flag": True, "none": None,
                            "min": "limits.min_days", "style": "diverging", "nested": {"x": "$quantity"}})
        self.assertEqual(node, MapValue((
            ("data", Ref("diff")),
            ("dates", ListValue((Ref("days", ("0",)), Const("2024-01-01")))),
            ("n", Const(3)),
            ("flag", Const(True)),
            ("none", Const(None)),
            ("min", LimitRef("min_days")),
            ("style", Const("diverging")),
            ("nested", MapValue((("x", Ref("quantity")),))),
        )))
        self.assertEqual(
            [r.text() for r in references(node)],
            ["$diff", "$days.0", "limits.min_days", "$quantity"],
        )

    def test_a_string_starting_with_dollar_must_be_a_reference(self):
        with self.assertRaises(ExpressionSyntaxError):
            parse_value("$5 off")

    def test_evaluate_value(self):
        node = bind(parse_value({"a": "$diff", "b": ["$diff.unit", "limits.min_days"], "c": "$quantity"}), ROOTS)
        self.assertEqual(
            evaluate_value(node, _Scope()),
            {"a": "grid_" + "a" * 32, "b": ["ppb", 2], "c": "no2_tropospheric_column"},
        )


class BindTests(unittest.TestCase):
    def test_roots(self):
        node = bind(parse_condition("$days == quantity and unknown"), ROOTS)
        self.assertEqual([r.root for r in references(node) if isinstance(r, Ref)], ["step", "input", None])

    def test_own(self):
        template = bind(parse_template("{product} for {quantity}"), ROOTS, own=True)
        self.assertEqual([r.root for r in references(template) if isinstance(r, Ref)], ["own", "input"])


class TemplateTests(unittest.TestCase):
    def test_parse(self):
        template = parse_template("Only {count($days)} days; at least {limits.min_days} from {candidates.rule}.")
        self.assertEqual(template, Template((
            "Only ", Func("count", Ref("days")), " days; at least ", LimitRef("min_days"),
            " from ", Ref("candidates", ("rule",), dollar=False), ".",
        )))

    def test_bad(self):
        cases = {
            "a {b": "unmatched {",
            "a } b": "unmatched }",
            "a {{b}}": "unmatched {",
            "a {} b": "a placeholder is empty",
            "a {x == 1} b": "a placeholder holds one reference, count(...) or exists(...)",
            "a {3} b": "a placeholder holds one reference, count(...) or exists(...)",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ExpressionSyntaxError) as caught:
                    parse_template(text)
                self.assertEqual(caught.exception.message, message)

    def test_position_inside_a_placeholder_is_from_the_start_of_the_text(self):
        with self.assertRaises(ExpressionSyntaxError) as caught:
            parse_template("ab {x.}")
        self.assertEqual(caught.exception.position, 5)

    def test_render(self):
        template = bind(parse_template(
            "{count($days)} days, mean {diff.summary.mean} {$diff.unit}, rule {diff.rule}, "
            "{exists($skipped)}, {product}"
        ), ROOTS, own=True)
        own = _Scope().steps["diff"]
        self.assertEqual(
            render_template(template, _Scope(), own=own),
            "3 days, mean 1.5 ppb, rule same weekday, no, TEMPO_NO2",
        )


class EvaluationTests(unittest.TestCase):
    def test_conditions(self):
        cases = {
            'quantity == "no2_tropospheric_column"': True,
            'mode != "region"': False,
            "count($days) < limits.min_days": False,
            "count($days) >= limits.min_days and $diff.summary.mean > 1": True,
            "$diff.summary.mean <= 1 or not exists($skipped)": True,
            "exists($diff.summary.values.1)": True,
            "exists($diff.summary.values.5)": False,
            "exists($diff.summary.nothing)": False,
            "exists($skipped.summary)": False,
            "$diff.summary.values.0 == 1": True,
            "$diff.quality == null": True,
            '"a" < "b"': True,
            "limits.min_coverage > 0.1": True,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIs(evaluate_condition(_cond(text), _Scope()), expected)

    def test_a_skipped_step_raises_so_the_runner_can_skip_on(self):
        with self.assertRaises(StepUnavailable) as caught:
            evaluate(_cond("count($skipped) > 1"), _Scope())
        self.assertEqual(caught.exception.step_id, "skipped")

    def test_short_circuit_avoids_a_skipped_step(self):
        self.assertFalse(evaluate_condition(_cond("exists($skipped) and count($skipped) > 1"), _Scope()))

    def test_missing_field(self):
        with self.assertRaises(MissingValue):
            evaluate(_cond("$diff.summary.nothing > 1"), _Scope())

    def test_type_errors(self):
        cases = {
            'quantity < 3': IncomparableValues,
            'quantity == 3': IncomparableValues,
            '$diff.quality < 1': IncomparableValues,
            "count($diff.unit) > 1": ValueTypeError,
            "quantity and true": ValueTypeError,
            "not quantity": ValueTypeError,
        }
        for text, error in cases.items():
            with self.subTest(text=text), self.assertRaises(error):
                evaluate_condition(_cond(text), _Scope())

    def test_a_condition_must_be_true_or_false(self):
        with self.assertRaises(ValueTypeError):
            evaluate_condition(_cond("$diff.summary.mean"), _Scope())

    def test_unbound_reference_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            evaluate(parse_condition("$days"), _Scope())


class StaticTypeTests(unittest.TestCase):
    def test_problems(self):
        types = {"n": "number", "s": "string"}

        def type_of(ref: Any) -> Any:
            return types.get(ref.name) if isinstance(ref, Ref) else "number"

        cases = {
            "n < 3": [],
            "s == \"x\"": [],
            "s == null": [],
            "$unknown == 3": [],
            "s < 3": ["cannot compare string with number using <"],
            "n == \"x\"": ["cannot compare number with string using =="],
            "true < false": ["cannot compare boolean with boolean using <"],
            "count($x) == \"a\"": ["cannot compare number with string using =="],
            "exists($x) > 1": ["cannot compare boolean with number using >"],
            "n and true": ["and joins true or false values, not number"],
            "not s": ["not takes a true or false value, not string"],
            "limits.x": ["a condition is true or false, not number"],
            "3": ["a condition is true or false, not number"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(type_problems(parse_condition(text), type_of), expected)


if __name__ == "__main__":
    unittest.main()
