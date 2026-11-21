"""The workflow spec models and the facts derived from them: shape, order, products, runnability."""
import copy
import unittest
from typing import Any

from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.results import Mode
from tta_backend.harness.workflows.expressions import Func, LimitRef, Ref
from tta_backend.harness.workflows.loader import SpecErrors, parse_spec
from tta_backend.harness.workflows.spec import CycleError, UseStep, WorkflowSpec

BASE: dict[str, Any] = {
    "name": "map",
    "version": 1,
    "status": "draft",
    "description": "Map one quantity over an area.",
    "examples": ["Plot NO2 over New Jersey."],
    "inputs": {
        "quantity": {"type": "quantity", "ask": "Which quantity?"},
        "area": {"type": "area", "ask": "Which area?"},
        "dates": {"type": "date_range", "ask": "Which dates?"},
    },
    "products": {"quantity": "registry_default"},
    "steps": [
        {"id": "data", "use": "sources.satellite.get_data",
         "with": {"quantity": "$quantity", "area": "$area", "dates": "$dates", "date_role": "range"}},
        {"id": "figure", "use": "render.map", "with": {"grid": "$data"}},
    ],
    "outputs": {"figures": ["$figure"], "data": ["$data"]},
}


def spec(**changes: Any) -> WorkflowSpec:
    raw = copy.deepcopy(BASE)
    raw.update(changes)
    return parse_spec(raw)


def shape_errors(raw: dict[str, Any]) -> list[str]:
    try:
        parse_spec(raw)
    except SpecErrors as failure:
        return [str(e) for e in failure.errors]
    return []


def step(**fields: Any) -> dict[str, Any]:
    return {"use": "render.map", "with": {"grid": "$data"}, **fields}


class ShapeTests(unittest.TestCase):
    def test_base_parses(self):
        parsed = spec()
        self.assertEqual(parsed.step_ids, ("data", "figure"))
        self.assertIsInstance(parsed.step("data"), UseStep)

    def test_shape_errors_are_collected_with_their_step(self):
        raw = copy.deepcopy(BASE)
        raw["name"] = "Map"
        raw["colour"] = "blue"
        raw["steps"][0]["when"] = "a = 1"
        raw["steps"][1]["date_role"] = "range"
        raw["steps"].append({"id": "both", "use": "render.map", "stop": "failed"})
        del raw["description"]
        self.assertEqual(shape_errors(raw), [
            "<inline>: Map v1: name: a workflow name is lower case with hyphens, like event-vs-baseline",
            "<inline>: Map v1: description is required",
            "<inline>: Map v1: step data: when: use == to compare at position 2",
            "<inline>: Map v1: step figure: date_role is not a known field",
            "<inline>: Map v1: step both: a step has exactly one of use, ask, agent, workflow or stop",
            "<inline>: Map v1: colour is not a known field",
        ])

    def test_field_rules(self):
        cases: list[tuple[str, Any, str]] = [
            ("version", 0, "version: Input should be greater than or equal to 1"),
            ("status", "done", "status: Input should be 'draft' or 'validated'"),
            ("description", "two\nlines", "description is one non-empty line"),
            ("examples", [], "examples: Tuple should have at least 1 item"),
            ("limits", {"min": True}, "limits.min"),
            ("limits", {"and": 1}, "a limit name may not be and"),
            ("products", {"quantity": []}, "a products list is non-empty without repeats"),
            ("products", "everything", "products"),
            ("inputs", {"Area": {"type": "area", "ask": "Which?"}}, "an input name is a lower-case name"),
            ("inputs", {"limits": {"type": "text", "ask": "Which?"}}, "an input name may not be limits"),
            ("outputs", {}, "outputs declare at least one finding"),
            ("outputs", {"figures": ["$figure.charts"]}, "a figure or data entry is a bare step reference"),
            ("outputs", {"findings": [{"name": "m", "value": "mean"}]}, "a reference to an input or step starts with $"),
            ("steps", [], "a spec has at least one step"),
            ("benchmark", ["a", "a"], "benchmark lists case ids, each once"),
        ]
        for field, value, expected in cases:
            with self.subTest(field=field, value=value):
                raw = copy.deepcopy(BASE)
                raw[field] = value
                errors = shape_errors(raw)
                self.assertTrue(any(expected in e for e in errors), errors)

    def test_step_rules(self):
        cases: list[tuple[dict[str, Any], str]] = [
            (step(id="input_area"), "a step id may not start with input"),
            (step(id="count"), "a step id may not be count"),
            (step(id="Fig"), "a step id is a lower-case name"),
            (step(id="f", after=["x", "x"]), "after lists each step once"),
            (step(id="f", label="x" * 81), "label is at most 80 characters"),
            (step(id="f", method="{a"), "method: unmatched { at position 0"),
            (step(id="f", for_each="days"), "a reference to an input or step starts with $"),
            ({"id": "f", "use": "Render"}, "use names a capability"),
            ({"id": "f", "use": "render.map", "with": {"Grid": "$data"}}, "with keys are lower-case parameter names"),
            ({"id": "f", "use": "render.map", "with": {"grid": "$data."}}, "with: a reference is"),
            ({"id": "f", "ask": {"kind": "pick_one", "question": "Which?"}}, "a pick_one question has options and no items"),
            ({"id": "f", "ask": {"kind": "text", "question": "Which?", "items": [1]}}, "a text question has no options or items"),
            ({"id": "f", "agent": {"instruction": "Say.", "returns": {}}}, "returns names at least one field"),
            ({"id": "f", "agent": {"instruction": "x" * 2001, "returns": {"text": "string"}}},
             "instruction is at most 2000 characters"),
            ({"id": "f", "agent": {"instruction": "Say.", "reads": ["$data.summary"] * 2, "returns": {"text": "string"}}},
             "reads lists each reference once"),
            ({"id": "f", "agent": {"instruction": "Say.", "reads": [f"$data.values.v{n}" for n in range(13)],
                                   "returns": {"text": "string"}}}, "reads lists at most 12 references"),
            ({"id": "f", "agent": {"instruction": "Say.", "returns": {f"f{n}": "string" for n in range(9)}}},
             "returns names at most 8 fields"),
            ({"id": "f", "workflow": "Map", "version": 1}, "workflow names a workflow"),
            ({"id": "f", "stop": "done", "when": "true", "reason": "x"}, "stop: Input should be"),
            ({"id": "f", "stop": "failed", "reason": "{"}, "reason: unmatched {"),
        ]
        for entry, expected in cases:
            with self.subTest(entry=entry):
                raw = copy.deepcopy(BASE)
                raw["steps"].append(entry)
                errors = shape_errors(raw)
                self.assertTrue(any(expected in e for e in errors), errors)

    def test_input_rules(self):
        cases: list[tuple[dict[str, Any], str | None]] = [
            ({"type": "text"}, "a required input has an ask"),
            ({"type": "choice", "ask": "Which?"}, "options are given for a choice input and only then"),
            ({"type": "text", "ask": "Which?", "options": ["a"]}, "options are given for a choice input"),
            ({"type": "choice", "options": ["a", "a"], "default": "a"}, "options are a non-empty list without repeats"),
            ({"type": "handle", "ask": "Which?"}, "kind is given for a handle input and only then"),
            ({"type": "handle", "kind": "cube", "ask": "Which?"}, "kind"),
            ({"type": "date", "min": 1, "ask": "Which?"}, "min and max apply to number and text inputs"),
            ({"type": "number", "min": 5, "max": 1, "ask": "Which?"}, "min is above max"),
            ({"type": "number", "default": 40, "min": 1, "max": 31}, "the default is outside min and max"),
            ({"type": "number", "default": "7"}, "the default of a number input is a number"),
            ({"type": "choice", "options": ["a"], "default": "b"}, "the default of a choice input is one of its options"),
            ({"type": "area", "default": "aoi_x"}, "an area input has no default other than null"),
            ({"type": "handle", "kind": "grid", "default": "grid_x"}, "a handle input has no default other than null"),
            ({"type": "area", "default": None}, None),
            ({"type": "handle", "kind": "grid", "default": None}, None),
            ({"type": "text", "default": None, "max": 3}, None),
            ({"type": "choice", "options": ["a"], "default": None}, None),
            ({"type": "date_range", "default": None}, None),
            ({"type": "date", "default": "yesterday"}, "the default of a date input is an ISO date"),
            ({"type": "date_range", "default": {"start": "2024-02-01", "end": "2024-01-01"}}, "the default's end is before its start"),
            ({"type": "date_list", "default": []}, "a non-empty list of ISO dates"),
            ({"type": "quantity", "default": "NO2"}, "a lower-case quantity name"),
            ({"type": "text", "default": "abc", "max": 2}, "the default's length is outside min and max"),
            ({"type": "number", "default": 7, "min": 1, "max": 31}, None),
            ({"type": "handle", "kind": "grid", "ask": "Which grid?"}, None),
            ({"type": "date_list", "default": ["2024-01-01"]}, None),
            ({"type": "monitor", "ask": "Which monitor?"}, None),
        ]
        for declared, expected in cases:
            with self.subTest(declared=declared):
                raw = copy.deepcopy(BASE)
                raw["inputs"]["extra"] = declared
                errors = shape_errors(raw)
                if expected is None:
                    self.assertEqual(errors, [])
                else:
                    self.assertTrue(any(expected in e for e in errors), errors)

    def test_yaml_dates_are_kept_as_iso_text(self):
        parsed = parse_spec(
            "name: map\nversion: 1\nstatus: validated\ndescription: Map.\nexamples: [Map it.]\n"
            "inputs:\n  dates: {type: date_range, default: {start: 2024-01-01, end: 2024-01-31}}\n"
            "steps:\n  - {id: show, use: render.map, with: {grid: $x, title: 2024-01-01}}\n"
            "outputs: {figures: [$show]}\nvalidated_on: 2026-11-01\n"
        )
        self.assertEqual(parsed.inputs["dates"].default, {"start": "2024-01-01", "end": "2024-01-31"})
        self.assertEqual(parsed.step("show").with_["title"], "2024-01-01")
        self.assertEqual(str(parsed.validated_on), "2026-11-01")

    def test_optional_inputs(self):
        parsed = spec(inputs={**BASE["inputs"], "stat": {"type": "choice", "options": ["mean", "max"], "default": "mean"}})
        self.assertTrue(parsed.inputs["stat"].optional)
        self.assertFalse(parsed.inputs["area"].optional)
        self.assertEqual(parsed.inputs["area"].handle_kind, "polygons")

    def test_output_forms(self):
        parsed = spec(outputs={
            "findings": [{"name": "mean", "value": "$data.summary.mean", "unit": "$data.unit", "where": "New Jersey", "optional": True}],
            "figures": [{"from": "$figure", "optional": True}],
            "data": ["$data"],
        })
        out = parsed.compiled_outputs
        self.assertEqual(out.findings[0].value, Ref("data", ("summary", "mean"), root="step"))
        self.assertTrue(out.findings[0].optional)
        self.assertEqual([(e.ref.name, e.optional) for e in out.figures], [("figure", True)])
        self.assertEqual([(e.ref.name, e.optional) for e in out.data], [("data", False)])


class CompiledTests(unittest.TestCase):
    def test_bound_references(self):
        parsed = spec(limits={"few": 2}, steps=[
            *BASE["steps"],
            {"id": "too_few", "stop": "insufficient_evidence", "when": "count($data.values.days) < limits.few",
             "reason": "Only {count($data.values.days)} days for {quantity}."},
        ])
        compiled = parsed.compiled("too_few")
        self.assertEqual(compiled.kind, "stop")
        assert compiled.reason is not None
        self.assertEqual(compiled.reason.slots, (Func("count", Ref("data", ("values", "days"), root="step")), Ref("quantity", dollar=False, root="input")))
        self.assertIn(LimitRef("few"), compiled.references())
        self.assertEqual(parsed.compiled("figure").with_.get("grid"), Ref("data", root="step"))

    def test_own_result_in_method(self):
        raw = copy.deepcopy(BASE)
        raw["steps"][0]["method"] = "{product} over {area}, mean {summary.mean}."
        roots = [r.root for r in parse_spec(raw).compiled("data").references() if isinstance(r, Ref)]
        self.assertEqual(roots, ["input", "input", "input", "own", "input", "own"])


class OrderTests(unittest.TestCase):
    def _branching(self) -> WorkflowSpec:
        return spec(steps=[
            {"id": "last", "use": "operations.difference", "with": {"a": "$left", "b": "$right"}},
            *BASE["steps"],
            {"id": "left", "use": "operations.aggregate_time", "with": {"data": "$data"}},
            {"id": "right", "use": "operations.aggregate_time", "with": {"data": "$data", "stat": "max"}},
            {"id": "note", "use": "explain.reliability", "with": {"handle": "$data"}, "after": ["figure"]},
        ])

    def test_dependencies_come_from_references_and_after(self):
        parsed = self._branching()
        self.assertEqual(parsed.dependencies("last"), {"left", "right"})
        self.assertEqual(parsed.dependencies("data"), frozenset())
        self.assertEqual(parsed.dependencies("note"), {"data", "figure"})

    def test_order_follows_dependencies_not_the_file(self):
        order = self._branching().order()
        self.assertEqual(order, ("data", "figure", "left", "right", "last", "note"))

    def test_independent_steps_share_no_dependency(self):
        parsed = self._branching()
        for a, b in (("left", "right"), ("figure", "left")):
            self.assertNotIn(a, parsed.dependencies(b))
            self.assertNotIn(b, parsed.dependencies(a))

    def test_inputs_are_not_dependencies(self):
        self.assertEqual(spec().dependencies("data"), frozenset())

    def test_cycles(self):
        parsed = spec(steps=[
            {"id": "a", "use": "operations.aggregate_time", "with": {"data": "$b"}},
            {"id": "b", "use": "operations.aggregate_time", "with": {"data": "$a"}},
            {"id": "c", "use": "operations.aggregate_time", "with": {"data": "$c"}},
            {"id": "d", "use": "render.map", "with": {"grid": "$a"}, "after": ["d"]},
        ], outputs={"figures": ["$d"]})
        self.assertEqual(parsed.cycles(), (("a", "b"), ("c",), ("d",)))
        with self.assertRaises(CycleError):
            parsed.order()

    def test_always_runs(self):
        parsed = spec(steps=[
            *BASE["steps"],
            {"id": "maybe", "use": "operations.aggregate_time", "with": {"data": "$data"}, "when": "exists($data.summary)"},
            {"id": "after_maybe", "use": "render.map", "with": {"grid": "$maybe"}},
            {"id": "soft", "use": "operations.aggregate_time", "with": {"data": "$data"}, "required": False},
        ])
        self.assertEqual(
            {s: parsed.always_runs(s) for s in parsed.step_ids},
            {"data": True, "figure": True, "maybe": False, "after_maybe": False, "soft": False},
        )


class RunnerFactsTests(unittest.TestCase):
    def test_runnable(self):
        self.assertTrue(spec().runnable)
        self.assertIsNone(spec().not_runnable_reason)

    def test_an_agent_step_is_runnable(self):
        say = {"id": "say", "agent": {"instruction": "Say.", "reads": ["$data.summary"], "returns": {"text": "string"}}}
        parsed = spec(steps=[*BASE["steps"], say])
        self.assertTrue(parsed.runnable)
        self.assertIsNone(parsed.not_runnable_reason)

    def test_an_ask_step_is_runnable(self):
        pick = {"id": "pick", "ask": {"kind": "pick_one", "question": "Which?", "options": ["a", "b"]}}
        parsed = spec(steps=[*BASE["steps"], pick])
        self.assertTrue(parsed.runnable)
        self.assertIsNone(parsed.not_runnable_reason)

    def test_not_runnable_names_the_first_step_and_feature(self):
        say = {"id": "say", "agent": {"instruction": "Say.", "reads": ["$data.summary"], "returns": {"text": "string"}}}
        pick = {"id": "pick", "ask": {"kind": "pick_one", "question": "Which?", "options": ["a", "b"]}}
        cases: list[tuple[dict[str, Any], str]] = [
            ({"id": "child", "workflow": "map", "version": 1, "with": {"area": "$area"}},
             "step child is a workflow step, which the runner does not run"),
            (step(id="each", for_each="$data.values.days"), "step each uses for_each, which the runner does not run"),
        ]
        for entry, reason in cases:
            with self.subTest(reason=reason):
                parsed = spec(steps=[*BASE["steps"], say, pick, entry, step(id="later", for_each="$data.values.days")])
                self.assertFalse(parsed.runnable)
                self.assertEqual(parsed.not_runnable_reason, reason)

    def test_approved_for(self):
        self.assertEqual(spec().approved_for("data"), "registry_default")
        self.assertEqual(spec(products="registry_default").approved_for("data"), "registry_default")
        self.assertEqual(spec(products={"quantity": ["TEMPO_NO2", "OMI_NO2"]}).approved_for("data"), ["TEMPO_NO2", "OMI_NO2"])
        self.assertIsNone(spec(products=None).approved_for("data"))
        self.assertIsNone(spec().approved_for("figure"))

    def test_mode(self):
        self.assertEqual(spec().mode(), Mode.draft("map", 1))
        validated = spec(status="validated", benchmark=["map_1"], validated_by="Science team", validated_on="2026-11-01")
        self.assertEqual(validated.mode(), Mode.validated("map", 1))

    def test_connectors_are_derived(self):
        registry = build_registry()
        self.assertEqual(spec().connectors(registry), {"earthdata"})
        render_only = spec(inputs={"grid": {"type": "handle", "kind": "grid", "ask": "Which grid?"}},
                           steps=[{"id": "figure", "use": "render.map", "with": {"grid": "$grid"}}],
                           outputs={"figures": ["$figure"]})
        self.assertEqual(render_only.connectors(registry), frozenset())
        # explain.lineage declares earthdata but does not require it.
        lineage = spec(steps=[*BASE["steps"], {"id": "lineage", "use": "explain.lineage", "with": {"handles": ["$data"]}}])
        self.assertEqual(lineage.connectors(registry), {"earthdata"})


if __name__ == "__main__":
    unittest.main()
