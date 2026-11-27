"""Loading specs against the real capabilities and dataset registry: every load check, packs on disk, the catalogue."""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.workflows.loader import (
    SpecError,
    SpecErrors,
    UnknownWorkflow,
    WorkflowCatalogue,
    approved_parameter,
    load_packs,
    load_spec,
    load_specs,
    parse_spec,
)
from tta_backend.harness.workflows.spec import WorkflowSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "workflows"
CAPABILITIES = build_registry()
DATASETS = load_registry()

INPUTS: dict[str, Any] = {
    "quantity": {"type": "quantity", "ask": "Which quantity?"},
    "area": {"type": "area", "ask": "Which area?"},
    "dates": {"type": "date_range", "ask": "Which dates?"},
}
GET = {"quantity": "$quantity", "area": "$area", "dates": "$dates", "date_role": "range"}

BASE: dict[str, Any] = {
    "name": "map",
    "version": 1,
    "status": "draft",
    "description": "Map one quantity over an area.",
    "examples": ["Plot NO2 over New Jersey."],
    "inputs": INPUTS,
    "products": {"quantity": ["TEMPO_NO2", "TROPOMI_NO2"]},
    "limits": {"min_coverage": 0.3},
    "steps": [
        {"id": "data", "use": "sources.satellite.get_data", "with": GET},
        {"id": "figure", "use": "render.map", "with": {"grid": "$data"}},
    ],
    "outputs": {"figures": ["$figure"], "data": ["$data"]},
}


def raw(**changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(BASE)
    out.update(copy.deepcopy(changes))
    return out


def with_steps(*extra: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return raw(steps=[*copy.deepcopy(BASE["steps"]), *extra], **changes)


def load(source: Any, datasets: Any = None) -> WorkflowSpec:
    return load_spec(source, CAPABILITIES, DATASETS if datasets is None else datasets)


def messages(source: Any, datasets: Any = None) -> list[str]:
    try:
        load(source, datasets)
    except SpecErrors as failure:
        return [e.message for e in failure.errors]
    return []


class LoadCheckTests(unittest.TestCase):
    """Each case: a spec that fails one check with its wording, and its corrected form that loads."""

    def assert_check(self, bad: dict[str, Any], expected: str, good: dict[str, Any] | None) -> None:
        found = messages(bad)
        self.assertIn(expected, found)
        if good is not None:
            self.assertEqual(messages(good), [])

    def run_cases(self, cases: list[tuple[str, dict[str, Any], str, dict[str, Any] | None]]) -> None:
        for name, bad, expected, good in cases:
            with self.subTest(name):
                self.assert_check(bad, expected, good)

    def test_base_loads(self):
        self.assertEqual(messages(BASE), [])

    def test_references(self):
        fig = {"id": "extra", "use": "render.map"}
        self.run_cases([
            ("unknown step", with_steps({**fig, "with": {"grid": "$nothing"}}), "$nothing names no input or step",
             with_steps({**fig, "with": {"grid": "$data"}})),
            ("unknown limit", with_steps({**fig, "with": {"grid": "$data"}, "when": "limits.few > 1"}),
             "limits.few is not a limit", with_steps({**fig, "with": {"grid": "$data"}, "when": "limits.min_coverage > 0.1"})),
            ("bare step in when", with_steps({**fig, "with": {"grid": "$data"}, "when": "data == null"}),
             "data in when is not an input; a step is written $data",
             with_steps({**fig, "with": {"grid": "$data"}, "when": "$data == null"})),
            ("bare unknown in when", with_steps({**fig, "with": {"grid": "$data"}, "when": "colour == \"red\""}),
             "colour names no input or step",
             with_steps({**fig, "with": {"grid": "$data"}, "when": "quantity == \"no2_tropospheric_column\""})),
            ("own field", with_steps({**fig, "with": {"grid": "$data"}, "method": "Drawn from {bogus}."}),
             "{bogus} names no input, step or result field",
             with_steps({**fig, "with": {"grid": "$data"}, "method": "Drawn from {data.product} over {area}."})),
            ("after", with_steps({**fig, "with": {"grid": "$data"}, "after": ["zz"]}), "after names zz, which is not a step",
             with_steps({**fig, "with": {"grid": "$data"}, "after": ["figure"]})),
            ("after itself", with_steps({**fig, "with": {"grid": "$data"}, "after": ["extra"]}), "after names the step itself", None),
            ("repeated id", with_steps({**fig, "id": "figure", "with": {"grid": "$data"}}), "step id figure is used more than once", None),
            ("id is an input", with_steps({**fig, "id": "area", "with": {"grid": "$data"}}), "step id area is also an input name", None),
            ("finding", raw(outputs={"findings": [{"name": "mean", "value": "$nope.summary.mean"}]}),
             "$nope.summary.mean names no input or step",
             raw(outputs={"findings": [{"name": "mean", "value": "$data.summary.mean", "unit": "$data.unit"}]})),
        ])

    def test_static_types_in_when(self):
        fig = {"id": "extra", "use": "render.map", "with": {"grid": "$data"}}
        choice = {**INPUTS, "mode": {"type": "choice", "options": ["region", "point"], "default": "region"}}
        self.run_cases([
            ("string against number", with_steps({**fig, "when": "quantity < 3"}),
             "when: cannot compare string with number using <", with_steps({**fig, "when": "quantity != \"x\""})),
            ("limit against string", with_steps({**fig, "when": "limits.min_coverage == \"high\""}),
             "when: cannot compare number with string using ==", with_steps({**fig, "when": "limits.min_coverage == 0.3"})),
            ("not an option", with_steps({**fig, "when": "mode == \"regions\""}, inputs=choice),
             "when compares mode with text that is not one of its options",
             with_steps({**fig, "when": "mode == \"region\""}, inputs=choice)),
        ])

    def test_capability_and_parameters(self):
        self.run_cases([
            ("unknown capability", with_steps({"id": "extra", "use": "render.globe", "with": {}}),
             "render.globe is not a registered capability", with_steps({"id": "extra", "use": "render.map", "with": {"grid": "$data"}})),
            ("unknown parameter", with_steps({"id": "extra", "use": "render.map", "with": {"grid": "$data", "colour": "red"}}),
             "render.map has no parameter colour", None),
            ("approved in with", raw(steps=[{"id": "data", "use": "sources.satellite.get_data", "with": {**GET, "approved": ["TEMPO_NO2"]}},
                                            BASE["steps"][1]]),
             "approved comes from products, not from with", None),
            ("required parameter", with_steps({"id": "extra", "use": "render.map", "with": {"style": "diverging"}}),
             "with does not give grid, which render.map requires", None),
            ("date_role", raw(steps=[{"id": "data", "use": "sources.satellite.get_data",
                                      "with": {k: v for k, v in GET.items() if k != "date_role"}}, BASE["steps"][1]]),
             "with does not give date_role, which sources.satellite.get_data requires", None),
            ("literal value", with_steps({"id": "extra", "use": "render.map", "with": {"grid": "$data", "style": "rainbow"}}),
             "with.style is not a valid value for render.map: Input should be 'sequential' or 'diverging'",
             with_steps({"id": "extra", "use": "render.map", "with": {"grid": "$data", "style": "diverging"}})),
            ("literal dates", raw(steps=[{"id": "data", "use": "sources.satellite.get_data",
                                          "with": {**GET, "dates": {"start": "2024-01-01", "end": "soon"}}}, BASE["steps"][1]]),
             "with.dates is not a valid value for sources.satellite.get_data: Input should be a valid date",
             raw(steps=[{"id": "data", "use": "sources.satellite.get_data",
                         "with": {**GET, "dates": {"start": "2024-01-01", "end": "2024-01-31"}}}, BASE["steps"][1]])),
            ("literal quantity", raw(steps=[{"id": "data", "use": "sources.satellite.get_data",
                                             "with": {**GET, "quantity": "no2_tropospheric_column"}}, BASE["steps"][1]]),
             "with.quantity for sources.satellite.get_data is a quantity input, like $quantity", None),
        ])

    def test_error_names_the_step_and_field(self):
        errors = _errors(raw(steps=[{"id": "data", "use": "sources.satellite.get_data",
                                     "with": {k: v for k, v in GET.items() if k != "date_role"}}, BASE["steps"][1]]))
        self.assertEqual([(e.step, e.message) for e in errors],
                         [("data", "with does not give date_role, which sources.satellite.get_data requires")])
        self.assertEqual(str(errors[0]),
                         "<inline>: map v1: step data: with does not give date_role, which sources.satellite.get_data requires")

    def test_kinds(self):
        grid_input = {**INPUTS, "base": {"type": "handle", "kind": "grid", "ask": "Which grid?"},
                      "table": {"type": "handle", "kind": "table", "ask": "Which table?"}}

        def extra(use: str, with_: dict[str, Any], **more: Any) -> dict[str, Any]:
            return with_steps({"id": "extra", "use": use, "with": with_}, **more)

        stat = {"id": "stat", "use": "operations.statistic", "with": {"data": "$data"}}
        self.run_cases([
            ("area input to a grid", extra("render.map", {"grid": "$area"}), "with.grid takes grid; input area is polygons",
             extra("render.map", {"grid": "$data", "polygons": ["$area"]})),
            ("plain input to a handle", extra("render.map", {"grid": "$quantity"}),
             "with.grid takes a handle; input quantity is a quantity", None),
            ("handle input of another kind", extra("render.map", {"grid": "$table"}, inputs=grid_input),
             "with.grid takes grid; input table is table", extra("render.map", {"grid": "$base"}, inputs=grid_input)),
            ("step without a handle", extra("operations.aggregate_time", {"data": "$figure"}),
             "with.data takes a handle; step figure returns none", extra("operations.aggregate_time", {"data": "$data"})),
            ("step of another kind", with_steps(stat, {"id": "extra", "use": "operations.aggregate_time", "with": {"data": "$stat"}}),
             "with.data takes grid; step stat returns scalar", None),
            ("handle field", with_steps(stat, {"id": "extra", "use": "operations.aggregate_time", "with": {"data": "$stat.handle"}}),
             "with.data takes grid; step stat returns scalar",
             with_steps({"id": "extra", "use": "operations.aggregate_time", "with": {"data": "$data.handle"}})),
            ("bare step to a value", extra("render.map", {"grid": "$data", "title": "$data"}),
             "with.title takes a value, not a handle; read a field of step data, like $data.summary",
             extra("render.map", {"grid": "$data", "title": "$data.product"})),
            ("area input to a value", extra("render.map", {"grid": "$data", "title": "$area"}),
             "with.title takes a value, not a handle; input area is a handle", None),
            ("one handle where a list is taken", with_steps(
                {"id": "series", "use": "operations.regional_series", "with": {"data": "$data"}},
                {"id": "extra", "use": "render.timeseries", "with": {"series": "$series"}}),
             "with.series takes a list of handles, like [$series]",
             with_steps({"id": "series", "use": "operations.regional_series", "with": {"data": "$data"}},
                        {"id": "extra", "use": "render.timeseries", "with": {"series": ["$series"]}})),
            ("a list where one handle is taken", extra("render.map", {"grid": ["$data"]}), "with.grid takes one handle, not a list", None),
            ("a limit to a handle", extra("render.map", {"grid": "limits.min_coverage"}), "with.grid takes a handle; a limit is a number", None),
        ])

    def test_cycles(self):
        self.run_cases([
            ("two steps", with_steps(
                {"id": "a", "use": "operations.aggregate_time", "with": {"data": "$b"}},
                {"id": "b", "use": "operations.aggregate_time", "with": {"data": "$a"}}),
             "steps a, b depend on one another in a cycle",
             with_steps({"id": "a", "use": "operations.aggregate_time", "with": {"data": "$data"}},
                        {"id": "b", "use": "operations.aggregate_time", "with": {"data": "$a"}})),
            ("through after", with_steps(
                {"id": "a", "use": "operations.aggregate_time", "with": {"data": "$data"}, "after": ["b"]},
                {"id": "b", "use": "operations.aggregate_time", "with": {"data": "$a"}}),
             "steps a, b depend on one another in a cycle", None),
            ("itself", with_steps({"id": "a", "use": "operations.aggregate_time", "with": {"data": "$a"}}),
             "the step depends on itself", None),
        ])

    def test_products_exist(self):
        no_quantity = dict(DATASETS)
        no_quantity["TEMPO_NO2"] = DATASETS["TEMPO_NO2"].model_copy(update={"quantity": None})
        self.assertIn("product NOPE_NO2 is not in the dataset registry", messages(raw(products={"quantity": ["TEMPO_NO2", "NOPE_NO2"]})))
        self.assertIn("product TEMPO_NO2 records no quantity in the dataset registry", messages(BASE, no_quantity))
        self.assertEqual(messages(raw(products={"quantity": ["TEMPO_NO2_NRT", "OMI_NO2"]})), [])

    def test_stop_has_when_and_reason(self):
        stop = {"id": "too_few", "stop": "insufficient_evidence"}
        self.run_cases([
            ("no reason", with_steps({**stop, "when": "count($data.values.dropped_dates) > 3"}), "a stop step has a when and a reason",
             with_steps({**stop, "when": "count($data.values.dropped_dates) > 3",
                         "reason": "{count($data.values.dropped_dates)} days were dropped; at most 3 may be."})),
            ("no when", with_steps({**stop, "reason": "Stopped."}), "a stop step has a when and a reason", None),
        ])

    def test_product_discipline(self):
        second = {"id": "second", "use": "sources.satellite.get_data", "with": {**GET, "product": "$data.product"}}
        third = {"id": "third", "use": "sources.satellite.get_point_series",
                 "with": {"quantity": "$quantity", "area": "$area", "dates": "$dates", "product": "$second.product"}}
        self.run_cases([
            ("two choosers", with_steps({**second, "with": GET}),
             "steps data, second each choose the product for quantity; pin one with product: $data.product, "
             'or put them in exclusive branches on one choice input, like when: mode == "region" and when: mode == "point"',
             with_steps(second)),
            ("pinned to a literal", with_steps({**second, "with": {**GET, "product": "TEMPO_NO2"}}),
             "with.product pins to another step retrieving quantity, like $<step>.product", None),
            ("transitive pin", with_steps(second, {**third, "with": {**third["with"], "product": "$third.product"}}),
             "the step depends on itself", with_steps(second, third)),
            ("pins in a loop beside the chooser", with_steps(
                {**second, "with": {**GET, "product": "$third.product"}},
                {**third, "with": {**third["with"], "product": "$second.product"}}),
             "steps second, third pin the product for quantity to one another, not to data", None),
        ])
        no_chooser = raw(steps=[
            {"id": "a", "use": "sources.satellite.get_data", "with": {**GET, "product": "$b.product"}},
            {"id": "b", "use": "sources.satellite.get_data", "with": {**GET, "product": "$a.product"}},
            {"id": "figure", "use": "render.map", "with": {"grid": "$a"}},
        ], outputs={"figures": ["$figure"]})
        self.assertIn("no step chooses the product for quantity; steps a, b pin to one another", messages(no_chooser))

    def _branches(self, region_when: str | None, point_when: str | None) -> dict[str, Any]:
        """A timeseries shape: a regional series from a grid, or a point series, chosen by mode."""
        inputs = {**INPUTS,
                  "mode": {"type": "choice", "options": ["region", "point"], "default": "region"},
                  "level": {"type": "choice", "options": ["surface", "column"], "default": "column"}}
        grid = {"id": "grid", "use": "sources.satellite.get_data", "with": GET}
        point = {"id": "point", "use": "sources.satellite.get_point_series",
                 "with": {"quantity": "$quantity", "area": "$area", "dates": "$dates"}}
        if region_when is not None:
            grid["when"] = region_when
        if point_when is not None:
            point["when"] = point_when
        return raw(inputs=inputs, steps=[
            grid,
            {"id": "regional", "use": "operations.regional_series", "with": {"data": "$grid"}},
            point,
            {"id": "region_chart", "use": "render.timeseries", "with": {"series": ["$regional"]}},
            {"id": "point_chart", "use": "render.timeseries", "with": {"series": ["$point"]}},
        ], outputs={"figures": [{"from": "$region_chart", "optional": True}, {"from": "$point_chart", "optional": True}]})

    def test_choosers_in_exclusive_branches_on_one_choice_input(self):
        self.assertEqual(messages(self._branches('mode == "region"', 'mode == "point"')), [])
        self.assertEqual(messages(self._branches('mode == "region" and limits.min_coverage > 0', '"point" == mode')), [])
        rejected = {
            "different inputs": ('mode == "region"', 'level == "surface"'),
            "one branch unconditioned": ('mode == "region"', None),
            "an inequality": ('mode == "region"', 'mode != "region"'),
            "a disjunction": ('mode == "region"', 'mode == "point" or limits.min_coverage > 0'),
            "the same option": ('mode == "region"', 'mode == "region"'),
        }
        for name, (region_when, point_when) in rejected.items():
            with self.subTest(name):
                found = messages(self._branches(region_when, point_when))
                self.assertEqual(len(found), 1, found)
                self.assertTrue(found[0].startswith("steps grid, point each choose the product for quantity;"), found)
                self.assertIn("exclusive branches on one choice input", found[0])

    def test_a_branch_is_inherited_from_the_steps_a_chooser_reads(self):
        inputs = {**INPUTS, "mode": {"type": "choice", "options": ["region", "place"], "default": "region"},
                  "place": {"type": "text", "ask": "Which place?"}}
        steps = [
            {"id": "grid", "use": "sources.satellite.get_data", "with": GET, "when": 'mode == "region"'},
            {"id": "picked", "use": "sources.areas.resolve", "with": {"place": "$place"}, "when": 'mode == "place"'},
            {"id": "elsewhere", "use": "sources.satellite.get_data", "with": {**GET, "area": "$picked"}},
            {"id": "figure", "use": "render.map", "with": {"grid": "$grid"}},
        ]
        self.assertEqual(messages(raw(inputs=inputs, steps=steps, outputs={"figures": [{"from": "$figure", "optional": True}]})), [])
        steps[1].pop("when")
        found = messages(raw(inputs=inputs, steps=steps, outputs={"figures": [{"from": "$figure", "optional": True}]}))
        self.assertTrue(any(m.startswith("steps grid, elsewhere each choose the product") for m in found), found)

    def test_a_branch_is_not_inherited_through_exists_or_after(self):
        inputs = {**INPUTS, "mode": {"type": "choice", "options": ["region", "point"], "default": "region"}}
        outputs = {"figures": [{"from": "$figure", "optional": True}]}
        for name, linked in (("exists", {"when": "not exists($c)"}), ("after", {"after": ["c"]})):
            with self.subTest(name):
                steps = [
                    {"id": "c", "use": "sources.areas.resolve", "with": {"place": "x"}, "when": 'mode == "region"'},
                    {"id": "a", "use": "sources.satellite.get_data", "with": GET, "when": 'mode == "point"'},
                    {"id": "b", "use": "sources.satellite.get_data", "with": GET, **linked},
                    {"id": "figure", "use": "render.map", "with": {"grid": "$a"}},
                ]
                # With mode point, c is skipped and both a and b run, each choosing a product.
                found = messages(raw(inputs=inputs, steps=steps, outputs=outputs))
                self.assertTrue(any(m.startswith("steps a, b each choose the product for quantity") for m in found), found)

    def test_a_spec_nested_too_deeply_is_an_error_not_a_crash(self):
        deep: Any = 1
        for _ in range(3000):
            deep = [deep]
        steps = [{"id": "data", "use": "sources.satellite.get_data", "with": {**GET, "dates": deep}}]
        with self.assertRaises(SpecErrors):
            parse_spec({**BASE, "steps": steps})
        text = "name: map\nsteps: " + "[" * 3000 + "]" * 3000
        with self.assertRaises(SpecErrors) as caught:
            parse_spec(text)
        self.assertEqual(caught.exception.errors[0].message, "the spec nests too deeply to read")
        when = "(" * 3000 + "$x" + ")" * 3000
        with self.assertRaises(SpecErrors):
            parse_spec(with_steps({"id": "extra", "use": "render.map", "with": {"grid": "$data"}, "when": when}))

    def test_nullable_inputs(self):
        inputs = {**INPUTS, "named": {"type": "text", "default": None},
                  "focus": {"type": "area", "default": None}, "where": {"type": "area", "default": None}}
        good = raw(inputs=inputs, steps=[
            {"id": "data", "use": "sources.satellite.get_data", "with": {**GET, "named": "$named"}},
            {"id": "figure", "use": "render.map", "with": {"grid": "$data", "polygons": ["$area"]}},
            {"id": "smoke", "use": "sources.smoke.get_extent", "with": {"date": "2024-06-01", "area": "$focus"},
             "when": "exists(focus) and named != null"},
        ])
        self.assertEqual(messages(good), [])
        cases = {
            "a parameter that needs a value": (
                {"id": "extra", "use": "render.map", "with": {"grid": "$data", "style": "$named"}},
                "with.style gets input named, which may be null, but render.map needs a value"),
            "a handle parameter that needs a value": (
                {"id": "extra", "use": "operations.aggregate_time", "with": {"data": "$focus"}},
                "with.data gets input focus, which may be null, but operations.aggregate_time needs a value"),
            "a nullable handle in a list of handles": (
                {"id": "extra", "use": "render.map", "with": {"grid": "$data", "polygons": ["$area", "$focus"]}},
                "with.polygons puts input focus, which may be null, inside a list or mapping; "
                "a null there is not a valid value"),
            "a nullable value in a mapping": (
                {"id": "extra", "use": "sources.satellite.get_data",
                 "with": {**GET, "dates": {"start": "$named", "end": "2024-06-02"}, "product": "$data.product"}},
                "with.dates puts input named, which may be null, inside a list or mapping; "
                "a null there is not a valid value"),
            "the kind still applies": (
                {"id": "extra", "use": "render.map", "with": {"grid": "$data", "points": "$where"}},
                "with.points takes points; input where is polygons"),
        }
        for name, (step, expected) in cases.items():
            with self.subTest(name):
                self.assertIn(expected, messages(with_steps(step, inputs=inputs)))

    def test_products_cover_injected_steps(self):
        self.run_cases([
            ("missing", raw(products=None),
             "products is required: steps data retrieve by quantity", BASE),
            ("not a quantity input", raw(products={"quantity": "registry_default", "area": ["TEMPO_NO2"]}),
             "products key area is not a quantity input", raw(products={"quantity": "registry_default"})),
            ("no entry", raw(inputs={**INPUTS, "other": {"type": "quantity", "ask": "Which other?"}},
                             products={"other": "registry_default"}),
             "products has no entry for quantity, which this step retrieves",
             raw(inputs={**INPUTS, "other": {"type": "quantity", "ask": "Which other?"}},
                 products={"other": "registry_default", "quantity": "registry_default"})),
        ])

    def test_outputs_always_run_or_are_optional(self):
        maybe = {"id": "maybe", "use": "operations.statistic", "with": {"data": "$data"}, "when": "exists($data.summary)"}
        soft = {"id": "soft", "use": "operations.statistic", "with": {"data": "$data"}, "required": False}
        after = {"id": "after", "use": "operations.aggregate_time", "with": {"data": "$data"}, "when": "limits.min_coverage > 0.1"}
        later = {"id": "later", "use": "render.map", "with": {"grid": "$after"}}
        self.run_cases([
            ("finding from a conditional step", with_steps(maybe, outputs={"findings": [{"name": "mean", "value": "$maybe.summary.value"}]}),
             "finding mean comes from step maybe, which may not run; mark it optional",
             with_steps(maybe, outputs={"findings": [{"name": "mean", "value": "$maybe.summary.value", "optional": True}]})),
            ("unit from a step that may fail", with_steps(soft, outputs={"findings": [{"name": "mean", "value": "$data.summary.mean", "unit": "$soft.unit"}]}),
             "finding mean comes from step soft, which may not run; mark it optional",
             with_steps(soft, outputs={"findings": [{"name": "mean", "value": "$data.summary.mean", "unit": "ppb"}]})),
            ("figure downstream of a condition", with_steps(after, later, outputs={"figures": ["$later"]}),
             "figures entry $later comes from step later, which may not run; mark it optional",
             with_steps(after, later, outputs={"figures": [{"from": "$later", "optional": True}]})),
            ("data downstream of a condition", with_steps(after, outputs={"data": ["$after"]}),
             "data entry $after comes from step after, which may not run; mark it optional",
             with_steps(after, outputs={"data": [{"from": "$after", "optional": True}]})),
            ("figure that does not render", raw(outputs={"figures": ["$data"]}), "figures entry $data is not a render step", None),
            ("data without a handle", raw(outputs={"data": ["$figure"]}), "data entry $figure returns no handle", None),
            ("figure from an input", raw(outputs={"figures": ["$area"]}), "figures entry $area is not a step", None),
            ("interpretation not from an agent", raw(outputs={"figures": ["$figure"], "interpretation": "$data.summary"}),
             "interpretation comes from an agent step", None),
        ])
        agent = {"id": "say", "agent": {"instruction": "Say what the map shows.", "reads": ["$data.summary"], "returns": {"text": "string"}},
                 "when": "exists($data.summary)"}
        self.assertIn("interpretation comes from step say, which may not run; mark it optional",
                      messages(with_steps(agent, outputs={"figures": ["$figure"], "interpretation": "$say.text"})))
        self.assertEqual(messages(with_steps(agent, outputs={"figures": ["$figure"],
                                                             "interpretation": {"from": "$say.text", "optional": True}})), [])

    def test_agent_steps(self):
        def agent(**fields: Any) -> dict[str, Any]:
            spec_ = {"instruction": "Say what the map shows.", "reads": ["$data.summary"],
                     "returns": {"text": "string", "ok": "boolean", "points": "list"}}
            spec_.update(fields)
            return {"id": "say", "agent": spec_}

        def make(step_: dict[str, Any], *more: dict[str, Any], outputs: dict[str, Any] | None = None) -> dict[str, Any]:
            return with_steps(step_, *more, outputs=outputs or {"figures": ["$figure"], "interpretation": "$say.text"})

        good = make(agent(reads=["$data.summary", "$data.quality", "$quantity", "$dates.start", "limits.min_coverage"]))
        self.assertEqual(messages(good), [])
        later = {"id": "later", "use": "render.map", "with": {"grid": "$data"}, "when": "$say.ok"}
        self.assertEqual(messages(make(agent(), later)), [])
        cases = [
            ("a bare use step in reads", make(agent(reads=["$data"])),
             "reads $data, a handle; read a field of it, like $data.summary"),
            ("a handle input in reads", make(agent(reads=["$area"])),
             "reads $area, a handle; an agent step reads summaries and small values"),
            ("an undeclared field", make(agent(), outputs={"figures": ["$figure"], "interpretation": "$say.words"}),
             "$say.words: agent step say returns no field words"),
            ("inside a scalar field", make(agent(), {**later, "when": "exists($say.text.more)"}),
             "$say.text.more: field text of agent step say is a string, with nothing inside it"),
            ("a type mismatch in when", make(agent(), {**later, "when": "$say.text > 3"}),
             "when: cannot compare string with number using >"),
            ("interpretation from a field that is not text", make(agent(), outputs={"figures": ["$figure"], "interpretation": "$say.ok"}),
             "interpretation reads a string field of agent step say, like $say.text"),
            ("interpretation from the whole answer", make(agent(), outputs={"figures": ["$figure"], "interpretation": "$say"}),
             "interpretation reads a string field of agent step say, like $say.text"),
            ("a finding from an agent", make(agent(), outputs={"figures": ["$figure"], "findings": [{"name": "said", "value": "$say.text"}]}),
             "finding said reads agent step say; a finding comes from capabilities, not a model"),
            ("data from an agent", make(agent(), outputs={"data": ["$say"]}),
             "data entry $say is an agent step, which returns no handle"),
            ("a figure from an agent", make(agent(), outputs={"figures": ["$say"]}),
             "figures entry $say is an agent step, which returns no chart"),
            ("an answer as a handle", make(agent(), {"id": "later", "use": "render.map", "with": {"grid": "$say"}}),
             "with.grid takes a handle; only a use step's result is a handle, and say is not one"),
        ]
        # A model's answer drives no capability and fills no fixed wording.
        refused = "reads agent step say; a model's answer is read only in when and in outputs.interpretation"
        statistic = {"id": "later", "use": "operations.statistic", "with": {"data": "$data"}}
        cases += [
            ("an answer in with", make(agent(), {**later, "with": {"grid": "$data", "title": "$say.text"}}), f"with {refused}"),
            ("an answer in a nested with value", make(agent(), {**statistic, "with": {"data": "$data", "stat": ["$say.points.0"]}}),
             f"with {refused}"),
            ("an answer in a method", make(agent(), {**later, "method": "The map shows {say.text}."}), f"method {refused}"),
            ("an answer in a stop reason", make(agent(), {"id": "halt", "stop": "failed", "when": "not $say.ok", "reason": "{say.text}"}),
             f"reason {refused}"),
            ("an answer in for_each", make(agent(), {**later, "for_each": "$say.points"}), f"for_each {refused}"),
            ("an answer read by another agent", make(agent(), {"id": "again", "agent": {
                "instruction": "Say it again.", "reads": ["$say.text"], "returns": {"text": "string"}}}), f"reads {refused}"),
        ]
        for name, source, expected in cases:
            with self.subTest(name):
                self.assertIn(expected, messages(source))
        # A list field is read whole or by index.
        listed = {"id": "later", "use": "render.map", "with": {"grid": "$data"}, "when": 'exists($say.points.0)'}
        self.assertEqual(messages(make(agent(), listed)), [])

    def test_validated_needs_benchmark_and_sign_off(self):
        found = messages(raw(status="validated"))
        self.assertEqual(found, [
            "a validated spec lists its benchmark cases",
            "a validated spec names who signed it off in validated_by",
            "a validated spec gives the sign-off date in validated_on",
        ])
        self.assertEqual(messages(raw(status="validated", benchmark=["map_nj_2024"], validated_by="Science team",
                                      validated_on="2026-11-01")), [])

    def test_all_errors_are_reported_together(self):
        bad = with_steps({"id": "a", "use": "render.globe"}, {"id": "b", "use": "render.map", "with": {"grid": "$nothing"}},
                         status="validated", products=None)
        self.assertGreaterEqual(len(messages(bad)), 5)

    def test_yaml_errors(self):
        self.assertEqual(messages("name: [unclosed"), ["the file is not valid YAML at line 1"])
        self.assertEqual(messages("- a list"), ["a spec is a mapping of fields"])


def _errors(source: Any) -> list[SpecError]:
    try:
        load(source)
    except SpecErrors as failure:
        return list(failure.errors)
    return []


class RealSpecTests(unittest.TestCase):
    def test_map(self):
        spec = load((FIXTURES / "map.yaml").read_text(encoding="utf-8"))
        self.assertEqual(spec.order(), ("data", "figure"))
        self.assertTrue(spec.runnable)
        self.assertEqual(spec.connectors(CAPABILITIES), {"earthdata"})
        self.assertEqual(approved_parameter(spec, "data", CAPABILITIES), "registry_default")
        self.assertIsNone(approved_parameter(spec, "figure", CAPABILITIES))

    def test_period_comparison(self):
        spec = load((FIXTURES / "period-comparison.yaml").read_text(encoding="utf-8"))
        self.assertEqual(spec.dependencies("b"), {"a"})
        self.assertEqual(spec.dependencies("diff"), {"mean_a", "mean_b"})
        self.assertEqual(spec.dependencies("mean_a"), {"a"})
        self.assertEqual(set(spec.order()[:2]), {"a", "b"})
        self.assertLess(spec.order().index("diff"), spec.order().index("stat"))
        approved = ["TEMPO_NO2", "TROPOMI_NO2", "OMI_NO2"]
        self.assertEqual(approved_parameter(spec, "a", CAPABILITIES), approved)
        self.assertEqual(approved_parameter(spec, "b", CAPABILITIES), approved)
        self.assertIsNone(approved_parameter(spec, "diff", CAPABILITIES))
        self.assertTrue(all(spec.always_runs(s) for s in spec.step_ids))


def _child(**changes: Any) -> dict[str, Any]:
    return raw(name="child", **changes)


def _parent(step: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return raw(name="parent", steps=[*copy.deepcopy(BASE["steps"]), step], **changes)


def _set_messages(sources: dict[str, Any]) -> list[str]:
    try:
        load_specs(sources, CAPABILITIES, DATASETS)
    except SpecErrors as failure:
        return [e.message for e in failure.errors]
    return []


class ChildWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ask_child = _child(steps=[
            *copy.deepcopy(BASE["steps"]),
            {"id": "confirm", "ask": {"kind": "pick_one", "question": "Which style?", "options": ["sequential", "diverging"]}},
        ])
        self.call = {"id": "run_child", "workflow": "child", "version": 1,
                     "with": {"quantity": "$quantity", "area": "$area", "dates": "$dates"}}

    def test_child_checks(self):
        cases = [
            ("not loaded", {**self.call, "version": 9}, "runs child v9, which is not loaded"),
            ("unknown input", {**self.call, "with": {**self.call["with"], "colour": "red"}}, "with gives colour, which is not an input of child"),
            ("missing input", {**self.call, "with": {"quantity": "$quantity", "area": "$area"}}, "with does not give dates, which child requires"),
            ("answer for a step that is not an ask", {**self.call, "answers": {"figure": "diverging"}}, "answers figure, which is not an ask step of child"),
        ]
        for name, step, expected in cases:
            with self.subTest(name):
                self.assertIn(expected, _set_messages({"child": self.ask_child, "parent": _parent(step)}))
        self.assertEqual(_set_messages({"child": self.ask_child, "parent": _parent({**self.call, "answers": {"confirm": "diverging"}})}), [])

    def test_validated_parent_needs_validated_children(self):
        sign_off = {"status": "validated", "benchmark": ["case_1"], "validated_by": "Science team", "validated_on": "2026-11-01"}
        found = _set_messages({"child": self.ask_child, "parent": _parent(self.call, **sign_off)})
        self.assertIn("a validated spec runs only validated workflows; child v1 is a draft", found)
        self.assertEqual(_set_messages({"child": {**self.ask_child, **sign_off}, "parent": _parent(self.call, **sign_off)}), [])

    def test_cycles_across_specs(self):
        call_parent = {"id": "back", "workflow": "parent", "version": 1,
                       "with": {"quantity": "$quantity", "area": "$area", "dates": "$dates"}}
        child = _child(steps=[*copy.deepcopy(BASE["steps"]), call_parent])
        self.assertIn("workflows child v1, parent v1 run one another in a cycle", _set_messages({"child": child, "parent": _parent(self.call)}))

    def test_load_spec_finds_children_in_a_catalogue(self):
        catalogue = load_specs({"child": self.ask_child}, CAPABILITIES, DATASETS)
        parent = load_spec(_parent(self.call), CAPABILITIES, DATASETS, catalogue=catalogue)
        self.assertEqual(parent.children(), (("child", 1),))
        self.assertFalse(parent.runnable)
        with self.assertRaises(SpecErrors):
            load_spec(_parent(self.call), CAPABILITIES, DATASETS)

    def test_repeated_version_in_a_set(self):
        self.assertIn("map v1 is also defined in first", _set_messages({"first": BASE, "second": BASE}))


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PackTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.map_text = (FIXTURES / "map.yaml").read_text(encoding="utf-8")

    def test_layout_loads(self):
        _write(self.root, "core/workflows/map/1.yaml", self.map_text)
        _write(self.root, "core/workflows/map/2.yaml", self.map_text.replace("version: 1", "version: 2").replace(
            "description: Map one quantity over an area for a date or date range.", "description: Map a quantity, second version."))
        _write(self.root, "science/workflows/period-comparison/1.yaml", (FIXTURES / "period-comparison.yaml").read_text(encoding="utf-8"))
        _write(self.root, "science/notes.md", "Not a workflow.")
        catalogue = load_packs(self.root, CAPABILITIES, DATASETS)
        self.assertEqual(len(catalogue), 3)
        self.assertEqual(catalogue.versions("map"), (1, 2))
        self.assertEqual(catalogue.latest("map").version, 2)
        self.assertEqual(catalogue.get("map", 1).version, 1)
        self.assertEqual([(s.name, s.version) for s in catalogue], [("map", 1), ("map", 2), ("period-comparison", 1)])
        self.assertIn(("map", 2), catalogue)
        self.assertEqual(catalogue.index(), (
            "map v2 (draft): Map a quantity, second version.\n"
            "period-comparison v1 (draft): Compare one quantity over an area between two periods, with the difference."
        ))
        with self.assertRaises(UnknownWorkflow):
            catalogue.get("map", 3)
        with self.assertRaises(UnknownWorkflow):
            catalogue.latest("trend")
        with self.assertRaises(LookupError):
            catalogue.versions("trend")

    def test_errors_across_files_are_reported_together(self):
        _write(self.root, "core/workflows/map/1.yaml", self.map_text)
        _write(self.root, "core/workflows/maps/1.yaml", self.map_text.replace("version: 1", "version: 3"))
        _write(self.root, "core/workflows/map/2.yaml", self.map_text)
        _write(self.root, "core/workflows/map/v3.yaml", self.map_text)
        _write(self.root, "core/workflows/map/03.yaml", self.map_text)
        _write(self.root, "core/workflows/stray.yaml", self.map_text)
        _write(self.root, "extra/workflows/broken/1.yaml", "name: broken\nversion: 1\nsteps: [\n")
        _write(self.root, "extra/workflows/odd/1.yaml", self.map_text.replace("name: map", "name: odd").replace("grid: $data", "grid: $nothing"))
        with self.assertRaises(SpecErrors) as caught:
            load_packs(self.root, CAPABILITIES, DATASETS)
        found = sorted((e.source, e.message) for e in caught.exception.errors)
        self.assertEqual(found, sorted([
            ("core/workflows/map/03.yaml", "a spec file is named by its version, like 1.yaml"),
            ("core/workflows/map/v3.yaml", "a spec file is named by its version, like 1.yaml"),
            ("core/workflows/map/2.yaml", "the spec's version is 1 but its file is 2.yaml"),
            ("core/workflows/maps/1.yaml", "the spec's name is map but its directory is maps"),
            ("core/workflows/maps/1.yaml", "the spec's version is 3 but its file is 1.yaml"),
            ("core/workflows/stray.yaml", "is not a workflow directory"),
            ("extra/workflows/broken/1.yaml", "the file is not valid YAML at line 4"),
            ("extra/workflows/odd/1.yaml", "$nothing names no input or step"),
        ]))
        odd = next(e for e in caught.exception.errors if e.source == "extra/workflows/odd/1.yaml")
        self.assertEqual((odd.workflow, odd.version, odd.step), ("odd", 1, "figure"))

    def test_missing_root(self):
        with self.assertRaises(SpecErrors):
            load_packs(self.root / "absent", CAPABILITIES, DATASETS)

    def test_empty_catalogue(self):
        self.assertEqual(len(load_packs(self.root, CAPABILITIES, DATASETS)), 0)
        self.assertEqual(WorkflowCatalogue().index(), "")


if __name__ == "__main__":
    unittest.main()
