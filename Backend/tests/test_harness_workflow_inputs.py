"""Workflow inputs: presence, defaults, types and ranges per input type, and area resolution."""
import tempfile
import unittest
from typing import Any

from tests.test_harness_grids import NJ
from tests.test_harness_workflow_runner import DATASETS, Script, fake_registry, make, spec
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.testing import FakePlaces, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory
from tta_backend.harness.workflows.inputs import InputProblem, ResolvedInputs, resolve_inputs
from tta_backend.harness.workflows.loader import load_spec
from tta_backend.harness.workflows.spec import WorkflowSpec

QUANTITIES = frozenset({"no2_tropospheric_column", "ozone_surface"})

INPUTS: dict[str, Any] = {
    "quantity": {"type": "quantity", "ask": "Which quantity?"},
    "area": {"type": "area", "ask": "Which area?"},
    "day": {"type": "date", "default": "2024-06-01"},
    "span": {"type": "date_range", "default": {"start": "2024-06-01", "end": "2024-06-03"}},
    "days": {"type": "date_list", "default": ["2024-06-01"]},
    "site": {"type": "monitor", "default": "34-013-0003"},
    "change": {"type": "choice", "options": ["absolute", "percent"], "default": "absolute"},
    "window": {"type": "number", "default": 7, "min": 1, "max": 31},
    "note": {"type": "text", "default": "none", "max": 20},
}
HANDLE_INPUTS: dict[str, Any] = {"grid": {"type": "handle", "kind": "scalar", "ask": "Which result?"}}


class InputCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.registry = fake_registry(Script())
        self.places = FakePlaces({"new jersey": NJ, "nj": NJ})
        self.env: RunEnv = await make_run_env(tmp.name, context=make_run_context(), places=self.places)
        self.spec = self.load(INPUTS)

    def load(self, inputs: dict[str, Any]) -> WorkflowSpec:
        return load_spec(spec([make("one")], {"data": ["$one"]}, inputs=inputs), self.registry, DATASETS)

    async def resolve(self, raw: dict[str, Any], workflow: WorkflowSpec | None = None) -> ResolvedInputs | InputProblem:
        return await resolve_inputs(self.env, workflow or self.spec, raw, self.registry, quantities=QUANTITIES)

    async def entries(self) -> list[tuple[str | None, str | None, str]]:
        found = await self.env.context.storage.traces.entries("user-a", self.env.run_id)
        return [(e.capability, e.step, e.status) for e in found]


class InputTests(InputCase):
    async def test_required_inputs_missing_are_listed_with_their_asks(self):
        for raw, names in (({}, ["quantity", "area"]), ({"area": "nj"}, ["quantity"]), ({"quantity": None, "area": "nj"}, ["quantity"])):
            with self.subTest(raw=raw):
                problem = await self.resolve(raw)
                assert isinstance(problem, InputProblem)
                self.assertEqual(problem.status, "needs_input")
                self.assertEqual([m.name for m in problem.missing], names)
                for item in problem.missing:
                    self.assertIn(item.ask, problem.reason)
        self.assertEqual(await self.entries(), [])

    async def test_defaults_are_applied(self):
        resolved = await self.resolve({"quantity": "ozone_surface", "area": "nj"})
        assert isinstance(resolved, ResolvedInputs)
        values = resolved.values
        self.assertEqual(values["day"], "2024-06-01")
        self.assertEqual(values["span"], {"start": "2024-06-01", "end": "2024-06-03"})
        self.assertEqual((values["window"], values["change"]), (7, "absolute"))
        self.assertTrue(str(values["area"]).startswith("polygons_"))
        self.assertEqual(resolved.names, {"area": "New Jersey"})
        self.assertEqual(resolved.disclosures, ("'nj' was resolved to New Jersey.",))

    async def test_nullable_inputs_are_none_unless_given_and_a_null_area_is_not_resolved(self):
        workflow = self.load({
            **INPUTS,
            "named": {"type": "text", "default": None},
            "focus": {"type": "area", "default": None},
            "base": {"type": "handle", "kind": "scalar", "default": None},
        })
        for raw in ({"quantity": "ozone_surface", "area": "nj"}, {"quantity": "ozone_surface", "area": "nj", "named": None, "focus": None}):
            with self.subTest(raw=raw):
                resolved = await self.resolve(raw, workflow)
                assert isinstance(resolved, ResolvedInputs)
                self.assertEqual((resolved.values["named"], resolved.values["focus"], resolved.values["base"]), (None, None, None))
                self.assertEqual(resolved.names, {"area": "New Jersey"})
        self.assertEqual([step for _, step, _ in await self.entries()], ["input.area", "input.area"])
        resolved = await self.resolve({"quantity": "ozone_surface", "area": "nj", "named": "TEMPO", "focus": "new jersey"}, workflow)
        assert isinstance(resolved, ResolvedInputs)
        self.assertEqual(resolved.values["named"], "TEMPO")
        self.assertTrue(str(resolved.values["focus"]).startswith("polygons_"))

    async def test_each_type_accepts_good_values_and_refuses_bad_ones_naming_the_input(self):
        good: list[tuple[str, Any, Any]] = [
            ("quantity", "no2_tropospheric_column", "no2_tropospheric_column"),
            ("day", "2024-02-29", "2024-02-29"),
            ("span", {"start": "2024-06-01", "end": "2024-06-01"}, {"start": "2024-06-01", "end": "2024-06-01"}),
            ("days", ["2024-06-02", "2024-06-01", "2024-06-02"], ["2024-06-02", "2024-06-01"]),
            ("site", "06-037-1103", "06-037-1103"),
            ("change", "percent", "percent"),
            ("window", 31, 31),
            ("window", 2.5, 2.5),
            ("note", "short note", "short note"),
        ]
        bad: list[tuple[str, Any, str]] = [
            ("quantity", "unobtainium", "Input quantity must be a quantity the dataset registry knows"),
            ("quantity", 5, "Input quantity must be a quantity"),
            ("day", "2024-02-30", "Input day must be an ISO date"),
            ("day", "June 1st", "Input day must be an ISO date"),
            ("span", "2024-06-01", "Input span must be a date range"),
            ("span", {"start": "2024-06-01"}, "Input span must be a date range"),
            ("span", {"start": "2024-06-03", "end": "2024-06-01"}, "Input span ends before it starts."),
            ("days", [], "Input days must be a list of ISO dates."),
            ("days", ["2024-06-01", "yesterday"], "Input days must be a list of ISO dates."),
            ("site", "has space", "Input site must be a monitor id."),
            ("change", "relative", "Input change must be one of: absolute, percent."),
            ("window", 0, "Input window must be at least 1."),
            ("window", 32, "Input window must be at most 31."),
            ("window", True, "Input window must be a number."),
            ("window", "7", "Input window must be a number."),
            ("window", float("nan"), "Input window must be a number."),
            ("note", "x" * 21, "Input note must be 1 to 20 characters long."),
            ("note", 3, "Input note must be text."),
        ]
        base = {"quantity": "ozone_surface", "area": "nj"}
        for name, value, expected in good:
            with self.subTest(good=name, value=value):
                resolved = await self.resolve({**base, name: value})
                assert isinstance(resolved, ResolvedInputs), resolved
                self.assertEqual(resolved.values[name], expected)
        for name, value, wording in bad:
            with self.subTest(bad=name, value=value):
                problem = await self.resolve({**base, name: value})
                assert isinstance(problem, InputProblem), problem
                self.assertEqual((problem.status, problem.category), ("failed", FailureCategory.BAD_INPUT))
                self.assertTrue(problem.reason.startswith(wording), problem.reason)
                if isinstance(value, str) and len(value) > 8:
                    self.assertNotIn(value, problem.reason)

    async def test_an_unknown_input_is_refused_without_echoing_a_long_key(self):
        problem = await self.resolve({"quantity": "ozone_surface", "area": "nj", "extra": 1})
        assert isinstance(problem, InputProblem)
        self.assertEqual(problem.reason, "Input extra is not an input of this workflow.")
        long_key = "k" * 200
        problem = await self.resolve({"quantity": "ozone_surface", "area": "nj", long_key: 1})
        assert isinstance(problem, InputProblem)
        self.assertNotIn(long_key, problem.reason)

    async def test_a_bad_input_writes_no_entry_even_with_area_wording(self):
        problem = await self.resolve({"quantity": "ozone_surface", "area": "nj", "window": 99})
        assert isinstance(problem, InputProblem)
        self.assertEqual(await self.entries(), [])
        self.assertEqual(self.places.asked, [])

    async def test_a_handle_input_must_be_the_users_own_of_the_declared_kind(self):
        workflow = self.load(HANDLE_INPUTS)
        made = await invoke(self.env, self.registry.get("ops.make"), {"label": "x"})
        resolved = await self.resolve({"grid": made.handle}, workflow)
        assert isinstance(resolved, ResolvedInputs)
        self.assertEqual(resolved.values["grid"], made.handle)
        area = await invoke(self.env, areas.resolve, {"place": "nj"})
        for value in ("scalar_" + "0" * 32, "not a handle", area.handle, 7):
            with self.subTest(value=value):
                problem = await self.resolve({"grid": value}, workflow)
                assert isinstance(problem, InputProblem)
                self.assertEqual(problem.reason, "Input grid must be a scalar handle from this conversation's results.")


class AreaTests(InputCase):
    async def test_wording_is_resolved_as_a_traced_input_step(self):
        for raw in ("New Jersey", {"place": "New Jersey"}):
            with self.subTest(raw=raw):
                resolved = await self.resolve({"quantity": "ozone_surface", "area": raw})
                assert isinstance(resolved, ResolvedInputs)
                self.assertTrue(str(resolved.values["area"]).startswith("polygons_"))
        self.assertEqual(await self.entries(), [("sources.areas.resolve", "input.area", "done")] * 2)
        self.assertEqual(self.places.asked, ["New Jersey", "New Jersey"])

    async def test_an_area_handle_is_checked_and_not_resolved_again(self):
        area = await invoke(self.env, areas.resolve, {"place": "nj"})
        resolved = await self.resolve({"quantity": "ozone_surface", "area": area.handle})
        assert isinstance(resolved, ResolvedInputs)
        self.assertEqual(resolved.values["area"], area.handle)
        self.assertEqual(resolved.names, {"area": "New Jersey"})
        self.assertEqual(len(await self.entries()), 1)

    async def test_a_handle_that_is_not_this_users_area_is_bad_input(self):
        scalar = await invoke(self.env, self.registry.get("ops.make"), {"label": "x"})
        for handle in ("polygons_" + "a" * 32, scalar.handle):
            with self.subTest(handle=handle):
                problem = await self.resolve({"quantity": "ozone_surface", "area": handle})
                assert isinstance(problem, InputProblem)
                self.assertEqual(problem.reason, "Input area must be an area handle from this conversation's results, or a place.")

    async def test_unresolvable_wording_ends_with_the_capabilitys_question(self):
        problem = await self.resolve({"quantity": "ozone_surface", "area": "Atlantis"})
        assert isinstance(problem, InputProblem)
        self.assertEqual(problem.status, "needs_input")
        assert problem.question is not None
        self.assertEqual(problem.question["kind"], "text")
        self.assertTrue(problem.reason.startswith("'Atlantis' could not be found as a place."))
        self.assertEqual(await self.entries(), [("sources.areas.resolve", "input.area", "waiting")])

    async def test_badly_shaped_area_values_are_bad_input(self):
        for raw in (5, {"place": 5}, {"where": "nj"}, "   ", "x" * 501):
            with self.subTest(raw=raw):
                problem = await self.resolve({"quantity": "ozone_surface", "area": raw})
                assert isinstance(problem, InputProblem)
                self.assertEqual(problem.category, FailureCategory.BAD_INPUT)
                self.assertTrue(problem.reason.startswith("Input area"))


if __name__ == "__main__":
    unittest.main()
