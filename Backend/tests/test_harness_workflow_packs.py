"""The shipped workflow specs under packs/: they load, run in the order and with the connectors stated here, and read only result fields their capabilities set."""
import unittest
from pathlib import Path

from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.workflows.expressions import Ref
from tta_backend.harness.workflows.loader import approved_parameter, load_packs
from tta_backend.harness.workflows.spec import UseStep, WorkflowSpec

PACKS = Path(__file__).resolve().parent.parent / "packs"
CAPABILITIES = build_registry()
DATASETS = load_registry()
CATALOGUE = load_packs(PACKS, CAPABILITIES, DATASETS)

# Step id order and connectors of each spec; a change to a procedure shows up here.
ORDERS: dict[str, tuple[str, ...]] = {
    "map": ("data", "figure"),
    "timeseries": ("data", "series", "figure", "average", "point", "point_figure", "point_average"),
    "vertical-profile": ("data", "profile", "figure"),
    "monitor-readings": (
        "find", "readings", "figure_series", "summary", "figure_map", "figure_table", "exceedances", "figure_exceedances",
    ),
    "smoke-map": ("extent", "figure"),
    "period-comparison": ("first_data", "second_data", "first_mean", "second_mean", "diff", "figure", "stat"),
    "satellite-vs-monitor": ("find", "data", "readings", "pair", "figure_series", "monitor_values", "figure_map"),
    "event-vs-baseline": (
        "event", "candidates", "available", "no_record", "too_few_available", "days", "checked", "too_few_confirmed",
        "baseline",
        "too_few_kept", "event_mean", "baseline_mean", "diff", "figure",
    ),
}
CONNECTORS: dict[str, frozenset[str]] = {
    "map": frozenset({"earthdata"}),
    "timeseries": frozenset({"earthdata"}),
    "vertical-profile": frozenset({"earthdata"}),
    "monitor-readings": frozenset({"epa_aqs"}),
    "smoke-map": frozenset(),
    "period-comparison": frozenset({"earthdata"}),
    "satellite-vs-monitor": frozenset({"earthdata", "epa_aqs"}),
    "event-vs-baseline": frozenset({"earthdata"}),
}

# What each capability a spec reads sets, read from the capability code. A name under
# summary, values or quality that is not listed here is not known to exist.
SCALARS = frozenset({"handle", "kind", "unit", "product"})
SECTIONS = ("summary", "values", "quality")
FIELDS: dict[str, dict[str, frozenset[str]]] = {
    "sources.satellite.get_data": {
        "summary": frozenset({"product", "area", "period", "parts"}),
        "values": frozenset({
            "area_name", "period_text", "rule", "qa_status", "date_role", "masking", "dropped_dates",
            "dropped_after_masking", "days_without_valid_data", "nearest_dates", "situation",
        }),
        "quality": frozenset({
            "valid_fraction", "valid_cells", "region_cells", "qa_status", "qa_pass_rate", "days_requested",
            "days_with_data", "missing_days", "missing_days_listed",
        }),
    },
    "sources.satellite.check_availability": {
        "values": frozenset({
            "area_name", "period_text", "available", "situation", "ruled_out", "granules", "days_with_data",
            "days_without_data",
        }),
    },
    "operations.select_comparison_days": {
        "values": frozenset({
            "days", "event_date", "day_type", "window_days", "limit", "latest", "selection", "situation",
        }),
    },
    "operations.check_comparison_days": {
        "values": frozenset({"days", "kept", "confirmed", "dropped", "event_date", "latest"}),
    },
    "sources.satellite.get_point_series": {
        "summary": frozenset({"points", "days_with_data", "first", "last"}),
        "values": frozenset({"area_name", "period_text", "rule", "situation"}),
    },
    "operations.aggregate_time": {
        "summary": frozenset({"stat", "time_steps", "time_steps_total", "mean", "min", "max", "max_location", "min_location", "cells"}),
        "values": frozenset({"basis"}),
    },
    "operations.regional_series": {
        "summary": frozenset({"stat", "points", "time_steps_total", "first", "last", "min", "max"}),
        "values": frozenset({"area_name", "weighting"}),
    },
    "operations.statistic": {
        "summary": frozenset({"value", "stat", "basis", "spread", "spread_basis", "count", "percentile"}),
    },
    "operations.difference": {
        "summary": frozenset({
            "change", "mean", "min", "max", "max_location", "min_location", "paired_cells", "cells_valid_in_either",
            "paired_fraction", "mean_difference", "percent_change", "percent_change_note", "withheld_cells",
            "area_exceeding_threshold", "products",
        }),
        "values": frozenset({"period_a", "period_b", "paired_percent"}),
    },
    "operations.pair_with_monitors": {
        "summary": frozenset({
            "monitors", "monitors_paired", "monitors_without_pairs", "paired_days", "pairs", "pooled_r", "per_monitor",
        }),
        "values": frozenset({"pairing", "ground_unit", "pollutant", "situation"}),
    },
    "operations.vertical_profile": {
        "summary": frozenset({"layers", "vertical_dim", "axes", "default_axis", "layer_order", "timesteps", "aggregation"}),
    },
    "sources.monitors.find": {
        "summary": frozenset({"monitors", "inside", "nearest_km", "farthest_km", "period"}),
        "values": frozenset({"monitor_ids"}),
    },
    "sources.monitors.readings": {
        "summary": frozenset({"monitors", "values", "min", "max", "mean", "period"}),
        "values": frozenset({"monitor_ids"}),
    },
    "sources.monitors.daily_summary": {
        "summary": frozenset({"rows", "monitors", "min", "max", "mean", "period"}),
        "values": frozenset({"standard", "value"}),
    },
    "sources.monitors.exceedance_days": {
        "summary": frozenset({"exceedance_days", "monitor_days_exceeding", "monitor_days_checked", "monitors", "max", "period"}),
        "values": frozenset({"standard", "level", "value", "days", "days_truncated"}),
    },
    "sources.smoke.get_extent": {
        "summary": frozenset({"plumes", "by_density", "area_km2", "first_start", "last_end"}),
        "values": frozenset({"source_url", "sha256"}),
        "quality": frozenset({"repaired_records", "skipped_records"}),
    },
}
RESULT_FIELDS = frozenset(CapabilityResult.model_fields)

# Latest version of each workflow. Version 2 of the three below adds the optional cadence input.
LATEST: dict[str, int] = {name: 1 for name in ORDERS} | {
    "map": 2, "timeseries": 2, "period-comparison": 2, "monitor-readings": 3,
}
# Step orders that differ in a later version, by (name, version).
VERSION_ORDERS: dict[tuple[str, int], tuple[str, ...]] = {
    ("monitor-readings", 2): (
        "find", "readings", "figure_series", "reading_mean", "reading_max", "summary", "figure_map", "figure_table",
        "exceedances", "figure_exceedances",
    ),
    ("monitor-readings", 3): (
        "find", "readings", "figure_series", "reading_mean", "reading_max", "summary", "figure_map", "figure_table",
        "exceedances", "figure_exceedances",
    ),
}
CADENCE_STEPS = {
    "map": ("data",),
    "timeseries": ("data", "point"),
    "period-comparison": ("first_data", "second_data"),
}


def unknown_path(capability: str, path: tuple[str, ...]) -> str | None:
    """Why a result path cannot be read from a capability's result, or None when it can."""
    known = FIELDS.get(capability, {})
    head = path[0]
    if head in SCALARS:
        return None if len(path) == 1 else f"{head} holds a single value"
    if head in SECTIONS:
        if len(path) == 1:
            return None
        if path[1] not in known.get(head, frozenset()):
            return f"{capability} is not known to set {head}.{path[1]}"
        return None
    if head in RESULT_FIELDS:
        return f"{head} is not read from a result"
    if head not in known.get("values", frozenset()):
        return f"{capability} is not known to set the value {head}"
    return None


def result_reads(spec: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
    """(where, capability, path) for every path read from a step's result, in steps and outputs."""
    found: list[tuple[str, str, tuple[str, ...]]] = []

    def capability_of(step_id: str) -> str:
        step = spec.step(step_id)
        assert isinstance(step, UseStep), f"{spec.name}: {step_id} is read as a result but is not a use step"
        return step.use

    def note(where: str, owner: str | None, ref: object) -> None:
        if not isinstance(ref, Ref) or not ref.path and ref.root != "own":
            return
        if ref.root == "step":
            found.append((where, capability_of(ref.name), ref.path))
        elif ref.root == "own":
            assert owner is not None
            found.append((where, capability_of(owner), (ref.name, *ref.path)))

    for step_id in spec.step_ids:
        for ref in spec.compiled(step_id).references():
            note(f"step {step_id}", step_id, ref)
    outputs = spec.compiled_outputs
    for finding in outputs.findings:
        for ref in finding.references():
            note(f"finding {finding.name}", None, ref)
    for entry in (*outputs.figures, *outputs.data):
        note("outputs", None, entry.ref)
    return found


class PackLoadTests(unittest.TestCase):
    def test_exactly_the_shipped_workflows_load(self):
        expected = [(name, version) for name in sorted(ORDERS) for version in range(1, LATEST[name] + 1)]
        self.assertEqual([(s.name, s.version) for s in CATALOGUE], expected)

    def test_each_is_a_runnable_draft(self):
        for spec in CATALOGUE:
            with self.subTest(spec.name):
                self.assertEqual(spec.status, "draft")
                self.assertTrue(spec.runnable, spec.not_runnable_reason)
                self.assertTrue(spec.examples)

    def test_the_index_has_one_line_per_workflow(self):
        lines = CATALOGUE.index().splitlines()
        self.assertEqual(len(lines), len(ORDERS))
        for line, name in zip(lines, sorted(ORDERS), strict=True):
            self.assertTrue(line.startswith(f"{name} v{LATEST[name]} (draft): "), line)

    def test_step_order(self):
        for name, order in ORDERS.items():
            for version in range(1, LATEST[name] + 1):
                with self.subTest(f"{name} v{version}"):
                    self.assertEqual(CATALOGUE.get(name, version).order(), VERSION_ORDERS.get((name, version), order))

    def test_connectors(self):
        for name, connectors in CONNECTORS.items():
            for version in range(1, LATEST[name] + 1):
                with self.subTest(f"{name} v{version}"):
                    self.assertEqual(CATALOGUE.get(name, version).connectors(CAPABILITIES), connectors)

    def test_specs_leave_approved_to_the_runner(self):
        for spec in CATALOGUE:
            for step in spec.steps:
                if isinstance(step, UseStep):
                    self.assertNotIn("approved", step.with_, f"{spec.name} {step.id}")


class CadenceTests(unittest.TestCase):
    def test_version_one_is_unchanged_and_takes_no_cadence(self):
        for name in CADENCE_STEPS:
            spec = CATALOGUE.get(name, 1)
            with self.subTest(name):
                self.assertNotIn("cadence", spec.inputs)
                for step_id in CADENCE_STEPS[name]:
                    self.assertNotIn("cadence", spec.step(step_id).with_)

    def test_version_two_takes_an_optional_cadence_choice(self):
        for name in CADENCE_STEPS:
            with self.subTest(name):
                self.assertEqual(CATALOGUE.versions(name), (1, 2))
                declared = CATALOGUE.get(name, 2).inputs["cadence"]
                self.assertEqual(declared.type, "choice")
                self.assertEqual(tuple(declared.options or ()), ("hourly", "daily", "monthly"))
                self.assertTrue(declared.optional)
                self.assertTrue(declared.nullable)

    def test_version_two_passes_the_cadence_to_every_retrieval(self):
        for name, step_ids in CADENCE_STEPS.items():
            spec = CATALOGUE.get(name, 2)
            retrievals = [
                s.id for s in spec.steps
                if isinstance(s, UseStep) and s.use in ("sources.satellite.get_data", "sources.satellite.get_point_series")
            ]
            with self.subTest(name):
                self.assertEqual(tuple(retrievals), step_ids)
                for step_id in retrievals:
                    self.assertEqual(spec.step(step_id).with_["cadence"], "$cadence")

    def test_vertical_profile_is_left_alone(self):
        self.assertNotIn("cadence", CATALOGUE.get("vertical-profile", 1).inputs)


class ProductTests(unittest.TestCase):
    def approved(self, name: str, step: str) -> object:
        return approved_parameter(CATALOGUE.get(name, 1), step, CAPABILITIES)

    def test_map_leaves_the_choice_to_the_registry(self):
        self.assertEqual(self.approved("map", "data"), "registry_default")

    def test_profile_is_limited_to_the_layered_product(self):
        self.assertEqual(self.approved("vertical-profile", "data"), ["TEMPO_O3PROF"])

    def test_both_periods_share_one_approved_list_and_one_product(self):
        first = self.approved("period-comparison", "first_data")
        self.assertIsInstance(first, list)
        self.assertEqual(first, self.approved("period-comparison", "second_data"))
        spec = CATALOGUE.get("period-comparison", 1)
        self.assertEqual(spec.dependencies("second_data"), {"first_data"})

    def test_the_monitor_comparison_is_limited_to_the_no2_products_and_asks_for_daily_values(self):
        self.assertEqual(self.approved("satellite-vs-monitor", "data"), ["TEMPO_NO2", "TROPOMI_NO2", "OMI_NO2"])
        self.assertEqual(CATALOGUE.get("satellite-vs-monitor", 1).step("data").with_["cadence"], "daily")

    def test_comparison_takes_the_second_period_minus_the_first(self):
        spec = CATALOGUE.get("period-comparison", 1)
        with_ = {key: (node.name if isinstance(node, Ref) else None) for key, node in spec.compiled("diff").with_.items}
        self.assertEqual((with_["a"], with_["b"]), ("first_mean", "second_mean"))

    def test_each_satellite_workflow_takes_an_optional_named_product(self):
        for name in ("map", "timeseries", "vertical-profile", "period-comparison", "satellite-vs-monitor"):
            declared = CATALOGUE.get(name, 1).inputs["named"]
            self.assertEqual(declared.type, "text", name)
            self.assertTrue(declared.optional, name)

    def test_listed_products_measure_a_quantity(self):
        for spec in CATALOGUE:
            if isinstance(spec.products, dict):
                for listed in spec.products.values():
                    if isinstance(listed, tuple):
                        for key in listed:
                            self.assertIsNotNone(DATASETS[key].quantity, key)


class ResultPathTests(unittest.TestCase):
    """The loader does not check the path after $step, so each is checked here against FIELDS."""

    def test_every_path_read_from_a_result_is_known(self):
        total = 0
        for spec in CATALOGUE:
            for where, capability, path in result_reads(spec):
                total += 1
                with self.subTest(f"{spec.name}: {where}: {capability} {'.'.join(path)}"):
                    self.assertIsNone(unknown_path(capability, path))
        self.assertGreater(total, 20)

    def test_input_paths_are_the_parts_of_a_date_range(self):
        for spec in CATALOGUE:
            for step_id in spec.step_ids:
                for ref in spec.compiled(step_id).references():
                    if isinstance(ref, Ref) and ref.root == "input" and ref.path:
                        with self.subTest(f"{spec.name}: {step_id}: {ref.text()}"):
                            self.assertEqual(spec.inputs[ref.name].type, "date_range")
                            self.assertIn(ref.path, {("start",), ("end",)})

    def test_the_guard_rejects_a_path_nothing_sets(self):
        get_data = "sources.satellite.get_data"
        self.assertIsNone(unknown_path(get_data, ("quality", "valid_fraction")))
        self.assertIsNone(unknown_path(get_data, ("summary",)))
        self.assertIsNone(unknown_path(get_data, ("product",)))
        self.assertIsNotNone(unknown_path(get_data, ("summary", "mean")))
        self.assertIsNotNone(unknown_path(get_data, ("product", "name")))
        self.assertIsNotNone(unknown_path(get_data, ("situation_text",)))
        self.assertIsNotNone(unknown_path("render.map", ("summary", "title")))
        self.assertIsNone(unknown_path("sources.monitors.exceedance_days", ("level",)))


if __name__ == "__main__":
    unittest.main()
