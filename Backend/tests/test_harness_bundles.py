"""The exploration bundles: what each holds, and that they follow the registry."""
import unittest

from pydantic import BaseModel

from tta_backend.harness.capabilities import Capability, CapabilityRegistry, Group, UnknownCapability
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.specialists.bundles import BUNDLES, Bundle, UnknownBundle, get_bundle

# Capabilities deliberately in no bundle.
UNREACHABLE: frozenset[str] = frozenset()

SNAPSHOT = {
    "analyse": [
        "operations.aggregate_time",
        "operations.daily_peak",
        "operations.difference",
        "operations.regional_series",
        "operations.statistic",
        "operations.threshold",
        "operations.vertical_profile",
        "render.map",
        "render.monitor_map",
        "render.polygon_map",
        "render.profile",
        "render.table",
        "render.timeseries",
        "sources.areas.resolve",
        "sources.monitors.annual_summary",
        "sources.monitors.daily_summary",
        "sources.monitors.exceedance_days",
        "sources.monitors.find",
        "sources.monitors.quarterly_summary",
        "sources.monitors.readings",
        "sources.satellite.check_availability",
        "sources.satellite.describe",
        "sources.satellite.get_data",
        "sources.satellite.get_point_series",
        "sources.satellite.search",
        "sources.smoke.get_extent",
    ],
    "investigate": [
        "explain.citations",
        "explain.lineage",
        "explain.reliability",
        "operations.aggregate_time",
        "operations.daily_peak",
        "operations.difference",
        "operations.regional_series",
        "operations.statistic",
        "operations.threshold",
        "operations.vertical_profile",
        "render.map",
        "render.monitor_map",
        "render.polygon_map",
        "render.profile",
        "render.table",
        "render.timeseries",
    ],
    "find-data": [
        "sources.monitors.find",
        "sources.satellite.check_availability",
        "sources.satellite.describe",
        "sources.satellite.search",
    ],
}


class _P(BaseModel):
    pass


async def _run(call):  # type: ignore[no-untyped-def]
    return call.done()


def _cap(name: str, *, retrieves: bool = False, workflow_only: bool = False) -> Capability[_P]:
    return Capability(
        name=name, description="A test capability.", params=_P, output_kind="scalar", label="Testing",
        method="Done.", retrieves=retrieves, retry_safe=True, workflow_only=workflow_only, run=_run,
    )


def _registry(*capabilities: Capability[_P]) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for group in sorted({c.group for c in capabilities}):
        registry.add_group(Group(group, f"The {group} group."))
    for capability in capabilities:
        registry.register(capability)
    return registry


class BundleTests(unittest.TestCase):
    def test_snapshot(self) -> None:
        registry = build_registry()
        found = {name: [c.name for c in b.capabilities(registry)] for name, b in BUNDLES.items()}
        self.assertEqual(SNAPSHOT, found)

    def test_workflow_only_is_excluded_everywhere(self) -> None:
        registry = _registry(
            _cap("sources.x.read"),
            _cap("sources.monitors.find", retrieves=True),
            _cap("sources.x.secret", workflow_only=True),
            _cap("operations.sum"),
            _cap("operations.regulatory", workflow_only=True),
            _cap("render.chart"),
            _cap("render.hidden", workflow_only=True),
            _cap("explain.why"),
            _cap("explain.hidden", workflow_only=True),
        )
        for bundle in BUNDLES.values():
            names = [c.name for c in bundle.capabilities(registry)]
            self.assertTrue(names)
            self.assertFalse({"sources.x.secret", "operations.regulatory", "render.hidden", "explain.hidden"} & set(names))

    def test_workflow_only_named_addition_is_excluded(self) -> None:
        registry = _registry(_cap("sources.x.find", retrieves=True, workflow_only=True))
        bundle = Bundle(name="find-data", description="A bundle.", extra=("sources.x.find",))
        self.assertEqual((), bundle.capabilities(registry))

    def test_new_source_joins_analyse_and_find_data(self) -> None:
        registry = _registry(
            _cap("sources.x.get", retrieves=True),
            _cap("sources.x.describe"),
            _cap("operations.sum"),
            _cap("sources.monitors.find", retrieves=True),
        )
        names = lambda n: [c.name for c in BUNDLES[n].capabilities(registry)]  # noqa: E731
        self.assertEqual(
            ["operations.sum", "sources.monitors.find", "sources.x.describe", "sources.x.get"], names("analyse")
        )
        self.assertEqual(["sources.monitors.find", "sources.x.describe"], names("find-data"))
        self.assertEqual(["operations.sum"], names("investigate"))

    def test_prefix_matches_on_a_dotted_boundary(self) -> None:
        registry = _registry(_cap("sources_extra.get"), _cap("sources.x.get"))
        names = [c.name for c in BUNDLES["analyse"].capabilities(registry)]
        self.assertEqual(["sources.x.get"], names)

    def test_unknown_bundle(self) -> None:
        with self.assertRaises(UnknownBundle):
            get_bundle("everything")
        for name in BUNDLES:
            self.assertIs(BUNDLES[name], get_bundle(name))

    def test_missing_named_addition_fails(self) -> None:
        registry = _registry(_cap("sources.x.describe"))
        with self.assertRaises(UnknownCapability):
            BUNDLES["find-data"].capabilities(registry)

    def test_groups_match_capabilities(self) -> None:
        registry = build_registry()
        for bundle in BUNDLES.values():
            groups = bundle.groups(registry)
            self.assertEqual(sorted({c.group for c in bundle.capabilities(registry)}), [g.name for g in groups])
            self.assertEqual([g.name for g in groups], sorted(g.name for g in groups))
        self.assertEqual(
            ["sources.monitors", "sources.satellite"],
            [g.name for g in BUNDLES["find-data"].groups(registry)],
        )

    def test_descriptions_are_one_line(self) -> None:
        for bundle in BUNDLES.values():
            self.assertTrue(bundle.description.strip())
            self.assertNotIn("\n", bundle.description)
        with self.assertRaises(ValueError):
            Bundle(name="analyse", description="two\nlines")

    def test_every_capability_is_reachable(self) -> None:
        registry = build_registry()
        reachable = {c.name for b in BUNDLES.values() for c in b.capabilities(registry)}
        every = {c.name for g in registry.groups() for c in registry.in_group(g.name)}
        self.assertEqual(set(), every - reachable - UNREACHABLE)
