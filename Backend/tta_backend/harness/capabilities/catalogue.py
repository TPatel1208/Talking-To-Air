"""The capabilities the harness offers, registered in a fixed order."""
from __future__ import annotations

from tta_backend.harness.capabilities import explain
from tta_backend.harness.capabilities.operations import compare, events, profile, reduce
from tta_backend.harness.capabilities.operations import group as operations_group
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.render import maps, monitor_map, profiles, tables, timeseries
from tta_backend.harness.capabilities.render import group as render_group
from tta_backend.harness.capabilities.sources import areas, monitors, satellite, smoke


def build_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for group in (
        areas.GROUP,
        satellite.GROUP,
        monitors.GROUP,
        smoke.GROUP,
        operations_group.GROUP,
        render_group.GROUP,
        explain.GROUP,
    ):
        registry.add_group(group)
    for capabilities in (
        areas.CAPABILITIES,
        satellite.CAPABILITIES,
        monitors.CAPABILITIES,
        smoke.CAPABILITIES,
        reduce.CAPABILITIES,
        compare.CAPABILITIES,
        events.CAPABILITIES,
        profile.CAPABILITIES,
        maps.CAPABILITIES,
        monitor_map.CAPABILITIES,
        timeseries.CAPABILITIES,
        profiles.CAPABILITIES,
        tables.CAPABILITIES,
        explain.CAPABILITIES,
    ):
        for capability in capabilities:
            registry.register(capability)
    return registry
