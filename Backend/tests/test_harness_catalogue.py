"""The catalogue registers every capability and group the package declares."""
import importlib
import pkgutil
import unittest
from typing import Any

import tta_backend.harness.capabilities as package
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.registry import Group


def declared() -> tuple[list[Capability[Any]], list[Group]]:
    """Every CAPABILITIES tuple and GROUP found by importing each module of the package."""
    capabilities: list[Capability[Any]] = []
    groups: list[Group] = []
    for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        module = importlib.import_module(info.name)
        capabilities.extend(getattr(module, "CAPABILITIES", ()))
        group = getattr(module, "GROUP", None)
        if isinstance(group, Group):
            groups.append(group)
    return capabilities, groups


class CatalogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()
        self.capabilities, self.groups = declared()

    def registered(self) -> list[Capability[Any]]:
        return [c for g in self.registry.groups() for c in self.registry.in_group(g.name)]

    def test_the_package_declares_capabilities(self):
        self.assertGreater(len(self.capabilities), 20)

    def test_every_declared_capability_is_registered(self):
        for capability in self.capabilities:
            self.assertIs(self.registry.get(capability.name), capability)
        self.assertEqual(len(self.registered()), len(self.capabilities))

    def test_every_declared_group_is_registered(self):
        for group in self.groups:
            self.assertIs(self.registry.group(group.name), group)

    def test_names_are_unique_and_each_belongs_to_a_registered_group(self):
        names = [c.name for c in self.registered()]
        self.assertEqual(len(names), len(set(names)))
        known = {g.name for g in self.registry.groups()}
        for capability in self.registered():
            self.assertIn(capability.group, known)

    def test_every_capability_exports_a_parameter_schema(self):
        for capability in self.registered():
            schema = self.registry.schema(capability.name)
            self.assertEqual(schema.get("type"), "object", capability.name)
            self.assertIn("properties", schema, capability.name)
            for param in capability.handle_params:
                self.assertIn("x-handle-kinds", schema["properties"][param], capability.name)

    def test_every_capability_is_described_labelled_and_has_a_method(self):
        for capability in self.registered():
            self.assertTrue(capability.description.strip(), capability.name)
            self.assertTrue(capability.label.strip(), capability.name)
            self.assertTrue(capability.method.strip(), capability.name)

    def test_the_index_has_one_line_per_group(self):
        lines = self.registry.index().splitlines()
        groups = self.registry.groups()
        self.assertEqual(len(lines), len(groups))
        for line, group in zip(lines, groups, strict=True):
            self.assertTrue(line.startswith(f"{group.name}: "))

    def test_every_group_has_usage_notes(self):
        for group in self.registry.groups():
            self.assertTrue(group.notes.strip(), group.name)

    def test_every_group_holds_a_capability(self):
        for group in self.registry.groups():
            self.assertTrue(self.registry.in_group(group.name), group.name)

    def test_each_call_builds_an_independent_registry(self):
        self.assertIsNot(build_registry(), build_registry())
