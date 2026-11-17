"""The supervisor's generated index and system prompt."""
import re
import unittest
from pathlib import Path

from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.specialists.bundles import BUNDLES
from tta_backend.harness.supervisor.index import supervisor_index
from tta_backend.harness.supervisor.prompt import supervisor_prompt
from tta_backend.harness.workflows.loader import load_packs

PACKS = Path(__file__).resolve().parent.parent / "packs"
CAPABILITIES = build_registry()
CATALOGUE = load_packs(PACKS, CAPABILITIES, load_registry())
TOOLS = ("run_workflow", "resume_run", "explore", "resolve_area", "search_datasets", "check_availability")
MAX_RULES_CHARS = 4000
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class SupervisorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = supervisor_index(CATALOGUE, CAPABILITIES)

    def test_every_runnable_workflow_appears_at_its_latest_version(self) -> None:
        runnable = 0
        for name in CATALOGUE.names():
            spec = CATALOGUE.latest(name)
            if spec.not_runnable_reason is None:
                runnable += 1
                self.assertIn(f"- {name} [{spec.status}]: {spec.description}", self.index)
                for input_name in spec.inputs:
                    self.assertIn(f"{input_name} (", self.index)
            else:
                self.assertNotIn(f"- {name} [", self.index)
        self.assertGreater(runnable, 0)

    def test_every_bundle_and_used_group_appears(self) -> None:
        for name, bundle in BUNDLES.items():
            self.assertIn(f"- {name}: {bundle.description}", self.index)
            for group in bundle.groups(CAPABILITIES):
                self.assertIn(f"- {group.name}: {group.description}", self.index)

    def test_no_capability_names_are_listed(self) -> None:
        for group in CAPABILITIES.groups():
            for capability in CAPABILITIES.in_group(group.name):
                self.assertNotIn(capability.name, self.index)

    def test_same_inputs_give_identical_text(self) -> None:
        self.assertEqual(self.index, supervisor_index(CATALOGUE, build_registry()))

    def test_no_handle_ids_or_dates(self) -> None:
        self.assertIsNone(DATE.search(self.index))
        self.assertIsNone(re.search(r"\b(?:h|handle)[_-][0-9a-f]{6,}", self.index))


    def test_input_formats_name_every_registry_quantity_and_the_date_shapes(self) -> None:
        from tta_backend.harness.capabilities.sources.satellite import QUANTITIES

        formats = self.index[self.index.index("Input formats (run_workflow):"):self.index.index("Exploration bundles")]
        for quantity in QUANTITIES:
            self.assertIn(quantity, formats)
        self.assertIn('{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}', formats)
        self.assertIn("place exactly as the user wrote it", formats)
        used = {declared.type for name in CATALOGUE.names() for declared in CATALOGUE.latest(name).inputs.values()}
        for kind in used:
            self.assertIn(f"- {kind}: ", formats)

class SupervisorPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = supervisor_index(CATALOGUE, CAPABILITIES)
        self.prompt = supervisor_prompt(self.index)

    def test_contains_the_index(self) -> None:
        self.assertIn(self.index, self.prompt)

    def test_names_the_six_tools(self) -> None:
        for tool in TOOLS:
            self.assertIn(tool, self.prompt)

    def test_has_no_old_procedure_terms(self) -> None:
        lowered = self.prompt.lower()
        for term in ("dispatch", "intent", "satellite agent", "ground agent"):
            self.assertNotIn(term, lowered)

    def test_rules_stay_short(self) -> None:
        self.assertLess(len(self.prompt) - len(self.index), MAX_RULES_CHARS)

    def test_prompt_is_stable_and_undated(self) -> None:
        self.assertEqual(self.prompt, supervisor_prompt(self.index))
        self.assertIsNone(DATE.search(self.prompt))


if __name__ == "__main__":
    unittest.main()
