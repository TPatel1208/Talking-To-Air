import re
import unittest


class PresetCollectionsTests(unittest.TestCase):
    def test_preset_collections_is_a_small_suggestion_list_not_an_exhaustive_registry(self):
        from datasets.preset_collections import PRESET_COLLECTIONS

        self.assertGreater(len(PRESET_COLLECTIONS), 0)
        self.assertLess(len(PRESET_COLLECTIONS), 15)
        for entry in PRESET_COLLECTIONS:
            self.assertIn("short_name", entry)
            self.assertIn("description", entry)

    def test_every_preset_is_grounded_in_the_registry_by_concept_id(self):
        """Regression guard for the AOD misrouting bug (2026-07-11): the
        preset labels the agent was told to search with ('MODIS_AOD_TERRA',
        'OMI_NO2', ...) were synthetic strings that returned ZERO search
        results, so the agent free-ranged onto unsupported products. Every
        preset must now carry a real CMR concept_id and short_name that match
        a registered collection exactly — the identifiers are pulled from the
        registry, so a preset can never again point at a resolve-to-nothing
        label."""
        from datasets.preset_collections import PRESET_COLLECTIONS
        from datasets.registry import load_registry

        registry = load_registry()
        by_concept_id = {cfg.collection_id: cfg for cfg in registry.values()}

        for entry in PRESET_COLLECTIONS:
            concept_id = entry.get("concept_id")
            self.assertIsNotNone(concept_id, f"preset missing concept_id: {entry}")
            # CMR concept-id shape, e.g. C3618500076-GES_DISC — the query key
            # that resolves to exactly one collection.
            self.assertRegex(
                concept_id, r"^C\d+-[A-Z0-9_]+$",
                f"preset concept_id is not a CMR concept-id: {concept_id!r}",
            )
            self.assertIn(
                concept_id, by_concept_id,
                f"preset concept_id {concept_id!r} matches no registered collection",
            )
            # short_name must be the registered collection's real short_name,
            # not a human label (the old bug), so the table never misleads.
            self.assertEqual(
                entry.get("short_name"), by_concept_id[concept_id].short_name,
                f"preset short_name disagrees with the registry for {concept_id}",
            )

    def test_the_prompt_table_instructs_search_by_concept_id(self):
        """The prompt must actually surface the concept_ids and tell the agent
        to search by them — the behavioral half of the fix."""
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt
        from datasets.preset_collections import PRESET_COLLECTIONS

        prompt = get_earthdata_agent_prompt()
        self.assertIn("concept_id", prompt)
        for entry in PRESET_COLLECTIONS:
            self.assertIn(entry["concept_id"], prompt)


if __name__ == "__main__":
    unittest.main()
