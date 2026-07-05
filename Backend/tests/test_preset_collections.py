import unittest


class PresetCollectionsTests(unittest.TestCase):
    def test_preset_collections_is_a_small_suggestion_list_not_an_exhaustive_registry(self):
        from datasets.preset_collections import PRESET_COLLECTIONS

        self.assertGreater(len(PRESET_COLLECTIONS), 0)
        self.assertLess(len(PRESET_COLLECTIONS), 15)
        for entry in PRESET_COLLECTIONS:
            self.assertIn("short_name", entry)
            self.assertIn("description", entry)


if __name__ == "__main__":
    unittest.main()
