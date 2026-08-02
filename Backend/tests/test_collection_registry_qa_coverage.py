"""Registration-time guard: a collection cannot join the registry with its QA
story left unstated.

Measured 2026-08-02 across four real granules, the Tier-2 CF-deterministic path
has never once fired on a real NASA product. Every ``flag_meanings`` vocabulary
encountered so far falls outside GOOD_TOKENS/BAD_TOKENS:

    TEMPO NO2 / HCHO   'normal suspicious bad'
                       -> 'suspicious' is unclassifiable, so the whole parse is
                          ambiguous and no mask is applied
    OMI HCHO           'good_number_of_samples_greater_than_0.1
                        good_number_of_samples_less_than_0.1
                        bad_or_not_computed'
                       -> all three unclassifiable

Real products describe their flags in phrases, not in the tidy single words the
vocabulary expects. So in practice every working mask in this system is a
Tier-1 pin, and a new collection registered without one gets no quality masking
at all -- silently, discovered only when someone reads a pass rate that says
"not applied". This test makes that discovery happen at registration instead.
"""
import unittest


class EveryCollectionStatesItsQAStoryTests(unittest.TestCase):
    def test_a_collection_either_pins_a_quality_rule_or_documents_having_no_flag(self):
        from tta_backend.datasets.registry import (
            COLLECTIONS_WITHOUT_QUALITY_FLAG,
            load_registry,
        )

        unaccounted = sorted(
            key for key, cfg in load_registry().items()
            if not cfg.quality_flag_var and key not in COLLECTIONS_WITHOUT_QUALITY_FLAG
        )

        self.assertEqual(
            unaccounted, [],
            "These collections pin no quality_flag_var and are not recorded as "
            "publishing none, so they will mask nothing without saying why. "
            "Verify with `python Backend/scripts/probe_tempo_qa.py <granule> "
            "<KEY>`: if the granule carries a flag band, pin it in "
            "collections.yaml; if it genuinely has none, record it in "
            f"COLLECTIONS_WITHOUT_QUALITY_FLAG with the evidence. {unaccounted}",
        )

    def test_the_no_flag_record_carries_evidence_rather_than_a_bare_name(self):
        """A bare set would let "I didn't look" and "I checked a granule" be
        written identically. Each entry states how it was established."""
        from tta_backend.datasets.registry import COLLECTIONS_WITHOUT_QUALITY_FLAG

        for key, reason in COLLECTIONS_WITHOUT_QUALITY_FLAG.items():
            self.assertTrue(
                reason and len(reason) > 20,
                f"{key} needs a reason describing how this was established",
            )

    def test_the_no_flag_record_does_not_contradict_a_pinned_collection(self):
        from tta_backend.datasets.registry import (
            COLLECTIONS_WITHOUT_QUALITY_FLAG,
            load_registry,
        )

        registry = load_registry()
        contradictions = sorted(
            key for key in COLLECTIONS_WITHOUT_QUALITY_FLAG
            if key in registry and registry[key].quality_flag_var
        )

        self.assertEqual(
            contradictions, [],
            f"recorded as having no quality flag, yet pin one: {contradictions}",
        )


if __name__ == "__main__":
    unittest.main()
