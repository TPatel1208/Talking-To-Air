"""Registration-time guard: a collection cannot join the registry with its QA
story left unstated.

Measured 2026-08-02 across five real granules, the Tier-2 CF-deterministic path
has never once fired on a real NASA product:

    TEMPO NO2 / HCHO   'normal suspicious bad'
                       -> 'suspicious' is unclassifiable, so the whole parse is
                          ambiguous and no mask is applied
    OMI HCHO           'good_number_of_samples_greater_than_0.1
                        good_number_of_samples_less_than_0.1
                        bad_or_not_computed'
                       -> all three unclassifiable
    OMI SO2            no flag_values/flag_meanings attributes AT ALL, only a
                       ValidRange -- Tier 2 has nothing to parse

Real products describe their flags in phrases rather than the tidy single words
the vocabulary expects, and some publish no CF flag attributes at all. So in
practice every working mask in this system is a Tier-1 pin, and a new
collection registered without one gets no quality masking -- silently,
discovered only when someone reads a pass rate that says "not applied". This
test makes that discovery happen at registration instead.
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


class AerDbdtFamilyIsTreatedIdenticallyTests(unittest.TestCase):
    """MODIS Aqua/Terra and VIIRS SNPP/NOAA-20 are the same GES DISC product:
    the Deep Blue + Dark Target combination gridded to 0.1 degrees daily
    (AER_DBDT_D10KM_L3_*). Verified 2026-08-02 against Earthdata, all four
    publish an identical variable list down to the names.

    They therefore have to be masked identically. Divergence between siblings
    is not a preference, it is one of them being wrong -- and it would show up
    as the same scene reading differently depending on which satellite the
    agent happened to pick.
    """

    FAMILY = ("MODIS_AOD_AQUA", "MODIS_AOD_TERRA", "VIIRS_AOD_SNPP", "VIIRS_AOD_NOAA20")

    def test_every_member_of_the_family_shares_one_masking_treatment(self):
        from tta_backend.datasets.registry import load_registry

        registry = load_registry()
        missing = [k for k in self.FAMILY if k not in registry]
        self.assertEqual(missing, [], f"not registered: {missing}")

        treatments = {
            key: (
                registry[key].primary_var,
                registry[key].units,
                registry[key].fill_value,
                registry[key].valid_min,
                registry[key].valid_max,
                registry[key].quality_flag_var,
            )
            for key in self.FAMILY
        }
        self.assertEqual(
            len(set(treatments.values())), 1,
            f"AER_DBDT siblings disagree about masking: {treatments}",
        )


class AerdaVariantsShareOneAlgorithmChoiceTests(unittest.TestCase):
    """The 24-hour and 3-hour AERDA cuts are the same product at two
    aggregation windows -- granule-verified 2026-08-02 as structurally
    identical, 432 bands across 121 groups each.

    Their primary_var is a choice among 121 platform x algorithm x QA-filter
    combinations, so the two drifting apart would mean the same question
    answered by different retrieval algorithms depending only on whether the
    user asked for a day or an afternoon. Cadence is deliberately excluded from
    the comparison: that is the one thing that legitimately differs.
    """

    def test_both_cuts_resolve_to_the_same_algorithm_and_bounds(self):
        from tta_backend.datasets.registry import load_registry

        registry = load_registry()
        pair = ("AERDA_AOD_NRT", "AERDA_AOD_NRT_3H")
        missing = [k for k in pair if k not in registry]
        self.assertEqual(missing, [], f"not registered: {missing}")

        treatments = {
            key: (
                registry[key].primary_var,
                registry[key].units,
                registry[key].fill_value,
                registry[key].valid_min,
                registry[key].valid_max,
                registry[key].quality_flag_var,
            )
            for key in pair
        }
        self.assertEqual(
            len(set(treatments.values())), 1,
            f"AERDA cuts disagree on the pinned algorithm or bounds: {treatments}",
        )

    def test_the_two_cuts_are_distinguishable_by_cadence(self):
        """Registering both only helps if a period label can tell them apart."""
        from tta_backend.datasets.registry import load_registry

        registry = load_registry()
        self.assertEqual(registry["AERDA_AOD_NRT"].cadence, "daily")
        self.assertEqual(registry["AERDA_AOD_NRT_3H"].cadence, "hourly")


class Omso2eQualityRuleRestsOnGranuleEvidenceTests(unittest.TestCase):
    """OMSO2e's rule was pinned only after reading a granule (2025-06-15, one
    granule, 2026-08-02) -- the assumption it replaced was wrong in a way no
    amount of reading metadata would have caught.

    QualityFlags_SO2 was assumed to be a bit mask. It is not: ValidRange is
    [0, 2], an ordinary enumerated flag. What the granule also showed is that
    flag=0 covers 57,508 cells and the finite ColumnAmountSO2 count is 57,508
    exactly -- flag=0 is the retrieved class and flag=1 carries no data at all.

    Value 2 never appears, so qa_bad_values [2] is the minimal-intervention
    pin: it matches the sibling OMI L3 product (OMI_HCHO) and drops only the
    class never observed to carry data. Pinning qa_good_values [0] instead
    would behave identically here but would discard anything flag=1 or 2 ever
    does carry, which no evidence supports.
    """

    def test_the_rule_is_stated_as_bad_values_not_a_good_value_whitelist(self):
        from tta_backend.datasets.registry import load_registry

        cfg = load_registry().get("OMI_SO2")
        self.assertIsNotNone(cfg, "OMI_SO2 is not registered")
        self.assertEqual(cfg.quality_flag_var, "QualityFlags_SO2")
        self.assertEqual(cfg.qa_bad_values, [2])
        self.assertIsNone(
            cfg.qa_good_values,
            "a good-value whitelist would drop flag classes no granule has "
            "shown to be unusable",
        )

    def test_the_pinned_bounds_are_the_files_own_valid_range(self):
        """Taken from the granule's ValidRange attribute rather than invented:
        [-10, 2000] DU, with the -1.2676506e+30 MissingValue sentinel."""
        from tta_backend.datasets.registry import load_registry

        cfg = load_registry()["OMI_SO2"]
        self.assertEqual(cfg.valid_min, -10.0)
        self.assertEqual(cfg.valid_max, 2000.0)
        self.assertAlmostEqual(cfg.fill_value, -1.2676506002282294e30)

    def test_omso2e_does_not_clip_its_negative_tail(self):
        """45.8% of the verified granule's finite values are negative (min
        -1.06 DU) -- the largest negative share of any product measured, so a
        zero floor would have destroyed nearly half the observations. Same
        doctrine as NO2 and HCHO, even though the units are DU."""
        from tta_backend.datasets.registry import load_registry

        cfg = load_registry()["OMI_SO2"]
        self.assertLess(cfg.valid_min, 0.0)


if __name__ == "__main__":
    unittest.main()
