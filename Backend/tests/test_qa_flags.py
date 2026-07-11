import unittest


class ParseFlagMeaningsTests(unittest.TestCase):
    def test_unambiguous_tokens_split_into_good_and_bad(self):
        from datasets.qa_flags import parse_flag_meanings

        parsed = parse_flag_meanings([0, 1, 2], ["good_quality", "bad_quality", "missing"])

        self.assertTrue(parsed.available)
        self.assertTrue(parsed.unambiguous)
        self.assertEqual(parsed.good_values, [0])
        self.assertEqual(parsed.bad_values, [1, 2])
        self.assertEqual(parsed.ambiguous_tokens, [])

    def test_ambiguous_token_outside_vocabulary_is_flagged_not_guessed(self):
        from datasets.qa_flags import parse_flag_meanings

        parsed = parse_flag_meanings(
            [0, 1, 2], ["good_quality", "partially_cloudy_usable", "missing"],
        )

        self.assertTrue(parsed.available)
        self.assertFalse(parsed.unambiguous)
        self.assertEqual(parsed.good_values, [0])
        self.assertEqual(parsed.bad_values, [2])
        self.assertEqual(parsed.ambiguous_tokens, ["partially_cloudy_usable"])
        self.assertEqual(parsed.ambiguous_values, [1])

    def test_space_separated_string_attrs_are_coerced(self):
        from datasets.qa_flags import parse_flag_meanings

        parsed = parse_flag_meanings("0 1 2", "good_quality bad_quality missing")

        self.assertTrue(parsed.unambiguous)
        self.assertEqual(parsed.good_values, [0])

    def test_missing_or_mismatched_attrs_are_not_available(self):
        from datasets.qa_flags import parse_flag_meanings

        self.assertFalse(parse_flag_meanings(None, None).available)
        self.assertFalse(parse_flag_meanings([0, 1], None).available)
        self.assertFalse(parse_flag_meanings([0, 1], ["good_quality"]).available)  # length mismatch


class ResolveQaInfoTierTests(unittest.TestCase):
    """T25 Phase 3: pinned yaml rule -> CF flag_meanings (deterministic or
    agent-inferred) -> no mask, one tier per test."""

    def test_tier1_pinned_good_values_wins_even_with_cf_attrs_present(self):
        from datasets.qa_flags import QA_VERIFIED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={"qa_good_values": [0]},
            flag_attrs={"flag_values": [0, 1], "flag_meanings": "good_quality bad_quality"},
        )

        self.assertEqual(qa_col_info, {"qa_good_values": [0]})
        self.assertEqual(provenance["qa_status"], QA_VERIFIED)
        self.assertEqual(provenance["qa_source"], "collections_yaml")

    def test_tier1_pinned_bad_values_used_when_good_not_set(self):
        from datasets.qa_flags import QA_VERIFIED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(yaml_info={"qa_bad_values": [2]}, flag_attrs={})

        self.assertEqual(qa_col_info, {"qa_bad_values": [2]})
        self.assertEqual(provenance["qa_status"], QA_VERIFIED)

    def test_tier2_unambiguous_cf_flags_apply_deterministically_no_model(self):
        from datasets.qa_flags import QA_CF_DETERMINISTIC, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={},
            flag_attrs={"flag_values": [0, 1, 2], "flag_meanings": "good_quality bad_quality missing"},
        )

        self.assertEqual(qa_col_info, {"qa_good_values": [0]})
        self.assertEqual(provenance["qa_status"], QA_CF_DETERMINISTIC)
        self.assertEqual(provenance["qa_source"], "cf_flag_meanings")
        self.assertEqual(provenance["qa_good_values"], [0])
        self.assertEqual(provenance["qa_bad_values"], [1, 2])

    def test_tier2_ambiguous_tokens_with_agent_proposal_are_applied_and_tagged_inferred(self):
        from datasets.qa_flags import QA_INFERRED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={},
            flag_attrs={
                "flag_values": [0, 1, 2],
                "flag_meanings": "good_quality partially_cloudy_usable missing",
            },
            proposed_good_tokens=["partially_cloudy_usable"],
        )

        self.assertEqual(sorted(qa_col_info["qa_good_values"]), [0, 1])
        self.assertEqual(provenance["qa_status"], QA_INFERRED)
        self.assertEqual(provenance["qa_ambiguous_tokens"], ["partially_cloudy_usable"])
        self.assertEqual(provenance["qa_inferred_tokens"], ["partially_cloudy_usable"])

    def test_tier2_ambiguous_tokens_without_proposal_do_not_guess(self):
        from datasets.qa_flags import QA_AMBIGUOUS_PENDING, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={},
            flag_attrs={
                "flag_values": [0, 1, 2],
                "flag_meanings": "good_quality partially_cloudy_usable missing",
            },
        )

        self.assertEqual(qa_col_info, {})
        self.assertEqual(provenance["qa_status"], QA_AMBIGUOUS_PENDING)
        self.assertEqual(provenance["qa_ambiguous_tokens"], ["partially_cloudy_usable"])

    def test_tier3_no_pinned_rule_and_no_cf_flags_discloses_not_applied(self):
        from datasets.qa_flags import QA_NOT_APPLIED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(yaml_info={}, flag_attrs={})

        self.assertEqual(qa_col_info, {})
        self.assertEqual(provenance["qa_status"], QA_NOT_APPLIED)
        self.assertEqual(provenance["qa_source"], "none")


class EmptyGoodSetTests(unittest.TestCase):
    """Follow-up review #1: an empty good-set must never become a keep-mask on
    ``isin([])`` -- an all-False mask that wipes the whole variable to NaN.
    Treat it as no usable good classification (go ambiguous / do not mask),
    never as "mask everything"."""

    def test_parse_all_bad_tokens_is_unambiguous_but_yields_empty_good(self):
        from datasets.qa_flags import parse_flag_meanings

        parsed = parse_flag_meanings([1, 2], ["missing", "cloudy"])

        self.assertTrue(parsed.available)
        self.assertTrue(parsed.unambiguous)  # nothing ambiguous...
        self.assertEqual(parsed.good_values, [])  # ...but no good class either
        self.assertEqual(parsed.bad_values, [1, 2])

    def test_cf_all_bad_tokens_go_ambiguous_instead_of_masking_everything(self):
        from datasets.qa_flags import QA_AMBIGUOUS_PENDING, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={},
            flag_attrs={"flag_values": [1, 2], "flag_meanings": "missing cloudy"},
        )

        # No qa_good_values emitted -> apply_quality_mask keys no isin([]) mask.
        self.assertEqual(qa_col_info, {})
        self.assertNotIn("qa_good_values", qa_col_info)
        self.assertEqual(provenance["qa_status"], QA_AMBIGUOUS_PENDING)
        self.assertEqual(provenance["qa_bad_values"], [1, 2])

    def test_pinned_empty_good_values_does_not_verify_a_wipeout(self):
        from datasets.qa_flags import QA_NOT_APPLIED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(yaml_info={"qa_good_values": []}, flag_attrs={})

        self.assertEqual(qa_col_info, {})
        self.assertEqual(provenance["qa_status"], QA_NOT_APPLIED)

    def test_pinned_empty_good_falls_through_to_cf_flag_meanings(self):
        from datasets.qa_flags import QA_CF_DETERMINISTIC, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={"qa_good_values": []},
            flag_attrs={"flag_values": [0, 1], "flag_meanings": "good_quality bad_quality"},
        )

        self.assertEqual(qa_col_info, {"qa_good_values": [0]})
        self.assertEqual(provenance["qa_status"], QA_CF_DETERMINISTIC)

    def test_pinned_empty_bad_falls_back_to_good_rule_rather_than_noop_verified(self):
        from datasets.qa_flags import QA_VERIFIED, resolve_qa_info

        qa_col_info, provenance = resolve_qa_info(
            yaml_info={"qa_good_values": [0], "qa_bad_values": []}, flag_attrs={},
        )

        self.assertEqual(qa_col_info, {"qa_good_values": [0]})
        self.assertEqual(provenance["qa_status"], QA_VERIFIED)


if __name__ == "__main__":
    unittest.main()
