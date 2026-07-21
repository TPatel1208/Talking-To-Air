import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class EarthdataAgentPromptT07Tests(unittest.TestCase):
    def test_prompt_tells_the_agent_satellite_and_ground_are_different_quantities(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("different physical quantities", prompt.lower())

    def test_prompt_routes_validation_requests_to_the_t07_tools(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("validate_against_ground", prompt)
        self.assertIn("exceedance_overlay", prompt)


class EarthdataAgentPromptT08Tests(unittest.TestCase):
    def test_prompt_routes_comparison_requests_to_the_compare_tool(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("compare", prompt)
        self.assertIn("mode=\"region\"", prompt)
        self.assertIn("mode=\"period\"", prompt)


class EarthdataAgentPromptT09Tests(unittest.TestCase):
    def test_prompt_calls_preview_dataset_before_any_retrieval(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        preview_step = prompt.index("preview_dataset")
        retrieve_step = prompt.index("safe_retrieve")
        self.assertLess(
            preview_step, retrieve_step,
            "preview_dataset must be called before safe_retrieve, so the researcher "
            "confirms product-and-region fit before the platform commits resources",
        )

    def test_prompt_tells_the_agent_to_report_a_missing_gibs_layer_plainly(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("no browse layer", prompt.lower())


class EarthdataAgentPromptT20Tests(unittest.TestCase):
    def test_prompt_routes_a_single_locations_history_to_point_timeseries(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("point_timeseries", prompt)
        self.assertIn("point-over-time", prompt.lower())

    def test_prompt_still_routes_area_mean_trends_to_conduct_temporal_statistic(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("conduct_temporal_statistic", prompt)


class EarthdataAgentPromptAvailabilityGroundingTests(unittest.TestCase):
    """Talking-to-air fix B: the agent must never assert availability from a
    prior claim quoted back in the task — only from a this-turn coverage check."""

    def test_prompt_forbids_stating_availability_without_a_this_turn_check(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        # Collapse whitespace so line wrapping in the prompt can't hide a phrase.
        prompt = " ".join(get_earthdata_agent_prompt().lower().split())

        self.assertIn("this turn", prompt)
        self.assertIn("not evidence", prompt)
        # Availability guidance appears before the No-Data Protocol acts on it.
        full = get_earthdata_agent_prompt()
        self.assertLess(
            full.index("Availability must be tool-grounded"),
            full.index("## No-Data Protocol"),
        )


class EarthdataAgentPromptT22Tests(unittest.TestCase):
    def test_prompt_offers_the_optional_suggested_followups_envelope_key(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("suggested_followups", prompt)
        self.assertIn("otherwise omit", prompt.lower())


class EarthdataAgentPromptT25Phase4Tests(unittest.TestCase):
    """T25 Phase 4: the prompt states the satellite arm is universal over
    gridded Earthdata collections, that ground/cross-source confirmation is
    an air-quality-only capability the agent must never promise elsewhere,
    and instructs the agent to consume describe_dataset's variable metadata
    and treat structured choice-errors as questions, not failures."""

    def test_prompt_states_satellite_handles_any_gridded_collection(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt().lower()

        self.assertIn("any regularly-gridded", prompt)
        self.assertIn("not in the preset table", prompt)

    def test_prompt_names_the_out_of_scope_grid_refusals(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt().lower()

        self.assertIn("curvilinear", prompt)
        self.assertIn("sinusoidal", prompt)

    def test_prompt_never_promises_ground_confirmation_outside_air_quality(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        # Collapse whitespace so line wrapping in the prompt can't hide a phrase.
        prompt = " ".join(get_earthdata_agent_prompt().lower().split())

        self.assertIn("air-quality-only", prompt)
        self.assertIn("no ground-truth confirmation step", prompt)
        self.assertIn("never offer, promise, or imply", prompt)

    def test_prompt_names_non_aq_domains_the_asymmetry_covers(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt().lower()

        self.assertIn("soil moisture", prompt)
        self.assertIn("land surface temperature", prompt)

    def test_prompt_instructs_using_describe_dataset_variable_metadata_to_choose(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("describe_dataset", prompt)
        self.assertIn("long_name", prompt)
        self.assertIn("advisory_notes", prompt)

    def test_prompt_instructs_relaying_choice_errors_as_questions_not_failures(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()

        self.assertIn("variable_choice_required", prompt)
        self.assertIn("dimension_choice_required", prompt)
        self.assertIn("not a failure", prompt.lower())


class EarthdataAgentPromptT36Phase3Tests(unittest.TestCase):
    """T36 Phase 3: the prompt has an "Explaining measurement reliability"
    section that routes reliability questions through explain_measurement and
    holds the grounding discipline — evidence-only, evidence-vs-verdict, and an
    honest empty-evidence path that offers to retrieve the companions (the P1
    loop) rather than confabulating confidence."""

    def _prompt(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        return get_earthdata_agent_prompt()

    def test_prompt_has_the_reliability_section_and_routes_to_the_tool(self):
        prompt = self._prompt()

        self.assertIn("Explaining measurement reliability", prompt)
        self.assertIn("explain_measurement", prompt)
        # Triggered by reliability/confidence phrasing (collapse whitespace so
        # line wrapping can't hide a phrase).
        lowered = " ".join(prompt.lower().split())
        self.assertIn("how reliable", lowered)
        self.assertIn("how confident", lowered)

    def test_prompt_holds_the_evidence_only_and_evidence_vs_verdict_guardrails(self):
        # Collapse whitespace so line wrapping can't hide a phrase.
        prompt = " ".join(self._prompt().lower().split())

        # Evidence-only: never assert a factor not in the returned evidence.
        self.assertIn("only the facts in the returned", prompt)
        self.assertIn("not in the returned evidence", prompt)
        # Evidence, not verdict.
        self.assertIn("never as a categorical verdict", prompt)
        # Caveats: coverage and uncertainty are surfaced.
        self.assertIn("coverage", prompt)
        self.assertIn("uncertainty", prompt)

    def test_prompt_empty_evidence_offers_retrieval_instead_of_confabulating(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("has_evidence", prompt)
        self.assertIn("no companion evidence", prompt)
        # Offers the QA flag / cloud fraction retrieval as a suggested followup
        # (the P1 loop), and refuses to manufacture confidence.
        self.assertIn("qa flag", prompt)
        self.assertIn("cloud fraction", prompt)
        self.assertIn("suggested_followups", prompt)
        self.assertIn("do not manufacture a confidence claim", prompt)

    def test_prompt_relaxes_summary_length_for_reliability_only_but_keeps_contract(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("reliability query class only", prompt)
        self.assertIn("the json envelope contract is unchanged", prompt)


class EarthdataAgentPromptCurrentDateTests(unittest.TestCase):
    """Root cause A (live 2026-07-19): the sub-agent refused valid present-day
    L3 AOD as "in the future, no observations exist" even after being shown the
    current date. The prompt's date line dangled ("Use this as the reference"
    pointed at nothing) and never told the agent the injected date is
    authoritative or that on-or-before-today dates are not "future". The prompt
    must anchor relative dates on the authoritative current-date banner and
    forbid refusing present dates as future from the model's own prior."""

    def _prompt(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        return get_earthdata_agent_prompt()

    def test_prompt_references_the_authoritative_current_date_banner(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("current date", prompt)
        self.assertIn("authoritative", prompt)
        # The old dangling "use this as the reference" (pointing at nothing) is gone.
        self.assertNotIn("use this as the reference", prompt)

    def test_prompt_forbids_refusing_present_dates_as_in_the_future(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("in the future", prompt)
        self.assertIn("no observations exist", prompt)
        # And routes doubt about a date's data to a tool check, not a refusal.
        self.assertIn("check_availability", prompt)


class EarthdataAgentPromptNrtLatencyTests(unittest.TestCase):
    """Root cause B (live 2026-07-19): "recent AOD, last week" dead-ended on the
    standard-latency MODIS AOD preset ("no data found") because standard L3 lags
    days behind real time and the prompt had no concept of NRT products. The
    prompt must explain standard-L3 latency, tell the agent to search for a
    Near Real-Time product before declaring recent data unavailable, and report
    a partially-filled recent window honestly instead of as a failure."""

    def _prompt(self):
        from config.earthdata_agent_prompt import get_earthdata_agent_prompt

        return get_earthdata_agent_prompt()

    def test_prompt_explains_standard_l3_latency_and_names_nrt(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("latency", prompt)
        self.assertIn("near real-time", prompt)
        self.assertIn("nrt", prompt)

    def test_prompt_tells_agent_to_search_for_nrt_before_declaring_no_recent_data(self):
        prompt = " ".join(self._prompt().lower().split())

        self.assertIn("search_datasets", prompt)
        # A short rolling NRT window may only partially fill a multi-day recent
        # request — report which days returned rather than calling it a failure.
        self.assertIn("rolling window", prompt)


if __name__ == "__main__":
    unittest.main()
