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


if __name__ == "__main__":
    unittest.main()
