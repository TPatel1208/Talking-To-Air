import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class SupervisorPromptT25Phase4Tests(unittest.TestCase):
    """T25 Phase 4: the supervisor prompt states the same asymmetry as the
    earthdata agent's — satellite is universal over gridded Earthdata
    collections, ground/cross-source confirmation is air-quality-only — so
    routing never promises a ground check for a non-AQ satellite query."""

    def test_prompt_states_satellite_is_not_limited_to_air_quality(self):
        from config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = SUPERVISOR_PROMPT.lower()

        self.assertIn("any regularly-gridded", prompt)
        self.assertIn("not just air quality", prompt)

    def test_prompt_states_ground_is_air_quality_pollutants_only(self):
        from config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = SUPERVISOR_PROMPT.lower()

        self.assertIn("air quality pollutants", prompt)
        self.assertIn("no2, pm2.5, o3, so2, co", prompt)

    def test_prompt_forbids_promising_ground_confirmation_for_non_aq_queries(self):
        from config.supervisor_prompt import SUPERVISOR_PROMPT

        # Collapse whitespace so line wrapping in the prompt can't hide a phrase.
        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertIn("never call ground, and never tell the user", prompt)
        self.assertIn("soil moisture", prompt)
        self.assertIn("land surface temperature", prompt)

    def test_ground_plus_satellite_routing_is_scoped_to_air_quality(self):
        from config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())
        routing_idx = prompt.index("ground + satellite: user explicitly requests")

        self.assertIn("air-quality pollutants only", prompt[routing_idx:routing_idx + 300])


if __name__ == "__main__":
    unittest.main()
