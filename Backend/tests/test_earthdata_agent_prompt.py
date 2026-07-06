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


if __name__ == "__main__":
    unittest.main()
