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


if __name__ == "__main__":
    unittest.main()
