import unittest


class GroundSensorAgentPromptT22Tests(unittest.TestCase):
    def test_prompt_offers_the_optional_suggested_followups_envelope_key(self):
        from tta_backend.config.ground_sensor_agent_prompt import GROUND_SYSTEM_PROMPT

        self.assertIn("suggested_followups", GROUND_SYSTEM_PROMPT)
        self.assertIn("otherwise omit", GROUND_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
