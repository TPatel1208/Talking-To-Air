import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class SubAgentEnvelopeTests(unittest.TestCase):
    def test_parses_a_well_formed_envelope(self):
        from models.agent_result import parse_sub_agent_envelope

        raw = '{"summary": "Found 3 monitors.", "artifact_ids": ["art_1"], "handles": ["obs_1", "obs_2"]}'

        envelope = parse_sub_agent_envelope(raw)

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.summary, "Found 3 monitors.")
        self.assertEqual(envelope.artifact_ids, ["art_1"])
        self.assertEqual(envelope.handles, ["obs_1", "obs_2"])

    def test_artifact_ids_and_handles_default_to_empty(self):
        from models.agent_result import parse_sub_agent_envelope

        envelope = parse_sub_agent_envelope('{"summary": "No data found."}')

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.artifact_ids, [])
        self.assertEqual(envelope.handles, [])

    def test_missing_summary_is_invalid(self):
        from models.agent_result import parse_sub_agent_envelope

        envelope = parse_sub_agent_envelope('{"artifact_ids": [], "handles": []}')

        self.assertIsNone(envelope)

    def test_malformed_json_is_invalid(self):
        from models.agent_result import parse_sub_agent_envelope

        envelope = parse_sub_agent_envelope("{summary: not valid json")

        self.assertIsNone(envelope)

    def test_plain_prose_is_invalid(self):
        from models.agent_result import parse_sub_agent_envelope

        envelope = parse_sub_agent_envelope("I found 3 monitors near Newark, NJ.")

        self.assertIsNone(envelope)


class AgentResultHandlesFieldTests(unittest.TestCase):
    def test_agent_result_defaults_handles_to_empty_list(self):
        from models.agent_result import AgentResult

        result = AgentResult(text="ok")

        self.assertEqual(result.handles, [])

    def test_agent_result_round_trips_handles_through_json(self):
        from models.agent_result import AgentResult, agent_result_to_json, parse_agent_result

        result = AgentResult(text="ok", handles=["obs_1"])

        parsed = parse_agent_result(agent_result_to_json(result))

        self.assertEqual(parsed.handles, ["obs_1"])


if __name__ == "__main__":
    unittest.main()
