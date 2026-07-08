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


class SubAgentEnvelopeSuggestedFollowupsTests(unittest.TestCase):
    """T22: an optional, additive field — offering none is always legitimate
    (story #7); the field must never be required, and prose that exceeds the
    two-suggestion cap is a malformed envelope (salvage territory), not a
    silently truncated list."""

    def test_suggested_followups_defaults_to_none_when_absent(self):
        from models.agent_result import parse_sub_agent_envelope

        envelope = parse_sub_agent_envelope('{"summary": "Found 3 monitors."}')

        self.assertIsNotNone(envelope)
        self.assertIsNone(envelope.suggested_followups)

    def test_suggested_followups_parses_when_present(self):
        from models.agent_result import parse_sub_agent_envelope

        raw = (
            '{"summary": "Found 3 monitors.", '
            '"suggested_followups": ["What about last month?", "Any exceedances nearby?"]}'
        )

        envelope = parse_sub_agent_envelope(raw)

        self.assertEqual(
            envelope.suggested_followups,
            ["What about last month?", "Any exceedances nearby?"],
        )

    def test_more_than_two_suggestions_is_an_invalid_envelope(self):
        from models.agent_result import parse_sub_agent_envelope

        raw = (
            '{"summary": "Found 3 monitors.", '
            '"suggested_followups": ["one?", "two?", "three?"]}'
        )

        self.assertIsNone(parse_sub_agent_envelope(raw))


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

    def test_agent_result_without_suggestions_omits_the_field_from_json(self):
        """T22: 'missing suggestions' means the key is absent from the wire
        payload, not present-and-null."""
        from models.agent_result import AgentResult, agent_result_to_json

        raw = agent_result_to_json(AgentResult(text="ok"))

        self.assertNotIn("suggested_followups", raw)

    def test_agent_result_round_trips_suggestions_through_json(self):
        from models.agent_result import AgentResult, agent_result_to_json, parse_agent_result

        result = AgentResult(text="ok", suggested_followups=["What about last month?"])

        parsed = parse_agent_result(agent_result_to_json(result))

        self.assertEqual(parsed.suggested_followups, ["What about last month?"])


if __name__ == "__main__":
    unittest.main()
