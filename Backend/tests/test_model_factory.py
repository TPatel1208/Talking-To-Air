import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class ModelFactoryTests(unittest.TestCase):
    def test_groq_provider_yields_a_model_bound_to_the_groq_api(self):
        from config.model_factory import build_chat_model
        from config.settings import Settings
        from langchain_groq import ChatGroq

        settings = Settings(groq_api_key="groq-secret")
        model = build_chat_model("groq", "openai/gpt-oss-120b", settings)

        self.assertIsInstance(model, ChatGroq)
        self.assertEqual(model.model_name, "openai/gpt-oss-120b")
        self.assertEqual(model.groq_api_key.get_secret_value(), "groq-secret")

    def test_google_provider_yields_a_model_bound_to_gemini(self):
        from config.model_factory import build_chat_model
        from config.settings import Settings
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = Settings(google_api_key="google-secret")
        model = build_chat_model("google", "gemini-2.5-flash", settings)

        self.assertIsInstance(model, ChatGoogleGenerativeAI)
        self.assertEqual(model.model, "gemini-2.5-flash")
        self.assertEqual(model.google_api_key.get_secret_value(), "google-secret")

    def test_unknown_provider_raises_configuration_error_at_construction(self):
        from config.model_factory import build_chat_model
        from config.settings import ConfigurationError, Settings

        settings = Settings()
        with self.assertRaisesRegex(ConfigurationError, "openai"):
            build_chat_model("openai", "gpt-4", settings)

    def test_structured_output_hook_delegates_to_the_model(self):
        from config.model_factory import structured_output

        class _Schema:
            pass

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def with_structured_output(self, schema):
                self.calls.append(schema)
                return "bound-model"

        fake = _FakeModel()
        result = structured_output(fake, _Schema)

        self.assertEqual(result, "bound-model")
        self.assertEqual(fake.calls, [_Schema])

    def test_structured_output_hook_carries_the_sub_agent_envelope_schema(self):
        """T15: the retry demotion's single re-prompt routes through this
        same seam with the real SubAgentEnvelope schema (asserted here
        hermetically — no live provider call)."""
        from config.model_factory import structured_output
        from models import SubAgentEnvelope

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def with_structured_output(self, schema):
                self.calls.append(schema)
                return "bound-model"

        fake = _FakeModel()
        result = structured_output(fake, SubAgentEnvelope)

        self.assertEqual(result, "bound-model")
        self.assertEqual(fake.calls, [SubAgentEnvelope])


if __name__ == "__main__":
    unittest.main()
