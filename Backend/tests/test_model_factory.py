import unittest


class ModelFactoryTests(unittest.TestCase):
    def test_groq_provider_yields_a_model_bound_to_the_groq_api(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings
        from langchain_groq import ChatGroq

        settings = Settings(groq_api_key="groq-secret")
        model = build_chat_model("groq", "openai/gpt-oss-120b", settings)

        self.assertIsInstance(model, ChatGroq)
        self.assertEqual(model.model_name, "openai/gpt-oss-120b")
        self.assertEqual(model.groq_api_key.get_secret_value(), "groq-secret")

    def test_google_provider_yields_a_model_bound_to_gemini(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = Settings(google_api_key="google-secret")
        model = build_chat_model("google", "gemini-2.5-flash", settings)

        self.assertIsInstance(model, ChatGoogleGenerativeAI)
        self.assertEqual(model.model, "gemini-2.5-flash")
        self.assertEqual(model.google_api_key.get_secret_value(), "google-secret")

    def test_unknown_provider_raises_configuration_error_at_construction(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import ConfigurationError, Settings

        settings = Settings()
        with self.assertRaisesRegex(ConfigurationError, "openai"):
            build_chat_model("openai", "gpt-4", settings)

    def test_structured_output_hook_delegates_to_the_model(self):
        from tta_backend.config.model_factory import structured_output

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
        from tta_backend.config.model_factory import structured_output
        from tta_backend.models import SubAgentEnvelope

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

    def test_suggested_followups_is_optional_in_the_schema_fed_to_structured_output(self):
        """T22: the follow-ups field is additive to T15's single schema
        source — proving it's optional here (not required) is what makes it
        safe to feed to both providers' with_structured_output without
        reopening either provider's call site. Hermetic — no live provider
        call, per the Testing Decisions' factory-seam extension."""
        from tta_backend.config.model_factory import structured_output
        from tta_backend.models import SubAgentEnvelope

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def with_structured_output(self, schema):
                self.calls.append(schema)
                return "bound-model"

        fake = _FakeModel()
        result = structured_output(fake, SubAgentEnvelope)

        self.assertEqual(result, "bound-model")
        bound_schema = fake.calls[0]
        field = bound_schema.model_fields["suggested_followups"]
        self.assertFalse(field.is_required())
        self.assertIsNone(field.default)

        # The same schema instance round-trips through the seam whether it
        # carries suggestions or not — both providers see one JSON schema.
        with_suggestions = bound_schema(summary="ok", suggested_followups=["Next question?"])
        without_suggestions = bound_schema(summary="ok")
        self.assertEqual(with_suggestions.suggested_followups, ["Next question?"])
        self.assertIsNone(without_suggestions.suggested_followups)


class RetryCeilingTests(unittest.TestCase):
    """Verify that LLM providers use the application's retry ceiling.

    Provider SDKs may retry failed requests multiple times by default. These
    retries happen inside a single LLM call, so backoff time can significantly
    increase request latency without appearing as separate LLM calls, and the
    input prompt is billed again on every attempt.

    The retry limit is configured when models are created rather than at
    individual call sites. This ensures every agent, including future agents,
    uses the same retry policy.
    """

    def test_groq_model_caps_retries_at_the_configured_ceiling(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings

        settings = Settings(groq_api_key="groq-secret", llm_max_retries=3)
        model = build_chat_model("groq", "openai/gpt-oss-120b", settings)

        self.assertEqual(model.max_retries, 3)

    def test_google_model_caps_retries_at_the_configured_ceiling(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings

        settings = Settings(google_api_key="google-secret", llm_max_retries=3)
        model = build_chat_model("google", "gemini-2.5-flash", settings)

        self.assertEqual(model.max_retries, 3)

    def test_the_default_ceiling_is_lower_than_the_sdk_default_of_six(self):
        """Pin the value, not just that it is passed through: raising it back
        to the SDK default should fail here."""
        from tta_backend.config.settings import Settings

        self.assertEqual(Settings().llm_max_retries, 2)

    def test_the_ceiling_is_environment_overridable(self):
        import os
        from unittest.mock import patch

        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "4"}):
            self.assertEqual(Settings().llm_max_retries, 4)

    def test_a_negative_ceiling_floors_at_zero(self):
        """Neither SDK accepts a negative retry count, so clamp here rather
        than let a misconfigured environment fail at the first call."""
        import os
        from unittest.mock import patch

        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "-1"}):
            self.assertEqual(Settings().llm_max_retries, 0)


if __name__ == "__main__":
    unittest.main()
