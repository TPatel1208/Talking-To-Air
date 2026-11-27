import unittest


class ModelFactoryTests(unittest.TestCase):
    def test_groq_provider_yields_a_model_bound_to_the_groq_api(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.infrastructure.settings import Settings
        from langchain_groq import ChatGroq

        settings = Settings(groq_api_key="groq-secret")
        model = build_chat_model("groq", "openai/gpt-oss-120b", settings)

        self.assertIsInstance(model, ChatGroq)
        self.assertEqual(model.model_name, "openai/gpt-oss-120b")
        self.assertEqual(model.groq_api_key.get_secret_value(), "groq-secret")

    def test_google_provider_yields_a_model_bound_to_gemini(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.infrastructure.settings import Settings
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = Settings(google_api_key="google-secret")
        model = build_chat_model("google", "gemini-2.5-flash", settings)

        self.assertIsInstance(model, ChatGoogleGenerativeAI)
        self.assertEqual(model.model, "gemini-2.5-flash")
        self.assertEqual(model.google_api_key.get_secret_value(), "google-secret")

    def test_unknown_provider_raises_configuration_error_at_construction(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.infrastructure.settings import ConfigurationError, Settings

        settings = Settings()
        with self.assertRaisesRegex(ConfigurationError, "openai"):
            build_chat_model("openai", "gpt-4", settings)


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
        from tta_backend.infrastructure.settings import Settings

        settings = Settings(groq_api_key="groq-secret", llm_max_retries=3)
        model = build_chat_model("groq", "openai/gpt-oss-120b", settings)

        self.assertEqual(model.max_retries, 3)

    def test_google_model_caps_retries_at_the_configured_ceiling(self):
        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.infrastructure.settings import Settings

        settings = Settings(google_api_key="google-secret", llm_max_retries=3)
        model = build_chat_model("google", "gemini-2.5-flash", settings)

        self.assertEqual(model.max_retries, 3)

    def test_the_default_ceiling_is_lower_than_the_sdk_default_of_six(self):
        """Pin the value, not just that it is passed through: raising it back
        to the SDK default should fail here."""
        from tta_backend.infrastructure.settings import Settings

        self.assertEqual(Settings().llm_max_retries, 2)

    def test_the_ceiling_is_environment_overridable(self):
        import os
        from unittest.mock import patch

        from tta_backend.infrastructure.settings import Settings

        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "4"}):
            self.assertEqual(Settings().llm_max_retries, 4)

    def test_a_negative_ceiling_floors_at_zero(self):
        """Neither SDK accepts a negative retry count, so clamp here rather
        than let a misconfigured environment fail at the first call."""
        import os
        from unittest.mock import patch

        from tta_backend.infrastructure.settings import Settings

        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "-1"}):
            self.assertEqual(Settings().llm_max_retries, 0)


class PortkeyProviderTests(unittest.TestCase):
    def _settings(self, **overrides):
        from tta_backend.infrastructure.settings import Settings

        return Settings(portkey_api_key="pk-offline", **overrides)

    def test_requests_go_to_the_configured_gateway_with_portkey_headers(self):
        from tta_backend.config.model_factory import build_chat_model

        model = build_chat_model(
            "portkey", "@slug/model", self._settings(portkey_base_url="http://portkey-gateway:8787/v1"),
            agent_type="satellite",
        )

        self.assertEqual(model.openai_api_base, "http://portkey-gateway:8787/v1")
        self.assertEqual(model.default_headers["x-portkey-api-key"], "pk-offline")
        self.assertEqual(model.default_headers["x-portkey-strict-open-ai-compliance"], "false")
        self.assertIn('"agent_type": "satellite"', model.default_headers["x-portkey-metadata"])
        self.assertNotIn("x-portkey-config", model.default_headers)

    def test_the_gateway_defaults_to_portkeys_hosted_one(self):
        import os
        from unittest.mock import patch

        from tta_backend.infrastructure.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings().portkey_base_url, "https://api.portkey.ai/v1")


if __name__ == "__main__":
    unittest.main()
