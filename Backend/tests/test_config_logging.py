import json
import logging
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class ConfigLoggingTests(unittest.TestCase):
    def setUp(self):
        from config import settings

        settings.get_settings.cache_clear()

    def tearDown(self):
        from config import settings

        settings.get_settings.cache_clear()

    def test_settings_loads_defaults_and_validates_required_startup_values(self):
        from config.settings import Settings, get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.llm_model, "gemini-2.5-flash")
        self.assertEqual(loaded.ground_agent_model, "openai/gpt-oss-20b")
        self.assertEqual(loaded.data_fetch_mode, "auto")
        self.assertEqual(loaded.harmony_processing_timeout_seconds, 600)
        loaded = Settings(db_password=None, google_api_key=None)
        with self.assertRaisesRegex(RuntimeError, "DB_PASSWORD, GOOGLE_API_KEY"):
            loaded.validate_startup()

    def test_settings_loads_harmony_processing_timeout(self):
        from config.settings import get_settings

        with patch.dict(os.environ, {"HARMONY_PROCESSING_TIMEOUT_SECONDS": "15"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.harmony_processing_timeout_seconds, 15)

    def test_settings_loads_earthdata_mcp_defaults_and_overrides(self):
        from config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.earthdata_mcp_url, "http://mcp:8765/mcp")
        self.assertIsNone(loaded.earthdata_mcp_token)

        with patch.dict(
            os.environ,
            {"EARTHDATA_MCP_URL": "http://mcp:9000/mcp", "EARTHDATA_MCP_TOKEN": "secret"},
            clear=True,
        ):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.earthdata_mcp_url, "http://mcp:9000/mcp")
        self.assertEqual(loaded.earthdata_mcp_token, "secret")

    def test_settings_loads_retrieval_gate_defaults_and_overrides(self):
        from config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.retrieval_soft_cap_bytes, 2 * 1024 ** 3)
        self.assertEqual(loaded.retrieval_hard_cap_bytes, 10 * 1024 ** 3)
        self.assertEqual(loaded.await_retrieval_poll_min_seconds, 2)
        self.assertEqual(loaded.await_retrieval_poll_max_seconds, 15)
        self.assertEqual(loaded.await_retrieval_timeout_seconds, 900)

        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_SOFT_CAP_BYTES": "1000",
                "RETRIEVAL_HARD_CAP_BYTES": "5000",
                "AWAIT_RETRIEVAL_POLL_MIN_SECONDS": "1",
                "AWAIT_RETRIEVAL_POLL_MAX_SECONDS": "20",
                "AWAIT_RETRIEVAL_TIMEOUT_SECONDS": "60",
            },
            clear=True,
        ):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.retrieval_soft_cap_bytes, 1000)
        self.assertEqual(loaded.retrieval_hard_cap_bytes, 5000)
        self.assertEqual(loaded.await_retrieval_poll_min_seconds, 1)
        self.assertEqual(loaded.await_retrieval_poll_max_seconds, 20)
        self.assertEqual(loaded.await_retrieval_timeout_seconds, 60)

    def test_settings_loads_subagent_trim_token_ceiling_default_and_override(self):
        from config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.subagent_trim_token_ceiling, 20000)

        with patch.dict(os.environ, {"SUBAGENT_TRIM_TOKEN_CEILING": "4000"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.subagent_trim_token_ceiling, 4000)

    def test_settings_loads_earthdata_agent_model_default_and_override(self):
        # Settings() constructed directly (not via get_settings()) so a
        # developer's local .env can't shadow the default being asserted here.
        from config.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            loaded = Settings()
        self.assertEqual(loaded.earthdata_agent_model, "openai/gpt-oss-120b")

        with patch.dict(os.environ, {"EARTHDATA_AGENT_MODEL": "some/other-model"}, clear=True):
            loaded = Settings()
        self.assertEqual(loaded.earthdata_agent_model, "some/other-model")

    def test_settings_earthdata_agent_model_falls_back_to_legacy_satellite_env_var(self):
        from config.settings import Settings

        with patch.dict(os.environ, {"SATELLITE_AGENT_MODEL": "legacy/model"}, clear=True):
            loaded = Settings()

        self.assertEqual(loaded.earthdata_agent_model, "legacy/model")

    def test_settings_loads_default_agent_providers(self):
        from config.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            loaded = Settings()

        self.assertEqual(loaded.supervisor_model_provider, "google")
        self.assertEqual(loaded.earthdata_agent_provider, "groq")
        self.assertEqual(loaded.ground_agent_provider, "groq")

    def test_settings_loads_agent_provider_overrides(self):
        from config.settings import Settings

        with patch.dict(
            os.environ,
            {
                "SUPERVISOR_MODEL_PROVIDER": "groq",
                "EARTHDATA_AGENT_PROVIDER": "google",
                "GROUND_AGENT_PROVIDER": "google",
            },
            clear=True,
        ):
            loaded = Settings()

        self.assertEqual(loaded.supervisor_model_provider, "groq")
        self.assertEqual(loaded.earthdata_agent_provider, "google")
        self.assertEqual(loaded.ground_agent_provider, "google")

    def test_validate_startup_requires_google_key_only_when_a_google_agent_is_configured(self):
        from config.settings import Settings

        # Default posture: supervisor on google, both subagents on groq.
        loaded = Settings(db_password="x", jwt_secret_key="x", google_api_key=None, groq_api_key="x")
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
            loaded.validate_startup()

        # No agent resolves to google -> GOOGLE_API_KEY is not required.
        loaded = Settings(
            db_password="x",
            jwt_secret_key="x",
            google_api_key=None,
            groq_api_key="x",
            supervisor_model_provider="groq",
        )
        loaded.validate_startup()

    def test_validate_startup_requires_groq_key_only_when_a_groq_agent_is_configured(self):
        from config.settings import Settings

        # Default posture: both subagents resolve to groq.
        loaded = Settings(db_password="x", jwt_secret_key="x", google_api_key="x", groq_api_key=None)
        with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
            loaded.validate_startup()

        # No agent resolves to groq -> GROQ_API_KEY is not required.
        loaded = Settings(
            db_password="x",
            jwt_secret_key="x",
            google_api_key="x",
            groq_api_key=None,
            earthdata_agent_provider="google",
            ground_agent_provider="google",
        )
        loaded.validate_startup()

    def test_settings_normalizes_invalid_modes(self):
        from config.settings import get_settings

        with patch.dict(os.environ, {"DATA_FETCH_MODE": "bogus", "LOG_FORMAT": "xml"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.data_fetch_mode, "auto")
        self.assertEqual(loaded.log_format, "text")

    def test_json_formatter_outputs_expected_fields_and_extra_values(self):
        from utils.logging import JsonFormatter

        record = logging.LogRecord(
            name="api",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Request completed",
            args=(),
            exc_info=None,
        )
        record._request_id = "req-1"

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["module"], "api")
        self.assertEqual(payload["message"], "Request completed")
        self.assertEqual(payload["request_id"], "req-1")
        self.assertIn("timestamp", payload)


if __name__ == "__main__":
    unittest.main()
