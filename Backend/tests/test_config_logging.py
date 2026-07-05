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

        self.assertEqual(loaded.llm_model, "llama-3.3-70b-versatile")
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
