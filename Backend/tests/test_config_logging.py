import json
import logging
import os
import unittest
from unittest.mock import patch

# T61: the identity-provider pair validate_startup() now requires. Every
# Settings(...) below has to satisfy it to reach the assertion it actually
# cares about, so it lives here -- the next required-var change edits one line
# rather than every construction in the file.
SUPABASE_KWARGS = {
    "supabase_url": "https://test-project.supabase.co",
    "supabase_publishable_key": "k",
}


class ConfigLoggingTests(unittest.TestCase):
    def setUp(self):
        from tta_backend.config import settings

        settings.get_settings.cache_clear()
        # get_settings() calls load_dotenv() on every cache miss, which reads
        # this checkout's real .env (repo root) straight into os.environ --
        # bypassing patch.dict(..., clear=True) below and leaking real
        # secrets/config into what these tests assert are un-set defaults.
        # Docker's backend-test image has no .env in its build context, so
        # this only bites on a host run with a real .env present (see
        # CLAUDE.md's "Optional deps / .env bleed" note). Neutralize it here
        # so these tests assert Settings' own defaults, not the developer's
        # local file.
        self._load_dotenv_patcher = patch("tta_backend.config.settings.load_dotenv")
        self._load_dotenv_patcher.start()
        self.addCleanup(self._load_dotenv_patcher.stop)

    def tearDown(self):
        from tta_backend.config import settings

        settings.get_settings.cache_clear()

    def test_settings_loads_defaults_and_validates_required_startup_values(self):
        from tta_backend.config.settings import Settings, get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.llm_model, "gemma-4-31b-it")
        self.assertEqual(loaded.ground_agent_model, "gemini-3.1-flash-lite")
        self.assertEqual(loaded.data_fetch_mode, "auto")
        self.assertEqual(loaded.harmony_processing_timeout_seconds, 600)
        loaded = Settings(db_password=None, google_api_key=None)
        with self.assertRaisesRegex(RuntimeError, "DB_PASSWORD, GOOGLE_API_KEY"):
            loaded.validate_startup()

    def test_settings_loads_harmony_processing_timeout(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {"HARMONY_PROCESSING_TIMEOUT_SECONDS": "15"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.harmony_processing_timeout_seconds, 15)

    def test_settings_loads_earthdata_mcp_defaults_and_overrides(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.earthdata_mcp_url, "http://mcp:8765/mcp")
        self.assertIsNone(loaded.earthdata_mcp_token)

    def test_agent_recursion_limit_default_has_headroom_for_a_multi_period_workflow(self):
        """Regression guard (2026-07-20 AOD wildfire session): the default was
        25, but a legitimate two-period, multi-day plot workflow (search → AOI →
        coverage → retrieve → poll → plot, ×2 periods) is ~two supersteps per
        tool call and runs right into that ceiling, surfacing as an opaque
        GraphRecursionError. The default must leave room for that real workflow;
        it stays tunable so a runaway loop can still be capped lower/higher."""
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertGreaterEqual(loaded.agent_recursion_limit, 40)

    def test_agent_recursion_limit_is_overridable(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {"AGENT_RECURSION_LIMIT": "60"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.agent_recursion_limit, 60)

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
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.retrieval_soft_cap_bytes, 2 * 1024 ** 3)
        self.assertEqual(loaded.retrieval_hard_cap_bytes, 10 * 1024 ** 3)
        self.assertEqual(loaded.await_retrieval_poll_min_seconds, 2)
        # Capped low on purpose: the backoff's ceiling is also how far behind
        # reality a narrated status can fall once it saturates. See the note on
        # the field in config/settings.py.
        self.assertEqual(loaded.await_retrieval_poll_max_seconds, 5)
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

    def test_settings_loads_bundle_open_gate_default_and_override(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        # 8 GiB, not the original 2: that number was sized to bound RAM, and
        # open_max_chunk_bytes does that now. What is left for this gate to
        # protect is extract-cache disk and the retrieval whose size the
        # provider could not estimate, and 2 GiB was quietly capping an
        # ordinary two-day TEMPO request in the name of a risk that had moved.
        self.assertEqual(loaded.bundle_open_max_uncompressed_bytes, 8 * 1024 ** 3)

        with patch.dict(os.environ, {"BUNDLE_OPEN_MAX_UNCOMPRESSED_BYTES": "1234"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.bundle_open_max_uncompressed_bytes, 1234)

    def test_settings_loads_subagent_trim_token_ceiling_default_and_override(self):
        from tta_backend.config.settings import get_settings

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
        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            loaded = Settings()
        self.assertEqual(loaded.earthdata_agent_model, "gemini-3.1-flash-lite")

        with patch.dict(os.environ, {"EARTHDATA_AGENT_MODEL": "some/other-model"}, clear=True):
            loaded = Settings()
        self.assertEqual(loaded.earthdata_agent_model, "some/other-model")

    def test_settings_earthdata_agent_model_falls_back_to_legacy_satellite_env_var(self):
        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {"SATELLITE_AGENT_MODEL": "legacy/model"}, clear=True):
            loaded = Settings()

        self.assertEqual(loaded.earthdata_agent_model, "legacy/model")

    def test_settings_loads_default_agent_providers(self):
        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            loaded = Settings()

        self.assertEqual(loaded.supervisor_model_provider, "google")
        self.assertEqual(loaded.earthdata_agent_provider, "google")
        self.assertEqual(loaded.ground_agent_provider, "google")

    def test_settings_loads_agent_provider_overrides(self):
        from tta_backend.config.settings import Settings

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
        from tta_backend.config.settings import Settings

        # Default posture: supervisor, earthdata, and ground agent all on google.
        loaded = Settings(db_password="x", **SUPABASE_KWARGS, google_api_key=None, groq_api_key="x")
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
            loaded.validate_startup()

        # No agent resolves to google -> GOOGLE_API_KEY is not required.
        loaded = Settings(
            db_password="x",
            **SUPABASE_KWARGS,
            google_api_key=None,
            groq_api_key="x",
            supervisor_model_provider="groq",
            earthdata_agent_provider="groq",
            ground_agent_provider="groq",
        )
        loaded.validate_startup()

    def test_validate_startup_requires_the_supabase_identity_provider(self):
        from tta_backend.config.settings import Settings

        # T61: unconditional, unlike the provider keys. api.py builds the
        # verifier's issuer as f"{supabase_url}/auth/v1" at import, so a None
        # URL constructs a verifier reading "None/auth/v1" that rejects every
        # token with "Invalid issuer." on a backend that booted clean. Naming
        # it at boot is the whole point of the check.
        loaded = Settings(
            db_password="x",
            google_api_key="x",
            groq_api_key="x",
            supabase_url=None,
            supabase_publishable_key="k",
        )
        with self.assertRaisesRegex(RuntimeError, "SUPABASE_URL"):
            loaded.validate_startup()

        # Unread by the backend until Phase 3 serves it from /config/auth, and
        # required anyway: the frontend cannot sign anyone in without it, so a
        # deploy carrying one of the pair is broken either way.
        loaded = Settings(
            db_password="x",
            google_api_key="x",
            groq_api_key="x",
            supabase_url="https://test-project.supabase.co",
            supabase_publishable_key=None,
        )
        with self.assertRaisesRegex(RuntimeError, "SUPABASE_PUBLISHABLE_KEY"):
            loaded.validate_startup()

    def test_validate_startup_passes_with_the_identity_provider_configured(self):
        from tta_backend.config.settings import Settings

        # The other half of the check above: proves the two branches reject a
        # missing value rather than rejecting everything.
        loaded = Settings(db_password="x", google_api_key="x", groq_api_key="x", **SUPABASE_KWARGS)
        loaded.validate_startup()  # must not raise

    def test_validate_startup_requires_groq_key_only_when_a_groq_agent_is_configured(self):
        from tta_backend.config.settings import Settings

        # A groq-configured subagent requires GROQ_API_KEY.
        loaded = Settings(
            db_password="x",
            **SUPABASE_KWARGS,
            google_api_key="x",
            groq_api_key=None,
            ground_agent_provider="groq",
        )
        with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
            loaded.validate_startup()

        # Default posture: no agent resolves to groq -> GROQ_API_KEY is not required.
        loaded = Settings(db_password="x", **SUPABASE_KWARGS, google_api_key="x", groq_api_key=None)
        loaded.validate_startup()

    def test_validate_startup_rejects_a_malformed_earthdata_mcp_url(self):
        from tta_backend.config.settings import ConfigurationError, Settings

        # A config typo (bad scheme, no host) is a bug to fix at boot, not an
        # outage the connection manager should retry (T17).
        loaded = Settings(
            db_password="x", **SUPABASE_KWARGS, google_api_key="x", groq_api_key="x",
            earthdata_mcp_url="not-a-url",
        )
        with self.assertRaisesRegex(ConfigurationError, "EARTHDATA_MCP_URL"):
            loaded.validate_startup()

    def test_validate_startup_accepts_a_well_formed_earthdata_mcp_url(self):
        from tta_backend.config.settings import Settings

        loaded = Settings(
            db_password="x", **SUPABASE_KWARGS, google_api_key="x", groq_api_key="x",
            earthdata_mcp_url="http://mcp:8765/mcp",
        )
        loaded.validate_startup()  # must not raise

    def test_settings_loads_debug_heap_profiling_flag_default_and_override(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertFalse(loaded.debug_heap_profiling_enabled)

        with patch.dict(os.environ, {"DEBUG_HEAP_PROFILING_ENABLED": "1"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertTrue(loaded.debug_heap_profiling_enabled)

    def test_settings_loads_mcp_call_timeout_and_chat_turn_timeout_defaults(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.mcp_call_timeout_seconds, 120)
        self.assertEqual(loaded.chat_turn_timeout_seconds, 1800)

    def test_settings_loads_mcp_call_timeout_and_chat_turn_timeout_from_env(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(
            os.environ,
            {"MCP_CALL_TIMEOUT_SECONDS": "45", "CHAT_TURN_TIMEOUT_SECONDS": "600"},
            clear=True,
        ):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.mcp_call_timeout_seconds, 45)
        self.assertEqual(loaded.chat_turn_timeout_seconds, 600)

    def test_validate_startup_rejects_a_chat_turn_timeout_not_greater_than_await_retrieval_timeout(self):
        from tta_backend.config.settings import ConfigurationError, Settings

        # A misconfiguration here would make every retrieval that runs the
        # full await_retrieval_timeout_seconds a guaranteed turn timeout.
        loaded = Settings(
            db_password="x", **SUPABASE_KWARGS, google_api_key="x", groq_api_key="x",
            await_retrieval_timeout_seconds=900,
            chat_turn_timeout_seconds=900,
        )
        with self.assertRaisesRegex(ConfigurationError, "CHAT_TURN_TIMEOUT_SECONDS"):
            loaded.validate_startup()

    def test_validate_startup_accepts_a_chat_turn_timeout_greater_than_await_retrieval_timeout(self):
        from tta_backend.config.settings import Settings

        loaded = Settings(
            db_password="x", **SUPABASE_KWARGS, google_api_key="x", groq_api_key="x",
            await_retrieval_timeout_seconds=900,
            chat_turn_timeout_seconds=1800,
        )
        loaded.validate_startup()  # must not raise

    def test_settings_normalizes_invalid_modes(self):
        from tta_backend.config.settings import get_settings

        with patch.dict(os.environ, {"DATA_FETCH_MODE": "bogus", "LOG_FORMAT": "xml"}, clear=True):
            get_settings.cache_clear()
            loaded = get_settings()

        self.assertEqual(loaded.data_fetch_mode, "auto")
        self.assertEqual(loaded.log_format, "text")

    def test_configure_logging_silences_the_benign_langchain_google_genai_schema_warning(self):
        """T45: langchain_google_genai._function_utils logs
        "Key '...' is not supported in schema, ignoring" at WARNING for
        every unrecognized JSON-schema keyword in a tool's pydantic model —
        known-benign, but noisy enough to drown real WARNING/ERROR events in
        the log auditor's correlation (QA note, 2026-07-17). Silenced at
        logger config rather than at each call site."""
        from tta_backend.utils.logging import configure_logging

        noisy_logger = logging.getLogger("langchain_google_genai._function_utils")
        unrelated_logger = logging.getLogger("api")

        configure_logging()

        self.assertFalse(noisy_logger.isEnabledFor(logging.WARNING))
        self.assertTrue(unrelated_logger.isEnabledFor(logging.WARNING))

    def test_json_formatter_outputs_expected_fields_and_extra_values(self):
        from tta_backend.utils.logging import JsonFormatter

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
