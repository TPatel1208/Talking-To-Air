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


class GeminiNativeToolSchemaTests(unittest.TestCase):
    """T64 Phase 0 F1/F3: langchain_google_genai's converter reduces a list of
    union items to STRING/OBJECT, and its bind_tools cannot carry a raw JSON
    schema. A tool that names its own JSON schema must reach the request
    untouched, bound the way create_agent binds (bind_tools(tools, tool_choice=...))."""

    def _declarations(self, tools):
        from langchain_core.messages import HumanMessage

        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.config.settings import Settings

        model = build_chat_model("google", "gemini-3.5-flash-lite", Settings(google_api_key="offline"))
        bound = model.bind_tools(tools, tool_choice=None)
        request = model._prepare_request([HumanMessage("hi")], **bound.kwargs)
        return {fd.name: fd for tool in request["config"].tools for fd in tool.function_declarations or []}

    def test_a_tool_carrying_a_native_json_schema_reaches_gemini_verbatim(self):
        from langchain_core.tools import StructuredTool

        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA
        from tta_backend.models.agent_request import DispatchArgs, dispatch_json_schema

        async def _noop(**_):
            return ""

        dispatch = StructuredTool.from_function(
            coroutine=_noop, name="dispatch", description="Dispatch tasks.", args_schema=DispatchArgs,
            metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
        )

        declaration = self._declarations([dispatch])["dispatch"]

        self.assertEqual(declaration.parameters_json_schema, dispatch_json_schema())
        self.assertIsNone(declaration.parameters)
        self.assertEqual(declaration.description, "Dispatch tasks.")

    def test_an_ordinary_tool_bound_alongside_is_converted_as_before(self):
        from langchain_core.tools import StructuredTool, tool

        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA
        from tta_backend.models.agent_request import DispatchArgs, dispatch_json_schema

        async def _noop(**_):
            return ""

        dispatch = StructuredTool.from_function(
            coroutine=_noop, name="dispatch", description="Dispatch tasks.", args_schema=DispatchArgs,
            metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
        )

        @tool
        def lookup(code: str) -> str:
            """Look a code up."""
            return code

        declarations = self._declarations([dispatch, lookup])

        self.assertIsNone(declarations["lookup"].parameters_json_schema)
        self.assertEqual(list(declarations["lookup"].parameters.properties), ["code"])

    def test_tool_choice_none_reaches_gemini_as_function_calling_mode_none(self):
        """The supervisor closes dispatch with tool_choice "none" (T64); the
        native-schema path must not drop it on the way to the request."""
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import StructuredTool

        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA, build_chat_model
        from tta_backend.config.settings import Settings
        from tta_backend.models.agent_request import DispatchArgs, dispatch_json_schema

        async def _noop(**_):
            return ""

        dispatch = StructuredTool.from_function(
            coroutine=_noop, name="dispatch", description="Dispatch tasks.", args_schema=DispatchArgs,
            metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
        )
        model = build_chat_model("google", "gemini-3.5-flash-lite", Settings(google_api_key="offline"))
        bound = model.bind_tools([dispatch], tool_choice="none")
        config = model._prepare_request([HumanMessage("hi")], **bound.kwargs)["config"]

        self.assertEqual(str(config.tool_config.function_calling_config.mode.value), "NONE")

    def test_native_and_converted_declarations_travel_in_one_tool(self):
        """The stock path sends every function declaration in a single Tool; a
        second function-declaration Tool is a request shape never verified
        against Gemini, so mixing must not create one."""
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import StructuredTool, tool

        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA, build_chat_model
        from tta_backend.config.settings import Settings
        from tta_backend.models.agent_request import DispatchArgs, dispatch_json_schema

        async def _noop(**_):
            return ""

        dispatch = StructuredTool.from_function(
            coroutine=_noop, name="dispatch", description="Dispatch tasks.", args_schema=DispatchArgs,
            metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
        )

        @tool
        def lookup(code: str) -> str:
            """Look a code up."""
            return code

        model = build_chat_model("google", "gemini-3.5-flash-lite", Settings(google_api_key="offline"))
        bound = model.bind_tools([dispatch, lookup], tool_choice=None)
        tools = model._prepare_request([HumanMessage("hi")], **bound.kwargs)["config"].tools

        self.assertEqual(len(tools), 1)
        self.assertEqual(sorted(fd.name for fd in tools[0].function_declarations), ["dispatch", "lookup"])


class PortkeyProviderTests(unittest.TestCase):
    """Dispatch must use the T64 native schema, where ``agent`` is required."""

    def _settings(self, **overrides):
        from tta_backend.config.settings import Settings

        return Settings(portkey_api_key="pk-offline", **overrides)

    def _dispatch_tool(self):
        from langchain_core.tools import StructuredTool

        from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA
        from tta_backend.models.agent_request import DispatchArgs, dispatch_json_schema

        async def _noop(**_):
            return ""

        return StructuredTool.from_function(
            coroutine=_noop, name="dispatch", description="Dispatch tasks.", args_schema=DispatchArgs,
            metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
        )

    def test_a_native_schema_tool_is_declared_with_that_schema(self):
        from langchain_core.tools import tool

        from tta_backend.config.model_factory import build_chat_model
        from tta_backend.models.agent_request import dispatch_json_schema

        @tool
        def lookup(code: str) -> str:
            """Look a code up."""
            return code

        model = build_chat_model("portkey", "@slug/model", self._settings())
        bound = model.bind_tools([self._dispatch_tool(), lookup], tool_choice="any")
        functions = {t["function"]["name"]: t["function"] for t in bound.kwargs["tools"]}

        self.assertEqual(functions["dispatch"]["parameters"], dispatch_json_schema())
        self.assertEqual(functions["dispatch"]["description"], "Dispatch tasks.")
        self.assertIn("agent", functions["dispatch"]["parameters"]["$defs"]["SatelliteRequest"]["required"])
        self.assertEqual(list(functions["lookup"]["parameters"]["properties"]), ["code"])

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

        from tta_backend.config.settings import Settings

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings().portkey_base_url, "https://api.portkey.ai/v1")


class CacheableSystemPromptTests(unittest.TestCase):
    """Claude via Portkey caches only a marked system prompt."""

    def test_claude_through_portkey_gets_a_cache_marker(self):
        from langchain_core.messages import SystemMessage

        from tta_backend.config.model_factory import cacheable_system_prompt

        prompt = cacheable_system_prompt("SYSTEM", "portkey", "@gcp-x/anthropic.claude-sonnet-5")

        self.assertIsInstance(prompt, SystemMessage)
        self.assertEqual(prompt.content, [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}])

    def test_other_models_keep_the_plain_string(self):
        from tta_backend.config.model_factory import cacheable_system_prompt

        for provider, model in (("google", "gemini-3.5-flash-lite"), ("groq", "claude-lookalike"), ("portkey", "@azure-x/gpt-5.4")):
            with self.subTest(provider=provider, model=model):
                self.assertEqual(cacheable_system_prompt("SYSTEM", provider, model), "SYSTEM")

    def test_the_marker_reaches_the_portkey_request(self):
        """Catches a langchain-openai upgrade that drops unknown block keys."""
        from langchain_core.messages import HumanMessage

        from tta_backend.config.model_factory import build_chat_model, cacheable_system_prompt
        from tta_backend.config.settings import Settings

        model = build_chat_model("portkey", "@x/claude", Settings(portkey_api_key="offline"))
        system = cacheable_system_prompt("SYSTEM", "portkey", "@x/claude")
        payload = model._get_request_payload([system, HumanMessage("hi")])

        self.assertEqual(payload["messages"][0]["content"][0]["cache_control"], {"type": "ephemeral"})

    def test_both_sub_agents_send_the_marked_prompt(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from langchain_core.messages import SystemMessage

        from tta_backend.agents import earthdata_agent, ground_sensor_agent

        for module, build in (
            (earthdata_agent, lambda: earthdata_agent.build_earthdata_agent("@x/claude", "portkey", mcp_tools={})),
            (ground_sensor_agent, lambda: ground_sensor_agent.build_ground_agent("@x/claude", "portkey")),
        ):
            with self.subTest(module=module.__name__):
                create_agent = Mock(return_value=SimpleNamespace())
                with patch.object(module, "build_chat_model", Mock(return_value=object())), \
                     patch.object(module, "create_agent", create_agent), \
                     patch.object(earthdata_agent, "build_satellite_tools", Mock(return_value=[])):
                    build()

                self.assertIsInstance(create_agent.call_args.kwargs["system_prompt"], SystemMessage)


if __name__ == "__main__":
    unittest.main()
