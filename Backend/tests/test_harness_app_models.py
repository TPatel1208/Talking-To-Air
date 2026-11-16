"""The app's harness models: each role built from the setting pair the agents already use, and the
cache breakpoint a portkey Claude model puts on the final message of each request."""
import dataclasses
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tta_backend.config.model_factory import _portkey_chat_class, build_chat_model, cacheable_system_prompt
from tta_backend.harness.models import ModelChoice, NamedModel, system_message
from tta_backend.infrastructure.settings import Settings
from tta_backend.services.harness_models import (
    AppModels,
    build_cached_chat_model,
    mark_final_message,
    takes_cache_breakpoints,
)

CLAUDE = "@gw/anthropic.claude-sonnet-5"
SETTINGS = Settings(
    llm_model=CLAUDE,
    supervisor_model_provider="portkey",
    earthdata_agent_model="gemini-3.5-flash-lite",
    earthdata_agent_provider="google",
    portkey_api_key="test-key",
)
MARK = {"type": "ephemeral"}


def built():
    """Patches both builders with mocks; each call is recorded as (provider, model, settings, agent_type)."""
    calls: list[tuple[str, str, Settings, str]] = []

    def plain(provider, model, settings, *, agent_type):  # type: ignore[no-untyped-def]
        calls.append((provider, model, settings, agent_type))
        return Mock()

    def cached(model, settings, *, agent_type):  # type: ignore[no-untyped-def]
        calls.append(("portkey+cache", model, settings, agent_type))
        return Mock()

    patches = (
        patch("tta_backend.services.harness_models.build_chat_model", side_effect=plain),
        patch("tta_backend.services.harness_models.build_cached_chat_model", side_effect=cached),
    )
    return calls, patches


class AppModelsTests(unittest.TestCase):
    def roles(self, settings: Settings, *roles: str):  # type: ignore[no-untyped-def]
        calls, (plain, cached) = built()
        with plain, cached:
            models = AppModels(settings)
            choices = {role: models.for_role(role) for role in roles}  # type: ignore[arg-type]
        return choices, calls

    def test_explore_and_supervisor_use_the_supervisors_model_and_agent_step_the_cheaper_one(self):
        choices, calls = self.roles(SETTINGS, "explore", "supervisor", "agent_step")
        self.assertEqual(choices["explore"].name, f"portkey:{CLAUDE}")
        self.assertEqual(choices["supervisor"].name, choices["explore"].name)
        self.assertEqual(choices["agent_step"].name, "google:gemini-3.5-flash-lite")
        self.assertEqual({(c[0], c[1]): c[3] for c in calls}, {
            ("portkey+cache", CLAUDE): "supervisor",
            ("google", "gemini-3.5-flash-lite"): "satellite",
        })
        self.assertTrue(all(c[2].llm_max_retries == 0 for c in calls))

    def test_sdk_retries_are_off_and_the_given_settings_are_untouched(self):
        settings = dataclasses.replace(SETTINGS, llm_max_retries=2)
        _, calls = self.roles(settings, "explore", "agent_step", "supervisor")
        self.assertEqual({c[2].llm_max_retries for c in calls}, {0})
        self.assertEqual(settings.llm_max_retries, 2)

    def test_a_role_is_built_once_and_has_no_fallbacks(self):
        calls, (plain, cached) = built()
        with plain, cached:
            models = AppModels(SETTINGS)
            first = models.for_role("agent_step")
            second = models.for_role("agent_step")
        self.assertIs(first, second)
        self.assertEqual(first.fallbacks, ())
        self.assertEqual(len(calls), 1)

    def test_the_names_pass_the_traces_checks_for_the_vetted_gateway_models(self):
        settings = Settings(
            llm_model="@gcp-internal-dataset-integration/anthropic.claude-sonnet-5",
            supervisor_model_provider="portkey",
            earthdata_agent_model="@gcp-internal-dataset-integration/anthropic.claude-haiku-4-5@20251001",
            earthdata_agent_provider="portkey",
        )
        choices, _ = self.roles(settings, "explore", "agent_step", "supervisor")
        self.assertEqual({role: c.name for role, c in choices.items()}, {
            "explore": "portkey:@gcp-internal-dataset-integration/anthropic.claude-sonnet-5",
            "agent_step": "portkey:@gcp-internal-dataset-integration/anthropic.claude-haiku-4-5@20251001",
            "supervisor": "portkey:@gcp-internal-dataset-integration/anthropic.claude-sonnet-5",
        })

    def test_an_unknown_role_has_no_model(self):
        with patch("tta_backend.services.harness_models.build_chat_model"):
            self.assertIsNone(AppModels(SETTINGS).for_role("other"))  # type: ignore[arg-type]

    def test_prompt_caching_is_on_for_portkey_claude_models_only(self):
        choices, calls = self.roles(SETTINGS, "explore", "supervisor", "agent_step")
        self.assertTrue(choices["explore"].prompt_cache)
        self.assertTrue(choices["supervisor"].prompt_cache)
        self.assertFalse(choices["agent_step"].prompt_cache)
        settings = dataclasses.replace(SETTINGS, earthdata_agent_provider="portkey", earthdata_agent_model="@gw/openai.gpt-5.5")
        choices, calls = self.roles(settings, "agent_step")
        self.assertFalse(choices["agent_step"].prompt_cache)
        self.assertEqual(calls[0][0], "portkey")

    def test_the_caching_rule_is_the_agents_cached_system_prompt_rule(self):
        for provider, model in [
            ("portkey", CLAUDE), ("portkey", "@gw/anthropic.CLAUDE-haiku-4-5"),
            ("portkey", "@gw/openai.gpt-5.5"), ("google", "claude-lookalike"), ("groq", "llama-4"),
        ]:
            with self.subTest(provider=provider, model=model):
                marked = isinstance(cacheable_system_prompt("x", provider, model), SystemMessage)
                self.assertEqual(takes_cache_breakpoints(provider, model), marked)

    def test_the_cached_model_is_built_as_the_factory_builds_a_portkey_model(self):
        plain = build_chat_model("portkey", CLAUDE, SETTINGS, agent_type="supervisor")
        cached = build_cached_chat_model(CLAUDE, SETTINGS, agent_type="supervisor")
        self.assertIsInstance(cached, _portkey_chat_class())
        clients = {"client", "async_client", "root_client", "root_async_client", "callbacks"}
        for field in sorted(plain.model_fields_set - clients):
            with self.subTest(field=field):
                self.assertEqual(getattr(cached, field), getattr(plain, field))
        self.assertEqual([type(c) for c in cached.callbacks], [type(c) for c in plain.callbacks])
        self.assertEqual(cached.extra_body, {"cache_control": MARK})


def payload(model, messages):  # type: ignore[no-untyped-def]
    return model._get_request_payload(messages)


def marks(sent: dict) -> list[int]:
    """The index of each payload message whose content carries a cache breakpoint."""
    return [
        i for i, m in enumerate(sent["messages"])
        if isinstance(m.get("content"), list) and any("cache_control" in b for b in m["content"] if isinstance(b, dict))
    ]


class FinalMessageMarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_cached_chat_model(CLAUDE, SETTINGS, agent_type="supervisor")
        choice = ModelChoice(primary=NamedModel(self.model, f"portkey:{CLAUDE}"), prompt_cache=True)
        self.system = system_message("You are a specialist.", choice)
        self.call = AIMessage(content="", tool_calls=[
            {"name": "lookup", "args": {"place": "NJ"}, "id": "call_1", "type": "tool_call"},
        ])

    def test_the_final_message_is_marked_whether_tool_assistant_or_user_and_no_earlier_one_is(self):
        conversations = {
            "user": [self.system, HumanMessage("Find NO2.")],
            "tool": [self.system, HumanMessage("Find NO2."), self.call, ToolMessage('{"status":"done"}', tool_call_id="call_1")],
            "assistant": [self.system, HumanMessage("Find NO2."), AIMessage("It is 2.5.")],
            "user after tool": [
                self.system, HumanMessage("Find NO2."), self.call, ToolMessage("{}", tool_call_id="call_1"),
                HumanMessage("Answer now."),
            ],
        }
        for name, messages in conversations.items():
            with self.subTest(final=name):
                sent = payload(self.model, messages)
                last = len(messages) - 1
                # The system prompt's own mark, then the final message's; with the top-level one, three in all.
                self.assertEqual(marks(sent), [0, last])
                self.assertEqual(sent["messages"][last]["content"][-1]["cache_control"], MARK)
                self.assertEqual(sent["extra_body"], {"cache_control": MARK})
                self.assertLessEqual(len(marks(sent)) + 1, 4)

    def test_the_messages_given_are_left_unmarked(self):
        human = HumanMessage(content=[{"type": "text", "text": "Find NO2."}])
        payload(self.model, [self.system, human])
        self.assertEqual(human.content, [{"type": "text", "text": "Find NO2."}])

    def test_nothing_to_mark_is_left_as_it_is(self):
        sent = payload(self.model, [self.system, HumanMessage("Find NO2."), self.call])
        self.assertEqual(marks(sent), [0])
        for content in ("", "   ", [{"type": "text", "text": ""}], None):
            with self.subTest(content=content):
                body = {"messages": [{"role": "user", "content": content}]}
                mark_final_message(body)
                self.assertEqual(body, {"messages": [{"role": "user", "content": content}]})
        mark_final_message({})

    def test_the_third_request_of_a_tool_loop(self):
        second = AIMessage(content="", tool_calls=[
            {"name": "measure", "args": {"area": "polygons_1"}, "id": "call_2", "type": "tool_call"},
        ])
        messages = [
            self.system, HumanMessage("Find NO2."),
            self.call, ToolMessage('{"handle":"polygons_1"}', tool_call_id="call_1"),
            second, ToolMessage('{"value":2.5}', tool_call_id="call_2"),
        ]
        sent = payload(self.model, messages)

        def called(call_id: str, name: str, args: dict) -> dict:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "type": "function", "id": call_id, "function": {"name": name, "arguments": json.dumps(args)},
            }]}

        self.assertEqual(sent["extra_body"], {"cache_control": MARK})
        self.assertEqual(sent["messages"], [
            {"role": "system", "content": [{"type": "text", "text": "You are a specialist.", "cache_control": MARK}]},
            {"role": "user", "content": "Find NO2."},
            called("call_1", "lookup", {"place": "NJ"}),
            {"role": "tool", "tool_call_id": "call_1", "content": '{"handle":"polygons_1"}'},
            called("call_2", "measure", {"area": "polygons_1"}),
            {"role": "tool", "tool_call_id": "call_2",
             "content": [{"type": "text", "text": '{"value":2.5}', "cache_control": MARK}]},
        ])


if __name__ == "__main__":
    unittest.main()
