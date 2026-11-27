"""Reaching a model: per-role choice, retry then fallback, the usage meter and the scripted fake."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import httpcore
import httpx
import openai
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.exceptions import ContextOverflowError, ModelAuthenticationError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tta_backend.harness.models import (
    CACHE_CONTROL,
    NO_BACKOFF,
    Backoff,
    ModelChoice,
    NamedModel,
    UsageMeter,
    is_transient,
    resilience_middleware,
    system_message,
)
from tta_backend.harness.run_context import NO_MODEL_REASON
from tta_backend.harness.testing import (
    ScriptedChatModel,
    ScriptedModels,
    ScriptExhausted,
    answer,
    make_run_env,
    rate_limited,
    tool_call,
)
from tta_backend.harness.trace import FailureCategory, StepFailure


class Finding(BaseModel):
    summary: str


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def lookup_tool(ran: list[str]):
    @tool
    def lookup(place: str) -> str:
        """Look up a place."""
        ran.append(place)
        return f"found {place}"

    return lookup


def choice(primary: ScriptedChatModel, *fallbacks: ScriptedChatModel) -> ModelChoice:
    return ModelChoice(
        primary=NamedModel(primary, "fake:primary"),
        fallbacks=tuple(NamedModel(model, f"fake:fallback-{i}") for i, model in enumerate(fallbacks, 1)),
    )


def agent_for(models: ModelChoice, ran: list[str], *, max_retries: int = 2, response_format=None):
    meter = UsageMeter(models)
    agent = create_agent(
        models.model,
        tools=[lookup_tool(ran)],
        middleware=[*resilience_middleware(models, max_retries=max_retries, backoff=NO_BACKOFF), meter],
        response_format=response_format,
    )
    return agent, meter


def ask(text: str = "Where?"):
    return {"messages": [HumanMessage(text)]}


class AgentWithScriptedModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_then_answer(self):
        model = ScriptedChatModel(script=[tool_call("lookup", {"place": "Boston"}), answer("Boston it is")])
        ran: list[str] = []
        agent, _ = agent_for(choice(model), ran)
        result = await agent.ainvoke(ask())
        self.assertEqual(ran, ["Boston"])
        self.assertEqual(result["messages"][-1].content, "Boston it is")
        self.assertEqual(model.tools_offered[0], ["lookup"])
        # The second call was sent the tool's result.
        sent = model.received[1]
        self.assertIsInstance(sent[-1], ToolMessage)
        self.assertEqual(sent[-1].content, "found Boston")

    def test_sync_invoke_runs_the_same_script(self):
        model = ScriptedChatModel(script=[tool_call("lookup", {"place": "Reno"}), answer("ok", usage=(4, 2))])
        ran: list[str] = []
        agent, meter = agent_for(choice(model), ran)
        result = agent.invoke(ask())
        self.assertEqual((ran, result["messages"][-1].content), (["Reno"], "ok"))
        usage = meter.usage()
        self.assertEqual((usage.model_calls, usage.tool_calls, usage.input_tokens), (2, 1, 4))

    async def test_retry_recovers_without_rerunning_the_tool(self):
        model = ScriptedChatModel(
            script=[tool_call("lookup", {"place": "Boston"}), rate_limited(), answer("done")]
        )
        ran: list[str] = []
        agent, meter = agent_for(choice(model), ran)
        result = await agent.ainvoke(ask())
        self.assertEqual(result["messages"][-1].content, "done")
        self.assertEqual(ran, ["Boston"])
        self.assertEqual(model.calls, 3)
        # The retried call was sent the same conversation as the failed one.
        self.assertEqual(model.received[1], model.received[2])
        usage = meter.usage()
        self.assertEqual((usage.model_calls, usage.failed_model_calls, usage.tool_calls), (2, 1, 1))
        self.assertEqual(usage.model_name, "fake:primary")

    async def test_fallback_answers_after_retries_run_out(self):
        primary = ScriptedChatModel(script=[rate_limited(), rate_limited(), rate_limited()])
        fallback = ScriptedChatModel(script=[answer("from the fallback", usage=(9, 4))])
        ran: list[str] = []
        agent, meter = agent_for(choice(primary, fallback), ran, max_retries=2)
        result = await agent.ainvoke(ask())
        self.assertEqual(result["messages"][-1].content, "from the fallback")
        self.assertEqual((primary.calls, fallback.calls), (3, 1))
        usage = meter.usage()
        self.assertEqual(usage.model_name, "fake:fallback-1")
        self.assertEqual(usage.models_used, ("fake:fallback-1",))
        self.assertEqual((usage.model_calls, usage.failed_model_calls), (1, 3))
        self.assertEqual((usage.input_tokens, usage.output_tokens), (9, 4))

    async def test_each_turn_tries_the_primary_first(self):
        primary = ScriptedChatModel(
            script=[tool_call("lookup", {"place": "Reno"}), rate_limited(), rate_limited()]
        )
        fallback = ScriptedChatModel(script=[answer("fallback finished")])
        ran: list[str] = []
        agent, meter = agent_for(choice(primary, fallback), ran, max_retries=1)
        result = await agent.ainvoke(ask())
        self.assertEqual(result["messages"][-1].content, "fallback finished")
        self.assertEqual(ran, ["Reno"])
        usage = meter.usage()
        self.assertEqual(usage.models_used, ("fake:primary", "fake:fallback-1"))
        self.assertEqual(usage.model_name, "fake:fallback-1")

    async def test_a_lasting_failure_falls_back_without_retrying(self):
        primary = ScriptedChatModel(script=[ModelAuthenticationError("refused")])
        fallback = ScriptedChatModel(script=[answer("ok")])
        agent, _ = agent_for(choice(primary, fallback), [])
        result = await agent.ainvoke(ask())
        self.assertEqual(result["messages"][-1].content, "ok")
        self.assertEqual(primary.calls, 1)

    async def test_last_error_propagates_when_every_model_fails(self):
        primary = ScriptedChatModel(script=[rate_limited()] * 2)
        fallback = ScriptedChatModel(script=[ContextOverflowError("too long")])
        agent, meter = agent_for(choice(primary, fallback), [], max_retries=1)
        with self.assertRaises(ContextOverflowError):
            await agent.ainvoke(ask())
        self.assertEqual((primary.calls, fallback.calls), (2, 1))
        self.assertEqual(meter.usage().failed_model_calls, 3)

    async def test_without_fallbacks_exhausted_retries_raise(self):
        model = ScriptedChatModel(script=[rate_limited()] * 3)
        agent, _ = agent_for(choice(model), [], max_retries=2)
        with self.assertRaises(Exception) as caught:
            await agent.ainvoke(ask())
        self.assertTrue(is_transient(caught.exception))
        self.assertEqual(model.calls, 3)

    async def test_meter_counts_calls_and_tokens(self):
        model = ScriptedChatModel(
            script=[
                tool_call("lookup", {"place": "A"}, usage=(10, 5)),
                tool_call("lookup", {"place": "B"}),  # no usage: counts as zero
                tool_call("Finding", {"summary": "both found"}, usage=(30, 7)),
            ]
        )
        ran: list[str] = []
        agent, meter = agent_for(choice(model), ran, response_format=ToolStrategy(Finding))
        result = await agent.ainvoke(ask())
        self.assertEqual(result["structured_response"], Finding(summary="both found"))
        self.assertEqual(ran, ["A", "B"])
        self.assertIn("Finding", model.tools_offered[0])
        usage = meter.usage()
        # The structured answer is a model call, not a tool call.
        self.assertEqual((usage.model_calls, usage.tool_calls), (3, 2))
        self.assertEqual((usage.input_tokens, usage.output_tokens), (40, 12))
        self.assertEqual(usage.failed_model_calls, 0)

    async def test_meter_sums_cache_read_and_creation_tokens(self):
        def cached(message: AIMessage, read: int | None, created: int | None) -> AIMessage:
            details = {k: v for k, v in (("cache_read", read), ("cache_creation", created)) if v is not None}
            message.usage_metadata = {
                "input_tokens": 100, "output_tokens": 5, "total_tokens": 105, "input_token_details": details,
            }
            return message

        model = ScriptedChatModel(script=[
            cached(tool_call("lookup", {"place": "A"}), 0, 80),
            cached(tool_call("lookup", {"place": "B"}), 80, 15),
            cached(tool_call("lookup", {"place": "C"}), 95, None),  # no creation: counts as zero
            answer("done", usage=(10, 1)),  # no details at all
        ])
        agent, meter = agent_for(choice(model), [])
        await agent.ainvoke(ask())
        usage = meter.usage()
        self.assertEqual((usage.input_tokens, usage.output_tokens), (310, 16))
        self.assertEqual((usage.cache_read_tokens, usage.cache_creation_tokens), (175, 95))

    async def test_an_overrun_script_is_not_retried(self):
        model = ScriptedChatModel(script=[])
        agent, _ = agent_for(choice(model), [])
        with self.assertRaises(ScriptExhausted):
            await agent.ainvoke(ask())
        self.assertEqual(model.calls, 1)


class ResilienceConfigTests(unittest.TestCase):
    def test_retry_sits_inside_fallback(self):
        primary, fallback = ScriptedChatModel(), ScriptedChatModel()
        layers = resilience_middleware(choice(primary, fallback), max_retries=3, backoff=Backoff(initial=0.5))
        self.assertIsInstance(layers[0], ModelFallbackMiddleware)
        self.assertIsInstance(layers[1], ModelRetryMiddleware)
        self.assertEqual(layers[0].models, [fallback])
        retry = layers[1]
        self.assertEqual((retry.max_retries, retry.on_failure, retry.initial_delay), (3, "error", 0.5))

    def test_no_fallbacks_means_retry_only(self):
        layers = resilience_middleware(choice(ScriptedChatModel()))
        self.assertEqual(len(layers), 1)
        self.assertIsInstance(layers[0], ModelRetryMiddleware)

    def test_transient_errors(self):
        self.assertTrue(is_transient(rate_limited()))
        self.assertTrue(is_transient(TimeoutError()))
        self.assertTrue(is_transient(ConnectionResetError()))
        self.assertTrue(is_transient(StatusError(429)))
        self.assertTrue(is_transient(StatusError(503)))
        self.assertFalse(is_transient(StatusError(400)))
        self.assertFalse(is_transient(ModelAuthenticationError("no")))
        self.assertFalse(is_transient(ContextOverflowError("long")))
        self.assertFalse(is_transient(ValueError("bug")))

    def test_openai_and_httpx_dropped_calls_are_transient(self):
        request = httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")
        for exc in (
            openai.APIConnectionError(request=request),
            openai.APITimeoutError(request=request),
            httpx.ReadTimeout("read timed out", request=request),
            httpx.ConnectTimeout("connect timed out", request=request),
            httpx.ConnectError("connection refused", request=request),
            httpx.RemoteProtocolError("peer closed connection", request=request),
            httpcore.ReadTimeout("read timed out"),
            httpcore.ConnectError("connection refused"),
        ):
            with self.subTest(type(exc).__name__):
                self.assertTrue(is_transient(exc))

    def test_openai_status_errors_follow_their_status(self):
        request = httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")

        def status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
            return cls("scripted", response=httpx.Response(status, request=request), body=None)

        self.assertTrue(is_transient(status_error(openai.RateLimitError, 429)))
        self.assertTrue(is_transient(status_error(openai.InternalServerError, 500)))
        for cls, status in (
            (openai.AuthenticationError, 401),
            (openai.PermissionDeniedError, 403),
            (openai.BadRequestError, 400),
        ):
            with self.subTest(cls.__name__):
                self.assertFalse(is_transient(status_error(cls, status)))
        self.assertFalse(is_transient(httpx.UnsupportedProtocol("no scheme")))

    def test_harness_imports_no_provider_sdk(self):
        code = (
            "import sys\n"
            "import tta_backend.harness.models, tta_backend.harness.testing\n"
            "print(sorted(m for m in ('openai', 'anthropic', 'groq') if m in sys.modules))\n"
        )
        backend = Path(__file__).resolve().parents[1]
        done = subprocess.run([sys.executable, "-c", code], cwd=backend, capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")

    def test_model_names_are_short_ids(self):
        model = ScriptedChatModel()
        self.assertEqual(NamedModel(model, "openai:gpt-5.5").name, "openai:gpt-5.5")
        for bad in ("", "has space", "https://key@host/v1", "x" * 101):
            with self.assertRaises(ValueError):
                NamedModel(model, bad)

    def test_choice_names_its_models_by_identity(self):
        primary, fallback = ScriptedChatModel(), ScriptedChatModel()
        models = choice(primary, fallback)
        self.assertIs(models.model, primary)
        self.assertEqual(models.name, "fake:primary")
        self.assertEqual(models.name_of(fallback), "fake:fallback-1")
        self.assertIsNone(models.name_of(ScriptedChatModel()))


def breakpoints(messages: list[BaseMessage]) -> list[int]:
    """The index of each message carrying a cache breakpoint."""
    return [
        i for i, m in enumerate(messages)
        if isinstance(m.content, list) and any(isinstance(b, dict) and "cache_control" in b for b in m.content)
    ]


class PromptCacheTests(unittest.IsolatedAsyncioTestCase):
    def cached(self, model: ScriptedChatModel, *, on: bool = True) -> ModelChoice:
        return ModelChoice(primary=NamedModel(model, "fake:primary"), prompt_cache=on)

    def test_off_by_default(self):
        models = ModelChoice(primary=NamedModel(ScriptedChatModel(), "fake:primary"))
        self.assertFalse(models.prompt_cache)
        self.assertEqual(system_message("Be brief.", models), "Be brief.")

    def test_the_system_prompt_is_one_text_block_with_an_ephemeral_breakpoint(self):
        message = system_message("Be brief.", self.cached(ScriptedChatModel()))
        self.assertIsInstance(message, SystemMessage)
        self.assertEqual(message.content, [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(CACHE_CONTROL, {"type": "ephemeral"})

    async def test_only_the_system_prompt_is_marked_and_the_state_holds_no_marking(self):
        model = ScriptedChatModel(script=[tool_call("lookup", {"place": "A"}), answer("done")])
        models = self.cached(model)
        agent = create_agent(models.model, tools=[lookup_tool([])], system_prompt=system_message("Be brief.", models))
        result = await agent.ainvoke(ask())
        for sent in model.received:
            self.assertEqual(breakpoints(sent), [0])
        self.assertEqual(breakpoints(result["messages"]), [])


class RunEnvModelsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    async def test_env_without_models_refuses_with_its_reason(self):
        env = await make_run_env(self._tmp.name)
        self.assertIsNone(env.models)
        with self.assertRaises(StepFailure) as caught:
            env.model_for("explore")
        self.assertEqual(caught.exception.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(caught.exception.reason, NO_MODEL_REASON.format(role="explore"))

    async def test_env_hands_out_the_role_it_was_given(self):
        models = choice(ScriptedChatModel())
        scripted = ScriptedModels({"agent_step": models})
        env = await make_run_env(self._tmp.name, models=scripted)
        self.assertIs(env.model_for("agent_step"), models)
        with self.assertRaises(StepFailure) as caught:
            env.model_for("supervisor")
        self.assertEqual(caught.exception.reason, NO_MODEL_REASON.format(role="supervisor"))
        self.assertEqual(scripted.asked, ["agent_step", "supervisor"])

    def test_every_role_shares_one_choice(self):
        models = choice(ScriptedChatModel())
        scripted = ScriptedModels.every_role(models)
        self.assertIs(scripted.for_role("explore"), models)
        self.assertIs(scripted.for_role("supervisor"), models)


if __name__ == "__main__":
    unittest.main()
