"""How the harness reaches a language model: the models a host offers per role,
the retry and fallback around each model call, the system prompt's cache
marking, and a meter of what a run used.

The host builds the chat models, through its gateway; the harness only drives them.
"""
from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

ModelRole = Literal["explore", "agent_step", "supervisor"]

MAX_NAME_CHARS = 100
# HTTP statuses a provider error may carry that are worth asking again.
RETRYABLE_STATUSES = frozenset({408, 409, 429})
# Dropped connections and timeouts from provider SDKs and their HTTP clients,
# matched by class name so the harness imports no provider SDK. The SDK names
# are shared by the openai, anthropic and groq clients, so any module matches.
TRANSIENT_SDK_ERRORS = frozenset({"APIConnectionError", "APITimeoutError"})
TRANSIENT_HTTP_ERRORS = {
    "httpx": frozenset({"TimeoutException", "NetworkError", "RemoteProtocolError"}),
    "httpcore": frozenset({"TimeoutException", "NetworkError", "RemoteProtocolError"}),
}
CACHE_CONTROL = {"type": "ephemeral"}


@dataclass(frozen=True)
class NamedModel:
    """A chat model and the short name the trace records for it, such as
    ``openai:gpt-5.5``: provider and model id, never a key or a URL."""

    model: BaseChatModel
    name: str

    def __post_init__(self) -> None:
        name = self.name
        if not name or len(name) > MAX_NAME_CHARS or any(c.isspace() for c in name) or "://" in name:
            raise ValueError("a model name is a short provider:model id, with no spaces or URL")


@dataclass(frozen=True)
class ModelChoice:
    """The model for a role, then the models to fall back to, in order.

    ``prompt_cache`` asks for a Claude-style cache breakpoint (a
    ``cache_control`` content block) on the system prompt; the host sets it
    only for models that take one, and marks the conversation itself in the
    payload its model sends.
    """

    primary: NamedModel
    fallbacks: tuple[NamedModel, ...] = ()
    prompt_cache: bool = False

    @property
    def model(self) -> BaseChatModel:
        """The primary model, as ``create_agent`` takes it."""
        return self.primary.model

    @property
    def name(self) -> str:
        return self.primary.name

    def name_of(self, model: Any) -> str | None:
        """The trace name of ``model`` when it is one of this choice's models."""
        for named in (self.primary, *self.fallbacks):
            if named.model is model:
                return named.name
        return None


class Models(Protocol):
    """The models a host offers this run, keyed by role."""

    def for_role(self, role: ModelRole) -> ModelChoice | None:
        """The models for ``role``; None when the host has none for it."""
        ...


@dataclass(frozen=True, kw_only=True)
class Backoff:
    """Waits between retries: ``initial * factor ** retry``, capped at ``maximum``,
    with +-25% jitter when ``jitter``. A factor of 0 waits ``initial`` every time."""

    initial: float = 1.0
    factor: float = 2.0
    maximum: float = 30.0
    jitter: bool = True


DEFAULT_BACKOFF = Backoff()
NO_BACKOFF = Backoff(initial=0.0, factor=0.0, maximum=0.0, jitter=False)


def _is_dropped_call(exc: Exception) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__ in TRANSIENT_SDK_ERRORS:
            return True
        root = cls.__module__.split(".", 1)[0]
        if cls.__name__ in TRANSIENT_HTTP_ERRORS.get(root, ()):
            return True
    return False


def is_transient(exc: Exception) -> bool:
    """Whether asking the same model again may succeed: a rate limit, a server
    error, a timeout or a dropped connection. Anything else, such as a refused
    credential or a request too long for the model, is not retried."""
    if isinstance(exc, ModelError):
        return exc.is_retryable
    if isinstance(exc, (TimeoutError, ConnectionError)) or _is_dropped_call(exc):
        return True
    # Provider SDK errors not mapped to ModelError still carry their HTTP status.
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in RETRYABLE_STATUSES or status >= 500)


def resilience_middleware(
    choice: ModelChoice,
    *,
    max_retries: int = 2,
    backoff: Backoff = DEFAULT_BACKOFF,
    retry_on: Callable[[Exception], bool] = is_transient,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Middleware that retries a failed model call up to ``max_retries`` times
    on ``retry_on`` errors, then tries each fallback model in turn, each with
    its own retries. Only the model call is repeated, never a tool.

    When every model has failed, the last error propagates out of the agent.
    The list goes first in ``create_agent(middleware=...)``; a meter goes after it.
    """
    retry = ModelRetryMiddleware(
        max_retries=max_retries,
        retry_on=retry_on,
        # Raise when retries run out, so the fallback sees the failure.
        on_failure="error",
        backoff_factor=backoff.factor,
        initial_delay=backoff.initial,
        max_delay=backoff.maximum,
        jitter=backoff.jitter,
    )
    if not choice.fallbacks:
        return [retry]
    first, *rest = (named.model for named in choice.fallbacks)
    # The first middleware is the outermost, so each model's attempt runs
    # through the retry inside the fallback.
    return [ModelFallbackMiddleware(first, *rest), retry]


def system_message(text: str, choice: ModelChoice) -> str | SystemMessage:
    """The system prompt as ``create_agent`` takes it: plain text, or with
    ``prompt_cache`` one text block carrying a cache breakpoint, which also
    covers the tools sent before it."""
    if not choice.prompt_cache:
        return text
    return SystemMessage(content=[{"type": "text", "text": text, "cache_control": dict(CACHE_CONTROL)}])


@dataclass(frozen=True)
class Usage:
    """What one agent run used. ``model_name`` is the model that answered the
    last call; ``models_used`` every model that answered one, in first-use order.

    ``cache_read_tokens`` and ``cache_creation_tokens`` are the parts of
    ``input_tokens`` read from and written to the provider's prompt cache, as
    langchain's ``input_token_details`` reports them.
    """

    model_calls: int
    failed_model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    model_name: str | None
    models_used: tuple[str, ...]


def _messages(response: Any) -> Sequence[BaseMessage]:
    if isinstance(response, ExtendedModelResponse):
        response = response.model_response
    if isinstance(response, AIMessage):
        return [response]
    if isinstance(response, ModelResponse):
        return response.result
    return []


def _tokens(message: AIMessage) -> tuple[int, int, int, int]:
    """Input, output, cache-read and cache-creation tokens; anything missing is 0."""
    usage = message.usage_metadata
    if usage is None:
        return 0, 0, 0, 0
    details = usage.get("input_token_details") or {}
    return (
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        int(details.get("cache_read") or 0),
        int(details.get("cache_creation") or 0),
    )


class UsageMeter(AgentMiddleware[Any, Any, Any]):
    """Counts model calls, tool calls and tokens for one agent run; build one per run.

    It goes last in the middleware list, inside retry and fallback, so it sees
    each attempt and the model that made it. Tokens come from ``usage_metadata``
    on the answers; an answer without it counts as zero.
    """

    def __init__(self, choice: ModelChoice) -> None:
        super().__init__()
        self._choice = choice
        self._lock = threading.Lock()
        self._model_calls = 0
        self._failed = 0
        self._tool_calls = 0
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_creation = 0
        self._last: str | None = None
        self._used: list[str] = []

    def usage(self) -> Usage:
        with self._lock:
            return Usage(
                model_calls=self._model_calls,
                failed_model_calls=self._failed,
                tool_calls=self._tool_calls,
                input_tokens=self._input,
                output_tokens=self._output,
                cache_read_tokens=self._cache_read,
                cache_creation_tokens=self._cache_creation,
                model_name=self._last,
                models_used=tuple(self._used),
            )

    def _answered(self, request: ModelRequest[Any], response: Any) -> None:
        name = self._choice.name_of(request.model) or "unknown"
        with self._lock:
            self._model_calls += 1
            for message in _messages(response):
                if isinstance(message, AIMessage):
                    tokens_in, tokens_out, cache_read, cache_creation = _tokens(message)
                    self._input += tokens_in
                    self._output += tokens_out
                    self._cache_read += cache_read
                    self._cache_creation += cache_creation
            self._last = name
            if name not in self._used:
                self._used.append(name)

    def _failed_call(self) -> None:
        with self._lock:
            self._failed += 1

    def _tool_ran(self) -> None:
        with self._lock:
            self._tool_calls += 1

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        try:
            response = handler(request)
        except GraphBubbleUp:
            raise
        except Exception:
            self._failed_call()
            raise
        self._answered(request, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception:
            self._failed_call()
            raise
        self._answered(request, response)
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        self._tool_ran()
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self._tool_ran()
        return await handler(request)
