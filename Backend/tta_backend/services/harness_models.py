"""The app's models for the agent harness, built through the model factory.

Each role maps to a setting pair:

- ``explore`` and ``supervisor``: ``SUPERVISOR_MODEL_PROVIDER`` and ``LLM_MODEL``, the stronger tier.
- ``agent_step``: ``EARTHDATA_AGENT_PROVIDER`` and ``EARTHDATA_AGENT_MODEL``, the cheaper tier.

No fallback model is configured today, so no choice has fallbacks.

The harness retries model calls itself, so these models are built with the SDK's own retries off.

Prompt caching is on for portkey Claude models only. The harness marks the system prompt. The
conversation is marked here, in the payload after langchain's conversion, since langchain_openai drops
``cache_control`` from assistant and tool messages: the final message's last content block gets a
breakpoint. With the top-level ``cache_control`` the factory already sends, a request holds three
breakpoints, within Anthropic's four.
"""
from __future__ import annotations

import dataclasses
from functools import cache
from typing import Any

from tta_backend.config.model_factory import _auto_cache_body, _portkey_chat_class, _portkey_headers, build_chat_model
from tta_backend.harness.models import CACHE_CONTROL, ModelChoice, ModelRole, NamedModel
from tta_backend.infrastructure.settings import Settings

# Role -> (agent_type label for the model's metrics, provider setting, model setting).
_ROLES: dict[ModelRole, tuple[str, str, str]] = {
    "explore": ("supervisor", "supervisor_model_provider", "llm_model"),
    "supervisor": ("supervisor", "supervisor_model_provider", "llm_model"),
    "agent_step": ("satellite", "earthdata_agent_provider", "earthdata_agent_model"),
}


def takes_cache_breakpoints(provider: str, model: str) -> bool:
    """Whether the model takes Claude's ``cache_control`` breakpoints; others cache on their own."""
    return provider == "portkey" and "claude" in model.lower()


def mark_final_message(payload: dict[str, Any]) -> None:
    """Put a cache breakpoint on the last content block of the payload's final message.

    String content becomes one text block. A message with no text to mark, such
    as an assistant turn holding only tool calls, is left as it is. Blocks are
    copied, never changed in place, since they may be the langchain message's own.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return
    final = messages[-1]
    content = final.get("content")
    if isinstance(content, str):
        if content.strip():
            final["content"] = [{"type": "text", "text": content, "cache_control": dict(CACHE_CONTROL)}]
        return
    if not isinstance(content, list):
        return
    last = next((i for i in range(len(content) - 1, -1, -1) if isinstance(content[i], dict)), None)
    if last is None:
        return
    block = content[last]
    if block.get("type") == "text" and not str(block.get("text") or "").strip():
        return
    final["content"] = [*content[:last], {**block, "cache_control": dict(CACHE_CONTROL)}, *content[last + 1:]]


@cache
def _cached_portkey_class() -> Any:
    class CachedPortkeyChat(_portkey_chat_class()):  # type: ignore[misc]
        """The portkey chat model, with a cache breakpoint on the final message of each request."""

        def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
            payload: dict[str, Any] = super()._get_request_payload(input_, stop=stop, **kwargs)
            mark_final_message(payload)
            return payload

    return CachedPortkeyChat


def build_cached_chat_model(model: str, settings: Settings, *, agent_type: str) -> Any:
    """A portkey model built with the arguments ``build_chat_model`` uses, marking the conversation."""
    from pydantic import SecretStr

    from tta_backend.utils.llm_timing import timing_callbacks

    return _cached_portkey_class()(
        model=model,
        api_key=SecretStr("unused"),
        base_url=settings.portkey_base_url,
        default_headers=_portkey_headers(settings, agent_type),
        extra_body=_auto_cache_body(model),
        max_retries=settings.llm_max_retries,
        callbacks=timing_callbacks(agent_type),
    )


class AppModels:
    """The ``Models`` the harness is given in the app; each role's model is built once, on first use."""

    def __init__(self, settings: Settings) -> None:
        self._settings = dataclasses.replace(settings, llm_max_retries=0)
        self._choices: dict[ModelRole, ModelChoice] = {}

    def for_role(self, role: ModelRole) -> ModelChoice | None:
        choice = self._choices.get(role)
        if choice is None:
            spec = _ROLES.get(role)
            if spec is None:
                return None
            agent_type, provider_setting, model_setting = spec
            provider = getattr(self._settings, provider_setting)
            model = getattr(self._settings, model_setting)
            cached = takes_cache_breakpoints(provider, model)
            if cached:
                chat = build_cached_chat_model(model, self._settings, agent_type=agent_type)
            else:
                chat = build_chat_model(provider, model, self._settings, agent_type=agent_type)
            choice = self._choices[role] = ModelChoice(
                primary=NamedModel(model=chat, name=f"{provider}:{model}"), prompt_cache=cached,
            )
        return choice
