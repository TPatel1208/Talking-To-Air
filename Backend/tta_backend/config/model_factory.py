"""Per-provider chat model construction.

The single seam where provider choice becomes a constructed LangChain chat
model — agent-construction code calls ``build_chat_model`` and never
imports a concrete provider class.
"""
from __future__ import annotations

from functools import cache
from typing import Any

from tta_backend.config.settings import ConfigurationError, Settings

_PROVIDERS = ("groq", "google")

# Tool metadata key naming a JSON Schema the tool's declaration must use
# verbatim (T64). The Gemini model below sends it as parameters_json_schema
# instead of converting the tool's args_schema.
NATIVE_JSON_SCHEMA = "native_json_schema"


def build_chat_model(
    provider: str,
    model: str,
    settings: Settings,
    *,
    agent_type: str = "unknown",
    fallback_model: str | None = None,
) -> Any:
    """Construct a chat model for ``provider`` + ``model``.

    ``agent_type`` labels this model's latency and token series. It belongs
    here rather than at the call sites' own metrics because this is the only
    place a chat model is constructed, so an agent added later is attributed
    without its author having to remember to ask -- the same argument that
    put the timing callback here. Use the vocabulary
    ``utils.metrics.AGENT_REQUESTS_TOTAL`` already uses ("satellite",
    "ground_sensor"), plus "supervisor", so the two metrics join.

    If ``fallback_model`` is provided and differs from ``model``, wraps the primary
    model in a fallback chain with the constructed fallback model.

    Raises ``ConfigurationError`` for any provider name outside the
    supported set, matching the existing fail-at-boot posture for other
    required runtime configuration.
    """
    # Attached here rather than per-agent because this is the only place a
    # chat model is constructed: an agent added later is timed without its
    # author having to remember to ask for it. See utils.llm_timing for why
    # provider latency needed its own series at all. ``max_retries`` is set
    # here for the same reason -- see settings.llm_max_retries.
    from tta_backend.utils.llm_timing import timing_callbacks

    if provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        primary = ChatGroq(
            model=model,
            api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
            max_retries=settings.llm_max_retries,
            callbacks=timing_callbacks(agent_type),
        )
    elif provider == "google":
        primary = _gemini_chat_class()(
            model=model,
            google_api_key=settings.google_api_key,
            max_retries=settings.llm_max_retries,
            callbacks=timing_callbacks(agent_type),
        )
    else:
        raise ConfigurationError(
            f"Unknown model provider {provider!r}; supported providers are {', '.join(_PROVIDERS)}"
        )

    if fallback_model and fallback_model != model:
        fallback = build_chat_model(
            provider,
            fallback_model,
            settings,
            agent_type=agent_type,
        )
        return primary.with_fallbacks([fallback])

    return primary


@cache
def _gemini_chat_class() -> Any:
    """ChatGoogleGenerativeAI whose bind_tools honours NATIVE_JSON_SCHEMA.

    langchain_google_genai's converter reduces a list of ``anyOf`` items to a
    bare STRING/OBJECT, and its own bind_tools round-trips every tool through
    a dict that cannot carry ``parameters_json_schema`` (T64 Phase 0 F1/F3).
    So tools naming a native schema are sent as one ready-made types.Tool,
    which the request builder passes through unconverted; every other tool is
    converted exactly as the parent does. Built lazily so importing this
    module never imports a provider SDK."""
    from google.genai import types
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai._function_utils import (
        convert_to_genai_function_declarations,
    )

    class GeminiChat(ChatGoogleGenerativeAI):
        def bind_tools(
            self,
            tools: Any,
            tool_config: Any = None,
            *,
            tool_choice: Any = None,
            **kwargs: Any,
        ) -> Any:
            native = [t for t in tools if _native_schema(t) is not None]
            if not native:
                return super().bind_tools(tools, tool_config, tool_choice=tool_choice, **kwargs)
            converted = convert_to_genai_function_declarations(
                [t for t in tools if _native_schema(t) is None]
            )
            # Every function declaration travels in one Tool, as on the stock
            # path; only special tools (search, code execution) stay separate.
            declarations = [fd for t in converted for fd in (t.function_declarations or [])]
            declarations += [
                types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters_json_schema=_native_schema(t),
                )
                for t in native
            ]
            formatted: list[Any] = [t for t in converted if not t.function_declarations]
            formatted.append(types.Tool(function_declarations=declarations))
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            if tool_config:
                kwargs["tool_config"] = tool_config
            return self.bind(tools=formatted, **kwargs)

    return GeminiChat


def _native_schema(tool: Any) -> Any:
    metadata = getattr(tool, "metadata", None)
    return metadata.get(NATIVE_JSON_SCHEMA) if isinstance(metadata, dict) else None


def structured_output(model: Any, schema: Any) -> Any:
    """Bind ``model`` to ``schema`` for structured output.

    Kept at the factory boundary (rather than inline in agent-construction
    code) so a provider-specific structured-output strategy can change here
    without reopening callers.
    """
    return model.with_structured_output(schema)
