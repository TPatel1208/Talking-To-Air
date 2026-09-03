"""Per-provider chat model construction.

The single seam where provider choice becomes a constructed LangChain chat
model — agent-construction code calls ``build_chat_model`` and never
imports a concrete provider class.
"""
from __future__ import annotations

from typing import Any

from tta_backend.config.settings import ConfigurationError, Settings

_PROVIDERS = ("groq", "google")


def build_chat_model(
    provider: str, model: str, settings: Settings, *, agent_type: str = "unknown"
) -> Any:
    """Construct a chat model for ``provider`` + ``model``.

    ``agent_type`` labels this model's latency and token series. It belongs
    here rather than at the call sites' own metrics because this is the only
    place a chat model is constructed, so an agent added later is attributed
    without its author having to remember to ask -- the same argument that
    put the timing callback here. Use the vocabulary
    ``utils.metrics.AGENT_REQUESTS_TOTAL`` already uses ("satellite",
    "ground_sensor"), plus "supervisor", so the two metrics join.

    Raises ``ConfigurationError`` for any provider name outside the
    supported set, matching the existing fail-at-boot posture for other
    required runtime configuration.
    """
    # Attached here rather than per-agent because this is the only place a
    # chat model is constructed: an agent added later is timed without its
    # author having to remember to ask for it. See utils.llm_timing for why
    # provider latency needed its own series at all.
    from tta_backend.utils.llm_timing import timing_callbacks

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            groq_api_key=settings.groq_api_key,
            callbacks=timing_callbacks(agent_type),
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            callbacks=timing_callbacks(agent_type),
        )
    raise ConfigurationError(
        f"Unknown model provider {provider!r}; supported providers are {', '.join(_PROVIDERS)}"
    )


def structured_output(model: Any, schema: Any) -> Any:
    """Bind ``model`` to ``schema`` for structured output.

    Kept at the factory boundary (rather than inline in agent-construction
    code) so a provider-specific structured-output strategy can change here
    without reopening callers.
    """
    return model.with_structured_output(schema)
