"""Per-provider chat model construction.

The single seam where provider choice becomes a constructed LangChain chat
model — agent-construction code calls ``build_chat_model`` and never
imports a concrete provider class.
"""
from __future__ import annotations

from typing import Any

from config.settings import ConfigurationError, Settings

_PROVIDERS = ("groq", "google")


def build_chat_model(provider: str, model: str, settings: Settings) -> Any:
    """Construct a chat model for ``provider`` + ``model``.

    Raises ``ConfigurationError`` for any provider name outside the
    supported set, matching the existing fail-at-boot posture for other
    required runtime configuration.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, groq_api_key=settings.groq_api_key)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key)
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
