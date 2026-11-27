"""Per-provider chat model construction.

The single seam where provider choice becomes a constructed LangChain chat
model — callers use ``build_chat_model`` and never import a concrete
provider class.
"""
from __future__ import annotations

from functools import cache
from typing import Any

from tta_backend.infrastructure.settings import ConfigurationError, Settings

_PROVIDERS = ("groq", "google", "portkey")


def build_chat_model(
    provider: str, model: str, settings: Settings, *, agent_type: str = "unknown"
) -> Any:
    """Construct a chat model for ``provider`` + ``model``.

    ``agent_type`` labels this model's latency and token series. It belongs
    here rather than at the call sites' own metrics because this is the only
    place a chat model is constructed, so a role added later is attributed
    without its author having to remember to ask -- the same argument that
    put the timing callback here.

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

        return ChatGroq(
            model=model,
            api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
            max_retries=settings.llm_max_retries,
            callbacks=timing_callbacks(agent_type),
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            max_retries=settings.llm_max_retries,
            callbacks=timing_callbacks(agent_type),
        )
    if provider == "portkey":
        from pydantic import SecretStr

        return _portkey_chat_class()(
            model=model,
            # The real key goes in a header; the client just needs a value.
            api_key=SecretStr("unused"),
            base_url=settings.portkey_base_url,
            default_headers=_portkey_headers(settings, agent_type),
            extra_body=_auto_cache_body(model),
            max_retries=settings.llm_max_retries,
            callbacks=timing_callbacks(agent_type),
        )
    raise ConfigurationError(
        f"Unknown model provider {provider!r}; supported providers are {', '.join(_PROVIDERS)}"
    )


@cache
def _portkey_chat_class() -> Any:
    """The chat class portkey models are built with. Imported lazily so
    importing this module never imports a provider SDK."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


def _portkey_headers(settings: Settings, agent_type: str) -> dict[str, str]:
    """The x-portkey-* headers. Strict compliance is off so provider-specific
    fields (e.g. Gemini's thought_signature) come through."""
    from portkey_ai import createHeaders

    headers: dict[str, str] = createHeaders(  # type: ignore[no-untyped-call]
        api_key=settings.portkey_api_key,
        strict_open_ai_compliance=False,
        config=settings.portkey_config or None,
        metadata={"agent_type": agent_type},
    )
    return headers


def _auto_cache_body(model: str) -> dict[str, Any] | None:
    """Asks Claude to cache up to the latest message on every call."""
    return {"cache_control": {"type": "ephemeral"}} if "claude" in model.lower() else None
