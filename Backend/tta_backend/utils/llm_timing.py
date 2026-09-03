"""Record per-call timing and provider token metrics for LLM requests.

The shared model factory instruments each provider request as an ``llm_call``
span, including failed calls. This fills the observability gap between the
existing data-processing phases and the end of a turn.

The timing is particularly useful for detecting provider-side retry/backoff:
ChatGoogleGenerativeAI can retry inside the provider SDK, below LangChain's
``on_retry`` callback, so those delays may otherwise appear only as an
unexplained increase in LLM latency.

The callback also records ``input_tokens``, ``output_tokens``, and
``input_token_details["cache_read"]``. ``cache_read`` shows how much of the
repeated system-prompt/tool-schema prefix was served from the provider's
prompt cache, allowing cache effectiveness to be measured directly.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from tta_backend.utils.phase_timing import record_phase

# A run whose end callback never arrives (a turn cancelled by the T38
# whole-turn deadline is the realistic case) would otherwise pin its entry
# forever. Insertion order is preserved, so the oldest entries are the ones
# evicted -- and an entry old enough to be evicted under this bound belongs
# to a call nobody is waiting on any more.
_MAX_TRACKED_RUNS = 256


def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """Best-effort model id for the log line.

    Read from whichever of the three places the provider integration
    populated -- they disagree between providers and across versions, and a
    missing model name must degrade to a usable log line rather than an
    exception on the hot path.
    """
    params = kwargs.get("invocation_params") or {}
    metadata = kwargs.get("metadata") or {}
    for candidate in (
        params.get("model"),
        params.get("model_name"),
        metadata.get("ls_model_name"),
        (serialized or {}).get("kwargs", {}).get("model"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def _usage_metadata(response: Any) -> dict[str, Any] | None:
    """The provider's usage block for a finished call, or None.

    Read from whichever place the provider integration populated, for the
    same reason :func:`_model_name` does: the integrations disagree with each
    other and across versions. The standard location is the ``AIMessage`` on
    the first generation; ``llm_output`` is the older shape some integrations
    still fill instead.
    """
    for batch in getattr(response, "generations", None) or ():
        for generation in batch or ():
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                return usage
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for key in ("usage_metadata", "token_usage"):
            usage = llm_output.get(key)
            if isinstance(usage, dict) and usage:
                return usage
    return None


def _usage_context(response: Any) -> dict[str, int]:
    """Token counts for a finished call, or ``{}`` if the provider reported none.

    A missing count is omitted, never recorded as a zero. This is the same
    rule ``_end`` applies to a span with no matching start, and it matters
    most for the field this exists to measure: a streaming response whose
    usage block never arrived would otherwise contribute ``cache_read=0``,
    and a cache that is working perfectly would read as a 0% hit rate. An
    absent measurement and a measured zero must not be the same number.

    Never raises -- a provider that changes this payload's shape must cost a
    telemetry field, not a turn.
    """
    try:
        usage = _usage_metadata(response)
        if usage is None:
            return {}
        context: dict[str, int] = {}
        for source_key, context_key in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ):
            value = usage.get(source_key)
            if isinstance(value, int) and not isinstance(value, bool):
                context[context_key] = value
        # cache_read is a *breakdown* of input_tokens, not an addition to it
        # (LangChain's UsageMetadata contract), so the hit rate for a call is
        # cache_read_tokens / input_tokens.
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            cache_read = details.get("cache_read")
            if isinstance(cache_read, int) and not isinstance(cache_read, bool):
                context["cache_read_tokens"] = cache_read
        return context
    except Exception:  # pragma: no cover -- defensive; see docstring
        return {}


# Context key -> the ``kind`` label it counts under. Only these three keys are
# forwarded to the counters; any other context a call site adds stays on the
# log line alone.
_TOKEN_KINDS = (
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("cache_read_tokens", "cache_read"),
)


def _record_token_counters(model: str, agent_type: str, context: dict[str, Any]) -> None:
    """Add whichever token counts ``context`` carries to the counters.

    Never raises, for the reason the rest of this module does not: it runs on
    the hot path, and a telemetry failure must not be why a turn fails.
    """
    try:
        from tta_backend.utils.metrics import record_llm_tokens

        for context_key, kind in _TOKEN_KINDS:
            tokens = context.get(context_key)
            if isinstance(tokens, int):
                record_llm_tokens(model, agent_type, kind, tokens)
    except Exception:  # pragma: no cover -- defensive; see docstring
        pass


class LlmTimingCallback(BaseCallbackHandler):
    """Times every provider request to the ``llm_call`` phase.

    One instance is shared by every call a model makes, including concurrent
    ones, so in-flight state is keyed by LangChain's per-run ``run_id`` and
    guarded by a lock -- the sub-agent dispatch path runs several models at
    once, and a single start timestamp would attribute one call's span to
    another.

    Sync (``BaseCallbackHandler``) rather than async on purpose: LangChain
    dispatches sync handlers from both its sync and async paths, so one
    class covers every call site. The executor hop that costs is on the
    order of milliseconds against spans this exists to measure in tens of
    seconds.

    ``agent_type`` is fixed at construction because that is where the answer
    is actually known: a chat model is built once per agent and every call
    through it belongs to that agent. Deriving it at call time instead would
    mean a contextvar the sub-agent dispatch path would have to set and the
    supervisor would have to unset around every delegation -- state that can
    be wrong, standing in for something that never changes.
    """

    def __init__(self, agent_type: str = "unknown") -> None:
        self._agent_type = agent_type
        self._started: dict[UUID, tuple[float, str]] = {}
        self._retries: dict[UUID, int] = {}
        self._lock = threading.Lock()

    def _begin(self, run_id: UUID, serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> None:
        model = _model_name(serialized, kwargs)
        with self._lock:
            self._started[run_id] = (time.monotonic(), model)
            while len(self._started) > _MAX_TRACKED_RUNS:
                self._started.pop(next(iter(self._started)), None)

    def _end(self, run_id: UUID, outcome: str, **context: Any) -> None:
        with self._lock:
            entry = self._started.pop(run_id, None)
            retries = self._retries.pop(run_id, 0)
        if entry is None:
            # No matching start: a handler attached mid-flight, or an end
            # delivered twice. Recording a span we never measured would put
            # a fabricated number in the histogram.
            return
        started, model = entry
        record_phase(
            "llm_call",
            time.monotonic() - started,
            model=model,
            agent_type=self._agent_type,
            outcome=outcome,
            retries=retries,
            **context,
        )
        # Counters after the log line, and keyed by the same ``model`` the span
        # was charged to, so a call's duration and its token cost are always
        # attributed to the same series -- the sub-agents run different models
        # concurrently, and split attribution would make cost per model a guess.
        _record_token_counters(model, self._agent_type, context)

    # on_chat_model_start, not on_llm_start, is what a chat model raises --
    # both are implemented because build_chat_model is not the only thing
    # that could ever be handed this callback.
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._begin(run_id, serialized, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._begin(run_id, serialized, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        # Usage is read here rather than in _end because this is the only
        # callback that receives the provider's response at all: on_llm_error
        # has no usage block to read, and a failed call is charged its span
        # with no token context.
        self._end(run_id, "success", **_usage_context(response))

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._end(run_id, "error", error_type=type(error).__name__)

    def on_retry(self, retry_state, *, run_id, **kwargs) -> None:
        """Count a LangChain-driven retry and log the sleep it is about to take.

        ``idle_for`` is tenacity's cumulative backoff so far, which is the
        number that answers "is this turn slow because it is sleeping".
        Retries inside the provider SDK never reach here -- see the module
        docstring; their absence next to a long ``llm_call`` is the finding,
        not a gap in this instrumentation.
        """
        try:
            attempt = getattr(retry_state, "attempt_number", 0)
            idle_for = getattr(retry_state, "idle_for", 0.0)
            with self._lock:
                self._retries[run_id] = attempt
                while len(self._retries) > _MAX_TRACKED_RUNS:
                    self._retries.pop(next(iter(self._retries)), None)
            record_phase(
                "llm_retry_sleep",
                float(idle_for or 0.0),
                attempt=attempt,
                run_id=str(run_id),
            )
        except Exception:  # pragma: no cover -- telemetry must not break a turn
            pass


def timing_callbacks(agent_type: str = "unknown") -> list[BaseCallbackHandler]:
    """The callback list to attach to a newly constructed chat model.

    One instance per built model, not the process-wide singleton this used to
    return. The singleton was correct while the only thing recorded was a
    duration keyed by run_id, but ``agent_type`` is a property of the *model*,
    and the only place it is known without inventing call-time state is the
    build. Three agents means three bounded in-flight dicts instead of one,
    which is the whole cost of making per-agent spend attributable.

    ``agent_type`` defaults rather than being required so this stays usable by
    anything that builds a model outside the three agents. A series appearing
    under ``agent_type="unknown"`` means exactly that happened, and is the
    signal to pass a real one.
    """
    return [LlmTimingCallback(agent_type)]
