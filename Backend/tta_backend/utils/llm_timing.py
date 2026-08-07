"""Per-call wall-clock timing for provider LLM requests.

A 2026-08-07 trace of a 372s "plot TEMPO NO2 over North America" turn
attributed 171s to the Harmony retrieval wait and 61s to the data pipeline
(open/aggregate/render), and left **107s -- 29% of the turn -- attributable
to nothing at all**: the last thing logged was the ``render`` phase, then
silence but for health checks, then the turn ended. T51's phase vocabulary
is entirely data work, so a slow model call had no series to land in.

This closes that gap at the one seam every chat model is built through
(:func:`config.model_factory.build_chat_model`), recording each provider
request to the ``llm_call`` phase.

Two things make the measurement worth more than it looks:

* ``ChatGoogleGenerativeAI`` defaults to ``max_retries=6``. Those retries
  happen *below* the LangChain seam, inside the provider SDK, so they raise
  no ``on_retry`` and emit no log -- a rate-limited call is simply one
  ``llm_call`` span an order of magnitude longer than its neighbours. That
  is the signature to look for, and this is what makes it visible.
* ``on_retry`` is instrumented anyway, for the retries LangChain *does*
  drive itself. Silence there alongside a long span is itself the evidence
  that the wait was the provider SDK's.

Timing is recorded for failed calls too. A call that spent 90s retrying and
then raised is precisely what a "why was that turn slow" investigation
needs, and charging only successes would hide the worst cases.
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
    """

    def __init__(self) -> None:
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
            outcome=outcome,
            retries=retries,
            **context,
        )

    # on_chat_model_start, not on_llm_start, is what a chat model raises --
    # both are implemented because build_chat_model is not the only thing
    # that could ever be handed this callback.
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._begin(run_id, serialized, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._begin(run_id, serialized, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._end(run_id, "success")

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


# Shared by every model build: the histogram is process-wide anyway, and a
# per-model instance would multiply the in-flight bookkeeping for no gain.
_CALLBACK = LlmTimingCallback()


def timing_callbacks() -> list[BaseCallbackHandler]:
    """The callback list to attach to a newly constructed chat model."""
    return [_CALLBACK]
