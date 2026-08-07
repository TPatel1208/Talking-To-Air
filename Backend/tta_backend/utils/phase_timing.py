"""Per-phase wall-clock timing for the retrieval/visualization pipeline (T51).

The only durations this backend recorded were per-HTTP-request and whole-turn
elapsed, so a slow turn could never be attributed to a phase and every
performance decision was a guess. :func:`phase_timer` closes that gap: it
records a phase's *own* span to the ``pipeline_phase_duration_seconds``
histogram and emits a structured ``phase_timing`` log event carrying the
phase, its duration, the thread it ran on, and whatever cheap size context
the call site can supply (cells in/out, bytes read).

Deliberately a separate channel from ``emit_status``/``stage_reached``
narration (:mod:`config.workflow_stages`). That vocabulary is closed on
purpose -- the frontend's workflow strip and the eval's stage-sequence
assertions key off it -- and its events carry cumulative elapsed since turn
start, which is the wrong shape for a phase duration. Phase timing is
developer telemetry; narration is a user-facing contract.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


def record_phase(phase: str, duration_seconds: float, /, **context: Any) -> None:
    """Record an already-measured span for ``phase``.

    The context-manager form below cannot express a span whose start and end
    arrive as two *separate* events -- a LangChain ``on_llm_start`` /
    ``on_llm_end`` callback pair, or the gap between two chunks of an
    ``astream``. Those call sites hold the duration and nothing else, so this
    is the seam they record through, and :meth:`_PhaseTimer._finish` routes
    here too rather than duplicating the log shape.

    ``phase`` and ``duration_seconds`` are positional-only so a context key
    may be named either without colliding with the parameter.

    Never raises: this is on the hot path, and a telemetry failure must not
    be the reason a turn fails. Every step (the metrics backend, the log
    call) is inside the guard.
    """
    try:
        from tta_backend.utils.metrics import observe_phase_duration

        observe_phase_duration(phase, duration_seconds)
        extra = {
            "_event": "phase_timing",
            "_phase": phase,
            "_duration_seconds": round(duration_seconds, 6),
            "_thread_id": threading.get_ident(),
        }
        extra.update({f"_{key}": value for key, value in context.items()})
        logger.info("phase_timing", extra=extra)
    except Exception:  # pragma: no cover -- defensive; see docstring
        pass


class _PhaseTimer:
    """Times one phase. Usable as either a sync or an async context manager --
    the pipeline crosses ``asyncio.to_thread`` boundaries, and a monotonic
    wall clock is the right measurement on both sides of one.

    Yields its mutable context dict, so a call site can record sizes it only
    learns *inside* the block::

        with phase_timer("crop", cells_in=da.size) as ctx:
            cropped = ...
            ctx["cells_out"] = cropped.size
    """

    def __init__(self, phase: str, context: dict[str, Any]):
        self.phase = phase
        self.context = context
        self._started: float | None = None

    def _start(self) -> dict[str, Any]:
        self._started = time.monotonic()
        return self.context

    def _finish(self) -> None:
        if self._started is None:  # pragma: no cover -- defensive
            return
        record_phase(self.phase, time.monotonic() - self._started, **self.context)

    def __enter__(self) -> dict[str, Any]:
        return self._start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A phase that raised still took time, and that time is exactly what a
        # "why was this turn slow" investigation needs -- record it, then let
        # the exception propagate untouched (False).
        self._finish()
        return False

    async def __aenter__(self) -> dict[str, Any]:
        return self._start()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._finish()
        return False


def phase_timer(phase: str, **context: Any) -> _PhaseTimer:
    """Time ``phase``, recording to the histogram and the ``phase_timing`` log.

    ``phase`` should be one of :data:`utils.metrics.PIPELINE_PHASES` (those
    labels are pre-declared at zero so the series exists before any traffic).
    ``context`` is cheap size/identity detail -- cells in, cells out, bytes
    read, member count -- logged with a leading underscore so the JSON
    formatter lifts it to a top-level field.
    """
    return _PhaseTimer(phase, dict(context))
