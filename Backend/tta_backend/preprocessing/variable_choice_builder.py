"""
preprocessing/variable_choice_builder.py
=========================================
Emit the variable picker on the active SSE stream. Building the picker is a
domain concern (domain/aggregation/variable_choice.py); dispatch fills each
option's ``prompt`` with ``fill_prompts`` from the original request it holds
(services/subagent_dispatch.py).
"""
from __future__ import annotations

import logging

import xarray as xr

from tta_backend.domain.aggregation.variable_choice import build_variable_choice, fill_prompts
from tta_backend.domain.aggregation.variable_resolver import Resolution

__all__ = ["build_variable_choice", "emit_variable_choice_payload", "fill_prompts"]

logger = logging.getLogger(__name__)


def emit_variable_choice_payload(resolution: Resolution, ds: xr.Dataset) -> None:
    """Build the picker from ``resolution``/``ds`` and emit it out-of-band on
    the active SSE stream (utils.streaming.emit_variable_choice) -- the seam a
    satellite tool calls when it catches a ``VariableChoiceRequired`` short-
    circuit. The candidate prompts are left blank here; run_satellite fills
    them from the original request once it captures the event.

    Best-effort: a build/emit failure is logged, never re-raised, since the
    caller still returns the compact, size-bounded ``MCPToolError`` that keeps the
    turn honest even if the richer picker couldn't be built."""
    from tta_backend.utils.streaming import emit_variable_choice

    try:
        choice = build_variable_choice(resolution, ds)
        emit_variable_choice(choice.model_dump())
    except Exception:
        logger.warning("emit_variable_choice_failed", exc_info=True)
