"""
services/variable_choice_registry.py
======================================
Backend memory of "which science variable did the model actually choose"
(PRD T25), keyed by the obs_/cube_ handle a retrieval eventually mints --
so a later plot/stat/compare call on that handle inherits the choice
instead of AggregationService.to_dataarray having to guess, or refuse, on
a multi-variable file all over again.

safe_retrieve already knows the requested ``variables`` list synchronously,
but the handle a retrieval will resolve to is only known once
await_retrieval reaches a terminal "ready" status -- so recording is a two-
step handoff: ``record_pending`` at submission time (keyed by job_handle),
``finalize`` once the job's handle is known.

In-memory, per-process, TTL-bounded -- mirrors the existing ArtifactStore/
GeocodingService caches (services/artifact_store.py, utils/plotting.py).
Nothing here is a system of record: the same handle is round-trippable
through the MCP's own export/rematerialize cycle at any time, so losing a
recorded choice on a process restart only means a later call falls through
to to_dataarray's next resolution tier (single-data-var file, or a
candidate-listing error) -- never a wrong answer.
"""
from __future__ import annotations

import time

_TTL_SECONDS = 24 * 60 * 60

_pending: dict[str, tuple[str, float]] = {}
_choices: dict[str, tuple[str, float]] = {}


def record_pending(job_handle: str, variable: str | None) -> None:
    """Record ``variable`` as the single unambiguous choice requested for
    ``job_handle``, once retrieve_subset submits it -- a no-op when
    ``variable`` is falsy (0 or >1 variables requested is not a choice to
    remember)."""
    if not job_handle or not variable:
        return
    _pending[job_handle] = (variable, time.time() + _TTL_SECONDS)


def finalize(job_handle: str, handle: str | None) -> None:
    """Promote a pending job's recorded choice to ``handle`` once the job
    reaches "ready" with that handle. A no-op when nothing was pending for
    ``job_handle`` (ambiguous/no-variable submission) or ``handle`` is
    falsy."""
    entry = _pending.pop(job_handle, None)
    if entry is None or not handle:
        return
    variable, expires_at = entry
    _choices[handle] = (variable, expires_at)


def get(handle: str) -> str | None:
    """Return the recorded variable choice for ``handle``, or None if none
    was recorded or it has expired."""
    entry = _choices.get(handle)
    if entry is None:
        return None
    variable, expires_at = entry
    if expires_at <= time.time():
        _choices.pop(handle, None)
        return None
    return variable
