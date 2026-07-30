"""
services/scope_registry.py
============================
Backend memory of "what scope did the researcher actually request" (PRD T46),
keyed by the obs_/cube_ handle a retrieval eventually mints — so a later
plot/stat call on that handle can stamp the *requested* scope alongside the
*delivered* one and disclose any silent substitution (a single-day request
served by a monthly mean, a time range clamped to coverage).

Exactly the two-step handoff variable_choice_registry uses: safe_retrieve /
point_timeseries know the requested ``{location, time_range}`` synchronously,
but the handle a retrieval resolves to is only known once await_retrieval
reaches a terminal "ready" status — so ``record_pending`` at submission time
(keyed by job_handle), ``finalize`` once the job's handle is known.

In-memory, per-process, TTL-bounded. Nothing here is a system of record:
losing a recorded scope on a process restart only means a later plot omits the
disclosure note (the Metadata tab still carries the delivered scope) — never a
wrong answer.
"""
from __future__ import annotations

import time

_TTL_SECONDS = 24 * 60 * 60

_pending: dict[str, tuple[dict, float]] = {}
_scopes: dict[str, tuple[dict, float]] = {}


def record_pending(job_handle: str, requested_scope: dict | None) -> None:
    """Record the requested scope for ``job_handle`` once a retrieval submits
    it — a no-op when ``job_handle`` is falsy or ``requested_scope`` carries no
    usable fact (nothing to disclose against)."""
    if not job_handle or not requested_scope:
        return
    scope = {k: v for k, v in requested_scope.items() if v}
    if not scope:
        return
    _pending[job_handle] = (scope, time.time() + _TTL_SECONDS)


def finalize(job_handle: str, handle: str | None) -> None:
    """Promote a pending job's recorded scope to ``handle`` once the job
    reaches "ready" with that handle. A no-op when nothing was pending for
    ``job_handle`` or ``handle`` is falsy."""
    entry = _pending.pop(job_handle, None)
    if entry is None or not handle:
        return
    scope, expires_at = entry
    _scopes[handle] = (scope, expires_at)


def get(handle: str) -> dict | None:
    """Return the recorded requested scope for ``handle``, or None if none was
    recorded or it has expired."""
    entry = _scopes.get(handle)
    if entry is None:
        return None
    scope, expires_at = entry
    if expires_at <= time.time():
        _scopes.pop(handle, None)
        return None
    return scope
