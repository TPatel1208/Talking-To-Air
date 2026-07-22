"""
services/jobs_service.py
==========================
Backend composite behind the jobs panel (PRD T05): the panel's list model is
built here, not in the endpoint, so the endpoint stays a thin HTTP wrapper.
Composes the earthdata-retrieval MCP's ``list_workspace`` (job handles plus
their static metadata) with a ``get_retrieval_status`` fan-out per handle
(status, progress, phase, message, obs_handle) into one durable job list —
populated from the backend on every page load, never from chat history.
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from earthdata_mcp.results import CATEGORY_NOT_FOUND, MCPToolError, parse_tool_result
from services.retrieval_composites import TERMINAL_STATUSES, annotate_paused

# Cap the get_retrieval_status fan-out: a workspace accumulates jobs over its
# lifetime (hundreds in practice) and the panel refetches the whole list on
# every load, so an unbounded gather would fire one MCP round-trip per job all
# at once. 8 keeps the panel responsive without stampeding the MCP/worker.
_STATUS_FANOUT_LIMIT = 8

# Statuses whose row is immutable, so it's cached and never re-fetched.
# TERMINAL_STATUSES includes "not_found" — a handle list_workspace still
# lists but get_retrieval_status no longer has a record of (an old evicted
# job). It is a dead handle that never reappears, and on a long-lived
# workspace these dominate: the live repro had 134 not_found handles vs 46
# real jobs, so re-polling them every 15s *was* the fan-out (a 180-job
# workspace = ~180 MCP round-trips/poll). parse_tool_result surfaces
# not_found as a structured passthrough ({status: "not_found"}), so it
# arrives here as an ordinary status string.
_CACHEABLE_STATUSES = TERMINAL_STATUSES

# The panel refetches the whole workspace on the frontend's live poll
# (hooks/useJobs.js). Without this cache, list_jobs re-issued
# get_retrieval_status for every finished-or-dead handle on every poll, for
# data that cannot have changed. Cache immutable rows process-side and serve
# them with no MCP call; only genuinely non-terminal jobs stay on the per-poll
# fan-out. Keyed by job_handle (workspace-unique, and only ever read back for a
# handle the caller's own list_workspace returned, so no cross-workspace
# leakage). Process-lifetime state, like the T31 credential injector.
_TERMINAL_STATUS_CACHE: dict[str, dict[str, Any]] = {}


def clear_terminal_status_cache() -> None:
    """Drop the cached terminal statuses. A test seam (the cache is
    process-global) and a hook for a future workspace-reset if one is needed."""
    _TERMINAL_STATUS_CACHE.clear()


async def list_jobs(tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    """Return the caller's workspace jobs, each merged with its live status.

    ``list_workspace`` returns every handle in the workspace as
    ``{handles: [{handle, type, created_at, summary}]}`` — filtered here to
    ``type == "job"`` and mapped to the field names the rest of this
    composite (and the frontend) expect. Active (non-terminal) jobs sort
    first, newest-first within each group, so a researcher sees what's still
    running before what's already finished.

    The per-handle ``get_retrieval_status`` fan-out is bounded and fault-
    isolated: a single job whose status can't be read degrades to a
    ``status: "error"`` row rather than failing the whole panel, so one bad
    handle never blanks every healthy sibling.
    """
    workspace_raw = await tools["list_workspace"].ainvoke({})
    workspace = parse_tool_result(workspace_raw)
    entries = [
        {"job_handle": handle["handle"], "created_at": handle.get("created_at"), **(handle.get("summary") or {})}
        for handle in workspace.get("handles", [])
        if handle.get("type") == "job"
    ]

    status_tool = tools["get_retrieval_status"]
    semaphore = asyncio.Semaphore(_STATUS_FANOUT_LIMIT)

    async def status_for(entry: dict[str, Any]) -> dict[str, Any]:
        handle = entry["job_handle"]
        # A job already known terminal never changes — serve it from the cache
        # and skip the MCP round-trip entirely (see _TERMINAL_STATUS_CACHE).
        cached = _TERMINAL_STATUS_CACHE.get(handle)
        if cached is not None:
            return cached
        async with semaphore:
            try:
                # annotate_paused: a Harmony-auto-paused job reports as
                # "running" forever (the MCP maps paused -> running); the
                # panel must show it as paused-with-guidance, not an
                # eternally processing row.
                status = annotate_paused(parse_tool_result(
                    await status_tool.ainvoke({"job_handle": handle})
                ))
            # Broad by design: this is the fault-isolation boundary the
            # docstring promises. MCPToolError covers responses the adapter
            # classifies; anything else (a transport error, a malformed
            # response parse_tool_result chokes on) must degrade the same
            # single row rather than reaching asyncio.gather and cancelling
            # every other in-flight status call. Never cached — a status read
            # that failed transiently must be retried on the next poll.
            except Exception as exc:
                return {"status": "error", "message": str(exc)}
        # Cache only immutable outcomes (terminal or dead handle);
        # paused/running/pending stay on the fan-out so a resume or completion
        # is still picked up.
        if status.get("status") in _CACHEABLE_STATUSES:
            _TERMINAL_STATUS_CACHE[handle] = status
        return status

    statuses = await asyncio.gather(*(status_for(entry) for entry in entries))

    jobs = [
        {**entry, **status}
        for entry, status in zip(entries, statuses)
    ]
    jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    jobs.sort(key=lambda job: job.get("status") in TERMINAL_STATUSES)
    return jobs


async def cancel_job(job_handle: str, tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Proxy the MCP's cancel tool — hidden from the agent, exposed to the UI.

    A cancel targeting a handle the workspace never had is a valid answer, not
    an error (T47 finding #3): the MCP models a missing job as data. Whichever
    way it signals the miss — a ``not_found`` error envelope (which
    ``parse_tool_result`` re-raises) or a structured ``{status: "not_found"}``
    passthrough — the endpoint answers 200 with an explicit ``not_found``
    status the panel can render honestly and refetch from, never a generic 404
    error or a ``cancelled``-looking body.
    """
    raw = await tools["cancel_retrieval"].ainvoke({"job_handle": job_handle})
    try:
        return parse_tool_result(raw)
    except MCPToolError as exc:
        if exc.category == CATEGORY_NOT_FOUND:
            return {"job_handle": job_handle, "status": "not_found"}
        raise
