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

from earthdata_mcp.results import parse_tool_result
from services.retrieval_composites import TERMINAL_STATUSES

# Cap the get_retrieval_status fan-out: a workspace accumulates jobs over its
# lifetime (hundreds in practice) and the panel refetches the whole list on
# every load, so an unbounded gather would fire one MCP round-trip per job all
# at once. 8 keeps the panel responsive without stampeding the MCP/worker.
_STATUS_FANOUT_LIMIT = 8


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
        async with semaphore:
            try:
                return parse_tool_result(
                    await status_tool.ainvoke({"job_handle": entry["job_handle"]})
                )
            # Broad by design: this is the fault-isolation boundary the
            # docstring promises. MCPToolError covers responses the adapter
            # classifies; anything else (a transport error, a malformed
            # response parse_tool_result chokes on) must degrade the same
            # single row rather than reaching asyncio.gather and cancelling
            # every other in-flight status call.
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

    statuses = await asyncio.gather(*(status_for(entry) for entry in entries))

    jobs = [
        {**entry, **status}
        for entry, status in zip(entries, statuses)
    ]
    jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    jobs.sort(key=lambda job: job.get("status") in TERMINAL_STATUSES)
    return jobs


async def cancel_job(job_handle: str, tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Proxy the MCP's cancel tool — hidden from the agent, exposed to the UI."""
    raw = await tools["cancel_retrieval"].ainvoke({"job_handle": job_handle})
    return parse_tool_result(raw)
