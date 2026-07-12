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


async def list_jobs(tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    """Return the caller's workspace jobs, each merged with its live status.

    ``list_workspace`` returns every handle in the workspace as
    ``{handles: [{handle, type, created_at, summary}]}`` — filtered here to
    ``type == "job"`` and mapped to the field names the rest of this
    composite (and the frontend) expect. Active (non-terminal) jobs sort
    first, newest-first within each group, so a researcher sees what's still
    running before what's already finished.
    """
    workspace_raw = await tools["list_workspace"].ainvoke({})
    workspace = parse_tool_result(workspace_raw)
    entries = [
        {"job_handle": handle["handle"], "created_at": handle.get("created_at"), **(handle.get("summary") or {})}
        for handle in workspace.get("handles", [])
        if handle.get("type") == "job"
    ]

    statuses = await asyncio.gather(*(
        tools["get_retrieval_status"].ainvoke({"job_handle": entry["job_handle"]})
        for entry in entries
    ))

    jobs = [
        {**entry, **parse_tool_result(status)}
        for entry, status in zip(entries, statuses)
    ]
    jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    jobs.sort(key=lambda job: job.get("status") in TERMINAL_STATUSES)
    return jobs


async def cancel_job(job_handle: str, tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Proxy the MCP's cancel tool — hidden from the agent, exposed to the UI."""
    raw = await tools["cancel_retrieval"].ainvoke({"job_handle": job_handle})
    return parse_tool_result(raw)
