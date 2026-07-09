"""
services/retrieval_composites.py
==================================
Backend composites over the earthdata-retrieval MCP that absorb the two
ergonomic hazards of durable retrieval: an LLM burning turns polling job
status (await_retrieval), and an LLM free-running an expensive retrieval
(safe_retrieve).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import BaseTool

from config.settings import Settings, get_settings
from config.workflow_stages import STAGE_ESTIMATE, STAGE_PROGRESS, STAGE_SUBMIT
from datasets.registry import load_registry
from earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError, parse_tool_result
from utils.streaming import emit_job_progress, emit_status

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled"}


def _supports_variable_subsetting(variables: list[str]) -> bool:
    """Best-effort registry check: False only when every requested variable
    resolves to a collection explicitly marked ``supports_variable_subsetting:
    false`` (a Harmony capability flag captured by
    scripts/generate_collection.py). ``safe_retrieve`` only ever sees an
    opaque ``dataset_handle``, not a collection id, so this matches by
    variable short name instead -- a name that collides across registry
    entries always agrees on this flag in practice, and an unregistered
    name defaults to True (today's send-it-and-see behavior) rather than
    guessing.
    """
    index: dict[str, bool] = {}
    for cfg in load_registry().values():
        names = {cfg.primary_var}
        if cfg.quality_flag_var:
            names.add(cfg.quality_flag_var)
        names.update(v.rsplit("/", 1)[-1] for v in cfg.variables)
        for name in names:
            index[name] = cfg.supports_variable_subsetting or index.get(name, False)

    resolved = [index[v] for v in variables if v in index]
    return all(resolved) if resolved else True


class RetrievalError(RuntimeError):
    """Base class for retrieval-composite failures."""


class RetrievalTimeoutError(TimeoutError, RetrievalError):
    """Raised when await_retrieval exceeds the configured polling timeout."""

    def __init__(self, message: str, *, job_handle: str, elapsed_seconds: float):
        super().__init__(message)
        self.job_handle = job_handle
        self.elapsed_seconds = elapsed_seconds


async def await_retrieval(
    job_handle: str,
    tools: dict[str, BaseTool],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Poll get_retrieval_status backend-side until the job reaches a terminal state.

    Spends one model turn instead of a polling loop: backs off from
    poll_min to poll_max seconds, emits a job_progress SSE event each poll,
    and returns the terminal status dict (including obs_handle on success).
    A failed/cancelled job is returned, not raised — the MCP's own stage/
    provider-prefixed error string is the caller's answer, verbatim.
    """
    settings = settings or get_settings()
    status_tool = tools["get_retrieval_status"]
    interval = settings.await_retrieval_poll_min_seconds
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + settings.await_retrieval_timeout_seconds

    while True:
        raw = await status_tool.ainvoke({"job_handle": job_handle})
        data = parse_tool_result(raw)
        status = data.get("status", "")
        progress = data.get("progress")
        emit_job_progress(
            job_handle,
            status,
            progress,
            data.get("phase"),
            data.get("message"),
        )
        emit_status(f"Retrieving data — {status}...", stage=STAGE_PROGRESS, detail=progress)
        if status in TERMINAL_STATUSES:
            return data

        if loop.time() >= deadline:
            elapsed = loop.time() - started
            raise RetrievalTimeoutError(
                f"Retrieval job {job_handle} did not reach a terminal state within "
                f"{settings.await_retrieval_timeout_seconds}s",
                job_handle=job_handle,
                elapsed_seconds=elapsed,
            )

        await asyncio.sleep(interval)
        interval = min(interval * 2, settings.await_retrieval_poll_max_seconds)


async def safe_retrieve(
    dataset_handle: str,
    aoi_handle: str,
    time_range: str,
    variables: list[str],
    tools: dict[str, BaseTool],
    *,
    output_format: str | None = None,
    confirmed: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run estimate -> gate -> retrieve as one deterministic call.

    The soft/hard cap numbers live in config, not prompts, so neither the
    model nor a persuasive user can talk past the guardrail:

    - at or below the soft cap: proceeds automatically.
    - between the soft and hard cap: pauses for confirmation unless the
      caller already has one (``confirmed=True``, e.g. a retry after the
      supervisor relayed the user's approval).
    - above the hard cap: refused unconditionally, even if ``confirmed``.
    """
    settings = settings or get_settings()
    emit_status("Estimating retrieval size...", stage=STAGE_ESTIMATE)
    estimate_raw = await tools["estimate_retrieval_size"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
    })
    estimate = parse_tool_result(estimate_raw)
    estimated_bytes = estimate.get("estimated_bytes", 0)

    if estimated_bytes > settings.retrieval_hard_cap_bytes:
        return {
            "status": "refused",
            "estimated_bytes": estimated_bytes,
            "message": (
                f"Estimated retrieval size (~{estimated_bytes:,} bytes) exceeds the "
                f"{settings.retrieval_hard_cap_bytes:,}-byte hard cap. Narrow the "
                "area of interest, time range, or variable list and try again."
            ),
        }

    if estimated_bytes > settings.retrieval_soft_cap_bytes and not confirmed:
        return {
            "status": "needs_confirmation",
            "estimated_bytes": estimated_bytes,
            "message": (
                f"Estimated retrieval size (~{estimated_bytes:,} bytes) is above the "
                f"{settings.retrieval_soft_cap_bytes:,}-byte soft cap. Reply to confirm "
                "or narrow the request."
            ),
        }

    emit_status("Submitting retrieval...", stage=STAGE_SUBMIT)
    # Skip the doomed subset attempt entirely for collections the registry
    # already knows don't support it (e.g. TROPOMI, MODIS AOD) -- avoids a
    # wasted failed-subset-then-full-retrieval round trip on every call.
    subset_variables = variables if _supports_variable_subsetting(variables) else []
    subset_raw = await tools["retrieve_subset"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "variables": subset_variables,
        "output_format": output_format,
    })
    subset = parse_tool_result(subset_raw)
    return {"status": "submitted", "estimated_bytes": estimated_bytes, **subset}


def _parse_time_span_days(time_range: str) -> int | None:
    """Days between an ISO 8601 'start/end' interval's two ends, or None if
    ``time_range`` isn't in that shape. An unparseable range is left for the
    MCP's own time_range validation to reject rather than folded into the
    too_large gate below."""
    try:
        start_str, end_str = time_range.split("/", 1)
        return (datetime.fromisoformat(end_str) - datetime.fromisoformat(start_str)).days
    except (ValueError, AttributeError):
        return None


async def point_timeseries(
    dataset_handle: str,
    location: str,
    time_range: str,
    variable: str,
    tools: dict[str, BaseTool],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Resolve AOI, gate the requested time span, submit a point-sampled
    timeseries retrieval, and await it to a terminal state (T20) — the
    point-timeseries analogue of safe_retrieve's gate-then-submit shape,
    chained straight into await_retrieval so the model spends one tool call
    instead of three.

    Point sampling is always on (this composite's identity is the MCP's
    AppEEARS-routed point path, never a gridded cube). Raises
    ``MCPToolError(category="too_large")`` before any MCP call when the
    requested span exceeds ``settings.retrieval_max_timeseries_days`` — a
    span gate rather than safe_retrieve's byte-size estimate, since a point
    series has no size to estimate.
    """
    settings = settings or get_settings()

    span_days = _parse_time_span_days(time_range)
    if span_days is not None and span_days > settings.retrieval_max_timeseries_days:
        raise MCPToolError(
            CATEGORY_TOO_LARGE,
            f"Requested time span ({span_days} days) exceeds the "
            f"{settings.retrieval_max_timeseries_days}-day point-timeseries limit.",
            suggestion="Narrow the time range and try again.",
        )

    aoi_raw = await tools["define_area_of_interest"].ainvoke({"location": location})
    aoi = parse_tool_result(aoi_raw)
    aoi_handle = aoi.get("handle") or aoi.get("aoi_handle")

    emit_status("Submitting point timeseries retrieval...", stage=STAGE_SUBMIT)
    submit_raw = await tools["retrieve_timeseries"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "variables": [variable],
        "point_sample": True,
    })
    submit = parse_tool_result(submit_raw)

    status = await await_retrieval(submit["job_handle"], tools, settings=settings)
    return {"aoi_handle": aoi_handle, **status}
