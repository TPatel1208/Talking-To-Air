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
from typing import Any

from langchain_core.tools import BaseTool

from config.settings import Settings, get_settings
from earthdata_mcp.results import parse_tool_result
from utils.streaming import emit_job_progress

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled"}


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
        emit_job_progress(
            job_handle,
            status,
            data.get("progress"),
            data.get("phase"),
            data.get("message"),
        )
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

    subset_raw = await tools["retrieve_subset"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "variables": variables,
        "output_format": output_format,
    })
    subset = parse_tool_result(subset_raw)
    return {"status": "submitted", "estimated_bytes": estimated_bytes, **subset}
