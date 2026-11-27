"""
services/retrieval_composites.py
==================================
Waiting on an earthdata-retrieval job backend-side: ``await_retrieval`` polls
its status until it reaches a terminal state, narrating each poll.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool

from tta_backend.infrastructure.settings import Settings, get_settings
from tta_backend.config.workflow_stages import STAGE_PROGRESS
from tta_backend.domain.earthdata.retrieval import (
    PAUSED_STATUS,
    POLL_TRANSIENT_FAILURE_LIMIT,
    TERMINAL_STATUSES,
    annotate_paused,
)
from tta_backend.earthdata_mcp.results import (
    CATEGORY_PROVIDER_UNAVAILABLE,
    MCPToolError,
    parse_tool_result,
)
from tta_backend.infrastructure.metrics import observe_harmony_fetch
from tta_backend.utils.streaming import emit_status

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalError",
    "RetrievalTimeoutError",
    "await_retrieval",
]


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

    Backs off from poll_min to poll_max seconds, emits a status line each
    poll, and returns the terminal status dict (including obs_handle on
    success).
    A failed/cancelled job is returned, not raised — the MCP's own stage/
    provider-prefixed error string is the caller's answer, verbatim.

    A provider-paused job (see :func:`annotate_paused`) is also returned
    immediately, with status ``"paused"`` and a plain-language ``note`` —
    polling it to the timeout would just narrate a job that cannot finish
    on its own.
    """
    settings = settings or get_settings()
    status_tool = tools["get_retrieval_status"]
    interval = settings.await_retrieval_poll_min_seconds
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + settings.await_retrieval_timeout_seconds

    consecutive_poll_failures = 0
    while True:
        try:
            raw = await status_tool.ainvoke({"job_handle": job_handle})
            data = annotate_paused(parse_tool_result(raw))
        except MCPToolError as exc:
            # Transient outages (the MCP's upstream call dying, a network
            # blip) are retried up to POLL_TRANSIENT_FAILURE_LIMIT
            # consecutive times — the job itself is still running at the
            # provider, so abandoning the await on the first blip throws
            # away real work. Anything non-transient (contract/user_input)
            # is a bug or a bad request and fails loud on the first poll.
            if exc.category != CATEGORY_PROVIDER_UNAVAILABLE:
                raise
            consecutive_poll_failures += 1
            if consecutive_poll_failures > POLL_TRANSIENT_FAILURE_LIMIT:
                raise
            logger.warning(
                "await_retrieval_poll_retry",
                extra={
                    "_event": "await_retrieval_poll_retry",
                    "_job_handle": job_handle,
                    "_consecutive_failures": consecutive_poll_failures,
                },
            )
            await asyncio.sleep(interval)
            interval = min(interval * 2, settings.await_retrieval_poll_max_seconds)
            continue
        consecutive_poll_failures = 0
        status = data.get("status", "")
        progress = data.get("progress")
        if status == PAUSED_STATUS:
            emit_status(
                "Retrieval paused by the data provider — the request may be too large.",
                stage=STAGE_PROGRESS,
                detail=progress,
            )
            return data
        # `phase` is harmony-retrieval-mcp's qualitative label for the job and
        # reads better mid-flight than the durable `status` it qualifies
        # ("queued at provider" vs. "submitted") — the jobs panel already
        # prefers it for exactly that reason (Frontend/src/utils/jobCard.js),
        # so this line had been showing the worse of two strings it already
        # held. Terminal responses ignore it for the same reason that panel
        # does: a cancel carries no phase, or a stale one, and a "cancelled"
        # line reading "materializing" is a contradiction.
        label = status if status in TERMINAL_STATUSES else (data.get("phase") or status)
        emit_status(f"Retrieving data — {label}...", stage=STAGE_PROGRESS, detail=progress)
        if status in TERMINAL_STATUSES:
            if status == "ready":
                # Only successful fetches, so the histogram means "download time".
                observe_harmony_fetch(loop.time() - started)
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
