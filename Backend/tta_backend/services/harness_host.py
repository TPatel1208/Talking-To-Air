"""The app's implementations of the harness's host ports, over existing services.

Nothing in production builds a harness run yet; these are what it will be
handed: the per-user earthdata session, retrieval opening, heavy-compute
admission, size and time limits, and place resolution.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from shapely.geometry import mapping

from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT, MCPToolError, call_tool, parse_tool_result
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.run_context import AnsweredRequests, Limits, Region
from tta_backend.infrastructure.settings import Settings
from tta_backend.services import admission
from tta_backend.services.open_handle import HandleGoneError, open_handle
from tta_backend.utils.plotting import RegionResolver


class McpEarthdataSession:
    """The earthdata service over one user's workspace-bound MCP tools.

    The service has no idempotency parameter, so a call repeating an
    idempotency key and request this session already answered returns the
    first answer without calling the service again; the same key with a
    different request is refused as a contract failure. The session keeps
    the most recent keys only. A saved job reference covers a resumed run in
    another process; only a crash between the service's reply and the harness
    saving the job can submit twice.
    """

    def __init__(self, tools: dict[str, BaseTool], *, remembered: int = 256) -> None:
        self._tools = tools
        self._answered = AnsweredRequests(remembered)

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            answer = self._answered.get(idempotency_key, tool, arguments)
            if answer is not None:
                return answer
        if tool not in self._tools:
            raise MCPToolError(CATEGORY_CONTRACT, f"The data service offers no tool named {tool!r}.")
        result = parse_tool_result(await call_tool(self._tools[tool], arguments))
        if idempotency_key is not None:
            self._answered.put(idempotency_key, tool, arguments, result)
        return result


def open_retrieval_over(tools: dict[str, BaseTool], user_id: str) -> OpenRetrieval:
    """An ``OpenRetrieval`` over one user's MCP tools; another user's handle is
    not found, and one whose data is gone and cannot be rematerialised is evicted."""

    async def open_retrieval(owner: str, record: HandleRecord) -> Any:
        if owner != user_id:
            raise HandleError("not_found", record.id)
        try:
            return await open_handle(record.id, tools)
        except HandleGoneError:
            raise HandleError("evicted", record.id) from None

    return open_retrieval


class AdmissionCompute:
    """Heavy work on the app's bounded pool, admitted like every chart tool's."""

    async def run[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T:
        return await admission.run_heavy(functools.partial(fn, *args, **kwargs))


def limits_from_settings(settings: Settings) -> Limits:
    """The harness's retrieval limits, from the same settings the current system uses."""
    return Limits(
        retrieval_soft_bytes=settings.retrieval_soft_cap_bytes,
        retrieval_hard_bytes=settings.retrieval_hard_cap_bytes,
        max_point_timeseries_days=settings.retrieval_max_timeseries_days,
        retrieval_timeout_seconds=float(settings.await_retrieval_timeout_seconds),
        poll_min_seconds=float(settings.await_retrieval_poll_min_seconds),
        poll_max_seconds=float(settings.await_retrieval_poll_max_seconds),
    )


class ResolverPlaces:
    """Places resolved by the app's ``RegionResolver``: presets, states, composites, buffers, geocoding."""

    def __init__(self, resolver: RegionResolver | None = None) -> None:
        self._resolver = resolver or RegionResolver()

    async def resolve(self, wording: str) -> Region | None:
        region = await self._resolver.aresolve_location(wording)
        if region is None:
            return None
        west, south, east, north = (float(v) for v in region["bounds"])
        return Region(
            name=str(region.get("display_name") or region.get("name") or wording),
            bounds=(west, south, east, north),
            geometry=dict(mapping(region["geometry"])),
            region_type=region.get("region_type"),
        )
