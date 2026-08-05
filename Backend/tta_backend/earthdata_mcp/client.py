"""
earthdata_mcp/client.py
========================
Async client for the earthdata-retrieval MCP: connects over streamable HTTP
(bearer token, per T01 settings), loads every tool the MCP exposes, and
fails loud if the MCP is unreachable or missing a tool this backend depends
on — a broken data layer should be discovered at boot, not mid-conversation.

The curated, model-facing subset is assembled separately in
earthdata_mcp.toolset; this module only deals with the raw connection.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from tta_backend.config.settings import Settings


@dataclass
class HeldSession:
    """The tool surface of one long-lived MCP session, plus a liveness ``probe``
    bound to THAT session.

    ``probe`` (the session's own ``send_ping``) lets the connection manager's
    heartbeat check the session its bound tools actually use — instead of a
    fresh throwaway session, which stays reachable even while the held one is a
    404 zombie after a server restart (server lost the session id; every request
    on it 404s, but a brand-new session connects fine). Probing the fresh one
    masked that death and left the whole tool surface wedged until a process
    restart. ``probe`` is ``None`` only for the legacy plain-loader test seam,
    where the manager falls back to its injected fresh-session probe."""

    tools: dict[str, BaseTool]
    probe: Callable[[], Awaitable[Any]] | None = None

# Model-facing curated surface (T11 decision record: MCP-first minimal
# toolset): discovery, AOI, coverage. Retrieval, citation, and provenance
# are demoted to internal-only — the model reaches them only through the
# safe_retrieve/await_retrieval composites and the T10 backend endpoints.
CURATED_TOOL_NAMES = (
    "search_datasets",
    "describe_dataset",
    "preview_dataset",
    "define_area_of_interest",
    "check_availability",
    "check_coverage",
)
# Used internally by the await_retrieval/safe_retrieve/open_handle/jobs/
# compare composites, or by backend endpoints directly; never exposed to
# the model as standalone tools.
INTERNAL_TOOL_NAMES = (
    "retrieve_subset",
    "estimate_retrieval_size",
    "export_result",
    "rematerialize",
    "list_workspace",
    "cancel_retrieval",
    # T08: the compare tool's period mode calls this directly to grid-align
    # two retrievals before differencing; deferred until this PRD per T02.
    "align",
    # T10: the data-download endpoints call this directly to materialize a
    # handle in a downloadable format (e.g. NetCDF); UI-initiated and
    # deterministic, so it stays off the model-facing surface.
    "convert_format",
    # T11: used by the await_retrieval composite's internal polling — the
    # model never polls status itself.
    "get_retrieval_status",
    # T11: bypasses the size-estimate gate, so it leaves the model surface;
    # stays required because the point-timeseries composite (T20) adopts it
    # as its engine (decision record 2026-07-06/07).
    "retrieve_timeseries",
    # T11: model-facing citation/provenance duplicated T10's backend
    # endpoints, which call these directly by name — demoted, not removed.
    "cite_dataset",
    "get_provenance",
    # T21: the discovery pane's granule-inspection endpoint calls this
    # directly (services/discovery_service.py) so a researcher can see what
    # a retrieval would pull before committing to it; stays off the model
    # surface (the agent keeps coverage + the size gate).
    "inspect_granules",
)
REQUIRED_TOOL_NAMES = CURATED_TOOL_NAMES + INTERNAL_TOOL_NAMES

# The parameters this backend actually sends to each required tool (derived
# from the composites/tool factories that call it directly — see
# services/discovery_service.py, services/jobs_service.py,
# services/provenance_service.py, services/retrieval_composites.py,
# services/data_download_service.py, tools/satellite_tools/comparison_tools.py
# — plus workspace_id, which earthdata_mcp/workspace.py injects into every
# call). A tool with no direct call site here (check_availability, per T11)
# only requires workspace_id: presence is already covered by
# REQUIRED_TOOL_NAMES, and this backend has no fixed param set to assert
# beyond that. PRD T17's connect-time schema check (earthdata_mcp/
# connection.py) verifies each name here appears in the tool's advertised
# input schema.
REQUIRED_TOOL_PARAMS: dict[str, tuple[str, ...]] = {
    "search_datasets": ("query", "filters", "workspace_id"),
    "describe_dataset": ("dataset_handle", "detail", "workspace_id"),
    "preview_dataset": ("dataset_handle", "aoi_handle", "time_range", "layer", "workspace_id"),
    "define_area_of_interest": ("location", "workspace_id"),
    "check_availability": ("workspace_id",),
    "check_coverage": ("dataset_handle", "aoi_handle", "time_range", "workspace_id"),
    "retrieve_subset": (
        "dataset_handle", "aoi_handle", "time_range", "variables", "output_format", "workspace_id",
    ),
    "estimate_retrieval_size": ("dataset_handle", "aoi_handle", "time_range", "workspace_id"),
    "export_result": ("handle", "workspace_id"),
    "rematerialize": ("handle", "workspace_id"),
    "list_workspace": ("workspace_id",),
    "cancel_retrieval": ("job_handle", "workspace_id"),
    "align": ("source_handles", "workspace_id"),
    "convert_format": ("source_handle", "output_format", "workspace_id"),
    "get_retrieval_status": ("job_handle", "workspace_id"),
    # T20: services/retrieval_composites.py::point_timeseries's direct call
    # site — never output_format, this composite is always point-sampled.
    "retrieve_timeseries": (
        "dataset_handle", "aoi_handle", "time_range", "variables", "point_sample", "workspace_id",
    ),
    "cite_dataset": ("dataset_handle", "workspace_id"),
    "get_provenance": ("handle", "workspace_id"),
    # T21: services/discovery_service.py::inspect_granules's direct call site.
    "inspect_granules": ("dataset_handle", "aoi_handle", "time_range", "limit", "workspace_id"),
}


class EarthdataMCPUnavailableError(RuntimeError):
    """Raised when the earthdata-retrieval MCP is unreachable or missing a required tool."""


def _client(settings: Settings) -> MultiServerMCPClient:
    """Build the earthdata-retrieval streamable-HTTP client. The bearer token
    here is this backend's *shared* MCP credential (T01), set once per
    connection; per-user Earthdata credentials flow separately as the
    ``edl_token`` tool argument (T31), never as a connection header, so one
    long-lived session serves every user's calls without header divergence."""
    headers = {}
    if settings.earthdata_mcp_token:
        headers["Authorization"] = f"Bearer {settings.earthdata_mcp_token}"
    return MultiServerMCPClient({
        "earthdata": {
            "url": settings.earthdata_mcp_url,
            "transport": "streamable_http",
            "headers": headers,
        }
    })


async def _list_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Connect to the earthdata-retrieval MCP and return every tool it
    exposes. Used by ``load_raw_mcp_tools`` (the standalone, non-session-held
    load path still used by backend endpoints and test fixtures) and
    ``probe_mcp_tools`` (steady-state heartbeat, listing only) -- both need
    the same connect-and-list step and the same unreachable-MCP wrapping.

    Note this is the *per-call-reconnect* path: ``get_tools()`` binds each
    returned tool to ``session=None`` so every ``ainvoke`` opens a fresh
    streamable-HTTP session. The connection manager's steady-state tool
    surface goes through ``open_earthdata_session`` instead, which holds one
    session open so bound tools reuse it (see that function's docstring)."""
    try:
        return await _client(settings).get_tools()
    except Exception as exc:
        raise EarthdataMCPUnavailableError(
            f"Could not reach earthdata-retrieval MCP at {settings.earthdata_mcp_url}: {exc}"
        ) from exc


@asynccontextmanager
async def open_earthdata_session(settings: Settings) -> AsyncIterator[HeldSession]:
    """Open **one** long-lived streamable-HTTP session to the earthdata-
    retrieval MCP and yield every tool it exposes, by name, bound to that
    session.

    This is the root-cause fix for the 2026-07-20 request storm. Tools loaded
    via ``MultiServerMCPClient.get_tools()`` carry ``session=None``, so
    langchain-mcp-adapters 0.3.0 opens a *fresh* session (POST initialize +
    notifications/initialized + GET SSE + POST tools/call) on every single
    ``ainvoke`` — a ~4-5x HTTP fan-out per logical tool call and a reconnect-
    storm surface whenever the server's SSE flaps. ``load_mcp_tools(session)``
    with a live session instead binds each tool to that one session, so its
    ``call_tool`` reuses the open connection (``await session.call_tool(...)``)
    with no per-call handshake.

    The session stays open for the whole ``async with`` body — the connection
    manager (earthdata_mcp/connection.py) holds it open across its ready +
    heartbeat phase and only lets this context exit (closing the session) on
    demotion, reconnect, or ``stop()``. Because a langchain-mcp-adapters
    session is anchored to the anyio task that entered it, the *same* task
    (the manager's connect loop) must own this ``async with`` for its lifetime;
    concurrent tool calls from other tasks (the ReAct ToolNode's
    ``asyncio.gather``) are safe — the MCP SDK's ``BaseSession`` multiplexes
    them by JSON-RPC request id over the shared streams.

    Connection-phase failures (unreachable server, a required tool gone
    missing) raise ``EarthdataMCPUnavailableError`` exactly like
    ``load_raw_mcp_tools``, so the manager's connect loop classifies them
    unchanged. Exceptions raised by the *body* (bind/on_ready/heartbeat, or a
    ``CancelledError`` from ``stop()``) propagate untouched and still close
    the session via the context manager's normal unwind."""
    connected = False
    try:
        async with _client(settings).session("earthdata") as session:
            tools = await load_mcp_tools(session)
            by_name = {tool.name: tool for tool in tools}
            _require_all_present(set(by_name.keys()), settings.earthdata_mcp_url)
            connected = True
            # Hand the manager a probe bound to THIS session (send_ping over the
            # held connection) so its heartbeat detects a 404-zombie session the
            # bound tools would otherwise keep failing on — see HeldSession.
            yield HeldSession(by_name, probe=session.send_ping)
    except EarthdataMCPUnavailableError:
        raise
    except Exception as exc:
        # Only a *connection-phase* failure (before ``connected``) is a
        # reachability problem to wrap. Anything after the session is up is a
        # body/teardown error (or CancelledError, a BaseException that skips
        # this clause entirely) — re-raise it as-is so the manager's own
        # classification (on_ready failure logging, etc.) stays intact.
        if connected:
            raise
        raise EarthdataMCPUnavailableError(
            f"Could not reach earthdata-retrieval MCP at {settings.earthdata_mcp_url}: {exc}"
        ) from exc


def _require_all_present(names: set[str], url: str) -> None:
    missing = [name for name in REQUIRED_TOOL_NAMES if name not in names]
    if missing:
        raise EarthdataMCPUnavailableError(
            f"earthdata-retrieval MCP at {url} is missing required tool(s): {', '.join(missing)}"
        )


async def load_raw_mcp_tools(settings: Settings) -> dict[str, BaseTool]:
    """Connect to the earthdata-retrieval MCP and return every tool it exposes, by name."""
    tools = await _list_mcp_tools(settings)
    by_name = {tool.name: tool for tool in tools}
    _require_all_present(set(by_name.keys()), settings.earthdata_mcp_url)
    return by_name


async def probe_mcp_tools(settings: Settings) -> set[str]:
    """Lightweight steady-state heartbeat check (PRD T40): list the MCP's
    tool names and confirm every required tool is still present -- no
    schema fetch beyond what listing requires, no workspace bind. Raises
    ``EarthdataMCPUnavailableError`` exactly like ``load_raw_mcp_tools``
    (unreachable, or a required tool went missing), so the connection
    manager's heartbeat can treat any probe exception as a single miss."""
    tools = await _list_mcp_tools(settings)
    names = {tool.name for tool in tools}
    _require_all_present(names, settings.earthdata_mcp_url)
    return names
