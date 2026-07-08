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

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.settings import Settings

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
}


class EarthdataMCPUnavailableError(RuntimeError):
    """Raised when the earthdata-retrieval MCP is unreachable or missing a required tool."""


async def load_raw_mcp_tools(settings: Settings) -> dict[str, BaseTool]:
    """Connect to the earthdata-retrieval MCP and return every tool it exposes, by name."""
    headers = {}
    if settings.earthdata_mcp_token:
        headers["Authorization"] = f"Bearer {settings.earthdata_mcp_token}"

    client = MultiServerMCPClient({
        "earthdata": {
            "url": settings.earthdata_mcp_url,
            "transport": "streamable_http",
            "headers": headers,
        }
    })
    try:
        tools = await client.get_tools()
    except Exception as exc:
        raise EarthdataMCPUnavailableError(
            f"Could not reach earthdata-retrieval MCP at {settings.earthdata_mcp_url}: {exc}"
        ) from exc

    by_name = {tool.name: tool for tool in tools}
    missing = [name for name in REQUIRED_TOOL_NAMES if name not in by_name]
    if missing:
        raise EarthdataMCPUnavailableError(
            f"earthdata-retrieval MCP at {settings.earthdata_mcp_url} is missing "
            f"required tool(s): {', '.join(missing)}"
        )
    return by_name
