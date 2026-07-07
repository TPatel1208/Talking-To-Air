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
    # stays required because the point-timeseries composite (new-series
    # PRD) adopts it as its engine (decision record 2026-07-06/07).
    "retrieve_timeseries",
    # T11: model-facing citation/provenance duplicated T10's backend
    # endpoints, which call these directly by name — demoted, not removed.
    "cite_dataset",
    "get_provenance",
)
REQUIRED_TOOL_NAMES = CURATED_TOOL_NAMES + INTERNAL_TOOL_NAMES


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
