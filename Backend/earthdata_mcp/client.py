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

# Model-facing curated surface (decision record §8.5): discovery, AOI,
# coverage, retrieval status, single-point timeseries, citation, provenance.
CURATED_TOOL_NAMES = (
    "search_datasets",
    "describe_dataset",
    "preview_dataset",
    "summarize_dataset",
    "define_area_of_interest",
    "check_availability",
    "check_coverage",
    "get_retrieval_status",
    "retrieve_timeseries",
    "cite_dataset",
    "get_provenance",
)
# Used internally by the await_retrieval/safe_retrieve/open_handle composites;
# never exposed to the model as standalone tools.
INTERNAL_TOOL_NAMES = (
    "retrieve_subset",
    "estimate_retrieval_size",
    "export_result",
    "rematerialize",
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
