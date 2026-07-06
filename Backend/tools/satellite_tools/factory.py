"""
tools/satellite_tools/factory.py
==================================
Builds the satellite agent's tool list for one request/session: the curated
model-facing earthdata-retrieval MCP tools (discovery, AOI, coverage,
retrieval) plus the handle-based plot/statistics tools, which close over
``mcp_tools`` so ``open_handle`` can reach export_result/rematerialize —
without the model ever seeing or supplying that dict itself.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from earthdata_mcp.toolset import curated_model_tools
from tools.satellite_tools.geocode_tools import geocode_location
from tools.satellite_tools.plot_tools import (
    make_conduct_temporal_statistic,
    make_plot_multiple,
    make_plot_singular,
)
from tools.satellite_tools.retrieval_tools import make_await_retrieval, make_safe_retrieve
from tools.satellite_tools.stat_tools import make_compute_statistic_tool, make_find_daily_peak


def build_satellite_tools(mcp_tools: dict[str, BaseTool]) -> list[BaseTool]:
    """Assemble the earthdata agent's tools, bound to this request's mcp_tools."""
    return [
        geocode_location,
        *curated_model_tools(mcp_tools),
        make_safe_retrieve(mcp_tools),
        make_await_retrieval(mcp_tools),
        make_plot_singular(mcp_tools),
        make_plot_multiple(mcp_tools),
        make_compute_statistic_tool(mcp_tools),
        make_conduct_temporal_statistic(mcp_tools),
        make_find_daily_peak(mcp_tools),
    ]
