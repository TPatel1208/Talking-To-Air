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

from earthdata_mcp.client import CURATED_TOOL_NAMES
from earthdata_mcp.toolset import curated_model_tools
from tools.satellite_tools.comparison_tools import make_compare
from tools.satellite_tools.plot_tools import (
    make_conduct_temporal_statistic,
    make_plot_multiple,
    make_plot_singular,
)
from tools.satellite_tools.retrieval_tools import make_await_retrieval, make_point_timeseries, make_safe_retrieve
from tools.satellite_tools.stat_tools import make_compute_statistic_tool, make_find_daily_peak
from tools.satellite_tools.validation_tools import make_exceedance_overlay, make_validate_against_ground


def _handle_tools(mcp_tools: dict[str, BaseTool]) -> list[BaseTool]:
    """The handle-based plot/statistics tools, bound to this request's mcp_tools."""
    return [
        make_safe_retrieve(mcp_tools),
        make_await_retrieval(mcp_tools),
        make_point_timeseries(mcp_tools),
        make_plot_singular(mcp_tools),
        make_plot_multiple(mcp_tools),
        make_compute_statistic_tool(mcp_tools),
        make_conduct_temporal_statistic(mcp_tools),
        make_find_daily_peak(mcp_tools),
        make_validate_against_ground(mcp_tools),
        make_exceedance_overlay(mcp_tools),
        make_compare(mcp_tools),
    ]


def build_satellite_tools(mcp_tools: dict[str, BaseTool]) -> list[BaseTool]:
    """Assemble the earthdata agent's tools, bound to this request's mcp_tools."""
    return [*curated_model_tools(mcp_tools), *_handle_tools(mcp_tools)]


def sanctioned_tool_names() -> list[str]:
    """The exact model-facing earthdata tool names, in the same assembly
    order as ``build_satellite_tools`` — the single source of truth for the
    supervisor's refusal-retry guidance, so it can never name a tool this
    backend doesn't register. None of the handle tools' constructors touch
    ``mcp_tools`` eagerly, so an empty dict is safe here.
    """
    return [*CURATED_TOOL_NAMES, *(t.name for t in _handle_tools({}))]
