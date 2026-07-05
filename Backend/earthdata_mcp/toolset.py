"""
earthdata_mcp/toolset.py
=========================
Assembles the workspace-bound tool set the earthdata agent and its
composites depend on: every raw MCP tool this backend requires, wrapped so
workspace_id is injected and invisible to the model. The curated,
model-facing subset is a small slice of that — composites (await_retrieval,
safe_retrieve) reach the rest (retrieve_subset, estimate_retrieval_size)
directly by name.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.tools import BaseTool

from config.settings import Settings

from .client import CURATED_TOOL_NAMES, load_raw_mcp_tools
from .workspace import bind_workspace


async def load_earthdata_tools(settings: Settings, user_id_getter: Callable[[], str]) -> dict[str, BaseTool]:
    """Load, validate, and workspace-bind every tool the composites need."""
    raw = await load_raw_mcp_tools(settings)
    return bind_workspace(raw, user_id_getter)


def curated_model_tools(tools: dict[str, BaseTool]) -> list[BaseTool]:
    """The subset of ``tools`` the model is allowed to see and call directly."""
    return [tools[name] for name in CURATED_TOOL_NAMES]
