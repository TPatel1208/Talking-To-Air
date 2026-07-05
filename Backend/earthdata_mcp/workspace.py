"""
earthdata_mcp/workspace.py
===========================
Binds every earthdata-retrieval MCP tool call to a workspace_id the model
never sees or invents. Each wrapped tool's schema drops workspace_id
entirely; the wrapper resolves it at call time via ``user_id_getter`` and
injects ``user-{user_id}``, so a researcher's handles persist across their
threads without the model being able to see or forge workspace identity.
"""
from __future__ import annotations

import copy
from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool


def bind_workspace(tools: dict[str, BaseTool], user_id_getter: Callable[[], str]) -> dict[str, BaseTool]:
    """Return copies of ``tools`` with workspace_id injected and hidden from the schema."""
    return {name: _bind_one(tool, user_id_getter) for name, tool in tools.items()}


def _bind_one(tool: BaseTool, user_id_getter: Callable[[], str]) -> BaseTool:
    schema = _schema_without_workspace_id(tool.args_schema)

    async def _call(**kwargs):
        kwargs["workspace_id"] = f"user-{user_id_getter()}"
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool.description,
        args_schema=schema,
    )


def _schema_without_workspace_id(schema):
    schema = copy.deepcopy(schema)
    properties = schema.get("properties", {})
    properties.pop("workspace_id", None)
    schema["required"] = [name for name in schema.get("required", []) if name != "workspace_id"]
    return schema
