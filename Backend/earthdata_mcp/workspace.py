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
import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from earthdata_mcp.results import parse_tool_result


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


def model_view_describe_dataset(tool: BaseTool) -> BaseTool:
    """Wrap an already workspace-bound ``describe_dataset`` tool so its
    model-facing result stays proportional to what the model actually uses
    (T13): variable names/units/advisories to subset, never every
    fill-value/valid-range record a many-variable collection carries.

    Applied only to the curated model-facing tool list
    (earthdata_mcp/toolset.py::curated_model_tools) — the original tool in
    the shared workspace-bound dict is left untouched, since discovery-pane
    consumers (services/discovery_service.py) call it directly by name and
    need the full per-variable records.
    """

    async def _call(**kwargs):
        raw = await tool.ainvoke(kwargs)
        result = parse_tool_result(raw)
        return json.dumps(_compact_describe_dataset_result(result))

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _compact_describe_dataset_result(result: dict) -> dict:
    variables = result.get("variables")
    if not isinstance(variables, list):
        return result
    compacted = dict(result)
    compacted["variables"] = [
        _compact_variable(var) if isinstance(var, dict) else var for var in variables
    ]
    return compacted


def _compact_variable(var: dict) -> dict:
    """name/long_name/units/advisory_notes plus a one-line mask_note derived
    from fill/range presence — the model needs variable names to subset, not
    every fill-value/valid-range record (T13 story #11)."""
    out: dict[str, Any] = {
        "name": var.get("name"),
        "long_name": var.get("long_name"),
        "units": var.get("units"),
        "advisory_notes": var.get("advisory_notes", []),
    }
    has_fill = bool(var.get("fill_values"))
    has_range = bool(var.get("valid_ranges"))
    if has_fill and has_range:
        out["mask_note"] = "fill values and a valid range are defined"
    elif has_fill:
        out["mask_note"] = "fill values are defined, no valid range"
    elif has_range:
        out["mask_note"] = "a valid range is defined, no fill values"
    else:
        out["mask_note"] = "no fill/range metadata"
    if "mask_metadata_note" in var:
        out["mask_metadata_note"] = var["mask_metadata_note"]
    return {k: v for k, v in out.items() if v is not None}
