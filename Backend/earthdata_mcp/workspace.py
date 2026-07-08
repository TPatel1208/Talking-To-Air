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

from config.workflow_stages import STAGE_AOI, STAGE_COVERAGE, STAGE_SEARCH
from earthdata_mcp.results import MCPToolError, call_tool, parse_tool_result
from utils.streaming import emit_status

# T19: one wrapper covers every curated discovery tool without touching the
# MCP — searching/resolving/checking are exactly the stages the model-facing
# discovery tools correspond to (earthdata_mcp/client.py's CURATED_TOOL_NAMES).
_STAGE_BY_TOOL_NAME: dict[str, tuple[str, str]] = {
    "search_datasets": (STAGE_SEARCH, "Searching datasets..."),
    "describe_dataset": (STAGE_SEARCH, "Inspecting dataset..."),
    "preview_dataset": (STAGE_SEARCH, "Previewing dataset..."),
    "define_area_of_interest": (STAGE_AOI, "Resolving area of interest..."),
    "check_availability": (STAGE_COVERAGE, "Checking availability..."),
    "check_coverage": (STAGE_COVERAGE, "Checking coverage..."),
}


def bind_workspace(tools: dict[str, BaseTool], user_id_getter: Callable[[], str]) -> dict[str, BaseTool]:
    """Return copies of ``tools`` with workspace_id injected and hidden from the schema."""
    return {name: _bind_one(tool, user_id_getter) for name, tool in tools.items()}


def _bind_one(tool: BaseTool, user_id_getter: Callable[[], str]) -> BaseTool:
    schema = _schema_without_workspace_id(tool.args_schema)
    stage_info = _STAGE_BY_TOOL_NAME.get(tool.name)

    async def _call(**kwargs):
        kwargs["workspace_id"] = f"user-{user_id_getter()}"
        if stage_info is not None:
            emit_status(stage_info[1], stage=stage_info[0])
        # T18: bind_workspace is the one place every model-facing MCP tool
        # call passes through — classify here (call_tool catches a raised
        # transport failure, parse_tool_result classifies the returned
        # content) and hand back the structured error envelope instead of a
        # raw exception. On success ``raw`` is returned unchanged, so a
        # backend composite's own parse_tool_result(raw) call downstream
        # behaves exactly as before; on error, that same downstream call
        # recognizes the envelope and re-raises the typed MCPToolError.
        try:
            raw = await call_tool(tool, kwargs)
            result = parse_tool_result(raw)
        except MCPToolError as exc:
            return exc.to_tool_json()
        # T19 story #3: surface the granule count once check_coverage's own
        # response is known, so a researcher sees why their request is
        # small or large before the (potentially long) retrieval wait.
        if tool.name == "check_coverage" and isinstance(result, dict) and "granule_count" in result:
            granule_count = result["granule_count"]
            emit_status(f"Checking coverage — {granule_count} granules...", stage=STAGE_COVERAGE, detail=granule_count)
        return raw

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
        try:
            result = parse_tool_result(raw)
        except MCPToolError as exc:
            # ``tool`` is already bind_workspace-wrapped, so ``raw`` here is
            # either real content or bind_workspace's own error envelope
            # (T18) — re-raised by parse_tool_result and passed straight
            # through unchanged rather than re-wrapped.
            return exc.to_tool_json()
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
