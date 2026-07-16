"""
services/discovery_service.py
================================
Backend composite behind the discovery pane (PRD T09): thin proxy over the
earthdata-retrieval MCP's search/describe/preview/coverage tools, so the
pane's direct (non-agent) use shares the same workspace-bound tools and
authenticated path as the agent — the pane cannot do anything the agent
couldn't.

``preview_dataset`` and ``check_coverage`` take a human-readable ``location``
rather than an ``aoi_handle`` — the pane never mints or stores an AOI handle
itself (pane state stays client-side, per the PRD); this module resolves it
via ``define_area_of_interest`` on every call, the same tool the agent uses.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from datasets.registry import CollectionConfig, load_registry
from datasets.variable_roles import ROLE_DISPLAY_ORDER, classify_inventory
from earthdata_mcp.results import CATEGORY_NO_DATA, parse_tool_result

# Mirrors the MCP's own inspect_granules contract (harmony-retrieval-mcp's
# server.py/tools/coverage.py): a modest default so a first look stays cheap,
# capped at CMR's own effective ceiling so an oversized request is narrowed
# rather than rejected.
_DEFAULT_GRANULE_LIMIT = 10
_MAX_GRANULE_LIMIT = 50


async def search_datasets(query: str, filters: dict | None, tools: dict[str, BaseTool]) -> dict[str, Any]:
    raw = await tools["search_datasets"].ainvoke({"query": query, "filters": filters})
    return parse_tool_result(raw)


async def describe_dataset(dataset_handle: str, tools: dict[str, BaseTool]) -> dict[str, Any]:
    raw = await tools["describe_dataset"].ainvoke({"dataset_handle": dataset_handle, "detail": False})
    result = parse_tool_result(raw)
    return _attach_inventory(result)


def _registry_config_for(result: dict[str, Any]) -> CollectionConfig | None:
    """The registered collection this describe_dataset result belongs to, matched
    on the ``concept_id`` (== registry ``collection_id``) or ``short_name`` the
    result carries — the same identity markers a real granule/CMR record uses.
    ``dataset_handle`` itself is opaque, so identity comes from the payload, not
    the handle. Returns None for an unregistered collection (the classifier
    still runs name-only, just without the registry's primary/qa hints)."""
    concept_id = result.get("concept_id")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    short_name = metadata.get("short_name") or result.get("short_name")
    for cfg in load_registry().values():
        if concept_id and cfg.collection_id == concept_id:
            return cfg
        if short_name and cfg.short_name and cfg.short_name.upper() == str(short_name).upper():
            return cfg
    return None


def _attach_inventory(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a classified, role-annotated ``inventory`` to a describe_dataset
    result (PRD T35). Additive — existing callers ignoring the new key are
    unaffected; the semantic classification lives backend-side next to its
    consumers, exactly as masking semantics live in ``datasets/mask_info.py``
    rather than the MCP. A result with no ``variables`` list is returned
    untouched (no inventory key), so an empty/thin UMM-Var view (e.g. MODIS AOD)
    honestly shows nothing rather than an invented one."""
    variables = result.get("variables")
    if not isinstance(variables, list) or not variables:
        return result

    cfg = _registry_config_for(result)
    classified = classify_inventory(
        variables,
        groups=cfg.groups if cfg else None,
        primary_var=cfg.primary_var if cfg else None,
        quality_flag_var=cfg.quality_flag_var if cfg else None,
    )
    counts: dict[str, int] = {}
    for entry in classified:
        counts[entry["role"]] = counts.get(entry["role"], 0) + 1

    result["inventory"] = {
        "variables": classified,
        "counts": counts,
        "roles_present": [role for role in ROLE_DISPLAY_ORDER if counts.get(role)],
        "collection_key": _registry_key_for(cfg),
    }
    return result


def _registry_key_for(cfg: CollectionConfig | None) -> str | None:
    if cfg is None:
        return None
    for key, candidate in load_registry().items():
        if candidate is cfg:
            return key
    return None


async def preview_dataset(
    dataset_handle: str,
    location: str | None,
    time_range: str | None,
    layer: str | None,
    tools: dict[str, BaseTool],
) -> dict[str, Any]:
    aoi_handle = await _resolve_aoi(location, tools)
    raw = await tools["preview_dataset"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "layer": layer,
    })
    return parse_tool_result(raw)


async def check_coverage(
    dataset_handle: str,
    location: str,
    time_range: str,
    tools: dict[str, BaseTool],
) -> dict[str, Any]:
    aoi_handle = await _resolve_aoi(location, tools)
    raw = await tools["check_coverage"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
    })
    return parse_tool_result(raw)


async def inspect_granules(
    dataset_handle: str,
    location: str,
    time_range: str,
    limit: int | None,
    tools: dict[str, BaseTool],
) -> dict[str, Any]:
    """List the granules a retrieval would pull, before the researcher
    commits to it (T21): the MCP's own records plus a count/total-size
    summary computed from them — no reshaping beyond that, no caching.

    An empty result is a plain answer, not an error (T18's no_data
    category, story #4): the response still carries ``granules: []`` and
    ``count: 0``, annotated with a ``note`` rather than raised as an
    ``MCPToolError``, so absence reads the same as everywhere else in the
    pane without discarding the (empty) result shape.
    """
    aoi_handle = await _resolve_aoi(location, tools)
    applied_limit = min(limit, _MAX_GRANULE_LIMIT) if limit else _DEFAULT_GRANULE_LIMIT
    raw = await tools["inspect_granules"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "limit": applied_limit,
    })
    result = parse_tool_result(raw)
    granules = result.get("granules") or []
    result["total_size_mb"] = sum(g.get("size_mb") or 0 for g in granules)
    result["limit_applied"] = applied_limit
    if not granules:
        result["note"] = {
            "category": CATEGORY_NO_DATA,
            "message": "No granules found for this dataset/area/period.",
        }
    return result


async def _resolve_aoi(location: str | None, tools: dict[str, BaseTool]) -> str | None:
    if not location:
        return None
    raw = await tools["define_area_of_interest"].ainvoke({"location": location})
    aoi = parse_tool_result(raw)
    return aoi.get("handle")
