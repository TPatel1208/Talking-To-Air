"""
services/provenance_service.py
================================
Backend composite behind the provenance pane (PRD T10): walks an artifact's
``source_handles`` through the MCP's ``get_provenance`` tool, one call per
handle, and merges the results into a single deduplicated, chronologically
ordered node list ("how was this made"). Handles with no ancestry (AOIs,
datasets) come back from the MCP as bare leaf descriptions and render as
leaf inputs, same as everything else.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from earthdata_mcp.results import parse_tool_result


async def get_lineage(source_handles: list[str], tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Merge the per-handle provenance walks of ``source_handles`` into one lineage.

    Each ``get_provenance`` call returns a node that may embed its own
    ancestors inline under ``inputs`` (e.g. a comparison's aligned
    intermediate, or the dataset/AOI leaves behind an observation). This
    flattens that nesting across every source handle into one node list,
    deduplicated by handle (two artifacts can share an upstream AOI or
    dataset) and ordered ancestors-first so it renders newest-last.
    """
    by_handle: dict[str, dict[str, Any]] = {}
    for handle in source_handles:
        raw = await tools["get_provenance"].ainvoke({"handle": handle})
        node = parse_tool_result(raw)
        _flatten_into(node, by_handle)

    return {"nodes": list(by_handle.values())}


def _flatten_into(node: dict[str, Any], by_handle: dict[str, dict[str, Any]]) -> None:
    for ancestor in node.get("inputs") or []:
        _flatten_into(ancestor, by_handle)

    handle = node["handle"]
    if handle not in by_handle:
        by_handle[handle] = {key: value for key, value in node.items() if key != "inputs"}


async def get_citations(source_handles: list[str], tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    """Cite every distinct dataset behind ``source_handles``, deduplicated.

    The datasets an artifact depends on only show up as ``dataset``-kind
    leaf nodes in its lineage, so this walks the same merged lineage
    ``get_lineage`` produces and cites each distinct dataset handle once.
    """
    lineage = await get_lineage(source_handles, tools)
    dataset_handles = [node["handle"] for node in lineage["nodes"] if node.get("kind") == "dataset"]

    citations = []
    for handle in dict.fromkeys(dataset_handles):
        raw = await tools["cite_dataset"].ainvoke({"dataset_handle": handle})
        citations.append(parse_tool_result(raw))
    return citations
