"""
services/provenance_service.py
================================
Backend composite behind the provenance pane (PRD T10): walks an artifact's
``source_handles`` through the MCP's ``get_provenance`` tool, one call per
handle, and merges the results into a single deduplicated node list
("how was this made").

The real MCP contract (harmony-retrieval-mcp ``tools/provenance.py``) answers
``{handle, ancestors: [{handle, depth}], events: [{event_type, detail,
created_at}], resolved_handle?, note?}`` — ancestry is a flat handle list,
never nested nodes, and nodes carry no ``kind``/``description`` of their own.
Kind is derived here from the handle's prefix (``dataset_`` → dataset, ...),
which is the MCP's own naming contract (workspace.models.HandleType).

History: this service originally consumed an imagined ``inputs``/``kind``/
``stage``/``at`` shape that only the test fixtures ever produced — against
the live server that made citations silently empty and methods.md crash
(QA 2026-07-17 blockers). Fixtures now mirror the real wire shape.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from tta_backend.earthdata_mcp.results import parse_tool_result

# Handle prefix → node kind, mirroring the MCP's HandleType vocabulary.
_KIND_BY_PREFIX = {
    "dataset": "dataset",
    "aoi": "aoi",
    "obs": "observation",
    "cube": "cube",
    "job": "job",
    "query": "query",
    "preview": "preview",
}


def kind_of_handle(handle: Any) -> str:
    """Derive a node's kind from its handle prefix (``dataset_x`` → ``dataset``)."""
    return _KIND_BY_PREFIX.get(str(handle).split("_", 1)[0], "step")


async def get_lineage(source_handles: list[str], tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Merge the per-handle provenance walks of ``source_handles`` into one lineage.

    Returns ``{"nodes": [...]}`` where each node is ``{handle, kind}`` plus,
    for ancestors, ``depth``, and for the walked handle itself, ``events``
    (re-ordered oldest-first for rendering; the MCP answers newest-first)
    and the MCP's explanatory ``note`` when there is one. Nodes are
    deduplicated by handle (two artifacts can share an upstream AOI or
    dataset) and ordered ancestors-first so lineage renders newest-last.
    """
    by_handle: dict[str, dict[str, Any]] = {}
    for handle in source_handles:
        raw = await tools["get_provenance"].ainvoke({"handle": handle})
        response = parse_tool_result(raw)

        ancestors = sorted(
            response.get("ancestors") or [],
            key=lambda ancestor: -(ancestor.get("depth") or 0),
        )
        for ancestor in ancestors:
            ancestor_handle = ancestor.get("handle")
            if ancestor_handle and ancestor_handle not in by_handle:
                by_handle[ancestor_handle] = {
                    "handle": ancestor_handle,
                    "kind": kind_of_handle(ancestor_handle),
                    "depth": ancestor.get("depth"),
                }

        # A job_ handle redirects to its obs_ handle's lineage; the response
        # names the redirect target so the node carries the real identity.
        target = response.get("resolved_handle") or response.get("handle") or handle
        if target not in by_handle:
            node: dict[str, Any] = {
                "handle": target,
                "kind": kind_of_handle(target),
                "events": list(reversed(response.get("events") or [])),
            }
            if response.get("note"):
                node["note"] = response["note"]
            by_handle[target] = node

    return {"nodes": list(by_handle.values())}


async def get_citations(source_handles: list[str], tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    """Cite every distinct dataset behind ``source_handles``, deduplicated.

    Dataset ancestry only shows up as ``dataset``-kind nodes in the merged
    lineage, so this walks the same lineage ``get_lineage`` produces and
    cites each distinct dataset handle once. Each citation is the MCP's
    ``cite_dataset`` record verbatim: ``{handle, concept_id, doi,
    doi_authority, collection_citations, reference_citation_count}``.
    """
    lineage = await get_lineage(source_handles, tools)
    dataset_handles = [node["handle"] for node in lineage["nodes"] if node.get("kind") == "dataset"]

    citations = []
    for handle in dict.fromkeys(dataset_handles):
        raw = await tools["cite_dataset"].ainvoke({"dataset_handle": handle})
        citations.append(parse_tool_result(raw))
    return citations
