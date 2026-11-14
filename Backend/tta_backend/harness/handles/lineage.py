"""How a handle was made, in the shape the provenance pane, citations and methods export read.

Derived handles are walked through the registry; at retrieval handles the walk
hands off to the earthdata service's provenance lookup. The result is
``{"nodes": [{handle, kind, depth?, events?, note?}, ...]}``, deduplicated by
handle and ordered ancestors first.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord, is_derived_id
from tta_backend.harness.handles.registry import HandleRegistry

# The app's provenance lookup over retrieval handles, given the requesting
# user so it answers from that user's workspace: (user_id, source_handles).
Provenance = Callable[[str, list[str]], Awaitable[dict[str, Any]]]


def _operation_event(record: HandleRecord) -> dict[str, Any]:
    return {
        "event_type": "operation",
        "detail": {
            "operation": record.operation,
            "parameters": record.parameters,
            "parents": list(record.parents),
            "release": record.release,
        },
        "created_at": record.created_at.isoformat(),
    }


async def lineage(registry: HandleRegistry, user_id: str, handles: Sequence[str], provenance: Provenance) -> dict[str, Any]:
    """The merged lineage of ``handles``; any handle the user does not own is ``not_found``."""
    roots = [await registry.get(user_id, handle) for handle in dict.fromkeys(handles)]
    root_ids = {record.id for record in roots}

    seen: set[str] = set()
    records: dict[str, HandleRecord] = {}
    finished: list[HandleRecord] = []
    missing: list[str] = []

    # Iterative depth-first walk, parents before children. A parent already
    # seen is never followed again, which also stops a cycle the registry
    # should never contain.
    for root in roots:
        if root.id in seen:
            continue
        seen.add(root.id)
        records[root.id] = root
        stack: list[tuple[HandleRecord, int]] = [(root, 0)]
        while stack:
            record, index = stack[-1]
            if index == len(record.parents):
                stack.pop()
                finished.append(record)
                continue
            stack[-1] = (record, index + 1)
            parent_id = record.parents[index]
            if parent_id in seen:
                continue
            seen.add(parent_id)
            try:
                parent = await registry.get(user_id, parent_id)
            except HandleError:
                missing.append(parent_id)
                continue
            records[parent_id] = parent
            stack.append((parent, 0))

    depth = _shortest_depths(roots, records)

    by_handle: dict[str, dict[str, Any]] = {}
    retrieval = [record.id for record in finished if record.origin == "retrieval"]
    if retrieval:
        for service_node in (await provenance(user_id, retrieval)).get("nodes") or []:
            by_handle.setdefault(service_node["handle"], service_node)
    for handle in missing:
        by_handle.setdefault(handle, {"handle": handle, "kind": handle.split("_", 1)[0] if is_derived_id(handle) else "step", "note": "Not available."})
    for record in finished:
        if record.origin != "derived":
            continue
        node: dict[str, Any] = {"handle": record.id, "kind": record.kind}
        if record.id not in root_ids:
            node["depth"] = depth[record.id]
        node["events"] = [_operation_event(record)]
        by_handle.setdefault(record.id, node)
    return {"nodes": list(by_handle.values())}


def _shortest_depths(roots: list[HandleRecord], records: dict[str, HandleRecord]) -> dict[str, int]:
    depth = {record.id: 0 for record in roots}
    frontier = [record.id for record in roots]
    while frontier:
        following: list[str] = []
        for handle in frontier:
            for parent in records[handle].parents:
                if parent in records and parent not in depth:
                    depth[parent] = depth[handle] + 1
                    following.append(parent)
        frontier = following
    return depth


def dataset_handles(lineage_result: dict[str, Any]) -> list[str]:
    """The distinct dataset handles behind a lineage, in order, for citations."""
    return list(dict.fromkeys(node["handle"] for node in lineage_result.get("nodes") or [] if node.get("kind") == "dataset"))
