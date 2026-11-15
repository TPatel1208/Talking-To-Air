"""Reading the earthdata service's provenance answers: node kinds and the merged lineage.

``get_provenance`` answers ``{handle, ancestors: [{handle, depth}], events:
[{event_type, detail, created_at}], resolved_handle?, note?}``: ancestry is a
flat handle list and nodes carry no kind of their own, so the kind comes from
the handle's prefix.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Handle prefix -> node kind, the earthdata service's own vocabulary.
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
    """A node's kind from its handle prefix (``dataset_x`` -> ``dataset``)."""
    return _KIND_BY_PREFIX.get(str(handle).split("_", 1)[0], "step")


def merge_lineage(walks: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Merge per-handle provenance answers, given as ``(handle asked about, response)``, into one lineage.

    Returns ``{"nodes": [...]}``; each node is ``{handle, kind}`` plus, for
    ancestors, ``depth``, and for the walked handle itself, ``events``
    (oldest first; the service answers newest first) and the service's
    ``note`` when there is one. Nodes are deduplicated by handle and ordered
    ancestors first.
    """
    by_handle: dict[str, dict[str, Any]] = {}
    for handle, response in walks:
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
