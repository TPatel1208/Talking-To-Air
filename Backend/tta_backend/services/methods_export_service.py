"""
services/methods_export_service.py
====================================
T10: assembles the Markdown a paper's methods section needs, deterministically,
from an artifact's own lineage and citations (provenance_service's output) —
never from the chat transcript, and never through an LLM, so the same
session always yields the same text. An optional LLM polish pass may later
rewrite the prose, but these structured facts are what get re-validated
against afterward.

Consumes the shapes provenance_service now produces from the real MCP:
lineage nodes ``{handle, kind, events?: [{event_type, detail, created_at}]}``
and citations ``{handle, doi, collection_citations: [...]}`` (CMR UMM-C
records verbatim). Every field access is defensive — an unexpected shape
must degrade to a blander line, never a KeyError that 500s the endpoint
(QA 2026-07-17 blocker).
"""
from __future__ import annotations

from typing import Any


def build_methods_markdown(
    artifact_title: str,
    aoi_description: str,
    time_window: str,
    lineage: dict[str, Any],
    citations: list[dict[str, Any]],
) -> str:
    nodes = lineage.get("nodes") or []

    lines = [
        f"## Methods — {artifact_title}",
        "",
        f"Data were retrieved for the area of interest **{aoi_description}** over "
        f"the period **{time_window}**.",
        "",
        "### Datasets",
        "",
    ]
    for node in nodes:
        if node.get("kind") == "dataset":
            citation = next(
                (c for c in citations if c.get("handle") == node.get("handle")),
                None,
            )
            title = _citation_title(citation) or node.get("handle", "unknown dataset")
            doi_suffix = f" (doi: {citation['doi']})" if citation and citation.get("doi") else ""
            lines.append(f"- {title}{doi_suffix}")
    lines += ["", "### Processing chain", ""]
    for index, node in enumerate(nodes, start=1):
        step = f"{index}. **{node.get('handle', 'unknown')}** ({node.get('kind', 'step')})"
        step_text = _step_text(node)
        if step_text:
            step += f" — {step_text}"
        lines.append(step)

    retrieval_dates = _retrieval_dates(nodes)
    lines += ["", "### Retrieval dates", ""]
    for date in retrieval_dates:
        lines.append(f"- {date}")

    lines += ["", "### References", ""]
    for index, citation in enumerate(citations, start=1):
        lines.append(f"{index}. {_citation_text(citation)}")

    lines.append("")
    return "\n".join(lines)


def _step_text(node: dict[str, Any]) -> str:
    events = node.get("events") or []
    return "; ".join(_event_text(event) for event in events)


def _event_text(event: dict[str, Any]) -> str:
    detail = event.get("detail") or {}
    detail_parts = ", ".join(f"{key} {value}" for key, value in detail.items())
    when = event.get("created_at") or "time unknown"
    suffix = f", {detail_parts}" if detail_parts else ""
    return f"{event.get('event_type', 'event')} ({when}{suffix})"


def _retrieval_dates(nodes: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for node in nodes:
        for event in node.get("events") or []:
            if event.get("event_type") == "materialized":
                date = str(event.get("created_at") or "")[:10]
                if date and date not in dates:
                    dates.append(date)
    return dates


def _citation_title(citation: dict[str, Any] | None) -> str | None:
    """The dataset's own title, from the first CMR CollectionCitations entry."""
    if not citation:
        return None
    for entry in citation.get("collection_citations") or []:
        title = entry.get("Title")
        if title:
            return str(title)
    return None


def _citation_text(citation: dict[str, Any]) -> str:
    """One reference line from a CMR citation record: the formal
    CollectionCitations fields when published, else the DOI, else the handle."""
    for entry in citation.get("collection_citations") or []:
        parts = [
            str(entry[key])
            for key in ("Creator", "Title", "Publisher", "ReleaseDate")
            if entry.get(key)
        ]
        if parts:
            text = ", ".join(parts)
            if citation.get("doi"):
                text += f", doi:{citation['doi']}"
            return text
    if citation.get("doi"):
        return f"doi:{citation['doi']}"
    return str(citation.get("handle", ""))
