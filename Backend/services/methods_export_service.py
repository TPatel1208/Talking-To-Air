"""
services/methods_export_service.py
====================================
T10: assembles the Markdown a paper's methods section needs, deterministically,
from an artifact's own lineage and citations (provenance_service's output) —
never from the chat transcript, and never through an LLM, so the same
session always yields the same text. An optional LLM polish pass may later
rewrite the prose, but these structured facts are what get re-validated
against afterward.
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
                (c for c in citations if c.get("dataset_handle") == node["handle"]),
                None,
            )
            doi_suffix = f" (doi: {citation['doi']})" if citation and citation.get("doi") else ""
            lines.append(f"- {node.get('description', node['handle'])}{doi_suffix}")
    lines += ["", "### Processing chain", ""]
    for index, node in enumerate(nodes, start=1):
        lines.append(f"{index}. **{node['handle']}** ({node.get('kind', 'step')}) — {_step_text(node)}")

    retrieval_dates = _retrieval_dates(nodes)
    lines += ["", "### Retrieval dates", ""]
    for date in retrieval_dates:
        lines.append(f"- {date}")

    lines += ["", "### References", ""]
    for index, citation in enumerate(citations, start=1):
        lines.append(f"{index}. {citation.get('citation', citation.get('doi', ''))}")

    lines.append("")
    return "\n".join(lines)


def _step_text(node: dict[str, Any]) -> str:
    events = node.get("events") or []
    if events:
        return "; ".join(_event_text(event) for event in events)
    return node.get("description", "")


def _event_text(event: dict[str, Any]) -> str:
    detail_parts = [f"{key} {value}" for key, value in event.items() if key not in ("stage", "at")]
    detail = f", {', '.join(detail_parts)}" if detail_parts else ""
    return f"{event['stage']} ({event['at']}{detail})"


def _retrieval_dates(nodes: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for node in nodes:
        for event in node.get("events") or []:
            if event.get("stage") == "materialized":
                date = event["at"][:10]
                if date not in dates:
                    dates.append(date)
    return dates
