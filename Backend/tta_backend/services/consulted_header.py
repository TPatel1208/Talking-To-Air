"""
consulted_header.py
-------------------
T64 D14: the "Data sources: ..." line that opens a sub-agent answer. It is
derived from what the turn's dispatch calls actually ran, so it can never
claim a dataset or ground check that did not happen; any header the model
writes itself is dropped.

A satellite run is named by the datasets its charts were drawn from; a ground
run is EPA AQS; a run that failed draws on nothing. Each dispatch item records
its sources when it runs, because the checkpointed copy keeps only chart
references and loses the dataset.
"""
from __future__ import annotations

import re
from typing import Any

from tta_backend.models import AgentResult, parse_dispatch_result

_HEADER_PREFIX = "Data sources:"
# The pre-rename header. A model imitating it may restate raw data above it,
# so everything above the last one is dropped.
_LEGACY_HEADER_LINE = re.compile(r"^[ \t]*Agent consulted:[^\n]*\n?", re.MULTILINE)
# A header at the very start of the text: the server's own, or a model's copy.
_LEADING_HEADER = re.compile(r"\A[ \t]*(?:Data sources|Agent consulted):[^\n]*(?:\n|\Z)")
# A model's closing "Data sources" line, which would repeat the header.
_TRAILING_SOURCES = re.compile(r"\n[ \t]*Data sources:[^\n]*\s*\Z")
_RAN = ("ok", "error", "duplicate")

GROUND_SOURCE = "EPA AQS"
SATELLITE_FALLBACK = "NASA Earthdata"


def _field(obj: Any, name: str) -> Any:
    """``name`` off a ChartPayload (extras are attributes) or a plain dict,
    read in place: a chart carries its whole grid, so it is never dumped."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _datasets(chart: Any) -> list[str]:
    provenances = [_field(chart, "provenance")]
    provenances += [_field(panel, "provenance") for panel in _field(chart, "panels") or []]
    return [p["dataset"] for p in provenances if isinstance(p, dict) and isinstance(p.get("dataset"), str) and p["dataset"]]


def merge_sources(*groups: list[str]) -> list[str]:
    """Every source in ``groups`` once: satellite datasets in first-seen
    order, then EPA AQS last."""
    seen: list[str] = []
    for source in (source for group in groups for source in group):
        if source not in seen:
            seen.append(source)
    return [s for s in seen if s != GROUND_SOURCE] + ([GROUND_SOURCE] if GROUND_SOURCE in seen else [])


def _failed(result: AgentResult) -> bool:
    """A run salvaged into an error answer with nothing produced: a
    rate-limited or crashed sub-agent comes back this way, not as an error."""
    produced = bool(result.charts or result.artifacts)
    return not produced and bool(result.metadata.get("salvaged") or result.metadata.get("error"))


def result_sources(agent: str, result: AgentResult) -> list[str]:
    """The sources one sub-agent run drew on; none when it failed."""
    if _failed(result):
        return []
    if agent == "ground":
        return [GROUND_SOURCE]
    datasets = merge_sources([d for chart in result.charts for d in _datasets(chart)]) or [SATELLITE_FALLBACK]
    return merge_sources(datasets, [GROUND_SOURCE] if _has_ground_series(result) else [])


def _has_ground_series(result: AgentResult) -> bool:
    """Whether a satellite run's output carries EPA AQS series:
    validate_against_ground tags each series it draws with its source_kind."""
    return any(
        isinstance(series, dict) and series.get("source_kind") == "ground"
        for artifact in result.artifacts
        for series in artifact.metadata.get("series") or []
    )


def consulted_sources(tool_content: Any) -> list[str]:
    """The sources a supervisor ToolMessage shows were consulted, from the
    dispatch items that ran (a rejected item never did). An item saved
    before items recorded sources is named by its agent. Empty for anything
    that is not a dispatch batch."""
    batch = parse_dispatch_result(tool_content)
    if batch is None:
        return []
    return merge_sources(*(
        item.sources if item.sources is not None
        else [GROUND_SOURCE] if item.agent == "ground" else [SATELLITE_FALLBACK]
        for item in batch.results if item.status in _RAN
    ))


def sources_header(sources: list[str]) -> str | None:
    return f"{_HEADER_PREFIX} {' · '.join(sources)}" if sources else None


def with_sources_header(text: str, sources: list[str]) -> str:
    """``text`` opened by the header for ``sources``.

    Dropped from ``text``: any header line it opens with (on the live stream
    the server's header precedes the answer); everything above the last
    legacy "Agent consulted" line, where a model imitating that header
    restated raw data first; and a closing "Data sources" line. A "Data
    sources" line inside the answer stays: models cite that way, and treating
    it as a header would cut the text above it."""
    legacy = list(_LEGACY_HEADER_LINE.finditer(text))
    body = (text[legacy[-1].end():] if legacy else text).strip()
    while (leading := _LEADING_HEADER.match(body)) is not None:
        body = body[leading.end():].lstrip()
    body = _TRAILING_SOURCES.sub("", body).strip()
    header = sources_header(sources)
    return f"{header}\n\n{body}" if header else body
