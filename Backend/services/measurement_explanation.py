"""
services/measurement_explanation.py
=====================================
PRD T36 Phase 3 -- the read-only bridge from P2's deterministic evidence to a
grounded, on-demand explanation. ``explain_measurement`` looks up a persisted
chart payload (``agent_charts``, where P2 already stored ``provenance.evidence``)
and returns a compact reliability-relevant subset: the evidence facts *verbatim*,
the masking ``qa_status``/``qa_source`` disclosure, the plotted variable/units,
and an explicit ``has_evidence`` bool.

No recomputation (P2 owns the numbers -- this returns the stored list unchanged),
no MCP call, no LLM. An unknown chart_id, a chart without provenance, or a
science-only chart with no companion evidence all resolve to
``has_evidence: false`` plus a plain-language ``reason`` -- never an error the
agent has to decode. The prompt's "Explaining measurement reliability" section
turns this subset into words, grounded strictly in it.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from repositories import chart_repository

logger = logging.getLogger(__name__)

_NO_EVIDENCE_REASON = (
    "No companion evidence was retrieved for this chart -- only the science "
    "variable was plotted, so there is no QA pass rate, uncertainty, or context "
    "fact to explain. The QA flag / cloud fraction would have to be retrieved to "
    "populate it."
)


def summarize_measurement_evidence(payload: dict[str, Any] | None) -> dict[str, Any]:
    """The pure reliability subset of one persisted chart payload -- the facts
    layer P3 explains, read straight off ``provenance`` with no recomputation.

    Returns ``evidence`` (P2's ``provenance.evidence`` list, verbatim), the
    ``masking`` qa_status/qa_source when present, the plotted variable/units, and
    an explicit ``has_evidence`` bool. Missing payload / provenance / evidence
    each yield ``has_evidence: false`` with a plain ``reason`` -- never raises.
    """
    if not isinstance(payload, dict):
        return {
            "has_evidence": False,
            "reason": "No chart was found for that id, so there is nothing to explain.",
        }

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return {
            "chart_id": payload.get("chart_id"),
            "variable": payload.get("variable"),
            "units": payload.get("units"),
            "has_evidence": False,
            "reason": "This chart carries no provenance, so there is no retrieved evidence to explain.",
        }

    evidence = provenance.get("evidence")
    evidence = evidence if isinstance(evidence, list) else []
    masking = provenance.get("masking")

    result: dict[str, Any] = {
        "chart_id": payload.get("chart_id"),
        "variable": provenance.get("variable") if provenance.get("variable") is not None else payload.get("variable"),
        "units": provenance.get("units") if provenance.get("units") is not None else payload.get("units"),
        "has_evidence": bool(evidence),
    }
    if evidence:
        # Verbatim -- P2 is the single source of truth for every number; this
        # accessor never re-derives a stat.
        result["evidence"] = evidence
    else:
        result["reason"] = _NO_EVIDENCE_REASON
    if isinstance(masking, dict):
        result["masking"] = {
            "qa_status": masking.get("qa_status"),
            "qa_source": masking.get("qa_source"),
        }
    return {k: v for k, v in result.items() if v is not None}


async def explain_measurement(
    chart_id: str,
    get_chart: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Look up the persisted chart and return its reliability subset. A lookup
    that fails (or an unknown id) degrades to ``has_evidence: false`` with a
    reason -- the agent never has to decode an error path (PRD T36 P3).

    ``chart_id`` is a free-text, LLM-supplied tool argument, so ownership is
    enforced here with the same semantics as every HTTP chart endpoint
    (api.py ``_get_owned_chart``): a chart whose stored ``user_id`` doesn't
    match the caller's reads exactly as not-found -- nothing about it leaks.
    """
    if get_chart is None:
        get_chart = chart_repository.get_chart
    try:
        payload = await get_chart(chart_id)
    except Exception:
        logger.warning("explain_measurement_lookup_failed", extra={"_chart_id": chart_id})
        payload = None
    if payload is not None and payload.get("user_id") is not None and payload.get("user_id") != user_id:
        payload = None
    return summarize_measurement_evidence(payload)
