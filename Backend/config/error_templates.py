"""
config/error_templates.py
===========================
PRD T18: deterministic answers for the two chat-level failure points where
no sub-agent turn ever produced text a model composed — the dispatch-time
"data layer isn't ready" gate (T17) and a salvage with nothing to salvage
(T15's envelope-parse failure with no prose at all). Both live in
services/subagent_dispatch.py, which the supervisor's tool wrappers and the
router fast path (T14) call identically, so templating here gives both
paths the same answer for the same failure (story #12) with no model call
in either.

Every template is filled only from observed facts (``stage``, ``detail``)
— never a guess at cause. This is what keeps the system honest when it's
wrong: it can misclassify a failure, but it can never narrate one.
"""
from __future__ import annotations

from earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_NOT_FOUND,
    CATEGORY_NO_DATA,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_TOO_LARGE,
    CATEGORY_USER_INPUT,
)

_TEMPLATES: dict[str, str] = {
    CATEGORY_USER_INPUT: "The {stage} could not proceed: {detail}",
    CATEGORY_NO_DATA: "The {stage} found no data: {detail}",
    CATEGORY_NOT_FOUND: "The {stage} referenced something that no longer exists: {detail}",
    CATEGORY_TOO_LARGE: "The {stage} was too large to complete: {detail}",
    CATEGORY_PROVIDER_UNAVAILABLE: (
        "The {stage} is temporarily unavailable. {detail} "
        "This was not a problem with your question — try again in a moment."
    ),
    CATEGORY_CONTRACT: (
        "The {stage} hit an internal error and could not complete. {detail} "
        "This has been logged; it was not a problem with your question."
    ),
}

_DEFAULT_DETAIL = "No further detail is available."


def render_error_answer(category: str, stage: str, detail: str | None = None) -> str:
    """A fixed, honest chat answer for ``category`` — filled only with
    observed facts (``stage``, ``detail``), no model in the loop. An
    unrecognized category still renders (falls back to the ``contract``
    template) rather than raising, matching the classifier's own
    additive-safe rule.

    ``detail`` is normalized to end with terminal punctuation: several
    templates continue with another sentence after it, and an unpunctuated
    detail produced run-ons like "...no further detail is available This
    has been logged..." (live 2026-07-16)."""
    template = _TEMPLATES.get(category, _TEMPLATES[CATEGORY_CONTRACT])
    detail = (detail or "").strip() or _DEFAULT_DETAIL
    if detail[-1] not in ".!?":
        detail += "."
    return template.format(stage=stage, detail=detail)


# T38: distinct from the taxonomy templates above — a turn deadline is not an
# MCPToolError category, it is stream_chat_events/_fast_path_events giving up
# on the whole turn rather than one classified failure. Names any job handles
# still running server-side (collected from this turn's job_progress events)
# so a researcher staring at the timeout answer knows the Jobs panel, not a
# retry, is where to look.
_TURN_TIMEOUT_TEMPLATE = (
    "This request ran out of time before it could finish. {jobs_detail}"
    "This was not a problem with your question — try again, or check the "
    "Jobs panel for anything still running."
)


def render_turn_timeout_answer(job_handles: list[str] | None = None) -> str:
    jobs_detail = f"Still running: {', '.join(job_handles)}. " if job_handles else ""
    return _TURN_TIMEOUT_TEMPLATE.format(jobs_detail=jobs_detail)


# T46: silent scope substitution. When a retrieval answers a materially
# different question than the one asked — a single-day request served by a
# monthly mean, a time range clamped to coverage, a region that resolved to a
# different place — the researcher must read that in the chat answer, not
# discover it in the Metadata tab's fine print. This is templated (never
# model-composed) for the same reason the taxonomy answers above are: the
# model can't paraphrase away an inconvenient correction it didn't want to
# admit. Rendered from provenance's requested_scope/delivered_scope facts in
# the dispatch layer (services/subagent_dispatch.py), the same seam that
# appends T18/T37 error answers.


def _date_part(value: str | None) -> str | None:
    """The YYYY-MM-DD of an ISO date or datetime, or None. Time-of-day is
    below the granularity any substitution disclosure cares about — a single-
    date request naturally widens to T00:00:00/T23:59:59, which is not a
    substitution to announce."""
    if not value:
        return None
    return str(value).split("T", 1)[0].strip() or None


def _requested_span(time_range: str | None) -> tuple[str | None, str | None]:
    """The (start, end) date parts of a requested ``start/end`` range (or a
    bare single date). Unparseable shapes yield (None, None) — nothing to
    compare against, so no note, rather than a false alarm."""
    if not time_range:
        return (None, None)
    parts = str(time_range).split("/", 1)
    start = _date_part(parts[0])
    end = _date_part(parts[1]) if len(parts) == 2 else start
    return (start, end)


def render_scope_note(requested_scope: dict | None, delivered_scope: dict | None) -> str | None:
    """A one-line disclosure when the delivered scope differs materially from
    the requested one, or ``None`` when they match (don't nag on an exact
    request). Filled only from observed facts — never a guess at why the
    substitution happened.

    "Differ materially" (decided against real T46 cases):
    - time: requested vs delivered date parts differ (any narrowing/widening,
      including a single day served by a whole month);
    - region: the delivered region/display name differs from the requested
      location string (exact-string fuzz — the goal is catching a
      *substitution*, not a synonym)."""
    requested_scope = requested_scope or {}
    delivered_scope = delivered_scope or {}

    clauses: list[str] = []

    req_start, req_end = _requested_span(requested_scope.get("time_range"))
    del_start = _date_part(delivered_scope.get("start_date"))
    del_end = _date_part(delivered_scope.get("end_date"))
    if req_start and del_start and (req_start, req_end) != (del_start, del_end):
        requested_time = req_start if req_start == req_end else f"{req_start} to {req_end}"
        delivered_time = del_start if del_start == del_end else f"{del_start} to {del_end}"
        cadence = str(delivered_scope.get("cadence") or "").strip()
        cadence_clause = f" (this is a {cadence} product)" if cadence and cadence.lower() != "unknown" else ""
        clauses.append(
            f"you asked for {requested_time}, but the data shown covers "
            f"{delivered_time}{cadence_clause}"
        )

    requested_location = str(requested_scope.get("location") or "").strip()
    delivered_region = str(delivered_scope.get("region_name") or "").strip()
    if requested_location and delivered_region and requested_location.lower() != delivered_region.lower():
        clauses.append(
            f"you asked about {requested_location}, but the data shown covers {delivered_region}"
        )

    if not clauses:
        return None
    return "Note: " + "; ".join(clauses) + "."
