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

_DEFAULT_DETAIL = "no further detail is available"


def render_error_answer(category: str, stage: str, detail: str | None = None) -> str:
    """A fixed, honest chat answer for ``category`` — filled only with
    observed facts (``stage``, ``detail``), no model in the loop. An
    unrecognized category still renders (falls back to the ``contract``
    template) rather than raising, matching the classifier's own
    additive-safe rule."""
    template = _TEMPLATES.get(category, _TEMPLATES[CATEGORY_CONTRACT])
    return template.format(stage=stage, detail=detail or _DEFAULT_DETAIL)
