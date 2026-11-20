"""
config/error_templates.py
===========================
Deterministic answers for chat-level failures where no model composed any
text: a turn that raised, ran out of time, or was shed by admission control.
The chat host (services/harness_chat.py) sends them with no model call.

Every template is filled only from observed facts (``stage``, ``detail``)
— never a guess at cause. This is what keeps the system honest when it's
wrong: it can misclassify a failure, but it can never narrate one.
"""
from __future__ import annotations

from tta_backend.earthdata_mcp.results import (
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


# Distinct from the taxonomy templates above — a turn deadline is not an
# MCPToolError category, it is the chat host giving up on the whole turn
# rather than one classified failure. Names any job handles still running
# server-side so a researcher staring at the timeout answer knows the Jobs
# panel, not a retry, is where to look.
_TURN_TIMEOUT_TEMPLATE = (
    "This request ran out of time before it could finish. {jobs_detail}"
    "This was not a problem with your question — try again, or check the "
    "Jobs panel for anything still running."
)


def render_turn_timeout_answer(job_handles: list[str] | None = None) -> str:
    jobs_detail = f"Still running: {', '.join(job_handles)}. " if job_handles else ""
    return _TURN_TIMEOUT_TEMPLATE.format(jobs_detail=jobs_detail)


# Admission control shed this turn: too many heavy reductions were already
# running or waiting (services/admission.py). Distinct from the timeout
# template above even though both end a turn, because the honest advice
# differs -- a timeout says "this may just be slow, check the Jobs panel",
# while this says "the server is busy, the same question will work shortly".
# Telling a researcher to check Jobs here would send them to an empty panel.
#
# Templated rather than model-composed for the reason the taxonomy answers
# are, and one more besides: the model never sees this. The turn ends before
# the supervisor could read a "busy" result and simply call the tool again --
# which would make an overloaded backend amplify its own load.
_OVERLOADED_TEMPLATE = (
    "The server is busy handling other large requests right now, so this one "
    "was not started. This was not a problem with your question — wait a "
    "moment and ask again. Narrowing the region or the time range also makes "
    "a request much cheaper to run."
)


def render_overloaded_answer() -> str:
    return _OVERLOADED_TEMPLATE
