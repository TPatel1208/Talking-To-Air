"""
domain/aggregation/variable_choice.py
=====================================
Turn a variable resolver ``Resolution`` into the deterministic, LLM-free
``VariableChoice`` picker payload the frontend renders.

The candidate options are the resolver's own ranked ``candidates``, their
units read straight off the file's CF attrs, and the message is a fixed
template filled only from counts. Nothing here is composed by the model.

The ``prompt`` on each option (the auto-send text that reconstructs the original
request with this variable substituted in) is left blank by
``build_variable_choice``: the original request text lives with the caller, which
fills each ``prompt`` via ``fill_prompts`` from the one request string it holds.
"""
from __future__ import annotations

import xarray as xr
from pydantic import BaseModel, Field

from tta_backend.domain.aggregation.variable_resolver import Resolution


class VariableChoiceOption(BaseModel):
    """One candidate the researcher may pick when the variable resolver
    couldn't confidently choose. Every field is a fact the resolver computed
    (name/category/units/valid_fraction/reasons) plus the pre-composed
    ``prompt`` the dispatch layer builds by templating the original request
    with this variable substituted in — clicking it auto-sends that prompt."""

    name: str
    category: str
    units: str | None = None
    valid_fraction: float | None = None
    reasons: list[str] = Field(default_factory=list)
    prompt: str


class VariableChoice(BaseModel):
    """The deterministic, LLM-free picker payload for a variable-choice
    turn. ``message`` is a template filled only from resolver facts (candidate
    count, excluded-empty count); ``candidates`` is the resolver's own ranked
    list, never capped and never paraphrased by the model. Rendered as inline
    chips (<=6) or a searchable, grouped modal (>6) by the frontend."""

    message: str
    candidates: list[VariableChoiceOption] = Field(default_factory=list)


def _units(ds: xr.Dataset, name: str) -> str | None:
    """The CF ``units`` attr of ``name`` in ``ds``, or None. A picker row shows
    units so a researcher can tell an AOD field from a reflectance without
    guessing -- read from the file's own metadata, never invented."""
    try:
        units = ds[name].attrs.get("units")
    except Exception:
        return None
    units = str(units).strip() if units not in (None, "") else ""
    return units or None


def build_variable_choice(resolution: Resolution, ds: xr.Dataset) -> VariableChoice:
    """Build the picker payload from a medium/low-confidence ``Resolution``.

    Candidates are the resolver's ranked ``candidates`` with any zero-valid
    entry filtered out (selecting one would plot blank, not offer a real
    choice) and no length cap -- a 400-variable file shows 400 rows for the
    frontend to group/search, never a truncated prose excerpt. ``prompt`` is
    left "" for the caller to fill (see module docstring).
    """
    options = [
        VariableChoiceOption(
            name=c.name,
            category=c.category,
            units=_units(ds, c.name),
            valid_fraction=c.valid_fraction,
            reasons=list(c.reasons),
            prompt="",
        )
        for c in resolution.candidates
        if c.valid_fraction is not None and c.valid_fraction > 0.0
    ]
    return VariableChoice(
        message=_message(resolution, len(options)),
        candidates=options,
    )


def fill_prompts(choice: VariableChoice, original_request: str) -> VariableChoice:
    """Return a copy of ``choice`` with each option's ``prompt`` composed from
    ``original_request`` and that option's variable name -- the full original
    request with the chosen variable appended (e.g. "plot NO2 over New Jersey
    last week" -> "... using tropospheric_NO2_column"), NOT a bare "use <var>".

    Reconstructing the whole request is deliberate: clicking a candidate
    re-sends a complete, self-contained turn with its own region/time, so the
    picker never depends on implicit context from earlier turns."""
    request = " ".join(str(original_request or "").split())
    filled = [
        opt.model_copy(update={"prompt": _compose_prompt(request, opt.name)})
        for opt in choice.candidates
    ]
    return choice.model_copy(update={"candidates": filled})


def _compose_prompt(request: str, name: str) -> str:
    """The auto-send text for one candidate. Empty request (nothing to
    reconstruct) falls back to a plain, still-actionable instruction rather
    than a leading ' using'."""
    if not request:
        return f"Use variable {name}."
    return f"{request} using {name}"


def _message(resolution: Resolution, shown: int) -> str:
    """The picker's headline. Low-confidence (no name) asks the researcher to
    choose outright; medium-confidence names the auto-pick and offers the
    alternatives. The empty-excluded count is disclosed either way so the
    candidate list reads as the populated subset, not the whole file."""
    excluded = _excluded_clause(resolution.excluded_empty_count)
    if resolution.name is None:
        return (
            f"This dataset has {shown} candidate variables I can't confidently "
            f"narrow down — pick one:{excluded}"
        )
    return (
        f"Showing {resolution.name} — I wasn't certain this is the right "
        f"variable. Pick another if not:{excluded}"
    )


def _excluded_clause(count: int) -> str:
    if count and count > 0:
        plural = "variable" if count == 1 else "variables"
        return f" ({count} {plural} excluded: no data in this range)"
    return ""
