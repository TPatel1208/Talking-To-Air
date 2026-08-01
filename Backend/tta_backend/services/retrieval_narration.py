"""
services/retrieval_narration.py
=================================
What to *call* an in-flight retrieval while the researcher waits for it.

A Harmony job spends minutes in "materializing", and for all of it the chat
had one thing to say: "Retrieving data — running...". The wait didn't feel
long because it *was* long so much as because it was informationally empty —
every fact that would make it legible (which variable, which place, which
dates, roughly how many bytes) is known synchronously at submission time and
was simply dropped on the floor. This module keeps those facts for the length
of the wait and renders them as one line.

Same two-step, per-process, TTL-bounded shape as its two sibling registries
(``scope_registry``, ``variable_choice_registry``): ``record`` at submit,
keyed by the job_handle, because the ``obs_``/``cube_`` handle a retrieval
resolves to is only known once the job reaches "ready".

**Unlike** those two there is no ``finalize``. A narration describes the
*wait*, not the result, so nothing downstream ever reads it against a result
handle — it is ``discard``-ed the moment the job goes terminal. That, plus an
expiry sweep for jobs that never reach a terminal state at all (a timed-out
await never comes back to discard), is what bounds the dict.

Nothing here is a system of record. Losing it degrades the status line back to
the bare "Retrieving data" wording — never a wrong answer, and never a failed
turn. That posture is deliberate: this is cosmetic, and cosmetics must not be
able to cost a researcher their result.
"""
from __future__ import annotations

import time
from datetime import date, datetime

_TTL_SECONDS = 60 * 60

_narrations: dict[str, tuple[dict, float]] = {}

# Spelled out rather than taken from ``strftime("%b")``, which is locale-
# dependent: the backend container and a developer's machine would render
# different months, and a status line is not worth a locale bug.
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Facts are joined with a middle dot, not a comma: a location is very often
# itself comma-shaped ("Newark, NJ"), and comma-joining it against the other
# facts produces a line whose structure the reader has to guess at. It also
# leaves the em-dash free to separate the subject from the job's phase in the
# one caller that composes both (services.retrieval_composites.await_retrieval).
_SEPARATOR = " · "


def record(
    job_handle: str | None,
    *,
    variable: str | None = None,
    location: str | None = None,
    time_range: str | None = None,
    estimated_bytes: int | None = None,
) -> None:
    """Remember what ``job_handle``'s retrieval is for, as of submission.

    Every argument is optional because the two call sites genuinely know
    different things: ``safe_retrieve`` holds a size estimate but only an
    opaque ``aoi_handle`` (never a place name), while ``point_timeseries``
    holds the place name but has no size to estimate. A no-op when the handle
    is falsy (a submit that returned none is a contract error its caller
    raises on — narration must not be what fails first) or when no argument
    carried a usable fact (an empty subject reads worse than the default
    wording it would replace).
    """
    if not job_handle:
        return
    facts = {
        # Retrievals request group-qualified names ("product/foo"); the group
        # is plumbing the researcher never typed.
        "variable": variable.rsplit("/", 1)[-1] if variable else None,
        "location": location or None,
        "time_range": time_range or None,
        "estimated_bytes": estimated_bytes,
    }
    facts = {key: value for key, value in facts.items() if value}
    if not facts:
        return
    _sweep_expired()
    _narrations[job_handle] = (facts, time.time() + _TTL_SECONDS)


def describe(job_handle: str) -> str | None:
    """One line naming what ``job_handle`` is retrieving, or None if nothing
    was recorded for it or the entry has expired.

    None is a real answer, not a failure: ``open_handle``'s rematerialize path
    awaits jobs this process never submitted, and the caller falls back to its
    own generic wording.
    """
    entry = _narrations.get(job_handle)
    if entry is None:
        return None
    facts, expires_at = entry
    if expires_at <= time.time():
        _narrations.pop(job_handle, None)
        return None

    # The variable leads: it is the thing the researcher actually asked about.
    # The rest are qualifiers that bound the wait, in narrowing order.
    parts = [
        facts.get("variable"),
        facts.get("location"),
        _format_time_range(facts.get("time_range")),
        _format_bytes(facts.get("estimated_bytes")),
    ]
    rendered = [part for part in parts if part]
    return _SEPARATOR.join(rendered) if rendered else None


def discard(job_handle: str) -> None:
    """Forget ``job_handle``'s narration once its job is terminal. Harmless
    for a job that was never recorded — ``await_retrieval`` calls this on every
    terminal poll, including for jobs it did not submit."""
    _narrations.pop(job_handle, None)


def _sweep_expired() -> None:
    """Drop entries whose TTL has passed.

    Needed because ``discard`` only fires on a terminal poll: a job that times
    out, or an await abandoned when its turn was cancelled, never comes back to
    clean up after itself. Run on ``record`` (the only write) so the sweep is
    paid for by the thing that grows the dict, and is bounded by how many
    retrievals a process submits in a TTL.
    """
    now = time.time()
    for handle in [h for h, (_facts, expires_at) in _narrations.items() if expires_at <= now]:
        _narrations.pop(handle, None)


def _format_time_range(time_range: str | None) -> str | None:
    """An ISO ``start/end`` interval as a short human date range.

    Returns None for anything unparseable rather than echoing the raw string:
    narration is cosmetic, a malformed range means the MCP's own ``time_range``
    validation is about to reject the job with a far more specific message, and
    putting ISO plumbing in front of the researcher buys nothing.
    """
    if not time_range:
        return None
    parts = str(time_range).split("/", 1)
    try:
        start = datetime.fromisoformat(parts[0].strip()).date()
        end = datetime.fromisoformat(parts[1].strip()).date() if len(parts) == 2 else start
    except (ValueError, IndexError):
        return None

    if start == end:
        return f"{_month_day(start)}, {start.year}"
    if start.year != end.year:
        return f"{_month_day(start)}, {start.year} – {_month_day(end)}, {end.year}"
    # Same year, so state it once at the end; same month, so don't repeat it.
    if start.month == end.month:
        return f"{_month_day(start)}–{end.day}, {start.year}"
    return f"{_month_day(start)} – {_month_day(end)}, {start.year}"


def _month_day(value: date) -> str:
    return f"{_MONTH_ABBR[value.month - 1]} {value.day}"


def _format_bytes(count: int | None) -> str | None:
    """A byte count at a readable scale, or None when there is no number to
    show.

    Carries a "~" and at most one decimal on purpose: this is the provider's
    *estimate*, and rendering it to the byte would imply a precision the
    estimator never claimed. A missing estimate (``estimate_retrieval_size``
    can decline to price a request at all) is not zero, and neither is
    reported.
    """
    if count is None or count <= 0:
        return None
    for unit, size in (("GB", 1000 ** 3), ("MB", 1000 ** 2), ("kB", 1000)):
        if count >= size:
            value = count / size
            return f"~{value:.0f} {unit}" if value >= 10 else f"~{value:.1f} {unit}"
    return f"~{int(count)} B"
