"""Building earthdata retrieval requests and reading job status, from registry facts.

Pure functions: no connection, no emitters, no per-request state. The current
system's composites and the harness's satellite source share them.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal

from tta_backend.datasets.registry import CollectionConfig, load_registry

# "not_found" is a handle the workspace still lists after the service evicted
# its job; it never changes again, so it counts as terminal.
TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled", "not_found"}

# Consecutive transient status-poll failures tolerated before the error surfaces.
POLL_TRANSIENT_FAILURE_LIMIT = 3

# The provider pauses jobs it considers too large to run unattended. The
# service reports such a job as "running"; only its message says "paused".
PAUSED_STATUS = "paused"

_PAUSED_MESSAGE_RE = re.compile(r"\bpaused\b", re.IGNORECASE)

PAUSED_NOTE = (
    "The data provider paused this job — usually because the request was too "
    "large to process automatically. It will not finish on its own: cancel "
    "it, then narrow the time range or area and try again."
)


def annotate_paused(status_data: dict[str, Any]) -> dict[str, Any]:
    """A non-terminal status whose provider message says the job is paused
    becomes status ``"paused"``, with a ``"paused at provider"`` phase and a
    plain-language ``note``. Anything else passes through untouched."""
    status = status_data.get("status", "")
    message = status_data.get("message") or ""
    if status in TERMINAL_STATUSES or not _PAUSED_MESSAGE_RE.search(message):
        return status_data
    return {
        **status_data,
        "status": PAUSED_STATUS,
        "phase": "paused at provider",
        "note": PAUSED_NOTE,
    }


def supports_variable_subsetting(variables: list[str]) -> bool:
    """False only when every requested variable resolves to a collection
    marked ``supports_variable_subsetting: false``. Matched by variable short
    name, since a caller may hold only an opaque dataset handle; an
    unregistered name defaults to True."""
    index: dict[str, bool] = {}
    for cfg in load_registry().values():
        names = {cfg.primary_var}
        if cfg.quality_flag_var:
            names.add(cfg.quality_flag_var)
        names.update(v.rsplit("/", 1)[-1] for v in cfg.variables)
        for name in names:
            index[name] = cfg.supports_variable_subsetting or index.get(name, False)

    resolved = [index[v] for v in variables if v in index]
    return all(resolved) if resolved else True


def _leaf(name: str | None) -> str:
    return str(name or "").rsplit("/", 1)[-1]


def pinned_companion_variables(variables: list[str]) -> list[str]:
    """The companions each requested science variable cannot be interpreted
    without: its collection's quality flag (without which it cannot be masked)
    and, for a layered product, its vertical axes (without which it has nothing
    to be plotted against).

    Matched by bare leaf name, since a caller may spell a variable
    group-qualified or bare. Each addition uses the spelling the matching
    request used; a companion in a different group from the science variable
    keeps the registry's own qualified spelling.
    """
    requested = {_leaf(v) for v in variables}
    by_leaf = {_leaf(v): v for v in variables}
    additions: list[str] = []
    for cfg in load_registry().values():
        companions: list[str] = [c for c in (cfg.quality_flag_var, *cfg.vertical_axis_vars) if c]
        if not companions:
            continue
        collection_leaves = {_leaf(cfg.primary_var)} | {_leaf(v) for v in cfg.variables}
        matched = requested & collection_leaves
        if not matched:
            continue
        science_as_asked = by_leaf[sorted(matched)[0]]
        for companion in companions:
            companion_leaf = _leaf(companion)
            if companion_leaf in requested:
                continue                  # the caller already asked for it
            if "/" not in science_as_asked and "/" not in companion:
                spelling = companion_leaf
            else:
                # Prefer the registry's own qualified spelling; fall back to the
                # group the caller addressed the science variable through.
                spelling = next(
                    (v for v in cfg.variables if _leaf(v) == companion_leaf and "/" in v),
                    f"{science_as_asked.rsplit('/', 1)[0]}/{companion_leaf}"
                    if "/" in science_as_asked else companion_leaf,
                )
            if spelling not in additions:
                additions.append(spelling)
    return additions


def normalize_time_range(time_range: str) -> str:
    """Widen a degenerate ``start/end`` interval where start == end, which the
    provider rejects: a date-only pair becomes that whole day, a timestamped
    pair gets one second of width. Anything unparseable is returned unchanged
    for the service's own validation to reject."""
    parts = time_range.split("/", 1)
    if len(parts) != 2 or parts[0] != parts[1]:
        return time_range
    start = parts[0].strip()
    try:
        parsed = datetime.fromisoformat(start)
    except ValueError:
        return time_range
    if "T" not in start:
        return f"{start}T00:00:00/{start}T23:59:59"
    return f"{start}/{(parsed + timedelta(seconds=1)).isoformat()}"


def time_span_days(time_range: str) -> int | None:
    """Days between an ISO 8601 'start/end' interval's two ends, or None when
    ``time_range`` is not in that shape."""
    try:
        start_str, end_str = time_range.split("/", 1)
        return (datetime.fromisoformat(end_str) - datetime.fromisoformat(start_str)).days
    except (ValueError, AttributeError):
        return None


def day_time_range(day: date) -> str:
    """The whole UTC day, as the service's ISO 8601 interval."""
    return f"{day.isoformat()}T00:00:00/{day.isoformat()}T23:59:59"


def span_time_range(start: date, end: date) -> str:
    """Every UTC day from ``start`` to ``end`` inclusive, as an ISO 8601 interval."""
    return f"{start.isoformat()}T00:00:00/{end.isoformat()}T23:59:59"


def request_variables(cfg: CollectionConfig) -> tuple[str, list[str]]:
    """The science variable a retrieval of ``cfg`` is opened on (its bare leaf
    name, as opened files name it) and the variable list the request sends.

    The list is the registry's science variable plus its pinned companions,
    or empty (the whole collection) when the collection does not support
    variable subsetting.
    """
    science = [cfg.primary_var]
    if not supports_variable_subsetting(science):
        return _leaf(cfg.primary_var), []
    return _leaf(cfg.primary_var), [*science, *pinned_companion_variables(science)]


SizeDecision = Literal["proceed", "confirm", "refuse"]


def size_gate(estimated_bytes: int | None, *, soft: int, hard: int, confirmed: bool) -> SizeDecision:
    """Whether a retrieval of ``estimated_bytes`` may proceed.

    Above the hard cap it is refused even when confirmed. Between the caps,
    or with no estimate at all, it needs the user's confirmation.
    """
    if estimated_bytes is not None and estimated_bytes > hard:
        return "refuse"
    if confirmed:
        return "proceed"
    if estimated_bytes is None or estimated_bytes > soft:
        return "confirm"
    return "proceed"


def bbox_location(bounds: Sequence[float]) -> str:
    """``"W,S,E,N"`` at six decimals, rounded outward, so the area the service
    retrieves always contains the area that is masked."""
    west, south, east, north = bounds
    return ",".join([
        f"{math.floor(west * 1e6) / 1e6:.6f}",
        f"{math.floor(south * 1e6) / 1e6:.6f}",
        f"{math.ceil(east * 1e6) / 1e6:.6f}",
        f"{math.ceil(north * 1e6) / 1e6:.6f}",
    ])


def _shift_month(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def nearby_dates(day: date, cadence: str, *, today: date, end: date | None = None) -> list[date]:
    """Dates to look at when ``day`` (or the span from ``day`` to ``end``) has
    no data, nearest first and never after ``today``: three days either side
    for hourly and daily data, the month either side for monthly data."""
    last = day if end is None else end
    if cadence == "monthly":
        candidates = [_shift_month(day, -1), _shift_month(last, 1)]
    else:
        candidates = [
            (day - timedelta(days=k)) if sign < 0 else (last + timedelta(days=k))
            for k in (1, 2, 3) for sign in (-1, 1)
        ]
    return [d for d in candidates if d <= today]


def search_hit(entry: dict[str, Any]) -> dict[str, Any]:
    """One ``search_datasets`` hit's fields in one mapping. The service lists
    each hit as ``{"handle", "summary": {"concept_id", "short_name",
    "entry_title", ...}}``; fields given beside the summary are read too."""
    summary = entry.get("summary")
    return {**entry, **summary} if isinstance(summary, dict) else dict(entry)


def search_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every hit of a ``search_datasets`` reply, each read with ``search_hit``."""
    entries = result.get("datasets") or result.get("results") or []
    return [search_hit(entry) for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def dataset_handle_from_search(result: dict[str, Any], collection_id: str) -> str | None:
    """The handle of the ``search_datasets`` hit whose concept id is exactly
    ``collection_id``, or None when no hit is that collection."""
    candidates = search_hits(result)
    if not candidates and isinstance(result.get("concept_id"), str):
        candidates = [result]
    for entry in candidates:
        if entry.get("concept_id") != collection_id:
            continue
        handle = entry.get("handle") or entry.get("dataset_handle")
        if isinstance(handle, str) and handle:
            return handle
    return None


def aoi_handle_from_result(result: dict[str, Any]) -> str | None:
    """The area-of-interest handle in a ``define_area_of_interest`` result."""
    handle = result.get("handle") or result.get("aoi_handle")
    return handle if isinstance(handle, str) and handle else None


# The service reports sizes in megabytes; read as mebibytes so a size checked
# against a byte limit is never under-counted.
_MEGABYTE = 1024 * 1024


def estimated_bytes(result: dict[str, Any]) -> int | None:
    """The bytes an ``estimate_retrieval_size`` result reports, or None when it cannot say.

    The service gives ``total_size_mb``: the sampled granules' mean size times
    the granule count, and 0 when no sampled granule reports a size. So 0 with
    granules to fetch means unknown, and 0 with none means nothing to fetch.
    """
    given = result.get("estimated_bytes")
    if isinstance(given, int | float) and not isinstance(given, bool) and given >= 0:
        return int(given)
    size = result.get("total_size_mb")
    if isinstance(size, bool) or not isinstance(size, int | float) or size < 0:
        return None
    if size > 0:
        return round(size * _MEGABYTE)
    granules = result.get("total_granules")
    return 0 if granules == 0 else None


def granule_count(result: dict[str, Any]) -> int | None:
    """The granule count a ``check_coverage`` result reports, or None when it reports none."""
    count = result.get("granule_count")
    if isinstance(count, bool) or not isinstance(count, int | float) or count < 0:
        return None
    return int(count)
