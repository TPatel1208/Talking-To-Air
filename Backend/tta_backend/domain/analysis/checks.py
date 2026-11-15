"""Whether two fields can be compared, and the named reason when they cannot.

Each check returns None when the fields agree, or a ``Mismatch`` naming the
reason and the facts it observed; callers word their own refusal from those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

MismatchReason = Literal["variable_mismatch", "unit_mismatch", "unit_missing", "disjoint_periods", "grid_mismatch"]

# Two grids match when, where their extents overlap, every coordinate of one
# has a coordinate of the other within this fraction of the grid spacing.
GRID_MATCH_FRACTION = 0.1
# The tolerance for an axis with a single coordinate on both sides.
_SINGLE_COORDINATE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Mismatch:
    reason: MismatchReason
    facts: dict[str, str]


def variable_mismatch(name_a: Any, name_b: Any) -> Mismatch | None:
    """Two named variables that differ; an unnamed side is not a mismatch."""
    a = str(name_a or "").strip()
    b = str(name_b or "").strip()
    if a and b and a != b:
        return Mismatch("variable_mismatch", {"a": a, "b": b})
    return None


def units_mismatch(units_a: Any, units_b: Any) -> Mismatch | None:
    """Units that differ, or units declared on only one side.

    Nothing is converted. One side without units cannot be confirmed
    commensurable with the other, so it is a mismatch too (``unit_missing``,
    naming the declared unit and the side missing one); only when neither side
    declares units do the fields compare.
    """
    a = str(units_a or "").strip()
    b = str(units_b or "").strip()
    if a and b:
        return Mismatch("unit_mismatch", {"a": a, "b": b}) if a != b else None
    if a or b:
        declared, missing_side = (a, "B") if a else (b, "A")
        return Mismatch("unit_missing", {"declared": declared, "missing_side": missing_side})
    return None


def axis_spacing(values: Any) -> float:
    """The smallest step between distinct coordinates of an axis; 0 for fewer than two."""
    coords = np.unique(np.asarray(values, dtype=float))
    return float(np.diff(coords).min()) if coords.size > 1 else 0.0


def _unmatched(x: np.ndarray, y: np.ndarray, tolerance: float) -> bool:
    """Whether a coordinate of ``x`` inside ``y``'s range has no coordinate of ``y`` within ``tolerance``."""
    inside = x[(x >= y[0] - tolerance) & (x <= y[-1] + tolerance)]
    if inside.size == 0:
        return False
    right = np.clip(np.searchsorted(y, inside), 0, y.size - 1)
    left = np.clip(right - 1, 0, y.size - 1)
    nearest = np.minimum(np.abs(inside - y[right]), np.abs(inside - y[left]))
    return bool((nearest > tolerance).any())


def _axes_match(a: Any, b: Any) -> bool:
    x, y = np.unique(np.asarray(a, dtype=float)), np.unique(np.asarray(b, dtype=float))
    if x.size == 0 or y.size == 0:
        return True
    steps = [s for s in (axis_spacing(x), axis_spacing(y)) if s > 0]
    tolerance = GRID_MATCH_FRACTION * min(steps) if steps else _SINGLE_COORDINATE_TOLERANCE
    return not (_unmatched(x, y, tolerance) or _unmatched(y, x, tolerance))


def _spacing_text(lat: Any, lon: Any) -> str:
    return f"{axis_spacing(lat):g} by {axis_spacing(lon):g} degree"


def grid_mismatch(lat_a: Any, lon_a: Any, lat_b: Any, lon_b: Any) -> Mismatch | None:
    """Two grids whose cells do not line up: where their extents overlap, a
    latitude or longitude of one has no counterpart in the other within
    ``GRID_MATCH_FRACTION`` of the grid spacing, as for grids of different
    resolution or offset from each other. Grids that do not overlap match.

    The facts are each grid's spacing.
    """
    if _axes_match(lat_a, lat_b) and _axes_match(lon_a, lon_b):
        return None
    return Mismatch("grid_mismatch", {"spacing_a": _spacing_text(lat_a, lon_a), "spacing_b": _spacing_text(lat_b, lon_b)})


def has_valid_data(da: xr.DataArray) -> bool:
    """Whether any value of ``da`` is finite."""
    return bool(np.isfinite(np.asarray(da.values, dtype=float)).any())


def time_range(da: xr.DataArray) -> tuple[str, str] | tuple[None, None]:
    """The first and last value of ``da``'s ``time`` coordinate, as text, or (None, None)."""
    if "time" not in da.coords:
        return None, None
    times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
    if not times:
        return None, None
    return times[0], times[-1]


def disjoint_periods(
    range_a: tuple[str | None, str | None], range_b: tuple[str | None, str | None],
) -> Mismatch | None:
    """Two time ranges that do not overlap at all; an unknown range is not a mismatch.

    The facts are the ranges' dates (first ten characters).
    """
    start_a, end_a = range_a
    start_b, end_b = range_b
    if start_a is None or end_a is None or start_b is None or end_b is None:
        return None
    if end_a < start_b or end_b < start_a:
        return Mismatch(
            "disjoint_periods",
            {"start_a": start_a[:10], "end_a": end_a[:10], "start_b": start_b[:10], "end_b": end_b[:10]},
        )
    return None
