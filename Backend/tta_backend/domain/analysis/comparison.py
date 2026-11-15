"""Differences between two fields and the statistics that describe them.

A difference is B minus A, cell by cell; a cell missing on either side is
missing in the difference and in every statistic computed over it. Means are
cos(latitude) area-weighted (``area_weighted_mean``).
"""
from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import xarray as xr

from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

# Percent change is withheld when A's mean is smaller than this fraction of
# A's own 2-98th-percentile magnitude range: relative to the field's own
# scale, so the guard holds across variables of very different magnitudes.
PERCENT_CHANGE_FLOOR_FRACTION = 0.05

# An aligned retrieval arrives as either; selection keeps whichever it was.
_XrT = TypeVar("_XrT", xr.DataArray, xr.Dataset)
# How one side is picked out of an aligned object: ("sel", label) or ("isel", position).
SourceSelector = tuple[str, Any]


def difference(da_a: xr.DataArray, da_b: xr.DataArray) -> xr.DataArray:
    """B minus A, cell by cell; NaN on either side stays NaN, never a 0."""
    return da_b - da_a


def percent_change_floor(a_paired_vals: np.ndarray) -> float | None:
    """The smallest A mean magnitude a percent change may be normalized by:
    ``PERCENT_CHANGE_FLOOR_FRACTION`` of |A|'s 2-98th-percentile range over
    the paired cells. None when A has no valid cells."""
    valid = np.abs(a_paired_vals[np.isfinite(a_paired_vals)])
    if valid.size == 0:
        return None
    spread = float(np.percentile(valid, 98) - np.percentile(valid, 2))
    return PERCENT_CHANGE_FLOOR_FRACTION * spread


def anomaly_stats(da_a: xr.DataArray, da_b: xr.DataArray, diff: xr.DataArray, threshold: float | None) -> dict[str, Any]:
    """Mean difference, percent change relative to A's mean, and optionally the
    area whose change magnitude reaches ``threshold``, over cells valid on both sides.

    Percent change is None, with a ``percent_change_note``, when A's mean is
    zero or under ``percent_change_floor``; the mean difference is always given.
    """
    diff_vals = np.asarray(diff.values, dtype=float)
    valid_diff = diff_vals[np.isfinite(diff_vals)]

    # A on the paired cells, kept as a DataArray so the weighted mean sees latitude.
    a_paired = da_a.where(diff.notnull())
    a_paired_vals = np.asarray(a_paired.values, dtype=float)

    mean_difference = area_weighted_mean(diff) if valid_diff.size else None
    mean_a = area_weighted_mean(a_paired) if np.isfinite(a_paired_vals).any() else None

    percent_change = None
    percent_change_note = None
    if mean_difference is not None and mean_a is not None:
        floor = percent_change_floor(a_paired_vals)
        if mean_a == 0.0 or (floor is not None and abs(mean_a) < floor):
            percent_change_note = (
                "Percent change withheld — baseline too small to normalize by: period A's "
                f"area-weighted mean ({mean_a:.3g}) is under {PERCENT_CHANGE_FLOOR_FRACTION:.0%} of "
                "the field's own 2–98th-percentile range, where a percentage would explode. "
                "The absolute mean difference is reported instead."
            )
        else:
            percent_change = (mean_difference / mean_a) * 100.0

    stats: dict[str, Any] = {
        "n_cells": int(valid_diff.size),
        "mean_difference": mean_difference,
        "percent_change": percent_change,
    }
    if percent_change_note is not None:
        stats["percent_change_note"] = percent_change_note

    if threshold is not None:
        exceeding = valid_diff[np.abs(valid_diff) >= threshold]
        stats["area_exceeding_threshold"] = {
            "threshold": threshold,
            "n_cells": int(exceeding.size),
            "fraction": (exceeding.size / valid_diff.size) if valid_diff.size else 0.0,
        }

    return stats


def region_stats(da: xr.DataArray, *, basis: str | None = None) -> dict[str, Any] | None:
    """Mean (area-weighted), median, max, min and cell count over ``da``'s
    valid cells, or None when none are valid. ``basis``, when given, names the
    field the statistics describe, such as ``period-mean``."""
    values = np.asarray(da.values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    stats: dict[str, Any] = {
        "mean": area_weighted_mean(da),
        "median": float(np.median(valid)),
        "max": float(np.max(valid)),
        "min": float(np.min(valid)),
        "n_pixels": int(valid.size),
    }
    if basis:
        stats["basis"] = basis
    return stats


def source_labels(aligned: xr.DataArray | xr.Dataset) -> list[str] | None:
    """The ``source`` coordinate's values as text, or None without one."""
    if "source" not in getattr(aligned, "coords", {}):
        return None
    return [str(v) for v in np.atleast_1d(aligned["source"].values)]


def aligned_source_order(
    aligned: xr.DataArray | xr.Dataset, handle_a: str | None, handle_b: str | None,
) -> tuple[SourceSelector, SourceSelector]:
    """The (A, B) selectors for an aligned cube: by label when the ``source``
    coordinate names both handles, otherwise by position (0 is A, 1 is B)."""
    labels = source_labels(aligned)
    if handle_a and handle_b and labels is not None and handle_a in labels and handle_b in labels:
        return ("sel", handle_a), ("sel", handle_b)
    return ("isel", 0), ("isel", 1)


def apply_source_selector(obj: _XrT, selector: SourceSelector) -> _XrT:
    kind, key = selector
    return obj.sel(source=key) if kind == "sel" else obj.isel(source=key)


def split_aligned(aligned: _XrT, handle_a: str | None = None, handle_b: str | None = None) -> tuple[_XrT, _XrT]:
    """The two sources of an aligned cube with a ``source`` dimension of length 2.

    Selected by label when the cube names ``handle_a`` and ``handle_b`` on its
    ``source`` coordinate, so a reordered cube cannot flip the difference's
    sign; otherwise by position, which relies on the aligner keeping input order.
    Raises ValueError for a cube without exactly two sources.
    """
    if "source" not in aligned.dims:
        raise ValueError(f"Aligned result has no 'source' dimension: dims={list(aligned.dims)}")
    n_sources = aligned.sizes["source"]
    if n_sources != 2:
        raise ValueError(f"Expected an aligned result with 2 sources, found {n_sources}.")
    sel_a, sel_b = aligned_source_order(aligned, handle_a, handle_b)
    return apply_source_selector(aligned, sel_a), apply_source_selector(aligned, sel_b)


def _spacing(values: np.ndarray) -> float:
    steps = np.abs(np.diff(np.sort(values)))
    steps = steps[steps > 0]
    return float(steps.min()) if steps.size else 0.0


def pair_cells(da_a: xr.DataArray, da_b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """A and B on A's cells: B's nearest cell within half of A's grid spacing,
    missing where B has no cell that close. Both are (lat, lon) fields."""
    lat_a, lon_a = np.asarray(da_a["lat"].values, dtype=float), np.asarray(da_a["lon"].values, dtype=float)
    lat_b, lon_b = np.asarray(da_b["lat"].values, dtype=float), np.asarray(da_b["lon"].values, dtype=float)
    if lat_a.shape == lat_b.shape and lon_a.shape == lon_b.shape and np.array_equal(lat_a, lat_b) and np.array_equal(lon_a, lon_b):
        return da_a, da_b.assign_coords(lat=da_a["lat"], lon=da_a["lon"])
    if lat_b.size == 0 or lon_b.size == 0:
        return da_a, da_a.where(False)
    b = da_b.sortby("lat").sortby("lon")
    lat_tol = max(_spacing(lat_a) / 2, 1e-6)
    lon_tol = max(_spacing(lon_a) / 2, 1e-6)
    paired = b.reindex(lat=da_a["lat"], method="nearest", tolerance=lat_tol)
    paired = paired.reindex(lon=da_a["lon"], method="nearest", tolerance=lon_tol)
    paired.attrs = dict(da_b.attrs)
    paired.name = da_b.name
    return da_a, paired


def shared_cells(da_a: xr.DataArray, da_b: xr.DataArray) -> int:
    """How many of A's cells have a cell of B paired with them, valid or not."""
    present = xr.ones_like(da_b, dtype=float)
    _, paired = pair_cells(da_a, present)
    return int(np.isfinite(np.asarray(paired.values, dtype=float)).sum())


def percent_change_field(da_a: xr.DataArray, da_b: xr.DataArray) -> tuple[xr.DataArray, int]:
    """100 x (B - A) / A per cell, and how many paired cells were withheld.

    A cell is withheld (missing) where |A| is zero or under the floor
    ``percent_change_floor`` sets from A's paired cells.
    """
    diff = difference(da_a, da_b)
    a_paired = da_a.where(diff.notnull())
    floor = percent_change_floor(np.asarray(a_paired.values, dtype=float)) or 0.0
    magnitude = np.abs(a_paired)
    keep = (magnitude > 0) & (magnitude >= floor)
    percent = (diff / a_paired.where(keep)) * 100.0
    withheld = int((diff.notnull() & ~keep).sum())
    return percent, withheld


def _location(field: xr.DataArray, index: int) -> dict[str, float]:
    """The cell at flat ``index`` of a (lat, lon) field."""
    lat_pos, lon_pos = np.unravel_index(index, field.shape)
    return {"lat": float(field["lat"].values[lat_pos]), "lon": float(field["lon"].values[lon_pos])}


def field_summary(da: xr.DataArray) -> dict[str, Any] | None:
    """Mean (area-weighted), min, max, their cells and the valid cell count of
    a (lat, lon) field, or None when no cell is valid."""
    field = da.transpose("lat", "lon")
    values = np.asarray(field.values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    masked = np.where(finite, values, np.nan)
    high, low = int(np.nanargmax(masked)), int(np.nanargmin(masked))
    return {
        "mean": area_weighted_mean(field),
        "min": float(masked.flat[low]),
        "max": float(masked.flat[high]),
        "max_location": _location(field, high),
        "min_location": _location(field, low),
        "cells": int(finite.sum()),
    }
