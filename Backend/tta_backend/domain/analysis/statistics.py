"""Statistics of a field or a series, each computed on the basis true to its name, and peaks.

Over a field with a time axis: max and min reduce time with the same
statistic, so they are the extremes over every valid observation; the mean is
the cos(latitude) area-weighted mean of the temporal-mean field; median, std
and percentiles are of the temporal-mean field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from tta_backend.domain.aggregation.aggregation_service import AggregationService, area_weighted_mean
from tta_backend.domain.analysis.reduction import reduce_over_time
from tta_backend.domain.geo import find_lat_coord, find_lon_coord

VALID_STATS = {"mean", "median", "max", "min", "std"}

_service = AggregationService()


def basis_statistic(stat: str, mean_field: xr.DataArray, extremum_field: xr.DataArray | None = None) -> tuple[float, str]:
    """``stat`` of a region and the basis it was computed on.

    ``mean_field`` is the temporal-mean field; ``extremum_field`` is the field
    reduced over time with ``stat`` itself, needed for max and min. Raises
    ValueError when no finite value remains.
    """
    if stat in ("max", "min"):
        if extremum_field is None:
            raise ValueError(f"{stat} needs the field reduced over time with {stat}")
        field_values = extremum_field.values
        return (
            _service.compute_values_stat(field_values[np.isfinite(field_values)], stat),
            f"{stat} over every valid observation (all timesteps)",
        )
    if stat == "mean":
        return area_weighted_mean(mean_field), "cos(latitude) area-weighted spatial mean of the temporal-mean field"
    values = mean_field.values
    valid = values[np.isfinite(values)]
    return _service.compute_values_stat(valid, stat), f"{stat} of the temporal-mean field (per-cell mean over time first)"


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _spread(values: np.ndarray) -> float | None:
    """The sample standard deviation, or None for fewer than two values."""
    return _finite_or_none(_service.compute_values_stat(values, "std")) if values.size > 1 else None


GRID_SPREAD_BASIS = "sample standard deviation across the valid cells of the temporal-mean field"
SERIES_SPREAD_BASIS = "sample standard deviation of the valid values"


def grid_statistic(da: xr.DataArray, stat: str, cadence: str, percentile: float | None = None) -> dict[str, Any] | None:
    """``stat`` (or the ``percentile``, 0-100) of a masked field and its basis,
    with the ``spread`` (``spread_basis``: the sample standard deviation across
    the temporal-mean field's valid cells) and ``count`` of those cells.

    The spread describes how the field varies across the area; it is not an
    uncertainty of the statistic. None when no cell holds a valid value.
    """
    mean_field, _ = reduce_over_time(da, "mean", cadence)
    values = np.asarray(mean_field.values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    if stat == "percentile":
        if percentile is None:
            raise ValueError("a percentile needs its rank")
        value = float(np.percentile(valid, percentile))
        basis = f"{percentile:g}th percentile of the temporal-mean field (per-cell mean over time first)"
    else:
        extremum = reduce_over_time(da, stat, cadence)[0] if stat in ("max", "min") else None
        value, basis = basis_statistic(stat, mean_field, extremum)
    return {
        "value": value, "basis": basis, "spread": _spread(valid), "spread_basis": GRID_SPREAD_BASIS,
        "count": int(valid.size),
    }


def series_statistic(values: np.ndarray, stat: str, percentile: float | None = None) -> dict[str, Any] | None:
    """``stat`` (or the ``percentile``) over every finite value and its basis,
    with the ``spread`` (``spread_basis``: their sample standard deviation) and
    ``count`` of those values; not an uncertainty. None when none is finite."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    if stat == "percentile":
        if percentile is None:
            raise ValueError("a percentile needs its rank")
        value = float(np.percentile(valid, percentile))
        basis = f"{percentile:g}th percentile of every valid value"
    else:
        value = _service.compute_values_stat(valid, stat)
        basis = f"{stat} of every valid value"
    return {
        "value": value, "basis": basis, "spread": _spread(valid), "spread_basis": SERIES_SPREAD_BASIS,
        "count": int(valid.size),
    }


class LatLonNotFound(ValueError):
    """A field without latitude and longitude dimensions."""


class PeakCoordinatesUnreadable(Exception):
    """The peak's latitude or longitude could not be read from the field's
    coordinates; its message is the underlying error's."""


@dataclass(frozen=True)
class Peak:
    value: float
    lat: float
    lon: float


def peak_cell(field: xr.DataArray) -> Peak | None:
    """The highest valid cell of a 2-D (lat, lon) field, or None when no cell is valid.

    Raises LatLonNotFound when the field has no latitude and longitude
    dimensions, and PeakCoordinatesUnreadable when an IndexError or TypeError
    reading the peak's coordinates stops it; any other error propagates.
    """
    lat_name = find_lat_coord(field)
    lon_name = find_lon_coord(field)
    lat_dim = lat_name if lat_name in field.dims else None
    lon_dim = lon_name if lon_name in field.dims else None
    if lat_dim is None or lon_dim is None:
        raise LatLonNotFound(f"no lat/lon dimensions in {list(field.dims)}")

    lat_array = field[lat_dim].values
    lon_array = field[lon_dim].values
    values = field.values
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return None

    masked_values = np.where(valid_mask, values, np.nan)
    flat_idx = np.nanargmax(masked_values)
    dim0_idx, dim1_idx = np.unravel_index(flat_idx, masked_values.shape)
    dims = list(field.dims)
    indices = [dim0_idx, dim1_idx]
    lat_idx = indices[dims.index(lat_dim)]
    lon_idx = indices[dims.index(lon_dim)]
    try:
        peak_lat = float(lat_array[lat_idx] if lat_array.ndim == 1 else lat_array[lat_idx, lon_idx])
        peak_lon = float(lon_array[lon_idx] if lon_array.ndim == 1 else lon_array[lat_idx, lon_idx])
    except (IndexError, TypeError) as error:
        raise PeakCoordinatesUnreadable(str(error)) from error
    return Peak(value=float(masked_values[dim0_idx, dim1_idx]), lat=peak_lat, lon=peak_lon)
