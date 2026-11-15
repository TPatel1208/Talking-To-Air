"""Reducing an already-masked field over time, or over its area at each time.

Both work on the shared grid layout: ``time`` (optional), ``lat`` and ``lon``.
Neither masks again: the field arrives masked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.domain import xarray_ops
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.aggregation.regional_reduction import reduce_keeping_axes

# The statistics a reduction over time or area supports.
REDUCTION_STATS: tuple[str, ...] = tuple(AggregationService._STAT_FUNCS)

_service = AggregationService()


def valid_time_indices(da: xr.DataArray) -> list[int]:
    """The positions on ``time`` that hold at least one finite value."""
    spatial = [d for d in da.dims if d != "time"]
    found = np.atleast_1d(np.asarray(xarray_ops.isfinite(da).any(spatial).values))
    return [i for i, ok in enumerate(found) if bool(ok)]


def reduce_over_time(da: xr.DataArray, stat: str, cadence: str) -> tuple[xr.DataArray, list[int]]:
    """``da`` reduced over ``time`` with ``stat``, and the time positions that took part.

    Only time steps holding a valid value take part. A mean weights each
    cadence bucket equally (``AggregationService.temporal_mean``), so densely
    sampled days do not dominate; the other statistics are per cell over every
    valid step. A field without ``time`` is returned as it is.
    """
    if stat not in _service._STAT_FUNCS:
        raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(_service._STAT_FUNCS)}")
    if "time" not in da.dims:
        return da, [0]
    indices = valid_time_indices(da)
    if not indices:
        reduced = da.isel(time=slice(0, 0)).mean(dim="time", skipna=True)
    else:
        valid = da.isel(time=indices)
        if stat == "mean":
            reduced = _service.temporal_mean(valid, "time", cadence)
        else:
            reduced = valid.reduce(_service._STAT_FUNCS[stat], dim="time")
    reduced.name = da.name
    reduced.attrs = dict(da.attrs)
    return reduced, indices


def regional_series(da: xr.DataArray, stat: str) -> pd.DataFrame:
    """One value of ``stat`` over the area per time step, as ``time`` and ``value`` columns.

    The mean is cos(latitude) area-weighted; the other statistics are per cell
    (``reduce_keeping_axes``). A step with no valid cell is left out, and rows
    are sorted by time.
    """
    reduced = reduce_keeping_axes(da, keep=("time",), stat=stat)
    values = np.atleast_1d(np.asarray(reduced.values, dtype=float))
    times = np.atleast_1d(np.asarray(da["time"].values))
    rows = [(pd.Timestamp(t), float(v)) for t, v in zip(times, values, strict=True) if np.isfinite(v)]
    frame = pd.DataFrame(rows, columns=["time", "value"])
    return frame.sort_values("time", kind="stable").reset_index(drop=True)
