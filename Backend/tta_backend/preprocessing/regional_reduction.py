"""
preprocessing/regional_reduction.py
====================================
Collapsing an analyzed region down to one number per position on a retained
axis — the shared middle of every chart that is not a map.

A timeseries and a vertical profile differ only in WHICH axis survives. The
reduction itself does not differ at all: the regional mean is cos(latitude)
area-weighted (so a chart line agrees with the statistics tool for the same
region), every other statistic is deliberately per-cell, and a slice the mask
emptied has no value rather than a zero. Each of those is a correction this
codebase has already made once, in ``conduct_temporal_statistic``. This module
exists so the profile tool inherits them instead of re-deriving them — a second
copy would not stay wrong visibly, it would drift by a few percent on
continental regions and agree everywhere a test looks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService, cos_lat_weights

_STATS = AggregationService._STAT_FUNCS


def reduce_keeping_axes(
    da: xr.DataArray, *, keep: Sequence[str], stat: str = "mean",
) -> xr.DataArray:
    """Reduce every dimension of ``da`` except ``keep`` to a single statistic.

    ``keep=("time",)`` is the timeseries: one regional value per timestep.
    ``keep=("time", "layer")`` is the profile's intermediate: one regional
    value per (timestep, layer), computed BEFORE time is collapsed, so the
    later temporal mean weights cadence buckets rather than granules (T56 D4).

    ``mean`` is the cos(latitude) area-weighted mean — the ONE regional-mean
    definition (``aggregation_service.area_weighted_mean``), falling back to an
    unweighted mean when no latitude dimension is collapsed. Every other
    statistic describes the cells themselves, not an area, so it stays
    unweighted; that split is ``conduct_temporal_statistic``'s and is preserved
    exactly.

    A position whose cells are all non-finite comes back NaN — absent, not
    zero. Callers drop those, the same way the timeseries loop dropped a
    timestep whose statistic raised.

    Reduced in ONE vectorized pass rather than a Python loop over slices.
    Numerically this is the same arithmetic, but a profile retains 24 layers
    times N granules, and a loop would walk the lazily-opened bundle's dask
    graph once per slice — turning a bounded read into a per-layer one.
    """
    if stat not in _STATS:
        raise ValueError(f"Unsupported stat '{stat}'. Valid: {sorted(_STATS)}")

    keep_dims = tuple(d for d in da.dims if d in set(keep))
    collapsed = [d for d in da.dims if d not in keep_dims]
    if not collapsed:
        return da

    # ±inf is not a measurement. ``compute_values_stat`` filtered on
    # ``np.isfinite`` before applying its NaN-aware function, so an infinity
    # was excluded rather than propagated through a whole slice; nulling it
    # here reproduces that exactly for the vectorized reduction.
    finite = da.where(np.isfinite(da))

    if stat == "mean":
        weights = cos_lat_weights(finite)
        if weights is None:
            reduced = finite.mean(dim=collapsed, skipna=True)
        else:
            reduced = finite.weighted(weights).mean(dim=collapsed, skipna=True)
    else:
        import warnings

        with warnings.catch_warnings():
            # An all-empty slice makes nanmax/nanmin/nanmedian warn before
            # returning the NaN that IS the honest answer here.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced = finite.reduce(_STATS[stat], dim=collapsed)

    # A weighted mean over an all-NaN slice can land on ±inf (zero weight
    # against zero contribution); "absent" is the one answer this returns.
    reduced = reduced.where(np.isfinite(reduced))
    reduced.name = da.name
    reduced.attrs.update(da.attrs)
    return reduced.transpose(*keep_dims) if keep_dims else reduced
