"""Turn an opened retrieval into the masked grid every analysis reads.

The order is the current system's: normalise longitudes on the whole
Dataset (so the science variable and its quality flag share one convention),
select the science variable, mask to the area's geometry and crop to its
bounds, then apply the registry's fill, valid-range and quality-flag masking
through ``AggregationService.resolve_and_mask``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from tta_backend.datasets.mask_info import col_info_for_variable, override_for
from tta_backend.datasets.registry import CollectionConfig
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.errors import CATEGORY_CONTRACT, CATEGORY_UNSUPPORTED_GRID, MCPToolError
from tta_backend.domain.geo import find_lat_coord, find_lon_coord, identify_time
from tta_backend.domain.masking.geometry import mask_data_by_geometry, sel_bounds

# The provenance keys a masked grid carries forward, out of everything
# resolve_and_mask reports.
_PROVENANCE_KEYS = (
    "fill_value_source",
    "valid_range_source",
    "applied",
    "qa_status",
    "qa_source",
    "qa_pass_rate",
    "qa_checked_pixels",
    "qa_passing_pixels",
)

# How many missing days a quality summary lists by date.
MAX_LISTED_DAYS = 31


@dataclass(frozen=True)
class MaskedGrid:
    """A science variable masked to its area and by its quality rules.

    ``data`` has ``lat``/``lon`` (and ``time`` when the data has one) and
    ``units`` in its attrs. ``masking`` is the small provenance a grid handle
    records. ``region_cells`` is how many grid cells lie inside the area on
    one time slice, the denominator of the valid-data fraction.
    ``region_area`` is those cells' cos(latitude)-weighted area, the
    denominator of a frame's coverage; None when it was not measured.
    """

    data: xr.DataArray
    masking: dict[str, Any]
    region_cells: int
    region_area: float | None = None

    def dataset(self) -> xr.Dataset:
        """The grid in the harness's grid layout: one data variable."""
        return self.data.to_dataset(name=str(self.data.name or "value"))


def normalize_longitudes[T: (xr.DataArray, xr.Dataset)](data: T, lon_coord: str) -> T:
    """Convert 0..360 longitude coordinates to -180..180 and keep them sorted."""
    lon_vals = np.asarray(data[lon_coord].values)
    finite_lons = lon_vals[np.isfinite(lon_vals)]
    if finite_lons.size == 0 or finite_lons.min() < 0 or finite_lons.max() <= 180:
        return data
    normalized = ((lon_vals + 180) % 360) - 180
    return data.assign_coords({lon_coord: normalized}).sortby(lon_coord)


def _select(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """The named variable, matched exactly or by bare leaf name; never another one."""
    names = [str(name) for name in ds.data_vars]
    leaf = variable.rsplit("/", 1)[-1]
    name = variable if variable in names else leaf if leaf in names else None
    if name is None:
        raise MCPToolError(
            CATEGORY_CONTRACT,
            f"The retrieved data does not hold the variable {variable!r}.",
        )
    return AggregationService().to_dataarray(ds, variable=name)


def registry_col_info(cfg: CollectionConfig) -> dict[str, Any]:
    """A registry entry as the masking code reads it, with any recorded override on top."""
    return {**cfg.model_dump(), **override_for(cfg.short_name.upper() if cfg.short_name else None)}


def _standard_layout(da: xr.DataArray, lat: str, lon: str) -> xr.DataArray:
    renames: dict[Any, str] = {}
    if lat != "lat":
        renames[lat] = "lat"
    if lon != "lon":
        renames[lon] = "lon"
    time = identify_time(da)
    if time is not None and time != "time" and time in da.dims:
        renames[time] = "time"
    return da.rename(renames) if renames else da


def _pinned_axes(ds: xr.Dataset, da: xr.DataArray, col_info: dict[str, Any]) -> dict[str, Any]:
    """The registry's pinned vertical axes the file publishes as data
    variables, on the science variable's dimensions, ready to attach as coordinates."""
    axes: dict[str, Any] = {}
    names = {str(name) for name in ds.data_vars}
    for pinned in col_info.get("vertical_axis_vars") or []:
        leaf = str(pinned).rsplit("/", 1)[-1]
        name = pinned if pinned in names else leaf if leaf in names else None
        if name is None or leaf in da.coords:
            continue
        var = ds[name]
        if set(var.dims) <= set(da.dims):
            axes[leaf] = (var.dims, var.data, dict(var.attrs))
    return axes


def mask_retrieval(
    ds: xr.Dataset,
    *,
    variable: str,
    geometry: dict[str, Any] | BaseGeometry,
    bounds: Sequence[float],
    product: CollectionConfig | None = None,
) -> MaskedGrid:
    """Mask ``variable`` of an opened retrieval to ``geometry`` and its quality rules.

    The masking facts come from the registry entry matching the file's own
    collection identity, as in the current system; when the file carries no
    identity, ``product``'s entry is used, and ``masking["registry_entry"]``
    says which. ``bounds`` is (west, south, east, north). A layered product's
    pinned vertical axes ride on the grid as coordinates, whether the file
    publishes them as coordinates or as data variables.
    """
    lon_name = find_lon_coord(ds)
    if lon_name:
        ds = normalize_longitudes(ds, lon_name)
    da = _select(ds, variable)
    lat, lon = find_lat_coord(da), find_lon_coord(da)
    if lat is None or lon is None:
        raise MCPToolError(
            CATEGORY_UNSUPPORTED_GRID,
            "Could not identify latitude/longitude coordinates on this product.",
        )
    units = da.attrs.get("units") or (product.units if product is not None else None)
    shape_ = geometry if isinstance(geometry, BaseGeometry) else shape(geometry)

    col_info = col_info_for_variable(da, ds)
    registry_entry = "file_identity" if col_info else None
    # Unlike the current system's map path, a file naming no collection falls back to the chosen product's entry.
    if not col_info and product is not None:
        col_info = registry_col_info(product)
        registry_entry = "chosen_product"
    axes = _pinned_axes(ds, da, col_info)
    if axes:
        da = da.assign_coords(axes)

    masked = mask_data_by_geometry(normalize_longitudes(da, lon), shape_)
    region_type = masked.attrs.get("region_type")
    region_cells = int(masked.attrs.get("region_cells", 0))
    area = masked.attrs.get("region_area")
    masked = sel_bounds(masked, lat, lon, tuple(bounds))  # type: ignore[arg-type]

    service = AggregationService()
    name = str(masked.name)
    masked, provenance = service.resolve_and_mask(masked, variable=name, col_info=col_info, source_ds=ds)
    quality_flag = service.qa_flag_variable(ds, da, col_info)

    masked = _standard_layout(masked, lat, lon)
    masked.name = name
    masked.attrs = {"units": units} if units else {}
    masking: dict[str, Any] = {"variable": name, "quality_flag": quality_flag}
    masking.update({key: provenance[key] for key in _PROVENANCE_KEYS if key in provenance})
    masking["region"] = "boundary_cells" if region_type == "boundary_cells" else "geometry"
    masking["registry_entry"] = registry_entry
    return MaskedGrid(
        data=masked, masking=masking, region_cells=region_cells,
        region_area=float(area) if isinstance(area, int | float) else None,
    )


def combine(grids: Sequence[MaskedGrid], days: Sequence[str]) -> MaskedGrid:
    """One grid from per-day grids of the same product and area, joined on time.

    A grid with no time coordinate takes the day it was retrieved for. The
    masking provenance keeps the keys every part agrees on and states the
    pass rate over all parts.
    """
    if not grids:
        raise ValueError("nothing to combine")
    parts = []
    for grid, day in zip(grids, days, strict=True):
        data = grid.data
        if "time" not in data.dims:
            data = data.expand_dims(time=[pd.Timestamp(day)])
        parts.append(data)
    joined = xr.concat(parts, dim="time", join="outer").sortby("time")
    joined.name = grids[0].data.name
    joined.attrs = dict(grids[0].data.attrs)

    first = grids[0].masking
    masking = {
        key: value for key, value in first.items()
        if key not in ("qa_pass_rate", "qa_checked_pixels", "qa_passing_pixels")
        and all(g.masking.get(key) == value for g in grids)
    }
    checked = [g.masking.get("qa_checked_pixels") for g in grids]
    passing = [g.masking.get("qa_passing_pixels") for g in grids]
    if all(isinstance(c, int) for c in checked) and all(isinstance(p, int) for p in passing):
        total_checked = sum(checked)  # type: ignore[arg-type]
        total_passing = sum(passing)  # type: ignore[arg-type]
        masking["qa_checked_pixels"] = total_checked
        masking["qa_passing_pixels"] = total_passing
        if total_checked:
            masking["qa_pass_rate"] = round(total_passing / total_checked, 6)
    masking["parts"] = len(grids)
    return MaskedGrid(
        data=joined, masking=masking, region_cells=grids[0].region_cells, region_area=grids[0].region_area,
    )


def _days(values: Any) -> list[str]:
    return [pd.Timestamp(v).date().isoformat() for v in np.atleast_1d(values)]


def quality_summary(grid: MaskedGrid, *, requested_days: Sequence[str]) -> dict[str, Any]:
    """Valid-data fraction, quality pass rate and which requested days had valid data.

    The valid-data fraction is valid cells over the area's cells, after all
    masking, over every time slice. A grid with no time coordinate counts
    every requested day as having data when any cell is valid.
    """
    data = grid.data
    finite = xr.apply_ufunc(np.isfinite, data, dask="allowed")
    valid = int(finite.sum())
    slices = int(data.size // max(1, data.sizes.get("lat", 1) * data.sizes.get("lon", 1)))
    denominator = grid.region_cells * slices
    fraction = min(1.0, valid / denominator) if denominator else 0.0

    if "time" in data.dims:
        spatial = [d for d in data.dims if d != "time"]
        per_time = np.atleast_1d(np.asarray(finite.any(spatial).values))
        with_data = {day for day, ok in zip(_days(data["time"].values), per_time, strict=True) if ok}
    else:
        with_data = set(requested_days) if valid else set()
    requested = list(dict.fromkeys(requested_days))
    missing = [day for day in requested if day not in with_data]

    summary: dict[str, Any] = {
        "valid_fraction": round(fraction, 4),
        "valid_cells": valid,
        "region_cells": grid.region_cells,
        "qa_status": grid.masking.get("qa_status"),
        "qa_pass_rate": grid.masking.get("qa_pass_rate"),
        "days_requested": len(requested),
        "days_with_data": len(requested) - len(missing),
        "missing_days": missing[:MAX_LISTED_DAYS],
    }
    if len(missing) > MAX_LISTED_DAYS:
        summary["missing_days_listed"] = MAX_LISTED_DAYS
    return summary
