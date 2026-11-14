"""Reduce a field to its (lat, lon) plane, selecting along any other dimension.

A size-1 dimension is dropped, an explicit selector picks a value along the
dimension it names, and a surviving time axis is averaged. Any other dimension
left over is refused with an error that lists its values.
"""
from __future__ import annotations

from typing import Any

import xarray as xr

from tta_backend.domain.errors import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError
from tta_backend.domain.geo import (
    LAT_COORD_CANDIDATES,
    LON_COORD_CANDIDATES,
    find_lat_coord,
    find_lon_coord,
    identify_time,
    is_vertical_dim,
)


def _non_selectable_dims(data_array: xr.DataArray) -> set[Any]:
    """Dims that never require an explicit selection: spatial (lat/lon) and
    time (the one transparent auto-reduction), both found by CF metadata."""
    non_selectable = set(LAT_COORD_CANDIDATES + LON_COORD_CANDIDATES)
    non_selectable |= {name for name in (find_lat_coord(data_array), find_lon_coord(data_array)) if name}
    time_name = identify_time(data_array)
    if time_name:
        non_selectable.add(time_name)
    return non_selectable


def _squeeze_size_one_dims(data_array: xr.DataArray) -> xr.DataArray:
    dims_to_squeeze = [d for d in data_array.dims if data_array.sizes[d] == 1]
    return data_array.squeeze(dims_to_squeeze) if dims_to_squeeze else data_array


def _dimension_choice_error(data_array: xr.DataArray, dim: str) -> MCPToolError:
    if dim in data_array.coords:
        coord = data_array[dim]
        values = [v.item() if hasattr(v, "item") else v for v in coord.values.tolist()] if coord.ndim == 1 else coord.values.tolist()
        units = coord.attrs.get("units", "")
    else:
        values = list(range(data_array.sizes[dim]))
        units = ""
    values_str = ", ".join(str(v) for v in values)
    units_note = f", units {units}" if units else ""
    name = data_array.name or "this variable"
    suggestion = f"Pass a dimension selector for '{dim}' with one of the values above."
    # A refused VERTICAL dimension is the moment a researcher wanted the whole
    # profile, not one level out of it -- and this refusal is where they already
    # are. Naming the tool here turns the dead end into the discovery path.
    # Kept conditional on the dimension actually being vertical:
    # suggesting a profile for a wavelength axis would send them somewhere that
    # cannot help.
    if is_vertical_dim(data_array, dim):
        suggestion = (
            f"'{dim}' is a vertical axis: call plot_vertical_profile to chart the "
            f"whole profile, or pass a '{dim}' selector with one of the values "
            "above to take a single level."
        )
    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"'{name}' has an additional dimension '{dim}' ({len(values)} values{units_note}) "
        f"with no selection made: {values_str}.",
        suggestion=suggestion,
    )


def _select_dim_nearest(data_array: xr.DataArray, dim_name: str, value: Any) -> xr.DataArray:
    """Select ``value`` along ``dim_name`` by nearest match, but refuse a
    request that falls outside the coordinate's own min--max range instead of
    silently snapping to an edge level. ``method="nearest"`` alone turns a
    units mismatch (a MERRA-2 ``lev`` coordinate in Pa with the model passing
    hPa, say) into a plausible-looking selection of the wrong level, with no
    signal that anything went wrong."""
    coord = data_array[dim_name] if dim_name in data_array.coords else None
    if coord is None:
        # A dim with no coordinate can only be selected by position —
        # _dimension_choice_error advertises indices 0..n-1 for exactly this
        # case, and xarray's .sel(method="nearest") refuses index-less dims
        # outright ("no associated coordinate or index").
        return _select_dim_positional(data_array, dim_name, value)
    if coord.ndim == 1 and coord.size > 0:
        try:
            requested = float(value)
        except (TypeError, ValueError):
            requested = None
        if requested is not None:
            cmin = float(coord.min())
            cmax = float(coord.max())
            if not (cmin <= requested <= cmax):
                raise _dimension_out_of_range_error(coord, dim_name, requested, cmin, cmax)
    return data_array.sel({dim_name: value}, method="nearest")


def _select_dim_positional(data_array: xr.DataArray, dim_name: str, value: Any) -> xr.DataArray:
    """Select by integer position along a coordinate-less ``dim_name``,
    keeping the same refuse-don't-snap contract as ``_select_dim_nearest``:
    a fractional or out-of-range index is a structured, range-naming error,
    never a clamp or a silent truncation."""
    size = data_array.sizes[dim_name]
    try:
        requested = float(value)
    except (TypeError, ValueError):
        requested = None
    if requested is None or not requested.is_integer() or not (0 <= int(requested) < size):
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"Dimension '{dim_name}' has no coordinate values, so it is selected "
            f"by position — got {value!r}, expected an integer index in "
            f"[0, {size - 1}].",
            suggestion=f"Pass a '{dim_name}' index between 0 and {size - 1}.",
        )
    return data_array.isel({dim_name: int(requested)})


def _dimension_out_of_range_error(
    coord: xr.DataArray, dim: str, value: float, cmin: float, cmax: float,
) -> MCPToolError:
    units = coord.attrs.get("units", "")
    units_note = f" {units}" if units else ""
    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"Selection {value}{units_note} for dimension '{dim}' is outside its "
        f"coordinate range [{cmin}, {cmax}]{units_note} -- refusing to snap to "
        f"the nearest edge level, which is a likely units mismatch (e.g. hPa "
        f"vs Pa).",
        suggestion=f"Pass a '{dim}' value within [{cmin}, {cmax}]{units_note}.",
    )


def _normalize_to_2d(data_array: xr.DataArray, dim_selector: dict[str, Any] | None = None) -> xr.DataArray:
    """
    Squeeze a DataArray down to 2D (lat, lon):
      1. Drop all size-1 dimensions (handles Time=1 cleanly).
      2. Apply any explicit ``dim_selector`` ({dim_name: coordinate value}).
      3. A surviving time dim still auto-reduces (mean) -- it's the
         one transparent reduction, already disclosed via aggregate()'s own
         cadence/n_granules meta; this is a defense-in-depth squeeze for a
         caller that reaches _normalize_to_2d before aggregate() reduces it.
      4. Any other dimension that still survives and isn't spatial is
         refused with a structured, candidate-listing error naming the
         dimension and its coordinate values -- never silently averaged.
    """
    non_selectable = _non_selectable_dims(data_array)
    time_name = identify_time(data_array)
    data_array = _squeeze_size_one_dims(data_array)

    if dim_selector:
        for dim_name, value in dim_selector.items():
            if dim_name in data_array.dims:
                data_array = _select_dim_nearest(data_array, dim_name, value)
        data_array = _squeeze_size_one_dims(data_array)

    if time_name and time_name in data_array.dims:
        data_array = data_array.mean(dim=time_name, skipna=True)

    extra_dims = [d for d in data_array.dims if d not in non_selectable]
    if extra_dims:
        raise _dimension_choice_error(data_array, str(extra_dims[0]))

    return data_array
