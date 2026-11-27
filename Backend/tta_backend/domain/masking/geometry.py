"""Mask a gridded field to a geometry, and crop it to the mask's footprint."""
from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon

from tta_backend.domain.errors import CATEGORY_UNSUPPORTED_GRID, MCPToolError
from tta_backend.domain.geo import ensure_supported_grid, find_lat_coord, find_lon_coord
from tta_backend.infrastructure.phase_timing import phase_timer

logger = logging.getLogger(__name__)


def geometry_mask(
    data_array: xr.DataArray,
    geometry: Union[Polygon, MultiPolygon]
) -> xr.DataArray:
    """
    Boolean (lat, lon) DataArray: True inside ``geometry``.

    The single rasterization seam: ``mask_data_by_geometry`` applies it to
    the data, and callers needing the region footprint as well (the
    evidence crop) read it off this same mask instead of rasterizing an
    all-ones twin — the mask depends only on the grid and the geometry,
    never on the data values.
    """
    # The affine transform below assumes a 1-D rectilinear lat/lon grid. A
    # 2-D curvilinear swath or a projected x/y grid would be silently
    # mis-masked here -- refuse with a specific, typed error instead.
    ensure_supported_grid(data_array)

    # Find lat/lon coordinates (handle different naming conventions)
    lat_coord = find_lat_coord(data_array)
    lon_coord = find_lon_coord(data_array)

    if lat_coord is None or lon_coord is None:
        # Typed like the curvilinear/projected refusals above, so the stat/
        # plot tools answer a classified error instead of letting a bare
        # ValueError escape off-taxonomy — live, that surfaced as a generic
        # "internal error" for a plot and a false "no data found" for a
        # point query (QA 2026-07-17, GPM).
        raise MCPToolError(
            CATEGORY_UNSUPPORTED_GRID,
            "Could not identify latitude/longitude coordinates on this product "
            f"(dims: {list(data_array.dims)}; coords: {list(data_array.coords.keys())}).",
            suggestion="Try a gridded (Level 3) product published on a lat/lon grid.",
        )

    # Get coordinate arrays
    lats = data_array[lat_coord].values
    lons = data_array[lon_coord].values

    # Calculate the affine transform for the raster
    # Ensure we have at least 2 points to calculate resolution
    if len(lons) < 2:
        lon_res = 1.0  # Default resolution
    else:
        lon_diff = lons[-1] - lons[0]
        lon_res = lon_diff / (len(lons) - 1) if lon_diff != 0 else 1.0

    if len(lats) < 2:
        lat_res = 1.0  # Default resolution
    else:
        lat_diff = lats[-1] - lats[0]
        lat_res = lat_diff / (len(lats) - 1) if lat_diff != 0 else 1.0

    transform = Affine.translation(lons[0] - lon_res/2, lats[0] - lat_res/2) * \
                Affine.scale(lon_res, lat_res)

    # Rasterize the geometry
    # 1 = inside geometry, 0 = outside
    mask_2d = rasterize(
        [(geometry, 1)],
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Empty-mask self-heal: a region smaller than a grid cell can cover
    # zero cell *centers*, so center-containment rasterization returns nothing
    # for data that is right there. Retry with all_touched -- which keeps any
    # cell the geometry intersects at all -- and, if that recovers cells,
    # return them disclosed as boundary_cells. If it's still empty the region
    # genuinely misses the grid, and today's no-data answer stands.
    region_type = None
    if not mask_2d.any():
        boundary_2d = rasterize(
            [(geometry, 1)],
            out_shape=(len(lats), len(lons)),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        if boundary_2d.any():
            mask_2d = boundary_2d
            region_type = "boundary_cells"

    # Boolean (True = INSIDE geometry = keep), carrying the grid's own
    # coordinates so it can be cropped/selected like the data itself.
    mask_da = xr.DataArray(
        mask_2d == 1,
        coords={lat_coord: lats, lon_coord: lons},
        dims=[lat_coord, lon_coord],
    )
    if region_type is not None:
        # A masking-time fidelity fact (depends only on grid + geometry):
        # callers read it off the mask to disclose region_type in provenance.
        mask_da.attrs["region_type"] = region_type
    return mask_da


def half_cell(coords: np.ndarray) -> float:
    """Half the (uniform) grid spacing of a 1-D coordinate axis, for extending
    a pixel-center extent to its pixel-edge extent. Mirrors render_overlay_png's
    resolution formula so overlay.bounds and the rendered raster agree: a
    regular grid's step is (max - min) / (n - 1); a single-cell axis has no
    spacing to measure, so it falls back to the 0.5° half-cell that
    render_overlay_png assumes there (res default 1.0)."""
    vals = np.asarray(coords, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size > 1:
        return abs(float(finite.max()) - float(finite.min())) / (finite.size - 1) / 2.0
    return 0.5


def sel_bounds(
    da: xr.DataArray, lat_coord: str, lon_coord: str, bounds: tuple[float, float, float, float],
) -> xr.DataArray:
    """
    Crop a DataArray to (minx, miny, maxx, maxy) bounds in a coordinate-order-
    safe way.  xarray slice() requires start <= stop when coords are increasing
    and start >= stop when decreasing.  We detect the direction and swap if needed
    so the crop never silently returns an empty array.
    """
    lat_vals = da[lat_coord].values
    lon_vals = da[lon_coord].values

    lat_min, lat_max = bounds[1], bounds[3]   # miny, maxy
    lon_min, lon_max = bounds[0], bounds[2]   # minx, maxx

    # If latitude is stored N→S (decreasing), slice must be (max, min)
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Longitude is almost always W→E (increasing), but handle both
    if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return da.sel({lat_coord: lat_slice, lon_coord: lon_slice})


def _crop_to_mask_footprint(
    data_array: xr.DataArray,
    mask_da: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Narrow ``data_array`` and its already-rasterized ``mask_da`` to the
    smallest window containing every cell the mask keeps.

    "Mean NO2 over New Jersey" against a continental grid otherwise applies a
    continental ``.where`` and reduces over an array that is >99% NaN by
    construction. Rasterization stays on the FULL grid -- it costs ~10% of the
    call and reads no data -- and both the data and that same mask are then
    sliced by index. The kept cells are therefore *identical* to the uncropped
    path's by construction, rather than by an argument about grid steps: a
    window re-rasterized from cropped axes derives a step differing in the 8th
    digit on the float32 axes real granules ship, which flips cell centers
    lying on the geometry's boundary (measured live: a 2% move on a small AOI).

    Returns the pair unchanged whenever the crop can't be taken -- an empty
    mask (the honest no-data answer), a mask riding dims the data doesn't
    have, or a region already spanning the granule. It is an optimization,
    and must never be the reason a turn fails or a number moves.
    """
    try:
        lat_dim, lon_dim = map(str, mask_da.dims)
        if lat_dim not in data_array.dims or lon_dim not in data_array.dims:
            return data_array, mask_da

        rows = np.flatnonzero(mask_da.any(dim=lon_dim).values)
        cols = np.flatnonzero(mask_da.any(dim=lat_dim).values)
        if rows.size == 0 or cols.size == 0:
            # Nothing kept anywhere: leave today's empty result exactly as is.
            return data_array, mask_da

        window = {
            lat_dim: slice(int(rows[0]), int(rows[-1]) + 1),
            lon_dim: slice(int(cols[0]), int(cols[-1]) + 1),
        }
        cropped = data_array.isel(window)
        if cropped.size == data_array.size:
            # A region covering the whole granule has nothing to crop: hand
            # back the original, so the event below counts savings not
            # attempts.
            return data_array, mask_da

        cropped_mask = mask_da.isel(window)
        logger.info(
            "aoi_crop_applied",
            extra={
                "_event": "aoi_crop_applied",
                "_cells_before": int(data_array.size),
                "_cells_after": int(cropped.size),
                "_crop_bounds": [
                    float(np.nanmin(cropped_mask[lon_dim].values)),
                    float(np.nanmin(cropped_mask[lat_dim].values)),
                    float(np.nanmax(cropped_mask[lon_dim].values)),
                    float(np.nanmax(cropped_mask[lat_dim].values)),
                ],
            },
        )
        return cropped, cropped_mask
    except Exception:  # pragma: no cover - defensive: never fail a turn to crop
        logger.warning("aoi_crop_skipped", exc_info=True)
        return data_array, mask_da


def mask_data_by_geometry(
    data_array: xr.DataArray,
    geometry: Union[Polygon, MultiPolygon],
    crop: bool = True,
) -> xr.DataArray:
    """
    Mask xarray data to only show values within a geometry boundary.
    Sets all points outside the geometry to NaN.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data with latitude/longitude coordinates
        Handles common coord names: 'lat'/'latitude', 'lon'/'longitude'
    geometry : Polygon or MultiPolygon
        Boundary geometry from RegionResult
    crop : bool, optional
        Narrow the data (and the mask) to the mask's footprint before the
        ``.where``, so the reduction doesn't run over a grid that is almost
        entirely NaN by construction. Pass ``False`` for the full-grid
        behavior; the values kept are identical either way.

    Returns
    -------
    xarray.DataArray
        Masked data array (copy, original unchanged)
    """
    mask_da = geometry_mask(data_array, geometry)

    if crop:
        # The crop and the .where are timed separately: the crop exists to
        # move work out of the .where, and one number couldn't show that.
        with phase_timer("crop", cells_in=int(data_array.size)) as timing:
            data_array, mask_da = _crop_to_mask_footprint(data_array, mask_da)
            timing["cells_out"] = int(data_array.size)

    # Rank is not this function's business, and pinning it to 2-D/3-D was a
    # false constraint: the ``.where`` below aligns by dimension NAME, and
    # ``geometry_mask`` above has already established that the grid is a
    # rectilinear lat/lon one it can rasterize. What that rank check actually
    # excluded was a field carrying a non-spatial axis besides time -- a
    # vertical profile product's (time, lat, lon, layer) -- which failed here
    # with a bare ValueError before any tool could reach its own dimension
    # handling. Requiring the two masked dims to be present is the real
    # precondition, and it is checked where it belongs, in geometry_mask.
    #
    # xarray aligns by dimension name, so this works for (lat, lon),
    # Harmony-reformatted grids ordered as (time, lon, lat), and a layered
    # field alike.
    with phase_timer("mask", cells_in=int(data_array.size), cropped=crop):
        masked = data_array.where(mask_da)
    # Carry the mask's fidelity signal: when geometry_mask self-healed
    # an empty mask to boundary cells, surface that here (``.where`` doesn't
    # keep the mask's attrs) so the tool can disclose region_type.
    if mask_da.attrs.get("region_type"):
        masked.attrs["region_type"] = mask_da.attrs["region_type"]
    # How many cells the ANALYZED REGION actually covers on this grid, recorded
    # here because this is the only place the rasterized mask exists -- a caller
    # re-deriving it would have to re-rasterize on the cropped axes, whose
    # float32 step differs in the 8th digit and flips cells on the boundary.
    #
    # It is the honest denominator for any "what fraction of the region had a
    # value" figure. The obvious alternative -- the cropped array's cell count --
    # is the BOUNDING BOX, and for a region shaped like anything but a rectangle
    # that silently reports empty ocean as missing data: a complete retrieval
    # over the continental US reads 60% covered, and the reader sees a data
    # problem that does not exist.
    masked.attrs["region_cells"] = int(mask_da.sum())
    # ...and how much AREA those cells cover, cos(latitude)-weighted with the
    # one weight definition (``cos_lat_weights``). A count and an area are not
    # interchangeable denominators: cells shrink toward the poles, so a
    # coverage figure whose numerator is area-weighted and whose denominator is
    # a cell count describes two different fields at once. The frame stack's
    # ``valid_fraction`` needs the area; per-layer profile coverage still reads
    # the count.
    #
    # Imported here rather than at module scope: aggregation_service reaches
    # back into this module's masking helpers, and a top-level import would
    # close the cycle.
    from tta_backend.domain.aggregation.aggregation_service import cos_lat_weights

    weights = cos_lat_weights(mask_da)
    masked.attrs["region_area"] = float(
        (mask_da if weights is None else mask_da * weights).sum()
    )
    return masked


def apply_mask_region_type(masked: xr.DataArray, region: dict[str, Any]) -> None:
    """Downgrade ``region['region_type']`` to the masking-time fact when the
    mask self-healed: a sub-cell region that ``geometry_mask`` rescued
    with ``all_touched`` is ``boundary_cells``, not the polygon/box/point the
    resolver first named. Mutates the caller-owned ``region`` dict in place.

    ``boundary_cells`` is rasterization fidelity; ``composite_union`` and
    ``buffer`` are shape provenance ("a construction, not a named place").
    Both can be true at once (a small construction on a coarse grid
    self-heals), so the prior value is preserved into ``region_origin`` here,
    in one place, rather than at each constructor. ``setdefault``, because a constructor that already stated its origin
    is the more specific answer."""
    mask_region_type = masked.attrs.get("region_type")
    if mask_region_type:
        if region.get("region_type"):
            region.setdefault("region_origin", region["region_type"])
        region["region_type"] = mask_region_type
