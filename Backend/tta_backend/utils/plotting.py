import logging
import asyncio
import functools
import json
import os
import threading
import httpx
import requests
import time

import numpy as np
import xarray as xr
# The object-oriented API, never pyplot. ``plt.subplots`` files every figure
# in a process-global registry that ``plt.close`` must later remove it from,
# and neither operation is synchronised -- so two exports rendering on two
# worker threads share one unguarded dict, and a render that raises before
# its close leaks its figure for the life of the process. A Figure with its
# own canvas has no registry to share and nothing to leak.
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from shapely.geometry import box, shape, Polygon, MultiPolygon
from rasterio.features import rasterize
from affine import Affine

from typing import Optional, Tuple, Union

from tta_backend.config.settings import get_settings
from tta_backend.datasets.us_states import US_STATES
from tta_backend.utils import region_buffer, region_composition, region_dispatch
from tta_backend.earthdata_mcp.results import (
    CATEGORY_DIMENSION_CHOICE_REQUIRED,
    CATEGORY_UNSUPPORTED_GRID,
    MCPToolError,
)
from tta_backend.utils.geo_utils import (
    LAT_COORD_CANDIDATES,
    LON_COORD_CANDIDATES,
    ensure_supported_grid,
    find_lat_coord,
    find_lon_coord,
    identify_time,
    is_vertical_dim,
)
from tta_backend.utils.phase_timing import phase_timer

logger = logging.getLogger(__name__)
_geocoding_service = None

# Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/)
# requires an identifying User-Agent naming the application and a means of
# contact. Reuses AQS_API_EMAIL (already required for EPA AQS API
# registration) as that contact address instead of adding a second
# deployment-operator email setting — an unfilled placeholder here isn't just
# a policy violation, Nominatim's edge actively 403s any User-Agent
# containing an @example.com/.org/.net address (verified live 2026-07-07),
# so a real address is required for geocoding to work at all.
NOMINATIM_USER_AGENT = f"talking-to-air/1.0 (contact: {get_settings().aqs_api_email})"


def get_geocoding_service() -> "GeocodingService":
    """Return the shared geocoder so agent and plotting tools share cache."""
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service


# Real (simplified Natural Earth 110m) boundaries for the multi-country
# presets, so "mean over the US" doesn't average in Sonora and the Atlantic
# (T42). Built once by scripts/build_preset_regions.py and checked in under
# datasets/ (data/ is a runtime volume, .dockerignore'd -- this asset must be
# baked into the image). Loaded lazily and cached here -- no runtime fetch.
_PRESET_REGIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "preset_regions.geojson"
)

# GeoJSON geometry types that carry no area, and therefore no boundary to
# disclose as one (see RegionResolver._geocoded_region). GeometryCollection is
# deliberately absent: it *may* contain a polygon, so it keeps the shape-based
# treatment rather than being pre-judged by its type tag.
_ZERO_AREA_GEOJSON_TYPES = frozenset(
    {"Point", "MultiPoint", "LineString", "MultiLineString"}
)


@functools.lru_cache(maxsize=1)
def load_preset_polygons() -> dict:
    """``{preset_id: shapely geometry}`` from the checked-in preset GeoJSON,
    parsed once. Returns ``{}`` (so callers fall back to bounding boxes) if
    the asset is missing or unreadable rather than failing region resolution
    outright."""
    try:
        with open(_PRESET_REGIONS_PATH, encoding="utf-8") as fh:
            fc = json.load(fh)
        return {
            feature["id"]: shape(feature["geometry"])
            for feature in fc.get("features", [])
        }
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Could not load preset polygons: %s", e)
        return {}


# T60 Phase 4: the 242 individually-addressable countries, the ``+`` grammar's
# second member vocabulary (D6). A **second** asset rather than more features in
# the one above, on two measurements (Phase 4 gate, V18/V19):
#
#   * ``georgia`` and ``antarctica`` are each a shipped preset id *and* an
#     ``ADMIN`` value. Merged, this dict comprehension silently reduces 304
#     features to 302 polygons -- last wins -- and ``"georgia"`` masks the
#     Caucasus while ``global_regions['georgia']`` still reports the U.S.
#     Southeast. ``AliasCollisionError`` never sees it: that guard compares
#     keys against ``global_regions``, one layer above the asset.
#   * Merged cold parse is 230-354 ms against 34.9 ms today, and every region
#     request pays it -- including ``"paris"``, which then geocodes anyway.
#     Separate, the 165 ms is paid only when a country token is typed.
_ADMIN0_COUNTRIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "admin0_countries.geojson"
)


@functools.lru_cache(maxsize=1)
def load_admin0_polygons() -> dict:
    """``{normalized ADMIN: {"name": ADMIN, "geometry": geom}}``, parsed once.

    Keys are already normalized by the builder, so ``"bahamas"`` resolves even
    though the source spells it ``"The Bahamas"`` -- the only one of the 242
    that ``_normalize_location_name`` changes (V19).

    Carrying the source's own ``ADMIN`` spelling alongside the geometry is why
    this returns a richer value than ``load_preset_polygons``: the key is
    lower-cased and ``"the "``-stripped, and an answer that cited *that* would
    say "Bahamas", "Eswatini" and "Hong Kong S.A.R" for places Natural Earth
    spells "The Bahamas", "eSwatini" and "Hong Kong S.A.R.". ``display_name``
    is what a T42 answer cites, so it gets the real spelling rather than a
    ``.title()`` reconstruction of a normalized key.

    Degrades to ``{}`` like ``load_preset_polygons``, which for this asset means
    every country token stops resolving and the composite grammar reports it as
    an unknown token. That is the fail-closed answer D3b requires: there is no
    bounding-box tier under this one to fall back to, by design (V21), and the
    geocoder is off the table for anything containing a ``+`` (D8)."""
    try:
        with open(_ADMIN0_COUNTRIES_PATH, encoding="utf-8") as fh:
            fc = json.load(fh)
        return {
            feature["id"]: {
                "name": feature["properties"]["name"],
                "geometry": shape(feature["geometry"]),
            }
            for feature in fc.get("features", [])
        }
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Could not load admin-0 country polygons: %s", e)
        return {}


# The Natural Earth features ``plot_map`` draws, and the one-time guard that
# makes them safe to reach from more than one thread.
#
# cartopy loads these lazily and without any locking of its own:
# ``NaturalEarthFeature.geometries()`` does an unguarded ``if key not in
# _NATURAL_EARTH_GEOM_CACHE`` check-then-set, and on a miss reaches
# ``Downloader.path()``, which checks ``target_path.exists()`` and otherwise
# writes the download straight to that final path -- no temp file, no atomic
# rename. Nothing bakes this data into the image, so a fresh container fetches
# it on its first render. That was harmless while every render was serialised
# on the event loop; once renders moved to worker threads, two cold exports
# could both miss the cache and write the same shapefile, and a third could
# read it half-written.
#
# Warming under a lock closes the whole window: after this returns, every
# feature's geometries are in cartopy's cache and the concurrent path is
# pure dict reads. A failed warm deliberately does NOT latch -- the render
# that follows would fail on the same download anyway, and a transient
# network blip should not poison every later export.
_CARTOPY_MAP_FEATURES = None
_CARTOPY_FEATURE_LOCK = threading.Lock()
_CARTOPY_FEATURES_WARMED = False


def _warm_cartopy_features() -> None:
    global _CARTOPY_MAP_FEATURES, _CARTOPY_FEATURES_WARMED

    if _CARTOPY_FEATURES_WARMED:
        return
    with _CARTOPY_FEATURE_LOCK:
        if _CARTOPY_FEATURES_WARMED:
            return
        if _CARTOPY_MAP_FEATURES is None:
            _CARTOPY_MAP_FEATURES = (cfeature.STATES, cfeature.COASTLINE, cfeature.BORDERS)
        try:
            for feature in _CARTOPY_MAP_FEATURES:
                feature.geometries()
        except Exception as e:
            logger.warning("Could not pre-load cartopy map features: %s", e)
            return
        _CARTOPY_FEATURES_WARMED = True


def plot_map(
    data_array: xr.DataArray,
    title: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    mask_geometry: Optional[Union[Polygon, MultiPolygon]] = None,
    cmap: str = "Spectral_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_scale: bool = True,
    add_gridlines: bool = True,
    time_slice: Optional[int] = None
):
    """
    Plot air quality data on a Cartopy map with proper extent and masking.

    Parameters
    ----------
    data_array : xarray.DataArray
        Variable to be plotted (e.g., NO2, O3, PM2.5)
        Can be 2D (lat, lon) or 3D (time, lat, lon)
        Must have latitude/longitude coordinates
    title : str, optional
        Plot title
    extent : tuple, optional
        Map bounds from RegionResult.bounds: (minx, miny, maxx, maxy))
        This is the CORRECT format from shapely.geometry.bounds
    mask_geometry : Polygon or MultiPolygon, optional
        Region geometry for masking data outside boundaries
        Use RegionResult.geometry here
    cmap : str, optional
        Matplotlib colormap name
        Good choices: 'viridis', 'plasma', 'YlOrRd', 'RdYlBu_r'
    vmin, vmax : float, optional
        Explicit color scale limits
        If None and percentile_scale=True, uses 2nd-98th percentile
    percentile_scale : bool, optional
        If True, automatically compute vmin/vmax from percentiles
        Ignored if vmin/vmax are explicitly provided
    add_gridlines : bool, optional
        Whether to add lat/lon gridlines with labels
    time_slice : int, optional
        For 3D data, which time index to plot
        If None, uses first time slice (index 0)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        For further customization or saving


    """

    # Populate cartopy's shapefile cache before anything else touches it --
    # see _warm_cartopy_features. Called here rather than at import time so a
    # process that never renders a map never fetches the data.
    _warm_cartopy_features()

    # --- 0. Handle 3D data - select time slice ---
    if data_array.ndim == 3:
        # Find the time dimension
        time_dims = ['time', 'Time', 'TIME', 't']
        time_dim = None

        for dim in time_dims:
            if dim in data_array.dims:
                time_dim = dim
                break

        if time_dim is None:
            # Assume first dimension is time if not found
            time_dim = data_array.dims[0]

        # Select time slice
        if time_slice is None:
            time_slice = 0

        logger.info("Selecting time slice %s from dimension '%s'", time_slice, time_dim)
        time_size = data_array.sizes[time_dim]

        if time_size == 1:
            data_array = data_array.isel({time_dim: 0})  # just take the only one
        else:
            time_slice = min(time_slice, time_size - 1)  # clamp to valid range
            data_array = data_array.isel({time_dim: time_slice})

    # --- 1. Apply geometry mask if provided ---
    if mask_geometry is not None:
        data_array = mask_data_by_geometry(data_array, mask_geometry)

    # --- 1.5. Find coordinate names (handle different conventions) ---
    lat_coord = find_lat_coord(data_array)
    lon_coord = find_lon_coord(data_array)

    if lat_coord is None or lon_coord is None:
        # Same typed refusal as geometry_mask — see the comment there.
        raise MCPToolError(
            CATEGORY_UNSUPPORTED_GRID,
            "Could not identify latitude/longitude coordinates on this product "
            f"(coords: {list(data_array.coords.keys())}).",
            suggestion="Try a gridded (Level 3) product published on a lat/lon grid.",
        )

    # --- 2. Compute adaptive color scale ---
    if vmin is None or vmax is None:
        if percentile_scale:
            valid_data = data_array.values[~np.isnan(data_array.values)]
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.percentile(valid_data, 2)
                if vmax is None:
                    vmax = np.percentile(valid_data, 98)
        else:
            # Fallback to min/max
            finite_values = data_array.values[np.isfinite(data_array.values)]
            if vmin is None:
                vmin = np.min(finite_values) if len(finite_values) else 0.0
            if vmax is None:
                vmax = np.max(finite_values) if len(finite_values) else 1.0

    # --- 3. Calculate figure size based on extent aspect ratio ---
    if extent:
        lon_range = extent[2] - extent[0]  # maxx - minx
        lat_range = extent[3] - extent[1]  # maxy - miny

        # Prevent divide by zero for degenerate regions
        if lat_range <= 0 or lon_range <= 0:
            logger.warning(
                "Degenerate extent detected (lon_range=%s, lat_range=%s). Using default figure size.",
                lon_range,
                lat_range,
            )
            fig_width, fig_height = 10, 6
        else:
            aspect_ratio = lon_range / lat_range

            # Base height of 6 inches, adjust width
            fig_height = 6
            fig_width = fig_height * aspect_ratio * 1.2  # 1.2 factor for map projection
            fig_width = np.clip(fig_width, 6, 14)  # Reasonable bounds
    else:
        fig_width, fig_height = 10, 6

    # --- 4. Create figure with Cartopy projection ---
    fig = Figure(figsize=(fig_width, fig_height), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # --- 5. Plot the data ---
    if lat_coord in data_array.dims and lon_coord in data_array.dims:
        data_array = data_array.transpose(lat_coord, lon_coord)

    plot_data = data_array.values

    im = ax.pcolormesh(
        data_array[lon_coord].values,
        data_array[lat_coord].values,
        plot_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='nearest'
    )

    # Add colorbar manually
    fig.colorbar(
        im,
        ax=ax,
        shrink=0.7,
        extend='both',
        label=data_array.attrs.get('long_name', data_array.name or '')[:40]
    )

    # --- 6. Set extent (FIXED: correct bounds format) ---
    if extent:
        # Convert from shapely bounds (minx, miny, maxx, maxy)
        # to Cartopy extent [lon_max, lat_max, lon_min, lat_min]
        cartopy_extent = [extent[0], extent[2], extent[1], extent[3]]
        ax.set_extent(cartopy_extent, crs=ccrs.PlateCarree())

    # --- 7. Add geographic features ---
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black', alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)

    # Optional: add region boundary outline
    if mask_geometry is not None:
        ax.add_geometries(
            [mask_geometry],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='red',
            linewidth=2,
            alpha=0.8
        )

    # --- 8. Add gridlines with labels ---
    if add_gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    # --- 9. Set title ---
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()

    return fig, ax

def _non_selectable_dims(data_array: xr.DataArray) -> set:
    """Dims that never require an explicit selection: spatial (lat/lon, by
    CF identification -- T24) and time (by CF identification -- T25, the one
    transparent auto-reduction)."""
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
    # are. Naming the tool here turns the dead end into the discovery path
    # (T56 D10). Kept conditional on the dimension actually being vertical:
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


def _select_dim_nearest(data_array: xr.DataArray, dim_name: str, value) -> xr.DataArray:
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


def _select_dim_positional(data_array: xr.DataArray, dim_name: str, value) -> xr.DataArray:
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


def _normalize_to_2d(data_array: xr.DataArray, dim_selector: dict | None = None) -> xr.DataArray:
    """
    Squeeze a DataArray down to 2D (lat, lon):
      1. Drop all size-1 dimensions (handles Time=1 cleanly).
      2. Apply any explicit ``dim_selector`` ({dim_name: coordinate value}).
      3. A surviving time dim (T25) still auto-reduces (mean) -- it's the
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
        raise _dimension_choice_error(data_array, extra_dims[0])

    return data_array

def geometry_mask(
    data_array: xr.DataArray,
    geometry: Union[Polygon, MultiPolygon]
) -> xr.DataArray:
    """
    Boolean (lat, lon) DataArray: True inside ``geometry``.

    The single rasterization seam: ``mask_data_by_geometry`` applies it to
    the data, and callers needing the region footprint as well (plot_tools'
    evidence crop) read it off this same mask instead of rasterizing an
    all-ones twin — the mask depends only on the grid and the geometry,
    never on the data values.
    """
    # The affine transform below assumes a 1-D rectilinear lat/lon grid. A
    # 2-D curvilinear swath or a projected x/y grid would be silently
    # mis-masked here -- refuse with a specific, typed error instead (T24).
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

    # Empty-mask self-heal (T42): a region smaller than a grid cell can cover
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


def sel_bounds(da, lat_coord, lon_coord, bounds):
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
    smallest window containing every cell the mask keeps (T50).

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
        lat_dim, lon_dim = mask_da.dims
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
        entirely NaN by construction (T50). Pass ``False`` for the pre-T50
        full-grid behavior; the values kept are identical either way.

    Returns
    -------
    xarray.DataArray
        Masked data array (copy, original unchanged)
    """
    mask_da = geometry_mask(data_array, geometry)

    if crop:
        # T51: the crop and the .where are timed separately because the whole
        # point of T50 was to move work from the second into the first -- one
        # combined number could not show that trade.
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
    # Carry the mask's fidelity signal (T42): when geometry_mask self-healed
    # an empty mask to boundary cells, surface that here (``.where`` doesn't
    # keep the mask's attrs) so the tool can disclose region_type.
    if mask_da.attrs.get("region_type"):
        masked.attrs["region_type"] = mask_da.attrs["region_type"]
    # How many cells the ANALYZED REGION actually covers on this grid, recorded
    # here because this is the only place the rasterized mask exists -- a caller
    # re-deriving it would have to re-rasterize on the cropped axes, whose
    # float32 step differs in the 8th digit and flips cells on the boundary
    # (the T50 finding).
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
    # a cell count describes two different fields at once -- Finding #13's
    # mismatch, arriving through the denominator instead of the numerator. The
    # frame stack's ``valid_fraction`` (T59 D10) needs the area; T56's
    # per-layer coverage still reads the count.
    #
    # Imported here rather than at module scope: aggregation_service reaches
    # back into this module's masking helpers, and a top-level import would
    # close the cycle.
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    weights = cos_lat_weights(mask_da)
    masked.attrs["region_area"] = float(
        (mask_da if weights is None else mask_da * weights).sum()
    )
    return masked


def apply_mask_region_type(masked: xr.DataArray, region: dict) -> None:
    """Downgrade ``region['region_type']`` to the masking-time fact when the
    mask self-healed (T42): a sub-cell region that ``geometry_mask`` rescued
    with ``all_touched`` is ``boundary_cells``, not the polygon/box/point the
    resolver first named. Mutates the caller-owned ``region`` dict in place.

    T60 D10a: ``region_type`` was carrying two orthogonal facts in one slot.
    ``boundary_cells`` is *rasterization fidelity*; ``composite_union`` (and
    Phase 5's ``buffer``) is *shape provenance* -- "this shape is a
    construction, not a named place". Both can be true at once, and the
    downgrade below used to destroy the second. It is the *likely* path, not a
    corner: a small construction on a coarse grid self-heals, and every
    masking site calls this before building provenance, so the researcher
    would never learn the shape was constructed.

    The prior value is preserved into ``region_origin`` here rather than at
    each constructor, so a future origin (D9's buffer) cannot forget to opt
    in. ``setdefault``, because a constructor that already stated its origin
    is the more specific answer."""
    mask_region_type = masked.attrs.get("region_type")
    if mask_region_type:
        if region.get("region_type"):
            region.setdefault("region_origin", region["region_type"])
        region["region_type"] = mask_region_type
class GeocodingService:
    """Free geocoding using Nominatim (OpenStreetMap) with polygon and bounding box"""

    def __init__(self, cache_ttl_seconds: int = 24 * 60 * 60):
        self.cache = {}
        self.cache_ttl_seconds = cache_ttl_seconds
        self.last_request = 0
        # Guards the read-modify-write on last_request below: geocode()
        # (sync, sometimes run in a worker thread) and ageocode() (async, on
        # the event loop) share this one throttle timestamp, and without
        # coordination two concurrent calls can both read a stale value and
        # both pass the 1 rps check at once -- Nominatim's usage policy 403s
        # on exactly that. A threading.Lock (not asyncio.Lock) works for
        # both paths because the critical section below is pure arithmetic,
        # held for microseconds -- the actual wait happens after release, so
        # the async path never blocks the event loop on a contended lock.
        self._throttle_lock = threading.Lock()

    def _reserve_throttle_slot(self) -> float:
        """Atomically claim the next allowed request slot (>=1s after the
        previously claimed one) and return how long the caller must still
        wait for it. Shared by geocode()/ageocode() so concurrent calls from
        either path always claim distinct, correctly-spaced slots instead of
        both reading the same last_request before either updates it."""
        with self._throttle_lock:
            now = time.time()
            next_slot = max(now, self.last_request + 1.0)
            self.last_request = next_slot
            return max(0.0, next_slot - now)

    def _cache_key(self, location_name: str) -> str:
        return " ".join(location_name.lower().strip().split())

    def _get_cached(self, location_name: str):
        key = self._cache_key(location_name)
        entry = self.cache.get(key)
        if not entry:
            logger.info("satellite_geocode_cache_miss", extra={"_location": location_name})
            return None

        expires_at, result = entry
        if expires_at >= time.time():
            logger.info("satellite_geocode_cache_hit", extra={"_location": location_name})
            return result

        self.cache.pop(key, None)
        logger.info("satellite_geocode_cache_miss", extra={"_location": location_name})
        return None

    def _store_cached(self, location_name: str, result: dict):
        key = self._cache_key(location_name)
        self.cache[key] = (time.time() + self.cache_ttl_seconds, result)

    def geocode(self, location_name):
        """Convert location name to coordinates, polygon, and bounding box"""
        cached = self._get_cached(location_name)
        if cached is not None:
            return cached

        # Rate limit: 1 request per second
        wait_seconds = self._reserve_throttle_slot()
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 1  # Request polygon boundary
        }
        headers = {
            'User-Agent': NOMINATIM_USER_AGENT
        }

        logger.info("satellite_geocode_requests", extra={"_location": location_name})

        try:
            # timeout is load-bearing. No production caller reaches this sync
            # path any more -- export_service, the last one, moved to
            # ``aresolve_location``/``ageocode`` precisely because the throttle
            # below sleeps and this request blocks. Keep the timeout anyway:
            # the path is still live (``resolve_location`` reaches it, and the
            # region suites mock at this seam), so whichever thread a future
            # caller runs it on, an unbounded hang is not an option. If that
            # caller is ever on the event loop again it freezes the whole
            # single-worker backend -- every SSE stream, heartbeat, and
            # /health -- which is the failure this bound exists to cap.
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()

            if data:
                item = data[0]

                # Centroid
                latitude = float(item['lat'])
                longitude = float(item['lon'])

                # Polygon (GeoJSON)
                polygon = item.get('geojson', None)

                # Bounding box: [south, north, west, east] as floats
                bbox = [float(coord) for coord in item.get('boundingbox', [])]

                result = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'display_name': item['display_name'],
                    'polygon': polygon,  # None if not available
                    'bbox': bbox         # None if not available
                }

                self._store_cached(location_name, result)
                return result
        except Exception as e:
            logger.warning("Geocoding error: %s", e)

        return None

    async def ageocode(self, location_name):
        """Async version of geocode() for agent tools running on the event loop."""
        cached = self._get_cached(location_name)
        if cached is not None:
            return cached

        wait_seconds = self._reserve_throttle_slot()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 1,
        }
        headers = {'User-Agent': NOMINATIM_USER_AGENT}

        logger.info("satellite_geocode_requests", extra={"_location": location_name})

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            if data:
                item = data[0]
                result = {
                    'latitude': float(item['lat']),
                    'longitude': float(item['lon']),
                    'display_name': item['display_name'],
                    'polygon': item.get('geojson', None),
                    'bbox': [float(coord) for coord in item.get('boundingbox', [])],
                }
                self._store_cached(location_name, result)
                return result
        except Exception as e:
            logger.warning("Geocoding error: %s", e)

        return None


class RegionResolver:
    """resolves user location inputs into singular plot or multiple plots"""
    def __init__(self, geocoding_service: GeocodingService | None = None):
        self.geocoding_service = geocoding_service or get_geocoding_service()
        # Define special global regions that don't need geocoding
        self.global_regions = {
        # --- Global ---
        'global': {'geometry': box(-180, -90, 180, 90), 'bounds': (-180, -90, 180, 90), 'name': 'Global'},
        'world':  {'geometry': box(-180, -90, 180, 90), 'bounds': (-180, -90, 180, 90), 'name': 'World'},
        'earth':  {'geometry': box(-180, -90, 180, 90), 'bounds': (-180, -90, 180, 90), 'name': 'Earth'},

        # --- Continents ---
        'north america': {'geometry': box(-168,   7,  -52, 84), 'bounds': (-168,   7,  -52, 84), 'name': 'North America'},
        'south america': {'geometry': box( -82, -56,  -34, 13), 'bounds': ( -82, -56,  -34, 13), 'name': 'South America'},
        'europe':        {'geometry': box( -25,  34,   45, 72), 'bounds': ( -25,  34,   45, 72), 'name': 'Europe'},
        'africa':        {'geometry': box( -18, -35,   52, 38), 'bounds': ( -18, -35,   52, 38), 'name': 'Africa'},
        'asia':          {'geometry': box(  26, -10,  180, 78), 'bounds': (  26, -10,  180, 78), 'name': 'Asia'},
        'oceania':       {'geometry': box( 110, -50,  180, 25), 'bounds': ( 110, -50,  180, 25), 'name': 'Oceania'},
        'antarctica':    {'geometry': box(-180, -90,  180, -60), 'bounds': (-180, -90,  180, -60), 'name': 'Antarctica'},

        # --- United States ---
        'united states':  {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'United States'},
        'usa':            {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'United States'},
        'us':             {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'United States'},
        'conus':          {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},
        'continental us': {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},
        'contiguous us':  {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},
        'lower 48':       {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},

        # --- US Regions ---
        'northeast us':  {'geometry': box( -80, 37,  -66, 48), 'bounds': ( -80, 37,  -66, 48), 'name': 'Northeast US'},
        'southeast us':  {'geometry': box( -92, 24,  -75, 37), 'bounds': ( -92, 24,  -75, 37), 'name': 'Southeast US'},
        'midwest us':    {'geometry': box(-104, 36,  -80, 50), 'bounds': (-104, 36,  -80, 50), 'name': 'Midwest US'},
        'southwest us':  {'geometry': box(-125, 31, -102, 42), 'bounds': (-125, 31, -102, 42), 'name': 'Southwest US'},
        'northwest us':  {'geometry': box(-125, 42, -110, 50), 'bounds': (-125, 42, -110, 50), 'name': 'Northwest US'},
        'great plains':  {'geometry': box(-105, 36,  -96, 50), 'bounds': (-105, 36,  -96, 50), 'name': 'Great Plains'},
        'great lakes':   {'geometry': box( -92, 41,  -76, 48), 'bounds': ( -92, 41,  -76, 48), 'name': 'Great Lakes'},

        # --- Europe Subregions ---
        'western europe':  {'geometry': box(-10, 35,  20, 60), 'bounds': (-10, 35,  20, 60), 'name': 'Western Europe'},
        'eastern europe':  {'geometry': box( 14, 44,  33, 55), 'bounds': ( 14, 44,  33, 55), 'name': 'Eastern Europe'},
        'northern europe': {'geometry': box(-25, 54,  32, 72), 'bounds': (-25, 54,  32, 72), 'name': 'Northern Europe'},
        'southern europe': {'geometry': box(-10, 35,  30, 46), 'bounds': (-10, 35,  30, 46), 'name': 'Southern Europe'},
        'scandinavia':     {'geometry': box(  4, 55,  32, 72), 'bounds': (  4, 55,  32, 72), 'name': 'Scandinavia'},

        # --- Asia Subregions ---
        'east asia':     {'geometry': box(100, 20, 145, 54), 'bounds': (100, 20, 145, 54), 'name': 'East Asia'},
        'southeast asia':{'geometry': box( 95, -10, 141, 28), 'bounds': ( 95, -10, 141, 28), 'name': 'Southeast Asia'},
        'south asia':    {'geometry': box( 61,  6,  97, 37), 'bounds': ( 61,  6,  97, 37), 'name': 'South Asia'},
        'central asia':  {'geometry': box( 46, 36,  87, 56), 'bounds': ( 46, 36,  87, 56), 'name': 'Central Asia'},
        'middle east':   {'geometry': box( 26, 12,  63, 42), 'bounds': ( 26, 12,  63, 42), 'name': 'Middle East'},

        # --- Africa Subregions ---
        'north africa':       {'geometry': box(-18, 15,  38, 38), 'bounds': (-18, 15,  38, 38), 'name': 'North Africa'},
        'sub-saharan africa': {'geometry': box(-18, -35, 52, 15), 'bounds': (-18, -35, 52, 15), 'name': 'Sub-Saharan Africa'},
        'west africa':        {'geometry': box(-18,  4,  16, 20), 'bounds': (-18,  4,  16, 20), 'name': 'West Africa'},
        'east africa':        {'geometry': box( 29, -12, 52, 16), 'bounds': ( 29, -12, 52, 16), 'name': 'East Africa'},
        'southern africa':    {'geometry': box( 11, -35, 40, -15), 'bounds': ( 11, -35, 40, -15), 'name': 'Southern Africa'},
    }
        # T60 Phase 3a: the 51 U.S. admin-1 units, from the generated table.
        # Ordinary presets -- an honest envelope here, upgraded to the real
        # boundary by ``_finalize_preset`` -- because unlike a coalition a
        # state *has* an honest bounding box (D3's objection does not apply;
        # no state crosses the antimeridian). ``display_name`` carries the
        # disambiguation D15's resolution order makes necessary: gate V12
        # measured "washington" geocoding live to Washington DC with 0.00%
        # overlap with Washington State.
        for _key, _state in US_STATES.items():
            if _key in self.global_regions:
                # D12a, applied to the merge itself. These go in by
                # assignment, so a key that already existed would be silently
                # replaced -- and Phase 4 adds countries, where "georgia" the
                # country meets "georgia" the state. Naming the key beats a
                # count assertion that only holds until someone updates it.
                raise region_dispatch.AliasCollisionError(
                    f"U.S. admin-1 key {_key!r} shadows an existing "
                    "global_regions preset; resolve the ambiguity explicitly "
                    "rather than letting one silently win"
                )
            self.global_regions[_key] = {
                'geometry': box(*_state['bounds']),
                'bounds': _state['bounds'],
                'name': _state['name'],
                'display_name': _state['display_name'],
            }
        # D12a: the T60 alias/coalition tables are hand-maintained, and a
        # table that silently shadows "us" or "georgia" is the cheapest way
        # to reintroduce a confident wrong region. Checked against this
        # instance's presets rather than left to review.
        region_dispatch.assert_no_alias_collisions(self.global_regions)
        # T60 Phase 4's table, same rule. Kept a separate call because it
        # deliberately does not open the country asset -- see its docstring;
        # doing so here would spend the 165 ms cold parse on every resolver,
        # including the ones that only ever resolve "paris".
        region_composition.assert_no_country_collisions(self.global_regions)

    # Preset keys that resolve to a real polygon (feature id in
    # preset_regions.geojson). Everything else in ``global_regions`` stays a
    # disclosed bounding box -- pure-ocean/quadrant concepts where a rectangle
    # is the honest answer (T42).
    _POLYGON_PRESET_IDS = {
        "united states": "united states", "usa": "united states", "us": "united states",
        "conus": "conus", "continental us": "conus",
        "contiguous us": "conus", "lower 48": "conus",
        "north america": "north america", "south america": "south america",
        "europe": "europe", "africa": "africa", "asia": "asia",
        "oceania": "oceania", "antarctica": "antarctica",
        # T60 coalitions. These are NOT ``global_regions`` keys (D3 -- a
        # coalition has no honest bounding box), so they are unreachable via
        # the exact-match gate below and arrive only through region_dispatch.
        "otc": "otc", "new england": "new england",
        # T60 Phase 3a: the 51 states. Feature id is the lowercased name, so
        # the preset key and the polygon id are the same string.
        **{key: key for key in US_STATES},
    }

    def _finalize_preset(self, preset: dict, key: str) -> dict:
        """Return a preset enriched with the T42 fidelity disclosure fields.
        A multi-country concept (US, a continent) is upgraded to its real
        Natural Earth polygon and labelled ``region_type: polygon``; every
        other preset stays the crude rectangle it is, honestly labelled
        ``bounding_box``. ``display_name`` is the human name the answer should
        cite. A copy, so the shared ``global_regions`` dict is never mutated."""
        region = dict(preset)
        region.setdefault("display_name", region.get("name", ""))
        polygon_id = self._POLYGON_PRESET_IDS.get(key)
        polygon = load_preset_polygons().get(polygon_id) if polygon_id else None
        if polygon is not None:
            region["geometry"] = polygon
            region["bounds"] = polygon.bounds
            region["region_type"] = "polygon"
        else:
            region.setdefault("region_type", "bounding_box")
        return region

    @staticmethod
    def _geocoded_region(geo_result: dict) -> dict:
        """Build a RegionResult from a geocoder hit, disclosing which kind of
        footprint it is: ``polygon`` when Nominatim returned a real boundary,
        or ``point_buffer`` when it didn't and we mint a 0.1° box around the
        centroid. ``display_name`` is the geocoder's own label, carried
        through so a wrong-place answer ("Paris, Texas") is catchable. Shared
        by the sync and async resolvers so a place can't resolve two ways.

        A GeoJSON hit is not automatically a boundary. Nominatim wraps a
        *point* result in GeoJSON as readily as an administrative area, so
        ``geojson is not None`` was never the same question as "did we get a
        footprint" -- measured in the T60 Phase 0 gate (V2), both ``"OTC"``
        (an aerodrome in Chad) and ``"northeastern us"`` (an ~11 m railway
        platform in Boston) return GeoJSON ``Point`` geometries and were being
        disclosed as ``polygon``. A zero-area geometry gets the same 0.1° box
        the no-geojson branch mints, because that is the same footprint by the
        same logic; ``point_buffer`` already names it honestly, and inventing a
        second value for "point, but wrapped in GeoJSON" would be a
        distinction with no consequence for the researcher."""
        geojson = geo_result.get("polygon")
        if geojson and geojson.get("type") not in _ZERO_AREA_GEOJSON_TYPES:
            geometry = shape(geojson)
            region_type = "polygon"
        else:
            lon, lat = geo_result["longitude"], geo_result["latitude"]
            delta = 0.1  # degrees
            geometry = box(lon - delta, lat - delta, lon + delta, lat + delta)
            region_type = "point_buffer"
        display_name = geo_result["display_name"]
        return {
            "geometry": geometry,
            "bounds": geometry.bounds,  # (minx, miny, maxx, maxy)
            "name": display_name,
            "display_name": display_name,
            "region_type": region_type,
        }

    @staticmethod
    def _normalize_location_name(location_name: str) -> str:
        """One normalization for both resolvers (T42 sync/async parity): lower,
        strip, collapse whitespace, and drop a leading "the " so "the
        Netherlands" and "the north america" resolve the same by either code
        path. Previously only the async path stripped "the ", so the same
        input could resolve two different ways."""
        normalized = " ".join(location_name.lower().strip().split())
        return normalized.removeprefix("the ")

    def resolve_location(self, location_name: str):
        """Convert location name to RegionResult with geometry"""
        # T60 D5/D11a: the ``+`` grammar reads the RAW string, ahead of
        # normalization, because the split has to happen before "the " is
        # stripped -- otherwise "ny + the nj" and "the ny + nj" resolve
        # differently. Raises (D14) rather than returning None on a bad token.
        composed = region_composition.dispatch_composite(location_name, self)
        if composed.claimed:
            return composed.region
        # T60 D9/D11b: the buffer grammar, and it runs AFTER composition
        # deliberately (gate V25). "within 50 km of NY + NJ" contains a "+",
        # so the composition grammar claims it first and hard-fails naming the
        # token -- the right answer, because D9 forbids X from resolving
        # recursively through COMPOSITE. Reversing the order would turn that
        # refusal into a buffer around a region D9 does not allow. This is the
        # sync twin, which is what this sync method needs -- but note that
        # export_service, once its only production caller, now takes
        # ``aresolve_location``. Nothing in production reaches this line today;
        # the region suites do.
        buffered = region_buffer.dispatch_buffer(location_name, self)
        if buffered.claimed:
            return buffered.region
        # Check for global regions first
        location_lower = self._normalize_location_name(location_name)
        dispatched = region_dispatch.dispatch(location_lower, self)
        if dispatched.claimed:
            # Claimed-and-failed returns None here and never reaches the
            # geocoder (D3b). Phase 1 surfaces that as the call sites'
            # existing "Could not resolve location: '<string>'" -- honest, and
            # it names the string but not the reason; D14's taxonomy error is
            # Phase 3's.
            return dispatched.region
        if location_lower in self.global_regions:
            return self._finalize_preset(self.global_regions[location_lower], location_lower)

        geo_result = self.geocoding_service.geocode(location_name)
        if geo_result is None:
            return None

        return self._geocoded_region(geo_result)

    async def aresolve_location(self, location_name: str):
        """Async version of resolve_location() for agent tool execution."""
        # T60 D5: the same gate, in the same place, for the same reason as the
        # sync twin -- the composition tier is pure, so both share one copy.
        composed = region_composition.dispatch_composite(location_name, self)
        if composed.claimed:
            return composed.region
        # T60 D9/D11b: the async twin, in the same position and by the same
        # V25 ordering argument as the sync one. This is the twin every
        # analysis tool reaches (stat/plot/validation call the async resolver
        # exclusively -- gate V24), and it is async precisely so the geocode
        # does not block the event loop.
        buffered = await region_buffer.adispatch_buffer(location_name, self)
        if buffered.claimed:
            return buffered.region
        location_lower = self._normalize_location_name(location_name)
        dispatched = region_dispatch.dispatch(location_lower, self)
        if dispatched.claimed:
            return dispatched.region
        if location_lower in self.global_regions:
            return self._finalize_preset(self.global_regions[location_lower], location_lower)

        geo_result = await self.geocoding_service.ageocode(location_name)
        if geo_result is None:
            return None

        return self._geocoded_region(geo_result)

    def plot_singular(self, data_array, location_name, **kwargs):
        """Plot data for a single location"""
        region = self.resolve_location(location_name)
        if region is None:
            raise ValueError(f"Could not find location: {location_name}")

        # Extract title if provided in kwargs, otherwise use default
        title = kwargs.pop('title', f"Air Quality over {region['name']}")

        return plot_map(
            data_array,
            title=title,
            extent=region['bounds'],
            mask_geometry=region['geometry'],
            **kwargs
        )
