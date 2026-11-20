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

from typing import Any, Optional, Tuple, Union

from tta_backend.infrastructure.settings import get_settings
from tta_backend.datasets.us_states import US_STATES
from tta_backend.utils import region_buffer, region_composition, region_dispatch
from tta_backend.earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError
from tta_backend.domain.geo import find_lat_coord, find_lon_coord
from tta_backend.domain.masking.geometry import mask_data_by_geometry

logger = logging.getLogger(__name__)
_geocoding_service = None

# Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/)
# requires an identifying User-Agent naming the application and a means of
# contact. An unfilled placeholder is refused: Nominatim's edge 403s any
# User-Agent containing an @example.com/.org/.net address, so a real address
# is required for geocoding to work at all.
NOMINATIM_USER_AGENT = f"talking-to-air/1.0 (contact: {get_settings().nominatim_contact_email})"


def get_geocoding_service() -> "GeocodingService":
    """Return the shared geocoder so agent and plotting tools share cache."""
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service


# Real (simplified Natural Earth 110m) boundaries for the multi-country
# presets, so "mean over the US" doesn't average in Sonora and the Atlantic.
# Built once by scripts/build_preset_regions.py and checked in under
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
def load_preset_polygons() -> dict[str, Any]:
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


# The 242 individually-addressable countries, usable in the ``+`` grammar. A
# separate asset from the one above, for two reasons:
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
def load_admin0_polygons() -> dict[str, Any]:
    """``{normalized ADMIN: {"name": ADMIN, "geometry": geom}}``, parsed once.

    Keys are already normalized by the builder, so ``"bahamas"`` resolves even
    though the source spells it ``"The Bahamas"`` -- the only one of the 242
    that ``_normalize_location_name`` changes.

    Carrying the source's own ``ADMIN`` spelling alongside the geometry is why
    this returns a richer value than ``load_preset_polygons``: the key is
    lower-cased and ``"the "``-stripped, and an answer that cited *that* would
    say "Bahamas", "Eswatini" and "Hong Kong S.A.R" for places Natural Earth
    spells "The Bahamas", "eSwatini" and "Hong Kong S.A.R.". ``display_name``
    is what an answer cites, so it gets the real spelling rather than a
    ``.title()`` reconstruction of a normalized key.

    Degrades to ``{}`` like ``load_preset_polygons``: country tokens then fail
    as unknown. Deliberately fail-closed: there's no bounding-box fallback, and
    anything containing ``+`` never goes to the geocoder."""
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
) -> tuple[Figure, Any]:
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
            time_dim = str(data_array.dims[0])

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
            fig_width, fig_height = 10.0, 6.0
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


class GeocodingService:
    """Free geocoding using Nominatim (OpenStreetMap) with polygon and bounding box"""

    def __init__(self, cache_ttl_seconds: int = 24 * 60 * 60) -> None:
        # normalized name -> (expires_at, geocode result)
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.cache_ttl_seconds = cache_ttl_seconds
        self.last_request = 0.0
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

    def _get_cached(self, location_name: str) -> dict[str, Any] | None:
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

    def _store_cached(self, location_name: str, result: dict[str, Any]) -> None:
        key = self._cache_key(location_name)
        self.cache[key] = (time.time() + self.cache_ttl_seconds, result)

    def geocode(self, location_name: str) -> dict[str, Any] | None:
        """Convert location name to coordinates, polygon, and bounding box"""
        cached = self._get_cached(location_name)
        if cached is not None:
            return cached

        # Rate limit: 1 request per second
        wait_seconds = self._reserve_throttle_slot()
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        url = "https://nominatim.openstreetmap.org/search"
        params: dict[str, str | int] = {
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

    async def ageocode(self, location_name: str) -> dict[str, Any] | None:
        """Async version of geocode() for agent tools running on the event loop."""
        cached = self._get_cached(location_name)
        if cached is not None:
            return cached

        wait_seconds = self._reserve_throttle_slot()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        url = "https://nominatim.openstreetmap.org/search"
        params: dict[str, str | int] = {
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
        'lower 48 states':         {'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},
        'continental united states':{'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},
        'contiguous united states':{'geometry': box(-125, 24, -66, 50), 'bounds': (-125, 24, -66, 50), 'name': 'Continental US'},

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
        # The 51 U.S. admin-1 units, from the generated table. Ordinary
        # presets (a state has an honest bounding box, upgraded to the real
        # boundary by ``_finalize_preset``). ``display_name`` disambiguates,
        # e.g. "washington" otherwise geocodes to Washington DC.
        for _key, _state in US_STATES.items():
            if _key in self.global_regions:
                # These go in by assignment, so an existing key would be
                # silently replaced (e.g. "georgia" the country vs the state).
                # Fail naming the key.
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
        # The alias/coalition tables are hand-maintained; one that silently
        # shadows "us" or "georgia" would resolve a confident wrong region, so
        # check against this instance's presets.
        region_dispatch.assert_no_alias_collisions(self.global_regions)
        # The country table, same rule. Kept a separate call because it
        # deliberately does not open the country asset -- see its docstring;
        # doing so here would spend the 165 ms cold parse on every resolver,
        # including the ones that only ever resolve "paris".
        region_composition.assert_no_country_collisions(self.global_regions)

    # Preset keys that resolve to a real polygon (feature id in
    # preset_regions.geojson). Everything else in ``global_regions`` stays a
    # disclosed bounding box -- pure-ocean/quadrant concepts where a rectangle
    # is the honest answer.
    _POLYGON_PRESET_IDS = {
        "united states": "united states", "usa": "united states", "us": "united states",
        "conus": "conus", "continental us": "conus",
        "contiguous us": "conus", "lower 48": "conus",
        "lower 48 states": "conus", "continental united states": "conus",
        "contiguous united states": "conus",
        "north america": "north america", "south america": "south america",
        "europe": "europe", "africa": "africa", "asia": "asia",
        "oceania": "oceania", "antarctica": "antarctica",
        # Coalitions. These are NOT ``global_regions`` keys (a coalition has
        # no honest bounding box), so they are unreachable via
        # the exact-match gate below and arrive only through region_dispatch.
        "otc": "otc", "new england": "new england",
        # The 51 states. Feature id is the lowercased name, so
        # the preset key and the polygon id are the same string.
        **{key: key for key in US_STATES},
    }

    def _finalize_preset(self, preset: dict[str, Any], key: str) -> dict[str, Any]:
        """Return a preset enriched with the region-fidelity disclosure fields.
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
    def _geocoded_region(geo_result: dict[str, Any]) -> dict[str, Any]:
        """Build a RegionResult from a geocoder hit, disclosing which kind of
        footprint it is: ``polygon`` when Nominatim returned a real boundary,
        or ``point_buffer`` when it didn't and we mint a 0.1° box around the
        centroid. ``display_name`` is the geocoder's own label, carried
        through so a wrong-place answer ("Paris, Texas") is catchable. Shared
        by the sync and async resolvers so a place can't resolve two ways.

        A GeoJSON hit is not automatically a boundary. Nominatim wraps a
        *point* result in GeoJSON as readily as an administrative area, so
        ``geojson is not None`` was never the same question as "did we get a
        footprint" -- both ``"OTC"``
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
        """One normalization for both resolvers (sync and async must agree):
        lower, strip, collapse whitespace, and drop a leading "the " and then
        a leading "entire " or "whole ", so "the Netherlands" and "the entire
        continental United States" resolve the same either way. Only the key
        is normalized; the geocoder still gets the user's own wording."""
        normalized = " ".join(location_name.lower().strip().split())
        normalized = normalized.removeprefix("the ")
        return normalized.removeprefix("entire ").removeprefix("whole ")

    def resolve_location(self, location_name: str) -> dict[str, Any] | None:
        """Convert location name to RegionResult with geometry"""
        # The ``+`` grammar reads the RAW string, before normalization, so
        # "ny + the nj" and "the ny + nj" resolve the same. Raises rather than
        # returning None on a bad token.
        composed = region_composition.dispatch_composite(location_name, self)
        if composed.claimed:
            return composed.region
        # The buffer grammar runs AFTER composition on purpose: "within 50 km
        # of NY + NJ" must fail in the composition grammar, since buffering a
        # composite isn't allowed. (Sync twin; production uses
        # ``aresolve_location``, but the region tests reach this.)
        buffered = region_buffer.dispatch_buffer(location_name, self)
        if buffered.claimed:
            return buffered.region
        # Check for global regions first
        location_lower = self._normalize_location_name(location_name)
        dispatched = region_dispatch.dispatch(location_lower, self)
        if dispatched.claimed:
            # Claimed-and-failed returns None here and never reaches the
            # geocoder; callers report "Could not resolve location".
            return dispatched.region
        if location_lower in self.global_regions:
            return self._finalize_preset(self.global_regions[location_lower], location_lower)

        geo_result = self.geocoding_service.geocode(location_name)
        if geo_result is None:
            return None

        return self._geocoded_region(geo_result)

    async def aresolve_location(self, location_name: str) -> dict[str, Any] | None:
        """Async version of resolve_location() for agent tool execution."""
        # Same gate as the sync twin; the composition tier is pure, so both
        # share one copy.
        composed = region_composition.dispatch_composite(location_name, self)
        if composed.claimed:
            return composed.region
        # Async twin, same ordering as the sync one. Every analysis tool uses
        # this one, so the geocode doesn't block the event loop.
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

    def plot_singular(self, data_array: xr.DataArray, location_name: str, **kwargs: Any) -> tuple[Figure, Any]:
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
