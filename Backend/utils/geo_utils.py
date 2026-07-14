"""Geographic utility helpers.

Bounding boxes may be provided as a comma-separated string, a flat four-item
list or tuple, or a single-item list/tuple wrapper around either form.
"""

from typing import Any, Tuple


LAT_COORD_CANDIDATES = ("lat", "latitude", "Latitude", "LAT")
LON_COORD_CANDIDATES = ("lon", "longitude", "Longitude", "LON", "long")

# CF unit strings a latitude/longitude axis is published against. The
# spelling varies across products (degrees_north / degree_N / degreesN),
# so the vocabulary is the identification signal, not any one spelling.
LAT_UNITS = frozenset({"degrees_north", "degree_north", "degrees_N", "degree_N", "degreesN", "degreeN"})
LON_UNITS = frozenset({"degrees_east", "degree_east", "degrees_E", "degree_E", "degreesE", "degreeE"})


def _candidate_vars(obj: Any) -> dict:
    """Map every variable name that could be a horizontal coordinate to its
    DataArray, scanning both coords and (for a Dataset) data_vars. A
    DataArray only carries coords."""
    if hasattr(obj, "data_vars"):  # Dataset
        names = list(obj.coords) + list(obj.data_vars)
    else:  # DataArray
        names = list(obj.coords)
    return {name: obj[name] for name in names}


_BOUNDS_SUFFIXES = ("_bnds", "_bounds", "_vertices", "_edges")


def _is_meta_match(var: Any, standard_name: str, units: frozenset) -> bool:
    if str(var.attrs.get("standard_name", "")).strip() == standard_name:
        return True
    return str(var.attrs.get("units", "")).strip() in units


def _tiebreak(names: list, cands: dict, dims) -> str | None:
    """Pick the axis variable among several metadata matches. Metadata
    matching widens the net onto bounds/edge variables (which carry the same
    units as their axis); prefer the actual coordinate axis over them."""
    if not names:
        return None
    non_bounds = [n for n in names if not n.endswith(_BOUNDS_SUFFIXES)]
    pool = non_bounds or names
    # Structurally the axis has the fewest dims and is a dimension
    # coordinate; a bounds/edge var carries an extra vertex dim. This
    # catches unseen bounds spellings the suffix list never enumerated.
    return min(pool, key=lambda n: (cands[n].ndim, 0 if n in dims else 1, list(cands).index(n)))


def _coordinate_refs(obj: Any) -> set:
    """Leaf names referenced by a science variable's CF `coordinates`
    attribute. Our group merge strips paths, so `geolocation/latitude` is
    matched as `latitude`."""
    raw = obj.attrs.get("coordinates") or getattr(obj, "encoding", {}).get("coordinates")
    if not raw:
        return set()
    return {token.split("/")[-1] for token in str(raw).split()}


def _pick(cands: dict, dims, standard_name: str, units: frozenset, names: tuple) -> str | None:
    # CF metadata is the primary, universal signal.
    meta = [n for n, v in cands.items() if _is_meta_match(v, standard_name, units)]
    picked = _tiebreak(meta, cands, dims)
    if picked is not None:
        return picked
    # Name allowlist is the fallback for non-CF files only.
    return next((n for n in names if n in cands), None)


def _identify(obj: Any, standard_name: str, units: frozenset, names: tuple) -> str | None:
    cands = _candidate_vars(obj)
    dims = set(obj.dims)
    # 1. The science variable's own `coordinates` pointer is authoritative,
    #    when it resolves. Opportunistic: if it doesn't, fall through.
    refs = _coordinate_refs(obj)
    if refs:
        restricted = {n: v for n, v in cands.items() if n in refs}
        picked = _pick(restricted, dims, standard_name, units, names)
        if picked is not None:
            return picked
    # 2./3. CF standard_name/units, then the name allowlist.
    return _pick(cands, dims, standard_name, units, names)


def identify_lat(obj: Any) -> str | None:
    """Return the name of the latitude variable on a Dataset or DataArray,
    identified by CF metadata rather than a variable-name allowlist."""
    return _identify(obj, "latitude", LAT_UNITS, LAT_COORD_CANDIDATES)


def identify_lon(obj: Any) -> str | None:
    """Return the name of the longitude variable on a Dataset or DataArray."""
    return _identify(obj, "longitude", LON_UNITS, LON_COORD_CANDIDATES)


TIME_NAME_CANDIDATES = ("time", "Time", "TIME", "t")


def _is_time_meta_match(var: Any) -> bool:
    if str(var.attrs.get("standard_name", "")).strip() == "time":
        return True
    if str(var.attrs.get("axis", "")).strip().upper() == "T":
        return True
    dtype = getattr(var, "dtype", None)
    return dtype is not None and _is_datetime_dtype(dtype)


def _is_datetime_dtype(dtype: Any) -> bool:
    import numpy as np

    return np.issubdtype(dtype, np.datetime64)


def identify_time(obj: Any) -> str | None:
    """Return the name of the time variable on a Dataset or DataArray,
    identified by CF metadata (``standard_name: time`` / ``axis: T`` /
    datetime64 dtype) rather than the literal name "time" -- so a
    MERRA-2-style ``valid_time`` dimension is still recognized as time (T25),
    the same CF-metadata-primary treatment T24 gave lat/lon. The bare-name
    allowlist remains the fallback for files with no CF hints at all."""
    cands = _candidate_vars(obj)
    dims = set(obj.dims)
    meta_matches = [n for n, v in cands.items() if _is_time_meta_match(v)]
    if meta_matches:
        return min(meta_matches, key=lambda n: (0 if n in dims else 1, list(cands).index(n)))
    return next((n for n in TIME_NAME_CANDIDATES if n in cands), None)


def normalise_bbox(bbox) -> Tuple[float, float, float, float]:
    """Return bbox as (min_lon, min_lat, max_lon, max_lat)."""
    while isinstance(bbox, (list, tuple)) and len(bbox) == 1:
        bbox = bbox[0]

    if isinstance(bbox, str):
        parts = [float(x) for x in bbox.split(",")]
    else:
        parts = [float(x) for x in bbox]

    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}: {bbox!r}")
    return parts[0], parts[1], parts[2], parts[3]


# Metre-like unit strings that mark a projected (x/y) grid rather than a
# geographic lat/lon one.
METRE_UNITS = frozenset({"m", "metre", "metres", "meter", "meters"})


def _projected_crs_name(obj: Any) -> str | None:
    """Return a CRS descriptor if ``obj`` carries a projected-grid marker
    (a CF ``grid_mapping`` variable / ``crs`` / ``spatial_ref``), else None."""
    for name, var in _candidate_vars(obj).items():
        gm = var.attrs.get("grid_mapping_name")
        if gm:
            return str(gm)
        if name in ("crs", "spatial_ref"):
            return str(var.attrs.get("crs_wkt") or name)
    return None


def _has_metre_xy(obj: Any) -> bool:
    cands = _candidate_vars(obj)

    def _is_metre(axis: str) -> bool:
        return axis in cands and str(cands[axis].attrs.get("units", "")).strip() in METRE_UNITS

    return _is_metre("x") and _is_metre("y")


def _grid_kind(obj: Any) -> str:
    """Classify ``obj``'s horizontal grid: ``rectilinear`` (1-D lat/lon, the
    supported case), ``curvilinear`` (2-D lat/lon swath), ``projected``
    (x/y + CRS), or ``none`` (no recognizable horizontal coordinates)."""
    lat, lon = identify_lat(obj), identify_lon(obj)
    if lat is not None and lon is not None:
        if obj[lat].ndim <= 1 and obj[lon].ndim <= 1:
            return "rectilinear"
        return "curvilinear"
    if _projected_crs_name(obj) is not None or _has_metre_xy(obj):
        return "projected"
    return "none"


def ensure_supported_grid(obj: Any) -> None:
    """Raise a T24-typed :class:`MCPToolError` when ``obj`` is on a grid the
    1-D rectilinear mask/plot math can't honor, so a researcher gets a
    specific "not supported yet" answer instead of a silent mis-mask or an
    opaque empty-coords crash. A rectilinear grid (or one with no
    recognizable coordinates — handled by the caller's own error) is a
    no-op."""
    kind = _grid_kind(obj)
    if kind not in ("curvilinear", "projected"):
        return
    from earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError

    if kind == "curvilinear":
        raise MCPToolError(
            CATEGORY_UNSUPPORTED_GRID,
            "This product is on a 2-D curvilinear (swath) grid; lat/lon plotting isn't supported yet.",
            suggestion="Try a gridded (Level 3) product for this region and time.",
        )
    crs = _projected_crs_name(obj) or "unknown"
    raise MCPToolError(
        CATEGORY_UNSUPPORTED_GRID,
        f"This product is on a projected grid (CRS: {crs}); lat/lon plotting isn't supported yet.",
        suggestion="Try a product published on a geographic lat/lon grid.",
    )


def find_lat_coord(da: Any) -> str | None:
    """Return the latitude coordinate name on a DataArray (thin wrapper over
    the canonical :func:`identify_lat`, kept for its many callers)."""
    name = identify_lat(da)
    return name if name in da.coords else None


def find_lon_coord(da: Any) -> str | None:
    """Return the longitude coordinate name on a DataArray."""
    name = identify_lon(da)
    return name if name in da.coords else None
