"""Geographic utility helpers.

Bounding boxes may be provided as a comma-separated string, a flat four-item
list or tuple, or a single-item list/tuple wrapper around either form.
"""

from typing import Any, Tuple


LAT_COORD_CANDIDATES = ("lat", "latitude", "Latitude", "LAT")
LON_COORD_CANDIDATES = ("lon", "longitude", "Longitude", "LON", "long")


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


def find_lat_coord(da: Any) -> str | None:
    """Return the first recognized latitude coordinate name on a DataArray."""
    return next((name for name in LAT_COORD_CANDIDATES if name in da.coords), None)


def find_lon_coord(da: Any) -> str | None:
    """Return the first recognized longitude coordinate name on a DataArray."""
    return next((name for name in LON_COORD_CANDIDATES if name in da.coords), None)
