"""Geographic utility helpers.

Bounding boxes may be provided as a comma-separated string, a flat four-item
list or tuple, or a single-item list/tuple wrapper around either form.
"""

from typing import Tuple


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
