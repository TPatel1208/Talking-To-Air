"""Server-rendered data overlay PNGs for the MapLibre map engine (T23).

Reprojects a lat/lon (EPSG:4326) grid to Web Mercator (EPSG:3857) so the
raster's internal pixel spacing matches what MapLibre's `image` source
expects when it projects the four corner coordinates, then colorizes
through the same LUT the payload ships (utils.colormaps) so the overlay,
the client colorbar, and the export renderer can never disagree on what a
value looks like.
"""
from __future__ import annotations

import io

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

_SRC_CRS = CRS.from_epsg(4326)
_DST_CRS = CRS.from_epsg(3857)


def render_overlay_png(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    lut: list[list[int]],
    vmin: float,
    vmax: float,
) -> bytes:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    values = np.asarray(values, dtype=float)

    # GDAL/rasterio convention: row 0 is the north (top) edge, i.e. a
    # negative per-row latitude step. Flip when the source is south->north.
    if lats.size > 1 and lats[0] < lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]

    lon_res = (lons[-1] - lons[0]) / (lons.size - 1) if lons.size > 1 else 1.0
    lat_res = (lats[-1] - lats[0]) / (lats.size - 1) if lats.size > 1 else -1.0

    src_transform = Affine.translation(lons[0] - lon_res / 2, lats[0] - lat_res / 2) * Affine.scale(lon_res, lat_res)
    left, right = lons[0] - lon_res / 2, lons[-1] + lon_res / 2
    top, bottom = lats[0] - lat_res / 2, lats[-1] + lat_res / 2

    height, width = values.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        _SRC_CRS, _DST_CRS, width, height, left, bottom, right, top
    )

    nodata = np.nan
    destination = np.full((dst_height, dst_width), nodata, dtype=np.float64)
    reproject(
        source=values,
        destination=destination,
        src_transform=src_transform,
        src_crs=_SRC_CRS,
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=_DST_CRS,
        dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )

    rgba = _colorize(destination, lut, vmin, vmax)

    buf = io.BytesIO()
    import matplotlib.image as mpimg

    mpimg.imsave(buf, rgba, format="png")
    return buf.getvalue()


def _colorize(arr: np.ndarray, lut: list[list[int]], vmin: float, vmax: float) -> np.ndarray:
    lut_arr = np.asarray(lut, dtype=np.uint8)
    n = len(lut_arr)
    valid = np.isfinite(arr)

    span = (vmax - vmin) or 1.0
    safe_arr = np.where(valid, arr, vmin)
    normalized = np.clip((safe_arr - vmin) / span, 0.0, 1.0)
    idx = np.clip((normalized * (n - 1)).round().astype(int), 0, n - 1)

    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[valid] = lut_arr[idx[valid]]
    return rgba
