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
    # The axes are 1-D and negligible; keep them in full precision, since the
    # affine transform below is built from differences between them.
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    # The grid is not: it is the whole scene, and forcing it to float64 here
    # doubled a caller's float32 field before anything had been drawn. Every
    # value on this path ends up quantized to one of 256 LUT entries, so the
    # extra mantissa buys nothing a reader could ever see. Preserve whatever
    # float the caller has (no copy in the common case) and only convert when
    # it is not floating point at all -- NaN is the no-data sentinel below, so
    # an integer grid genuinely cannot be used as-is.
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)

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
    # Match the source dtype: rasterio requires it, and a float64 destination
    # for a float32 source would reintroduce the doubling avoided above -- the
    # reprojected grid is the larger of the two arrays.
    destination = np.full((dst_height, dst_width), nodata, dtype=values.dtype)
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

    # Normalize through a single working buffer. The arithmetic is unchanged;
    # only the number of live copies is. Written out as
    #   np.clip((np.where(...) - vmin) / span, 0, 1) * (n-1) ... .astype(int)
    # every operator materializes another full-size array -- six of them, at
    # the reprojected grid's size -- and they are all alive at once. That,
    # plus an int64 index for a value that never exceeds 255, is most of what
    # made a native-resolution render cost tens of bytes per cell.
    # np.where fills no-data with vmin (not NaN) so nothing invalid survives
    # into the integer cast below; those pixels are zeroed out afterwards.
    scaled = np.where(valid, arr, vmin)
    scaled -= vmin
    scaled /= span
    np.clip(scaled, 0.0, 1.0, out=scaled)
    scaled *= n - 1
    np.round(scaled, out=scaled)
    # Clipped to [0, 1] before scaling, so the index is already in range --
    # the outer np.clip this replaces was guarding arithmetic that cannot
    # leave the interval. A LUT is 256 entries (utils.colormaps), so the
    # index fits a byte; numpy's default int is int64, eight times wider.
    idx = scaled.astype(np.uint8 if n <= 256 else np.uint16)
    del scaled

    # Gather straight into the output rather than allocating zeros and filling
    # the valid pixels through a second fancy-index (which built two more
    # temporaries sized by the valid count).
    rgba = lut_arr[idx]
    np.logical_not(valid, out=valid)
    rgba[valid] = 0          # no-data stays fully transparent, as before
    return rgba
