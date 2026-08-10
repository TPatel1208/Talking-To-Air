"""T59 Risk 2 — does the bucketed block mean stay lazy? (PRD "Architectural Constraint")

Question
--------
Phase 3 plans to compute frames as::

    da.groupby(bucket).mean(time).coarsen(lat=k, lon=k, boundary="pad").mean()

``groupby(...).mean()`` alone emits N fields at **native** resolution. At the
render path's measured ~21 B/cell that is the unbounded OOM of 2026-08-05,
multiplied by N. The PRD's claim is that composing ``.coarsen(...).mean()``
onto the grouped result *before* ``.compute()`` keeps the materialized output
at N x 20,000 (~80 KB/frame).

"Lazy in principle" is not evidence. If dask in fact materializes the grouped
intermediate at native resolution the memory budget is wrong by ~250x and
Phase 3 needs a chunked rewrite -- so this is measured on a genuine
multi-granule TEMPO open, through the production open + mask path, before
anything is built on it.

boundary="pad", never "trim": trim silently clips up to k-1 rows and columns
off the region edge, and the edge of a masked region is where the interesting
gradients are.

Peak RSS is read from ``/proc/self/status`` VmHWM, reset per-stage via
``/proc/self/clear_refs`` so each stage's high-water mark is its own rather
than the process's since start. tracemalloc runs alongside because it does
capture numpy allocations (T51 note) and attributes them to a traceback,
which VmHWM cannot.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


# ---------------------------------------------------------------- measurement


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return -1


def reset_peak_rss() -> None:
    """Reset VmHWM so the next stage's peak is attributable to that stage.

    Linux clears the peak-RSS high-water mark on a write of "5" to
    ``clear_refs``. Without it VmHWM is monotonic since process start and every
    stage after the first would report the first stage's number.
    """
    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError as exc:  # pragma: no cover - kernel without CONFIG_PROC_PAGE
        print(f"  ! could not reset VmHWM ({exc}); peaks are since process start")


class Stage:
    """Measure wall time, peak RSS and peak Python-tracked allocation."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        reset_peak_rss()
        self.rss_before = _status_kb("VmRSS")
        tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        _cur, self.tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.rss_after = _status_kb("VmRSS")
        self.rss_peak = _status_kb("VmHWM")
        print(
            f"  [{self.label}] {self.seconds:.2f}s   "
            f"RSS {self.rss_before / 1024:.0f} -> {self.rss_after / 1024:.0f} MB   "
            f"peak RSS {self.rss_peak / 1024:.0f} MB   "
            f"tracemalloc peak {self.tm_peak / 1e6:.1f} MB"
        )
        return False


def mb(nbytes: float) -> str:
    return f"{nbytes / 1e6:,.1f} MB"


# ------------------------------------------------------------------- pipeline


def bucket_labels(stamps: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    """The same bucketing ``_cadence_weighted_mean`` uses."""
    if cadence == "monthly":
        buckets = stamps.to_period("M")
    else:
        buckets = stamps.floor({"hourly": "h", "daily": "D"}[cadence])
    return np.asarray([str(b) for b in buckets])


def coarsen_factors(n_lat: int, n_lon: int, target_cells: int) -> tuple[int, int]:
    """Block size that lands at or just under ``target_cells`` output cells.

    One shared factor per axis derived from the area ratio, so blocks stay
    square-ish in index space and a plume is not smeared preferentially along
    one axis.
    """
    total = n_lat * n_lon
    if total <= target_cells:
        return 1, 1
    k = int(np.ceil((total / target_cells) ** 0.5))
    return max(k, 1), max(k, 1)


def open_and_mask(path: str, bbox, col_info: dict):
    """The production open + mask path, exactly as characterize_d4 established."""
    with Stage("open bundle"):
        ds = _open_netcdf_bundle(path)
        da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    print(f"  variable={da.name} dims={dict(da.sizes)} time_dim={time_dim}")
    print(f"  dask-backed at open: {da.chunks is not None}")
    if da.chunks is not None:
        print(f"  chunks: {[tuple(sorted(set(c))) for c in da.chunks]}")

    if bbox is not None:
        lat, lon = find_lat_coord(da), find_lon_coord(da)
        minx, miny, maxx, maxy = bbox
        sel = {
            lat: slice(miny, maxy) if da[lat][0] < da[lat][-1] else slice(maxy, miny),
            lon: slice(minx, maxx) if da[lon][0] < da[lon][-1] else slice(maxx, minx),
        }
        da, ds = da.sel(sel), ds.sel(sel)
        print(f"  cropped to {bbox}: {dict(da.sizes)}")

    with Stage("resolve + mask"):
        masked = SVC._resolve_and_mask(
            da,
            col_info=col_info,
            collection_id=col_info.get("collection_id"),
            source_ds=ds,
        )
    field, counts = masked.data, masked.counts
    print(f"  masking: qa_status={masked.provenance.get('qa_status')} "
          f"dask-backed after mask: {field.chunks is not None}")

    valid = SVC._valid_time_indices(field, time_dim, counts)
    print(f"  timesteps: {field.sizes[time_dim]} total, {len(valid)} valid after masking")
    field = field.isel({time_dim: valid})
    return ds, field, time_dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--target-cells", type=int, default=20_000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--arm", choices=("fused", "native"), default="fused",
        help="fused = groupby().mean().coarsen().mean() computed as one graph (the "
             "design under test). native = groupby().mean().compute() first, then "
             "coarsen -- the shape the PRD says must NOT happen. Run each in its own "
             "process: peak RSS is not comparable across arms sharing a heap.",
    )
    ap.add_argument(
        "--allow-native-mb", type=float, default=400.0,
        help="refuse the native arm above this estimated stack size rather than "
             "OOM-killing the backend (the big bundles have done that before).",
    )
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    bbox = tuple(args.bbox) if args.bbox else None

    print(f"\n{'=' * 78}\nbundle: {args.bundle}   target cells: {args.target_cells:,}")
    ds, field, time_dim = open_and_mask(args.bundle, bbox, col_info)

    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    labels = bucket_labels(stamps, cadence)
    n_buckets = len(pd.unique(labels))
    lat, lon = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat], field.sizes[lon]
    k_lat, k_lon = coarsen_factors(n_lat, n_lon, args.target_cells)
    out_lat = int(np.ceil(n_lat / k_lat))
    out_lon = int(np.ceil(n_lon / k_lon))

    native_stack = n_buckets * n_lat * n_lon * 4
    coarse_stack = n_buckets * out_lat * out_lon * 4
    print(f"\n  arm={args.arm}")
    print(f"  cadence={cadence}  buckets={n_buckets}  native grid={n_lat}x{n_lon} "
          f"({n_lat * n_lon:,} cells)")
    print(f"  coarsen k=({k_lat},{k_lon}) -> {out_lat}x{out_lon} = {out_lat * out_lon:,} cells/frame")
    print(f"  N x native  = {mb(native_stack)}   (float32, the shape that must NOT materialize)")
    print(f"  N x coarse  = {mb(coarse_stack)}   (float32, the intended materialization)")

    grouper = xr.DataArray(labels, dims=[time_dim], coords={time_dim: field[time_dim]}).rename("bucket")

    # ---- the graph under test, composed but NOT computed -------------------
    grouped = field.groupby(grouper).mean(dim=time_dim, skipna=True)
    frames = grouped.coarsen({lat: k_lat, lon: k_lon}, boundary="pad").mean(skipna=True)

    print("\n  --- laziness of the composed graph (before compute) ---")
    print(f"  grouped lazy : {grouped.chunks is not None}  shape={grouped.shape}")
    print(f"  frames  lazy : {frames.chunks is not None}  shape={frames.shape}")
    if frames.chunks is not None:
        print(f"  frames chunks: {[tuple(sorted(set(c))) for c in frames.chunks]}")
        print(f"  frames graph tasks: {len(frames.data.dask):,}")
        biggest = int(np.prod([max(c) for c in frames.chunks])) * 4
        print(f"  largest frames chunk: {mb(biggest)}")

    result = {
        "bundle": args.bundle,
        "arm": args.arm,
        "cadence": cadence,
        "n_buckets": n_buckets,
        "native_grid": [n_lat, n_lon],
        "coarsen_k": [k_lat, k_lon],
        "frame_grid": [out_lat, out_lon],
        "cells_per_frame": out_lat * out_lon,
        "native_stack_bytes": native_stack,
        "coarse_stack_bytes": coarse_stack,
        "grouped_lazy": grouped.chunks is not None,
        "frames_lazy": frames.chunks is not None,
        "frames_graph_tasks": len(frames.data.dask) if frames.chunks is not None else None,
    }

    if args.arm == "native":
        est_mb = native_stack / 1e6
        if est_mb > args.allow_native_mb:
            print(f"\n  REFUSED: native arm would materialize {mb(native_stack)} > "
                  f"--allow-native-mb {args.allow_native_mb} MB. The estimate IS the "
                  f"finding; raise the cap deliberately if you want to pay it.")
            result["native"] = {"refused_estimate_mb": round(est_mb, 1)}
        else:
            print("\n  --- computing grouped at NATIVE resolution, then coarsening ---")
            with Stage("grouped.compute() [native]") as st:
                native = grouped.compute()
            print(f"  native intermediate {native.shape} {mb(native.nbytes)} {native.dtype}")
            with Stage("coarsen after") as st2:
                out = native.coarsen({lat: k_lat, lon: k_lon}, boundary="pad").mean(skipna=True)
            result["native"] = {
                "grouped_seconds": round(st.seconds, 3),
                "grouped_peak_rss_mb": round(st.rss_peak / 1024, 1),
                "grouped_tracemalloc_peak_mb": round(st.tm_peak / 1e6, 1),
                "native_intermediate_nbytes": int(native.nbytes),
                "coarsen_seconds": round(st2.seconds, 3),
                "coarsen_peak_rss_mb": round(st2.rss_peak / 1024, 1),
            }
    else:
        print("\n  --- computing frames (fused graph) ---")
        with Stage("frames.compute()") as st:
            out = frames.compute()
        result["fused"] = {
            "seconds": round(st.seconds, 3),
            "rss_before_mb": round(st.rss_before / 1024, 1),
            "peak_rss_mb": round(st.rss_peak / 1024, 1),
            "peak_rss_delta_mb": round((st.rss_peak - st.rss_before) / 1024, 1),
            "rss_after_mb": round(st.rss_after / 1024, 1),
            "tracemalloc_peak_mb": round(st.tm_peak / 1e6, 1),
            "out_shape": list(out.shape),
            "out_nbytes": int(out.nbytes),
            "out_dtype": str(out.dtype),
        }
        print(f"  materialized shape {out.shape}  nbytes {mb(out.nbytes)}  dtype {out.dtype}")
        print(f"  peak RSS above pre-compute baseline: "
              f"{(st.rss_peak - st.rss_before) / 1024:.1f} MB   "
              f"(N x native would be {mb(native_stack)})")

        # boundary="pad" must not blank the edge: if coarsen's mean did not skip
        # the NaN pad, the trailing row/col would be NaN wherever padding landed.
        vals = out.values
        finite_cols = np.isfinite(vals).any(axis=tuple(range(vals.ndim - 1)))
        finite_rows = np.isfinite(vals).any(axis=(0, 2)) if vals.ndim == 3 else None
        print(f"  pad check: trailing coarse column finite: {bool(finite_cols[-1])}  "
              f"trailing coarse row finite: "
              f"{bool(finite_rows[-1]) if finite_rows is not None else 'n/a'}")
        result["fused"]["last_col_finite"] = bool(finite_cols[-1])
        result["fused"]["last_row_finite"] = (
            bool(finite_rows[-1]) if finite_rows is not None else None
        )

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
