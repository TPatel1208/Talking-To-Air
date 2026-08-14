"""T59 Phase 3 -- the bucketed reduction, checked against a real bundle.

The unit tests are deterministic fixtures that reproduce each mechanism at its
smallest. This runs the same code through the production open + mask path on a
materialized TEMPO NO2 Harmony bundle and checks the four claims that only real
data can falsify:

1. **D14 tier one is exact.** ``mean(frames)`` must equal
   ``_cadence_weighted_mean`` -- the reduction every multi-granule map already
   goes through -- to float32 noise, on a real coverage pattern with real swath
   tiling. Reported as the D16 metric so a residual is a number rather than a
   boolean.
2. **D9's pooled scale is not the union of per-frame clips.** The fixture makes
   them 400x apart by construction; this measures how far apart they are on a
   real scene, which is the number that decides whether D9 was worth the
   divergence from the Map tab's scale.
3. **D10's coverage inflation.** Phase 2 measured 99.6% apparent against 94.7%
   true on block-meaned frames. This checks the shipped ``valid_fraction``
   against what the frame grid would have reported for the same frames.
4. **One graph walk, bounded materialization.** Peak RSS and the materialized
   frame stack against the ``N x native`` shape that must not appear.

Reproduce with::

    docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_frame_stack.py \
        /data/harmony/job_52a95bb4cb79e2ee/result.nc.zip'
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from tta_backend.preprocessing.aggregation_service import AggregationService, cos_lat_weights
from tta_backend.preprocessing.frame_stack import build_frame_stack
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return -1


def reset_peak_rss() -> None:
    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError as exc:  # pragma: no cover
        print(f"  ! could not reset VmHWM ({exc})")


def delta_pct(f, m, weights) -> float:
    """D16's metric, spelled the same way the characterization test spells it."""
    mask = np.isfinite(f) & np.isfinite(m)
    diff = np.abs(f - m)[mask]
    mag = np.abs(m)[mask]
    w = np.broadcast_to(weights, f.shape)[mask]
    den = float((mag * w).sum())
    return 100.0 * float((diff * w).sum()) / den if den > 0 else 0.0


def open_and_mask(path: str, bbox, col_info: dict):
    """The production open + mask path, as characterize_d4 established it."""
    ds = _open_netcdf_bundle(path)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    if bbox is not None:
        lat, lon = find_lat_coord(da), find_lon_coord(da)
        minx, miny, maxx, maxy = bbox
        sel = {
            lat: slice(miny, maxy) if da[lat][0] < da[lat][-1] else slice(maxy, miny),
            lon: slice(minx, maxx) if da[lon][0] < da[lon][-1] else slice(maxx, minx),
        }
        da, ds = da.sel(sel), ds.sel(sel)
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
    )
    return ds, masked, time_dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--target-cells", type=int, default=20_000)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--pad-span-hours", type=int, default=3,
                    help="widen the requested span past the data on both sides, so "
                         "genuinely empty buckets have to appear on the axis")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    bbox = tuple(args.bbox) if args.bbox else None

    print(f"\n{'=' * 78}\nbundle: {args.bundle}")
    reset_peak_rss()
    t0 = time.perf_counter()
    ds, masked, time_dim = open_and_mask(args.bundle, bbox, col_info)
    field, counts = masked.data, masked.counts
    print(f"  open+mask {time.perf_counter() - t0:.1f}s  "
          f"peak RSS {_status_kb('VmHWM') / 1024:.0f} MB  "
          f"qa_status={masked.provenance.get('qa_status')}")

    valid = SVC._valid_time_indices(field, time_dim, counts)
    print(f"  timesteps: {field.sizes[time_dim]} total, {len(valid)} valid after masking")
    field = field.isel({time_dim: valid})

    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)
    lat_dim, lon_dim = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat_dim], field.sizes[lon_dim]
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    pad = pd.Timedelta(hours=args.pad_span_hours)
    span = ((stamps.min() - pad).isoformat(), (stamps.max() + pad).isoformat())
    print(f"  cadence={cadence}  native grid={n_lat}x{n_lon} ({n_lat * n_lon:,} cells)")
    print(f"  requested span (padded {args.pad_span_hours}h each side): {span[0]} .. {span[1]}")

    # ---------------------------------------------------------------- reduce
    reset_peak_rss()
    rss_before = _status_kb("VmRSS")
    t0 = time.perf_counter()
    stack = build_frame_stack(
        field, time_dim=time_dim, cadence=cadence, span=span,
        target_cells=args.target_cells, max_frames=args.max_frames,
        region_area=field.attrs.get("region_area"), qa_counts=counts,
    )
    seconds = time.perf_counter() - t0
    peak = _status_kb("VmHWM")
    native_stack_bytes = len(stack.frames) * n_lat * n_lon * 8
    print(f"\n  build_frame_stack {seconds:.1f}s   "
          f"RSS {rss_before / 1024:.0f} -> {_status_kb('VmRSS') / 1024:.0f} MB   "
          f"peak {peak / 1024:.0f} MB")
    print(f"  tier={stack.tier}  frames={len(stack.frames)}  "
          f"buckets/frame={stack.buckets_per_frame}  k={stack.coarsen_k}  "
          f"cells/frame={stack.cells_per_frame:,} (ceiling {args.target_cells:,})")
    print(f"  materialized {stack.values.nbytes / 1e6:.2f} MB {stack.values.dtype}   "
          f"N x native float64 would be {native_stack_bytes / 1e6:,.0f} MB")

    empty = [f for f in stack.frames if f.n_granules == 0]
    print(f"  empty buckets on the axis: {len(empty)} of {len(stack.frames)}")
    if empty:
        print(f"    first: {empty[0].t_start}  valid_fraction={empty[0].valid_fraction} "
              f"qa_pass_rate={empty[0].qa_pass_rate}")

    result = {
        "bundle": args.bundle,
        "cadence": cadence,
        "native_grid": [n_lat, n_lon],
        "tier": stack.tier,
        "n_frames": len(stack.frames),
        "buckets_per_frame": stack.buckets_per_frame,
        "coarsen_k": list(stack.coarsen_k),
        "cells_per_frame": stack.cells_per_frame,
        "empty_buckets": len(empty),
        "seconds": round(seconds, 2),
        "peak_rss_mb": round(peak / 1024, 1),
        "materialized_mb": round(stack.values.nbytes / 1e6, 3),
        "n_times_native_float64_mb": round(native_stack_bytes / 1e6, 1),
        "delta": stack.delta,
        "value_range": list(stack.value_range) if stack.value_range else None,
    }

    # ---- claim 1: D14 tier one is exact ------------------------------------
    print("\n  --- claim 1: the period map IS the average of the frames ---")
    native = build_frame_stack(
        field, time_dim=time_dim, cadence=cadence, span=span,
        target_cells=n_lat * n_lon, max_frames=args.max_frames, qa_counts=counts,
    )
    production = SVC._cadence_weighted_mean(field, time_dim, cadence).compute()
    weights = np.asarray(cos_lat_weights(production).values)[:, None] \
        if cos_lat_weights(production) is not None else 1.0
    if production.dims[0] != lat_dim:  # (lon, lat) ordering
        weights = np.asarray(cos_lat_weights(production).values)[None, :]
    prod = np.asarray(production.values, dtype="float64")
    derived = np.asarray(native.period_values, dtype="float64")
    stack_mean = np.nanmean(
        np.asarray(native.values, dtype="float64"), axis=0,
    ) if native.tier == "cadence" else None

    result["tier1"] = {
        "tier": native.tier,
        "period_vs_cadence_weighted_mean_pct": delta_pct(derived, prod, weights),
    }
    print(f"    frame-0 vs _cadence_weighted_mean: "
          f"{result['tier1']['period_vs_cadence_weighted_mean_pct']:.6f} %  "
          f"(float32 storage noise is the floor here)")
    if stack_mean is not None:
        result["tier1"]["stack_mean_vs_period_pct"] = delta_pct(stack_mean, derived, weights)
        print(f"    mean(frames) vs frame 0:          "
              f"{result['tier1']['stack_mean_vs_period_pct']:.6f} %")

    # ---- claim 2: pooled scale vs the union of per-frame clips --------------
    print("\n  --- claim 2: pooled 2-98 vs a union of per-frame clips (D9) ---")
    per_frame = np.asarray(native.values, dtype="float64")
    lows, highs = [], []
    for index in range(per_frame.shape[0]):
        plane = per_frame[index]
        finite = plane[np.isfinite(plane)]
        if finite.size:
            lows.append(float(np.percentile(finite, 2)))
            highs.append(float(np.percentile(finite, 98)))
    union = (min(lows), max(highs)) if lows else None
    result["pooled_vs_union"] = {"pooled": list(native.value_range), "union": list(union)}
    print(f"    pooled : {native.value_range[0]:.4e} .. {native.value_range[1]:.4e}")
    print(f"    union  : {union[0]:.4e} .. {union[1]:.4e}")
    if native.value_range[1]:
        widening = (union[1] - union[0]) / (native.value_range[1] - native.value_range[0])
        result["pooled_vs_union"]["union_is_wider_by"] = widening
        print(f"    the union's ramp is {widening:.2f}x as wide")

    # ---- claim 3: D10's coverage inflation ---------------------------------
    print("\n  --- claim 3: native coverage vs what the frame grid would say (D10) ---")
    coarse = np.asarray(stack.values, dtype="float64")
    inflation = []
    for index, frame in enumerate(stack.frames):
        if frame.n_granules == 0:
            continue
        apparent = float(np.isfinite(coarse[index]).mean())
        inflation.append((frame.valid_fraction, apparent))
    if inflation:
        true_med = float(np.median([t for t, _ in inflation]))
        app_med = float(np.median([a for _, a in inflation]))
        result["coverage"] = {
            "native_median": true_med,
            "frame_grid_median": app_med,
            "inflation_pp": (app_med - true_med) * 100.0,
        }
        print(f"    native (shipped) median {true_med:.4f}   "
              f"frame-grid median {app_med:.4f}   "
              f"inflation {(app_med - true_med) * 100:.2f} pp")

    print("\n  --- per-frame disclosure, first 8 ---")
    for frame in stack.frames[:8]:
        peak_value = frame.statistics.get("max")
        print(f"    {frame.t_start}  n={frame.n_granules:>2}  "
              f"valid={frame.valid_fraction:.4f}  "
              f"qa={'--' if frame.qa_pass_rate is None else f'{frame.qa_pass_rate:.4f}'}  "
              f"max={'--' if peak_value is None else f'{peak_value:.3e}'}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
