"""T59 Phase 3 -- which term in the fused reduction costs the memory.

Phase 2 measured the BARE composition ``groupby().mean().coarsen().mean()`` on
a 36-granule full-domain TEMPO open at ~1.1 GB peak. Phase 3 hangs four more
consumers off the same native intermediate in the same ``compute()``: the
per-frame statistics, the coverage area, the pooled histogram and (in the
coarsened tier) the delta terms. Laziness is preserved either way -- but every
consumer of a chunk keeps that chunk alive until it has finished with it, so
fusing them multiplies the CONCURRENT working set even though nothing
materializes at native resolution.

The full-domain production-shaped call was OOM-killed in a 3.9 GB container
(cgroup ``oom_kill``), so this walks the arms from cheapest to most complete,
each as its own ``compute()`` with its own peak, and stops where it dies. The
last arm to survive is the answer.

Run in the background; each arm is a fresh I/O pass over the bundle::

    docker exec tta-backend sh -c 'cd /app && nohup python -u \
        scripts/probe_t59_frame_stack_memory.py \
        /data/harmony/job_d175709729a518f2/result.nc.zip \
        > /tmp/t59_mem.log 2>&1 & echo started'
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing import frame_stack as fs
from tta_backend.preprocessing.aggregation_service import AggregationService
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
        print(f"  ! could not reset VmHWM ({exc})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None)
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--target-cells", type=int, default=20_000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()

    print(f"\n{'=' * 78}\nbundle: {args.bundle}", flush=True)
    t0 = time.perf_counter()
    ds = _open_netcdf_bundle(args.bundle)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    if args.bbox:
        lat, lon = find_lat_coord(da), find_lon_coord(da)
        minx, miny, maxx, maxy = args.bbox
        sel = {
            lat: slice(miny, maxy) if da[lat][0] < da[lat][-1] else slice(maxy, miny),
            lon: slice(minx, maxx) if da[lon][0] < da[lon][-1] else slice(maxx, minx),
        }
        da, ds = da.sel(sel), ds.sel(sel)
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
    )
    field, counts = masked.data, masked.counts
    field = field.isel({time_dim: SVC._valid_time_indices(field, time_dim, counts)})
    print(f"  open+mask {time.perf_counter() - t0:.1f}s  peak RSS "
          f"{_status_kb('VmHWM') / 1024:.0f} MB", flush=True)

    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)
    lat_dim, lon_dim = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat_dim], field.sizes[lon_dim]
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    pad = pd.Timedelta(hours=3)
    span = ((stamps.min() - pad).isoformat(), (stamps.max() + pad).isoformat())

    # Rebuild exactly what build_frame_stack builds, so the arms below differ
    # from production only by which consumers are attached.
    axis = fs._axis_starts(span, cadence)
    axis_labels = [s.isoformat() for s in axis]
    grouper = xr.DataArray(
        np.array([s.isoformat() for s in fs._bucket_starts(stamps, cadence)]),
        dims=[time_dim], coords={time_dim: field[time_dim]},
    ).rename("bucket")
    spatial = [d for d in field.dims if d != time_dim]
    contributed = np.isfinite(field).any(spatial)
    buckets = field.groupby(grouper).mean(dim=time_dim, skipna=True).reindex(
        bucket=axis_labels,
    )
    granules = contributed.groupby(grouper).sum(dim=time_dim).reindex(
        bucket=axis_labels,
    ).fillna(0.0)
    period_native = buckets.mean(dim="bucket", skipna=True)
    k_lat, k_lon = fs.coarsen_factors(n_lat, n_lon, args.target_cells)
    coarse = fs._block_mean(buckets, lat_dim, lon_dim, k_lat, k_lon)
    period_coarse = fs._block_mean(period_native, lat_dim, lon_dim, k_lat, k_lon)
    edges = fs._pooled_bin_edges(field, None, counts)

    native_stack_mb = len(axis) * n_lat * n_lon * 8 / 1e6
    print(f"  cadence={cadence}  grid={n_lat}x{n_lon} ({n_lat * n_lon:,} cells)  "
          f"buckets={len(axis)}  k=({k_lat},{k_lon})", flush=True)
    print(f"  N x native float64 = {native_stack_mb:,.0f} MB "
          f"(the shape that must not materialize)", flush=True)

    # Frame 0 goes LAST, deliberately. At full domain the run died the moment it
    # was added, but that arm was second -- so "the cross-bucket reduction is
    # expensive" and "one more consumer is expensive" were not yet separable.
    # Every other consumer reduces WITHIN a chunk and drops it; only the period
    # mean has to combine chunks along the axis they are chunked on. Stacking the
    # within-chunk ones first and the cross-chunk one last is what tells them
    # apart.
    within_chunk = {
        "granules + coverage": lambda: {
            "n_granules": granules,
            "observed_area": fs._observed_area(buckets, (lat_dim, lon_dim)),
        },
        "per-frame statistics": lambda: fs._native_statistics(buckets, (lat_dim, lon_dim)),
        "pooled histogram": lambda: {
            "pooled_histogram": (
                fs._histogram(buckets, edges) + fs._histogram(period_native, edges)
            ),
        },
    }
    cumulative: dict = {"values": coarse}
    arms = [("1 frames only (Phase 2's shape)", dict(cumulative))]
    for index, (label, build) in enumerate(within_chunk.items(), start=2):
        cumulative.update(build())
        arms.append((f"{index} + {label}", dict(cumulative)))
    cumulative["period_values"] = period_coarse
    arms.append(("5 + frame 0, a CROSS-BUCKET reduction (= production)", dict(cumulative)))

    results = {
        "bundle": args.bundle, "grid": [n_lat, n_lon], "buckets": len(axis),
        "coarsen_k": [k_lat, k_lon], "n_times_native_mb": round(native_stack_mb, 1),
        "arms": [],
    }
    for label, variables in arms:
        reset_peak_rss()
        before = _status_kb("VmRSS")
        t = time.perf_counter()
        out = xr.Dataset(variables).compute()
        seconds = time.perf_counter() - t
        peak = _status_kb("VmHWM")
        materialized = sum(int(v.nbytes) for v in out.data_vars.values())
        print(f"  [{label}] {seconds:.1f}s   RSS {before / 1024:.0f} -> "
              f"{_status_kb('VmRSS') / 1024:.0f} MB   peak {peak / 1024:.0f} MB   "
              f"materialized {materialized / 1e6:.2f} MB", flush=True)
        results["arms"].append({
            "arm": label, "seconds": round(seconds, 1),
            "peak_rss_mb": round(peak / 1024, 1),
            "rss_before_mb": round(before / 1024, 1),
            "materialized_mb": round(materialized / 1e6, 3),
        })
        del out
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(results, fh, indent=2)

    print("\n  all arms survived", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
