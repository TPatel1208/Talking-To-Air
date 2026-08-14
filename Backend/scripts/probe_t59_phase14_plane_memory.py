"""T59 Phase 14 slice 0 -- what three planes cost AT THE EXTENT GATE'S BOUNDARY.

Two measurements exist and have never met:

* ``MAX_FRAME_NATIVE_CELLS = 4,000,000`` (frame_stack.py:66) was MEASURED, on a
  ONE-statistic build: CONUS 1250x3000 = 3.75M cells at 54 hourly buckets peaked
  at **1,342 MB**, against the next point up (the full 2950x5771 TEMPO domain)
  being OOM-killed by the kernel in the 3.9 GB container.
* Phase 11 G1 measured a THREE-statistic build at **1.64x and 1.91x** the
  one-statistic build's peak RSS -- but on REGIONAL bundles, where the absolute
  deltas are +32.5 MB and +84.4 MB. Forty times smaller than the number that set
  the gate, and nowhere near the boundary the gate defends.

Phase 14 makes every build a three-statistic one, so the ratio has to be
measured where the constant lives. 1.91 x 1,342 MB is ~2.56 GB in a container
that has already been OOM-killed on this exact code path.

Unlike ``probe_t59_frame_stack_memory.py``, which rebuilds the composition by
hand to isolate which CONSUMER costs the memory, this calls the real
``build_frame_stack`` with the real ``statistics`` argument -- the thing Phase 14
is about to wire -- so what it measures is the shipped code and not a model of
it.

**One arm per process, always.** Peak RSS is ``VmHWM``, a high-water mark that
never falls, so two arms sharing a heap measure the larger one twice. The arms
are also walked cheapest-first because the failure mode here is a cgroup
``oom_kill``, not a Python exception: a killed arm shows up as a missing result,
never a traceback.

    docker exec tta-backend sh -c 'cd /app && python -u \
        scripts/probe_t59_phase14_plane_memory.py \
        /data/harmony/job_d175709729a518f2/result.nc.zip \
        --crop 1250 3000 --arm mean_only'
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from tta_backend.preprocessing import frame_stack as fs
from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()

ARMS = {
    "mean_only": ("mean",),
    "mean_max_min": ("mean", "max", "min"),
}


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return -1


def reset_peak_rss() -> None:
    """Clear ``VmHWM`` so the peak measured is the REDUCTION's, not the open's.

    The open+mask pass is identical between arms, and leaving its high-water
    mark in place would floor both arms at the same number and shrink the ratio
    under test toward 1.0 by construction.
    """
    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError as exc:  # pragma: no cover
        print(f"  ! could not reset VmHWM ({exc})", flush=True)


def _centre_crop(da, lat_dim: str, lon_dim: str, ny: int, nx: int):
    """The middle ``ny`` x ``nx`` of the grid.

    Centred rather than cornered because a corner of the TEMPO domain is mostly
    off-swath NaN, and a crop with nothing in it measures an empty reduction.
    """
    n_lat, n_lon = da.sizes[lat_dim], da.sizes[lon_dim]
    ny, nx = min(ny, n_lat), min(nx, n_lon)
    lat0, lon0 = (n_lat - ny) // 2, (n_lon - nx) // 2
    return da.isel({
        lat_dim: slice(lat0, lat0 + ny), lon_dim: slice(lon0, lon0 + nx),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--arm", choices=sorted(ARMS), default="mean_only")
    ap.add_argument("--crop", type=int, nargs=2, default=None,
                    help="native ny nx to centre-crop to (the extent under test)")
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    statistics = ARMS[args.arm]

    print(f"\n{'=' * 78}\nbundle: {args.bundle}\narm: {args.arm} "
          f"statistics={statistics}", flush=True)
    t0 = time.perf_counter()
    ds = _open_netcdf_bundle(args.bundle)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    lat_dim, lon_dim = find_lat_coord(da), find_lon_coord(da)
    if args.crop:
        da = _centre_crop(da, lat_dim, lon_dim, *args.crop)
        ds = _centre_crop(ds, lat_dim, lon_dim, *args.crop)
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"),
        source_ds=ds,
    )
    field, counts = masked.data, masked.counts
    field = field.isel({time_dim: SVC._valid_time_indices(field, time_dim, counts)})
    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)
    n_lat, n_lon = field.sizes[lat_dim], field.sizes[lon_dim]
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    n_buckets = len(fs._axis_starts(
        (stamps.min().isoformat(), stamps.max().isoformat()), cadence,
    ))
    print(f"  open+mask {time.perf_counter() - t0:.1f}s  peak RSS "
          f"{_status_kb('VmHWM') / 1024:.0f} MB", flush=True)
    print(f"  cadence={cadence}  grid={n_lat}x{n_lon} ({n_lat * n_lon:,} cells/interval)"
          f"  buckets={n_buckets}  gate={fs.MAX_FRAME_NATIVE_CELLS:,}", flush=True)

    refusal = fs.frame_gate(field, time_dim=time_dim, cadence=cadence)
    print(f"  frame_gate: {'PASS' if refusal is None else refusal.reason}", flush=True)

    reset_peak_rss()
    before = _status_kb("VmRSS")
    t = time.perf_counter()
    stack = fs.build_frame_stack(
        field,
        time_dim=time_dim,
        cadence=cadence,
        # ``_attach_frames`` passes region_area from the geometry mask; there is
        # no AOI here, and it denominates coverage rather than allocating
        # anything, so its absence cannot move the number under test.
        qa_counts=counts,
        statistics=statistics,
    )
    seconds = time.perf_counter() - t
    peak = _status_kb("VmHWM")

    shipped = int(stack.values.nbytes + stack.period_values.nbytes)
    for plane in stack.planes.values():
        shipped += int(plane.values.nbytes + plane.period_values.nbytes)

    print(f"  [{args.arm}] {seconds:.1f}s   RSS {before / 1024:.0f} -> "
          f"{_status_kb('VmRSS') / 1024:.0f} MB   PEAK {peak / 1024:.0f} MB", flush=True)
    print(f"  frames={len(stack.frames)}  k={stack.coarsen_k}  "
          f"cells/frame={stack.cells_per_frame:,}  tier={stack.tier}  "
          f"planes={sorted(stack.planes)}", flush=True)
    print(f"  shipped arrays {shipped / 1e6:.3f} MB", flush=True)

    result = {
        "bundle": args.bundle, "arm": args.arm, "statistics": list(statistics),
        "grid": [n_lat, n_lon], "native_cells": n_lat * n_lon,
        "buckets": n_buckets, "n_frames": len(stack.frames),
        "coarsen_k": list(stack.coarsen_k), "tier": stack.tier,
        "seconds": round(seconds, 1),
        "peak_rss_mb": round(peak / 1024, 1),
        "rss_before_mb": round(before / 1024, 1),
        "shipped_mb": round(shipped / 1e6, 3),
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    print(f"\n  arm survived: {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
