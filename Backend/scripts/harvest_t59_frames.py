"""T59 Phase 2, harvest arm — produce REAL coarsened TEMPO frame stacks.

The storage benchmark needs frames that compress the way production frames
will, and gzip ratio on a geophysical field is dominated by two things a
synthetic array gets wrong: the NaN fraction (TEMPO swaths leave most of the
domain empty at any one hour) and the smoothness of the surviving values. So
the stacks are harvested from real materialized bundles through the production
open + mask path, then block-mean coarsened exactly the way Phase 3 will.

Emits, per bundle, a float32 ``.npy`` at each requested cell budget plus a
sidecar JSON of the facts the benchmark needs to interpret it (NaN fraction,
native grid, coarsen factor, bucket count).

Both budgets are computed in ONE ``dask.compute`` so they share the grouped
subgraph -- the expensive part is the mask and the groupby, and paying it twice
would double a 5-minute harvest for no extra information.

float32 is the cast under test: the whole pipeline is float64 (verified on
job_d175709729a518f2), so the frame blob's dtype is a deliberate narrowing and
the benchmark should measure what it actually costs to store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import dask
import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


def coarsen_k(n_lat: int, n_lon: int, target_cells: int) -> int:
    if n_lat * n_lon <= target_cells:
        return 1
    return max(int(np.ceil((n_lat * n_lon / target_cells) ** 0.5)), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out-dir", default="/tmp/t59_frames")
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--cells", type=int, nargs="+", default=[20_000, 8_000])
    ap.add_argument(
        "--per-granule", action="store_true",
        help="coarsen every timestep instead of every cadence bucket. Bucket frames "
             "are what the viewer stores, so they are the default -- but three "
             "bundles only yield ~96 of them, and the N=100 row must be built from "
             "real distinct frames rather than by repeating one (a repeat gzips to "
             "nothing and would flatter the whole table). Per-granule frames are "
             "equally real TEMPO fields and there are 107 of them at one grid.",
    )
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(args.bundle))

    t0 = time.perf_counter()
    ds = _open_netcdf_bundle(args.bundle)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
    )
    field = masked.data
    valid = SVC._valid_time_indices(field, time_dim, masked.counts)
    field = field.isel({time_dim: valid})
    t_mask = time.perf_counter() - t0

    lat, lon = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat], field.sizes[lon]

    cadence = SVC._cadence(ds, col_info.get("collection_id"), da.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    if cadence == "monthly":
        labels = np.asarray([str(b) for b in stamps.to_period("M")])
    else:
        labels = np.asarray(
            [str(b) for b in stamps.floor({"hourly": "h", "daily": "D"}[cadence])]
        )
    grouper = xr.DataArray(labels, dims=[time_dim],
                           coords={time_dim: field[time_dim]}).rename("bucket")
    if args.per_granule:
        grouped = field
        frame_dim = time_dim
    else:
        grouped = field.groupby(grouper).mean(dim=time_dim, skipna=True)
        frame_dim = "bucket"

    graphs, ks = [], []
    for target in args.cells:
        k = coarsen_k(n_lat, n_lon, target)
        ks.append(k)
        graphs.append(grouped.coarsen({lat: k, lon: k}, boundary="pad").mean(skipna=True))

    t1 = time.perf_counter()
    computed = dask.compute(*graphs)
    t_compute = time.perf_counter() - t1

    infix = "gran" if args.per_granule else ""
    meta = {
        "bundle": args.bundle,
        "tag": tag,
        "frame_kind": "granule" if args.per_granule else "cadence_bucket",
        "variable": str(da.name),
        "units": da.attrs.get("units", ""),
        "cadence": cadence,
        "n_granules": int(len(stamps)),
        "n_frames": int(grouped.sizes[frame_dim]),
        "native_grid": [int(n_lat), int(n_lon)],
        "mask_seconds": round(t_mask, 2),
        "compute_seconds": round(t_compute, 2),
        "budgets": [],
    }

    for target, k, arr in zip(args.cells, ks, computed):
        vals = np.asarray(arr.values, dtype=np.float32)
        path = os.path.join(args.out_dir, f"{tag}_{infix}{target}.npy")
        np.save(path, vals)
        finite = int(np.isfinite(vals).sum())
        entry = {
            "target_cells": target,
            "k": int(k),
            "shape": [int(x) for x in vals.shape],
            "cells_per_frame": int(vals.shape[1] * vals.shape[2]),
            "nan_fraction": round(1.0 - finite / vals.size, 4),
            "finite_min": float(np.nanmin(vals)) if finite else None,
            "finite_max": float(np.nanmax(vals)) if finite else None,
            "path": path,
        }
        meta["budgets"].append(entry)
        print(f"  {tag} target={target:>6,} k={k:<3} shape={vals.shape} "
              f"cells/frame={entry['cells_per_frame']:,} "
              f"NaN={100 * entry['nan_fraction']:.1f}%  -> {path}")

    with open(os.path.join(args.out_dir, f"{tag}_{infix}meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  {tag}: {meta['n_granules']} granules -> {meta['n_frames']} "
          f"{meta['frame_kind']} frames, mask {t_mask:.1f}s compute {t_compute:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
