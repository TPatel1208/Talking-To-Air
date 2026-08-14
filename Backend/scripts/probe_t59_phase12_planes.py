"""T59 Phase 12 -- does the PORTED reduction reproduce the gate's numbers?

Phase 11 measured G2/G4/G5 with standalone xarray in
``probe_t59_max_gate.py``, outside product code. Phase 12 moved that logic
into ``build_frame_stack`` as a ``statistics=`` parameter. This script runs the
product code against the same live bundle at the same bbox and checks the
answers land where the gate put them:

    retention of the native per-bucket max : 100.0% at every percentile
    extent overstatement                   : ~24.7x (ceiling k^2 = 25)
    both associativity identities          : exactly 0.0

It is a CONFIRMATION, not a discovery. If a number disagrees with
docs/t59-phase11-max-gate.md, the port has a bug -- the gate's numbers are what
the GO verdict is built on.

    docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_phase12_planes.py \\
        /data/harmony/job_a5c9813780a9300b/result.nc.zip \\
        --bbox -106.6458459 25.83706 -93.5078217 36.5004529'
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.preprocessing.frame_stack import build_frame_stack
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


def open_and_mask(path: str, bbox, col_info: dict):
    """``probe_t59_max_gate.open_and_mask``, unchanged -- the same open, crop
    and mask the gate measured through, so only the reduction differs."""
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
    field = masked.data
    valid = SVC._valid_time_indices(field, time_dim, masked.counts)
    field = field.isel({time_dim: valid})
    print(f"  cropped+masked: {dict(field.sizes)}  time_dim={time_dim}")
    return ds, field, time_dim, masked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--collection", default="TEMPO_NO2")
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    bbox = tuple(args.bbox) if args.bbox else None
    print(f"\n{'=' * 78}\nbundle: {args.bundle}")
    ds, field, time_dim, masked = open_and_mask(args.bundle, bbox, col_info)
    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)

    stack = build_frame_stack(
        field, time_dim=time_dim, cadence=cadence,
        qa_counts=masked.counts, statistics=("mean", "max", "min"),
    )
    print(f"  tier={stack.tier} frames={len(stack.frames)} "
          f"buckets_per_frame={stack.buckets_per_frame} k={stack.coarsen_k} "
          f"cells_per_frame={stack.cells_per_frame}")

    plane = stack.planes["max"]

    # --- G2: retention of the native per-bucket max, block max vs block mean.
    # The native reference, computed here the way the gate did.
    lat, lon = find_lat_coord(field), find_lon_coord(field)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    floor = {"hourly": "h", "daily": "D"}.get(cadence)
    starts = (
        stamps.to_period("M").to_timestamp() if floor is None else stamps.floor(floor)
    )
    grouper = xr.DataArray(
        np.asarray([s.isoformat() for s in starts]),
        dims=[time_dim], coords={time_dim: field[time_dim]},
    ).rename("bucket")
    # Reindexed onto the SHIPPED axis, which spans the whole requested range
    # including the intervals nothing was retrieved for -- 48 stops here over
    # 26 occupied buckets, and comparing them positionally would compare a
    # frame against a different hour's peak.
    native_max = field.groupby(grouper).max(dim=time_dim, skipna=True).reindex(
        bucket=[f.t_start for f in stack.frames],
    ).compute()
    native_peaks = np.asarray([
        float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        for v in np.asarray(native_max.values, dtype="float64")
    ])
    shipped_peaks = np.asarray([
        float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        for v in np.asarray(plane.values, dtype="float64")
    ])
    mean_peaks = np.asarray([
        float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        for v in np.asarray(stack.values, dtype="float64")
    ])
    ok = np.isfinite(native_peaks) & (native_peaks != 0)
    if stack.buckets_per_frame == 1:
        ratio = shipped_peaks[ok] / native_peaks[ok] * 100
        mean_ratio = mean_peaks[ok] / native_peaks[ok] * 100
        print("\n  G2 retention of the native per-bucket max "
              f"(n={int(ok.sum())} frames, gate: 100.0 everywhere):")
        print(f"    block max  p10={np.percentile(ratio, 10):.2f} "
              f"p50={np.percentile(ratio, 50):.2f} p90={np.percentile(ratio, 90):.2f} "
              f"worst={ratio.min():.2f} %")
        print(f"    block mean p10={np.percentile(mean_ratio, 10):.2f} "
              f"p50={np.percentile(mean_ratio, 50):.2f} p90={np.percentile(mean_ratio, 90):.2f} "
              f"worst={mean_ratio.min():.2f} %   (the SHIPPED mean plane, for contrast --"
              " the mean of a bucket, not the gate's block-mean-of-max)")
    else:
        print("\n  G2 skipped: the frames are grouped cadence buckets here, so a "
              "per-BUCKET native max is not what any frame claims to be.")

    # --- G5 identity B: does grouping temporally before or after the spatial
    # block max change the answer? Only exists where group > 1.
    if stack.buckets_per_frame > 1:
        group = stack.buckets_per_frame
        per_bucket = build_frame_stack(
            field, time_dim=time_dim, cadence=cadence, qa_counts=masked.counts,
            statistics=("max", "min"),
            max_frames=len(stack.frames) * group,
        )
        print(f"\n  G5 identity B (group={group}, gate: exactly 0.0):")
        for name, reducer in (("max", np.fmax.reduce), ("min", np.fmin.reduce)):
            after = stack.planes[name].values
            before = np.stack([
                reducer(per_bucket.planes[name].values[start:start + group], axis=0)
                for start in range(0, per_bucket.planes[name].values.shape[0], group)
            ])
            both = np.isfinite(after) & np.isfinite(before)
            worst = float(np.abs(after[both] - before[both]).max()) if both.any() else 0.0
            print(f"    {name}: shapes {after.shape} vs {before.shape}  "
                  f"NaN pattern matches={np.array_equal(np.isnan(after), np.isnan(before))}  "
                  f"exact on {int(both.sum())} finite cells="
                  f"{np.array_equal(after[both], before[both])}  max abs diff={worst}")

    # --- G4: the extent overstatement.
    print(f"\n  G4 extent overstatement (gate: 24.699x pooled, ceiling 25):")
    print(f"    {plane.extent_overstatement}")

    # --- G5: both identities, and the mean's own for contrast.
    print("\n  G5 identities (gate: exactly 0.0 on both):")
    for name in ("max", "min"):
        agreement = stack.planes[name].frame_grid_delta
        print(f"    {name} plane frame_grid_delta headline={agreement['headline']!r} "
              f"max_abs={agreement['max_abs']!r}")
    print(f"    mean plane frame_grid_delta headline={stack.frame_grid_delta['headline']!r} "
          "(non-zero: the block mean and the across-frame mean do not commute)")
    print(f"    mean plane delta={stack.delta if stack.delta is None else stack.delta['headline']!r}")

    # G3 at scale: no +-inf anywhere on a shipped plane, the specific leak an
    # all-NaN block would produce if the reducer fell back to its identity.
    print("\n  G3 +-inf leak check (gate: 0 on both bundles):")
    for name in ("max", "min"):
        for label, arr in (
            ("frames", stack.planes[name].values),
            ("period", stack.planes[name].period_values),
        ):
            print(f"    {name} {label}: {int(np.isinf(arr).sum())} inf cells, "
                  f"{int(np.isnan(arr).sum())} NaN of {arr.size}")

    print(f"\n  value ranges: mean={stack.value_range}")
    print(f"                max ={plane.value_range}")
    print(f"                min ={stack.planes['min'].value_range}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
