"""T59 Phase 2, fidelity arm — is 20,000 cells worth 2.5x the bytes over 8,000?

The storage arm says what each budget costs. It cannot say what either buys,
and "settled cell budget" (PRD Phase 2) is a trade, not a size. So this
measures what a frame at each budget still shows of the native field:

  peak retention   max and p98 per frame, against the SAME frame at native
                   resolution. A scrubber exists to find events; a budget that
                   flattens the peak has failed at its one job regardless of
                   how well it compresses.
  mean fidelity    the cos-lat weighted spatial mean. A block mean should carry
                   this almost exactly; the check is that the coarsening is not
                   quietly biased.
  coverage         the finite fraction. Block-meaning a mostly-empty frame
                   fills cells that had no observation in most of their block,
                   which makes a sparse hour look better covered than it is --
                   a D10 disclosure concern, measured here rather than assumed.

Also runs the incumbent ``_downsample_grid`` STRIDE at the same output size, so
D5's "a stride deletes plumes; a block mean preserves their mass" is a measured
claim rather than a quoted one. That is the existing payload path
(``_MAX_GRID_CELLS = 8_000``), so it is what frames are being compared against.

Run on the small regional bundle: its mask is ~7s, and peak retention is a
property of the field's spatial structure, not of the domain's size.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


def coarsen_k(n_lat: int, n_lon: int, target: int) -> int:
    if n_lat * n_lon <= target:
        return 1
    return max(int(np.ceil((n_lat * n_lon / target) ** 0.5)), 1)


def stride_like_production(arr: np.ndarray, target: int) -> np.ndarray:
    """``_downsample_grid``'s thinning, applied per frame.

    Reimplemented rather than imported because the production function takes
    (lats, lons, 2-D arr) and returns the coordinates too; the arithmetic --
    ``scale = (total / target) ** 0.5`` then ``arr[::step_r, ::step_c]`` -- is
    copied exactly so the comparison is against what actually ships.
    """
    n_r, n_c = arr.shape[-2:]
    total = n_r * n_c
    if total <= target:
        return arr
    scale = (total / target) ** 0.5
    step_r = max(int(np.ceil(scale)), 1)
    step_c = max(int(np.ceil(scale)), 1)
    return arr[..., ::step_r, ::step_c]


def stats(vals: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Per-frame statistics, NaN-safe, over a (frame, lat, lon) stack."""
    flat = vals.reshape(vals.shape[0], -1)
    finite = np.isfinite(flat)
    out = {"max": [], "p99": [], "p98": [], "mean": [], "finite_frac": []}
    for i in range(flat.shape[0]):
        row = flat[i][finite[i]]
        out["finite_frac"].append(float(finite[i].mean()))
        if row.size == 0:
            for k in ("max", "p99", "p98", "mean"):
                out[k].append(float("nan"))
            continue
        out["max"].append(float(row.max()))
        out["p99"].append(float(np.percentile(row, 99)))
        out["p98"].append(float(np.percentile(row, 98)))
        if weights is None:
            out["mean"].append(float(row.mean()))
        else:
            w = np.broadcast_to(weights, vals.shape[1:]).reshape(-1)[finite[i]]
            out["mean"].append(float((row * w).sum() / w.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def retention(ref: dict, test: dict, key: str) -> dict:
    """Per-frame ratio test/ref as a DISTRIBUTION, in percent.

    The median alone is not enough to choose between a block mean and a stride.
    A stride is a sampler: when it lands on the peak it reproduces it exactly,
    and when it misses it loses it entirely -- so its median can beat a block
    mean's while its worst frames are far worse. A block mean attenuates every
    peak by a predictable amount and never misses one. Those are different
    risks and only the spread distinguishes them, so p10/p50/p90 and the share
    of badly-degraded frames are all reported.
    """
    a, b = np.asarray(ref[key], dtype=float), np.asarray(test[key], dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0)
    if not ok.any():
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "frac_below_50pct": float("nan"), "n": 0}
    r = b[ok] / a[ok] * 100
    return {
        "p10": round(float(np.percentile(r, 10)), 2),
        "p50": round(float(np.percentile(r, 50)), 2),
        "p90": round(float(np.percentile(r, 90)), 2),
        "frac_below_50pct": round(float((r < 50).mean()), 3),
        "n": int(ok.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--cells", type=int, nargs="+", default=[20_000, 8_000])
    ap.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="crop before measuring. REQUIRED for the full-domain bundles: this "
             "arm needs the native stack in memory to compute a per-frame "
             "percentile, and 32 buckets x 17M cells is 4.4 GB of float64 in a "
             "3.9 GB container -- the exact materialization the Risk 2 probe "
             "refused to pay.",
    )
    ap.add_argument("--json-out", default="/tmp/t59_cell_budget.json")
    args = ap.parse_args()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()

    ds = _open_netcdf_bundle(args.bundle)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    if args.bbox:
        _lat, _lon = find_lat_coord(da), find_lon_coord(da)
        minx, miny, maxx, maxy = args.bbox
        sel = {
            _lat: slice(miny, maxy) if da[_lat][0] < da[_lat][-1] else slice(maxy, miny),
            _lon: slice(minx, maxx) if da[_lon][0] < da[_lon][-1] else slice(maxx, minx),
        }
        da, ds = da.sel(sel), ds.sel(sel)
        print(f"  cropped to {tuple(args.bbox)}: {dict(da.sizes)}")
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
    )
    field = masked.data
    valid = SVC._valid_time_indices(field, time_dim, masked.counts)
    field = field.isel({time_dim: valid})

    lat, lon = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat], field.sizes[lon]
    cadence = SVC._cadence(ds, col_info.get("collection_id"), da.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    labels = np.asarray(
        [str(b) for b in (stamps.to_period("M") if cadence == "monthly"
                          else stamps.floor({"hourly": "h", "daily": "D"}[cadence]))]
    )
    grouper = xr.DataArray(labels, dims=[time_dim],
                           coords={time_dim: field[time_dim]}).rename("bucket")
    grouped = field.groupby(grouper).mean(dim=time_dim, skipna=True)

    print(f"\n  native {n_lat}x{n_lon} = {n_lat * n_lon:,} cells, "
          f"{grouped.sizes['bucket']} buckets")
    native = grouped.compute()
    nat_vals = np.asarray(native.values, dtype=np.float32)
    w_native = np.cos(np.deg2rad(np.asarray(native[lat].values)))[:, None]
    ref = stats(nat_vals, w_native)
    units = da.attrs.get("units", "")
    print(f"  native per-frame max: median {np.nanmedian(ref['max']):.4g} {units}")

    results = {"bundle": args.bundle, "native_cells": int(n_lat * n_lon),
               "n_frames": int(grouped.sizes["bucket"]), "budgets": []}

    for target in args.cells:
        k = coarsen_k(n_lat, n_lon, target)
        block = grouped.coarsen({lat: k, lon: k}, boundary="pad").mean(skipna=True).compute()
        b_vals = np.asarray(block.values, dtype=np.float32)
        w_block = np.cos(np.deg2rad(np.asarray(block[lat].values)))[:, None]
        b_stats = stats(b_vals, w_block)

        s_vals = stride_like_production(nat_vals, target)
        s_stats = stats(s_vals)

        entry = {
            "target_cells": target,
            "k": int(k),
            "block_shape": [int(x) for x in b_vals.shape],
            "block_cells": int(b_vals.shape[1] * b_vals.shape[2]),
            "stride_cells": int(s_vals.shape[1] * s_vals.shape[2]),
            "block": {
                k2: retention(ref, b_stats, k2) for k2 in ("max", "p99", "p98", "mean")
            } | {"median_finite_frac": round(float(np.median(b_stats["finite_frac"])), 4)},
            "stride": {
                k2: retention(ref, s_stats, k2) for k2 in ("max", "p99", "p98", "mean")
            } | {"median_finite_frac": round(float(np.median(s_stats["finite_frac"])), 4)},
            "native_median_finite_frac": round(float(np.median(ref["finite_frac"])), 4),
        }
        results["budgets"].append(entry)

        print(f"\n  --- target {target:,} cells (k={k}) ---")
        for name, cells, ent in (
            ("block mean", entry["block_cells"], entry["block"]),
            ("stride    ", entry["stride_cells"], entry["stride"]),
        ):
            print(f"    {name}  {cells:,} cells"
                  + ("   (the incumbent _downsample_grid)" if "stride" in name else ""))
            for stat in ("max", "p99", "p98", "mean"):
                d = ent[stat]
                print(f"      {stat:<5} retained  p10 {d['p10']:>7.2f}  p50 {d['p50']:>7.2f}  "
                      f"p90 {d['p90']:>7.2f} %   frames <50%: {100 * d['frac_below_50pct']:.0f}%")
        print(f"    finite fraction  native {entry['native_median_finite_frac']:.3f}  "
              f"block {entry['block']['median_finite_frac']:.3f}  "
              f"stride {entry['stride']['median_finite_frac']:.3f}"
              f"   <- block INFLATES apparent coverage (D10)")

    with open(args.json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
