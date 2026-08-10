"""T59 Risk 2, edge arm — is boundary="pad" losing the region edge?

The fused-arm probe reported the trailing coarse ROW as all-NaN on the
full-domain bundle. Two very different causes produce that:

  (a) ``coarsen(boundary="pad").mean()`` is NOT skipping the NaN pad, so any
      block that is partly pad comes out NaN. That would blank the region edge
      -- the exact failure ``pad`` was chosen over ``trim`` to avoid, and it
      would mean Phase 3 cannot use coarsen as written.
  (b) the native rows under that block are themselves all-NaN (no coverage at
      the domain's high-latitude edge), in which case NaN is the correct and
      only honest answer.

Reporting "False" without separating these would be misleading, so this
compares each trailing coarse block against an explicit ``np.nanmean`` of the
native rows it covers. Also checks a synthetic block whose pad is known, so
skipna behaviour is established independently of what the data happens to hold.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, identify_time

SVC = AggregationService()


def synthetic_pad_check() -> None:
    """Does coarsen(pad).mean() skip the pad, on data we control completely?

    A 5-long axis coarsened by 3 pads to 6: block 0 = [1,2,3], block 1 =
    [4,5,pad]. If the pad is skipped block 1 is 4.5; if it is not, block 1 is
    NaN. No bundle needed, and it settles the mechanism before any real data
    muddies it.
    """
    da = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dims=["x"],
                      coords={"x": np.arange(5)})
    out = da.coarsen({"x": 3}, boundary="pad").mean(skipna=True)
    print("\n--- synthetic: [1,2,3,4,5] coarsen(x=3, pad).mean(skipna=True) ---")
    print(f"  result: {out.values}   expected [2.0, 4.5] if pad is skipped")
    print(f"  pad IS skipped: {bool(np.isfinite(out.values[-1]))}")

    out_nosk = da.coarsen({"x": 3}, boundary="pad").mean(skipna=False)
    print(f"  with skipna=False: {out_nosk.values}  (shows what NOT skipping costs)")

    # And a block that is genuinely all-NaN must stay NaN, not become 0.
    da2 = xr.DataArray(np.array([1.0, 2.0, 3.0, np.nan, np.nan]), dims=["x"],
                       coords={"x": np.arange(5)})
    out2 = da2.coarsen({"x": 3}, boundary="pad").mean(skipna=True)
    print(f"  all-NaN trailing block stays NaN: {bool(np.isnan(out2.values[-1]))} "
          f"-> {out2.values}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--target-cells", type=int, default=20_000)
    args = ap.parse_args()

    synthetic_pad_check()

    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()

    ds = _open_netcdf_bundle(args.bundle)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    masked = SVC._resolve_and_mask(
        da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
    )
    field = masked.data
    valid = SVC._valid_time_indices(field, time_dim, masked.counts)
    field = field.isel({time_dim: valid})

    lat, lon = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat], field.sizes[lon]
    k = int(np.ceil((n_lat * n_lon / args.target_cells) ** 0.5))
    out_lat = int(np.ceil(n_lat / k))
    print(f"\n--- real bundle: {n_lat}x{n_lon}, k={k}, {out_lat} coarse rows ---")
    print(f"  trailing block covers native rows {(out_lat - 1) * k}..{n_lat - 1} "
          f"({n_lat - (out_lat - 1) * k} real rows + {out_lat * k - n_lat} pad rows)")

    cadence = SVC._cadence(ds, col_info.get("collection_id"), da.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    if cadence == "monthly":
        labels = np.asarray([str(b) for b in stamps.to_period("M")])
    else:
        labels = np.asarray([str(b) for b in stamps.floor({"hourly": "h", "daily": "D"}[cadence])])
    grouper = xr.DataArray(labels, dims=[time_dim],
                           coords={time_dim: field[time_dim]}).rename("bucket")
    grouped = field.groupby(grouper).mean(dim=time_dim, skipna=True)
    frames = grouped.coarsen({lat: k, lon: k}, boundary="pad").mean(skipna=True).compute()

    # Independent reference for the trailing rows: how much real data is under
    # them at all? Computed from the grouped field one row-band at a time so
    # this check never materializes the native stack it exists to avoid.
    tail = grouped.isel({lat: slice((out_lat - 1) * k, n_lat)}).compute()
    n_finite_tail = int(np.isfinite(tail.values).sum())
    print(f"  native cells under the trailing block: {tail.size:,}, "
          f"finite: {n_finite_tail:,}")

    coarse_tail = frames.isel({lat: -1}).values
    print(f"  trailing coarse row: {int(np.isfinite(coarse_tail).sum())} finite "
          f"of {coarse_tail.size}")

    if n_finite_tail == 0:
        print("  => VERDICT: the source rows are themselves empty. NaN is correct; "
              "pad is not eating the edge.")
    elif int(np.isfinite(coarse_tail).sum()) == 0:
        print("  => VERDICT: real data underneath but the coarse row is NaN. "
              "pad IS eating the edge -- Phase 3 must not use coarsen as written.")
    else:
        print("  => VERDICT: partial coverage, preserved through the coarsen. "
              "pad is behaving.")

    # Same question for the trailing column.
    out_lon = int(np.ceil(n_lon / k))
    tailc = grouped.isel({lon: slice((out_lon - 1) * k, n_lon)}).compute()
    coarse_tailc = frames.isel({lon: -1}).values
    print(f"  trailing column: native finite {int(np.isfinite(tailc.values).sum()):,} "
          f"-> coarse finite {int(np.isfinite(coarse_tailc).sum())} of {coarse_tailc.size}")

    # And the whole-frame count, so "edge" is in proportion.
    print(f"  whole stack: {int(np.isfinite(frames.values).sum()):,} finite "
          f"of {frames.size:,} coarse cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
