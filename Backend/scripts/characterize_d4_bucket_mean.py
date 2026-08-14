"""D4 characterization (PRD T59 Phase 1 gate).

Question
--------
``_cadence_weighted_mean`` weights each timestep ``1 / (granules in its cadence
bucket)``. Summing that over timesteps is algebraically the mean of the
per-bucket means -- but ``.weighted(w).mean(skipna=True)`` renormalizes PER
PIXEL over the timesteps that survived masking. So at a pixel where a bucket
holds 4 granules and 3 were QA-masked, that bucket contributes weight 1/4, not
1. A mean-of-bucket-means gives it weight 1.

Finding #11's stated intent is that each cadence bucket counts equally
regardless of how many granules landed in it. If the delta below is material,
the current implementation does not deliver that intent at masked pixels, and
T59 D4 becomes a correctness fix rather than a viewer prerequisite.

Metric (T59 D16)
----------------
Over cells finite in BOTH fields, with ``w = cos(latitude)``::

    headline  = sum(w * |F - M|) / sum(w * |M|)
    companion = max|F - M| in physical units

Never a mean of per-pixel ratios (geophysical fields go near zero) and never a
signed mean (cancellation would report ~0 for a field off by +-10%).

Runs the REAL pipeline: ``_open_netcdf_bundle`` for the concat/time ordering,
``AggregationService._resolve_and_mask`` for the production NaN pattern.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.preprocessing.aggregation_service import AggregationService, cos_lat_weights
from tta_backend.services.open_handle import _open_netcdf_bundle
from tta_backend.utils.geo_utils import identify_time

SVC = AggregationService()


def bucket_labels(stamps: pd.DatetimeIndex, cadence: str) -> pd.Index:
    """The same bucketing ``_cadence_weighted_mean`` uses."""
    if cadence == "monthly":
        return stamps.to_period("M")
    return stamps.floor({"hourly": "h", "daily": "D"}[cadence])


def mean_of_bucket_means(da: xr.DataArray, time_dim: str, cadence: str) -> xr.DataArray:
    """Group into cadence buckets, mean within each, then mean across buckets.

    Each bucket contributes weight 1 at every pixel where it has ANY finite
    value -- which is the difference from the weighted form.
    """
    stamps = pd.to_datetime(np.asarray(da[time_dim].values))
    labels = bucket_labels(stamps, cadence)
    grouper = xr.DataArray(
        np.asarray([str(b) for b in labels]), dims=[time_dim], coords={time_dim: da[time_dim]},
    )
    per_bucket = da.groupby(grouper.rename("bucket")).mean(dim=time_dim, skipna=True)
    return per_bucket.mean(dim="bucket", skipna=True)


def partial_bucket_count(da: xr.DataArray, time_dim: str, cadence: str) -> xr.DataArray:
    """Per pixel, how many cadence buckets are *partially* covered.

    A partial bucket holds more than one granule and, at this pixel, has at
    least one finite value and at least one NaN. That is the exact and only
    condition under which the weighted form and the mean-of-bucket-means form
    can disagree: everywhere else both give each bucket weight 1.

    Splitting the delta on this is what turns "the two differ" into "the two
    differ *for this reason*" -- without it the mechanism is inferred rather
    than shown, and the QA-on/QA-off comparison already proved inference wrong
    here once.
    """
    stamps = pd.to_datetime(np.asarray(da[time_dim].values))
    labels = bucket_labels(stamps, cadence)
    grouper = xr.DataArray(
        np.asarray([str(b) for b in labels]), dims=[time_dim], coords={time_dim: da[time_dim]},
    ).rename("bucket")
    finite = np.isfinite(da)
    n_finite = finite.groupby(grouper).sum(dim=time_dim)
    n_total = finite.groupby(grouper).count(dim=time_dim)
    partial = (n_total > 1) & (n_finite > 0) & (n_finite < n_total)
    return partial.sum(dim="bucket")


def split_by_partial(f: xr.DataArray, m: xr.DataArray, partial: xr.DataArray) -> None:
    weights = cos_lat_weights(m)
    both = np.isfinite(f) & np.isfinite(m)
    for label, sel in (("pixels with NO partial bucket", partial == 0),
                       ("pixels with >=1 partial bucket", partial > 0)):
        mask = both & sel
        diff = np.abs(f - m).where(mask)
        mag = np.abs(m).where(mask)
        if weights is None:
            num, den = float(diff.sum(skipna=True)), float(mag.sum(skipna=True))
        else:
            num = float((diff * weights).sum(skipna=True))
            den = float((mag * weights).sum(skipna=True))
        n = int(mask.sum())
        pct = 100.0 * num / den if den > 0 else 0.0
        print(f"  {label:<32}: {n:>9,} cells   delta {pct:.4f} %   "
              f"max|F-M| {float(diff.max(skipna=True)) if n else 0:.4g}")


def delta_metrics(f: xr.DataArray, m: xr.DataArray) -> dict:
    weights = cos_lat_weights(m)
    both = np.isfinite(f) & np.isfinite(m)
    diff = np.abs(f - m).where(both)
    mag = np.abs(m).where(both)

    if weights is None:
        num = float(diff.sum(skipna=True))
        den = float(mag.sum(skipna=True))
    else:
        num = float((diff * weights).sum(skipna=True))
        den = float((mag * weights).sum(skipna=True))

    n_both = int(both.sum())
    # "Materially different" per pixel: >0.1% of that pixel's own magnitude,
    # reported as a COUNT so a small headline driven by a few extreme pixels is
    # distinguishable from one spread thinly everywhere.
    rel = (diff / mag.where(mag > 0)).where(both)
    return {
        "headline_pct": 100.0 * num / den if den > 0 else float("nan"),
        "max_abs_diff": float(diff.max(skipna=True)) if n_both else float("nan"),
        "cells_compared": n_both,
        "cells_finite_only_in_weighted": int((np.isfinite(m) & ~np.isfinite(f)).sum()),
        "cells_finite_only_in_bucketmean": int((np.isfinite(f) & ~np.isfinite(m)).sum()),
        "pixels_over_0.1pct": int((rel > 0.001).sum()),
        "pixels_over_1pct": int((rel > 0.01).sum()),
        "pixels_over_5pct": int((rel > 0.05).sum()),
        "max_rel_pct": 100.0 * float(rel.max(skipna=True)) if n_both else float("nan"),
    }


def registry_col_info(key: str) -> dict:
    """The collection's pinned facts, as the tool layer supplies them.

    A Harmony bundle member carries no ``short_name`` global attr, so neither
    the cadence nor the QA-flag variable resolves from the file alone -- and
    with ``cadence == "unknown"`` ``_cadence_weighted_mean`` silently degrades
    to a plain unweighted mean, which would make this characterization compare
    two things that are trivially identical. The real plot path always has
    ``col_info`` (``col_info_for_variable``), so supplying it here is what makes
    the probe production-faithful rather than a lucky no-op.
    """
    from tta_backend.datasets.registry import load_registry

    return load_registry()[key].model_dump()


def run(path: str, *, apply_qa: bool, bbox: tuple | None, col_info: dict) -> None:
    print(f"\n{'=' * 72}\nbundle: {path}   QA masking: {'ON' if apply_qa else 'OFF'}")
    ds = _open_netcdf_bundle(path)
    da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    print(f"variable: {da.name}   dims: {dict(da.sizes)}   time_dim: {time_dim}")

    if bbox is not None:
        from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord

        lat, lon = find_lat_coord(da), find_lon_coord(da)
        minx, miny, maxx, maxy = bbox
        sel = {
            lat: slice(miny, maxy) if da[lat][0] < da[lat][-1] else slice(maxy, miny),
            lon: slice(minx, maxx) if da[lon][0] < da[lon][-1] else slice(maxx, minx),
        }
        da = da.sel(sel)
        ds = ds.sel(sel)
        print(f"cropped to {bbox}: {dict(da.sizes)}")

    # Fill/valid-range only when QA is off, so the delta attributable to QA
    # masking specifically is isolable from the delta fill masking alone causes.
    #
    # Nulling ``quality_flag_var`` is NOT enough to suppress QA: with no pinned
    # name, ``_resolve_qa_flag_var`` falls through to its Tier-2/3 tiers and
    # re-finds ``main_data_quality_flag`` by its flag_values/flag_meanings
    # attrs. The flag has to leave the Dataset entirely.
    effective = dict(col_info)
    qa_source = ds
    if not apply_qa:
        qf = col_info.get("quality_flag_var")
        effective["quality_flag_var"] = None
        if qf and qf in ds.data_vars:
            qa_source = ds.drop_vars([qf])

    masked = SVC._resolve_and_mask(
        da,
        col_info=effective,
        collection_id=col_info.get("collection_id"),
        source_ds=qa_source,
    )
    field, counts = masked.data, masked.counts
    prov = masked.provenance
    print(f"masking: qa_status={prov.get('qa_status')} "
          f"flag={prov.get('quality_flag_var')} rate={prov.get('qa_pass_rate')}")

    valid = SVC._valid_time_indices(field, time_dim, counts)
    print(f"timesteps: {field.sizes[time_dim]} total, {len(valid)} valid after masking")
    field = field.isel({time_dim: valid})

    cadence = SVC._cadence(ds, col_info.get("collection_id"), da.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    labels = bucket_labels(stamps, cadence)
    sizes = pd.Series(1, index=labels).groupby(level=0).size()
    print(f"cadence: {cadence}   buckets: {len(sizes)}   "
          f"granules/bucket min={sizes.min()} max={sizes.max()} mean={sizes.mean():.2f}")
    print(f"buckets with >1 granule: {int((sizes > 1).sum())} of {len(sizes)}")

    nan_frac = float(np.isnan(field).mean())
    print(f"NaN fraction after masking: {100 * nan_frac:.1f}%")

    print("computing weighted mean (current behaviour)...")
    m = SVC._cadence_weighted_mean(field, time_dim, cadence).compute()
    print("computing mean-of-bucket-means (D4 proposal)...")
    f = mean_of_bucket_means(field, time_dim, cadence).compute()

    metrics = delta_metrics(f, m)
    units = da.attrs.get("units", "")
    print(f"\n--- D16 delta ({'QA ON' if apply_qa else 'QA OFF'}) ---")
    print(f"  headline  sum(w|F-M|)/sum(w|M|) : {metrics['headline_pct']:.4f} %")
    print(f"  max |F-M|                       : {metrics['max_abs_diff']:.6g} {units}")
    print(f"  max per-pixel relative          : {metrics['max_rel_pct']:.3f} %")
    print(f"  cells compared                  : {metrics['cells_compared']:,}")
    for key in ("pixels_over_0.1pct", "pixels_over_1pct", "pixels_over_5pct"):
        n = metrics[key]
        pct = 100.0 * n / metrics["cells_compared"] if metrics["cells_compared"] else 0.0
        print(f"  {key:<32}: {n:,} ({pct:.2f}%)")
    print(f"  finite only in weighted         : {metrics['cells_finite_only_in_weighted']:,}")
    print(f"  finite only in bucket-mean      : {metrics['cells_finite_only_in_bucketmean']:,}")

    print("\n--- mechanism: split by partial-bucket coverage ---")
    partial = partial_bucket_count(field, time_dim, cadence).compute()
    split_by_partial(f, m, partial)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--qa-off", action="store_true", help="also run with QA masking suppressed")
    ap.add_argument("--collection", default="TEMPO_NO2", help="registry key for col_info")
    args = ap.parse_args()

    col_info = registry_col_info(args.collection)
    bbox = tuple(args.bbox) if args.bbox else None
    run(args.bundle, apply_qa=True, bbox=bbox, col_info=col_info)
    if args.qa_off:
        run(args.bundle, apply_qa=False, bbox=bbox, col_info=col_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
