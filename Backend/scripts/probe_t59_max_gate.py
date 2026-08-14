"""T59 Phase 11 -- the max gate: can a peak survive the rendering grid?

D6a proposes a temporal-max-per-pixel plane (block max spatially, not block
mean) as a follow-on statistic beside the shipped mean. This is the gate that
measures the four things that proposal rests on, per
docs/t59-phase11-prompt.md. It writes no product code -- ``frame_stack.py`` is
untouched -- and composes the candidate reductions directly with xarray/dask,
the same way ``probe_t59_lazy_coarsen.py`` (Phase 2's gate) and
``bench_t59_cell_budget.py`` (D5a's benchmark) did before ``build_frame_stack``
existed to import from.

G1 -- does a 3rd and 4th reduction (max, min) cost more passes over the bundle
      than today's 1-statistic (mean) fused graph? Answered two ways: an exact
      read count on synthetic delayed-load data (the instrument
      ``test_the_shipped_array_delta_costs_no_extra_pass`` uses), and real-
      bundle wall-clock/RSS between isolated single-process arms.
G2 -- on the real bundle at its own k, what does block max / block mean /
      stride retain of the native per-bucket max?
G3 -- does ``coarsen(boundary="pad").max(skipna=True)``/``.min(...)`` skip the
      pad and hold an all-NaN block at NaN, never +-inf?
G4 -- the extent-overstatement metric: a block-max pixel paints its single
      peak cell's value across all k^2 cells it covers. Measured as a ratio of
      weighted (cos-lat) AREAS -- the area actually at each block's own max,
      versus the area the rendered block covers -- pooled over every finite
      block on the chart, never a mean of per-block ratios.
G5 -- do the two "max is associative" identities hold bit-exact on real data:
      block-max(period max) == max-over-buckets(block-max(per-bucket max)),
      and (for the coarsened tier) grouping temporally before or after the
      spatial block max gives the same answer.

Reproduce inside the container (bundles live at /data/harmony, host has no
netCDF stack)::

    docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_max_gate.py \\
        /data/harmony/job_a5c9813780a9300b/result.nc.zip \\
        --bbox -106.6458459 25.83706 -93.5078217 36.5004529 \\
        --gate all --json-out /tmp/p11_cadence.json'

G1's real-bundle arm must run as two separate processes (peak RSS is not
comparable across arms sharing a heap, the same caveat
``probe_t59_lazy_coarsen.py`` states) -- use ``--gate g1-real --arm mean_only``
and ``--gate g1-real --arm mean_max_min`` separately.
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

from tta_backend.preprocessing.aggregation_service import AggregationService, cos_lat_weights
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
    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError as exc:  # pragma: no cover
        print(f"  ! could not reset VmHWM ({exc}); peaks are since process start")


class Stage:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        reset_peak_rss()
        self.rss_before = _status_kb("VmRSS")
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.rss_after = _status_kb("VmRSS")
        self.rss_peak = _status_kb("VmHWM")
        print(
            f"  [{self.label}] {self.seconds:.2f}s   "
            f"RSS {self.rss_before / 1024:.0f} -> {self.rss_after / 1024:.0f} MB   "
            f"peak RSS {self.rss_peak / 1024:.0f} MB"
        )
        return False


def mb(nbytes: float) -> str:
    return f"{nbytes / 1e6:,.1f} MB"


# ------------------------------------------------------------------- pipeline


def coarsen_factors(n_lat: int, n_lon: int, target_cells: int) -> tuple[int, int]:
    total = n_lat * n_lon
    if total <= target_cells:
        return 1, 1
    k = max(1, int(np.ceil((total / target_cells) ** 0.5)))
    return k, k


def stride_like_production(arr: np.ndarray, target: int) -> np.ndarray:
    """``_downsample_grid``'s thinning, reimplemented per ``bench_t59_cell_budget.py``."""
    n_r, n_c = arr.shape[-2:]
    total = n_r * n_c
    if total <= target:
        return arr
    scale = (total / target) ** 0.5
    step_r = max(int(np.ceil(scale)), 1)
    step_c = max(int(np.ceil(scale)), 1)
    return arr[..., ::step_r, ::step_c]


def open_and_mask(path: str, bbox, col_info: dict):
    """The production open + mask path, exactly as characterize_d4 established."""
    with Stage("open bundle"):
        ds = _open_netcdf_bundle(path)
        da = SVC.to_dataarray(ds)
    time_dim = identify_time(da)
    print(f"  variable={da.name} dims={dict(da.sizes)} time_dim={time_dim}")

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
            da, col_info=col_info, collection_id=col_info.get("collection_id"), source_ds=ds,
        )
    field = masked.data
    valid = SVC._valid_time_indices(field, time_dim, masked.counts)
    print(f"  timesteps: {field.sizes[time_dim]} total, {len(valid)} valid after masking")
    field = field.isel({time_dim: valid})
    return ds, field, time_dim


def bucket_labels(stamps: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    if cadence == "monthly":
        buckets = stamps.to_period("M")
    else:
        buckets = stamps.floor({"hourly": "h", "daily": "D"}[cadence])
    return np.asarray([str(b) for b in buckets])


def block_reduce(field: xr.DataArray, lat_dim: str, lon_dim: str, k_lat: int, k_lon: int, how: str):
    if (k_lat, k_lon) == (1, 1):
        return field
    c = field.coarsen({lat_dim: k_lat, lon_dim: k_lon}, boundary="pad")
    return getattr(c, how)(skipna=True)


def _bare_grid(da: xr.DataArray, lat_dim: str, lon_dim: str) -> xr.DataArray:
    """Strip the coarsened lat/lon coordinate and rename to grid-neutral dims.

    ``coarsen(...).max()`` reduces the COORDINATE array (the lat/lon values
    themselves) with the same reducer as the data, so a block-max's coarse
    latitude coordinate differs numerically from a block-mean's or a native
    array's. Packing differently-reduced arrays into one ``xr.Dataset`` under
    the shared name ``latitude``/``longitude`` makes xarray align them on
    those coordinate VALUES -- an outer join that reindexes every variable
    onto the union of all three coordinate sets, silently inflating every
    array to a mostly-NaN grid nobody asked for. Renaming to dims with no
    coordinate index at all sidesteps alignment entirely: same dim NAME
    across statistics, no index to disagree about, plain positional stacking.
    """
    renamed = da.rename({lat_dim: "grid_lat", lon_dim: "grid_lon"})
    return renamed.drop_vars(["grid_lat", "grid_lon"], errors="ignore")


def _common(args):
    from tta_backend.datasets.registry import load_registry

    col_info = load_registry()[args.collection].model_dump()
    bbox = tuple(args.bbox) if args.bbox else None
    ds, field, time_dim = open_and_mask(args.bundle, bbox, col_info)
    cadence = SVC._cadence(ds, col_info.get("collection_id"), field.name, col_info)
    stamps = pd.to_datetime(np.asarray(field[time_dim].values))
    labels = bucket_labels(stamps, cadence)
    grouper = xr.DataArray(labels, dims=[time_dim], coords={time_dim: field[time_dim]}).rename("bucket")
    lat, lon = find_lat_coord(field), find_lon_coord(field)
    n_lat, n_lon = field.sizes[lat], field.sizes[lon]
    k_lat, k_lon = coarsen_factors(n_lat, n_lon, args.target_cells)
    n_buckets = len(pd.unique(labels))
    print(f"  cadence={cadence}  buckets={n_buckets}  native grid={n_lat}x{n_lon} "
          f"({n_lat * n_lon:,} cells)  k=({k_lat},{k_lon})")
    return dict(
        ds=ds, field=field, time_dim=time_dim, cadence=cadence, grouper=grouper,
        lat=lat, lon=lon, n_lat=n_lat, n_lon=n_lon, k_lat=k_lat, k_lon=k_lon,
        n_buckets=n_buckets,
    )


# ---------------------------------------------------------------------- G1


def g1_synthetic() -> dict:
    """Exact read count: 1-stat vs 3-stat fused graph, synthetic delayed loads.

    The instrument ``test_the_shipped_array_delta_costs_no_extra_pass`` uses.
    A real bundle cannot be instrumented at the level of "reads" this exactly
    -- dask task counts on a real netCDF backend include chunk-cache and
    coordinate reads that vary by chunking, so this settles the STRUCTURAL
    question (does xarray/dask fuse three groupby-then-coarsen reductions off
    one read of each source chunk) and ``g1_real`` settles the magnitude one.
    """
    import dask
    import dask.array as dask_array

    def make_field(n=6):
        loads: list[int] = []

        def chunk(i):
            block = np.full((1, 4, 4), float(i))

            def _load():
                loads.append(i)
                return block

            return dask_array.from_delayed(dask.delayed(_load)(), shape=block.shape, dtype="float64")

        science = dask_array.concatenate([chunk(i) for i in range(n)], axis=0)
        da = xr.DataArray(
            science, dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{(d // 2) + 1:02d}T{(d % 2) * 12:02d}:00" for d in range(n)],
                    dtype="datetime64[ns]",
                ),
                "lat": np.arange(4.0), "lon": np.arange(4.0),
            },
            name="no2",
        )
        return da, loads

    def run(stats: list[str]) -> list[int]:
        da, loads = make_field()
        grouper = xr.DataArray(
            [str(pd.Timestamp(t).floor("D")) for t in da["time"].values],
            dims=["time"], coords={"time": da["time"]},
        ).rename("bucket")
        pieces = {}
        for name in stats:
            grouped = getattr(da.groupby(grouper), name)(dim="time", skipna=True)
            coarse = grouped.coarsen({"lat": 2, "lon": 2}, boundary="pad")
            pieces[name] = _bare_grid(getattr(coarse, name)(skipna=True), "lat", "lon")
        xr.Dataset(pieces).compute()
        return sorted(loads)

    one = run(["mean"])
    three = run(["mean", "max", "min"])
    print(f"  synthetic 6-granule bundle (3 daily buckets), 1-stat (mean) reads : {one}")
    print(f"  synthetic 6-granule bundle (3 daily buckets), 3-stat (mean/max/min) reads: {three}")
    result = {
        "one_stat_reads": one, "one_stat_count": len(one),
        "three_stat_reads": three, "three_stat_count": len(three),
        "extra_reads": len(three) - len(one),
        "verdict": "PASS: no extra pass" if len(three) == len(one) else "FAIL: extra pass(es)",
    }
    print(f"  => {result['verdict']} (expected 6 reads either way; got {len(one)} vs {len(three)})")
    return result


def g1_real(args, common: dict) -> dict:
    field, time_dim, grouper = common["field"], common["time_dim"], common["grouper"]
    lat, lon, k_lat, k_lon = common["lat"], common["lon"], common["k_lat"], common["k_lon"]

    if args.arm == "mean_only":
        grouped = field.groupby(grouper).mean(dim=time_dim, skipna=True)
        frames = block_reduce(grouped, lat, lon, k_lat, k_lon, "mean")
        with Stage("compute [mean_only]") as st:
            out = frames.compute()
        entry = {
            "arm": "mean_only", "seconds": round(st.seconds, 3),
            "peak_rss_mb": round(st.rss_peak / 1024, 1),
            "peak_rss_delta_mb": round((st.rss_peak - st.rss_before) / 1024, 1),
            "out_nbytes": int(out.nbytes),
        }
    else:
        g_mean = field.groupby(grouper).mean(dim=time_dim, skipna=True)
        g_max = field.groupby(grouper).max(dim=time_dim, skipna=True)
        g_min = field.groupby(grouper).min(dim=time_dim, skipna=True)
        pieces = {
            "mean": _bare_grid(block_reduce(g_mean, lat, lon, k_lat, k_lon, "mean"), lat, lon),
            "max": _bare_grid(block_reduce(g_max, lat, lon, k_lat, k_lon, "max"), lat, lon),
            "min": _bare_grid(block_reduce(g_min, lat, lon, k_lat, k_lon, "min"), lat, lon),
        }
        with Stage("compute [mean_max_min]") as st:
            out = xr.Dataset(pieces).compute()
        entry = {
            "arm": "mean_max_min", "seconds": round(st.seconds, 3),
            "peak_rss_mb": round(st.rss_peak / 1024, 1),
            "peak_rss_delta_mb": round((st.rss_peak - st.rss_before) / 1024, 1),
            "out_nbytes": int(sum(out[v].nbytes for v in out.data_vars)),
        }
    print(f"  arm={args.arm}: {entry}")
    return entry


# ------------------------------------------------------------- G2, G3, G4, G5


def _native_stats_dataset(common: dict) -> xr.Dataset:
    """The ONE fused compute every one of G2-G5 reads from -- materializing
    mean/max/min at NATIVE resolution beside their block reductions, so G4's
    "costs no extra pass" requirement is structural rather than promised."""
    field, time_dim, grouper = common["field"], common["time_dim"], common["grouper"]
    lat, lon, k_lat, k_lon = common["lat"], common["lon"], common["k_lat"], common["k_lon"]

    g_mean = field.groupby(grouper).mean(dim=time_dim, skipna=True)
    g_max = field.groupby(grouper).max(dim=time_dim, skipna=True)
    g_min = field.groupby(grouper).min(dim=time_dim, skipna=True)

    # Each block-reduced piece below is packed through ``_bare_grid``: coarsen
    # reduces the lat/lon COORDINATE with the same reducer as the data, so
    # block-mean-of-max's coarse latitude differs numerically from block-max-
    # of-max's, and packing them under the shared name "latitude" would make
    # xarray outer-join them onto the union -- see ``_bare_grid``'s docstring.
    ds = xr.Dataset({
        "native_mean": g_mean, "native_max": g_max, "native_min": g_min,
        "block_mean_of_mean": _bare_grid(
            block_reduce(g_mean, lat, lon, k_lat, k_lon, "mean"), lat, lon),
        "block_max_of_max": _bare_grid(
            block_reduce(g_max, lat, lon, k_lat, k_lon, "max"), lat, lon),
        "block_mean_of_max": _bare_grid(
            block_reduce(g_max, lat, lon, k_lat, k_lon, "mean"), lat, lon),
        "block_min_of_min": _bare_grid(
            block_reduce(g_min, lat, lon, k_lat, k_lon, "min"), lat, lon),
    })
    with Stage("compute [native + block, mean/max/min]") as st:
        out = ds.compute()
    print(f"  materialized {mb(sum(out[v].nbytes for v in out.data_vars))} total")
    for v in out.data_vars:
        print(f"    {v}: shape={out[v].shape} dims={out[v].dims}")
    return out


def _stats_of(vals: np.ndarray) -> dict:
    flat = vals.reshape(vals.shape[0], -1)
    finite = np.isfinite(flat)
    out_max = []
    for i in range(flat.shape[0]):
        row = flat[i][finite[i]]
        out_max.append(float(row.max()) if row.size else float("nan"))
    return np.asarray(out_max)


def _retention(ref: np.ndarray, test: np.ndarray) -> dict:
    n = min(len(ref), len(test))
    a, b = ref[:n], test[:n]
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0)
    if not ok.any():
        return {"p10": None, "p50": None, "p90": None, "worst": None, "n": 0}
    r = b[ok] / a[ok] * 100
    return {
        "p10": round(float(np.percentile(r, 10)), 2),
        "p50": round(float(np.percentile(r, 50)), 2),
        "p90": round(float(np.percentile(r, 90)), 2),
        "worst": round(float(r.min()), 2),
        "n": int(ok.sum()),
    }


def g2(computed: xr.Dataset, common: dict, target_cells: int) -> dict:
    """What block max / block mean / stride retain of the native per-bucket max."""
    native_max = np.asarray(computed["native_max"].values, dtype="float64")
    block_max = np.asarray(computed["block_max_of_max"].values, dtype="float64")
    block_mean = np.asarray(computed["block_mean_of_max"].values, dtype="float64")
    stride = stride_like_production(native_max, target_cells)

    ref = _stats_of(native_max)
    result = {
        "k": [common["k_lat"], common["k_lon"]],
        "block_max_cells": int(block_max.shape[-2] * block_max.shape[-1]),
        "block_max": _retention(ref, _stats_of(block_max)),
        "block_mean": _retention(ref, _stats_of(block_mean)),
        "stride": _retention(ref, _stats_of(stride)),
    }
    print(f"  G2 retention of native per-bucket max (n={result['block_max']['n']} frames):")
    for name in ("block_max", "block_mean", "stride"):
        d = result[name]
        print(f"    {name:<10} p10={d['p10']} p50={d['p50']} p90={d['p90']} worst={d['worst']} %")

    # The sparsest bucket, named the way Phase 10 named its 0.38%-covered
    # frame -- if block max is indistinguishable from block mean anywhere,
    # it is here, on the frame with almost nothing to reduce.
    finite_per_bucket = np.isfinite(native_max).reshape(native_max.shape[0], -1).sum(axis=1)
    n_native_cells = native_max.shape[1] * native_max.shape[2]
    sparsest = int(np.argmin(np.where(finite_per_bucket > 0, finite_per_bucket, np.iinfo(np.int64).max)))
    bm_ratio = _stats_of(block_max)[sparsest] / ref[sparsest] * 100 if np.isfinite(ref[sparsest]) else None
    bmean_ratio = _stats_of(block_mean)[sparsest] / ref[sparsest] * 100 if np.isfinite(ref[sparsest]) else None
    print(f"  sparsest bucket: index {sparsest}, {int(finite_per_bucket[sparsest])} of "
          f"{n_native_cells:,} native cells finite "
          f"({100 * finite_per_bucket[sparsest] / n_native_cells:.2f}%) -- "
          f"block max retains {bm_ratio:.1f}%, block mean retains {bmean_ratio:.1f}%"
          if bm_ratio is not None else "  sparsest bucket: none finite")
    result["sparsest_bucket"] = {
        "index": sparsest,
        "finite_native_cells": int(finite_per_bucket[sparsest]),
        "of_native_cells": int(n_native_cells),
        "finite_pct": round(100 * float(finite_per_bucket[sparsest]) / n_native_cells, 4),
        "block_max_retention_pct": round(bm_ratio, 2) if bm_ratio is not None else None,
        "block_mean_retention_pct": round(bmean_ratio, 2) if bmean_ratio is not None else None,
    }
    return result


def g3(computed: xr.Dataset, common: dict) -> dict:
    """coarsen(pad).max()/.min(): pad skipped, all-NaN block stays NaN, never +-inf."""
    print("\n  --- synthetic control ---")
    da = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dims=["x"], coords={"x": np.arange(5)})
    out_max = da.coarsen({"x": 3}, boundary="pad").max(skipna=True)
    out_min = da.coarsen({"x": 3}, boundary="pad").min(skipna=True)
    print(f"  [1,2,3,4,5] coarsen(x=3,pad).max(skipna=True) = {out_max.values}  expect [3.0, 5.0]")
    print(f"  [1,2,3,4,5] coarsen(x=3,pad).min(skipna=True) = {out_min.values}  expect [1.0, 4.0]")
    max_pad_ok = bool(np.isfinite(out_max.values[-1]) and out_max.values[-1] == 5.0)
    min_pad_ok = bool(np.isfinite(out_min.values[-1]) and out_min.values[-1] == 4.0)

    da_allnan = xr.DataArray(
        np.array([1.0, 2.0, 3.0, np.nan, np.nan]), dims=["x"], coords={"x": np.arange(5)},
    )
    an_max = da_allnan.coarsen({"x": 3}, boundary="pad").max(skipna=True)
    an_min = da_allnan.coarsen({"x": 3}, boundary="pad").min(skipna=True)
    print(f"  trailing-block-with-real-data [1,2,3,nan,nan]: max={an_max.values} min={an_min.values} "
          f"(block 1 = [nan,nan,pad], expect NaN not inf)")

    da_fullynan = xr.DataArray(np.full(5, np.nan), dims=["x"], coords={"x": np.arange(5)})
    fn_max = da_fullynan.coarsen({"x": 3}, boundary="pad").max(skipna=True)
    fn_min = da_fullynan.coarsen({"x": 3}, boundary="pad").min(skipna=True)
    print(f"  fully-NaN [nan]*5: max={fn_max.values} min={fn_min.values} (expect [nan, nan], no inf)")
    fully_nan_ok = bool(
        np.all(np.isnan(fn_max.values)) and not np.any(np.isinf(fn_max.values))
        and np.all(np.isnan(fn_min.values)) and not np.any(np.isinf(fn_min.values))
    )

    print("\n  --- real bundle ---")
    block_max = np.asarray(computed["block_max_of_max"].values, dtype="float64")
    block_min = np.asarray(computed["block_min_of_min"].values, dtype="float64")
    n_inf_max = int(np.isinf(block_max).sum())
    n_inf_min = int(np.isinf(block_min).sum())
    n_nan_max = int(np.isnan(block_max).sum())
    n_nan_min = int(np.isnan(block_min).sum())
    print(f"  block_max_of_max: {n_inf_max} +-inf cells (want 0), {n_nan_max} NaN cells of {block_max.size}")
    print(f"  block_min_of_min: {n_inf_min} +-inf cells (want 0), {n_nan_min} NaN cells of {block_min.size}")

    # Trailing block, the specific edge probe_t59_coarsen_edge.py used for mean.
    native_max = np.asarray(computed["native_max"].values, dtype="float64")
    k_lat, k_lon = common["k_lat"], common["k_lon"]
    out_lat = block_max.shape[-2]
    tail_native = native_max[:, (out_lat - 1) * k_lat:, :]
    n_finite_tail = int(np.isfinite(tail_native).sum())
    tail_coarse = block_max[:, -1, :]
    n_finite_coarse_tail = int(np.isfinite(tail_coarse).sum())
    print(f"  trailing row: native finite cells under it = {n_finite_tail}, "
          f"coarse trailing row finite = {n_finite_coarse_tail} of {tail_coarse.size}")

    result = {
        "synthetic_pad_skipped_max": max_pad_ok,
        "synthetic_pad_skipped_min": min_pad_ok,
        "synthetic_fully_nan_block_stays_nan_no_inf": fully_nan_ok,
        "real_bundle_inf_cells_max": n_inf_max,
        "real_bundle_inf_cells_min": n_inf_min,
        "real_bundle_nan_cells_max": n_nan_max,
        "real_bundle_nan_cells_min": n_nan_min,
        "trailing_row_native_finite": n_finite_tail,
        "trailing_row_coarse_finite": n_finite_coarse_tail,
        "verdict": (
            "PASS: pad skipped, all-NaN stays NaN, no +-inf leaked"
            if (max_pad_ok and min_pad_ok and fully_nan_ok and n_inf_max == 0 and n_inf_min == 0)
            else "FAIL: see fields above"
        ),
    }
    print(f"  => {result['verdict']}")
    return result


def g4(computed: xr.Dataset, common: dict) -> dict:
    """Extent overstatement: area painted at a block's peak value versus the
    area that actually held it, pooled as weighted sums (D16's rule) over
    every finite block on the chart -- never a mean of per-block ratios."""
    native_max = np.asarray(computed["native_max"].values, dtype="float64")
    lat_vals = np.asarray(computed[common["lat"]].values, dtype="float64")
    n_buckets, n_lat, n_lon = native_max.shape
    k_lat, k_lon = common["k_lat"], common["k_lon"]
    out_lat = int(np.ceil(n_lat / k_lat))
    out_lon = int(np.ceil(n_lon / k_lon))
    pad_lat, pad_lon = out_lat * k_lat - n_lat, out_lon * k_lon - n_lon

    weight_row = np.cos(np.deg2rad(lat_vals))  # length n_lat
    weight_grid = np.broadcast_to(weight_row[:, None], (n_lat, n_lon))
    real_mask = np.ones((n_lat, n_lon), dtype=bool)

    weight_padded = np.pad(weight_grid, ((0, pad_lat), (0, pad_lon)), constant_values=0.0)
    real_padded = np.pad(real_mask, ((0, pad_lat), (0, pad_lon)), constant_values=False)

    total_native_peak_weighted = 0.0
    total_rendered_weighted = 0.0
    per_frame_ratio = []
    if k_lat == 1 and k_lon == 1:
        print("  k=(1,1): no block reduction runs, overstatement is undefined (1.0x by construction)")
        return {
            "k": [1, 1], "pooled_ratio": 1.0, "note": "k=(1,1): no spatial reduction runs",
        }

    for b in range(n_buckets):
        native_padded = np.pad(
            native_max[b], ((0, pad_lat), (0, pad_lon)), constant_values=np.nan,
        )
        blocks = native_padded.reshape(out_lat, k_lat, out_lon, k_lon)
        wblocks = weight_padded.reshape(out_lat, k_lat, out_lon, k_lon)
        rblocks = real_padded.reshape(out_lat, k_lat, out_lon, k_lon)

        with np.errstate(all="ignore"):
            block_max = np.where(
                np.isfinite(blocks).any(axis=(1, 3)),
                np.nanmax(np.where(np.isfinite(blocks), blocks, -np.inf), axis=(1, 3)),
                np.nan,
            )
        is_max = blocks == block_max[:, None, :, None]
        finite_block = np.isfinite(block_max)

        native_peak_weighted = (wblocks * is_max).sum(axis=(1, 3))
        rendered_weighted = (wblocks * rblocks).sum(axis=(1, 3))

        native_peak_weighted = np.where(finite_block, native_peak_weighted, 0.0)
        rendered_weighted = np.where(finite_block, rendered_weighted, 0.0)

        frame_native = float(native_peak_weighted.sum())
        frame_rendered = float(rendered_weighted.sum())
        if frame_native > 0:
            per_frame_ratio.append(frame_rendered / frame_native)

        total_native_peak_weighted += frame_native
        total_rendered_weighted += frame_rendered

    pooled_ratio = (
        total_rendered_weighted / total_native_peak_weighted
        if total_native_peak_weighted > 0 else float("nan")
    )
    per_frame_ratio = np.asarray(per_frame_ratio)
    result = {
        "k": [k_lat, k_lon],
        "k_squared": k_lat * k_lon,
        "pooled_ratio": round(pooled_ratio, 3),
        "per_frame_p50": round(float(np.median(per_frame_ratio)), 3) if len(per_frame_ratio) else None,
        "per_frame_p90": round(float(np.percentile(per_frame_ratio, 90)), 3) if len(per_frame_ratio) else None,
        "per_frame_max": round(float(per_frame_ratio.max()), 3) if len(per_frame_ratio) else None,
        "n_frames": len(per_frame_ratio),
    }
    print(f"  G4 extent overstatement, pooled over {result['n_frames']} frames, k={k_lat}x{k_lon} "
          f"(k^2={result['k_squared']}):")
    print(f"    pooled ratio (rendered area / native peak area) = {result['pooled_ratio']}x")
    print(f"    per-frame p50={result['per_frame_p50']}x p90={result['per_frame_p90']}x "
          f"max={result['per_frame_max']}x")
    return result


def g5(computed: xr.Dataset, common: dict) -> dict:
    """Bit-exact identities: does the order of temporal-max and spatial-block-
    max matter? Should not -- max is associative."""
    native_max = np.asarray(computed["native_max"].values, dtype="float64")  # (bucket, lat, lon)
    block_max_of_max = np.asarray(computed["block_max_of_max"].values, dtype="float64")  # (bucket, out_lat, out_lon)

    # Identity A: block-max(period max) vs max-over-buckets(block-max(per-bucket max)).
    period_native_max = np.nanmax(native_max, axis=0)  # (lat, lon), global native max
    k_lat, k_lon = common["k_lat"], common["k_lon"]
    n_lat, n_lon = period_native_max.shape
    out_lat = int(np.ceil(n_lat / k_lat))
    out_lon = int(np.ceil(n_lon / k_lon))
    pad_lat, pad_lon = out_lat * k_lat - n_lat, out_lon * k_lon - n_lon
    padded_period = np.pad(period_native_max, ((0, pad_lat), (0, pad_lon)), constant_values=np.nan)
    blocks = padded_period.reshape(out_lat, k_lat, out_lon, k_lon)
    with np.errstate(all="ignore"):
        method_a = np.where(
            np.isfinite(blocks).any(axis=(1, 3)),
            np.nanmax(np.where(np.isfinite(blocks), blocks, -np.inf), axis=(1, 3)),
            np.nan,
        )

    with np.errstate(all="ignore"):
        method_b = np.where(
            np.isfinite(block_max_of_max).any(axis=0),
            np.nanmax(np.where(np.isfinite(block_max_of_max), block_max_of_max, -np.inf), axis=0),
            np.nan,
        )

    both_finite = np.isfinite(method_a) & np.isfinite(method_b)
    both_nan = np.isnan(method_a) & np.isnan(method_b)
    agree_shape = method_a.shape == method_b.shape
    exact_on_finite = bool(np.array_equal(method_a[both_finite], method_b[both_finite]))
    nan_pattern_matches = bool(np.array_equal(np.isnan(method_a), np.isnan(method_b)))
    max_abs_diff = (
        float(np.abs(method_a[both_finite] - method_b[both_finite]).max())
        if both_finite.any() else 0.0
    )

    print("\n  G5 identity A: block-max(global native max) vs max-over-buckets(block-max(per-bucket max))")
    print(f"    shapes match: {agree_shape}   both finite where expected: {int(both_finite.sum())} cells")
    print(f"    NaN pattern matches: {nan_pattern_matches}   exact equality on finite cells: {exact_on_finite}")
    print(f"    max abs diff on finite cells: {max_abs_diff:.6e}")

    result = {
        "identity_a": {
            "nan_pattern_matches": nan_pattern_matches,
            "exact_on_finite": exact_on_finite,
            "max_abs_diff": max_abs_diff,
            "n_compared": int(both_finite.sum()),
        }
    }

    # Identity B (coarsened tier only): group temporally before or after the
    # spatial block max -- should give the same answer, since max commutes.
    group = common.get("buckets_per_frame_for_g5")
    if group and group > 1:
        n_buckets = native_max.shape[0]
        n_frames = int(np.ceil(n_buckets / group))
        pad_b = n_frames * group - n_buckets
        padded_buckets = np.pad(
            block_max_of_max, ((0, pad_b), (0, 0), (0, 0)), constant_values=np.nan,
        )
        grouped_shape = (n_frames, group) + padded_buckets.shape[1:]
        with np.errstate(all="ignore"):
            method_c = np.nanmax(
                np.where(np.isfinite(padded_buckets), padded_buckets, -np.inf).reshape(grouped_shape),
                axis=1,
            )
            method_c = np.where(
                np.isfinite(padded_buckets).reshape(grouped_shape).any(axis=1), method_c, np.nan,
            )

        padded_native = np.pad(
            native_max, ((0, pad_b), (0, 0), (0, 0)), constant_values=np.nan,
        )
        native_grouped_shape = (n_frames, group) + padded_native.shape[1:]
        with np.errstate(all="ignore"):
            temporal_first = np.nanmax(
                np.where(np.isfinite(padded_native), padded_native, -np.inf).reshape(native_grouped_shape),
                axis=1,
            )
            temporal_first = np.where(
                np.isfinite(padded_native).reshape(native_grouped_shape).any(axis=1), temporal_first, np.nan,
            )
        padded_lat, padded_lon = temporal_first.shape[-2], temporal_first.shape[-1]
        blocks_d = np.pad(
            temporal_first, ((0, 0), (0, pad_lat), (0, pad_lon)), constant_values=np.nan,
        ).reshape(n_frames, out_lat, k_lat, out_lon, k_lon)
        with np.errstate(all="ignore"):
            method_d = np.where(
                np.isfinite(blocks_d).any(axis=(2, 4)),
                np.nanmax(np.where(np.isfinite(blocks_d), blocks_d, -np.inf), axis=(2, 4)),
                np.nan,
            )

        both_finite_2 = np.isfinite(method_c) & np.isfinite(method_d)
        nan_pattern_matches_2 = bool(np.array_equal(np.isnan(method_c), np.isnan(method_d)))
        exact_on_finite_2 = bool(np.array_equal(method_c[both_finite_2], method_d[both_finite_2]))
        max_abs_diff_2 = (
            float(np.abs(method_c[both_finite_2] - method_d[both_finite_2]).max())
            if both_finite_2.any() else 0.0
        )
        print(f"\n  G5 identity B (group={group}): block-max-per-bucket-then-group-max "
              f"vs group-max-then-block-max")
        print(f"    NaN pattern matches: {nan_pattern_matches_2}   exact: {exact_on_finite_2}   "
              f"max abs diff: {max_abs_diff_2:.6e}")
        result["identity_b"] = {
            "group": group,
            "nan_pattern_matches": nan_pattern_matches_2,
            "exact_on_finite": exact_on_finite_2,
            "max_abs_diff": max_abs_diff_2,
            "n_compared": int(both_finite_2.sum()),
        }

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                     metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--collection", default="TEMPO_NO2")
    ap.add_argument("--target-cells", type=int, default=20_000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--gate", choices=("g1-synthetic", "g1-real", "g2345", "all"), default="all",
    )
    ap.add_argument("--arm", choices=("mean_only", "mean_max_min"), default="mean_only",
                     help="g1-real only; run each in its own process for a clean peak-RSS baseline")
    ap.add_argument(
        "--max-frames", type=int, default=60,
        help="D7's MAX_FRAMES, to derive buckets_per_frame for G5's identity B "
             "on a coarsened-tier bundle",
    )
    args = ap.parse_args()

    print(f"\n{'=' * 78}\nbundle: {args.bundle}   gate: {args.gate}   target cells: {args.target_cells:,}")
    result: dict = {"bundle": args.bundle, "gate": args.gate}

    if args.gate == "g1-synthetic":
        result["g1_synthetic"] = g1_synthetic()
    elif args.gate == "g1-real":
        common = _common(args)
        result["g1_real"] = g1_real(args, common)
    else:
        common = _common(args)
        n_buckets = common["n_buckets"]
        group = 1 if n_buckets <= args.max_frames else int(np.ceil(n_buckets / args.max_frames))
        common["buckets_per_frame_for_g5"] = group
        print(f"  buckets_per_frame (derived from MAX_FRAMES={args.max_frames}): {group}")

        if args.gate == "all":
            result["g1_synthetic"] = g1_synthetic()

        computed = _native_stats_dataset(common)
        result["g2"] = g2(computed, common, args.target_cells)
        result["g3"] = g3(computed, common)
        result["g4"] = g4(computed, common)
        result["g5"] = g5(computed, common)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
