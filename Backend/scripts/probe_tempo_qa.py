"""Probe: on a REAL TEMPO NO2 L3 granule, does the pinned QA mask actually run,
and what does it change? Prints, asserts nothing.

This is the verification behind the fix/QA_flagging diagnosis. The screenshot
that started it showed "QA pass rate: Not applied - semantics unknown" on a
16-granule TEMPO NO2 daily mean, with a min of -3.29e+14. Three claims came out
of reading the code; this script tests all three against a real file instead of
a synthetic fixture (probe_reads.py already covers the synthetic case):

  1. QA masking is skipped because ``main_data_quality_flag`` was never
     retrieved -- not because resolution failed. Tested by running the SAME
     masking call twice: once on the full file, once on a file with the flag
     dropped (what Harmony variable-subsetting returns today). The two
     ``qa_status`` values are the finding.
  2. The negative minimum is QA noise, not a bad valid_min. Tested by
     reporting min/mean before and after the mask.
  3. The frontend's stat cards describe the decimated render grid, not the
     science field. Tested by computing the full-field cos(lat)-weighted mean
     and comparing it to the unweighted mean of the real _downsample_grid
     output -- the same thinning the browser receives.

    python Backend/scripts/probe_tempo_qa.py <path-to-granule.nc>
"""
import pathlib
import sys

import numpy as np
import xarray as xr

_BACKEND = pathlib.Path(__file__).resolve().parent.parent  # Backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tta_backend.datasets.registry import load_registry  # noqa: E402
from tta_backend.preprocessing.aggregation_service import AggregationService  # noqa: E402
from tta_backend.tools.satellite_tools.plot_tools import _downsample_grid  # noqa: E402


def _groups(path):
    """Every group in the file, root first. TEMPO L3 keeps the science bands in
    /product and the lat/lon coords at the root, so both are needed."""
    import netCDF4

    found = [""]
    with netCDF4.Dataset(path) as nc:
        def walk(node, prefix):
            for name, child in node.groups.items():
                full = f"{prefix}/{name}".lstrip("/")
                found.append(full)
                walk(child, full)
        walk(nc, "")
    return found


def open_merged(path):
    """Mirror the MCP ``open_handle`` contract: merge every group down to BARE
    leaf names (no group prefix), stamping ``group_path`` on each band. The
    registry pins ``quality_flag_var: main_data_quality_flag`` bare, so this is
    exactly the naming the mask has to match -- if this probe has to prefix
    anything to make it work, the fix is wrong.
    """
    merged = xr.Dataset()
    for group in _groups(path):
        try:
            ds = xr.open_dataset(path, group=group or None, decode_times=False)
        except (OSError, ValueError):
            continue
        for name, var in ds.data_vars.items():
            if name in merged.data_vars:
                continue                      # first group wins, as open_handle does
            var.attrs["group_path"] = group or "/"
            merged[name] = var
        merged = merged.assign_coords({k: v for k, v in ds.coords.items()
                                       if k not in merged.coords})
    return merged


def _latlon(da):
    lat = next((d for d in da.dims if str(d).lower().startswith("lat")), None)
    lon = next((d for d in da.dims if str(d).lower().startswith("lon")), None)
    return lat, lon


def _weighted_mean(da, lat_dim):
    """Area-weighted (cos-lat) mean over the full field -- the honest number."""
    weights = np.cos(np.deg2rad(da[lat_dim]))
    return float(da.weighted(weights.fillna(0)).mean().values)


def describe(label, da):
    arr = np.asarray(da.values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"  {label:<22} no finite values")
        return
    print(f"  {label:<22} n={finite.size:>10,}  "
          f"min={finite.min():>12.4e}  mean={finite.mean():>12.4e}  max={finite.max():>12.4e}  "
          f"neg={100.0 * (finite < 0).mean():5.1f}%")


def main(path):
    cfg = load_registry()["TEMPO_NO2"]
    col_info = cfg.model_dump()
    science, flag = cfg.primary_var, cfg.quality_flag_var

    print(f"=== file: {path}")
    print(f"groups: {_groups(path)}")

    ds = open_merged(path)
    print(f"merged data_vars ({len(ds.data_vars)}): {sorted(ds.data_vars)[:12]}")
    print(f"science var present: {science in ds.data_vars}")
    print(f"flag var present   : {flag in ds.data_vars}   <- claim 1 hinges on this name")
    if science not in ds.data_vars:
        print("!! science variable missing; wrong file?")
        return
    if flag in ds.data_vars:
        fa = ds[flag].attrs
        print(f"flag CF attrs      : flag_values={fa.get('flag_values')!r} "
              f"flag_meanings={fa.get('flag_meanings')!r}")
        print(f"flag group_path    : {fa.get('group_path')!r}")
    sa = ds[science].attrs
    print(f"science CF attrs   : _FillValue={sa.get('_FillValue')!r} "
          f"valid_range={sa.get('valid_range')!r} units={sa.get('units')!r}")
    print(f"registry pins      : fill={col_info.get('fill_value')} "
          f"valid_min={col_info.get('valid_min')} valid_max={col_info.get('valid_max')} "
          f"qa_good_values={col_info.get('qa_good_values')}")

    svc = AggregationService()
    da = ds[science]

    print()
    print("=== claim 2: what the QA mask actually removes ===")
    describe("raw (unmasked)", da)

    # (a) today's shape: Harmony subset to the science variable only.
    subset_only = ds[[science]]
    masked_a, prov_a = svc.resolve_and_mask(
        subset_only[science], variable=science, col_info=col_info, source_ds=subset_only)
    describe("masked, flag ABSENT", masked_a)

    # (b) post-fix shape: the flag rode along with the science variable.
    masked_b, prov_b = svc.resolve_and_mask(
        da, variable=science, col_info=col_info, source_ds=ds)
    describe("masked, flag PRESENT", masked_b)

    print()
    print("=== claim 1: the two provenance outcomes ===")
    keys = ("qa_status", "qa_source", "qa_note", "qa_pass_rate",
            "qa_checked_pixels", "qa_passing_pixels")
    for label, prov in (("flag ABSENT (today)", prov_a), ("flag PRESENT (post-fix)", prov_b)):
        print(f"  {label}:")
        for k in keys:
            if k in prov:
                print(f"      {k:<20} {prov[k]!r}")

    print()
    print("=== claim 3: full field vs the grid the browser actually gets ===")
    field = masked_b if flag in ds.data_vars else masked_a
    field = field.squeeze(drop=True)
    lat_dim, lon_dim = _latlon(field)
    if lat_dim is None or lon_dim is None or field.ndim != 2:
        print(f"  skipped: need a 2-D lat/lon field, got dims={field.dims}")
        return

    field = field.transpose(lat_dim, lon_dim)
    arr = np.asarray(field.values, dtype="float64")
    lats = np.asarray(field[lat_dim].values, dtype="float64")
    lons = np.asarray(field[lon_dim].values, dtype="float64")

    full_finite = arr[np.isfinite(arr)]
    full_unweighted = float(full_finite.mean())
    full_weighted = _weighted_mean(field, lat_dim)

    _, _, thin = _downsample_grid(lats, lons, arr)
    thin_finite = thin[np.isfinite(thin)]
    thin_mean = float(thin_finite.mean())

    print(f"  full grid           cells={arr.size:>10,}  finite={full_finite.size:>10,}")
    print(f"  decimated (browser) cells={thin.size:>10,}  finite={thin_finite.size:>10,}"
          f"   <- 'Sample count'")
    print(f"  full,  cos-lat weighted mean : {full_weighted:>12.4e}   <- the honest number")
    print(f"  full,  unweighted mean       : {full_unweighted:>12.4e}")
    print(f"  decimated, unweighted mean   : {thin_mean:>12.4e}   <- what the card shows")
    if full_weighted:
        print(f"  card error vs honest mean    : "
              f"{100.0 * (thin_mean - full_weighted) / abs(full_weighted):+.2f}%")
    print(f"  full min/max      : {full_finite.min():.4e} / {full_finite.max():.4e}")
    print(f"  decimated min/max : {thin_finite.min():.4e} / {thin_finite.max():.4e}"
          f"   <- extremes the card reports")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
