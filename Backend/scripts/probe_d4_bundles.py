"""Probe materialized Harmony bundles for a D4 characterization candidate.

D4 (PRD T59) asks whether the period map -- today's cadence-bucket-WEIGHTED
temporal mean -- differs materially from the mean-of-cadence-bucket-means. The
two are algebraically identical except where per-pixel NaNs make
``.weighted().mean(skipna=True)`` renormalize, so the probe needs a bundle that
is (a) multi-granule, (b) spread across more than one cadence bucket, and
(c) carries a QA flag whose masking actually produces those NaNs.

Read-only: peeks at one member per bundle and prints what it is.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import xarray as xr

HARMONY = Path("/data/harmony")


def peek(path: Path) -> None:
    try:
        zf = zipfile.ZipFile(path)
    except Exception as exc:
        print(f"  !! not a zip: {exc}")
        return
    names = [n for n in zf.namelist() if n.endswith((".nc", ".nc4"))]
    if not names:
        print(f"  !! no netCDF members ({zf.namelist()[:3]})")
        return

    print(f"  members: {len(names)}")
    tmp = Path("/tmp/d4_peek.nc")
    with zf.open(names[0]) as src, open(tmp, "wb") as dst:
        dst.write(src.read())

    groups: dict[str, list[str]] = {}
    try:
        root = xr.open_dataset(tmp, engine="h5netcdf")
        groups[""] = list(root.data_vars)
        short = root.attrs.get("short_name") or root.attrs.get("shortName") or "?"
        tstart = root.attrs.get("time_coverage_start", "?")
        root.close()
    except Exception as exc:
        print(f"  !! root open failed: {exc}")
        return

    import h5py

    with h5py.File(tmp, "r") as h5:
        for key in h5:
            if isinstance(h5[key], h5py.Group):
                groups[key] = list(h5[key].keys())

    print(f"  short_name: {short}   time_coverage_start: {tstart}")
    for g, vs in groups.items():
        label = g or "(root)"
        print(f"  {label}: {vs[:12]}{' ...' if len(vs) > 12 else ''}")

    qa = [
        f"{g}/{v}" if g else v
        for g, vs in groups.items()
        for v in vs
        if "quality" in v.lower() or "qa" in v.lower() or v.lower().endswith("_flag")
    ]
    print(f"  QA-flag candidates: {qa or 'NONE'}")

    # Timestamps across every member -- this is what decides whether the bundle
    # spans more than one cadence bucket, which is the whole point of D4.
    stamps = []
    for n in names:
        with zf.open(n) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        try:
            with xr.open_dataset(tmp, engine="h5netcdf") as ds:
                stamps.append(ds.attrs.get("time_coverage_start", "?"))
        except Exception:
            stamps.append("?")
    stamps.sort()
    print(f"  span: {stamps[0]}  ->  {stamps[-1]}")
    days = {s[:10] for s in stamps if s != "?"}
    hours = {s[:13] for s in stamps if s != "?"}
    print(f"  distinct days: {len(days)}   distinct hours: {len(hours)}")


def main() -> None:
    targets = sys.argv[1:]
    paths = [Path(t) for t in targets] if targets else sorted(HARMONY.glob("*/result.nc.zip"))
    for p in paths:
        print(f"\n=== {p}  ({p.stat().st_size / 1e6:.0f} MB)")
        peek(p)


if __name__ == "__main__":
    main()
