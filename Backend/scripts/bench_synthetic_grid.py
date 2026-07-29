"""Rebuild T51's synthetic TEMPO-L3-shaped bundle and run all three cases.

Not a fixture and not a test — a one-shot reproduction of the exact grid the
T51 numbers in docs/prds/prd-t52-l4-zarr-cube-cache.md were taken on, so
case 3's number is comparable to cases 1 and 2 rather than being a different
measurement wearing the same table.
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (BACKEND_DIR, os.path.join(BACKEND_DIR, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

LATS, LONS, MEMBERS = 1750, 3250, 4


def build_bundle(directory: str) -> str:
    import numpy as np
    import xarray as xr

    lats = np.linspace(20.0, 55.0, LATS).astype("float32")
    lons = np.linspace(-130.0, -65.0, LONS).astype("float32")
    rng = np.random.default_rng(0)

    zip_path = os.path.join(directory, "bundle.nc.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member in range(MEMBERS):
            values = rng.random((1, LATS, LONS), dtype="float32") * 5e-6
            ds = xr.Dataset(
                {"vertical_column_troposphere": (("time", "latitude", "longitude"), values)},
                coords={
                    "time": [np.datetime64(f"2026-07-01T{12 + member:02d}:00:00")],
                    "latitude": lats,
                    "longitude": lons,
                },
                attrs={"ShortName": "TEMPO_NO2_L3_SYNTHETIC"},
            )
            member_path = os.path.join(directory, f"granule_{member:02d}.nc4")
            ds.to_netcdf(member_path, engine="h5netcdf")
            zf.write(member_path, arcname=f"granule_{member:02d}.nc4")
            os.remove(member_path)
    return zip_path


def main() -> None:
    import bench_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        bundle = build_bundle(tmp)
        print(f"bundle bytes: {os.path.getsize(bundle):,}")
        report = bench_pipeline.run_benchmark(
            bundle, variable="vertical_column_troposphere", aoi="-74.9,39.1,-73.1,40.9", runs=5
        )
        print(bench_pipeline.format_report(report))


if __name__ == "__main__":
    main()
