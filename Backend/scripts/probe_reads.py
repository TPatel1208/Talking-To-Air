"""Probe: how many times is a lazily-opened multi-granule bundle actually read
across masking + the aggregate reduction? Prints, asserts nothing.

This is the measurement behind the T55 counter fusion (commit bc8acb2): it
counts *materialisations* of each granule chunk by wrapping every block in a
``dask.delayed`` that appends to a list when it runs, so an extra pass over the
bundle shows up as duplicated entries rather than as a wall-clock difference
that could be noise. Run it before and after touching AggregationService's
reduction path -- the read count is the thing to keep down, and CPU time alone
will not tell you a second full read happened.

    python Backend/scripts/probe_reads.py
"""
import pathlib
import sys

import dask
import dask.array
import numpy as np
import xarray as xr

_BACKEND = pathlib.Path(__file__).resolve().parent.parent  # Backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tta_backend.preprocessing.aggregation_service import AggregationService  # noqa: E402

PINNED = {"quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}


def bundle(loads, n=3):
    def chunk(name, index, values, dtype):
        block = np.asarray(values, dtype=dtype)[None, ...]

        def _load():
            loads.append(f"{name}[{index}]")
            return block

        return dask.array.from_delayed(dask.delayed(_load)(), shape=block.shape, dtype=dtype)

    science = dask.array.concatenate(
        [chunk("no2", i, [[1.0, 2.0], [3.0, 4.0 + i]], "float64") for i in range(n)], axis=0)
    flag = dask.array.concatenate(
        [chunk("flag", i, [[0, 1], [0, 0]], "int64") for i in range(n)], axis=0)
    return xr.Dataset(
        {"no2": (("time", "lat", "lon"), science),
         "main_data_quality_flag": (("time", "lat", "lon"), flag)},
        coords={"time": np.array(["2024-01-01", "2024-01-02", "2024-01-03"][:n], dtype="datetime64[ns]"),
                "lat": [10.0, 20.0], "lon": [30.0, 40.0]},
    )


print("=== full aggregate() ===")
loads = []
res = AggregationService().aggregate(bundle(loads), variable="no2", col_info=PINNED)
print("after aggregate() returns:", len(loads), loads)
res.ds["no2"].values  # force the lazy graph; the count above vs below is the point
print("after forcing .values   :", len(loads), loads)
print("masking:", {k: v for k, v in res.meta["masking"].items() if k.startswith("qa_")})

print()
print("=== masking only (resolve_and_mask) ===")
loads = []
ds = bundle(loads)
masked, _prov = AggregationService().resolve_and_mask(
    ds["no2"], variable="no2", col_info=PINNED, source_ds=ds)
print("after resolve_and_mask  :", len(loads), loads)
print("chunks still lazy?      :", masked.chunks is not None)

print()
print("=== is the delayed itself memoized across two computes? ===")
loads2 = []


def _l():
    loads2.append("x")
    return np.array([[1.0, 2.0]])


arr = dask.array.from_delayed(dask.delayed(_l)(), shape=(1, 2), dtype="float64")
d = xr.DataArray(arr, dims=("time", "lon"))
d.sum().compute()
d.mean().compute()
print("two separate computes ->", len(loads2), "loads (expect 2 if no memoization)")
