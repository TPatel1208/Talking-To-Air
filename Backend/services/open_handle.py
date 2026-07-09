"""
services/open_handle.py
========================
The single seam between an ``obs_``/``cube_`` handle and an opened dataset.

Wraps ``export_result`` with a bounded eviction-recovery loop
(rematerialize -> await -> re-export) so every plot/statistics tool sees
either an opened Dataset/Table or a clear error — never a missing file.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from config.workflow_stages import STAGE_OPEN
from earthdata_mcp.results import parse_tool_result
from services.retrieval_composites import await_retrieval
from utils.streaming import emit_status

logger = logging.getLogger(__name__)


class OpenHandleError(RuntimeError):
    """Raised when a handle cannot be opened, even after one rematerialize attempt."""


async def open_handle(handle: str, tools: dict[str, BaseTool]) -> Any:
    """Resolve ``handle`` to an opened xarray Dataset (Zarr/NetCDF) or Arrow table (Parquet).

    On an expired/evicted export, attempts exactly one rematerialize -> await
    -> re-export cycle; a second failure raises with the MCP's own
    structured message verbatim.
    """
    emit_status("Opening retrieved data...", stage=STAGE_OPEN)
    export = await _export(handle, tools)
    if export.get("status") != "ready":
        export = await _recover(handle, tools)
    return await asyncio.to_thread(_open, export["storage_uri"], export["media_type"])


async def _export(handle: str, tools: dict[str, BaseTool]) -> dict:
    raw = await tools["export_result"].ainvoke({"handle": handle})
    return parse_tool_result(raw)


async def _recover(handle: str, tools: dict[str, BaseTool]) -> dict:
    emit_status("Rematerializing expired data...", stage=STAGE_OPEN)
    remat_raw = await tools["rematerialize"].ainvoke({"handle": handle})
    remat = parse_tool_result(remat_raw)
    if remat.get("status") == "not_found":
        raise OpenHandleError(remat.get("message") or f"Handle '{handle}' not found and cannot be rematerialized.")

    job_handle = remat.get("job_handle")
    if job_handle:
        status = await await_retrieval(job_handle, tools)
        if status.get("status") != "ready":
            raise OpenHandleError(status.get("message") or f"Rematerializing handle '{handle}' failed.")

    second_export = await _export(handle, tools)
    if second_export.get("status") != "ready":
        raise OpenHandleError(
            second_export.get("message") or f"Handle '{handle}' still not ready after rematerialize."
        )
    return second_export


def _open(storage_uri: str, media_type: str) -> Any:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise OpenHandleError(
            f"Opening non-local URIs (scheme '{parsed.scheme}') is not yet supported: {storage_uri}"
        )
    path = url2pathname(parsed.path)

    mt = (media_type or "").lower()
    if "zarr" in mt:
        import xarray as xr

        return xr.open_zarr(path)
    if "parquet" in mt:
        import pyarrow.parquet as pq

        return pq.read_table(path)
    if "netcdf" in mt:
        return _open_netcdf(path)
    raise OpenHandleError(f"Unsupported media_type '{media_type}' for exported handle.")


def _open_netcdf(path: str) -> Any:
    """Open a NetCDF file, descending into HDF5 subgroups when the root
    group carries no data variables.

    Some providers (e.g. TEMPO L3, OMI L3) nest their science variables
    under a subgroup such as /product -- and their lon/lat under a
    *different sibling* subgroup such as /geolocation -- leaving the root
    group empty. xr.open_dataset(path) alone only sees the root group,
    which AggregationService then reports as "Dataset has no data
    variables." rather than any group-specific error.

    Every non-empty group is merged into one Dataset by name (unchanged,
    so a caller relying on a known variable name like
    "vertical_column_troposphere" still finds it bare -- no group
    prefixing). Any lon/lat-like variable is then promoted from an
    ordinary data variable to a coordinate, wherever it happens to live,
    so it travels with whichever science variable gets selected downstream
    instead of being lost -- or, worse, mistaken for the science variable
    itself when a merged Dataset's first "data var" is actually longitude.

    Detected dynamically off the file itself (not the dataset registry) so
    it also covers datasets collections.yaml hasn't been told about yet,
    and generalizes to arbitrary group layouts rather than assuming
    "/product" and "/geolocation" by name.
    """
    import xarray as xr

    groups = _open_all_groups(path)
    root = groups.pop("/", None)
    if root is not None and root.data_vars:
        return root  # genuinely flat file; nothing nested to merge

    group_datasets = [g for g in groups.values() if g.data_vars]
    if not group_datasets:
        return root if root is not None else xr.Dataset()
    if len(group_datasets) == 1:
        return _promote_lat_lon_coords(group_datasets[0])

    try:
        merged = xr.merge(group_datasets, compat="override", join="override")
    except (ValueError, xr.MergeError):
        merged = group_datasets[0]
    return _promote_lat_lon_coords(merged)


def _open_all_groups(path: str) -> dict[str, Any]:
    """Open every HDF5 group in the file, keyed by group path ("/" for the
    root). Tries h5netcdf first -- pure-Python via h5py (already a
    dependency), so no compiled netCDF-C library needed -- then falls back
    to netCDF4 for classic-format files h5netcdf can't read. Either
    backend's absence, or a file neither can open as grouped, degrades to
    "just the root dataset" rather than a crash.
    """
    import xarray as xr

    for engine in ("h5netcdf", "netcdf4"):
        try:
            return dict(xr.open_groups(path, engine=engine))
        except (ImportError, OSError, ValueError):
            continue
    return {"/": xr.open_dataset(path)}


def _promote_lat_lon_coords(ds: Any) -> Any:
    """Mark lat/lon-like data variables as coordinates instead of ordinary
    data variables, so they survive variable selection (e.g.
    AggregationService.to_dataarray) instead of being dropped -- or, worse,
    mistaken for the science variable -- when a grouped product splits its
    lon/lat into a sibling subgroup from its science data."""
    from utils.geo_utils import LAT_COORD_CANDIDATES, LON_COORD_CANDIDATES

    candidates = set(LAT_COORD_CANDIDATES) | set(LON_COORD_CANDIDATES)
    to_promote = [name for name in ds.data_vars if name in candidates]
    return ds.set_coords(to_promote) if to_promote else ds
