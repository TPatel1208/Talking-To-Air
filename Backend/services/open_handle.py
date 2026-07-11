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


class UnreadableExportError(OpenHandleError):
    """Raised when an export reported "ready" but the file on disk isn't a
    readable NetCDF/HDF5 dataset — an error-response body or an incomplete/
    empty file saved in place of the granule. Distinct from OpenHandleError
    so open_handle can recognize this transient-looking failure and re-
    materialize once (the same self-heal used for evictions) before giving up."""


async def open_handle(handle: str, tools: dict[str, BaseTool]) -> Any:
    """Resolve ``handle`` to an opened xarray Dataset (Zarr/NetCDF) or Arrow table (Parquet).

    On an expired/evicted export, attempts exactly one rematerialize -> await
    -> re-export cycle; a second failure raises with the MCP's own
    structured message verbatim.
    """
    emit_status("Opening retrieved data...", stage=STAGE_OPEN)
    export = await _export(handle, tools)
    recovered = False
    if export.get("status") != "ready":
        export = await _recover(handle, tools)
        recovered = True
    try:
        return await asyncio.to_thread(_open, export["storage_uri"], export["media_type"])
    except UnreadableExportError:
        # A "ready" export whose file won't open is almost always a transient
        # bad retrieval (an error-response body or an incomplete/empty file
        # saved in place of the granule) — the same class of failure eviction
        # recovery already heals, and the reason a manual retry "just works".
        # Re-materialize once and re-open; only a freshly retrieved file that
        # is *also* unreadable is a real failure, and it propagates with the
        # actionable UnreadableExportError message rather than being retried
        # forever. If we already re-materialized (eviction path), don't loop.
        if recovered:
            raise
        emit_status("Retrieved file was unreadable; re-materializing...", stage=STAGE_OPEN)
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

    # The root group can carry the shared grid coordinates (lat/lon/time)
    # with no data_vars of its own -- a TEMPO L3 single-variable subset
    # splits the science variable into /product but leaves latitude,
    # longitude and time as coordinate variables in the root. Merge the
    # root back in so those coordinates ride along with the science
    # variable; drop it and find_lat_coord sees an empty coord set and
    # every plot/statistics tool fails with "Could not find lat/lon
    # coordinates." even though the granule's grid was right there.
    to_merge = group_datasets if root is None else [root, *group_datasets]
    try:
        merged = xr.merge(to_merge, compat="override", join="override")
    except (ValueError, xr.MergeError):
        merged = group_datasets[0]
    return _promote_lat_lon_coords(merged)


def _open_all_groups(path: str) -> dict[str, Any]:
    """Open every HDF5 group in the file, keyed by group path ("/" for the
    root). Tries h5netcdf first -- pure-Python via h5py (already a
    dependency), so no compiled netCDF-C library needed -- then falls back
    to netCDF4 for classic-format files h5netcdf can't read. These two
    engines between them cover every NetCDF variant (classic-3 via netCDF4,
    NetCDF-4/HDF5 via h5netcdf), so if *both* fail to open the file it isn't
    readable data at all.

    In that case, raise UnreadableExportError with the readers' own errors
    rather than falling back to a bare ``xr.open_dataset(path)`` (no
    ``engine=``). That naked call only re-runs xarray's backend guessing,
    which — on a file with no recognizable NetCDF/HDF5 magic (a zero-byte
    file or an error-response body saved as .nc4) — raises the notoriously
    misleading "did not find a match in any of xarray's currently installed
    IO backends" message, sending users to install packages that are already
    installed. The real cause is an incomplete/failed retrieval, and
    open_handle re-materializes once to heal it.
    """
    import xarray as xr

    errors: list[str] = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return dict(xr.open_groups(path, engine=engine))
        except ImportError:
            continue  # engine not installed — try the other
        except (OSError, ValueError) as exc:
            errors.append(f"{engine}: {exc}")
            continue
    raise UnreadableExportError(
        f"Retrieved file at '{path}' is not a readable NetCDF/HDF5 dataset — "
        "this is usually an incomplete or failed retrieval (e.g. an error "
        "response saved in place of the granule); retrying the retrieval "
        "typically resolves it. Underlying reader errors: "
        + ("; ".join(errors) if errors else "no NetCDF engine (h5netcdf/netCDF4) is installed")
        + "."
    )


def _promote_lat_lon_coords(ds: Any) -> Any:
    """Mark lat/lon-like data variables as coordinates instead of ordinary
    data variables, so they survive variable selection (e.g.
    AggregationService.to_dataarray) instead of being dropped -- or, worse,
    mistaken for the science variable -- when a grouped product splits its
    lon/lat into a sibling subgroup from its science data.

    Identification is the canonical CF-metadata-primary one (T24), so a
    product whose lat/lon carry standard_name/units under a spelling no
    name allowlist would guess is still promoted."""
    from utils.geo_utils import identify_lat, identify_lon

    identified = [identify_lat(ds), identify_lon(ds)]
    to_promote = [name for name in identified if name in ds.data_vars]
    return ds.set_coords(to_promote) if to_promote else ds
