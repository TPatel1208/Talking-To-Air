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
import os
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from config.settings import get_settings
from config.workflow_stages import STAGE_OPEN
from earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError, parse_tool_result
from services.retrieval_composites import await_retrieval
from utils.streaming import emit_status

logger = logging.getLogger(__name__)

try:
    # Bound dask's threaded scheduler process-wide: every in-flight task can
    # hold a whole granule chunk, and the default worker count (one per CPU)
    # lets a single aggregation stage n_cpus granules at once — reintroducing
    # the memory spike the lazy bundle open below exists to avoid. Two workers
    # keep granule reads pipelined without staging the bundle.
    import dask

    dask.config.set(num_workers=2)
except ImportError:  # pragma: no cover — dask is a declared dependency
    dask = None

# Bundle members are extracted here (under tempfile.gettempdir()) rather than
# a per-call tempdir: members are opened lazily, so their files are read well
# after _open_netcdf_bundle returns, and derived Datasets give no safe hook to
# know when the last reader is done. Entries are keyed by bundle identity
# (reused on repeat opens) and swept by age on each new extraction.
_EXTRACT_CACHE_DIR_NAME = "tta_bundle_extract"
_EXTRACT_CACHE_TTL_SECONDS = 3600.0
_EXTRACT_COMPLETE_MARKER = ".complete"


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
    if "hdf4" in mt or "native-archive" in mt:
        # The MCP materialized the provider's native distribution (HDF4 or a
        # mixed archive) because no NetCDF conversion service exists for the
        # collection. No local reader can open these, and re-retrieving
        # returns the same bytes — the actionable move is a different product.
        raise OpenHandleError(
            f"This dataset is distributed in a native format ('{media_type}') that the "
            "visualization pipeline cannot open. Retrying the retrieval will not help — "
            "suggest a different collection for this variable (an L3/L4 NetCDF product) instead."
        )
    if "bundle" in mt:
        return _open_netcdf_bundle(path)
    if "netcdf" in mt:
        return _open_netcdf(path)
    raise OpenHandleError(f"Unsupported media_type '{media_type}' for exported handle.")


def _open_netcdf(path: str, chunks: dict | None = None) -> Any:
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

    # A zip archive labeled plain netCDF (a bundle materialized before the
    # MCP's content sniffing existed, or a mislabeled legacy row): both
    # NetCDF engines reject it with "file signature not found", which the
    # UnreadableExportError path below misreads as a failed retrieval and
    # sends callers into pointless retries. Route by the bytes instead.
    if _is_zipfile(path):
        return _open_netcdf_bundle(path)

    groups = _open_all_groups(path, chunks)
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


def _is_zipfile(path: str) -> bool:
    import zipfile

    return zipfile.is_zipfile(path)


def _open_netcdf_bundle(path: str) -> Any:
    """Open a ``application/netcdf-bundle+zip`` export — a zip of NetCDF
    granule subsets — into one Dataset, concatenated on ``time``.

    The MCP ships every OPeNDAP subset and every multi-granule Harmony
    result as one of these bundles (its own ``_open_netcdf_bundle`` in
    ``tools/_dataio.py`` is the reference implementation). Each member is
    opened through :func:`_open_netcdf`, so grouped products (TEMPO/OMI L3)
    get the same group-merging and lat/lon promotion a bare NetCDF export
    gets, and variable names stay bare (no group prefixes) — unlike the
    MCP's flattener, whose prefixed names this backend's callers don't use.

    Members are opened *lazily* (one dask chunk per file), never eagerly
    loaded: the previous load-everything-then-concat shape held the whole
    day in memory twice, and a full-day TEMPO NO2 bundle OOM-killed the
    backend (live 2026-07-12). Laziness means the extracted files must
    outlive this call, so members land in a TTL-pruned cache directory
    keyed by bundle identity (see :func:`_extract_bundle_cached`) instead
    of a delete-before-return tempdir — a repeat open of the same bundle
    also skips re-extraction. Before any extraction, the bundle's total
    uncompressed size is gated (:func:`_gate_bundle_size`) so an
    arbitrarily large request refuses deterministically rather than taking
    the process down.

    Each member's CF time decodes against its *own* units at open time
    (xarray's default; time is an index coordinate, decoded even when data
    variables stay lazy), so granules with per-file epochs (e.g. MERRA-2
    daily) concat on absolute timestamps, not raw offsets. Members whose
    singleton time dim has no coordinate variable get one synthesized from
    their CMR granule date attrs, mirroring the MCP.
    """
    import zipfile

    import xarray as xr

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise UnreadableExportError(
            f"Retrieved bundle at '{path}' is not a readable zip archive — this is "
            "usually an incomplete or failed retrieval; retrying the retrieval "
            f"typically resolves it. Underlying error: {exc}"
        )

    with zf:
        # Granule filenames sort chronologically, so name order is time order.
        names = sorted(n for n in zf.namelist() if not n.endswith("/"))
        if not names:
            raise UnreadableExportError(
                f"Retrieved bundle at '{path}' is an empty archive — this is usually "
                "a failed retrieval; retrying the retrieval typically resolves it."
            )
        _gate_bundle_size(zf, path)
        extract_dir = _extract_bundle_cached(zf, path, names)

    chunks = _lazy_chunks()
    members: list[Any] = []
    for name in names:
        member_path = os.path.join(extract_dir, *name.split("/"))
        ds = _open_netcdf(member_path, chunks=chunks)
        members.append(_synthesize_member_time_coord(ds))

    if len(members) == 1:
        return members[0]
    normalized = [_strip_concat_unsafe_coord_attrs(ds) for ds in members]
    try:
        return xr.concat(normalized, dim="time")
    except Exception as exc:
        raise OpenHandleError(
            f"Could not combine the {len(members)} granules in bundle '{path}' onto a "
            f"shared time axis: {exc}"
        )


def _gate_bundle_size(zf: Any, path: str) -> None:
    """Refuse a bundle whose members' total *uncompressed* size exceeds the
    configured cap, before anything is extracted or opened.

    The retrieval-time byte caps (services.retrieval_composites) gate the
    provider's estimate at submit time; this gate catches what they can't —
    decompression, dtype widening and multi-granule concatenation all happen
    at open time. Raising the same structured too_large error the retrieval
    gates use keeps the failure on the T18 deterministic-error surface (the
    agent relays "narrow the request") instead of the OOM killer's."""
    granules = [info for info in zf.infolist() if not info.is_dir()]
    total = sum(info.file_size for info in granules)
    limit = get_settings().bundle_open_max_uncompressed_bytes
    if total <= limit:
        return
    raise MCPToolError(
        CATEGORY_TOO_LARGE,
        f"This result bundle holds {len(granules)} granule file(s) totalling "
        f"~{total:,} bytes uncompressed, over the {limit:,}-byte limit this "
        f"deployment can open safely (bundle: '{path}').",
        suggestion="Narrow the time range, area of interest, or variable list and retrieve again.",
    )


def _lazy_chunks() -> dict | None:
    """``chunks={}`` (one dask chunk per variable per file) when dask is
    installed, so bundle members stay on disk until a compute needs them and
    reductions stream granule-by-granule; None otherwise, where xarray's
    plain lazy arrays materialize at concat and the size gate is the only
    protection."""
    return None if dask is None else {}


def _extract_bundle_cached(zf: Any, path: str, names: list[str]) -> str:
    """Extract ``zf``'s members into a cache entry that outlives this call
    and return its directory.

    Lazily-opened members read these files long after the bundle open
    returns — any compute up to the end of the tool call — and derived
    Datasets keep no reference through which a per-call cleanup could know
    when the last reader is done. So entries live under one process-local
    cache root, keyed by the bundle file's identity (path/size/mtime),
    touched on reuse, and pruned by age on each new extraction
    (:func:`_prune_extract_cache`). Extraction goes into a staging dir
    renamed atomically into place, so a concurrent open of the same bundle
    either wins the rename or adopts the winner's completed entry."""
    import hashlib
    import shutil
    import tempfile

    root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
    os.makedirs(root, exist_ok=True)
    _prune_extract_cache(root)

    stat = os.stat(path)
    key = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    final_dir = os.path.join(root, key)
    marker = os.path.join(final_dir, _EXTRACT_COMPLETE_MARKER)
    if os.path.exists(marker):
        os.utime(final_dir, None)  # keep a hot entry out of the pruner's reach
        return final_dir

    staging = tempfile.mkdtemp(prefix=f"staging-{key}-", dir=root)
    try:
        for name in names:
            zf.extract(name, staging)
        with open(os.path.join(staging, _EXTRACT_COMPLETE_MARKER), "w"):
            pass
        os.rename(staging, final_dir)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if os.path.exists(marker):
            # Lost the race: a concurrent open of this bundle finished
            # extracting between the marker check and the rename.
            return final_dir
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_dir


def _prune_extract_cache(root: str, ttl_seconds: float = _EXTRACT_CACHE_TTL_SECONDS) -> None:
    """Sweep cache entries (completed or abandoned staging dirs) untouched
    for longer than the TTL. Reuse touches an entry's mtime, so only bundles
    nothing has opened for a full TTL are removed — far longer than any
    single tool call keeps lazy readers on them."""
    import shutil
    import time

    cutoff = time.time() - ttl_seconds
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
        except OSError:
            continue


def _synthesize_member_time_coord(ds: Any) -> Any:
    """Give a bundle member a real, indexed ``time`` coordinate before concat.

    Some daily L3 products (e.g. OMI_MINDS_NO2d) carry a differently-cased
    singleton time dimension (``Time``) with no coordinate variable at all —
    the granule's date lives only in the ``RangeBeginningDate``/
    ``RangeBeginningTime`` global attrs (standard CMR/UMM-G granule temporal
    metadata). Left alone, ``xr.concat(dim="time")`` fabricates a brand-new
    unindexed stacking dimension instead of reusing it. No-op when ``time``
    already exists or the date attrs are absent. (Ported from the MCP's
    ``_synthesize_bundle_time_coord``.)
    """
    import numpy as np

    if "time" in ds.dims:
        return ds
    candidates = [d for d in ds.dims if str(d).lower() == "time" and ds.sizes[d] == 1]
    if not candidates:
        return ds
    date = ds.attrs.get("RangeBeginningDate")
    if not date:
        return ds
    time_str = f"{date}T{ds.attrs.get('RangeBeginningTime', '00:00:00').rstrip('Z')}"
    ds = ds.rename({candidates[0]: "time"})
    return ds.assign_coords({"time": [np.datetime64(time_str)]})


def _strip_concat_unsafe_coord_attrs(ds: Any) -> Any:
    """Drop ``units``/``calendar`` from coords so cross-granule concat doesn't
    trip xarray's attribute-equality check when granules were written at
    different times. Time values are already decoded to datetime64 per member,
    so nothing downstream needs these attrs to interpret the axis."""
    ds = ds.copy()
    for coord in ds.coords:
        for attr in ("units", "calendar"):
            ds[coord].attrs.pop(attr, None)
            ds[coord].encoding.pop(attr, None)
    return ds


def _open_all_groups(path: str, chunks: dict | None = None) -> dict[str, Any]:
    """Open every HDF5 group in the file, keyed by group path ("/" for the
    root). Tries h5netcdf first -- pure-Python via h5py (already a
    dependency), so no compiled netCDF-C library needed -- then falls back
    to netCDF4 for classic-format files h5netcdf can't read. These two
    engines between them cover every NetCDF variant (classic-3 via netCDF4,
    NetCDF-4/HDF5 via h5netcdf), so if *both* fail to open the file it isn't
    readable data at all. ``chunks`` is forwarded to the reader (bundle
    members pass ``{}`` for lazy dask-backed variables; bare exports keep
    the default eager-on-access behavior).

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
            return dict(xr.open_groups(path, engine=engine, chunks=chunks))
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
