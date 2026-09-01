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
import hashlib
import logging
import math
import os
import threading
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from tta_backend.config.settings import get_settings
from tta_backend.config.workflow_stages import STAGE_OPEN
from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError, parse_tool_result
from tta_backend.services.retrieval_composites import await_retrieval
from tta_backend.utils.phase_timing import phase_timer
from tta_backend.utils.streaming import emit_status

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

# Serializes the actual HDF5/netCDF-C-library call inside a concurrent bundle
# member open (_open_bundle_members_concurrently). h5netcdf/netCDF4 release
# the GIL for their I/O, so Python threads genuinely overlap *inside* the C
# extension — but that's a different guarantee from the underlying HDF5 C
# library itself being safe for concurrent calls across different file
# handles, which the reference HDF5 build only provides with a special
# --enable-threadsafe compile flag (and even then by serializing everything
# behind one global lock, not truly parallelizing it). Nothing in this
# deployment establishes that guarantee, so this lock takes the same
# "serialize the actual library call" stance HDF5's own thread-safe build
# would — the surrounding thread-pool structure still overlaps the pure-Python
# per-member work (path handling, group merging, time-coord synthesis) that
# doesn't touch the C library.
_hdf5_open_lock = threading.Lock()


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
    cached = _serve_from_index(handle)
    if cached is not None:
        return cached
    export = await _export(handle, tools)
    recovered = False
    if export.get("status") != "ready":
        export = await _recover(handle, tools)
        recovered = True
    try:
        ds = await _open_export(export)
        _consider_cube(export, ds)
        return ds
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
        ds = await _open_export(export)
        _consider_cube(export, ds)
        return ds


async def _open_export(export: dict) -> Any:
    """Open a ready export off the event loop, carrying the whole response.

    The delivered-content fields (``content_digest``/``partial``, upstream PRD
    023) are what the cube key is derived from (T54), so the reader needs the
    response rather than just the two fields the URI-and-media-type call used
    to pass down.
    """
    return await asyncio.to_thread(
        _open, export["storage_uri"], export["media_type"], export=export
    )


def _serve_from_index(handle: str) -> Any | None:
    """T54: the cube for ``handle``, before the MCP is asked anything.

    T52 put the cache *behind* ``export_result``, so the key could not be
    computed until the round-trip it was meant to shield you from had already
    completed. The round-trip is cheap; its failure modes are not — an evicted
    source sends ``open_handle`` into ``rematerialize`` -> ``await_retrieval``,
    *minutes* of rebuilding the input to an answer already sitting on local
    disk, and a crash-restarting MCP makes that cube unreachable outright.

    Never raises: this is a shortcut, and a shortcut that can fail a turn is
    worse than no shortcut. Any trouble falls through to the ordinary
    verify-first path below.
    """
    from tta_backend.services import cube_cache

    # Deliberately *not* wrapped in a ``phase_timer`` (T51). A miss here is
    # followed by the real open, and emitting an ``open`` sample for both would
    # put a near-zero duration into that histogram on every single miss —
    # corrupting the distribution T51 exists to make trustworthy, and breaking
    # its one-open-phase-per-open_handle contract. The dedicated
    # ``cube_index_hits/misses`` counters are what make this path measurable.
    try:
        return cube_cache.lookup_by_handle(handle)
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        logger.debug("cube_index_lookup_failed", exc_info=True)
        return None


def _cube_identity(export: dict) -> tuple[str, str, str] | None:
    """``(cache_key, source_identity, local_path)`` for a cubeable export.

    None for anything the interpretation pipeline doesn't touch: a Zarr export
    is already a cube, a Parquet export is a table, and a native-archive export
    never opens at all. Cheap enough (one ``stat`` plus a hash) to recompute at
    both the read and the write call site rather than threading it through
    ``asyncio.to_thread``.

    T54: the identity now prefers what the export says it *delivered*
    (``content_digest``, upstream PRD 023) over the file's mtime and size, so a
    replay that re-fetched the same granules keeps its cube instead of
    rebuilding it — and one that genuinely drifted moves the key and misses.
    """
    from tta_backend.services.cube_cache import cache_key, netcdf_engine_signature, source_identity

    media_type = export.get("media_type") or ""
    mt = media_type.lower()
    if "bundle" not in mt and "netcdf" not in mt:
        return None
    parsed = urlparse(export.get("storage_uri") or "")
    if parsed.scheme != "file":
        return None
    path = url2pathname(parsed.path)
    try:
        source = source_identity(
            path,
            handle=export.get("handle") or "",
            content_digest=export.get("content_digest"),
            partial=bool(export.get("partial")),
        )
    except OSError:
        return None
    return cache_key(source, OPEN_PIPELINE_VERSION, netcdf_engine_signature()), source, path


def _consider_cube(export: dict, ds: Any) -> None:
    """Let the cube cache earn a write off this open (T52).

    Scheduled here, in the async caller, rather than inside ``_open``: that
    runs on a worker thread via ``asyncio.to_thread``, where there is no
    running event loop to attach a background task to. Never raises — a caller
    is mid-answer, and nothing about caching may cost them that.
    """
    from tta_backend.services import cube_cache

    try:
        identity = _cube_identity(export)
        if identity is None:
            return
        key, source, path = identity
        cube_cache.consider_write(
            ds,
            key,
            source=source,
            pipeline_version=OPEN_PIPELINE_VERSION,
            engine=cube_cache.netcdf_engine_signature(),
            # T54: recorded in the manifest and indexed once the cube lands, so
            # the next question about this handle can find it without asking
            # the MCP where its bytes are.
            handle=export.get("handle") or "",
            # The cube is written from a *lazy* Dataset still reading out of
            # the bundle-extract cache, whose pruner sweeps by mtime on a
            # 1-hour TTL and never sees those reads. Pin it for the write.
            pin=extract_cache_dir_for(path),
        )
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        logger.debug("cube_consider_failed", exc_info=True)


async def _export(handle: str, tools: dict[str, BaseTool]) -> dict:
    # T51: the MCP round-trip is its own phase — a turn that felt slow "opening
    # data" may have spent all of it here, waiting on the MCP, and never
    # touched a byte of the file.
    async with phase_timer("export", handle=handle) as timing:
        raw = await tools["export_result"].ainvoke({"handle": handle})
        export = parse_tool_result(raw)
        # The MCP echoes the handle, but the cube key and the index are both
        # derived from it (T54), so pin it to the handle we actually asked
        # about rather than trusting the echo to be the same string.
        export["handle"] = handle
        timing["status"] = export.get("status")
        timing["size_bytes"] = export.get("size_bytes")
    return export


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


def _open(storage_uri: str, media_type: str, *, export: dict | None = None) -> Any:
    """Time the read as the ``open`` phase and dispatch by media type (T51).

    Timed here rather than inside each per-format branch so a Zarr, a bare
    NetCDF and a bundle all land in one comparable distribution. A bundle's
    ``extract`` phase nests *inside* this one: ``open`` is the whole
    file-to-Dataset span, and ``open - extract`` is the per-member open cost.

    ``export`` is the full ready response, carrying the delivered-content
    fields the cube key is derived from (T54). Optional so the direct callers
    that only ever hold a URI and a media type (the live-smoke contract checks)
    keep working — they simply key on filesystem identity, exactly as T52 did.
    """
    # ``members`` defaults to the single-file answer; the bundle path below
    # overwrites it with its own member count, which it already knows -- far
    # cheaper than re-reading the archive here just to count.
    with phase_timer("open", media_type=media_type, members=1) as timing:
        return _open_by_media_type(storage_uri, media_type, timing, export=export)


def _open_by_media_type(
    storage_uri: str, media_type: str, timing: dict | None = None, *, export: dict | None = None
) -> Any:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise OpenHandleError(
            f"Opening non-local URIs (scheme '{parsed.scheme}') is not yet supported: {storage_uri}"
        )
    path = url2pathname(parsed.path)

    mt = (media_type or "").lower()
    if "zarr" in mt:
        import xarray as xr

        # The MCP's transform tools (compare/regrid) ship derived cubes as a
        # *zipped* Zarr store (cube.zarr.zip) under the same "application/zarr"
        # media type as a directory store, so route by the bytes. zarr-python 3
        # dropped v2's ZipStore-from-".zip"-suffix inference: a plain
        # xr.open_zarr(path) treats the zip file as an empty directory and
        # raises "No group found in store ... at path ''" (live TEMPO NO2
        # compare, 2026-07-16). Mirrors the MCP's own reader
        # (tools/_dataio.py): ZipStore reads chunk blobs on demand, and the
        # store is written consolidated=False.
        if _is_zipfile(path):
            import zarr

            return xr.open_zarr(zarr.storage.ZipStore(path, mode="r"), consolidated=False)
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
    export = _export_or_synthetic(export, storage_uri, media_type)
    if "bundle" in mt:
        return _serve_from_cube_or_open(export, timing, lambda: _open_netcdf_bundle(path, timing))
    if "netcdf" in mt:
        return _serve_from_cube_or_open(
            export, timing, lambda: _open_netcdf(path, chunks=_lazy_chunks(), timing=timing)
        )
    raise OpenHandleError(f"Unsupported media_type '{media_type}' for exported handle.")


def _export_or_synthetic(export: dict | None, storage_uri: str, media_type: str) -> dict:
    """The ready export, or the minimum a direct ``_open`` caller implies.

    A synthetic one carries no ``content_digest``, so it keys on filesystem
    identity and is never index-servable — the honest answer for a caller that
    never held a delivered-content record in the first place."""
    return export if export is not None else {"storage_uri": storage_uri, "media_type": media_type}


def _serve_from_cube_or_open(export: dict, timing: dict | None, opener: Any) -> Any:
    """Return the cached cube for this export if there is a good one, else run
    the open pipeline (T52).

    Deliberately *outside* the interpreting functions themselves, so
    :func:`pipeline_source_fingerprint` tracks interpretation only and cache
    plumbing changes don't spuriously invalidate every cube.
    """
    from tta_backend.services.cube_cache import lookup, note_handle, reconcile_handle
    from tta_backend.utils.metrics import record_cache_hit, record_cache_miss

    identity = _cube_identity(export)
    if identity is None:
        return opener()
    key, _source, _path = identity
    handle = export.get("handle") or ""
    # We are on the verified path: an ``export_result`` for this handle has
    # just told us what it currently delivers. If the index still points at a
    # cube built from *different* delivered content, that cube is stale — and
    # because the key is derived from content rather than from a file's mtime,
    # "different" here means the data genuinely changed. Drop it (T54).
    reconcile_handle(handle, key)
    cached = lookup(key)
    if cached is not None:
        record_cache_hit("zarr")
        if timing is not None:
            timing["cube"] = "hit"
        # A cube written before the index existed earns its entry here, on an
        # ordinary verified hit — no migration to run, and the *next* question
        # about it skips the round-trip.
        note_handle(handle, key)
        return cached
    record_cache_miss()
    if timing is not None:
        timing["cube"] = "miss"
    return opener()


def _open_netcdf(path: str, chunks: dict | None = None, *, timing: dict | None = None) -> Any:
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
        return _open_netcdf_bundle(path, timing)

    groups = _open_groups_bounded(path, chunks)
    # GPM-style products (IMERG) carry no netCDF dimension scales, so both
    # engines invent placeholder dims (phony_dim_N) and the science variable
    # opens with no lat/lon identity at all. The files DO declare the real
    # dims per variable (GPM's DimensionNames attr) — honor that before any
    # merging/promotion, same coordinate-discovery doctrine as T24.
    groups = {key: _apply_declared_dimension_names(gds) for key, gds in groups.items()}
    root = groups.pop("/", None)
    if root is not None and root.data_vars and not any(gds.data_vars for gds in groups.values()):
        # Genuinely flat file; nothing nested to merge. (Root data_vars alone
        # don't qualify: GPM 3CMB puts header strings in the root and ALL its
        # science data in nested /Grids groups — the old root-only return
        # silently discarded every science variable, live 2026-07-17.)
        return root

    # Merging by bare name destroys group membership, which is classification
    # evidence (datasets/variable_roles' qa_statistics/geolocation/product
    # priors). Stamp each variable's source group as a ``group_path`` attr so
    # post-open classification sees the same group a describe_dataset
    # inventory's slash-qualified name carries.
    #
    # A group with no data_vars of its own is kept, not skipped. It can still
    # be carrying the coordinates that make a science variable interpretable:
    # TEMPO_O3PROF_L3 declares its per-pixel vertical axes
    # (ozone_profile_pressure in hPa, ozone_profile_altitude in km, both
    # (time, lat, lon, layer)) as CF AUXILIARY COORDINATES in /support_data,
    # so that whole group opens with data_vars empty. Skipping it dropped both
    # axes silently -- the retrieval asked for three variables, Harmony
    # delivered three, and the opened Dataset had one. Nothing raises in that
    # state: ``ozone_profile`` still opens 4-D with ``layer`` intact, there is
    # simply nothing left to say what altitude a layer is at. The root group
    # was already exempted from this skip for the same reason (see the merge
    # below); the root is just the case that happened to be found first.
    group_datasets = []
    for group_key, gds in groups.items():
        group_path = group_key.strip("/")
        if group_path:
            for var in gds.data_vars.values():
                var.attrs.setdefault("group_path", group_path)
        group_datasets.append((group_path, gds))
    if not any(gds.data_vars for _, gds in group_datasets):
        return root if root is not None else xr.Dataset()

    # A leaf name that appears in MORE than one group must not be merged by
    # bare name: xr.merge(compat="override") silently keeps whichever group
    # iterates first — plausible numbers from possibly the wrong variable, on
    # exactly the unregistered products this dynamic path exists to support.
    # Colliding variables keep their group-qualified name ("product/foo",
    # "support_data/foo") instead: an explicit qualified request still
    # resolves exactly, and a bare ambiguous one falls through to
    # AggregationService.to_dataarray's refuse-with-candidates error rather
    # than a silent pick (the T25 doctrine).
    leaf_counts: dict[str, int] = {}
    for _, gds in group_datasets:
        for name in gds.data_vars:
            leaf_counts[name] = leaf_counts.get(name, 0) + 1
    collided = {name for name, count in leaf_counts.items() if count > 1}
    if collided:
        logger.warning(
            "netcdf_group_leaf_name_collision",
            extra={"_event": "netcdf_group_leaf_name_collision", "_names": sorted(collided), "_path": path},
        )
        group_datasets = [
            (
                group_path,
                gds.rename({n: f"{group_path}/{n}" for n in gds.data_vars if n in collided})
                if group_path and collided.intersection(gds.data_vars)
                else gds,
            )
            for group_path, gds in group_datasets
        ]
    group_datasets = [gds for _, gds in group_datasets]

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


def _open_netcdf_bundle(path: str, timing: dict | None = None) -> Any:
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
        if timing is not None:
            # The caller's ``open`` phase context defaults to the single-file
            # answer; this is the one place that knows the real member count.
            timing["members"] = len(names)
        # T51: the unzip is its own phase nested inside ``open``, with a
        # hit/miss flag -- the extract cache makes a repeat open skip it
        # entirely, which one merged distribution would hide.
        with phase_timer("extract", members=len(names)) as extract_timing:
            extract_dir, extract_timing["cached"] = _extract_bundle_cached(zf, path, names)

    # Deliberately NOT pipelined with the open step below (i.e. opening
    # member 1 while member 2 is still extracting), even though both phases
    # are already I/O-bound and bounded by the same setting. _extract_bundle_
    # cached's caching is all-or-nothing (marks the whole directory
    # ``.complete`` and renames it atomically only once every member has
    # extracted); if extraction failed partway *after* some members had
    # already been lazily opened, this function's own cleanup would
    # shutil.rmtree the staging directory while a caller's already-returned,
    # lazy (chunks={}) Dataset still points at files inside it -- a
    # since-deleted-file read failing silently downstream instead of a clean
    # bundle-open error. Extraction completing in full before any open
    # starts is what keeps that failure path safe.
    chunks = _lazy_chunks()
    members = _open_bundle_members_concurrently(extract_dir, names, chunks)

    if len(members) == 1:
        return members[0]
    normalized = [_strip_concat_unsafe_coord_attrs(ds) for ds in members]
    try:
        combined = xr.concat(normalized, dim="time")
    except Exception as exc:
        raise OpenHandleError(
            f"Could not combine the {len(members)} granules in bundle '{path}' onto a "
            f"shared time axis: {exc}"
        )
    return _order_bundle_time(combined, path)


def _run_bounded_failfast(items: list, fn: Any, workers: int) -> list:
    """Run ``fn(item)`` for every item in ``items``, bounded to ``workers``
    concurrent threads, preserving ``items`` order in the returned list.

    Fails fast: once any item raises, every not-yet-started item is cancelled
    rather than left to run to completion — a plain ``ThreadPoolExecutor.map``/
    ``submit`` would submit everything up front and (via ``shutdown(wait=True)``
    on exit) still run every already-queued item before the exception could
    propagate, silently doing the full amount of work anyway on a mid-bundle
    failure and, for an "open" workload, leaking however many file handles
    those extra opens created until the next GC. The first failure (in
    ``items`` order) is what propagates, matching a plain sequential loop's
    behavior. Falls back to a plain sequential loop at ``workers<=1``."""
    from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for future in not_done:
            future.cancel()  # best-effort: only removes not-yet-started work
        for future in futures:
            if future in done:
                future.result()  # raises on the first failure, in items order
        return [future.result() for future in futures]


def _open_bundle_members_concurrently(extract_dir: str, names: list[str], chunks: dict | None) -> list[Any]:
    """Open every bundle member and synthesize its time coordinate, using a
    small bounded thread pool instead of one file at a time.

    Members open *lazily* (``chunks``, see :func:`_lazy_chunks`), so this
    parallelizes only the metadata/header read each open performs — the
    h5netcdf/netCDF4 engines release the GIL for their file I/O, letting
    Python threads overlap the surrounding pure-Python per-member work — not
    any array materialization. RAM stays flat (no more data is pulled into
    memory than the sequential loop pulled) while wall-clock time on a bundle
    with many members drops. The HDF5-touching call itself is still
    serialized through :data:`_hdf5_open_lock` (see its own comment): GIL
    release doesn't establish that the underlying HDF5 C library is safe for
    concurrent calls across file handles, and nothing in this deployment
    proves that build property. Bounded by ``granule_concurrency`` (the same
    setting used for extraction below) so a 50+ granule bundle doesn't spin
    up 50 threads at once; ``names`` order is preserved in the result
    regardless of completion order, so the duplicate-timestamp keep-first
    logic in :func:`_order_bundle_time` stays deterministic."""

    def _open_one(name: str) -> Any:
        member_path = os.path.join(extract_dir, *name.split("/"))
        with _hdf5_open_lock:
            ds = _open_netcdf(member_path, chunks=chunks)
        return _synthesize_member_time_coord(ds)

    workers = min(get_settings().granule_concurrency, len(names))
    return _run_bounded_failfast(names, _open_one, workers)


def _gate_bundle_size(zf: Any, path: str) -> None:
    """Refuse a bundle whose members' total *uncompressed* size exceeds the
    configured cap, before anything is extracted or opened.

    Not a memory bound, despite where it came from. It was written when an
    ungated open could OOM-kill the process, but ``_open_groups_bounded`` now
    holds peak RAM to the chunk ceiling regardless of how large the bundle is,
    so a bundle's total size no longer predicts what opening it costs. What
    this still bounds is *disk* — every member is extracted into the extract
    cache, and nothing else limits that — and the retrieval whose size the
    provider could not estimate, which ``retrieval_composites`` waves through
    on confirmation while naming this gate as the backstop.

    Raising the same structured too_large error the retrieval gates use keeps
    the failure on the T18 deterministic-error surface (the agent relays
    "narrow the request") rather than surfacing as a disk-full OSError."""
    granules = [info for info in zf.infolist() if not info.is_dir()]
    total = sum(info.file_size for info in granules)
    limit = get_settings().bundle_open_max_uncompressed_bytes
    if total <= limit:
        return
    raise MCPToolError(
        CATEGORY_TOO_LARGE,
        f"This result bundle holds {len(granules)} granule file(s) totalling "
        f"~{total:,} bytes uncompressed, over the {limit:,}-byte limit this "
        f"deployment can extract to local disk (bundle: '{path}').",
        suggestion="Narrow the time range, area of interest, or variable list and retrieve again.",
    )


def _open_groups_bounded(path: str, chunks: dict | None) -> dict:
    """Open every group in ``path`` with no dask chunk larger than
    ``open_max_chunk_bytes``.

    This is the pipeline's memory bound. ``chunks={}`` is widely read as "one
    dask chunk per variable per file", but that is not what xarray does: it
    means *inherit the file's HDF5 chunk grid*, and only degrades to one
    whole-array chunk when the variable was written contiguously. Measured::

        contiguous              chunks={} -> ((1,), (600,), (900,))
        hdf5-chunked 1x128x128  chunks={} -> ((1,), (128,...), (128,...))

    So without this function the size of the unit dask allocates — and with
    it every intermediate a reduction holds — is decided by whichever layout
    the provider wrote, an uncontrolled external property. A time-mean over
    contiguous 2950x5771 float64 members, at the shipped 32 MiB budget::

        granules      2      4      8     16
        before     1072   1819   2484   2874  MiB   <- grows with the day
        after        295    409    426    492  MiB   <- levels off

    The "before" column is why a 15-granule day — one ordinary day of TEMPO
    NO2 over North America — SIGKILLed the backend, and why the answer could
    not be a granule cap: 15 granules is the question, not an abuse of it.

    The ceiling has to be applied *here*, at open. Splitting afterwards with
    ``.chunk()`` does not bound anything, because each smaller output chunk
    still has to materialize the whole source chunk it slices — same 8-granule
    reduction, 272.6 MiB opened bounded vs 1222.6 MiB opened whole and
    rechunked after.

    The common case (a provider that chunked its file sensibly) is already
    under budget and returns after the first open, untouched — re-chunking it
    would only straddle the on-disk grid and re-decompress. Both opens are
    header-only, so the second one costs no data read."""
    groups = _narrow_packed_dtypes(_open_all_groups(path, chunks))
    if chunks is None:  # no dask: nothing is chunked, so nothing to bound
        return groups
    spec = _chunk_ceiling_spec(groups, get_settings().open_max_chunk_bytes)
    if spec is None:
        return groups
    for gds in groups.values():
        gds.close()
    return _narrow_packed_dtypes(_open_all_groups(path, spec))


# float64. Nothing in the read/reduce path materializes anything wider, and
# most of it reaches this width whatever the file stored (see
# _chunk_ceiling_spec).
_WIDEST_WORKING_ITEMSIZE = 8

# Integer on-disk widths whose every value is exactly representable in a
# float32 mantissa (24 bits). int32 is deliberately absent: 31 bits of
# magnitude do not fit, so narrowing one would silently round real data.
_FLOAT32_SAFE_PACKED_ITEMSIZE = 2


def _narrow_packed_dtypes(groups: dict) -> dict:
    """Undo CF unpacking's gratuitous widening to float64.

    ``scale_factor``/``add_offset`` are conventionally written float64, and
    xarray takes the decoded dtype from the attribute rather than from the
    stored values — so an int16 variable unpacks to float64 and every
    downstream intermediate inherits four bytes per cell for information the
    file never had. int16 resolves ~4.5 decimal digits; float32 holds ~7.

    This is a pure waste-removal and nothing more. A variable stored natively
    as float64 is left alone: narrowing *that* trades real precision for
    memory, which is a scientific call belonging to whoever reads the numbers,
    not to the opener."""
    import numpy as np

    for gds in groups.values():
        for name, var in gds.data_vars.items():
            stored = var.encoding.get("dtype")
            if stored is None or var.dtype != np.float64:
                continue
            stored = np.dtype(stored)
            if stored.kind not in "iu" or stored.itemsize > _FLOAT32_SAFE_PACKED_ITEMSIZE:
                continue
            gds[name] = var.astype("float32", keep_attrs=True)
            gds[name].encoding = dict(var.encoding)
    return groups


def _chunk_ceiling_spec(groups: dict, limit: int) -> dict | None:
    """``{dim: size}`` that keeps every variable's chunk within ``limit``
    bytes, or None when they all already fit.

    Starts from the chunking the file already has and *halves* the largest
    dimension until the worst variable fits. Halving matters: a size that
    divides the on-disk chunk keeps whole HDF5 chunks inside one dask chunk,
    so bounding never turns one read into several overlapping decompressions.
    Time needs no special case — it is already 1 per bundle member, so the
    same rule leaves it alone and splits the spatial dims instead."""
    variables = [var for gds in groups.values() for var in gds.data_vars.values()]
    if not variables:
        return None
    # Floored at the pipeline's widest working dtype, because a chunk's cost is
    # set by the widest array it passes *through*, not the dtype it settles at.
    # Two ways a narrower stored dtype still costs eight bytes a cell:
    # CF unpacking produces the scale_factor's float64 before
    # _narrow_packed_dtypes casts it down, and the statistics layer promotes to
    # float64 for its accumulators (see aggregation_service.area_weighted_mean).
    # Sizing on the settled dtype handed those variables twice the cells for
    # the same budget -- measured at 8 MiB over a 2048x2048 grid, 56.2 MiB peak
    # for a packed variable and 46.2 MiB for a native float32 one, against
    # 33.2 MiB for the float64 whose chunks the ceiling thought were identical.
    itemsize = max(
        max((var.dtype.itemsize for var in variables), default=4),
        _WIDEST_WORKING_ITEMSIZE,
    )

    spec: dict[Any, int] = {}
    for var in variables:
        chunk_sizes = (
            {dim: max(sizes) for dim, sizes in zip(var.dims, var.chunks)}
            if var.chunks is not None
            else dict(var.sizes)
        )
        for dim, size in chunk_sizes.items():
            spec[dim] = max(spec.get(dim, 0), int(size))

    def worst_bytes() -> int:
        return max(
            math.prod(spec.get(dim, int(var.sizes[dim])) for dim in var.dims) * itemsize
            for var in variables
        )

    if worst_bytes() <= limit:
        return None
    while worst_bytes() > limit:
        widest = max(spec, key=lambda dim: spec[dim])
        if spec[widest] <= 1:
            break  # every dim is already singleton; one cell is over budget
        spec[widest] = max(1, spec[widest] // 2)
    return spec


def _lazy_chunks() -> dict | None:
    """The *request* for dask-backed opening — ``{}`` when dask is installed,
    None otherwise, where xarray's plain lazy arrays materialize whole at the
    first compute and the size gate is the only protection.

    ``{}`` asks xarray to inherit whatever chunking the file already has; it
    does NOT mean "one chunk per variable per file", which is only what it
    degrades to on a contiguously-written variable. What actually bounds the
    result is :func:`_open_groups_bounded`, which every caller goes through —
    this function decides *whether* to use dask, not how much a chunk may
    cost."""
    return None if dask is None else {}


def _extract_bundle_cached(zf: Any, path: str, names: list[str]) -> tuple[str, bool]:
    """Extract ``zf``'s members into a cache entry that outlives this call
    and return ``(directory, was_already_cached)``.

    The hit/miss flag is reported by this function rather than probed by the
    caller because only this function knows which branch it actually took --
    a caller-side check would have to re-derive the cache key and could still
    race the pruner between the check and the extraction (T51).

    Lazily-opened members read these files long after the bundle open
    returns — any compute up to the end of the tool call — and derived
    Datasets keep no reference through which a per-call cleanup could know
    when the last reader is done. So entries live under one process-local
    cache root, keyed by the bundle file's identity (path/size/mtime),
    touched on reuse, and pruned by age on each new extraction
    (:func:`_prune_extract_cache`). Extraction goes into a staging dir
    renamed atomically into place, so a concurrent open of the same bundle
    either wins the rename or adopts the winner's completed entry."""
    import shutil
    import tempfile

    root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
    os.makedirs(root, exist_ok=True)
    _prune_extract_cache(root)

    key = _extract_cache_key(path)
    final_dir = os.path.join(root, key)
    marker = os.path.join(final_dir, _EXTRACT_COMPLETE_MARKER)
    if os.path.exists(marker):
        os.utime(final_dir, None)  # keep a hot entry out of the pruner's reach
        return final_dir, True

    staging = tempfile.mkdtemp(prefix=f"staging-{key}-", dir=root)
    try:
        _extract_members_concurrently(zf, names, staging)
        with open(os.path.join(staging, _EXTRACT_COMPLETE_MARKER), "w"):
            pass
        os.rename(staging, final_dir)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if os.path.exists(marker):
            # Lost the race: a concurrent open of this bundle finished
            # extracting between the marker check and the rename.
            return final_dir, True
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_dir, False


def _extract_cache_key(path: str) -> str:
    stat = os.stat(path)
    return hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]


def extract_cache_dir_for(path: str) -> str:
    """Where :func:`_extract_bundle_cached` would put this bundle's members.

    Derived from the same bundle identity, without opening the archive, so a
    caller that only needs the *location* (T52's cube writer, which pins the
    directory it is lazily reading from) doesn't have to re-derive the key."""
    import tempfile

    return os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME, _extract_cache_key(path))


def _extract_members_concurrently(zf: Any, names: list[str], dest: str) -> None:
    """Extract every member of ``zf`` into ``dest`` using a small bounded
    thread pool instead of one file at a time.

    ``zipfile.ZipFile`` serializes reads of the shared archive file through
    its own internal lock (``ZipFile._lock``, thread-safe since Python 3.6),
    so concurrent ``extract()`` calls from multiple threads decompress
    correctly — each only contends briefly for that lock rather than the
    whole extraction happening sequentially. Bounded by the same
    ``granule_concurrency`` setting as the per-member open above, so a 50+
    granule bundle doesn't spin up 50 threads at once. Fails fast (see
    :func:`_run_bounded_failfast`): a bad member's failure stops any
    not-yet-started extraction rather than extracting every remaining member
    anyway before the caller's cleanup path gets to run."""
    workers = min(get_settings().granule_concurrency, len(names))
    _run_bounded_failfast(names, lambda name: zf.extract(name, dest), workers)


def _entry_size_bytes(path: str) -> int:
    """On-disk size of one cache entry. Missing files are skipped rather than
    raising: a concurrent prune may be removing this entry as we measure it,
    and a size that is slightly stale is harmless where an exception is not."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def _prune_extract_cache(
    root: str,
    ttl_seconds: float = _EXTRACT_CACHE_TTL_SECONDS,
    max_bytes: int | None = None,
) -> None:
    """Bound the extract cache, by age and then by size.

    **Age.** Entries (completed or abandoned staging dirs) untouched for longer
    than the TTL are swept. Reuse touches an entry's mtime, so only bundles
    nothing has opened for a full TTL are removed — far longer than any single
    tool call keeps lazy readers on them.

    **Size.** Whatever the TTL leaves is then trimmed to ``max_bytes``, least
    recently used first. Age alone never bounded this cache: a single bundle may
    extract up to ``bundle_open_max_uncompressed_bytes`` (8 GiB), and every
    bundle opened inside one TTL window is retained, so N busy retrievals hold
    N x 8 GiB with nothing counting the total. It was the only one of the three
    on-disk stores without a cap, and unbounded disk growth is not hypothetical
    in this project's history.

    Deliberately in that order. Evicting by size first would discard entries the
    TTL was about to reclaim for free, and could evict a *hot* entry to make
    room for a stale one that has simply not aged out yet.

    T52: an entry a cube write is currently reading from is pinned and skipped —
    by **both** passes. That writer holds a *lazy* Dataset pointing into this
    directory, and its reads don't touch mtime, so without the pin a long enough
    write can have its own source files deleted underneath it. The size pass
    needs the pin for exactly the same reason the age pass does: a torn write is
    a worse outcome than a full disk, so disk pressure is not a licence to break
    one.
    """
    import shutil
    import time

    from tta_backend.services.cube_cache import is_pinned

    if max_bytes is None:
        max_bytes = get_settings().bundle_extract_cache_max_bytes

    cutoff = time.time() - ttl_seconds
    try:
        entries = list(os.scandir(root))
    except OSError:
        return

    survivors: list[tuple[float, int, str]] = []
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            mtime = entry.stat(follow_symlinks=False).st_mtime
            if is_pinned(entry.path):
                continue  # pinned entries are neither swept nor counted against the cap
            if mtime < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
                continue
            survivors.append((mtime, _entry_size_bytes(entry.path), entry.path))
        except OSError:
            continue

    total = sum(size for _mtime, size, _path in survivors)
    if total <= max_bytes:
        return

    # Oldest access first, which for this cache is oldest mtime: a hit calls
    # os.utime on the entry (see _extract_bundle_cached), so mtime is a genuine
    # last-used stamp rather than a creation date.
    for _mtime, size, path in sorted(survivors):
        if total <= max_bytes:
            break
        shutil.rmtree(path, ignore_errors=True)
        total -= size
        logger.info(
            "extract_cache_evicted",
            extra={"_event": "extract_cache_evicted", "_path": path, "_bytes": size},
        )


def sweep_extract_cache() -> None:
    """Prune the extract cache at startup, before anything extracts.

    The pruner's only other trigger is the beginning of a new extraction, which
    means a backend that stops receiving bundle retrievals never runs it again
    and holds whatever the last busy period left — over a quiet weekend, that is
    disk pinned by a cache nobody is using. ``cube_cache._evict_for`` names this
    exact failure while explaining why the cube store does not copy this design.

    So this is the third of the three startup sweeps, beside
    ``cube_cache.sweep_store`` and ``frame_store.sweep_store``. Never raises:
    reclaiming disk is not worth failing a boot over, and a fresh container has
    no cache directory at all.
    """
    import tempfile

    root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
    if not os.path.isdir(root):
        return
    try:
        _prune_extract_cache(root)
    except Exception:  # noqa: BLE001 — a boot is worth more than the disk
        logger.warning("extract_cache_startup_sweep_failed", exc_info=True)


def extract_cache_size_bytes() -> int:
    """Total on-disk size of the bundle-extract TTL cache
    (:func:`_extract_bundle_cached`), in bytes. Feeds the
    bundle_extract_cache_bytes gauge -- an unexpectedly large cache is a
    plausible, checkable culprit for a memory/disk plateau, distinct from
    Python heap growth. 0 (not an error) when the directory doesn't exist
    yet -- nothing has opened a bundle in this process."""
    import tempfile

    root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue  # pruned/removed concurrently -- not this call's problem
    return total


def _order_bundle_time(ds: Any, path: str) -> Any:
    """Put a concatenated bundle onto a monotonic, duplicate-free time axis.

    The members were concatenated in filename order on the MCP's assumption
    that "names sort chronologically". Two failures follow from trusting it:

    - A provider whose names don't sort against their dates leaves a
      non-monotonic ``time`` axis, and a later ``sel(time=slice(...))``
      silently returns the wrong subset. Sorting by the decoded timestamps
      makes ordering independent of the naming scheme.
    - Overlapping orbits and reprocessed granules can carry *identical*
      timestamps, which double-count that granule in a mean and break
      ``sel(time=...)`` with an opaque non-unique-index error. Keep the first
      occurrence (in the original name order, so the choice is deterministic)
      and disclose the drop in a ``bundle_duplicate_timestamps`` event.

    No-op when there is no ``time`` coordinate to order by."""
    import numpy as np

    if "time" not in ds.coords or "time" not in ds.dims:
        return ds

    # De-duplicate in the current (name) order so keep-first is deterministic,
    # *then* sort — sorting first would make "first" depend on argsort's
    # tie-breaking among equal timestamps.
    times = np.asarray(ds["time"].values)
    _, first_idx = np.unique(times, return_index=True)
    if first_idx.size != times.size:
        keep = np.sort(first_idx)  # preserve name order among the survivors
        logger.info(
            "bundle_duplicate_timestamps",
            extra={
                "_event": "bundle_duplicate_timestamps",
                "_duplicate_count": int(times.size - first_idx.size),
                "_kept": int(first_idx.size),
                "_path": path,
            },
        )
        ds = ds.isel(time=keep)
    return ds.sortby("time")


def _synthesize_member_time_coord(ds: Any) -> Any:
    """Give a bundle member a real, indexed ``time`` coordinate before concat.

    Two shapes of attr-dated L3 product need this — in both, the granule's
    date lives only in the ``RangeBeginningDate``/``RangeBeginningTime``
    global attrs (standard CMR/UMM-G granule temporal metadata):

    - a differently-cased singleton time dimension with no coordinate
      variable (e.g. OMI_MINDS_NO2d's ``Time``), which gets renamed and
      assigned the synthesized timestamp; and
    - no time-like dimension at all (e.g. HAQ TROPOMI monthly L3: 2-D
      lat/lon only), which gets a size-1 indexed ``time`` via expand_dims.

    Left alone, ``xr.concat(dim="time")`` fabricates a brand-new *unindexed*
    stacking dimension in both cases, and every downstream time selection
    dies inside xarray with "no associated coordinate or index" (live
    TROPOMI, 2026-07-16). No-op when ``time`` already exists or the date
    attrs are absent or unparseable. (Extends the MCP's
    ``_synthesize_bundle_time_coord``.)
    """
    import numpy as np

    if "time" in ds.dims:
        return ds
    date = ds.attrs.get("RangeBeginningDate")
    if not date:
        return ds
    time_str = f"{date}T{ds.attrs.get('RangeBeginningTime', '00:00:00').rstrip('Z')}"
    try:
        timestamp = np.datetime64(time_str)
    except ValueError:
        return ds  # malformed attrs — synthesis is best-effort, never fatal
    candidates = [d for d in ds.dims if str(d).lower() == "time" and ds.sizes[d] == 1]
    if candidates:
        ds = ds.rename({candidates[0]: "time"})
        # Preserve a real per-granule time coordinate the differently-cased dim
        # already carried (Finding #15): only OMI_MINDS_NO2d's case -- a Time
        # dim with no coordinate variable -- needs the attr timestamp. When the
        # dim already indexes a real datetime (a genuine overpass time),
        # overwriting it with the attr date's (midnight) stamp would collapse
        # two same-day granules to identical timestamps, and _order_bundle_time
        # would dedup one away -- halving a "daily average" from real distinct
        # observations.
        if not _has_real_time_coord(ds):
            ds = ds.assign_coords({"time": [timestamp]})
        return ds
    return ds.expand_dims(time=[timestamp])


def _has_real_time_coord(ds: Any) -> bool:
    """Whether ``ds`` already carries a genuine datetime ``time`` coordinate
    (an indexed datetime64 with at least one non-NaT value) -- as opposed to a
    bare dimension or an integer index that only *names* time. Synthesis fills
    the latter but must never clobber the former (Finding #15)."""
    import numpy as np

    if "time" not in ds.coords:
        return False
    values = np.asarray(ds["time"].values)
    if not np.issubdtype(values.dtype, np.datetime64):
        return False
    return not bool(np.all(np.isnat(values)))


def _strip_concat_unsafe_coord_attrs(ds: Any) -> Any:
    """Drop ``units``/``calendar`` from TIME coords so cross-granule concat
    doesn't trip xarray's attribute-equality check when granules were written
    against different epochs ("seconds since <this granule's start>"). Time
    values are already decoded to datetime64 per member, so nothing downstream
    needs these attrs to interpret the axis.

    That last sentence is the whole justification, and it is true of time and
    nothing else. Applied to EVERY coordinate — as it was until 2026-08-08 —
    it took ``hPa`` off TEMPO_O3PROF's pressure axis and ``km`` off its
    altitude axis. Units are precisely how a vertical axis is identified
    (``geo_utils.vertical_axis_kind``), so the profile came back with 24 values
    and nothing to plot them against: no error, no empty result, just a missing
    axis. And only on a multi-granule bundle, since a single-member bundle
    returns before this runs — which is why it survived a full synthetic suite
    and surfaced on the first live retrieval.

    A coordinate counts as time if it decoded to datetime64 or still carries a
    CF "<unit> since <epoch>" encoding (an undecoded axis). Nothing else is
    touched.
    """
    import numpy as np

    ds = ds.copy()
    for coord in ds.coords:
        var = ds[coord]
        units = str(var.attrs.get("units", "") or var.encoding.get("units", "") or "")
        is_time = np.issubdtype(var.dtype, np.datetime64) or " since " in units.lower()
        if not is_time:
            continue
        for attr in ("units", "calendar"):
            var.attrs.pop(attr, None)
            var.encoding.pop(attr, None)
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


def _apply_declared_dimension_names(ds: Any) -> Any:
    """Rename engine-invented placeholder dims to the names the file itself
    declares via per-variable ``DimensionNames`` attributes (the GPM
    convention for HDF5 files without netCDF dimension scales).

    Dims whose declared name matches an existing 1-D variable are
    ``swap_dims``-ed so that variable becomes a real dimension coordinate
    (lat/lon/time get their identity back); dims whose declared name is
    otherwise unclaimed are plain-renamed. Opportunistic and safe by
    construction: any disagreement between variables, duplicate target, or
    name collision leaves the dataset exactly as it opened — this must
    never break a product that was already working.
    """
    mapping: dict[str, str] = {}
    for var in ds.variables.values():
        declared = var.attrs.get("DimensionNames") or getattr(var, "encoding", {}).get("DimensionNames")
        if declared is None:
            continue
        if isinstance(declared, bytes):
            declared = declared.decode("utf-8", "replace")
        names = [name.strip() for name in str(declared).split(",")]
        if len(names) != len(var.dims):
            continue
        for dim, name in zip(var.dims, names):
            if not name or name == dim:
                continue
            if mapping.get(dim, name) != name:
                return ds  # two variables disagree on this dim's name
            mapping[dim] = name

    mapping = {dim: name for dim, name in mapping.items() if dim in ds.dims}
    if not mapping:
        return ds
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        return ds  # two dims claim the same name
    if any(name in ds.dims for name in targets):
        return ds  # target name already names a different dim

    swaps: dict[str, str] = {}
    renames: dict[str, str] = {}
    for dim, name in mapping.items():
        declared_var = ds.variables.get(name)
        if declared_var is not None and declared_var.dims == (dim,):
            swaps[dim] = name
        elif name not in ds.variables:
            renames[dim] = name
        else:
            return ds  # name taken by a variable that can't be this dim's axis
    try:
        if swaps:
            ds = ds.swap_dims(swaps)
        if renames:
            ds = ds.rename_dims(renames)
    except (ValueError, KeyError):  # pragma: no cover — belt over the checks above
        return ds
    return ds


def pipeline_source_fingerprint() -> str:
    """A hash of the source of every function that *interprets* an opened file.

    ``_open_netcdf`` and its collaborators do not merely read files: they
    rename declared dimensions, qualify collided leaves, merge groups,
    synthesize time coordinates and promote lat/lon. Each of those has been
    wrong and then fixed, and a cube written before a fix reproduces the old
    interpretation forever with nothing in its output to reveal it.

    :data:`OPEN_PIPELINE_SOURCE_FINGERPRINT` pins this value so that editing
    any of them without bumping :data:`OPEN_PIPELINE_VERSION` fails CI rather
    than silently leaving stale cubes servable.
    """
    import inspect

    digest = hashlib.sha256()
    for fn in _PIPELINE_SOURCE_FUNCTIONS:
        digest.update(inspect.getsource(fn).encode())
    return digest.hexdigest()


def _promote_lat_lon_coords(ds: Any) -> Any:
    """Mark lat/lon-like data variables as coordinates instead of ordinary
    data variables, so they survive variable selection (e.g.
    AggregationService.to_dataarray) instead of being dropped -- or, worse,
    mistaken for the science variable -- when a grouped product splits its
    lon/lat into a sibling subgroup from its science data.

    Identification is the canonical CF-metadata-primary one (T24), so a
    product whose lat/lon carry standard_name/units under a spelling no
    name allowlist would guess is still promoted."""
    from tta_backend.utils.geo_utils import identify_lat, identify_lon

    identified = [identify_lat(ds), identify_lon(ds)]
    to_promote = [name for name in identified if name in ds.data_vars]
    return ds.set_coords(to_promote) if to_promote else ds


# ---------------------------------------------------------------------------
# T52: the open pipeline's interpretation version
# ---------------------------------------------------------------------------
# Participates in every cube cache key (services/cube_cache.py). BUMP THIS
# whenever any function in _PIPELINE_SOURCE_FUNCTIONS below changes the
# *interpretation* of a file -- which, in practice, means any time you touch
# one at all. Every cube built by the old logic is then unreachable by
# construction, including the negative-cache entries that recorded which
# sources the old logic could not cube. The pinned fingerprint below makes
# that enforcement rather than convention: test_cube_cache.py fails if the
# source moves and this doesn't.
#
# Bumped to "2" for _narrow_packed_dtypes. The open-time chunk ceiling landing
# in the same change is NOT why: chunking decides how a read is divided, never
# what it returns, and a cube written before it is bit-identical to one written
# after. The narrowing is the part that qualifies -- a packed int16 variable
# now decodes float32 where it used to decode float64, so a warm cube would
# keep serving the wider dtype for the same source indefinitely. The values
# agree to float32 precision and the old cube is if anything the more precise
# of the two, but "same source, different dtype depending on cache warmth" is
# exactly the silent divergence this version exists to prevent.
#
# Bumped to "3" for keeping coordinate-only groups, and again to "4" for
# stripping units from time coordinates only. Both qualify for the same reason:
# a cube written by the old logic is missing something a reader needs and
# carries no sign of it. "3" lost every auxiliary coordinate that lived in a
# coordinate-only group (for TEMPO_O3PROF, both vertical axes); "4" kept the
# coordinates but stripped the units that identify them as vertical axes at
# all. Either way a warm cube would keep serving an unplottable profile
# forever, with values that look entirely correct.
OPEN_PIPELINE_VERSION = "4"

_PIPELINE_SOURCE_FUNCTIONS = (
    _open_netcdf,
    _open_netcdf_bundle,
    # Decides the dtype a packed variable lands in, so it is interpretation and
    # belongs under the guard. Its siblings from the same change deliberately
    # do NOT: _open_groups_bounded and _chunk_ceiling_spec only decide how a
    # read is divided into tasks, and listing them would evict every cube in
    # the store for a chunk-budget tweak that cannot change a single value.
    _narrow_packed_dtypes,
    _apply_declared_dimension_names,
    _promote_lat_lon_coords,
    _synthesize_member_time_coord,
    _order_bundle_time,
    _strip_concat_unsafe_coord_attrs,
)

# Regenerate with:
#   docker compose --profile test run --build --rm backend-test \
#     python -c "from tta_backend.services.open_handle import pipeline_source_fingerprint; print(pipeline_source_fingerprint())"
# and bump OPEN_PIPELINE_VERSION in the same edit -- EXCEPT when the source
# moved without its meaning moving. The tta_backend.* namespace refactor
# rewrote two function-local import lines inside _open_netcdf_bundle and
# _promote_lat_lon_coords; that shifts the hash while every cube written by the
# old logic stays a correct interpretation, so the fingerprint was re-pinned
# and the version deliberately left at "1". Bumping it would have evicted the
# whole cube cache for a rename. The cube-cache *wiring*
# deliberately lives outside these functions (in _open_by_media_type) so this
# fingerprint tracks interpretation only, and cache plumbing changes don't
# spuriously invalidate every cube.
OPEN_PIPELINE_SOURCE_FINGERPRINT = "0682d593fca8bc02f86dd6850218764f6ec4753d12365722419f69f29b21b5fe"
