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
import threading
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from config.settings import get_settings
from config.workflow_stages import STAGE_OPEN
from earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError, parse_tool_result
from services.retrieval_composites import await_retrieval
from utils.phase_timing import phase_timer
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
    # T51: the MCP round-trip is its own phase — a turn that felt slow "opening
    # data" may have spent all of it here, waiting on the MCP, and never
    # touched a byte of the file.
    async with phase_timer("export", handle=handle) as timing:
        raw = await tools["export_result"].ainvoke({"handle": handle})
        export = parse_tool_result(raw)
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


def _open(storage_uri: str, media_type: str) -> Any:
    """Time the read as the ``open`` phase and dispatch by media type (T51).

    Timed here rather than inside each per-format branch so a Zarr, a bare
    NetCDF and a bundle all land in one comparable distribution. A bundle's
    ``extract`` phase nests *inside* this one: ``open`` is the whole
    file-to-Dataset span, and ``open - extract`` is the per-member open cost.
    """
    # ``members`` defaults to the single-file answer; the bundle path below
    # overwrites it with its own member count, which it already knows -- far
    # cheaper than re-reading the archive here just to count.
    with phase_timer("open", media_type=media_type, members=1) as timing:
        return _open_by_media_type(storage_uri, media_type, timing)


def _open_by_media_type(storage_uri: str, media_type: str, timing: dict | None = None) -> Any:
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
    if "bundle" in mt:
        return _open_netcdf_bundle(path, timing)
    if "netcdf" in mt:
        return _open_netcdf(path, timing=timing)
    raise OpenHandleError(f"Unsupported media_type '{media_type}' for exported handle.")


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

    groups = _open_all_groups(path, chunks)
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
    group_datasets = []
    for group_key, gds in groups.items():
        if not gds.data_vars:
            continue
        group_path = group_key.strip("/")
        if group_path:
            for var in gds.data_vars.values():
                var.attrs.setdefault("group_path", group_path)
        group_datasets.append((group_path, gds))
    if not group_datasets:
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
