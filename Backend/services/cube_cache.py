"""
services/cube_cache.py
======================
T52 — an L4 cache of the *output* of the open pipeline, stored as Zarr.

``open_handle`` re-does the whole interpretation pipeline on every call —
export round-trip, unzip, N per-member ``_open_netcdf`` (each behind the HDF5
lock, each re-running group discovery, ``DimensionNames`` renaming, leaf-
collision qualification, merging and lat/lon promotion), then concat, dedupe
and sort — against bytes that have not changed. The extract cache saves only
the unzip. This module caches everything above it.

The dependency direction is one-way: ``open_handle`` calls in here; this
module never calls back into ``open_handle``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

_CUBE_DIR_NAME = "cube.zarr"
_MANIFEST_NAME = "manifest.json"
_REFUSED_NAME = "refused.json"
_STAGING_PREFIX = "staging-"

# The escape for ``/`` in a variable name. Collided leaves are renamed to
# "group/leaf" (open_handle's T25 qualification) and ``/`` is Zarr's group
# separator, so those names cannot round-trip as flat variables. Never rely on
# Zarr to preserve them: escape on write, restore from the manifest's name map
# on read.
_SLASH_ESCAPE = "﹨"  # SMALL REVERSE SOLIDUS — not a legal char in CF names

# The ``.encoding`` keys downstream actually reads. AggregationService uses
# *encoding presence* as the signal that xarray decoded the data, in order to
# scale a packed CF ``valid_range`` into physical units; a cube that dropped
# these would hand back physical floats carrying a still-packed valid_range,
# and masking a ~6e17 field against ``<= 30000`` wipes the entire variable
# silently.
_PERSISTED_ENCODING_KEYS = ("scale_factor", "add_offset", "_FillValue")


def source_identity(path: str) -> str:
    """A cheap identity for the exported file at ``path``.

    Filesystem metadata only — path, size, mtime — mirroring the extract
    cache's existing key (``_extract_bundle_cached``). Full content hashing is
    deliberately rejected: reading a 2 GiB bundle to decide whether to read the
    bundle defeats the lookup.

    Defined as a *function*, not an inlined tuple, so that when the MCP grows a
    stronger identifier (an artifact digest, an export UUID, a retrieval
    manifest) adoption is a one-line change here and every caller keeps
    working.
    """
    stat = os.stat(path)
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"


def cache_key(source: str, pipeline_version: str, engine: str) -> str:
    """The cube's identity: the bytes, the interpretation, and the reader.

    ``pipeline_version`` is the heart of the design. ``_open_netcdf`` does not
    merely read files, it *interprets* them, and each of those interpretations
    has been wrong and then fixed (GPM ``DimensionNames`` recovery, TROPOMI
    no-time-dim ``expand_dims``, cased-``Time``-dim coordinate preservation,
    collision qualification). A cube written before a fix and served after it
    reproduces the old interpretation forever with nothing in the output to
    reveal it — so the version participates in the key and every such fix
    invalidates its own stale cubes.

    ``engine`` is in the key because the pipeline is not deterministic across
    environments: ``_open_all_groups`` tries ``h5netcdf`` and falls back to
    ``netcdf4``, the two decode attributes differently, and which one runs
    depends on what is installed rather than on the code. A version bump
    cannot catch that.
    """
    payload = f"{source}\x00{pipeline_version}\x00{engine}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def netcdf_engine_signature() -> str:
    """Which NetCDF readers this environment actually has, in the order
    ``_open_all_groups`` tries them.

    The engine that ends up reading a given file isn't knowable before opening
    it, but the key needs a token now. The *environment's* reader inventory is
    the right granularity: it is what differs between a host checkout and the
    Docker image, it is deterministic, and it is free to compute. The engine a
    cube was actually written with is recorded in its manifest for debugging.
    """
    import importlib.util

    return "+".join(
        name for name in ("h5netcdf", "netcdf4") if importlib.util.find_spec(name) is not None
    ) or "none"


# ---------------------------------------------------------------------------
# Store layout
# ---------------------------------------------------------------------------
# <root>/<key>/cube.zarr      the cube itself
# <root>/<key>/manifest.json  written LAST; its presence is the completion
#                             marker, exactly as `.complete` is for the
#                             extract cache
# <root>/<key>/refused.json   a negative-cache entry: this source cannot
#                             satisfy the round-trip contract under this
#                             pipeline version, so don't retry it every open


def _store_root() -> str:
    return get_settings().cube_store_dir


def _entry_dir(key: str) -> str:
    return os.path.join(_store_root(), key)


def _manifest_path(key: str) -> str:
    return os.path.join(_entry_dir(key), _MANIFEST_NAME)


def _cube_path(key: str) -> str:
    return os.path.join(_entry_dir(key), _CUBE_DIR_NAME)


def _read_manifest(key: str) -> dict | None:
    try:
        with open(_manifest_path(key), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _escape_name(name: str) -> str:
    return name.replace("/", _SLASH_ESCAPE)


def _capture_encoding(ds: Any) -> dict[str, dict]:
    """The subset of each variable's ``.encoding`` downstream reads.

    Captured *before* the write and restored *after* the read rather than left
    on the Dataset: leaving ``scale_factor``/``add_offset`` in ``.encoding``
    would make ``to_zarr`` re-pack the already-decoded physical values, and
    stripping it without persisting it is the silent-variable-wipe bug this
    exists to prevent.
    """
    captured: dict[str, dict] = {}
    for name, var in list(ds.variables.items()):
        enc = {k: var.encoding[k] for k in _PERSISTED_ENCODING_KEYS if k in var.encoding}
        if enc:
            captured[str(name)] = {k: _jsonable(v) for k, v in enc.items()}
    return captured


def _jsonable(value: Any) -> Any:
    """numpy scalars -> plain Python, so the manifest is real JSON."""
    item = getattr(value, "item", None)
    return item() if callable(item) and getattr(value, "shape", ()) == () else value


def _stat_sweep(cube_dir: str) -> tuple[int, int]:
    """(file count, total bytes) under ``cube_dir`` — an ``os.scandir`` walk,
    no data read. This is what catches a missing or truncated chunk, which a
    metadata-only ``open_zarr`` cannot."""
    files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(cube_dir):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
            files += 1
    return files, total


def write_cube(ds: Any, key: str, *, source: str = "", pipeline_version: str = "", engine: str = "") -> bool:
    """Write ``ds`` to the store under ``key``. Returns whether a cube landed.

    Staging dir -> fsync -> atomic rename -> manifest, adapting
    ``_extract_bundle_cached``'s pattern including its lost-the-race adoption
    branch. ``fsync`` before the rename matters: on a crash a rename can be
    durable while the contents are not.
    """
    root = _store_root()
    os.makedirs(root, exist_ok=True)
    final_dir = _entry_dir(key)
    if os.path.exists(_manifest_path(key)):
        return True  # already cubed by a concurrent writer
    if is_refused(key):
        return False  # known-unsafe under this pipeline version; don't retry

    settings = get_settings()
    projected = int(getattr(ds, "nbytes", 0) or 0)
    if projected > settings.cube_write_max_bytes:
        # A size policy, not a contract failure: the source is perfectly fine,
        # it is merely too big to cube safely. No negative-cache entry — this
        # check is O(1) and re-running it on every open costs nothing.
        logger.info(
            "cube_write_skipped_too_large",
            extra={
                "_event": "cube_write_skipped_too_large",
                "_cube_key": key,
                "_projected_bytes": projected,
                "_limit_bytes": settings.cube_write_max_bytes,
            },
        )
        return False
    evict_to_fit(projected)

    staging = tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{key}-", dir=root)
    try:
        encoding_map = _capture_encoding(ds)
        name_map = _build_name_map(ds)
        writable = _prepare_for_write(ds, name_map)
        cube_dir = os.path.join(staging, _CUBE_DIR_NAME)
        # consolidated=True: we own these stores, unlike the MCP's
        # consolidated=False ones that _open special-cases.
        writable.to_zarr(cube_dir, mode="w", consolidated=True)
        _fsync_tree(staging)
        files, total = _stat_sweep(cube_dir)
        manifest = {
            "source_identity": source,
            "pipeline_version": pipeline_version,
            "engine": engine,
            "variables": sorted(str(n) for n in ds.data_vars),
            "name_map": name_map,
            "encoding_map": encoding_map,
            "chunk_file_count": files,
            "total_bytes": total,
        }
        with open(os.path.join(staging, _MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        os.rename(staging, final_dir)
    except OSError:
        # Environmental, not the data's fault (a full volume, a vanished
        # staging dir). Blacklisting a perfectly cubeable source over a
        # transient disk error would keep it uncached until the next pipeline-
        # version bump, so this propagates instead of refusing.
        shutil.rmtree(staging, ignore_errors=True)
        if os.path.exists(_manifest_path(key)):
            return True  # lost the race; the winner's cube is just as good
        raise
    except Exception as exc:  # noqa: BLE001 — a contract failure, see _refuse
        shutil.rmtree(staging, ignore_errors=True)
        _refuse(key, f"{type(exc).__name__}: {exc}")
        return False
    return True


class CubeContractError(RuntimeError):
    """This Dataset cannot be represented faithfully as a cube."""


def _build_name_map(ds: Any) -> dict[str, str]:
    """``{escaped: original}`` for every slash-qualified variable name.

    Raises if escaping would make two variables collide: silently merging a
    collided leaf back into one variable is exactly the failure T25's qualified
    names exist to prevent, and it would be invisible in the output.
    """
    name_map = {_escape_name(str(n)): str(n) for n in ds.variables if "/" in str(n)}
    existing = {str(n) for n in ds.variables}
    for escaped, original in name_map.items():
        if escaped in existing:
            raise CubeContractError(
                f"escaping {original!r} collides with existing variable {escaped!r}"
            )
    return name_map


def _refuse(key: str, reason: str) -> None:
    """Record a negative-cache entry so a known-unsafe source is not retried on
    every open. Keyed like any other entry, so it carries the pipeline version
    — a later fix un-blacklists it with nothing to remember to clear."""
    logger.warning(
        "cube_write_refused",
        extra={"_event": "cube_write_refused", "_cube_key": key, "_reason": reason},
    )
    entry = _entry_dir(key)
    try:
        os.makedirs(entry, exist_ok=True)
        with open(os.path.join(entry, _REFUSED_NAME), "w", encoding="utf-8") as f:
            json.dump({"reason": reason}, f)
    except OSError:
        pass  # a refusal we couldn't record just means we retry — never fatal


def is_refused(key: str) -> bool:
    return os.path.exists(os.path.join(_entry_dir(key), _REFUSED_NAME))


def _prepare_for_write(ds: Any, name_map: dict[str, str]) -> Any:
    """Escape slash-qualified names, strip every ``.encoding`` key, and chunk.

    Stripping encoding is what makes ``to_zarr`` write the decoded physical
    values it was handed rather than re-packing them (or tripping over the
    source file's chunk/compressor encoding). It operates on a shallow copy,
    whose Variables carry their own ``attrs``/``encoding`` dicts — the caller
    is still holding this Dataset to answer the current question, and a
    background writer that stripped *its* encoding would trigger landmine 1
    live rather than on a later read.
    """
    writable = ds.copy(deep=False)
    if name_map:
        writable = writable.rename({original: escaped for escaped, original in name_map.items()})
    for name in list(writable.variables):
        writable[name].encoding = {}
    return _apply_cube_chunks(writable)


# T51 measured that the T50 crop does NOT push down to a hyperslab read: with
# ``chunks={}`` (one chunk per variable per file), a 700x reduction in cells
# reduced bought 3.47x wall-clock and *zero* reduction in bytes read -- dask
# materializes the whole granule chunk and the crop only trims it in memory.
# So the cube's headroom is the 86.8 MiB, not the 0.17 s, and a spatially
# monolithic cube would make every regional question pay the full continental
# read all over again. Hence: moderate spatial chunks, and as much of the time
# axis as fits in the budget (regional subset + reduce over time is the shape
# every question about this data takes).
_CUBE_SPATIAL_CHUNK = 256
_CUBE_TARGET_CHUNK_BYTES = 8 * 1024 ** 2


def _apply_cube_chunks(ds: Any) -> Any:
    try:
        return ds.chunk(_cube_chunks(ds))
    except (ImportError, ValueError):  # no dask, or a shape it can't rechunk
        return ds


def _cube_chunks(ds: Any) -> dict:
    """``{dim: chunk_size}`` for the cube: spatial dims capped, time as long as
    the byte budget allows.

    Chunking finer than the data is pure overhead (more files, more metadata,
    no smaller read), so a dim shorter than the cap is left in one piece.
    """
    itemsize = max((var.dtype.itemsize for var in ds.data_vars.values()), default=4)
    spatial = {
        dim: min(int(size), _CUBE_SPATIAL_CHUNK)
        for dim, size in ds.sizes.items()
        if str(dim).lower() != "time"
    }
    cells = 1
    for size in spatial.values():
        cells *= size
    budget = max(1, _CUBE_TARGET_CHUNK_BYTES // max(1, cells * itemsize))
    chunks = dict(spatial)
    for dim, size in ds.sizes.items():
        if str(dim).lower() == "time":
            chunks[dim] = min(int(size), budget)
    return chunks


def _fsync_tree(path: str) -> None:
    """fsync every file then the directories, so a crash cannot leave a
    durably-renamed entry pointing at contents that never reached disk."""
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            _fsync(os.path.join(dirpath, name), directory=False)
        _fsync(dirpath, directory=True)


def _fsync(path: str, *, directory: bool) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return  # best-effort durability; never fail a write over it
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def lookup(key: str) -> Any | None:
    """Return the cached Dataset for ``key``, or None on a miss.

    Transparently fallible, in the same self-heal posture ``open_handle``
    already takes toward a bad export: *any* failure on this path — a manifest
    that won't parse, an integrity check that doesn't add up, a store that
    won't open — drops the entry, logs a named event and returns a miss, so the
    caller falls through to the lazy path and the turn still answers. A cache
    that can refuse is strictly better than one that can be subtly wrong.
    """
    manifest = _read_manifest(key)
    if manifest is None:
        return None
    try:
        if not validate(key, manifest):
            _invalidate(key, "integrity_check_failed")
            return None
        ds = _read_cube(key, manifest)
        _touch(key)
        return ds
    except Exception as exc:  # noqa: BLE001 — the whole point is to never propagate
        _invalidate(key, f"{type(exc).__name__}: {exc}")
        return None


def validate(key: str, manifest: dict | None = None) -> bool:
    """Whether the on-disk cube still matches what its manifest recorded.

    A file-count and total-bytes sweep via ``os.scandir``, reading no data.
    Cheap enough to run on every hit, and it catches exactly what a
    metadata-only ``open_zarr`` cannot: a chunk file deleted or truncated
    underneath the store, which otherwise only surfaces much later inside a
    downstream compute.
    """
    manifest = manifest if manifest is not None else _read_manifest(key)
    if manifest is None:
        return False
    files, total = _stat_sweep(_cube_path(key))
    return files == manifest.get("chunk_file_count") and total == manifest.get("total_bytes")


def _invalidate(key: str, reason: str) -> None:
    logger.warning(
        "cube_read_failed",
        extra={"_event": "cube_read_failed", "_cube_key": key, "_reason": reason},
    )
    shutil.rmtree(_entry_dir(key), ignore_errors=True)


def sweep_store() -> None:
    """Remove orphaned staging directories and manifest-less entries.

    Called at startup: a crash mid-write leaves a staging dir behind, and an
    entry whose manifest never landed is by definition incomplete. Neither is
    ever served (the manifest is the completion marker), so this reclaims
    space rather than fixing correctness.
    """
    root = _store_root()
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        complete = os.path.exists(os.path.join(entry.path, _MANIFEST_NAME))
        # A negative-cache entry is complete by construction — it has no
        # manifest because it has no cube, and sweeping it would make every
        # restart retry a source already known unsafe.
        refused = os.path.exists(os.path.join(entry.path, _REFUSED_NAME))
        if entry.name.startswith(_STAGING_PREFIX) or not (complete or refused):
            shutil.rmtree(entry.path, ignore_errors=True)


def _entries() -> list[tuple[float, int, str]]:
    """``(last_access, bytes, key)`` for every complete cube in the store.

    Sizes come from each manifest's recorded ``total_bytes`` rather than a walk
    — the per-hit sweep already validates that number, so re-deriving it here
    would double the store's I/O on every write for no new information."""
    out: list[tuple[float, int, str]] = []
    try:
        scanned = list(os.scandir(_store_root()))
    except OSError:
        return out
    for entry in scanned:
        if not entry.is_dir(follow_symlinks=False) or entry.name.startswith(_STAGING_PREFIX):
            continue
        manifest = _read_manifest(entry.name)
        if manifest is None:
            continue
        try:
            last_access = entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue
        out.append((last_access, int(manifest.get("total_bytes") or 0), entry.name))
    return out


def store_size_bytes() -> int:
    """Total bytes of complete cubes in the store. Feeds the
    ``cube_store_bytes`` gauge -- "the cache isn't helping" and "the cache is
    never populated" are different problems with different fixes."""
    return sum(size for _access, size, _key in _entries())


def evict_to_fit(incoming_bytes: int) -> int:
    """Evict coldest-first until ``incoming_bytes`` fits under the cap.
    Returns how many cubes were evicted.

    LRU by last access, run *before* each write. Deliberately not the extract
    cache's age-only TTL: reads there don't touch mtime, so a hot entry would
    be evicted at one hour, and its sweep only fires on new extractions, so a
    cache that stops growing never prunes at all.
    """
    from utils.metrics import record_cube_eviction

    limit = get_settings().cube_store_max_bytes
    entries = sorted(_entries())  # oldest access first
    total = sum(size for _access, size, _key in entries)
    evicted = 0
    for _access, size, key in entries:
        if total + incoming_bytes <= limit:
            break
        shutil.rmtree(_entry_dir(key), ignore_errors=True)
        total -= size
        evicted += 1
        record_cube_eviction()
        logger.info(
            "cube_evicted",
            extra={"_event": "cube_evicted", "_cube_key": key, "_bytes": size},
        )
    return evicted


def _touch(key: str) -> None:
    """Record a hit as an access, so LRU means what it says. The extract
    cache's own pruner learned this the hard way: reads that don't touch mtime
    make a hot entry indistinguishable from an abandoned one."""
    try:
        os.utime(_entry_dir(key), None)
    except OSError:
        pass


def _read_cube(key: str, manifest: dict) -> Any | None:
    import xarray as xr

    ds = xr.open_zarr(_cube_path(key), consolidated=True)
    name_map = manifest.get("name_map") or {}
    if name_map:
        ds = ds.rename({escaped: original for escaped, original in name_map.items()})
    for name, enc in (manifest.get("encoding_map") or {}).items():
        if name in ds.variables:
            ds[name].encoding.update(enc)
    return ds


# ---------------------------------------------------------------------------
# Write policy: background, and earn your way in
# ---------------------------------------------------------------------------

# How long a queued write waits for the process to go quiet before giving up.
# Bounded so a permanently busy backend does not accumulate write tasks (each
# holding a lazy Dataset, and with it a pinned extract dir) forever.
_TURN_IDLE_WAIT_SECONDS = 1800.0

# Open counts live in-process, keyed by source_identity. ``--workers 1`` makes
# that correct today, and a restart losing counts costs at most one extra
# uncached turn -- Postgres would add a hot-path write for durability nobody
# needs. Swept by TTL so a long-lived process doesn't accumulate one entry per
# source it ever saw.
_OPEN_COUNT_TTL_SECONDS = 24 * 3600.0
_open_counts: dict[str, tuple[int, float]] = {}

# Keys with a write already scheduled or running, so two misses in the same
# turn queue one cube, not two.
_writes_in_flight: set[str] = set()

# Extract directories a write is currently reading from. _prune_extract_cache
# skips these: it sweeps by directory mtime on a 1-hour TTL and reads do not
# touch mtime, so a long write can otherwise have its source files deleted out
# from under its lazy Dataset.
_pinned_paths: set[str] = set()

_write_semaphore: Any = None
_active_turns = 0
_idle_event: Any = None


def note_open(source: str) -> int:
    """Count an open of ``source`` and return the running total."""
    import time

    now = time.monotonic()
    _sweep_open_counts(now)
    count, _first = _open_counts.get(source, (0, now))
    _open_counts[source] = (count + 1, now)
    return count + 1


def _sweep_open_counts(now: float) -> None:
    stale = [src for src, (_c, seen) in _open_counts.items() if now - seen > _OPEN_COUNT_TTL_SECONDS]
    for src in stale:
        del _open_counts[src]


def consider_write(
    ds: Any,
    key: str,
    *,
    source: str,
    pipeline_version: str = "",
    engine: str = "",
    pin: str | None = None,
) -> Any | None:
    """Schedule a background cube write if this source has earned one.

    Returns the ``asyncio.Task`` (so callers and tests can await it) or None
    when no write was scheduled. Never raises: a caller is in the middle of
    answering a question, and nothing about caching may cost them that.

    Cubing is triggered on the **second** open, never the first, so one-shot
    questions pay nothing. The task cubes the Dataset that second open
    *already produced* -- ``to_zarr`` on that object, not a re-derivation -- so
    the cube is pipeline output by construction and the ``pipeline_version``
    stamped in its manifest is provably the version that produced those arrays.
    """
    import asyncio

    try:
        if note_open(source) < 2:
            return None
        if key in _writes_in_flight or is_refused(key) or os.path.exists(_manifest_path(key)):
            return None
        _writes_in_flight.add(key)
        return asyncio.create_task(
            _write_task(ds, key, source=source, pipeline_version=pipeline_version, engine=engine, pin=pin)
        )
    except RuntimeError:
        # No running loop (a sync call site). Caching is best-effort.
        _writes_in_flight.discard(key)
        return None


async def _write_task(
    ds: Any,
    key: str,
    *,
    source: str,
    pipeline_version: str,
    engine: str,
    pin: str | None,
) -> None:
    import asyncio

    try:
        # Never two cubes building at once: concurrency buys nothing (same
        # disk) and multiplies the memory risk on a --workers 1 process.
        async with _semaphore():
            if not await _await_quiet_turn():
                logger.info(
                    "cube_write_abandoned_busy",
                    extra={"_event": "cube_write_abandoned_busy", "_cube_key": key},
                )
                return
            with pin_path(pin):
                await asyncio.to_thread(
                    write_cube, ds, key, source=source, pipeline_version=pipeline_version, engine=engine
                )
    except Exception as exc:  # noqa: BLE001 — a background task must never surface
        logger.warning(
            "cube_write_failed",
            extra={"_event": "cube_write_failed", "_cube_key": key, "_reason": f"{type(exc).__name__}: {exc}"},
        )
    finally:
        _writes_in_flight.discard(key)


def _semaphore() -> Any:
    import asyncio

    global _write_semaphore
    if _write_semaphore is None:
        _write_semaphore = asyncio.Semaphore(1)
    return _write_semaphore


def _idle() -> Any:
    import asyncio

    global _idle_event
    if _idle_event is None:
        _idle_event = asyncio.Event()
        _idle_event.set()
    return _idle_event


async def _await_quiet_turn() -> bool:
    """Block until no turn is in flight. False if the wait timed out.

    A full cube read/compress/write on the cold path would tax the turn the
    researcher is actually watching and re-enter the memory regime the lazy
    open exists to escape (the bundle-open OOM). Note the guarantee is "never
    *start* during a turn": a write already under way runs to completion, since
    a thread doing ``to_zarr`` cannot be interrupted. Its cost is bounded by
    the per-cube write cap and the semaphore above; a pausable writer is a
    later refinement.
    """
    import asyncio

    try:
        await asyncio.wait_for(_idle().wait(), timeout=_TURN_IDLE_WAIT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        return False
    return True


@contextlib.contextmanager
def active_turn():
    """Mark a chat turn as in flight for the duration of the block."""
    global _active_turns
    _active_turns += 1
    _idle().clear()
    try:
        yield
    finally:
        _active_turns -= 1
        if _active_turns <= 0:
            _active_turns = 0
            _idle().set()


def turn_is_active() -> bool:
    return _active_turns > 0


@contextlib.contextmanager
def pin_path(path: str | None):
    """Protect ``path`` from the bundle-extract cache's pruner for the
    duration of the block (see :data:`_pinned_paths`)."""
    if path is None:
        yield
        return
    _pinned_paths.add(os.path.abspath(path))
    try:
        yield
    finally:
        _pinned_paths.discard(os.path.abspath(path))


def is_pinned(path: str) -> bool:
    return os.path.abspath(path) in _pinned_paths


def reset_for_test() -> None:
    """Drop this module's process-lifetime state. A test seam, in the shape of
    ``jobs_service.clear_terminal_status_cache``."""
    global _write_semaphore, _idle_event, _active_turns
    _open_counts.clear()
    _writes_in_flight.clear()
    _pinned_paths.clear()
    _write_semaphore = None
    _idle_event = None
    _active_turns = 0
