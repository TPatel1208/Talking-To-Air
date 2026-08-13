"""
services/frame_store.py
=======================
T59 Phase 4 — where a frame stack lives once it has been computed.

D13's split, which is the whole design of this module: the frame **axis** and
every per-frame disclosure go in the chart's jsonb row; only the float32
**values** go in a bounded LRU blob store. The split is by durability, and it
buys three things:

* the slider is fully labeled and functional *before* the blob lands, because
  the axis arrives with the chart payload;
* eviction *degrades* rather than breaks, precisely because the axis survives
  in Postgres when the values do not;
* NaN needs no sentinel — it is a float32 bit pattern in the blob, and it never
  has to survive JSON, which has no way to spell it.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from tta_backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_BLOB_NAME = "frames.f32.gz"
_MANIFEST_NAME = "manifest.json"
_STAGING_PREFIX = "staging-"

# Measured on real harvested stacks (Phase 2 §2): level 9 buys 0.0-0.2% for
# 1.1-1.9x the CPU, level 1 costs 1.0-1.6% for 9-29% less. 6 is the nginx and
# zlib default and is correct here for the same reason it is correct there.
_GZIP_LEVEL = 6

#: The values, laid out as ``(1 + n_frames, ny, nx)`` float32: the period mean
#: is plane 0 — D5's "frame 0" — and the frames follow in axis order.
_PERIOD_INDEX = 0


@dataclass(frozen=True)
class FrameBlob:
    """A stored stack, as it goes on the wire: still compressed.

    The store never decodes what it holds. Serving the gzipped bytes straight
    through is not only cheaper, it is the only form in which the entry's
    integrity has been checked — decompressing to re-compress would put a
    second encoder between the checked bytes and the client.
    """

    gzipped: bytes
    #: The digest of exactly these bytes, recorded at write time and verified
    #: on the way out. It doubles as the ETag: the blob is immutable (D8), so
    #: an entity tag derived from its content can never go stale, and it costs
    #: nothing extra because the integrity check computes it anyway.
    etag: str


def store_frame_stack(stack: Any, *, pipeline_version: str) -> dict:
    """Store ``stack``'s values and return the block for the chart's jsonb row.

    The return value is the *whole* frontend contract minus the field itself:
    the labeled axis, the grid the values are laid out on, the pooled scale and
    every per-frame disclosure D10 requires. It carries ``_key`` only if the
    values actually landed — a caller holding a block without one has a chart
    with a drawn, labeled, unscrubbable axis, which is also exactly what an
    evicted chart looks like later.
    """
    block = _axis_block(stack, pipeline_version=pipeline_version)
    key = write_frames(stack, pipeline_version=pipeline_version)
    if key is not None:
        block["_key"] = key
    return block


def write_frames(stack: Any, *, pipeline_version: str) -> str | None:
    """Persist ``stack``'s float32 planes. Returns the entry key, or None.

    Never raises. A frame stack is a rendering convenience, and nothing about
    storing one may cost a caller the chart they asked for — the same posture
    ``_render_and_store_overlay`` takes toward a failed PNG render.

    **The key is minted, not derived from the content.** A content-addressed
    key would let two charts share one entry, and charts are ownership-scoped
    (every route goes through ``_get_owned_chart``): sharing a blob between
    them would manufacture exactly the cross-user cache this feature's
    ``private`` caching says does not exist. Identity is instead recorded *in*
    the entry — the pipeline version it was produced under, and the shape it
    holds — and checked on the way out.
    """
    try:
        payload = _pack(stack)
        # Before the write, and against the REAL bytes, which are known here
        # because the entry is already compressed: no estimate, no optimistic
        # and conservative pair of guesses, nothing to self-correct on the
        # following write.
        evict_to_fit(len(payload))
        key = uuid.uuid4().hex
        _write_entry(
            key, payload, pipeline_version=pipeline_version, shape=_shape(stack),
        )
        return key
    except Exception:  # noqa: BLE001 — degrade to a chart with no scrubber
        logger.warning("frame_write_failed", exc_info=True)
        return None


def read_frames(key: str, *, pipeline_version: str) -> FrameBlob | None:
    """Serve the stored blob for ``key``, or None on a miss.

    Transparently fallible, in ``cube_cache``'s posture: no failure on this
    path ever propagates. A store that can refuse is strictly better than one
    that can be subtly wrong, and here "subtly wrong" means a field someone
    would read. The caller degrades to a labeled, unscrubbable axis, which is a
    state D8 already requires the frontend to handle.

    An entry that fails *integrity* — a digest that does not add up, a blob
    that will not open, a superseded pipeline version — is dropped as well as
    missed, because it can only go on failing while charging the cap for the
    privilege. A missing or unparseable manifest is only a miss: the manifest
    is the completion marker, so its absence is indistinguishable from a write
    still in flight, and :func:`sweep_store` reclaims the genuinely abandoned
    ones at startup where that ambiguity does not exist.
    """
    if not key:
        return None
    manifest = _read_manifest(key)
    if manifest is None:
        return None
    # D8's guard, and the reason it is a *drop* rather than a refusal: the
    # version only moves forward, so an entry produced under a superseded
    # interpretation of the source files can never be served again. Keeping it
    # would charge the store for bytes no request can ever reach.
    if str(manifest.get("pipeline_version") or "") != pipeline_version:
        _invalidate(key, "pipeline_version_superseded")
        return None
    try:
        with open(_blob_path(key), "rb") as f:
            payload = f.read()
        if not _intact(payload, manifest):
            _invalidate(key, "integrity_check_failed")
            return None
    except OSError as exc:
        _invalidate(key, f"{type(exc).__name__}: {exc}")
        return None
    _touch(key)
    return FrameBlob(gzipped=payload, etag=str(manifest.get("blob_sha256")))


def _intact(payload: bytes, manifest: dict) -> bool:
    """Whether these bytes are the bytes that were written.

    Length first because it is free and catches the common failure — a write
    cut short by a full volume, a container killed mid-write — and then the
    digest, which is the only check that sees a bit flip that left the length
    alone. Both run on bytes already in hand for serving, so neither costs a
    second read, and neither decompresses: re-inflating to verify would put a
    decoder between the checked bytes and the client for no added confidence.

    Truncation is the case worth naming. Half a stack still gunzips to
    *something* — whole frames of float32 that reshape cleanly and render as a
    plausible field with the tail of the scrub silently absent.
    """
    if len(payload) != int(manifest.get("blob_bytes") or -1):
        return False
    return hashlib.sha256(payload).hexdigest() == manifest.get("blob_sha256")


def _invalidate(key: str, reason: str) -> None:
    logger.warning(
        "frame_read_failed",
        extra={"_event": "frame_read_failed", "_frame_key": key, "_reason": reason},
    )
    shutil.rmtree(_entry_dir(key), ignore_errors=True)


def sweep_store(current_pipeline_version: str) -> None:
    """Remove everything the store can never serve. Called at startup.

    Three kinds of dead weight: a staging directory a crash left behind, an
    entry whose manifest never landed (incomplete by definition, since the
    manifest is the completion marker), and entries produced under a superseded
    pipeline version, which :func:`read_frames` refuses and which would
    otherwise sit counting against the cap until LRU ground them out one write
    at a time.

    ``current_pipeline_version`` is a parameter rather than an import for the
    same reason ``cube_cache.sweep_store`` takes one: ``open_handle`` owns that
    constant, and this module stores what it is told without interpreting it.
    Required rather than defaulted, so a caller cannot quietly skip the reclaim
    by forgetting it.
    """
    try:
        entries = list(os.scandir(_store_root()))
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        manifest = _read_manifest(entry.name)
        superseded = (
            manifest is not None
            and str(manifest.get("pipeline_version") or "") != current_pipeline_version
        )
        if entry.name.startswith(_STAGING_PREFIX) or manifest is None:
            shutil.rmtree(entry.path, ignore_errors=True)
        elif superseded:
            logger.info(
                "frame_reclaimed_stale_pipeline_version",
                extra={
                    "_event": "frame_reclaimed_stale_pipeline_version",
                    "_frame_key": entry.name,
                    "_was_version": manifest.get("pipeline_version"),
                    "_current_version": current_pipeline_version,
                    "_bytes": manifest.get("blob_bytes"),
                },
            )
            shutil.rmtree(entry.path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Eviction: an LRU sweeper, and nothing else
# ---------------------------------------------------------------------------


def _entries() -> list[tuple[float, int, str]]:
    """``(last_access, bytes_on_disk, key)`` for every complete entry.

    **Real bytes on disk**, walked, not ``values.nbytes``. ``cube_store``
    measured its write cap against the uncompressed in-memory footprint while
    accounting the store in disk bytes, and that mismatch charged cubes up to
    5x what they cost. Frames are stored gzipped — measured at x1.36 on a dense
    regional field and x3.25 full-domain — so the same mistake here would be
    the same mistake three times over on exactly the sparse stacks that cost
    least.
    """
    out: list[tuple[float, int, str]] = []
    try:
        scanned = list(os.scandir(_store_root()))
    except OSError:
        return out
    for entry in scanned:
        if not entry.is_dir(follow_symlinks=False) or entry.name.startswith(_STAGING_PREFIX):
            continue
        if _read_manifest(entry.name) is None:
            continue  # incomplete: the manifest is the completion marker
        try:
            out.append((entry.stat(follow_symlinks=False).st_mtime,
                        _entry_bytes(entry.path), entry.name))
        except OSError:
            continue
    return out


def _entry_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def store_size_bytes() -> int:
    """Total bytes of complete entries in the store."""
    return sum(size for _access, size, _key in _entries())


def evict_to_fit(incoming_bytes: int) -> int:
    """Evict coldest-first until ``incoming_bytes`` fits under the cap.
    Returns how many entries were evicted.

    LRU by last access rather than by age: a stack being scrubbed right now is
    the one that must survive, and an age TTL cannot tell it from an abandoned
    one.

    An LRU sweeper is the *whole* policy — no per-entry cap, deliberately.
    ``cube_write_max_store_fraction`` exists to stop an entry big enough to
    evict the whole store to fit itself and then be evicted by the next write;
    D7 caps frames at 60 and D5 caps cells at 20,000, so an entry is at most
    4.58 MiB raw against a 1 GiB store and that thrash cannot occur. Adding a
    cap anyway would be cargo-culting the store this one borrowed from. The
    corollary, stated so nobody has to rediscover it: an entry larger than the
    entire store would clear the store and still not fit — unreachable at 0.4%
    of the cap, and the shape of the bound, not a case to code around.
    """
    limit = get_settings().frame_store_max_bytes
    entries = sorted(_entries())  # oldest access first
    total = sum(size for _access, size, _key in entries)
    evicted = 0
    for _access, size, key in entries:
        if total + incoming_bytes <= limit:
            break
        shutil.rmtree(_entry_dir(key), ignore_errors=True)
        total -= size
        evicted += 1
        logger.info(
            "frame_entry_evicted",
            extra={"_event": "frame_entry_evicted", "_frame_key": key, "_bytes": size},
        )
    return evicted


def _touch(key: str) -> None:
    """Record a hit as an access, so LRU means what it says. The extract
    cache's pruner learned this the hard way: reads that don't touch mtime make
    a hot entry indistinguishable from an abandoned one."""
    try:
        os.utime(_entry_dir(key), None)
    except OSError:
        pass


def _pack(stack: Any) -> bytes:
    """The planes, gzipped, as one buffer.

    Compressed streaming-wise rather than by concatenating the arrays first:
    the period plane and the frames are written into the same compressor, so
    nothing ever holds a second copy of the stack. ``mtime=0`` because a gzip
    header carries a timestamp by default, and a byte-identical stack must
    produce byte-identical output for the ETag to mean anything.
    """
    import numpy as np

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=_GZIP_LEVEL, mtime=0) as gz:
        for plane in (stack.period_values[np.newaxis, ...], stack.values):
            gz.write(np.ascontiguousarray(plane, dtype="float32").tobytes())
    return buffer.getvalue()


def _write_entry(
    key: str, payload: bytes, *, pipeline_version: str, shape: list[int],
) -> None:
    """Stage, fsync, rename, then manifest — the cube store's write protocol.

    ``fsync`` before the rename matters: on a crash a rename can be durable
    while the contents are not. The manifest is written inside the staging
    directory and lands with it, so its presence is the completion marker and a
    half-written entry is never served.
    """
    root = _store_root()
    os.makedirs(root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{key}-", dir=root)
    try:
        blob = os.path.join(staging, _BLOB_NAME)
        with open(blob, "wb") as f:
            f.write(payload)
        manifest = {
            "pipeline_version": pipeline_version,
            "shape": shape,
            "blob_bytes": len(payload),
            "blob_sha256": hashlib.sha256(payload).hexdigest(),
        }
        with open(os.path.join(staging, _MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        _fsync_tree(staging)
        os.rename(staging, _entry_dir(key))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _store_root() -> str:
    return get_settings().frame_store_dir


def _entry_dir(key: str) -> str:
    return os.path.join(_store_root(), key)


def _blob_path(key: str) -> str:
    return os.path.join(_entry_dir(key), _BLOB_NAME)


def _manifest_path(key: str) -> str:
    return os.path.join(_entry_dir(key), _MANIFEST_NAME)


def _read_manifest(key: str) -> dict | None:
    try:
        with open(_manifest_path(key), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fsync_tree(path: str) -> None:
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            _fsync(os.path.join(dirpath, name))
        _fsync(dirpath)


def _fsync(path: str) -> None:
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


def _shape(stack: Any) -> list[int]:
    n_frames, ny, nx = (int(v) for v in stack.values.shape)
    return [n_frames + 1, ny, nx]


def _axis_block(stack: Any, *, pipeline_version: str) -> dict:
    """Everything about a frame stack except its values.

    Rounded and coerced to plain Python here rather than at the call site: this
    lands in jsonb, and a numpy scalar or a float32 is not JSON-serializable.
    """
    # Local, not module-scope: ``frame_stack`` imports xarray, and ``api.py``
    # imports this module at module scope — so a top-level import here would
    # pull the whole scientific stack into API startup, which is exactly the
    # cost ``warmup.warm_render_path`` exists to schedule deliberately.
    from tta_backend.preprocessing.frame_stack import POOLED_SCALE_BASIS

    return {
        "frames": [
            {
                "t_start": frame.t_start,
                "t_end": frame.t_end,
                "n_granules": int(frame.n_granules),
                "valid_fraction": float(frame.valid_fraction),
                # Absent, never zero: "QA rejected everything" and "QA never
                # ran" are different facts about an empty-looking frame.
                "qa_pass_rate": (
                    None if frame.qa_pass_rate is None else float(frame.qa_pass_rate)
                ),
                "statistics": {k: float(v) if k != "count" else int(v)
                               for k, v in frame.statistics.items()},
            }
            for frame in stack.frames
        ],
        # How to read the blob, for a consumer that has one. The realized cell
        # count, never D5's 20,000 ceiling: integer coarsening quantizes, so a
        # real grid lands at 70-96% of what was asked for.
        "shape": _shape(stack),
        "dtype": "float32",
        "period_index": _PERIOD_INDEX,
        "lats": [round(float(v), 6) for v in stack.lats],
        "lons": [round(float(v), 6) for v in stack.lons],
        "cells_per_frame": int(stack.cells_per_frame),
        "cadence": stack.cadence,
        "tier": stack.tier,
        "buckets_per_frame": int(stack.buckets_per_frame),
        "coarsen_k": [int(k) for k in stack.coarsen_k],
        "delta": stack.delta,
        # The second delta, beside the first and never folded into it. `delta`
        # asks whether the coarser temporal aggregation is a different
        # measurement (natively, so a +1.2 and a -1.2 in one block cannot
        # cancel); this asks whether the two planes in THIS blob are each
        # other's average. Phase 8 measured 3.74% and 3.28% for the same chart
        # — neither bounds the other, and the smaller one was the one on
        # screen.
        "frame_grid_delta": stack.frame_grid_delta,
        "value_range": (
            None if stack.value_range is None
            else [float(v) for v in stack.value_range]
        ),
        # What that range actually is, in one self-describing line, for the
        # same reason ``QA_PASS_RATE_BASIS`` exists and the same way the D16
        # delta already carries its own basis: a 2-98 pooled across every frame
        # AND the period mean, at native resolution before the block mean, is
        # not inferable from two numbers — and D9 turns on this scale being the
        # scrubber's alone, never the Map tab's.
        "scale_basis": POOLED_SCALE_BASIS,
        # D8: frames are never served or rebuilt across a version change, and
        # the stamp is what makes that checkable on a chart written months ago.
        "pipeline_version": pipeline_version,
    }
