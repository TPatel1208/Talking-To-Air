"""
services/overlay_store.py
=========================
Where server-rendered map-overlay PNGs live.

Overlays are served only through the authenticated ``/chart/{id}/overlay.png``
route, so the store sits outside the public output directory. Each entry is
one ``<uuid>.png`` file.

The store is bounded by ``overlay_store_max_bytes`` and evicts the
least-recently-read overlays first. A missing overlay is a normal state, not an
error: the chart payload keeps the grid the frontend's canvas fallback draws
from, so an evicted overlay costs resolution, never the chart.
"""
from __future__ import annotations

import logging
import os
import re
import uuid

from tta_backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_SUFFIX = ".png"
_ENTRY_NAME = re.compile(r"[0-9a-f]{32}\.png")
_STAGING_PREFIX = "staging-"


def write_overlay(png_bytes: bytes) -> str:
    """Store one overlay and return the path recorded in the chart payload.

    The bytes are written to a staging file, fsynced, and renamed into place,
    so an entry name only ever refers to a complete PNG. A process killed
    mid-write leaves a staging file, which is never served and is removed by
    :func:`sweep_store`.
    """
    evict_to_fit(len(png_bytes))
    root = _store_root()
    os.makedirs(root, exist_ok=True)
    name = f"{uuid.uuid4().hex}{_SUFFIX}"
    path = os.path.join(root, name)
    staging = os.path.join(root, f"{_STAGING_PREFIX}{name}")
    try:
        with open(staging, "wb") as f:
            f.write(png_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(staging, path)
    except BaseException:
        _remove(staging)
        raise
    return path


def read_overlay(stored_path: str) -> bytes | None:
    """The overlay's bytes, or None when it is not in the store.

    ``stored_path`` is resolved by its file name inside the store as currently
    configured, not opened as given. A path recorded under a different store
    directory still reads, and a path naming anything other than a store entry
    is never opened.

    A successful read refreshes the entry's mtime, which is the access time
    eviction orders by. Filesystem atime is not used because volumes are
    commonly mounted ``noatime``.
    """
    path = _resolve(stored_path)
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError:
        return None
    _touch(path)
    return content


def sweep_store() -> None:
    """Remove staging files left by writes that never completed. Called at
    startup, when no write can be in flight."""
    try:
        scanned = list(os.scandir(_store_root()))
    except OSError:
        return
    for entry in scanned:
        if entry.name.startswith(_STAGING_PREFIX) and entry.is_file(follow_symlinks=False):
            _remove(entry.path)


def store_size_bytes() -> int:
    """Total bytes of the overlays in the store."""
    return sum(size for _access, size, _path in _entries())


def evict_to_fit(incoming_bytes: int) -> int:
    """Evict least-recently-read overlays until ``incoming_bytes`` fits under
    the cap. Returns how many were evicted."""
    limit = get_settings().overlay_store_max_bytes
    entries = sorted(_entries())  # oldest access first
    total = sum(size for _access, size, _path in entries)
    evicted = 0
    for _access, size, path in entries:
        if total + incoming_bytes <= limit:
            break
        if not _remove(path):
            continue
        total -= size
        evicted += 1
        logger.info(
            "overlay_evicted",
            extra={"_event": "overlay_evicted", "_overlay": os.path.basename(path), "_bytes": size},
        )
    return evicted


def _entries() -> list[tuple[float, int, str]]:
    """``(last_access, bytes_on_disk, path)`` for every overlay in the store."""
    out: list[tuple[float, int, str]] = []
    try:
        scanned = list(os.scandir(_store_root()))
    except OSError:
        return out
    for entry in scanned:
        if not _ENTRY_NAME.fullmatch(entry.name):
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        out.append((stat.st_mtime, stat.st_size, entry.path))
    return out


def _resolve(stored_path: str) -> str | None:
    name = os.path.basename(str(stored_path or "").replace("\\", "/"))
    if not _ENTRY_NAME.fullmatch(name):
        return None
    return os.path.join(_store_root(), name)


def _remove(path: str) -> bool:
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def _touch(path: str) -> None:
    try:
        os.utime(path, None)
    except OSError:
        pass


def _store_root() -> str:
    return get_settings().overlay_store_dir
