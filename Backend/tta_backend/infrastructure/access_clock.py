"""Access stamps for the stores that order their entries by last access through file times.

A file clock can give a read and the write after it the same time, and a store
then breaks the tie some other way, by key or size, so the entry just read can
be the one evicted. Stamps from here are strictly increasing within a process,
so the more recent access is always newer.
"""
from __future__ import annotations

import os
import threading
import time

# NTFS keeps file times in 100 ns units, so stamps a microsecond apart stay distinct on it.
STAMP_STEP_NS = 1_000

_lock = threading.Lock()
_last_ns = 0


def access_stamp_ns() -> int:
    """Now, or just after the last stamp this process gave."""
    global _last_ns
    with _lock:
        _last_ns = max(time.time_ns(), _last_ns + STAMP_STEP_NS)
        return _last_ns


def stamp_access(path: str) -> None:
    """Set ``path``'s access and modification times to a fresh stamp. Raises OSError."""
    stamp = access_stamp_ns()
    os.utime(path, ns=(stamp, stamp))
