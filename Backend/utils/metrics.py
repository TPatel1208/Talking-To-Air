"""Small in-process counters for lightweight operational metrics."""
from __future__ import annotations

from collections import Counter
from threading import Lock


_COUNTERS: Counter[str] = Counter()
_LOCK = Lock()


def increment_metric(name: str, amount: int = 1) -> None:
    """Increment a named counter."""
    if amount <= 0:
        return
    with _LOCK:
        _COUNTERS[name] += amount


def get_metric(name: str) -> int:
    """Return the current value for a named counter."""
    with _LOCK:
        return _COUNTERS[name]


def snapshot_metrics() -> dict[str, int]:
    """Return a copy of all current counters."""
    with _LOCK:
        return dict(_COUNTERS)


def reset_metrics() -> None:
    """Clear counters. Intended for tests."""
    with _LOCK:
        _COUNTERS.clear()
