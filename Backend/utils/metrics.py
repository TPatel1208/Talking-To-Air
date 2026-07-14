"""Prometheus metrics and legacy in-process counters."""
from __future__ import annotations

from collections import Counter as LegacyCounter
from threading import Lock
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


_LEGACY_COUNTERS: LegacyCounter[str] = LegacyCounter()
_LOCK = Lock()

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Completed HTTP requests.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request wall-clock duration in seconds.",
    ["method", "path"],
)
AGENT_REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Completed subagent calls.",
    ["agent_type", "outcome"],
)
ENVELOPE_SALVAGED_TOTAL = Counter(
    "envelope_salvaged_total",
    "Sub-agent final messages salvaged from prose after failing envelope parsing.",
    ["agent_type"],
)
HARMONY_FETCH_DURATION_SECONDS = Histogram(
    "harmony_fetch_duration_seconds",
    "Harmony job duration from submission through download completion.",
)
HARMONY_TIMEOUTS_TOTAL = Counter(
    "harmony_timeouts_total",
    "Harmony processing timeouts.",
)
CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Cache hits by cache level.",
    ["cache_level"],
)
CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Cache misses that require a remote fetch.",
)
DB_POOL_CONNECTIONS_ACTIVE = Gauge(
    "db_pool_connections_active",
    "Current active PostgreSQL pool connections.",
)

_PROMETHEUS_COUNTER_ALIASES = {
    "harmony_jobs_timed_out": HARMONY_TIMEOUTS_TOTAL,
}


def increment_metric(name: str, amount: int = 1) -> None:
    """Increment a named compatibility counter."""
    if amount <= 0:
        return
    with _LOCK:
        _LEGACY_COUNTERS[name] += amount
    collector = _PROMETHEUS_COUNTER_ALIASES.get(name)
    if collector is not None:
        collector.inc(amount)


def get_metric(name: str) -> int:
    """Return the current value for a named compatibility counter."""
    with _LOCK:
        return _LEGACY_COUNTERS[name]


def snapshot_metrics() -> dict[str, int]:
    """Return a copy of compatibility counters."""
    with _LOCK:
        return dict(_LEGACY_COUNTERS)


def reset_metrics() -> None:
    """Clear compatibility counters. Intended for tests."""
    with _LOCK:
        _LEGACY_COUNTERS.clear()


def render_prometheus_metrics() -> bytes:
    return generate_latest()


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


def observe_http_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=str(status_code)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration_seconds)


def record_agent_request(agent_type: str, outcome: str) -> None:
    AGENT_REQUESTS_TOTAL.labels(agent_type=agent_type, outcome=outcome).inc()


def record_envelope_salvaged(agent_type: str) -> None:
    ENVELOPE_SALVAGED_TOTAL.labels(agent_type=agent_type).inc()


def observe_harmony_fetch(duration_seconds: float) -> None:
    HARMONY_FETCH_DURATION_SECONDS.observe(duration_seconds)


def record_cache_hit(cache_level: str) -> None:
    CACHE_HITS_TOTAL.labels(cache_level=cache_level).inc()


def record_cache_miss() -> None:
    CACHE_MISSES_TOTAL.inc()


def set_db_pool_connections_active(value: int | float | None) -> None:
    if value is not None:
        DB_POOL_CONNECTIONS_ACTIVE.set(value)


def initialize_labelsets() -> None:
    """Create zero-valued time series for expected low-cardinality labels."""
    for method in ("GET", "POST", "DELETE", "OPTIONS"):
        for path in ("/health", "/metrics", "/chat", "/sessions"):
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code="200")
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path)
    for agent_type in ("satellite", "ground_sensor"):
        for outcome in ("success", "failure", "timeout"):
            AGENT_REQUESTS_TOTAL.labels(agent_type=agent_type, outcome=outcome)
    for agent_type in ("earthdata", "ground sensor"):
        ENVELOPE_SALVAGED_TOTAL.labels(agent_type=agent_type)
    for cache_level in ("memory", "zarr", "postgis"):
        CACHE_HITS_TOTAL.labels(cache_level=cache_level)


initialize_labelsets()
