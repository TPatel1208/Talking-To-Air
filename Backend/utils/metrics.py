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
    "Retrieval job duration, from the start of the backend-side await through "
    "the job reaching 'ready'. Ready jobs only -- a failed or cancelled job "
    "downloaded nothing, and folding those in would make these percentiles "
    "describe a different population than the name claims.",
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
PROCESS_RSS_BYTES = Gauge(
    "process_rss_bytes",
    "Resident set size of the backend process in bytes.",
)
MATPLOTLIB_OPEN_FIGURES = Gauge(
    "matplotlib_open_figures",
    "Number of matplotlib figures currently open (created but not yet closed).",
)
BUNDLE_EXTRACT_CACHE_BYTES = Gauge(
    "bundle_extract_cache_bytes",
    "Total on-disk size of the bundle-extract TTL cache in bytes.",
)
CUBE_STORE_BYTES = Gauge(
    "cube_store_bytes",
    "Total on-disk size of the T52 Zarr cube cache in bytes.",
)
CUBE_EVICTIONS_TOTAL = Counter(
    "cube_evictions_total",
    "Cubes evicted from the cube store to stay under CUBE_STORE_MAX_BYTES.",
)
PIPELINE_PHASE_DURATION_SECONDS = Histogram(
    "pipeline_phase_duration_seconds",
    "Wall-clock duration of one retrieval/visualization pipeline phase in seconds.",
    ["phase"],
)

# The closed vocabulary of phases, kept here (not in utils.phase_timing) with
# the other labelsets this module pre-declares. Deliberately NOT the same
# vocabulary as config.workflow_stages.ALL_STAGES: that one is a user-facing
# narration contract the frontend's workflow strip and the eval's stage-
# sequence assertions key off, and its ``stage_reached`` events carry
# *cumulative* elapsed since turn start rather than a phase's own span.
PIPELINE_PHASES = ("export", "extract", "open", "crop", "mask", "aggregate", "render")

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


def observe_phase_duration(phase: str, duration_seconds: float) -> None:
    PIPELINE_PHASE_DURATION_SECONDS.labels(phase=phase).observe(duration_seconds)


def record_cache_hit(cache_level: str) -> None:
    CACHE_HITS_TOTAL.labels(cache_level=cache_level).inc()


def record_cache_miss() -> None:
    CACHE_MISSES_TOTAL.inc()


def record_cube_eviction() -> None:
    CUBE_EVICTIONS_TOTAL.inc()


def set_db_pool_connections_active(value: int | float | None) -> None:
    if value is not None:
        DB_POOL_CONNECTIONS_ACTIVE.set(value)


def _current_process_rss_bytes() -> int | None:
    """Current (not peak) resident set size, read from /proc/self/status --
    ru_maxrss is a high-water mark that only ever climbs, which would hide a
    plateau-then-shrink shape; VmRSS reflects the process's actual current
    footprint. None on a non-Linux host (this deploys via Docker/Linux; a
    host run just skips the gauge instead of raising)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def refresh_process_gauges() -> None:
    """Refresh the process-health gauges (RSS, open matplotlib figures,
    bundle extract-cache size) -- called on each /metrics scrape rather than
    a timer, so idle time costs nothing and every scrape is current as of
    itself. Each source is best-effort: a missing/uninstalled dependency
    skips that one gauge rather than failing the whole scrape."""
    rss = _current_process_rss_bytes()
    if rss is not None:
        PROCESS_RSS_BYTES.set(rss)

    try:
        import matplotlib.pyplot as plt

        MATPLOTLIB_OPEN_FIGURES.set(len(plt.get_fignums()))
    except ImportError:
        pass

    try:
        from services.open_handle import extract_cache_size_bytes

        BUNDLE_EXTRACT_CACHE_BYTES.set(extract_cache_size_bytes())
    except ImportError:
        pass

    try:
        from services.cube_cache import store_size_bytes

        CUBE_STORE_BYTES.set(store_size_bytes())
    except ImportError:
        pass


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
    for phase in PIPELINE_PHASES:
        PIPELINE_PHASE_DURATION_SECONDS.labels(phase=phase)


initialize_labelsets()
