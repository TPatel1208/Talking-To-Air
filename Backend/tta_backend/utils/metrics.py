"""Prometheus metrics and legacy in-process counters."""
from __future__ import annotations

import ctypes
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
# Admission control (services/admission.py). These exist to make N -- how many
# heavy reductions may hold memory at once -- a measured number rather than the
# extrapolation it currently is: it was derived by arithmetic from an 8 GiB
# ceiling, while every other memory constant in this codebase was measured on
# real data.
ADMISSION_IN_FLIGHT = Gauge(
    "admission_in_flight",
    "Heavy reductions currently holding a memory permit.",
)
ADMISSION_QUEUED = Gauge(
    "admission_queued",
    "Callers waiting for a memory permit.",
)
# Labelled by surface because the two outcomes differ in cost, not just in
# origin: a shed export is a retry the user barely notices, while a shed chat
# turn is a lost turn that has already spent its LLM tokens. A single total
# would average the urgent case into the routine one.
ADMISSION_SHED_TOTAL = Counter(
    "admission_shed_total",
    "Requests refused because the admission queue was full.",
    ["surface"],
)
ADMISSION_WAIT_SECONDS = Histogram(
    "admission_wait_seconds",
    "Time spent waiting for a memory permit, including admissions that did not wait.",
)
# The measurement that would let N be re-derived. Sampled per section rather
# than at scrape time (see current_process_rss_bytes) and bucketed generously:
# the interesting range spans an idle 200 MiB floor to the 1,342 MB largest
# admitted reduction, and the question a reader asks is which order of
# magnitude the process is sitting at while N are resident, not its exact
# bytes.
ADMISSION_RSS_BYTES = Histogram(
    "admission_rss_bytes",
    "Process RSS observed at the end of a heavy section (a lower bound on its peak).",
    buckets=(
        256 * 1024 ** 2, 512 * 1024 ** 2, 768 * 1024 ** 2,
        1024 * 1024 ** 2, 1536 * 1024 ** 2, 2048 * 1024 ** 2,
        3072 * 1024 ** 2, 4096 * 1024 ** 2, 6144 * 1024 ** 2,
        8192 * 1024 ** 2, float("inf"),
    ),
)
PROCESS_RSS_BYTES = Gauge(
    "process_rss_bytes",
    "Resident set size of the backend process in bytes.",
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
# T54, counted separately from cube_hits/cube_misses on purpose: a cube hit
# says the cache saved the open pipeline, an *index* hit says it also saved the
# MCP round-trip that used to gate the lookup. Those are different wins, and
# collapsing them would make the reorder's own contribution unmeasurable.
CUBE_INDEX_HITS_TOTAL = Counter(
    "cube_index_hits_total",
    "Cubes served straight from the handle index, with no export_result round-trip.",
)
CUBE_INDEX_MISSES_TOTAL = Counter(
    "cube_index_misses_total",
    "Handle-index lookups that fell through to the verify-first path.",
)
CUBE_INDEX_INVALIDATIONS_TOTAL = Counter(
    "cube_index_invalidations_total",
    "Indexed cubes dropped because a verified export delivered different content.",
)
# Token accounting for provider chat completions (utils/llm_timing.py). The
# ``llm_call`` histogram made model *latency* visible; this makes model *cost*
# visible, which nothing in this backend could measure before -- so "what does
# a turn cost" and "is the constant prefix hitting the provider's prompt cache"
# were both unanswerable.
#
# ``kind`` is one of:
#   input       -- total prompt tokens billed for the call
#   output      -- completion tokens, thinking/reasoning tokens included
#   cache_read  -- the SUBSET of ``input`` that hit the provider's prompt cache
#
# cache_read is a *breakdown* of input, never additional to it (that is
# LangChain's UsageMetadata contract: input_token_details decomposes
# input_tokens). So the cache hit rate is cache_read/input, and summing the
# three kinds does not give a meaningful total -- an alert that does so is
# double-counting the cached tokens.
#
# Deliberately NOT pre-declared in initialize_labelsets() like every other
# labelset here: the model label is configuration (config/settings.py resolves
# it from the environment per agent), not a closed vocabulary, so hardcoding
# ids here would duplicate _VETTED_MODELS and go stale the first time one is
# swapped. A model with no series has genuinely never been called.
# ``agent_type`` uses the SAME vocabulary as AGENT_REQUESTS_TOTAL above
# ("satellite", "ground_sensor", plus "supervisor" which never makes a subagent
# call), so tokens-per-agent-call joins across the two without a relabel. The
# codebase carries two other spellings for the same concept -- "earthdata" in
# the trim middleware, "ground sensor" in ENVELOPE_SALVAGED_TOTAL -- and this
# deliberately matches neither of those: the join that pays is against
# agent_requests_total.
#
# The label exists because ``model`` cannot substitute for it. Both sub-agents
# default to the same model id (gemini-3.1-flash-lite), so without this the
# earthdata agent's spend and the ground agent's spend are one indivisible
# series -- and they have very different prefixes, which is exactly what
# anyone reading this metric is trying to compare.
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Provider tokens billed for chat completions, by model, agent_type and "
    "kind. 'cache_read' is the subset of 'input' that hit the prompt cache, "
    "not an addition to it.",
    ["model", "agent_type", "kind"],
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
PIPELINE_PHASES = (
    "export", "extract", "open", "crop", "mask", "aggregate", "render",
    # T55: the QA pass-rate counting inside the masking path. Its own phase,
    # not folded into "mask", because it adds an eager reduction to a
    # lazily-opened bundle and what it costs is exactly what must stay visible.
    "qa_pass_rate",
    # The two phases above the data pipeline, added after a 2026-08-07 trace
    # of a 372s plot turn left 107s (29%) attributable to nothing: every
    # phase here is data work, so model latency and graph overhead had no
    # series to land in and the gap read as a hole in the timeline.
    #
    # "llm_call" is one provider request as LangChain sees it, from
    # on_chat_model_start to on_llm_end. Provider-side retries live *inside*
    # that span (ChatGoogleGenerativeAI defaults to max_retries=6 with
    # backoff, and retries beneath the LangChain seam raise no callback), so
    # a retry storm shows up here as one long call rather than many -- which
    # is exactly the shape that was invisible before.
    "llm_call",
    # "llm_retry_sleep" is the cumulative backoff a LangChain-driven retry
    # has slept when it fires. Separate from "llm_call" because it is a
    # component *of* one: a long call with retry sleep is rate limiting, a
    # long call without it is the provider genuinely taking that long.
    "llm_retry_sleep",
    # The provenance build, split out 2026-08-07 after llm_call/agent_step
    # narrowed a 870s turn's unexplained 303s (34.8%) to the window between the
    # "render" timer closing and the tool returning. The only work there is
    # _attach_reproducibility, and nothing separated its two expensive halves.
    #
    # "provenance" is that whole span; "evidence" and "related_variables" are
    # the two passes inside it that read the *unaggregated* Dataset -- every
    # granule, every band -- rather than the reduced array the chart was drawn
    # from. Timed separately because the fix differs: evidence computes
    # per-band area-weighted stats, related_variables only classifies an
    # inventory, and which one dominates decides which to change.
    "provenance",
    "evidence",
    "related_variables",
    # "frames" is T59's bucketed reduction, the one behind a map's scrubber.
    # Its own phase for the same reason "qa_pass_rate" is: it is a whole extra
    # graph walk over a lazily-opened bundle -- Phase 3 measured 9.2 s against
    # a 12-15 s open+mask on a real regional TEMPO bundle -- so a plot that
    # gains a scrubber gains that time, and a feature's cost has to be visible
    # rather than smeared across "aggregate".
    "frames",
    # "agent_step" is one LangGraph superstep: the wall-clock gap between
    # consecutive `updates` chunks, attributed to the node that produced the
    # later one. Deliberately not folded into "llm_call" -- it also covers
    # checkpointer writes and graph overhead, so the *difference* between the
    # two is what separates "the model was slow" from "we were slow around it".
    "agent_step",
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


def observe_phase_duration(phase: str, duration_seconds: float) -> None:
    PIPELINE_PHASE_DURATION_SECONDS.labels(phase=phase).observe(duration_seconds)


def record_llm_tokens(model: str, agent_type: str, kind: str, tokens: int) -> None:
    """Add ``tokens`` to the ``model``/``agent_type``/``kind`` series.

    Called only for a count the provider actually reported -- see
    utils.llm_timing._usage_context for why an absent count must not be
    recorded as a zero.
    """
    LLM_TOKENS_TOTAL.labels(model=model, agent_type=agent_type, kind=kind).inc(tokens)


def record_cache_hit(cache_level: str) -> None:
    CACHE_HITS_TOTAL.labels(cache_level=cache_level).inc()


def record_cache_miss() -> None:
    CACHE_MISSES_TOTAL.inc()


def record_cube_eviction() -> None:
    CUBE_EVICTIONS_TOTAL.inc()


def record_cube_index_hit() -> None:
    CUBE_INDEX_HITS_TOTAL.inc()


def record_cube_index_miss() -> None:
    CUBE_INDEX_MISSES_TOTAL.inc()


def record_cube_index_invalidation() -> None:
    CUBE_INDEX_INVALIDATIONS_TOTAL.inc()


def set_db_pool_connections_active(value: int | float | None) -> None:
    if value is not None:
        DB_POOL_CONNECTIONS_ACTIVE.set(value)


def set_admission_occupancy(in_flight: int, queued: int) -> None:
    """Publish how many heavy reductions are resident and how many wait.

    One function for both gauges because they are read together and are only
    meaningful together: in_flight alone cannot distinguish a backend at its
    limit with nobody waiting (healthy, well-sized) from one at its limit with
    a queue behind it (undersized), and those call for opposite responses.
    """
    ADMISSION_IN_FLIGHT.set(in_flight)
    ADMISSION_QUEUED.set(queued)


def record_admission_shed(surface: str) -> None:
    ADMISSION_SHED_TOTAL.labels(surface=surface).inc()


def observe_admission_wait(seconds: float) -> None:
    ADMISSION_WAIT_SECONDS.observe(seconds)


def observe_admission_rss(rss_bytes: int) -> None:
    ADMISSION_RSS_BYTES.observe(rss_bytes)


def current_process_rss_bytes() -> int | None:
    """Public name for the RSS reader, for callers outside a /metrics scrape.

    ``refresh_process_gauges`` reads RSS when Prometheus asks, which is the
    right cadence for a process-health gauge and the wrong one for admission: a
    reduction that climbs to 1.3 GB and releases between two scrapes never
    appears, and those are the peaks that cause an OOM. services/admission.py
    samples per section instead, and goes through here rather than reaching
    for the private reader.
    """
    return _current_process_rss_bytes()


# Named so the parse can be exercised on a host that has no /proc -- the field
# this reads is the one thing about the Linux path that can silently regress
# (VmRSS is current, the VmHWM two lines above it is the high-water mark), and
# a guard that only runs in the container is no guard on the dev machine.
_LINUX_STATUS_PATH = "/proc/self/status"


def _linux_rss_bytes() -> int | None:
    """VmRSS from /proc/self/status, in bytes. None off Linux."""
    try:
        with open(_LINUX_STATUS_PATH) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    """Win32 PROCESS_MEMORY_COUNTERS. Field order is the ABI contract — the
    struct is matched by layout, not by name, so these must not be reordered."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_rss_bytes() -> int | None:
    """WorkingSetSize for this process, in bytes. None off Windows.

    The working set is the physical memory currently resident for the process —
    the Windows analogue of VmRSS, and what psutil reports as ``rss`` on this
    platform. Deliberately **not** ``PeakWorkingSetSize``, which is the same
    high-water-mark trap as ``ru_maxrss``: it only ever climbs, so it would hide
    exactly the plateau-then-shrink shape this gauge exists to make visible.

    Uses ``K32GetProcessMemoryInfo`` (exported from kernel32 since Windows 7)
    rather than psapi's ``GetProcessMemoryInfo``, so there is no second DLL to
    resolve. ``GetCurrentProcess()`` is a pseudo-handle that needs no close.
    """
    if not hasattr(ctypes, "WinDLL"):
        return None
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory_info = kernel32.K32GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            ctypes.c_uint32,
        ]
        get_memory_info.restype = wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        if not get_memory_info(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.WorkingSetSize)
    except (AttributeError, OSError, ValueError):
        return None


def _psutil_rss_bytes() -> int | None:
    """Current RSS via psutil, if it happens to be installed. None otherwise.

    Not a dependency — the two supported environments (Docker/Linux for the
    deployment, Windows for local development) are both covered without it, and
    adding a C extension to the runtime image to satisfy a platform nobody
    deploys on would be the wrong trade. This exists so a contributor on macOS
    or BSD, where there is no stdlib way to read current RSS, gets a working
    gauge by installing psutil rather than a silently dead one.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 — a metrics read must never break a scrape
        return None


def _current_process_rss_bytes() -> int | None:
    """Current (not peak) resident set size of **this** process, in bytes.

    Per-process by design: each process exposes its own /metrics, so scraping an
    aggregate here would double-count against Prometheus's own aggregation. If
    the backend is ever run with multiple uvicorn workers, each reports its own
    RSS under its own target, which is what makes a per-worker leak visible at
    all — an aggregate would average it away.

    Peak is deliberately avoided in every reader below: ``ru_maxrss`` and
    ``PeakWorkingSetSize`` only ever climb, which would hide the
    plateau-then-shrink shape (T45) this gauge was added to make falsifiable.

    Returns None only on a platform none of the readers cover, in which case
    ``refresh_process_gauges`` leaves the gauge alone rather than raising or
    publishing a fabricated zero — a missing series is honest, a zero one is not,
    and a zero is exactly how the Linux-only version hid on Windows.
    """
    for reader in (_linux_rss_bytes, _windows_rss_bytes, _psutil_rss_bytes):
        rss = reader()
        if rss is not None and rss > 0:
            return rss
    return None


def refresh_process_gauges() -> None:
    """Refresh the process-health gauges (RSS, bundle extract-cache size,
    cube-store size) -- called on each /metrics scrape rather than a timer,
    so idle time costs nothing and every scrape is current as of itself.
    Each source is best-effort: a missing/uninstalled dependency skips that
    one gauge rather than failing the whole scrape.

    There is deliberately no open-figure gauge here. T45 added one, reading
    ``plt.get_fignums()``, back when the PNG export built figures through
    ``pyplot`` and could leak them. ``export_service`` and ``plotting`` --
    the only modules in this process that build figures -- now use the
    object-oriented API (``Figure`` + ``FigureCanvasAgg``), which files
    nothing in pyplot's process-global registry, so that gauge could only
    ever report 0. Reading it also meant importing ``pyplot`` here, which
    was the one thing left in the whole process still pulling in the very
    registry the render path was moved off. The regression it watched for
    is pinned in CI instead, by
    ``test_export_event_loop_offload.NoProductionModuleImportsPyplotTests``.
    """
    rss = _current_process_rss_bytes()
    if rss is not None:
        PROCESS_RSS_BYTES.set(rss)

    try:
        from tta_backend.services.open_handle import extract_cache_size_bytes

        BUNDLE_EXTRACT_CACHE_BYTES.set(extract_cache_size_bytes())
    except ImportError:
        pass

    try:
        from tta_backend.services.cube_cache import store_size_bytes

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
