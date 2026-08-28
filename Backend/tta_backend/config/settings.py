"""Centralized runtime configuration for the backend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlsplit

from dotenv import load_dotenv

from tta_backend.utils.connector_crypto import ConnectorCryptoError, build_multi_fernet

_VALID_FETCH_MODES = {"auto", "harmony", "opendap", "s3"}
_VALID_LOG_FORMATS = {"text", "json"}

# Model ids reach the provider SDK unvalidated (see config/model_factory.py),
# so a wrong one is not discoverable until it 404s on the first call that uses
# it -- and for the supervisor that is every turn, making it an outage rather
# than a degraded path. Boot is the cheaper place to learn it.
#
# Adding an id here is the record that someone confirmed it exists on the
# provider *and* on the billing tier being deployed. A provider absent from
# this map is not gated at all, so bringing one under the gate stays a
# deliberate act instead of silently starting to fail boots.
_VETTED_MODELS: dict[str, frozenset[str]] = {
    "google": frozenset({
        "gemini-3.7-flash",
        "gemini-3.1-flash-lite",
    }),
}


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded once from environment at startup/import."""

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-3.7-flash"))
    ground_agent_model: str = field(
        default_factory=lambda: os.getenv(
            "GROUND_AGENT_MODEL",
            "gemini-3.1-flash-lite",
        )
    )
    earthdata_agent_model: str = field(
        default_factory=lambda: os.getenv(
            "EARTHDATA_AGENT_MODEL",
            os.getenv("SATELLITE_AGENT_MODEL", "gemini-3.1-flash-lite"),
        )
    )
    supervisor_model_provider: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_MODEL_PROVIDER", "google")
    )
    earthdata_agent_provider: str = field(
        default_factory=lambda: os.getenv("EARTHDATA_AGENT_PROVIDER", "google")
    )
    ground_agent_provider: str = field(
        default_factory=lambda: os.getenv("GROUND_AGENT_PROVIDER", "google")
    )
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))

    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: _int_env("DB_PORT", 5432))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "talking_to_air_memory")))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", os.getenv("POSTGRES_USER", "postgres")))
    db_password: str | None = field(default_factory=lambda: os.getenv("DB_PASSWORD"))
    db_pool_min_size: int = field(default_factory=lambda: _int_env("DB_POOL_MIN_SIZE", 1))
    db_pool_max_size: int = field(default_factory=lambda: _int_env("DB_POOL_MAX_SIZE", 10))

    cors_origins: list[str] = field(default_factory=lambda: _csv(os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost")))
    data_fetch_mode: str = field(default_factory=lambda: os.getenv("DATA_FETCH_MODE", "auto").strip().lower())
    # Bounds the thread pool open_handle uses to extract and lazily open a
    # multi-granule bundle's members concurrently (services/open_handle.py).
    # Both steps are I/O-bound (zip decompression, HDF5/netCDF header reads)
    # and members open lazily (chunks={}), so raising this speeds up a 50+
    # granule retrieval without materializing more data into RAM — it does
    # not interact with dask's separate num_workers=2 compute-scheduler cap.
    granule_concurrency: int = field(default_factory=lambda: max(1, _int_env("GRANULE_CONCURRENCY", 4)))
    memory_cache_max_bytes: int = field(
        default_factory=lambda: max(1, _int_env("MEMORY_CACHE_MAX_BYTES", 500 * 1024 * 1024))
    )
    csv_export_max_granules: int = field(default_factory=lambda: max(1, _int_env("CSV_EXPORT_MAX_GRANULES", 50)))
    s3_force_fetch: bool = field(default_factory=lambda: os.getenv("S3_FORCE_FETCH", "").strip() == "1")
    harmony_processing_timeout_seconds: int = field(
        default_factory=lambda: max(1, _int_env("HARMONY_PROCESSING_TIMEOUT_SECONDS", 600))
    )

    earthdata_token: str | None = field(default_factory=lambda: os.getenv("EARTHDATA_TOKEN"))
    earthdata_mcp_url: str = field(default_factory=lambda: os.getenv("EARTHDATA_MCP_URL", "http://mcp:8765/mcp"))
    earthdata_mcp_token: str | None = field(default_factory=lambda: os.getenv("EARTHDATA_MCP_TOKEN"))
    retrieval_soft_cap_bytes: int = field(
        default_factory=lambda: max(1, _int_env("RETRIEVAL_SOFT_CAP_BYTES", 2 * 1024 ** 3))
    )
    retrieval_hard_cap_bytes: int = field(
        default_factory=lambda: max(1, _int_env("RETRIEVAL_HARD_CAP_BYTES", 10 * 1024 ** 3))
    )
    await_retrieval_poll_min_seconds: int = field(
        default_factory=lambda: max(1, _int_env("AWAIT_RETRIEVAL_POLL_MIN_SECONDS", 2))
    )
    # The backoff doubles 2 -> 4 -> 8 -> cap, so once saturated the narrated
    # status trails reality by up to a full cap: a job that finished at t=30s
    # kept being reported as running until the next poll. At the old 15s cap
    # that was 7.5s of staleness on average and 15s at worst — 4% noise on a
    # 3-minute Harmony job, but 25-45% on a 30-second retrieval, i.e. the
    # backoff cost the most exactly where there was least to hide it and the
    # fast retrievals never got to feel fast. 5s buys that back for a handful
    # of extra status calls per job.
    await_retrieval_poll_max_seconds: int = field(
        default_factory=lambda: max(1, _int_env("AWAIT_RETRIEVAL_POLL_MAX_SECONDS", 5))
    )
    await_retrieval_timeout_seconds: int = field(
        default_factory=lambda: max(1, _int_env("AWAIT_RETRIEVAL_TIMEOUT_SECONDS", 900))
    )
    # T38: the one seam every MCP call passes through (earthdata_mcp/results.py
    # call_tool) times out an individual tool.ainvoke — generous by default
    # because retrieve_subset submissions and large export_result calls are
    # legitimately slow, but bounded so a wedged call can't pin a turn forever.
    mcp_call_timeout_seconds: int = field(
        default_factory=lambda: max(1, _int_env("MCP_CALL_TIMEOUT_SECONDS", 120))
    )
    # T38: the whole-turn deadline stream_chat_events/_fast_path_events enforce
    # around their event loop. Must stay comfortably above
    # await_retrieval_timeout_seconds (validate_startup asserts this) so a
    # legitimate long retrieval poll is never misread as a hung turn.
    chat_turn_timeout_seconds: int = field(
        default_factory=lambda: max(1, _int_env("CHAT_TURN_TIMEOUT_SECONDS", 1800))
    )
    # Storm containment (2026-07-20): every MCP tool opens a fresh streamable-
    # HTTP session per ainvoke (no persistent session), so a wedged call can
    # churn reconnect requests for its whole mcp_call_timeout window and the
    # ReAct loop retries it up to the graph budget — observed as 600+ MCP
    # requests/15s for 20+ minutes without ever reaching the plot step. The
    # call_tool circuit breaker (earthdata_mcp/results.py) short-circuits every
    # remaining MCP call in a turn once this many *consecutive* transport
    # failures accrue; one success resets the count.
    mcp_transport_failure_ceiling: int = field(
        default_factory=lambda: max(1, _int_env("MCP_TRANSPORT_FAILURE_CEILING", 3))
    )
    # Half-open recovery for the breaker above (2026-07-20). The MCP *server*
    # (earthdata-mcp) crash-restarts intermittently (~10-15s outage windows);
    # the heaviest MCP caller — compare, 4+ sequential calls — reliably lands in
    # one and trips the breaker. A permanently-sticky trip then fails the whole
    # (up to 30-min) turn for a 12-second blip. Once tripped, the breaker waits
    # this cooldown, then lets ONE call through as a half-open probe: success
    # closes it (the turn continues), failure re-arms the cooldown. So a brief
    # restart self-heals within ~cooldown of the server coming back, while a
    # genuine sustained storm still admits at most one (bounded) probe per
    # cooldown — never the 600-calls/15s churn the breaker exists to stop.
    mcp_transport_recovery_cooldown_seconds: float = field(
        default_factory=lambda: max(1.0, _float_env("MCP_TRANSPORT_RECOVERY_COOLDOWN_SECONDS", 5.0))
    )
    # The LangGraph superstep budget every agent's astream runs under. Raised
    # from LangGraph's historical default (25) to 40 after the 2026-07-20 AOD
    # wildfire session: a legitimate two-period, multi-day plot workflow (search
    # → AOI → coverage → retrieve → poll → plot, ×2 periods) is ~two supersteps
    # per tool call and hit the 25 ceiling as an opaque GraphRecursionError,
    # not a runaway loop. 40 (~18 tool round-trips) leaves room for that real
    # workflow while still capping a genuine loop. Kept explicit and tunable so
    # the ceiling stays an intentional choice (the storm diagnosis called out
    # that it must never be an accidental default).
    agent_recursion_limit: int = field(
        default_factory=lambda: max(1, _int_env("AGENT_RECURSION_LIMIT", 40))
    )
    retrieval_max_timeseries_days: int = field(
        default_factory=lambda: max(1, _int_env("RETRIEVAL_MAX_TIMESERIES_DAYS", 366))
    )
    # Gate on a result bundle's *uncompressed* size before open_handle extracts
    # and opens it.
    #
    # This was a RAM gate, and is no longer one: open_max_chunk_bytes below
    # bounds peak memory by the chunk rather than by the bundle, so a bundle's
    # total size stopped predicting what opening it costs (measured: 16
    # granules, 2874 MiB before, 492 MiB after, flat in granule count). Two
    # jobs it still does, which is why it stays:
    #
    #   1. Disk. Opening extracts every member into the extract cache under
    #      tempfile.gettempdir(), where entries outlive the call by design (see
    #      _extract_bundle_cached) — this is the only thing bounding that.
    #   2. The un-priced retrieval. The submit-time caps gate the *provider's
    #      estimate*, and retrieval_composites explicitly proceeds on
    #      confirmation when that estimate comes back None, naming this gate as
    #      the backstop. Nothing else stands behind that path.
    #
    # Raised 2 -> 8 GiB accordingly: 2 GiB was chosen as a memory number and
    # silently amounted to a ~40-granule ceiling, which is under two days of
    # TEMPO NO2 over North America — a limit on ordinary questions, imposed by
    # a setting whose stated reason no longer applied. 8 GiB is ~160 granules
    # and sized to the disk it actually protects.
    #
    # NOT a granule cap by another name, and there is no other one: the knob
    # that looked like it (satellite_max_results_cap) was dead code and was
    # deleted rather than left to imply a guarantee it never made.
    bundle_open_max_uncompressed_bytes: int = field(
        default_factory=lambda: max(1, _int_env("BUNDLE_OPEN_MAX_UNCOMPRESSED_BYTES", 8 * 1024 ** 3))
    )
    # The ceiling on a single dask chunk, and with it the pipeline's whole
    # memory profile: a chunk is the unit every task allocates, so peak RAM is
    # roughly this times the in-flight task count (dask num_workers=2, see
    # services/open_handle.py) times the intermediates one expression holds.
    # It is what makes memory constant in granule count rather than a property
    # of the layout the provider happened to write — see _open_groups_bounded.
    #
    # Swept on the shape that used to die — a 16-granule time-mean over
    # contiguous 2950x5771 float64 members. Four interleaved repetitions,
    # median, so disk-cache drift hits every case alike (a single-shot sweep of
    # this is worth nothing; the first pass read 14.1 s where the median is
    # 17.7 s):
    #
    #     unbounded -> 3069 MiB, 15.9 s      32 MiB -> 492 MiB, 17.7 s
    #        16 MiB ->  318 MiB, 22.7 s      64 MiB -> 850 MiB, 15.8 s
    #
    # 32 MiB is the default: ~11% wall clock for a 6.2x cut in peak. The
    # trade gets bad in both directions from there — 16 MiB costs 43% for
    # another 174 MiB, while 64 MiB buys the last 11% back at 850 MiB, which
    # is a third of the way to the spike this exists to prevent. Below 16 MiB
    # is pointless regardless: the floor is the output field itself (~260 MiB
    # here), not the chunks.
    #
    # The number to watch when tuning is concurrency, not the table: this peak
    # is per in-flight request while dask's num_workers cap is process-wide, so
    # N researchers plotting continents at once pay it N times. Lower it via
    # OPEN_MAX_CHUNK_BYTES if that ever bites before the throughput does.
    #
    # None of this touches a provider that already chunked its file sensibly:
    # such a file is under budget, opens once, and is left exactly as written.
    open_max_chunk_bytes: int = field(
        default_factory=lambda: max(1, _int_env("OPEN_MAX_CHUNK_BYTES", 32 * 1024 ** 2))
    )
    # The public chart-output directory. Mounted unauthenticated at /outputs
    # (api.py) and shared with the frontend nginx container through the
    # `plot_outputs` named volume, so anything written here is world-readable.
    #
    # A setting for the same reason cube_store_dir is one: it was three
    # ``APP_ROOT``-relative module constants, each with an import-time
    # os.makedirs, so importing the backend created `Backend/outputs/` in the
    # checkout — the cube-store problem again, and invisible for longer because
    # an *empty* directory is omitted from `git status` entirely, even under
    # --ignored. Only api.py's copy is live; StaticFiles needs the directory to
    # exist when it is mounted.
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "/app/outputs"))
    # T23: the server-rendered map overlay PNGs (tools/satellite_tools/
    # plot_tools.py). Deliberately OUTSIDE output_dir, which is served
    # unauthenticated — overlays are only reachable through the authenticated
    # /chart/{id}/overlay.png route, which checks chart ownership first.
    #
    # Also a setting rather than an ``APP_ROOT``-relative constant, and for the
    # same reason: the constant was computed at import and created
    # `Backend/overlay_store/` in the checkout, so the suite wrote into the
    # developer's own store. Gitignored, so that state survived branch switches
    # while never appearing in `git status`.
    overlay_store_dir: str = field(
        default_factory=lambda: os.getenv("OVERLAY_STORE_DIR", "/app/overlay_store/overlays")
    )
    # T52: the L4 Zarr cube cache (services/cube_cache.py). A backend-only
    # Docker named volume, NOT a tempdir — cubes cost minutes to build and
    # `tta-backend` is rebuilt constantly, so a tempdir store would be empty
    # every time you looked and would surface as "the cache doesn't help"
    # (the overlay_store failure mode exactly).
    cube_store_dir: str = field(default_factory=lambda: os.getenv("CUBE_STORE_DIR", "/app/cube_store"))
    # A fixed byte cap, deliberately not a percentage of free space — that
    # silently expands to fill any disk, which is how docker_data.vhdx reached
    # 296 GB against ~12 GB live. Eviction is LRU by last access, run before
    # each write.
    cube_store_max_bytes: int = field(
        default_factory=lambda: max(1, _int_env("CUBE_STORE_MAX_BYTES", 4 * 1024 ** 3))
    )
    # The largest share of the store one cube may occupy, as a fraction of
    # cube_store_max_bytes.
    #
    # A fraction rather than an absolute byte count because the only thing this
    # still guards is *thrash*: a cube big enough to evict the whole store to
    # fit itself, and then be evicted by the next one, is worse than never
    # caching it. That is inherently relative to the store's size, so it should
    # not have to be retuned whenever the store is resized. At 0.5 at least two
    # cubes always coexist.
    #
    # It replaces cube_write_max_bytes, which was measured against ds.nbytes —
    # the *uncompressed in-memory* footprint — while the store accounts entries
    # by real bytes on disk. Zarr writes compressed: 1.9x on incompressible
    # noise, 5.5x on a fill-heavy TEMPO grid, so the old 1 GiB cap was charging
    # cubes up to five times what they cost and rejecting them hardest on the
    # multi-granule retrievals the cache exists for (a 16-granule TEMPO bundle
    # is 2.855 GiB of nbytes and ~0.5 GiB of disk). Its other stated reason —
    # that writing is heavier than the lazy open — was true when the source
    # arrived in granule-sized chunks and is not now: with open_max_chunk_bytes
    # bounding the read, the same write peaks at 81.6 MiB, in the background,
    # behind a semaphore.
    cube_write_max_store_fraction: float = field(
        default_factory=lambda: min(1.0, max(0.0, _float_env("CUBE_WRITE_MAX_STORE_FRACTION", 0.5)))
    )
    # T59 Phase 4: the frame blob store (services/frame_store.py). Its own
    # backend-only named volume, and deliberately NOT inside overlay_store:
    # that store has no eviction policy and grows forever, which is precisely
    # why frames get a separate *bounded* one instead of joining it.
    frame_store_dir: str = field(
        default_factory=lambda: os.getenv("FRAME_STORE_DIR", "/app/frame_store")
    )
    # A quarter of cube_store's cap, and fixed bytes for the same reason: a
    # percentage of free space silently expands to fill any disk, which is how
    # docker_data.vhdx reached 296 GB. The proportion is the right one — a
    # frame stack is a rendering convenience D8 already refuses to regenerate,
    # while a cube is minutes of recompute.
    #
    # Entry size is bounded by construction (D7 caps frames at 60, D5 caps
    # cells at 20,000, so an entry is at most 60 x 20,000 x 4 = 4.58 MiB raw),
    # so this store needs an LRU sweeper and *no per-entry cap*: the thrash
    # cube_write_max_store_fraction guards against — one entry big enough to
    # evict the whole store to fit itself — cannot occur at 4.58 MiB against
    # 1 GiB. Measured typical entries are 1.30-1.41 MB gzipped, ~300-760
    # charts' worth.
    frame_store_max_bytes: int = field(
        default_factory=lambda: max(1, _int_env("FRAME_STORE_MAX_BYTES", 1024 ** 3))
    )
    # T54: serve a cube straight off the handle->cube index, skipping the
    # export_result round-trip entirely. Defaults ON — that skip is the whole
    # point, and it is what makes an upstream eviction or a crash-restart cost
    # a local Zarr read instead of minutes of rematerializing. Set to 0 to
    # restore T52's unconditional verify-first ordering without a redeploy, so
    # the optimization can be switched off in production if it is ever
    # implicated in a bad answer.
    cube_skip_export_verify: bool = field(
        default_factory=lambda: os.getenv("CUBE_SKIP_EXPORT_VERIFY", "1").strip() != "0"
    )
    # T53: the discovery-metadata cache at the bind_workspace seam
    # (earthdata_mcp/tool_cache.py). Collection metadata changes on the order
    # of never and AOI resolution from a location string is deterministic, so
    # the TTL is a safety net rather than a freshness mechanism — freshness is
    # handled by keeping coverage/availability off the allowlist entirely.
    mcp_metadata_cache_ttl_seconds: int = field(
        default_factory=lambda: max(1, _int_env("MCP_METADATA_CACHE_TTL_SECONDS", 3600))
    )
    # An entry cap so a long-running process cannot grow unbounded on
    # search-query variety (search_datasets keys on a free-text query).
    mcp_metadata_cache_max_entries: int = field(
        default_factory=lambda: max(1, _int_env("MCP_METADATA_CACHE_MAX_ENTRIES", 512))
    )
    aqs_api_email: str = field(default_factory=lambda: os.getenv("AQS_API_EMAIL", "your_email@example.com"))
    aqs_api_key: str = field(default_factory=lambda: os.getenv("AQS_API_KEY", "your_aqs_key"))

    subagent_trim_token_ceiling: int = field(
        default_factory=lambda: max(1, _int_env("SUBAGENT_TRIM_TOKEN_CEILING", 20000))
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "text").strip().lower())
    # T45: gates the /debug/heap-snapshot tracemalloc endpoint. Off by
    # default -- tracemalloc adds per-allocation overhead, so this is an
    # opt-in diagnostic for chasing a specific memory incident, not a
    # standing production setting.
    debug_heap_profiling_enabled: bool = field(
        default_factory=lambda: os.getenv("DEBUG_HEAP_PROFILING_ENABLED", "").strip() == "1"
    )
    long_request_seconds: float = field(default_factory=lambda: float(os.getenv("LONG_REQUEST_SECONDS", "30")))
    # T61: Supabase is the identity provider; our own Postgres stays. Both are
    # required at boot (validate_startup) now that the auth middleware verifies
    # against them -- see the note there for why a missing URL has to be fatal
    # rather than merely absent. Typed Optional anyway, because the dataclass
    # field is what validate_startup inspects: a required field with no default
    # would raise a TypeError at construction, before the check that names it.
    # The URL is stripped of whitespace and any trailing slash: the issuer is
    # built as f"{supabase_url}/auth/v1", and a stray slash rejects every token
    # with no hint as to why. No default is possible for either -- a placeholder
    # URL would verify tokens from the wrong project rather than failing.
    supabase_url: str | None = field(
        default_factory=lambda: os.getenv("SUPABASE_URL", "").strip().rstrip("/") or None
    )
    supabase_publishable_key: str | None = field(
        default_factory=lambda: os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip() or None
    )

    # T30: per-user connector secret storage (MultiFernet). Comma-separated so
    # a rotation can carry an old + new key simultaneously -- unset entirely
    # means the connectors feature degrades to a structured 503 rather than
    # blocking boot (ground/EPA-only deployments never need this).
    connector_encryption_key: str | None = field(default_factory=lambda: os.getenv("CONNECTOR_ENCRYPTION_KEY"))

    # T23 MapLibre basemap/terrain sources -- free-tier defaults, no API key.
    # Configuration (not code) so a keyed/self-hosted provider can be swapped
    # in without a redeploy as traffic grows; see the T23 PRD's "Further
    # Notes" on these providers' lack of an SLA.
    map_basemap_light_url: str = field(
        default_factory=lambda: os.getenv(
            "MAP_BASEMAP_LIGHT_URL", "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
        )
    )
    map_basemap_dark_url: str = field(
        default_factory=lambda: os.getenv(
            "MAP_BASEMAP_DARK_URL", "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png"
        )
    )
    map_terrain_dem_url: str = field(
        default_factory=lambda: os.getenv(
            "MAP_TERRAIN_DEM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        )
    )
    map_basemap_attribution: str = field(
        default_factory=lambda: os.getenv("MAP_BASEMAP_ATTRIBUTION", "© CARTO © OpenStreetMap contributors")
    )
    map_terrain_attribution: str = field(
        default_factory=lambda: os.getenv("MAP_TERRAIN_ATTRIBUTION", "Terrain tiles: AWS Terrain Tiles")
    )

    def __post_init__(self) -> None:
        if self.data_fetch_mode not in _VALID_FETCH_MODES:
            object.__setattr__(self, "data_fetch_mode", "auto")
        if self.log_format not in _VALID_LOG_FORMATS:
            object.__setattr__(self, "log_format", "text")

    @property
    def db_kwargs(self) -> dict:
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }

    def validate_startup(self) -> None:
        missing = []
        if not self.db_password:
            missing.append("DB_PASSWORD")
        configured_providers = {
            self.supervisor_model_provider,
            self.earthdata_agent_provider,
            self.ground_agent_provider,
        }
        if "google" in configured_providers and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if "groq" in configured_providers and not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        # T61: unconditional, unlike the provider keys above -- every route is
        # authenticated, so there is no configuration in which the backend is
        # useful without an identity provider. Fatal here rather than absent
        # because api.py builds the verifier's issuer as an f-string at import:
        # a None URL yields the literal "None/auth/v1", which constructs fine
        # and then rejects every token with "Invalid issuer." on a backend that
        # booted clean and reports healthy. One RuntimeError beats that.
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        # Not read by the backend until T61 Phase 3 serves it from /config/auth.
        # Required now regardless: the frontend cannot sign anyone in without
        # it, so a deploy carrying one of the pair and not the other is broken
        # either way, and boot is a cheaper place to learn that than the login
        # screen. Drop this branch if Phase 3 ends up somewhere else.
        if not self.supabase_publishable_key:
            missing.append("SUPABASE_PUBLISHABLE_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
        unvetted = []
        for label, provider, model in (
            ("LLM_MODEL", self.supervisor_model_provider, self.llm_model),
            ("EARTHDATA_AGENT_MODEL", self.earthdata_agent_provider, self.earthdata_agent_model),
            ("GROUND_AGENT_MODEL", self.ground_agent_provider, self.ground_agent_model),
        ):
            vetted = _VETTED_MODELS.get(provider)
            if vetted is not None and model not in vetted:
                unvetted.append(f"{label}={model!r} for provider {provider!r}")
        if unvetted:
            raise RuntimeError(
                "Unvetted model id(s): "
                + "; ".join(unvetted)
                + ". Confirm each id exists on the provider and on the billing tier "
                "being deployed, then add it to _VETTED_MODELS in config/settings.py."
            )

        # T38: a turn deadline that isn't comfortably above the retrieval
        # poll-loop's own deadline would make every retrieval that runs the
        # full await_retrieval_timeout_seconds a guaranteed turn timeout.
        if self.chat_turn_timeout_seconds <= self.await_retrieval_timeout_seconds:
            raise ConfigurationError(
                f"CHAT_TURN_TIMEOUT_SECONDS ({self.chat_turn_timeout_seconds}) must be greater "
                f"than AWAIT_RETRIEVAL_TIMEOUT_SECONDS ({self.await_retrieval_timeout_seconds})"
            )

        # A malformed earthdata-retrieval MCP URL is a config typo to fix,
        # not an outage — it must fail loudly at boot rather than being
        # retried forever by the connection manager (T17).
        parsed_mcp_url = urlsplit(self.earthdata_mcp_url)
        if parsed_mcp_url.scheme not in ("http", "https") or not parsed_mcp_url.netloc:
            raise ConfigurationError(
                f"Invalid EARTHDATA_MCP_URL {self.earthdata_mcp_url!r}: must be an http(s) URL"
            )

        # T30: an unset CONNECTOR_ENCRYPTION_KEY degrades the connectors
        # feature to a 503 (ground/EPA-only deployments don't need it) -- but
        # a *set-and-malformed* key is a half-configured secret store, worse
        # than none, so it fails boot loudly rather than surfacing as a
        # confusing per-request decrypt error later.
        if self.connector_encryption_key:
            try:
                build_multi_fernet(self.connector_encryption_key)
            except ConnectorCryptoError as exc:
                raise ConfigurationError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
