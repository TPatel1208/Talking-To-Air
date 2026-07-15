"""Centralized runtime configuration for the backend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlsplit

from dotenv import load_dotenv

from utils.connector_crypto import ConnectorCryptoError, build_multi_fernet

_VALID_FETCH_MODES = {"auto", "harmony", "opendap", "s3"}
_VALID_LOG_FORMATS = {"text", "json"}


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


@dataclass(frozen=True)
class Settings:
    """Application settings loaded once from environment at startup/import."""

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    ground_agent_model: str = field(
        default_factory=lambda: os.getenv(
            "GROUND_AGENT_MODEL",
            "openai/gpt-oss-20b",
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
        default_factory=lambda: os.getenv("GROUND_AGENT_PROVIDER", "groq")
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
    satellite_max_results_cap: int = field(default_factory=lambda: max(1, _int_env("SATELLITE_MAX_RESULTS_CAP", 20)))
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
    await_retrieval_poll_max_seconds: int = field(
        default_factory=lambda: max(1, _int_env("AWAIT_RETRIEVAL_POLL_MAX_SECONDS", 15))
    )
    await_retrieval_timeout_seconds: int = field(
        default_factory=lambda: max(1, _int_env("AWAIT_RETRIEVAL_TIMEOUT_SECONDS", 900))
    )
    retrieval_max_timeseries_days: int = field(
        default_factory=lambda: max(1, _int_env("RETRIEVAL_MAX_TIMESERIES_DAYS", 366))
    )
    # Gate on a result bundle's *uncompressed* size before open_handle extracts
    # and opens it. The retrieval byte caps above gate the estimate at submit
    # time; this one catches what they can't — decompression, dtype widening,
    # and multi-granule concatenation happen at open time, and an ungated open
    # OOM-killed the backend on a full-day TEMPO NO2 bundle (live 2026-07-12).
    bundle_open_max_uncompressed_bytes: int = field(
        default_factory=lambda: max(1, _int_env("BUNDLE_OPEN_MAX_UNCOMPRESSED_BYTES", 2 * 1024 ** 3))
    )
    aqs_api_email: str = field(default_factory=lambda: os.getenv("AQS_API_EMAIL", "your_email@example.com"))
    aqs_api_key: str = field(default_factory=lambda: os.getenv("AQS_API_KEY", "your_aqs_key"))

    subagent_trim_token_ceiling: int = field(
        default_factory=lambda: max(1, _int_env("SUBAGENT_TRIM_TOKEN_CEILING", 20000))
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "text").strip().lower())
    long_request_seconds: float = field(default_factory=lambda: float(os.getenv("LONG_REQUEST_SECONDS", "30")))
    jwt_secret_key: str | None = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY"))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expiration_minutes: int = field(default_factory=lambda: max(1, _int_env("JWT_EXPIRATION_MINUTES", 60)))

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
        if not self.jwt_secret_key:
            missing.append("JWT_SECRET_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

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
