"""Centralized runtime configuration for the backend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


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

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemma-4-26b-a4b-it"))
    ground_agent_model: str = field(
        default_factory=lambda: os.getenv(
            "GROUND_AGENT_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        )
    )
    satellite_agent_model: str = field(default_factory=lambda: os.getenv("SATELLITE_AGENT_MODEL", "openai/gpt-oss-20b"))
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
    edl_username: str = field(default_factory=lambda: os.getenv("EDL_USERNAME", ""))
    edl_password: str = field(default_factory=lambda: os.getenv("EDL_PASSWORD", ""))
    aqs_api_email: str = field(default_factory=lambda: os.getenv("AQS_API_EMAIL", "your_email@example.com"))
    aqs_api_key: str = field(default_factory=lambda: os.getenv("AQS_API_KEY", "your_aqs_key"))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "text").strip().lower())
    long_request_seconds: float = field(default_factory=lambda: float(os.getenv("LONG_REQUEST_SECONDS", "30")))
    jwt_secret_key: str | None = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY"))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expiration_minutes: int = field(default_factory=lambda: max(1, _int_env("JWT_EXPIRATION_MINUTES", 60)))

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
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.jwt_secret_key:
            missing.append("JWT_SECRET_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
