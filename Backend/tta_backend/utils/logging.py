"""Logging setup helpers with optional JSON output."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from tta_backend.config.settings import Settings, get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                payload[key[1:]] = value
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    # langchain_google_genai logs "Key '...' is not supported in schema,
    # ignoring" at WARNING for every unrecognized JSON-schema keyword in a
    # tool's pydantic model -- known-benign (the schema still converts fine),
    # but frequent enough to drown real WARNING/ERROR log signal. Set above
    # the root/settings level regardless of it, and unconditionally (not
    # just on first configure_logging() call) since tests and reconfigures
    # must not let it slip back to WARNING.
    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(settings.log_level)
        return

    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root.addHandler(handler)
    root.setLevel(settings.log_level)
