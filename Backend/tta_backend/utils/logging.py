"""Logging setup helpers with optional JSON output."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from tta_backend.config.settings import Settings, get_settings

# EPA AQS authenticates with email= and key= query parameters, so every AQS
# request URL carries the credential -- in httpx's INFO request line, in the
# Header EPA echoes back, and in httpx exception messages. Only the values
# go; the rest of the query stays readable for debugging. The [?&] anchor
# keeps look-alikes such as monkey= or api_key= untouched.
_CREDENTIAL_QUERY_PARAM = re.compile(r"([?&](?:email|key)=)[^&\s'\"#]+")

# Loggers whose records can carry request URLs. A filter here redacts the
# record itself, so it holds for every handler, including ones added later.
# uvicorn.error logs unhandled app tracebacks on its own handler and never
# reaches root, so the root handler filters below cannot see it.
_REDACTED_LOGGERS = ("httpx", "httpcore", "uvicorn.error")


def redact_credentials(text: str) -> str:
    return _CREDENTIAL_QUERY_PARAM.sub(r"\1REDACTED", text)


class RedactCredentialsFilter(logging.Filter):
    """Rewrites a record's message and traceback with credential query
    values replaced by REDACTED. Never drops a record."""

    _traceback_formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_credentials(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = self._traceback_formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_credentials(record.exc_text)
        return True


def _install_redaction(filterer: logging.Filterer) -> None:
    if not any(isinstance(f, RedactCredentialsFilter) for f in filterer.filters):
        filterer.addFilter(RedactCredentialsFilter())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            # exc_text holds the traceback RedactCredentialsFilter scrubbed.
            payload["exc_info"] = record.exc_text or self.formatException(record.exc_info)
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

    # Unconditional for the same reason. The logger filters cover httpx's
    # URLs on any handler; the handler filters cover everything else that
    # reaches root, including the backend's own lines.
    for name in _REDACTED_LOGGERS:
        _install_redaction(logging.getLogger(name))

    root = logging.getLogger()
    if root.handlers:
        for existing in root.handlers:
            _install_redaction(existing)
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
    _install_redaction(handler)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
