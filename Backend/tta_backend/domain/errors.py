"""Classified tool errors, and the categories a caller can act on."""
from __future__ import annotations

import json
from typing import Any

CATEGORY_USER_INPUT = "user_input"
CATEGORY_NO_DATA = "no_data"
CATEGORY_NOT_FOUND = "not_found"
CATEGORY_TOO_LARGE = "too_large"
CATEGORY_PROVIDER_UNAVAILABLE = "provider_unavailable"
CATEGORY_CONTRACT = "contract"
# A granule whose horizontal grid isn't a 1-D rectilinear lat/lon grid
# (2-D curvilinear swath, or projected x/y+CRS) — a known, honest limitation
# rather than a crash or a silently mis-masked result.
CATEGORY_UNSUPPORTED_GRID = "unsupported_grid"
# The backend refuses to invent a scientific choice it was never given.
# CATEGORY_VARIABLE_CHOICE_REQUIRED — a multi-variable file with no explicit
# variable and no recorded choice (AggregationService.to_dataarray).
# CATEGORY_DIMENSION_CHOICE_REQUIRED — a surviving non-spatial, non-time
# dimension (e.g. a MERRA-2 vertical level) with no selection
# (_normalize_to_2d). Both name their
# candidates in the message so a researcher (or the model) can pick
# deliberately instead of receiving whichever value happened to be first.
CATEGORY_VARIABLE_CHOICE_REQUIRED = "variable_choice_required"
CATEGORY_DIMENSION_CHOICE_REQUIRED = "dimension_choice_required"
# Per-user Earthdata credential errors the MCP raises against an injected
# edl_token. _classify_dict handles them generically; they exist to drive the
# actionable message templates below.
CATEGORY_TOKEN_INVALID = "token_invalid"
CATEGORY_TOKEN_EXPIRED = "token_expired"
CATEGORY_EULA_NOT_ACCEPTED = "eula_not_accepted"


class MCPToolError(Exception):
    """A classified MCP tool outcome: what happened (``category``), a human
    ``message``, an optional actionable ``suggestion``, and a ``raw_preview``
    of whatever the adapter actually returned (for the debug log line —
    never shown to a researcher)."""

    def __init__(self, category: str, message: str, *, suggestion: str | None = None, raw_preview: str | None = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.suggestion = suggestion
        self.raw_preview = raw_preview

    def to_dict(self) -> dict[str, Any]:
        """``{"category", "message", "suggestion"}`` — the nested shape
        every ``{"error": ...}`` envelope (tool JSON, pane 4xx/5xx bodies)
        builds from, dropping ``suggestion`` when there isn't one."""
        body: dict[str, Any] = {"category": self.category, "message": self.message}
        if self.suggestion:
            body["suggestion"] = self.suggestion
        return body

    def to_tool_json(self) -> str:
        """Structured JSON tool result shape: what a model-facing tool
        call returns instead of raising, and what a backend composite's own
        ``parse_tool_result(raw)`` call recognizes and re-raises from."""
        return json.dumps({"error": self.to_dict()})
