"""
Parse earthdata-retrieval MCP tool call results into plain dicts, or raise a
typed, categorized MCPToolError (PRD T18).

A LangChain tool built from an MCP tool returns its structured JSON payload
wrapped in a list of content blocks (``[{"type": "text", "text": "<json>"}]``)
rather than a parsed dict — this unwraps that shape. But the same channel
also carries FastMCP validation-error prose, a tool-raised ValueError's
message text, and (via ``call_tool``) a connection failure raised instead of
returned — every shape that isn't clean JSON used to detonate as an
unhandled exception somewhere different. ``parse_tool_result`` is the one
place every consumer already calls; it is now the single classification
point: a dict back, or ``MCPToolError`` with a category a caller can act on.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

CATEGORY_USER_INPUT = "user_input"
CATEGORY_NO_DATA = "no_data"
CATEGORY_NOT_FOUND = "not_found"
CATEGORY_TOO_LARGE = "too_large"
CATEGORY_PROVIDER_UNAVAILABLE = "provider_unavailable"
CATEGORY_CONTRACT = "contract"

# The MCP's own structured answers (T18 story #9) — first-class results, not
# errors; passed through as plain dicts for each composite's existing
# status-based branching (e.g. services/open_handle.py) to interpret.
_STRUCTURED_PASSTHROUGH_STATUSES = {"not_found", "pending", "expired"}


class MCPToolError(Exception):
    """A classified MCP tool outcome: what happened (``category``), a human
    ``message``, an optional actionable ``suggestion``, and a ``raw_preview``
    of whatever the adapter actually returned (for the debug log line T18
    story #10 asks for — never shown to a researcher)."""

    def __init__(self, category: str, message: str, *, suggestion: str | None = None, raw_preview: str | None = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.suggestion = suggestion
        self.raw_preview = raw_preview

    def to_dict(self) -> dict:
        """``{"category", "message", "suggestion"}`` — the nested shape
        every ``{"error": ...}`` envelope (tool JSON, pane 4xx/5xx bodies)
        builds from, dropping ``suggestion`` when there isn't one."""
        body: dict[str, Any] = {"category": self.category, "message": self.message}
        if self.suggestion:
            body["suggestion"] = self.suggestion
        return body

    def to_tool_json(self) -> str:
        """Structured JSON tool result shape (T18): what a model-facing tool
        call returns instead of raising, and what a backend composite's own
        ``parse_tool_result(raw)`` call recognizes and re-raises from."""
        return json.dumps({"error": self.to_dict()})


_VALIDATION_ERROR_RE = re.compile(r"^\s*\d+\s+validation error(s)?\s+for\b", re.IGNORECASE)

# langchain_mcp_adapters wraps a tool-raised exception's text in this prefix
# (live-verified 2026-07-08 against the real MCP: "Error calling tool
# 'define_area_of_interest': Neither Nominatim..."). It carries no
# information a researcher needs — strip it before matching or surfacing.
_ADAPTER_ERROR_PREFIX_RE = re.compile(r"^Error calling tool '[^']*':\s*")

# Known ValueError prefixes/substrings raised by the MCP's area/coverage/
# retrieval tools (harmony-retrieval-mcp/src/earthdata_mcp/tools/area.py,
# providers/base.py) — a researcher-fixable input, so these classify as
# user_input rather than the contract fallback. Unknown ValueError prose
# (a new raise site the MCP adds later) still classifies safely as contract
# instead of crashing (T18 Implementation Decisions: additive-safe).
_USER_INPUT_PATTERNS: tuple[tuple[str, str], ...] = (
    # Both the single-source ("Nominatim found no results for location ...")
    # and the fallback-exhausted ("Neither Nominatim nor USGS WBD found
    # results for location ...") messages share this substring — live-
    # verified 2026-07-07/08 against the real MCP (F6), which also wraps the
    # tool's raised message in an adapter-added "Error calling tool 'X': "
    # prefix this backend never anchors against.
    ("results for location", "Try a more specific location name."),
    ("location must be provided", "Provide a location."),
    (
        "Ambiguous location",
        "Use the HUC or FIPS prefix to disambiguate (e.g. 'HUC 0204' or 'FIPS 34023').",
    ),
)


def parse_tool_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return _classify_dict(raw)
    if isinstance(raw, str):
        return _classify_text(raw)
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return _classify_text(block["text"])
        raise _log(MCPToolError(
            CATEGORY_CONTRACT,
            "The data service returned an empty or unrecognized result.",
            raw_preview=repr(raw)[:300],
        ))
    raise _log(MCPToolError(
        CATEGORY_CONTRACT,
        "The data service returned an unrecognized result shape.",
        raw_preview=repr(raw)[:300],
    ))


def _classify_dict(data: dict) -> dict:
    error = data.get("error")
    if isinstance(error, dict) and "category" in error and "message" in error:
        # bind_workspace's own error envelope (T18), round-tripped back
        # through this same classifier by a backend composite that calls
        # parse_tool_result on what bind_workspace returned.
        raise _log(MCPToolError(
            error["category"],
            error["message"],
            suggestion=error.get("suggestion"),
            raw_preview=error.get("raw_preview"),
        ))
    return data


def _classify_text(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise _log(_classify_prose(text)) from None
    if isinstance(parsed, dict):
        return _classify_dict(parsed)
    return parsed


def _classify_prose(text: str) -> MCPToolError:
    stripped = _ADAPTER_ERROR_PREFIX_RE.sub("", text.strip(), count=1)

    if _VALIDATION_ERROR_RE.match(stripped):
        # FastMCP rejected the call's own parameters — always a bug on this
        # backend's side (or the model's), never the researcher's; T18
        # Implementation Decisions: "always a bug, always logged loud."
        return MCPToolError(
            CATEGORY_CONTRACT,
            "The data service rejected the request as malformed.",
            raw_preview=stripped[:300],
        )

    for pattern, suggestion in _USER_INPUT_PATTERNS:
        if pattern in stripped:
            return MCPToolError(CATEGORY_USER_INPUT, stripped, suggestion=suggestion, raw_preview=stripped[:300])

    if "time_range" in stripped and "ISO-8601" in stripped:
        return MCPToolError(
            CATEGORY_USER_INPUT,
            stripped,
            suggestion="Use an ISO-8601 'start/end' time range, e.g. '2024-01-01/2024-01-31'.",
            raw_preview=stripped[:300],
        )

    return MCPToolError(
        CATEGORY_CONTRACT,
        "The data service returned an unrecognized error.",
        raw_preview=stripped[:300],
    )


def _log(exc: MCPToolError) -> MCPToolError:
    """T18 story #10: every classified error logged with its category and
    raw payload — contract errors loud (always a bug), everything else at
    debug (taxonomy gaps stay discoverable without flooding normal-operation
    logs with researcher-fixable input errors)."""
    log = logger.error if exc.category == CATEGORY_CONTRACT else logger.debug
    log(
        "earthdata_mcp_tool_error",
        extra={"_event": "earthdata_mcp_tool_error", "_category": exc.category, "_raw_preview": exc.raw_preview},
    )
    return exc


async def call_tool(tool: Any, kwargs: dict) -> Any:
    """Invoke ``tool.ainvoke(kwargs)`` and return its raw, unclassified
    result — or raise ``MCPToolError`` (category ``provider_unavailable``)
    for a transport/session-level failure.

    A connection failure is *raised* by langchain_mcp_adapters, never
    returned as tool content (content only carries MCP-side ``isError``
    results, which ``parse_tool_result`` classifies separately) — this is
    the seam that catches it so no consumer ever sees a bare
    ``httpcore``/``httpx``/``mcp`` transport exception.

    Live-verified 2026-07-08 (MCP stopped mid-session): the streamable-HTTP
    transport's structured concurrency (anyio task groups) wraps the actual
    ``httpx.ConnectError``/``httpcore.ConnectError`` in a ``BaseExceptionGroup``
    rather than raising it bare — ``except*`` unwraps that regardless of
    whether the underlying exception arrived grouped or not (PEP 654).
    """
    import httpcore
    import httpx
    from mcp.shared.exceptions import McpError

    try:
        return await tool.ainvoke(kwargs)
    except* (httpcore.ConnectError, httpx.ConnectError, httpx.ConnectTimeout, McpError) as eg:
        detail = eg.exceptions[0] if eg.exceptions else eg
        raise _log(MCPToolError(
            CATEGORY_PROVIDER_UNAVAILABLE,
            "The satellite data layer is temporarily unavailable.",
            suggestion="Try again in a moment.",
            raw_preview=str(detail)[:300],
        )) from eg
