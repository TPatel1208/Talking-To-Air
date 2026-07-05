"""Parse earthdata-retrieval MCP tool call results into plain dicts.

A LangChain tool built from an MCP tool returns its structured JSON payload
wrapped in a list of content blocks (``[{"type": "text", "text": "<json>"}]``)
rather than a parsed dict — this unwraps that shape.
"""
from __future__ import annotations

import json
from typing import Any


def parse_tool_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return json.loads(block["text"])
    raise ValueError(f"Could not parse earthdata MCP tool result: {raw!r}")
