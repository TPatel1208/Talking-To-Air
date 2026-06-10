from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


PNG_PATH_RE = re.compile(r"(/outputs/[\w\-./]+\.png|[\w\-./]+\.png)")


def flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "thinking":
                    continue
                parts.append(block.get("text", "") or str(block.get("content", "")))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def truncate_text(
    text: str,
    max_chars: int,
    agent_name: str,
    request_id: str | None = None,
) -> str:
    if len(text) <= max_chars:
        return text
    logger.warning(
        "response_truncated",
        extra={
            "_agent_name": agent_name,
            "_original_length": len(text),
            "_final_length": max_chars,
            "_request_id": request_id,
        },
    )
    return text[:max_chars]


def extract_last_text(
    result: dict,
    fallback: str,
    max_chars: int = 2000,
    agent_name: str = "unknown",
    request_id: str | None = None,
) -> str:
    """Return the last non-empty text content from an agent invoke() result."""
    for msg in reversed(result.get("messages", [])):
        if not (hasattr(msg, "content") and msg.content):
            continue
        text = flatten_text_content(msg.content)
        if text:
            return truncate_text(text, max_chars, agent_name, request_id)
    return fallback


def normalize_image_url(raw: str) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("/outputs/"):
        return raw
    filename = raw.replace("\\", "/").split("/")[-1]
    return f"/outputs/{filename}"
