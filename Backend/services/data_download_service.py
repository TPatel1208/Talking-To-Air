"""
services/data_download_service.py
====================================
T10: backend seam behind the artifact data-download endpoints. Calls the
MCP's ``convert_format`` tool to materialize a handle in a downloadable
format (e.g. NetCDF), then streams the converted file straight off disk —
extends the existing export/download machinery (open_handle's file://
export contract) rather than adding a parallel download system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from earthdata_mcp.results import parse_tool_result


class DataDownloadError(RuntimeError):
    """Raised when a handle cannot be converted or streamed for download."""


async def export_converted(handle: str, target_format: str, tools: dict[str, BaseTool]) -> dict[str, Any]:
    """Convert ``handle`` to ``target_format`` via the MCP; return its export descriptor."""
    raw = await tools["convert_format"].ainvoke({"handle": handle, "target_format": target_format})
    result = parse_tool_result(raw)
    if result.get("status") != "ready":
        raise DataDownloadError(
            result.get("message") or f"Could not convert handle '{handle}' to {target_format}."
        )
    return result


def iter_file_chunks(storage_uri: str, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Stream a ``file://`` URI's bytes in fixed-size chunks."""
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise DataDownloadError(
            f"Streaming non-local URIs (scheme '{parsed.scheme}') is not yet supported: {storage_uri}"
        )
    path = Path(url2pathname(parsed.path))

    async def _chunks() -> AsyncIterator[bytes]:
        with path.open("rb") as handle_file:
            while True:
                chunk = handle_file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return _chunks()


async def iter_converted_chunks(
    handle: str,
    target_format: str,
    tools: dict[str, BaseTool],
    chunk_size: int = 64 * 1024,
) -> AsyncIterator[bytes]:
    """Convert ``handle`` to ``target_format`` via the MCP, then stream its bytes."""
    export = await export_converted(handle, target_format, tools)
    async for chunk in iter_file_chunks(export["storage_uri"], chunk_size):
        yield chunk
