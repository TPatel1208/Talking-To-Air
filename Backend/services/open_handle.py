"""
services/open_handle.py
========================
The single seam between an ``obs_``/``cube_`` handle and an opened dataset.

Wraps ``export_result`` with a bounded eviction-recovery loop
(rematerialize -> await -> re-export) so every plot/statistics tool sees
either an opened Dataset/Table or a clear error — never a missing file.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from langchain_core.tools import BaseTool

from config.workflow_stages import STAGE_OPEN
from earthdata_mcp.results import parse_tool_result
from services.retrieval_composites import await_retrieval
from utils.streaming import emit_status

logger = logging.getLogger(__name__)


class OpenHandleError(RuntimeError):
    """Raised when a handle cannot be opened, even after one rematerialize attempt."""


async def open_handle(handle: str, tools: dict[str, BaseTool]) -> Any:
    """Resolve ``handle`` to an opened xarray Dataset (Zarr/NetCDF) or Arrow table (Parquet).

    On an expired/evicted export, attempts exactly one rematerialize -> await
    -> re-export cycle; a second failure raises with the MCP's own
    structured message verbatim.
    """
    emit_status("Opening retrieved data...", stage=STAGE_OPEN)
    export = await _export(handle, tools)
    if export.get("status") != "ready":
        export = await _recover(handle, tools)
    return await asyncio.to_thread(_open, export["storage_uri"], export["media_type"])


async def _export(handle: str, tools: dict[str, BaseTool]) -> dict:
    raw = await tools["export_result"].ainvoke({"handle": handle})
    return parse_tool_result(raw)


async def _recover(handle: str, tools: dict[str, BaseTool]) -> dict:
    emit_status("Rematerializing expired data...", stage=STAGE_OPEN)
    remat_raw = await tools["rematerialize"].ainvoke({"handle": handle})
    remat = parse_tool_result(remat_raw)
    if remat.get("status") == "not_found":
        raise OpenHandleError(remat.get("message") or f"Handle '{handle}' not found and cannot be rematerialized.")

    job_handle = remat.get("job_handle")
    if job_handle:
        status = await await_retrieval(job_handle, tools)
        if status.get("status") != "ready":
            raise OpenHandleError(status.get("message") or f"Rematerializing handle '{handle}' failed.")

    second_export = await _export(handle, tools)
    if second_export.get("status") != "ready":
        raise OpenHandleError(
            second_export.get("message") or f"Handle '{handle}' still not ready after rematerialize."
        )
    return second_export


def _open(storage_uri: str, media_type: str) -> Any:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise OpenHandleError(
            f"Opening non-local URIs (scheme '{parsed.scheme}') is not yet supported: {storage_uri}"
        )
    path = url2pathname(parsed.path)

    mt = (media_type or "").lower()
    if "zarr" in mt:
        import xarray as xr

        return xr.open_zarr(path)
    if "parquet" in mt:
        import pyarrow.parquet as pq

        return pq.read_table(path)
    if "netcdf" in mt:
        import xarray as xr

        return xr.open_dataset(path)
    raise OpenHandleError(f"Unsupported media_type '{media_type}' for exported handle.")
