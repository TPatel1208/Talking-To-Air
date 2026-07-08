"""
retrieval_tools.py
-------------------
Model-facing LangChain wrappers over the two retrieval composites
(services.retrieval_composites): ``safe_retrieve`` (estimate -> gate ->
retrieve) and ``await_retrieval`` (backend-side polling). Both close over
``mcp_tools`` so the model never sees or supplies the raw MCP tool dict —
same closure pattern as the handle-based plot/statistics tools.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from earthdata_mcp.results import MCPToolError
from services.retrieval_composites import RetrievalTimeoutError
from services.retrieval_composites import await_retrieval as _await_retrieval
from services.retrieval_composites import safe_retrieve as _safe_retrieve


def make_safe_retrieve(mcp_tools: dict[str, BaseTool]):
    @tool
    async def safe_retrieve(
        dataset_handle: str,
        aoi_handle: str,
        time_range: str,
        variables: list[str],
        output_format: Optional[str] = None,
        confirmed: bool = False,
    ) -> str:
        """
        Estimate a retrieval's size, then gate or submit it — the one call
        that stands in for retrieve_subset/estimate_retrieval_size.

        Always call this instead of retrieving directly. Returns one of:
          - status "submitted": retrieval started; pass the returned
            job_handle to await_retrieval.
          - status "needs_confirmation": ask the researcher before retrying
            this same call with confirmed=True.
          - status "refused": above the hard cap; narrow the AOI, time
            range, or variable list instead of retrying as-is.

        Args:
            dataset_handle : dataset_ handle from search_datasets.
            aoi_handle     : aoi_ handle from define_area_of_interest.
            time_range     : ISO 8601 interval, e.g. '2024-01-01/2024-01-31'.
            variables      : variable short names to retrieve.
            output_format  : optional output format hint.
            confirmed      : set True only after the researcher has approved
                              a prior "needs_confirmation" response.
        """
        try:
            result = await _safe_retrieve(
                dataset_handle,
                aoi_handle,
                time_range,
                variables,
                mcp_tools,
                output_format=output_format,
                confirmed=confirmed,
            )
        except MCPToolError as exc:
            return json.dumps({"error": exc.to_dict()})
        return json.dumps(result)

    return safe_retrieve


def make_await_retrieval(mcp_tools: dict[str, BaseTool]):
    @tool
    async def await_retrieval(job_handle: str) -> str:
        """
        Block until a retrieval job (from safe_retrieve) reaches a terminal
        state, spending one turn instead of polling get_retrieval_status
        yourself. Returns the terminal status, including the obs_/cube_
        handle on success. A failed/cancelled job is returned, not raised —
        report its message to the researcher verbatim.

        Args:
            job_handle : job_ handle returned by safe_retrieve.
        """
        try:
            result = await _await_retrieval(job_handle, mcp_tools)
        except RetrievalTimeoutError as exc:
            return json.dumps({"status": "timeout", "message": str(exc), "job_handle": job_handle})
        except MCPToolError as exc:
            return json.dumps({"error": exc.to_dict()})
        return json.dumps(result)

    return await_retrieval
