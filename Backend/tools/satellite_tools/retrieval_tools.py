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
from services.open_handle import OpenHandleError, open_handle
from services.retrieval_composites import RetrievalTimeoutError
from services.retrieval_composites import await_retrieval as _await_retrieval
from services.retrieval_composites import point_timeseries as _point_timeseries
from services.retrieval_composites import safe_retrieve as _safe_retrieve
from tools.satellite_tools.plot_tools import _save_chart


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


def _series_from_table(table, variable: str) -> tuple[list[str], list[float]]:
    """Extract a sorted (times, values) series from a point-sampled Parquet
    table: the time column (``time``/``date``, else the first column) paired
    with ``variable``'s column (else the first remaining column)."""
    import pandas as pd

    columns = table.column_names
    time_col = next((c for c in ("time", "date") if c in columns), columns[0])
    value_col = variable if variable in columns else next((c for c in columns if c != time_col), None)
    if value_col is None:
        return [], []

    times_raw = table.column(time_col).to_pylist()
    values_raw = table.column(value_col).to_pylist()
    paired = sorted(
        (pd.Timestamp(t).isoformat(), round(float(v), 6))
        for t, v in zip(times_raw, values_raw)
        if v is not None
    )
    if not paired:
        return [], []
    times, values = zip(*paired)
    return list(times), list(values)


def _table_units(table) -> str:
    metadata = table.schema.metadata or {}
    units = metadata.get(b"units")
    return units.decode() if units else ""


def make_point_timeseries(mcp_tools: dict[str, BaseTool]):
    @tool
    async def point_timeseries(
        dataset_handle: str,
        location: str,
        time_range: str,
        variable: str,
    ) -> str:
        """
        Retrieve a pollutant's history at one place over time and render it
        as a line chart — the one call for "how did X change at [place]
        over [period]" questions, standing in for define_area_of_interest +
        safe_retrieve + await_retrieval + a chart tool.

        Always point-sampled (the MCP routes this to AppEEARS, never a
        gridded cube). For area-mean trends over a region, retrieve a cube
        with safe_retrieve/await_retrieval and use
        conduct_temporal_statistic instead.

        Args:
            dataset_handle : dataset_ handle from search_datasets.
            location       : place name or point to sample the series at.
            time_range     : ISO 8601 interval, e.g. '2024-01-01/2024-01-31'.
                              Refused if it exceeds the configured span gate.
            variable       : single variable short name to sample.

        Returns:
            JSON string — compact chart summary (T13) with the artifact id,
            or the terminal job status verbatim if the retrieval failed.
        """
        try:
            status = await _point_timeseries(dataset_handle, location, time_range, variable, mcp_tools)
        except MCPToolError as exc:
            return json.dumps({"error": exc.to_dict()})

        if status.get("status") != "ready":
            return json.dumps(status)

        handle = status.get("obs_handle")
        try:
            table = await open_handle(handle, mcp_tools)
        except MCPToolError as exc:
            return json.dumps({"error": exc.to_dict()})
        except OpenHandleError as exc:
            return json.dumps({"error": f"Failed to open handle '{handle}': {exc}"})

        times, values = _series_from_table(table, variable)
        if not times:
            return json.dumps({"error": f"No data found in the point timeseries result for '{variable}'."})

        units = _table_units(table)
        payload = {
            "type": "timeseries",
            "title": f"{variable} at {location}",
            "variable": variable,
            "units": units,
            "stat": "point",
            "times": times,
            "values": values,
            "provenance": {
                "variable": variable,
                "start_date": times[0],
                "end_date": times[-1],
                "region_name": location,
                "aggregation": "point sample",
                "units": units,
                "source_handles": [handle],
            },
            "query": {
                "dataset": dataset_handle,
                "start_date": times[0],
                "end_date": times[-1],
                "aggregation": "point sample",
                "chart_parameters": {"chart_type": "timeseries", "location": location},
            },
            "export": {
                "type": "timeseries",
                "variable": variable,
                "units": units,
                "region_name": location,
                "aggregation": "point sample",
                "chart_parameters": {"chart_type": "timeseries", "location": location},
                "source_handles": [handle],
            },
            "metadata": {"source_handles": [handle]},
        }
        return _save_chart(payload, f"{variable}_{location}_point_timeseries")

    return point_timeseries
