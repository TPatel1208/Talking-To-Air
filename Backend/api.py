import asyncio
import inspect
import uuid
import os
import re
import sys
import json
import logging
import time
import csv
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.supervisor_agent import build_agent, list_sessions, delete_session
from config.settings import get_settings
from models import parse_agent_result, parse_chart_payload
from repositories.chart_repository import ensure_chart_table, get_chart, save_chart
from utils.db import close_db_pool, init_db_pool, validate_config
from utils.logging import configure_logging
from utils.streaming import stream_response

agent = None
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    validate_config()
    await init_db_pool()
    await ensure_chart_table()

    logger.info("startup_begin", extra={"_model": settings.llm_model})
    agent = await build_agent(settings.llm_model)
    app.state.agent = agent
    logger.info("startup_complete")
    try:
        yield
    finally:
        agent = None
        app.state.agent = None
        await close_db_pool()
        logger.info("shutdown_complete")


app = FastAPI(title="Talking to Air API", lifespan=lifespan)
origins = settings.cors_origins

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str] = None


def normalize_image_url(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("/outputs/"):
        return raw
    filename = raw.replace("\\", "/").split("/")[-1]
    return f"/outputs/{filename}"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _persist_chart_payload_async(thread_id: str, chart) -> dict:
    payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
    if payload.get("chart_id"):
        stored = await get_chart(payload["chart_id"])
        if stored:
            return stored
    return await save_chart(thread_id, payload)


def _safe_export_name(payload: dict, suffix: str) -> str:
    name = payload.get("title") or payload.get("metadata", {}).get("name") or payload.get("type") or "chart"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name)).strip("-").lower()[:80] or "chart"
    return f"{safe}.{suffix}"


def _export_data_dict(export: dict) -> dict:
    fetch_params = export.get("fetch_params") or {}
    return {
        "variable": export.get("variable") or fetch_params.get("variable", ""),
        "units": export.get("units", ""),
        "bbox": ",".join(str(v) for v in fetch_params.get("bbox", [])),
        "source": export.get("source", ""),
        "fetch_params": fetch_params,
    }


def _export_lat_lon_names(da):
    lat_coord = next((c for c in ["lat", "latitude", "Latitude"] if c in da.coords), None)
    lon_coord = next((c for c in ["lon", "longitude", "Longitude"] if c in da.coords), None)
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
    return lat_coord, lon_coord


def _export_data_array(export: dict, collapse_to_2d: bool = True):
    from preprocessing.aggregation_service import AggregationService
    from tools.satellite_tools.plot_tools import _normalize_longitudes, _sel_bounds
    from utils.data_utils import _load_data
    from utils.plotting import RegionResolver, mask_data_by_geometry

    da = _load_data(_export_data_dict(export))
    lat_coord, lon_coord = _export_lat_lon_names(da)
    da = _normalize_longitudes(da, lon_coord)

    region = None
    region_name = export.get("region_name")
    if region_name:
        try:
            region = RegionResolver().resolve_location(region_name)
        except Exception:
            region = None

    if region:
        da = mask_data_by_geometry(da, region["geometry"])
        bounds = region["bounds"]
    else:
        bounds = (export.get("fetch_params") or {}).get("bbox")

    if bounds:
        lat_coord, lon_coord = _export_lat_lon_names(da)
        da = _sel_bounds(da, lat_coord, lon_coord, bounds)

    if collapse_to_2d:
        aggregation = AggregationService().aggregate(
            da,
            variable=export.get("variable") or (export.get("fetch_params") or {}).get("variable"),
            stat=(export.get("aggregation_meta") or {}).get("stat", "mean"),
        )
        da = next(iter(aggregation.ds.data_vars.values()))
        lat_coord, lon_coord = _export_lat_lon_names(da)
        if da.dims.index(lat_coord) != 0:
            da = da.transpose(lat_coord, lon_coord)

    return da


def _iter_heatmap_csv_rows(export: dict, panel_name: str | None = None):
    import numpy as np

    da = _export_data_array(export, collapse_to_2d=True)
    lat_coord, lon_coord = _export_lat_lon_names(da)
    lats = da[lat_coord].values
    lons = da[lon_coord].values
    values = da.values.astype(float)
    variable = export.get("variable", "")
    units = export.get("units", "")

    for row_idx, col_idx in zip(*np.where(np.isfinite(values))):
        row = []
        if panel_name is not None:
            row.append(panel_name)
        row.extend([
            variable,
            float(lats[row_idx]),
            float(lons[col_idx]),
            float(values[row_idx, col_idx]),
            units,
        ])
        yield row


def _write_heatmap_csv(writer, export: dict, panel_name: str | None = None) -> None:
    writer.writerows(_iter_heatmap_csv_rows(export, panel_name))


def _unique_headers(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    headers = []
    for value in values:
        base = value or "granule"
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


def _iter_aggregated_heatmap_csv_rows(export: dict, panel_name: str | None = None):
    import numpy as np
    import pandas as pd
    from preprocessing.aggregation_service import AggregationService

    da = _export_data_array(export, collapse_to_2d=False)
    lat_coord, lon_coord = _export_lat_lon_names(da)
    if "time" not in da.dims:
        yield from _iter_heatmap_csv_rows(export, panel_name)
        return

    meta = export.get("aggregation_meta") or {}
    granule_dates = list(meta.get("granule_dates") or [])
    if not granule_dates:
        granule_dates = [pd.Timestamp(v).isoformat()[:10] for v in da["time"].values]

    cap = settings.csv_export_max_granules
    capped = len(granule_dates) > cap
    granule_dates = granule_dates[:cap]
    granule_headers = _unique_headers(granule_dates)

    if capped:
        yield [f"# CSV granule columns capped at {cap}; additional granules omitted."]

    header = []
    if panel_name is not None:
        header.append("panel")
    header.extend(["variable", "latitude", "longitude", *granule_headers, "mean", "units"])
    yield header

    aggregation = AggregationService().aggregate(
        da,
        variable=export.get("variable") or (export.get("fetch_params") or {}).get("variable"),
        stat=meta.get("stat", "mean"),
    )
    mean_da = next(iter(aggregation.ds.data_vars.values()))
    lat_coord, lon_coord = _export_lat_lon_names(mean_da)
    if mean_da.dims.index(lat_coord) != 0:
        mean_da = mean_da.transpose(lat_coord, lon_coord)
    if da.dims[-2:] != (lat_coord, lon_coord):
        time_dim = next(d for d in da.dims if d not in (lat_coord, lon_coord))
        da = da.transpose(time_dim, lat_coord, lon_coord)

    lats = mean_da[lat_coord].values
    lons = mean_da[lon_coord].values
    mean_values = mean_da.values.astype(float)
    granule_count = min(len(granule_dates), da.sizes["time"])
    granule_values = da.isel(time=slice(0, granule_count)).values.astype(float)
    variable = export.get("variable", "")
    units = export.get("units", "")

    valid_mask = np.isfinite(mean_values)
    if granule_count:
        valid_mask = valid_mask | np.any(np.isfinite(granule_values), axis=0)

    for row_idx, col_idx in np.argwhere(valid_mask):
        mean_value = mean_values[row_idx, col_idx]
        row_granules = [
            float(value) if np.isfinite(value) else ""
            for value in granule_values[:, row_idx, col_idx]
        ]
        row = []
        if panel_name is not None:
            row.append(panel_name)
        row.extend([
            variable,
            float(lats[row_idx]),
            float(lons[col_idx]),
            *row_granules,
            float(mean_value) if np.isfinite(mean_value) else "",
            units,
        ])
        yield row


def _write_aggregated_heatmap_csv(writer, export: dict, panel_name: str | None = None) -> None:
    writer.writerows(_iter_aggregated_heatmap_csv_rows(export, panel_name))


def _timeseries_rows(export: dict):
    import numpy as np
    import pandas as pd
    from preprocessing.aggregation_service import AggregationService

    da = _export_data_array(export, collapse_to_2d=False)
    if "time" not in da.dims:
        raise ValueError("Time-series export requires a time dimension.")

    stat = export.get("aggregation") or export.get("chart_parameters", {}).get("stat") or "mean"
    service = AggregationService()
    if stat not in AggregationService._STAT_FUNCS:
        raise ValueError(f"Unsupported time-series statistic: {stat}")

    rows = []
    for i in range(da.sizes["time"]):
        arr = da.isel(time=i).values.astype(float)
        valid = arr[np.isfinite(arr)]
        if not len(valid):
            continue
        rows.append([
            export.get("variable", ""),
            pd.Timestamp(da["time"].values[i]).isoformat(),
            stat,
            service.compute_values_stat(valid, stat),
            export.get("units", ""),
        ])
    return rows


def _iter_chart_csv_rows(payload: dict):
    export = payload.get("export") or {}
    if not export:
        raise ValueError("This chart does not include full-resolution export metadata.")

    export_type = export.get("type")

    if export_type == "heatmap_multi":
        for idx, panel in enumerate(export.get("panels") or []):
            if panel.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                yield from _iter_aggregated_heatmap_csv_rows(panel, panel.get("region_name") or f"panel-{idx + 1}")
            else:
                if idx == 0:
                    yield ["panel", "variable", "latitude", "longitude", "value", "units"]
                yield from _iter_heatmap_csv_rows(panel, panel.get("region_name") or f"panel-{idx + 1}")
    elif export_type == "timeseries":
        yield ["variable", "time", "stat", "value", "units"]
        yield from _timeseries_rows(export)
    else:
        if export.get("aggregation_meta", {}).get("n_granules", 1) > 1:
            yield from _iter_aggregated_heatmap_csv_rows(export)
        else:
            yield ["variable", "latitude", "longitude", "value", "units"]
            yield from _iter_heatmap_csv_rows(export)


def _iter_chart_csv_chunks(payload: dict, chunk_size: int = 64 * 1024):
    output = io.StringIO()
    writer = csv.writer(output)

    for row in _iter_chart_csv_rows(payload):
        writer.writerow(row)
        if output.tell() >= chunk_size:
            yield output.getvalue().encode("utf-8")
            output.seek(0)
            output.truncate(0)

    remaining = output.getvalue()
    if remaining:
        yield remaining.encode("utf-8")


def _build_chart_csv(payload: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(_iter_chart_csv_rows(payload))

    return output.getvalue()


def _plot_heatmap_axis(ax, export: dict, title: str) -> None:
    da = _export_data_array(export, collapse_to_2d=True)
    lat_coord, lon_coord = _export_lat_lon_names(da)
    mesh = ax.pcolormesh(da[lon_coord].values, da[lat_coord].values, da.values.astype(float), shading="auto", cmap="Spectral_r")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return mesh


def _build_chart_png(payload: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    export = payload.get("export") or {}
    if not export:
        raise ValueError("This chart does not include full-resolution export metadata.")

    export_type = export.get("type")
    if export_type == "heatmap_multi":
        panels = export.get("panels") or []
        if not panels:
            raise ValueError("Comparison chart has no export panels.")
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
        mesh = None
        for idx, panel in enumerate(panels):
            mesh = _plot_heatmap_axis(axes[0][idx], panel, panel.get("region_name") or f"Panel {idx + 1}")
        if mesh is not None:
            fig.colorbar(mesh, ax=axes.ravel().tolist(), label=export.get("units", ""))
    elif export_type == "timeseries":
        rows = _timeseries_rows(export)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot([row[1] for row in rows], [row[3] for row in rows], marker="o", linewidth=1.5)
        ax.set_title(payload.get("title") or export.get("variable") or "Time series")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{export.get('aggregation', 'value')} ({export.get('units', '')})")
        ax.tick_params(axis="x", rotation=30)
    else:
        from utils.plotting import RegionResolver, plot_map

        da = _export_data_array(export, collapse_to_2d=True)
        region = None
        region_name = export.get("region_name")
        if region_name:
            try:
                region = RegionResolver().resolve_location(region_name)
            except Exception:
                region = None
        fig, ax = plot_map(
            da,
            title=payload.get("title") or export.get("region_name") or "Chart",
            extent=region["bounds"] if region else export.get("fetch_params", {}).get("bbox"),
            mask_geometry=region["geometry"] if region else None,
            cmap=payload.get("cmap") or export.get("chart_parameters", {}).get("cmap") or "Spectral_r",
        )

    fig.tight_layout()
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/chart/{chart_id}/export.csv")
async def export_chart_csv(chart_id: str):
    payload = await get_chart(chart_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Chart not found")
    try:
        export = payload.get("export") or {}
        if not export:
            raise ValueError("This chart does not include full-resolution export metadata.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StreamingResponse(
        _iter_chart_csv_chunks(payload),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_export_name(payload, "csv")}"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chart/{chart_id}/export.png")
async def export_chart_png(chart_id: str):
    payload = await get_chart(chart_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Chart not found")
    try:
        content = await asyncio.to_thread(_build_chart_png, payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return Response(
        content=content,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{_safe_export_name(payload, "png")}"'},
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    active_agent = getattr(app.state, "agent", None) or agent
    if active_agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    thread_id = req.thread_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    async def generate():
        response_text = ""
        image_urls    = []
        tool_calls    = []
        started = time.monotonic()

        try:
            async for event_type, data in stream_response(active_agent, req.message, thread_id):
                logger.debug(
                    "stream_event",
                    extra={"_request_id": request_id, "_thread_id": thread_id, "_event_type": event_type},
                )

                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    yield sse("tool_call", {"name": data["name"], "args": data["args"]})

                elif event_type == "status":
                    yield sse("status", {"message": data.get("message", "")})

                elif event_type == "tool_result":
                    content = data.get("content", "")

                    structured_result = parse_agent_result(content)
                    if structured_result is not None:
                        for chart in structured_result.charts:
                            yield sse("chart", await _persist_chart_payload_async(thread_id, chart))
                        continue

                    chart = parse_chart_payload(content)
                    if chart is not None:
                        yield sse("chart", await _persist_chart_payload_async(thread_id, chart))
                        continue

                    # Legacy: detect .png paths for any remaining static image tools
                    png_match = re.search(r'(/outputs/[\w\-./]+\.png|[\w\-./]+\.png)', content)
                    if png_match:
                        url = normalize_image_url(png_match.group(1))
                        if url:
                            image_urls.append(url)
                            yield sse("image", {"url": url})

                elif event_type == "image":
                    url = normalize_image_url(data.get("path", ""))
                    if url:
                        image_urls.append(url)
                        yield sse("image", {"url": url})

                elif event_type == "text":
                    if isinstance(data, str):
                        structured_result = parse_agent_result(data)
                        if structured_result is not None:
                            response_text += structured_result.text
                            for chart in structured_result.charts:
                                yield sse("chart", await _persist_chart_payload_async(thread_id, chart))
                        else:
                            response_text += data
                    elif isinstance(data, list):
                        for block in data:
                            if isinstance(block, str):
                                response_text += block
                            elif isinstance(block, dict):
                                if block.get("type") == "text":
                                    response_text += block.get("text", "")
                            elif hasattr(block, "text"):
                                response_text += block.text

            yield sse("done", {
                "thread_id":  thread_id,
                "response":   response_text,
                "image_urls": image_urls,
                "tool_calls": tool_calls,
            })
            elapsed = time.monotonic() - started
            if elapsed >= settings.long_request_seconds:
                logger.warning(
                    "long_running_request",
                    extra={"_request_id": request_id, "_thread_id": thread_id, "_elapsed_seconds": round(elapsed, 3)},
                )
            else:
                logger.info(
                    "request_completed",
                    extra={"_request_id": request_id, "_thread_id": thread_id, "_elapsed_seconds": round(elapsed, 3)},
                )

        except Exception as e:
            logger.exception(
                "agent_failure",
                extra={"_request_id": request_id, "_thread_id": thread_id},
            )
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions")
async def get_sessions():
    try:
        return {"sessions": await list_sessions()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{thread_id}/history")
async def get_history(thread_id: str):
    try:
        active_agent = getattr(app.state, "agent", None) or agent
        if active_agent is None:
            raise HTTPException(status_code=503, detail="Agent is not ready")
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(active_agent, "aget_state"):
            state = await active_agent.aget_state(config)
        else:
            maybe_state = active_agent.get_state(config)
            state = await maybe_state if inspect.isawaitable(maybe_state) else maybe_state
        if not state or not state.values:
            return {"messages": []}

        raw_messages = state.values.get("messages", [])
        result = []

        for msg in raw_messages:
            role = getattr(msg, "type", None)

            if role == "human":
                result.append({
                    "role":      "user",
                    "content":   msg.content if isinstance(msg.content, str) else "",
                    "toolCalls": [],
                    "imageUrls": [],
                })

            elif role == "ai":
                tool_calls = []
                seen_tool_ids = set()
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tid = tc.get("id", "")
                        seen_tool_ids.add(tid)
                        tool_calls.append({
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        })

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            # plain string block (Gemini stores response text this way)
                            content += block
                        elif isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                content += block.get("text", "")
                            elif btype == "thinking":
                                pass  # skip internal chain-of-thought
                            elif btype == "tool_use":
                                # tool dispatch in content list
                                tid = block.get("id", "")
                                if tid not in seen_tool_ids:
                                    seen_tool_ids.add(tid)
                                    tool_calls.append({
                                        "name": block.get("name", ""),
                                        "args": block.get("input", {}),
                                    })
                        elif hasattr(block, "text"):
                            content += block.text

                result.append({
                    "role":      "assistant",
                    "content":   content,
                    "toolCalls": tool_calls,
                    "imageUrls": [],
                    "charts":    [],
                })

            elif role == "tool":
                # Flatten tool result content — can be str or list of blocks
                if isinstance(msg.content, str):
                    tool_text = msg.content
                elif isinstance(msg.content, list):
                    parts = []
                    for block in msg.content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict):
                            parts.append(block.get("text", "") or str(block.get("content", "")))
                        elif hasattr(block, "text"):
                            parts.append(block.text)
                    tool_text = " ".join(parts)
                else:
                    tool_text = str(msg.content)

                # Find all .png paths anywhere in the tool result
                for png_match in re.finditer(r'(/outputs/[\w\-./]+\.png|[\w\-./]+\.png)', tool_text):
                    url = normalize_image_url(png_match.group(1))
                    if url:
                        for m in reversed(result):
                            if m["role"] == "assistant":
                                if url not in m["imageUrls"]:
                                    m["imageUrls"].append(url)
                                break

                charts = []
                structured_result = parse_agent_result(tool_text)
                if structured_result is not None:
                    charts.extend(structured_result.charts)
                else:
                    chart = parse_chart_payload(tool_text)
                    if chart is not None:
                        charts.append(chart)

                for chart in charts:
                    chart_payload = await _persist_chart_payload_async(thread_id, chart)
                    for m in reversed(result):
                        if m["role"] == "assistant":
                            m.setdefault("charts", [])
                            if chart_payload not in m["charts"]:
                                m["charts"].append(chart_payload)
                            break

        merged = []
        for msg in result:
            if (
                msg["role"] == "assistant"
                and merged
                and merged[-1]["role"] == "assistant"
            ):
                prev = merged[-1]
                prev["toolCalls"].extend(msg["toolCalls"])
                if msg["content"]:
                    prev["content"] += ("\n\n" if prev["content"] else "") + msg["content"]
                for url in msg["imageUrls"]:
                    if url not in prev["imageUrls"]:
                        prev["imageUrls"].append(url)
                for chart in msg.get("charts", []):
                    prev.setdefault("charts", [])
                    if chart not in prev["charts"]:
                        prev["charts"].append(chart)
            else:
                merged.append(msg)

        return {"messages": merged}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{thread_id}")
async def remove_session(thread_id: str):
    try:
        await delete_session(thread_id)
        return {"deleted": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/debug/{thread_id}")
async def debug_history(thread_id: str):
    active_agent = getattr(app.state, "agent", None) or agent
    if active_agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    config = {"configurable": {"thread_id": thread_id}}
    if hasattr(active_agent, "aget_state"):
        state = await active_agent.aget_state(config)
    else:
        maybe_state = active_agent.get_state(config)
        state = await maybe_state if inspect.isawaitable(maybe_state) else maybe_state
    raw = state.values.get("messages", [])
    return [
        {
            "type": getattr(m, "type", None),
            "content_type": type(m.content).__name__,
            "content_preview": str(m.content)[:300],
            "has_tool_calls": bool(getattr(m, "tool_calls", None)),
        }
        for m in raw
    ]
