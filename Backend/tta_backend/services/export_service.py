from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import csv
import functools
import io
import re
import threading
from typing import Any, AsyncIterator, NamedTuple

import logging

from tta_backend.utils.colormaps import resolve as resolve_colormap

logger = logging.getLogger(__name__)

# Exports run on their own bounded pool, never on the default executor.
#
# ``asyncio.to_thread`` submits to the process-wide default ThreadPoolExecutor
# -- sized ``min(32, cpu_count + 4)``, so six threads on a two-core container
# -- and that pool is also where every request's JWT verification runs
# (api.py), alongside handle opens, overlay reads, and the to_thread sites in
# plot_tools/stat_tools. Ten concurrent exports, each holding a thread for a
# multi-second grid reduction, would saturate it and leave arriving requests
# queued at authentication: the freeze relocated from the event loop to the
# executor rather than being bounded.
#
# A private pool bounds it. Exports queue among themselves, the default
# executor stays free for auth and the rest of the app, and the ceiling also
# caps how many full-resolution grids can be in memory at once.
_EXPORT_MAX_WORKERS = 4
_export_executor: concurrent.futures.ThreadPoolExecutor | None = None
_export_executor_lock = threading.Lock()


def _get_export_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _export_executor

    if _export_executor is None:
        with _export_executor_lock:
            if _export_executor is None:
                _export_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_EXPORT_MAX_WORKERS,
                    thread_name_prefix="export",
                )
    return _export_executor


async def _to_export_thread(func, *args):
    """``asyncio.to_thread`` against the export pool.

    The context copy is not incidental: ``run_in_executor`` starts the call
    with an empty context, and ``current_user_id()`` is a ContextVar the
    workspace-bound MCP tools read to decide whose data to open. Losing it
    does not raise -- the export opens the wrong workspace or none -- so this
    reproduces exactly what ``to_thread`` does for its callers.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        _get_export_executor(), functools.partial(ctx.run, func, *args),
    )


# How many CSV rows one worker-thread hop converts. Small enough that the
# rows it holds are a rounding error next to the grid they come from, large
# enough that a full-resolution export is not thousands of hops.
_CSV_ROW_BLOCK = 20_000


async def materialize_first_chunk(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Await ``chunks``' first chunk before the caller commits to a response
    (T37): the common export failures (not-ready tools, missing handle,
    evicted export) surface while producing that first chunk, and letting
    them raise *here* — before any 200 header is sent — turns a silently
    truncated download into a clean 4xx/5xx. Returns an iterator that
    replays the materialized chunk and then streams the rest."""
    try:
        first = await chunks.__anext__()
    except StopAsyncIteration:
        first = None
    return _replay_then_stream(first, chunks)


async def _resolve_export_region(region_name: str):
    """Resolve an export's region, tolerating a *lookup failure* and refusing
    to tolerate a *refusal* (T60 D14).

    Both export paths used to wrap this in a bare ``except Exception: region =
    None``, which is a reasonable guard against a Nominatim timeout -- an
    unmasked chart beats a failed download -- and a catastrophe against a
    region the resolver deliberately declined to guess at. With no region the
    export renders ``extent=None, mask_geometry=None``: a chart over the
    **entire globe**, carrying the region name the researcher asked for. That
    is the T46 silent-scope-substitution failure arriving through the export
    path, and D14's raised error would have walked straight into it.

    So ``MCPToolError`` propagates (the API's handler turns it into a clean
    4xx naming the token) and everything else still degrades as before.
    Extracted to one function because two call sites drifting apart on this
    question is how one of them quietly keeps the old behaviour.

    Asynchronous because of the resolver it reaches. The sync twin's throttle
    is a bare ``time.sleep`` and its request a blocking ``requests.get`` --
    the geocoder's own comment named this module as the caller that ran them
    on the event loop -- and because the throttle is process-global, the waits
    of concurrent exports stack rather than overlap. ``aresolve_location``
    yields the loop across both, so a queue of uncached regions costs the same
    wall-clock time and freezes nothing while it waits.
    """
    from tta_backend.earthdata_mcp.results import MCPToolError
    from tta_backend.utils.plotting import RegionResolver

    try:
        return await RegionResolver().aresolve_location(region_name)
    except MCPToolError:
        raise
    except Exception:
        logger.warning("export_region_lookup_failed", extra={"_region": region_name})
        return None


async def _replay_then_stream(first: bytes | None, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT, MCPToolError

    if first is not None:
        yield first
    try:
        async for chunk in rest:
            yield chunk
    except MCPToolError as exc:
        # The 200 is already committed — the honest remaining move is a
        # clearly marked trailer so the file self-identifies as truncated
        # (T37 story #2). Category only, never the message: the trailer is
        # file content a researcher may share onward.
        logger.exception("export_stream_failed_mid_stream", extra={"_category": exc.category})
        yield f"\n# EXPORT INCOMPLETE — {exc.category}\n".encode("utf-8")
    except Exception:
        logger.exception("export_stream_failed_mid_stream", extra={"_category": CATEGORY_CONTRACT})
        yield f"\n# EXPORT INCOMPLETE — {CATEGORY_CONTRACT}\n".encode("utf-8")


class _HeatmapGrid(NamedTuple):
    """A materialised 2-D grid plus the indices of its finite cells.

    Computed once, on a worker thread, and then read from block by block --
    so the arrays are touched off the event loop and the Python rows they
    become are built off it too, without ever holding every row at once.
    """

    lats: Any
    lons: Any
    values: Any
    indices: Any


def _new_figure(width: float, height: float):
    """A figure with its own Agg canvas, and nothing filed anywhere global.

    The canvas is attached eagerly rather than left to ``savefig`` because
    attaching it is what keeps this off ``pyplot``: ``plt.subplots`` would
    give a canvas *and* an entry in a process-global registry that concurrent
    renders on worker threads would then share. ``plot_map`` builds its own
    rather than calling this, because its axes need a cartopy projection at
    construction time.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def _at(values: Any, index: int):
    """``values[index]`` when it exists, else None — so one short array (an
    axis a product doesn't publish, say) leaves a blank cell rather than
    truncating or misaligning every column after it."""
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    return values[index]


class ExportService:
    def __init__(self, csv_export_max_granules: int = 50):
        self.csv_export_max_granules = csv_export_max_granules

    def safe_export_name(self, payload: dict[str, Any], suffix: str) -> str:
        name = payload.get("title") or payload.get("metadata", {}).get("name") or payload.get("type") or "chart"
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name)).strip("-").lower()[:80] or "chart"
        return f"{safe}.{suffix}"

    async def iter_chart_csv_chunks(
        self,
        payload: dict[str, Any],
        tools: dict[str, Any],
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        output = io.StringIO()
        writer = csv.writer(output)

        async for row in self.iter_chart_csv_rows(payload, tools):
            writer.writerow(row)
            if output.tell() >= chunk_size:
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        remaining = output.getvalue()
        if remaining:
            yield remaining.encode("utf-8")

    async def build_chart_png(self, payload: dict[str, Any], tools: dict[str, Any]) -> bytes:
        """Render the chart to PNG bytes without freezing the event loop.

        The split below is the whole point of this method's shape. Everything
        that must ``await`` -- opening a source handle, reading its granule,
        looking up a region -- happens *here*, on the loop. Everything that
        follows is pure synchronous compute (cartopy projection, geometry
        masking, ``savefig(dpi=220)``) and is handed to a worker thread.

        Before the split this was an ``async def`` with an entirely
        synchronous body, so a single export froze the one uvicorn worker's
        one loop for its whole duration: every other user's SSE stream, the
        heartbeat nginx times out without, and the ``/health`` check Docker
        restarts the container over. The awaits cannot simply move into the
        thread with the rest -- there is no loop in there to await on -- which
        is why the prefetch is a separate phase rather than a wrapper.
        """
        export = payload.get("export") or {}
        if not export:
            raise ValueError("This chart does not include full-resolution export metadata.")

        export_type = export.get("type")
        if export_type == "heatmap_multi":
            panels = export.get("panels") or []
            if not panels:
                raise ValueError("Comparison chart has no export panels.")
            # One panel at a time, drawn and released before the next is
            # opened. Prefetching them all into a list first reads more
            # simply and holds every panel's full-resolution grid at once --
            # on a path that has already been OOM-killed once, a four-panel
            # compare over a full-domain product is exactly where that shows
            # up. Once ``pcolormesh`` has drawn a panel it owns its own copy,
            # so the source array is dead here and the next iteration's
            # rebinding drops it.
            fig = _new_figure(6 * len(panels), 5)
            axes = fig.subplots(1, len(panels), squeeze=False)
            mesh = None
            for idx, panel in enumerate(panels):
                da = await self._export_data_array(panel, tools, collapse_to_2d=True)
                mesh = await _to_export_thread(
                    self._plot_heatmap_axis,
                    axes[0][idx],
                    panel,
                    da,
                    panel.get("region_name") or f"Panel {idx + 1}",
                )
                del da
            if mesh is not None:
                fig.colorbar(mesh, ax=axes.ravel().tolist(), label=export.get("units", ""))
            prefetched: Any = fig
        elif export_type == "timeseries":
            prefetched = await self._timeseries_rows(export, tools)
        elif export_type == "profile":
            prefetched = None
        else:
            da = await self._export_data_array(export, tools, collapse_to_2d=True)
            region = None
            region_name = export.get("region_name")
            if region_name:
                region = await _resolve_export_region(region_name)
            prefetched = (da, region)

        return await _to_export_thread(self._render_png, payload, export, prefetched)

    def _render_png(self, payload: dict[str, Any], export: dict[str, Any], prefetched: Any) -> bytes:
        """The synchronous half: figure, draw, save. Runs on a worker thread,
        never on the event loop -- see :meth:`build_chart_png`."""
        export_type = export.get("type")
        if export_type == "heatmap_multi":
            # Already drawn, panel by panel, as each panel's array arrived --
            # see build_chart_png. Only the save is left.
            fig = prefetched
        elif export_type == "timeseries":
            rows = prefetched
            fig = _new_figure(9, 5)
            ax = fig.subplots()
            ax.plot([row[1] for row in rows], [row[3] for row in rows], marker="o", linewidth=1.5)
            ax.set_title(payload.get("title") or export.get("variable") or "Time series")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{export.get('aggregation', 'value')} ({export.get('units', '')})")
            ax.tick_params(axis="x", rotation=30)
        elif export_type == "profile":
            fig = self._plot_profile_figure(payload, export)
        else:
            from tta_backend.utils.plotting import plot_map

            da, region = prefetched
            fig, ax = plot_map(
                da,
                title=payload.get("title") or export.get("region_name") or "Chart",
                extent=region["bounds"] if region else None,
                mask_geometry=region["geometry"] if region else None,
                cmap=payload.get("colormap", {}).get("name") or resolve_colormap(export.get("variable")).name,
            )

        fig.tight_layout()
        output = io.BytesIO()
        # No ``plt.close`` to pair with: the figure was never registered
        # anywhere, so it is collected with the rest of this frame -- including
        # when ``savefig`` raises, which used to skip the close and leak.
        fig.savefig(output, format="png", dpi=220, bbox_inches="tight")
        return output.getvalue()

    async def iter_chart_csv_rows(self, payload: dict[str, Any], tools: dict[str, Any]):
        export = payload.get("export") or {}
        if not export:
            raise ValueError("This chart does not include full-resolution export metadata.")

        export_type = export.get("type")
        if export_type == "heatmap_multi":
            for idx, panel in enumerate(export.get("panels") or []):
                if panel.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                    async for row in self._iter_aggregated_heatmap_csv_rows(
                        panel,
                        tools,
                        panel.get("region_name") or f"panel-{idx + 1}",
                    ):
                        yield row
                else:
                    if idx == 0:
                        yield ["panel", "variable", "latitude", "longitude", "value", "units"]
                    async for row in self._iter_heatmap_csv_rows(panel, tools, panel.get("region_name") or f"panel-{idx + 1}"):
                        yield row
        elif export_type == "timeseries":
            yield ["variable", "time", "stat", "value", "units"]
            for row in await self._timeseries_rows(export, tools):
                yield row
        elif export_type == "profile":
            for row in self._profile_rows(export):
                yield row
        else:
            if export.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                async for row in self._iter_aggregated_heatmap_csv_rows(export, tools):
                    yield row
            else:
                yield ["variable", "latitude", "longitude", "value", "units"]
                async for row in self._iter_heatmap_csv_rows(export, tools):
                    yield row

    # ── Vertical profile (T56) ──────────────────────────────────────────────
    #
    # Both profile exports read the payload and nothing else. Every other chart
    # type re-opens its source granule because what it displays is a thinned
    # version of what it measured; a profile's 24 numbers per axis ARE the
    # measurement, so a re-read could only reproduce them -- and the export
    # keeps working after the source handle is evicted, which no other type
    # manages.

    _PROFILE_AXES = ("pressure", "altitude")

    def _profile_rows(self, export: dict[str, Any]):
        """One row per layer, carrying BOTH vertical axes and the per-layer
        spread of each.

        The spread column is the honest half of a regional profile: the vertical
        grid is fixed aloft and terrain-following near the surface, so the
        regional-mean axis is exact for the upper layers and an approximation
        below, and only a per-layer number distinguishes them. A CSV without it
        reads as though every layer sat at a single, definite pressure.
        """
        axes = export.get("vertical") or {}
        header = ["variable", "layer", "value", "units"]
        for kind in self._PROFILE_AXES:
            header += [kind, f"{kind}_units", f"{kind}_spread"]
        header.append("valid_fraction")
        yield header

        variable = export.get("variable") or ""
        units = export.get("units") or ""
        layers = export.get("layers") or []
        values = export.get("values") or []
        valid = export.get("valid_fraction") or []
        for index, layer in enumerate(layers):
            row = [variable, layer, _at(values, index), units]
            for kind in self._PROFILE_AXES:
                axis = axes.get(kind) or {}
                row += [
                    _at(axis.get("values"), index),
                    axis.get("units", ""),
                    _at(axis.get("spread"), index),
                ]
            row.append(_at(valid, index))
            yield row

    def _plot_profile_figure(self, payload: dict[str, Any], export: dict[str, Any]):
        """The profile as a static line chart, drawn against its physical axis.

        Plotting against the axis rather than the layer index is what makes the
        orientation right: TEMPO_O3PROF stores layer 0 at the TOP, so a chart
        that trusts the index draws the atmosphere upside down and looks
        entirely plausible doing it. With pressures on y and the axis reversed,
        the surface lands at the bottom whichever order the array arrived in --
        so the payload's measured ``layer_order`` is disclosure for the reader,
        never an input here. Pressure additionally gets a log scale: the top
        layer sits three orders of magnitude below the surface, and linearly the
        whole upper atmosphere collapses onto one pixel. Altitude needs neither,
        since it already increases upward.
        """
        kind = export.get("default_axis") or "pressure"
        axis = (export.get("vertical") or {}).get(kind) or {}
        values = export.get("values") or []
        axis_values = axis.get("values") or list(range(len(values)))

        fig = _new_figure(6, 8)
        ax = fig.subplots()
        ax.plot(values, axis_values, marker="o", linewidth=1.5)
        ax.set_title(payload.get("title") or export.get("variable") or "Vertical profile")
        ax.set_xlabel(f"{export.get('variable', 'value')} ({export.get('units', '')})")
        ax.set_ylabel(f"{kind} ({axis.get('units', '')})")
        if kind == "pressure":
            positive = [v for v in axis_values if isinstance(v, (int, float)) and v > 0]
            if len(positive) == len(axis_values) and positive:
                ax.set_yscale("log")
            ax.invert_yaxis()  # pressure falls with height
        ax.grid(True, which="both", alpha=0.3)
        return fig

    def _export_lat_lon_names(self, da):
        lat_coord = next((c for c in ["lat", "latitude", "Latitude"] if c in da.coords), None)
        lon_coord = next((c for c in ["lon", "longitude", "Longitude"] if c in da.coords), None)
        if lat_coord is None or lon_coord is None:
            raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
        return lat_coord, lon_coord

    async def _export_data_array(self, export: dict[str, Any], tools: dict[str, Any], collapse_to_2d: bool = True):
        """Open the source granule and narrow it to what the export draws.

        Only the two genuinely asynchronous steps happen here: the handle open
        (already offloaded by T16) and the region lookup. The rest -- variable
        resolution, longitude normalisation, geometry masking, the bounds
        selection and the dask reduction -- is synchronous CPU work on a full
        grid and belongs on a worker thread, not on the one loop that also
        carries every other user's stream.

        The region lookup has to move *ahead* of the narrowing to get there:
        it is a coroutine, and there is no loop inside the thread to await it
        on. It reads the payload and nothing from the array, so the data is
        unaffected -- but the ordering is visible in one case. An export whose
        variable cannot be resolved *and* whose region the resolver refuses
        now surfaces the region refusal rather than the variable error. That
        is the better of the two to lead with (D14's refusal is the one that
        silently becomes a globe-wide chart if it is ever dropped), and the
        cost is a geocode spent on an export that was going to fail anyway.
        """
        from tta_backend.services.open_handle import open_handle

        source_handles = export.get("source_handles") or []
        if not source_handles:
            raise ValueError("This chart does not include a source handle for full-resolution export.")
        ds = await open_handle(source_handles[0], tools)

        region = None
        region_name = export.get("region_name")
        if region_name:
            region = await _resolve_export_region(region_name)

        return await _to_export_thread(
            self._narrow_data_array, ds, export, source_handles[0], region, collapse_to_2d,
        )

    def _narrow_data_array(self, ds, export: dict[str, Any], handle: str, region, collapse_to_2d: bool):
        """The synchronous half of :meth:`_export_data_array`. Runs on a
        worker thread; raises exactly what it raised on the loop, since
        ``to_thread`` re-raises into the awaiting caller unchanged."""
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired
        from tta_backend.tools.satellite_tools.plot_tools import _normalize_longitudes, _sel_bounds
        from tta_backend.utils.plotting import mask_data_by_geometry

        from tta_backend.earthdata_mcp.results import MCPToolError

        # Pass the source handle so a stored payload whose ``variable`` wasn't
        # persisted (or was persisted as None) still inherits the science
        # variable recorded for this handle at retrieval time (T25). If it
        # can't be resolved -- a genuinely multi-science-variable file with no
        # recorded choice -- surface it as the export path's own ValueError
        # (a clean 422) instead of letting the structured MCPToolError escape
        # mid-stream on the CSV path, where the response has already started
        # and no handler can turn it into a proper error response.
        try:
            da = AggregationService().to_dataarray(ds, variable=export.get("variable"), handle=handle)
        except VariableChoiceRequired as exc:
            # T49's interactive picker is a chat-turn affordance; a raw export
            # download has no chat surface to attach it to, so this path keeps
            # the pre-T49 behavior -- surface the bounded refusal as the export's
            # own clean 422 ValueError.
            raise ValueError(exc.mcp_error.message) from exc
        except MCPToolError as exc:
            raise ValueError(exc.message) from exc
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        da = _normalize_longitudes(da, lon_coord)

        bounds = None
        if region:
            da = mask_data_by_geometry(da, region["geometry"])
            bounds = region["bounds"]

        if bounds:
            lat_coord, lon_coord = self._export_lat_lon_names(da)
            da = _sel_bounds(da, lat_coord, lon_coord, bounds)

        if collapse_to_2d:
            aggregation = AggregationService().aggregate(
                da,
                variable=export.get("variable"),
                stat=(export.get("aggregation_meta") or {}).get("stat", "mean"),
            )
            da = next(iter(aggregation.ds.data_vars.values()))
            lat_coord, lon_coord = self._export_lat_lon_names(da)
            if da.dims.index(lat_coord) != 0:
                da = da.transpose(lat_coord, lon_coord)

        return da

    async def _iter_heatmap_csv_rows(self, export: dict[str, Any], tools: dict[str, Any], panel_name: str | None = None):
        da = await self._export_data_array(export, tools, collapse_to_2d=True)
        grid = await _to_export_thread(self._heatmap_grid, da)
        variable = export.get("variable", "")
        units = export.get("units", "")
        # Blocked rather than materialised whole: turning a full-resolution
        # grid into one row per finite cell is the CSV's other blocking half,
        # and handing the *entire* conversion to one thread would trade a
        # frozen loop for a list of every row in memory at once. A block at a
        # time keeps both bounded, and the generator still streams.
        for start in range(0, len(grid.indices), _CSV_ROW_BLOCK):
            rows = await _to_export_thread(
                self._heatmap_rows, grid, start, variable, units, panel_name,
            )
            for row in rows:
                yield row

    def _heatmap_grid(self, da) -> _HeatmapGrid:
        import numpy as np

        lat_coord, lon_coord = self._export_lat_lon_names(da)
        values = da.values.astype(float)
        return _HeatmapGrid(
            lats=da[lat_coord].values,
            lons=da[lon_coord].values,
            values=values,
            indices=np.argwhere(np.isfinite(values)),
        )

    def _heatmap_rows(self, grid: _HeatmapGrid, start: int, variable: str, units: str, panel_name: str | None):
        rows = []
        for row_idx, col_idx in grid.indices[start:start + _CSV_ROW_BLOCK]:
            row = []
            if panel_name is not None:
                row.append(panel_name)
            row.extend([
                variable,
                float(grid.lats[row_idx]),
                float(grid.lons[col_idx]),
                float(grid.values[row_idx, col_idx]),
                units,
            ])
            rows.append(row)
        return rows

    def _unique_headers(self, values: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        headers = []
        for value in values:
            base = value or "granule"
            counts[base] = counts.get(base, 0) + 1
            headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
        return headers

    async def _iter_aggregated_heatmap_csv_rows(self, export: dict[str, Any], tools: dict[str, Any], panel_name: str | None = None):
        import pandas as pd

        da = await self._export_data_array(export, tools, collapse_to_2d=False)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        if "time" not in da.dims:
            async for row in self._iter_heatmap_csv_rows(export, tools, panel_name):
                yield row
            return

        meta = export.get("aggregation_meta") or {}
        granule_dates = list(meta.get("granule_dates") or [])
        if not granule_dates:
            granule_dates = [pd.Timestamp(v).isoformat()[:10] for v in da["time"].values]

        cap = self.csv_export_max_granules
        capped = len(granule_dates) > cap
        granule_dates = granule_dates[:cap]
        granule_headers = self._unique_headers(granule_dates)

        if capped:
            yield [f"# CSV granule columns capped at {cap}; additional granules omitted."]

        header = []
        if panel_name is not None:
            header.append("panel")
        header.extend(["variable", "latitude", "longitude", *granule_headers, "mean", "units"])
        yield header

        # The reduction and both materialisations -- the per-granule cube and
        # the reduced mean -- in one worker-thread hop. This is the heaviest
        # single step on the CSV path: it holds the whole cube, not just a
        # plane, so leaving it on the loop froze the worker for the length of
        # a multi-granule read.
        grid, granule_values = await _to_export_thread(
            self._aggregated_heatmap_grid, da, export, meta, len(granule_dates),
        )
        variable = export.get("variable", "")
        units = export.get("units", "")
        for start in range(0, len(grid.indices), _CSV_ROW_BLOCK):
            rows = await _to_export_thread(
                self._aggregated_heatmap_rows,
                grid, granule_values, start, variable, units, panel_name,
            )
            for row in rows:
                yield row

    def _aggregated_heatmap_grid(self, da, export: dict[str, Any], meta: dict[str, Any], granule_cap: int):
        import numpy as np
        from tta_backend.preprocessing.aggregation_service import AggregationService

        aggregation = AggregationService().aggregate(
            da,
            variable=export.get("variable"),
            stat=meta.get("stat", "mean"),
        )
        mean_da = next(iter(aggregation.ds.data_vars.values()))
        lat_coord, lon_coord = self._export_lat_lon_names(mean_da)
        if mean_da.dims.index(lat_coord) != 0:
            mean_da = mean_da.transpose(lat_coord, lon_coord)
        if da.dims[-2:] != (lat_coord, lon_coord):
            time_dim = next(d for d in da.dims if d not in (lat_coord, lon_coord))
            da = da.transpose(time_dim, lat_coord, lon_coord)

        mean_values = mean_da.values.astype(float)
        granule_count = min(granule_cap, da.sizes["time"])
        granule_values = da.isel(time=slice(0, granule_count)).values.astype(float)

        valid_mask = np.isfinite(mean_values)
        if granule_count:
            valid_mask = valid_mask | np.any(np.isfinite(granule_values), axis=0)

        grid = _HeatmapGrid(
            lats=mean_da[lat_coord].values,
            lons=mean_da[lon_coord].values,
            values=mean_values,
            indices=np.argwhere(valid_mask),
        )
        return grid, granule_values

    def _aggregated_heatmap_rows(self, grid: _HeatmapGrid, granule_values, start: int,
                                 variable: str, units: str, panel_name: str | None):
        import numpy as np

        rows = []
        for row_idx, col_idx in grid.indices[start:start + _CSV_ROW_BLOCK]:
            mean_value = grid.values[row_idx, col_idx]
            row_granules = [
                float(value) if np.isfinite(value) else ""
                for value in granule_values[:, row_idx, col_idx]
            ]
            row = []
            if panel_name is not None:
                row.append(panel_name)
            row.extend([
                variable,
                float(grid.lats[row_idx]),
                float(grid.lons[col_idx]),
                *row_granules,
                float(mean_value) if np.isfinite(mean_value) else "",
                units,
            ])
            rows.append(row)
        return rows

    async def _timeseries_rows(self, export: dict[str, Any], tools: dict[str, Any]):
        if export.get("aggregation") == "point sample":
            return await self._point_sample_timeseries_rows(export, tools)

        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = await self._export_data_array(export, tools, collapse_to_2d=False)
        if "time" not in da.dims:
            raise ValueError("Time-series export requires a time dimension.")

        stat = export.get("aggregation") or export.get("chart_parameters", {}).get("stat") or "mean"
        if stat not in AggregationService._STAT_FUNCS:
            raise ValueError(f"Unsupported time-series statistic: {stat}")

        # One hop for the whole loop: the output is one row per timestep, so
        # it is bounded by the number of granules, but reaching it reads every
        # timestep's grid in full.
        return await _to_export_thread(self._reduce_timeseries_rows, da, export, stat)

    def _reduce_timeseries_rows(self, da, export: dict[str, Any], stat: str):
        import numpy as np
        import pandas as pd
        from tta_backend.preprocessing.aggregation_service import AggregationService

        service = AggregationService()
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

    async def _point_sample_timeseries_rows(self, export: dict[str, Any], tools: dict[str, Any]):
        from tta_backend.services.open_handle import open_handle
        from tta_backend.tools.satellite_tools.retrieval_tools import _series_from_table

        source_handles = export.get("source_handles") or []
        if not source_handles:
            raise ValueError("This chart does not include a source handle for full-resolution export.")

        variable = export.get("variable", "")
        table = await open_handle(source_handles[0], tools)
        # The one export path that still ran its materialisation on the loop.
        # _series_from_table calls to_pylist() on both columns and then sorts
        # a generator that builds a pd.Timestamp per row -- for a multi-year
        # hourly series that is ~90k Timestamp constructions plus an O(n log n)
        # sort, which is the same freeze the gridded paths were moved off.
        times, values = await _to_export_thread(_series_from_table, table, variable)

        stat = export.get("aggregation", "")
        units = export.get("units", "")
        return [[variable, time, stat, value, units] for time, value in zip(times, values)]

    def _plot_heatmap_axis(self, ax, export: dict[str, Any], da, title: str):
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        mesh = ax.pcolormesh(
            da[lon_coord].values,
            da[lat_coord].values,
            da.values.astype(float),
            shading="auto",
            cmap=resolve_colormap(export.get("variable")).name,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return mesh
