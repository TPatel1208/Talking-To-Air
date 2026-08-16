from __future__ import annotations

import csv
import io
import re
from typing import Any, AsyncIterator

import logging

from tta_backend.utils.colormaps import resolve as resolve_colormap

logger = logging.getLogger(__name__)


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


def _resolve_export_region(region_name: str):
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
    question is how one of them quietly keeps the old behaviour."""
    from tta_backend.earthdata_mcp.results import MCPToolError
    from tta_backend.utils.plotting import RegionResolver

    try:
        return RegionResolver().resolve_location(region_name)
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
                mesh = await self._plot_heatmap_axis(
                    axes[0][idx],
                    panel,
                    tools,
                    panel.get("region_name") or f"Panel {idx + 1}",
                )
            if mesh is not None:
                fig.colorbar(mesh, ax=axes.ravel().tolist(), label=export.get("units", ""))
        elif export_type == "timeseries":
            rows = await self._timeseries_rows(export, tools)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot([row[1] for row in rows], [row[3] for row in rows], marker="o", linewidth=1.5)
            ax.set_title(payload.get("title") or export.get("variable") or "Time series")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{export.get('aggregation', 'value')} ({export.get('units', '')})")
            ax.tick_params(axis="x", rotation=30)
        elif export_type == "profile":
            fig = self._plot_profile_figure(payload, export, plt)
        else:
            from tta_backend.utils.plotting import RegionResolver, plot_map

            da = await self._export_data_array(export, tools, collapse_to_2d=True)
            region = None
            region_name = export.get("region_name")
            if region_name:
                region = _resolve_export_region(region_name)
            fig, ax = plot_map(
                da,
                title=payload.get("title") or export.get("region_name") or "Chart",
                extent=region["bounds"] if region else None,
                mask_geometry=region["geometry"] if region else None,
                cmap=payload.get("colormap", {}).get("name") or resolve_colormap(export.get("variable")).name,
            )

        fig.tight_layout()
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=220, bbox_inches="tight")
        plt.close(fig)
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

    def _plot_profile_figure(self, payload: dict[str, Any], export: dict[str, Any], plt):
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

        fig, ax = plt.subplots(figsize=(6, 8))
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
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired
        from tta_backend.tools.satellite_tools.plot_tools import _normalize_longitudes, _sel_bounds
        from tta_backend.services.open_handle import open_handle
        from tta_backend.utils.plotting import RegionResolver, mask_data_by_geometry

        from tta_backend.earthdata_mcp.results import MCPToolError

        source_handles = export.get("source_handles") or []
        if not source_handles:
            raise ValueError("This chart does not include a source handle for full-resolution export.")
        ds = await open_handle(source_handles[0], tools)
        # Pass the source handle so a stored payload whose ``variable`` wasn't
        # persisted (or was persisted as None) still inherits the science
        # variable recorded for this handle at retrieval time (T25). If it
        # can't be resolved -- a genuinely multi-science-variable file with no
        # recorded choice -- surface it as the export path's own ValueError
        # (a clean 422) instead of letting the structured MCPToolError escape
        # mid-stream on the CSV path, where the response has already started
        # and no handler can turn it into a proper error response.
        try:
            da = AggregationService().to_dataarray(
                ds, variable=export.get("variable"), handle=source_handles[0],
            )
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

        region = None
        region_name = export.get("region_name")
        if region_name:
            region = _resolve_export_region(region_name)

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
        import numpy as np

        da = await self._export_data_array(export, tools, collapse_to_2d=True)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        lats = da[lat_coord].values
        lons = da[lon_coord].values
        values = da.values.astype(float)
        variable = export.get("variable", "")
        units = export.get("units", "")

        for row_idx, col_idx in zip(*np.where(np.isfinite(values))):
            row = []
            if panel_name is not None:
                row.append(panel_name)
            row.extend([variable, float(lats[row_idx]), float(lons[col_idx]), float(values[row_idx, col_idx]), units])
            yield row

    def _unique_headers(self, values: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        headers = []
        for value in values:
            base = value or "granule"
            counts[base] = counts.get(base, 0) + 1
            headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
        return headers

    async def _iter_aggregated_heatmap_csv_rows(self, export: dict[str, Any], tools: dict[str, Any], panel_name: str | None = None):
        import numpy as np
        import pandas as pd
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
            row_granules = [float(value) if np.isfinite(value) else "" for value in granule_values[:, row_idx, col_idx]]
            row = []
            if panel_name is not None:
                row.append(panel_name)
            row.extend([variable, float(lats[row_idx]), float(lons[col_idx]), *row_granules, float(mean_value) if np.isfinite(mean_value) else "", units])
            yield row

    async def _timeseries_rows(self, export: dict[str, Any], tools: dict[str, Any]):
        if export.get("aggregation") == "point sample":
            return await self._point_sample_timeseries_rows(export, tools)

        import numpy as np
        import pandas as pd
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = await self._export_data_array(export, tools, collapse_to_2d=False)
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

    async def _point_sample_timeseries_rows(self, export: dict[str, Any], tools: dict[str, Any]):
        from tta_backend.services.open_handle import open_handle
        from tta_backend.tools.satellite_tools.retrieval_tools import _series_from_table

        source_handles = export.get("source_handles") or []
        if not source_handles:
            raise ValueError("This chart does not include a source handle for full-resolution export.")

        variable = export.get("variable", "")
        table = await open_handle(source_handles[0], tools)
        times, values = _series_from_table(table, variable)

        stat = export.get("aggregation", "")
        units = export.get("units", "")
        return [[variable, time, stat, value, units] for time, value in zip(times, values)]

    async def _plot_heatmap_axis(self, ax, export: dict[str, Any], tools: dict[str, Any], title: str):
        da = await self._export_data_array(export, tools, collapse_to_2d=True)
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
