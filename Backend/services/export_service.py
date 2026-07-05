from __future__ import annotations

import csv
import io
import re
from typing import Any, AsyncIterator, Iterable


class ExportService:
    def __init__(self, csv_export_max_granules: int = 50):
        self.csv_export_max_granules = csv_export_max_granules

    def safe_export_name(self, payload: dict[str, Any], suffix: str) -> str:
        name = payload.get("title") or payload.get("metadata", {}).get("name") or payload.get("type") or "chart"
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name)).strip("-").lower()[:80] or "chart"
        return f"{safe}.{suffix}"

    def iter_chart_csv_chunks(self, payload: dict[str, Any], chunk_size: int = 64 * 1024) -> Iterable[bytes]:
        output = io.StringIO()
        writer = csv.writer(output)

        for row in self.iter_chart_csv_rows(payload):
            writer.writerow(row)
            if output.tell() >= chunk_size:
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        remaining = output.getvalue()
        if remaining:
            yield remaining.encode("utf-8")

    async def iter_chart_csv_chunks_async(
        self,
        payload: dict[str, Any],
        tools: dict[str, Any],
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        output = io.StringIO()
        writer = csv.writer(output)

        async for row in self.iter_chart_csv_rows_async(payload, tools):
            writer.writerow(row)
            if output.tell() >= chunk_size:
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        remaining = output.getvalue()
        if remaining:
            yield remaining.encode("utf-8")

    def build_chart_csv(self, payload: dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(self.iter_chart_csv_rows(payload))
        return output.getvalue()

    def build_chart_png(self, payload: dict[str, Any]) -> bytes:
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
                mesh = self._plot_heatmap_axis(axes[0][idx], panel, panel.get("region_name") or f"Panel {idx + 1}")
            if mesh is not None:
                fig.colorbar(mesh, ax=axes.ravel().tolist(), label=export.get("units", ""))
        elif export_type == "timeseries":
            rows = self._timeseries_rows(export)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot([row[1] for row in rows], [row[3] for row in rows], marker="o", linewidth=1.5)
            ax.set_title(payload.get("title") or export.get("variable") or "Time series")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{export.get('aggregation', 'value')} ({export.get('units', '')})")
            ax.tick_params(axis="x", rotation=30)
        else:
            from utils.plotting import RegionResolver, plot_map

            da = self._export_data_array(export, collapse_to_2d=True)
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

    async def build_chart_png_async(self, payload: dict[str, Any], tools: dict[str, Any]) -> bytes:
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
                mesh = await self._plot_heatmap_axis_async(
                    axes[0][idx],
                    panel,
                    tools,
                    panel.get("region_name") or f"Panel {idx + 1}",
                )
            if mesh is not None:
                fig.colorbar(mesh, ax=axes.ravel().tolist(), label=export.get("units", ""))
        elif export_type == "timeseries":
            rows = await self._timeseries_rows_async(export, tools)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot([row[1] for row in rows], [row[3] for row in rows], marker="o", linewidth=1.5)
            ax.set_title(payload.get("title") or export.get("variable") or "Time series")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{export.get('aggregation', 'value')} ({export.get('units', '')})")
            ax.tick_params(axis="x", rotation=30)
        else:
            from utils.plotting import RegionResolver, plot_map

            da = await self._export_data_array_async(export, tools, collapse_to_2d=True)
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
                extent=region["bounds"] if region else None,
                mask_geometry=region["geometry"] if region else None,
                cmap=payload.get("cmap") or export.get("chart_parameters", {}).get("cmap") or "Spectral_r",
            )

        fig.tight_layout()
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output.getvalue()

    def iter_chart_csv_rows(self, payload: dict[str, Any]):
        export = payload.get("export") or {}
        if not export:
            raise ValueError("This chart does not include full-resolution export metadata.")

        export_type = export.get("type")
        if export_type == "heatmap_multi":
            for idx, panel in enumerate(export.get("panels") or []):
                if panel.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                    yield from self._iter_aggregated_heatmap_csv_rows(
                        panel,
                        panel.get("region_name") or f"panel-{idx + 1}",
                    )
                else:
                    if idx == 0:
                        yield ["panel", "variable", "latitude", "longitude", "value", "units"]
                    yield from self._iter_heatmap_csv_rows(panel, panel.get("region_name") or f"panel-{idx + 1}")
        elif export_type == "timeseries":
            yield ["variable", "time", "stat", "value", "units"]
            yield from self._timeseries_rows(export)
        else:
            if export.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                yield from self._iter_aggregated_heatmap_csv_rows(export)
            else:
                yield ["variable", "latitude", "longitude", "value", "units"]
                yield from self._iter_heatmap_csv_rows(export)

    async def iter_chart_csv_rows_async(self, payload: dict[str, Any], tools: dict[str, Any]):
        export = payload.get("export") or {}
        if not export:
            raise ValueError("This chart does not include full-resolution export metadata.")

        export_type = export.get("type")
        if export_type == "heatmap_multi":
            for idx, panel in enumerate(export.get("panels") or []):
                if panel.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                    async for row in self._iter_aggregated_heatmap_csv_rows_async(
                        panel,
                        tools,
                        panel.get("region_name") or f"panel-{idx + 1}",
                    ):
                        yield row
                else:
                    if idx == 0:
                        yield ["panel", "variable", "latitude", "longitude", "value", "units"]
                    async for row in self._iter_heatmap_csv_rows_async(panel, tools, panel.get("region_name") or f"panel-{idx + 1}"):
                        yield row
        elif export_type == "timeseries":
            yield ["variable", "time", "stat", "value", "units"]
            for row in await self._timeseries_rows_async(export, tools):
                yield row
        else:
            if export.get("aggregation_meta", {}).get("n_granules", 1) > 1:
                async for row in self._iter_aggregated_heatmap_csv_rows_async(export, tools):
                    yield row
            else:
                yield ["variable", "latitude", "longitude", "value", "units"]
                async for row in self._iter_heatmap_csv_rows_async(export, tools):
                    yield row

    def _export_lat_lon_names(self, da):
        lat_coord = next((c for c in ["lat", "latitude", "Latitude"] if c in da.coords), None)
        lon_coord = next((c for c in ["lon", "longitude", "Longitude"] if c in da.coords), None)
        if lat_coord is None or lon_coord is None:
            raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
        return lat_coord, lon_coord

    def _export_data_array(self, export: dict[str, Any], collapse_to_2d: bool = True):
        raise RuntimeError("Chart data export requires the async export path.")

    async def _export_data_array_async(self, export: dict[str, Any], tools: dict[str, Any], collapse_to_2d: bool = True):
        from preprocessing.aggregation_service import AggregationService
        from tools.satellite_tools.plot_tools import _normalize_longitudes, _sel_bounds
        from services.open_handle import open_handle
        from utils.plotting import RegionResolver, mask_data_by_geometry

        source_handles = export.get("source_handles") or []
        if not source_handles:
            raise ValueError("This chart does not include a source handle for full-resolution export.")
        ds = await open_handle(source_handles[0], tools)
        da = AggregationService().to_dataarray(ds, variable=export.get("variable"))
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        da = _normalize_longitudes(da, lon_coord)

        region = None
        region_name = export.get("region_name")
        if region_name:
            try:
                region = RegionResolver().resolve_location(region_name)
            except Exception:
                region = None

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

    def _iter_heatmap_csv_rows(self, export: dict[str, Any], panel_name: str | None = None):
        import numpy as np

        da = self._export_data_array(export, collapse_to_2d=True)
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

    async def _iter_heatmap_csv_rows_async(self, export: dict[str, Any], tools: dict[str, Any], panel_name: str | None = None):
        import numpy as np

        da = await self._export_data_array_async(export, tools, collapse_to_2d=True)
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

    def _iter_aggregated_heatmap_csv_rows(self, export: dict[str, Any], panel_name: str | None = None):
        import numpy as np
        import pandas as pd
        from preprocessing.aggregation_service import AggregationService

        da = self._export_data_array(export, collapse_to_2d=False)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        if "time" not in da.dims:
            yield from self._iter_heatmap_csv_rows(export, panel_name)
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
            variable=export.get("variable") or (export.get("fetch_params") or {}).get("variable"),
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

    async def _iter_aggregated_heatmap_csv_rows_async(self, export: dict[str, Any], tools: dict[str, Any], panel_name: str | None = None):
        import numpy as np
        import pandas as pd
        from preprocessing.aggregation_service import AggregationService

        da = await self._export_data_array_async(export, tools, collapse_to_2d=False)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        if "time" not in da.dims:
            async for row in self._iter_heatmap_csv_rows_async(export, tools, panel_name):
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

    def _timeseries_rows(self, export: dict[str, Any]):
        import numpy as np
        import pandas as pd
        from preprocessing.aggregation_service import AggregationService

        da = self._export_data_array(export, collapse_to_2d=False)
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

    async def _timeseries_rows_async(self, export: dict[str, Any], tools: dict[str, Any]):
        import numpy as np
        import pandas as pd
        from preprocessing.aggregation_service import AggregationService

        da = await self._export_data_array_async(export, tools, collapse_to_2d=False)
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

    def _plot_heatmap_axis(self, ax, export: dict[str, Any], title: str):
        da = self._export_data_array(export, collapse_to_2d=True)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        mesh = ax.pcolormesh(
            da[lon_coord].values,
            da[lat_coord].values,
            da.values.astype(float),
            shading="auto",
            cmap="Spectral_r",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return mesh

    async def _plot_heatmap_axis_async(self, ax, export: dict[str, Any], tools: dict[str, Any], title: str):
        da = await self._export_data_array_async(export, tools, collapse_to_2d=True)
        lat_coord, lon_coord = self._export_lat_lon_names(da)
        mesh = ax.pcolormesh(
            da[lon_coord].values,
            da[lat_coord].values,
            da.values.astype(float),
            shading="auto",
            cmap="Spectral_r",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return mesh
