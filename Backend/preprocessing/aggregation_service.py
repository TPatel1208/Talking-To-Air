from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datasets.registry import load_registry


@dataclass(frozen=True)
class AggregatedResult:
    ds: xr.Dataset
    meta: dict[str, Any]


class AggregationService:
    """Single entry point for satellite data validity filtering and reductions."""

    _STAT_FUNCS = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": np.nanstd,
    }

    def aggregate(
        self,
        data: xr.Dataset | xr.DataArray,
        collection_id: str | None = None,
        stat: str = "mean",
        *,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
        keep_time: bool = False,
    ) -> AggregatedResult:
        if stat not in self._STAT_FUNCS:
            raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(self._STAT_FUNCS)}")

        da = self.to_dataarray(data, collection_id=collection_id, variable=variable, col_info=col_info)
        da = self.apply_quality_mask(
            da,
            data if isinstance(data, xr.Dataset) else None,
            col_info or self._collection_info(collection_id, variable),
            variable=variable,
        )

        if "time" not in da.dims:
            reduced = da
            valid_indices = [0]
        else:
            valid_indices = self._valid_time_indices(da)
            if not valid_indices:
                reduced = da.isel(time=slice(0, 0)).mean(dim="time", skipna=True)
            else:
                valid_da = da.isel(time=valid_indices)
                if keep_time and valid_da.sizes.get("time", 0) == 1:
                    reduced = valid_da
                else:
                    reduced = valid_da.reduce(self._STAT_FUNCS[stat], dim="time")

        result_ds = reduced.to_dataset(name=da.name or variable or "value")
        result_ds.attrs.update(getattr(data, "attrs", {}))
        result_ds.attrs["n_granules"] = len(valid_indices)
        result_ds.attrs["cadence"] = self._cadence(data, collection_id, variable, col_info)

        return AggregatedResult(
            ds=result_ds,
            meta=self._build_meta(data, len(valid_indices), self._cadence(data, collection_id, variable, col_info), stat, valid_indices),
        )

    def to_dataarray(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        collection_id: str | None = None,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        if isinstance(data, xr.DataArray):
            return data
        if not data.data_vars:
            raise RuntimeError("Dataset has no data variables.")

        info = col_info or self._collection_info(collection_id, variable)
        primary_var = info.get("primary_var")
        name = next(
            (v for v in data.data_vars if v == primary_var),
            next((v for v in data.data_vars if variable and variable.lower() in v.lower()), next(iter(data.data_vars))),
        )
        da = data[name]
        if variable:
            da.name = variable
        return da

    def apply_quality_mask(
        self,
        da: xr.DataArray,
        ds: xr.Dataset | None = None,
        col_info: dict[str, Any] | None = None,
        *,
        apply_quality_flag: bool = True,
        variable: str | None = None,
    ) -> xr.DataArray:
        col_info = col_info or {}
        actual_fill = col_info.get("fill_value", da.attrs.get("_FillValue"))
        valid_min = col_info.get("valid_min", da.attrs.get("valid_min"))
        valid_max = col_info.get("valid_max", da.attrs.get("valid_max"))

        if actual_fill is not None:
            da = da.where(~np.isclose(da, actual_fill, rtol=0, atol=abs(float(actual_fill)) * 1e-3))
        if valid_min is not None:
            da = da.where(da >= valid_min)
        if valid_max is not None:
            da = da.where(da <= valid_max)

        qf_var = col_info.get("quality_flag_var")
        if apply_quality_flag and ds is not None and qf_var and qf_var in ds.data_vars:
            qf = ds[qf_var]
            bad_mask = (qf == 2) if variable == "OMI_HCHO" else (qf != 0)
            da = da.where(~bad_mask)
        return da

    def compute_values_stat(self, values: np.ndarray, stat: str) -> float:
        if stat not in self._STAT_FUNCS:
            raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(self._STAT_FUNCS)}")
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            raise ValueError("No finite values available for statistic.")
        return float(self._STAT_FUNCS[stat](valid))

    def _valid_time_indices(self, da: xr.DataArray) -> list[int]:
        indices = []
        for i in range(da.sizes["time"]):
            if bool(np.isfinite(da.isel(time=i).values).any()):
                indices.append(i)
        return indices

    def _collection_info(self, collection_id: str | None, variable: str | None) -> dict[str, Any]:
        registry = load_registry()
        if collection_id:
            for cfg in registry.values():
                if cfg.collection_id == collection_id:
                    return cfg.model_dump()
        if variable and variable in registry:
            return registry[variable].model_dump()
        return {}

    def _cadence(self, data: xr.Dataset | xr.DataArray, collection_id: str | None, variable: str | None, col_info: dict[str, Any] | None) -> str:
        attrs = getattr(data, "attrs", {}) or {}
        if attrs.get("cadence"):
            return str(attrs["cadence"])
        info = col_info or self._collection_info(collection_id, variable)
        return str(info.get("cadence", "daily"))

    def _build_meta(
        self,
        data: xr.Dataset | xr.DataArray,
        n_granules: int,
        cadence: str,
        stat: str,
        valid_indices: list[int],
    ) -> dict[str, Any]:
        times = []
        if "time" in getattr(data, "coords", {}):
            all_times = [str(t) for t in data["time"].values]
            times = [all_times[i] for i in valid_indices if i < len(all_times)]

        start = self._date_only(times[0]) if times else ""
        end = self._date_only(times[-1]) if times else ""
        cadence_label = {"hourly": "hourly", "daily": "daily", "monthly": "monthly"}.get(cadence, cadence)
        granule_str = f"{n_granules} {cadence_label} granule{'s' if n_granules != 1 else ''}"

        if n_granules <= 1:
            period = "Single Snapshot"
        elif cadence == "monthly" and n_granules == 12:
            period = "Annual"
        elif cadence == "hourly" and n_granules >= 10:
            period = "Daily"
        elif start and end and start != end:
            period = f"{start} to {end}"
        else:
            period = start or "Single Snapshot"

        stat_label = stat.capitalize()
        date_range = f"{start} to {end}" if start and end and start != end else (start or end)
        year_label = start[:4] if start[:4] and start[:4] == end[:4] else date_range
        aggregation_label = f"{period} {stat_label}, {granule_str}"
        if date_range:
            aggregation_label = f"{aggregation_label}, {date_range}"

        return {
            "aggregation_label": aggregation_label,
            "title_suffix": f"{period} {stat_label} ({year_label}, {granule_str})" if year_label else f"{period} {stat_label} ({granule_str})",
            "granule_dates": [self._date_only(t) for t in times],
            "n_granules": int(n_granules),
            "cadence": cadence,
            "stat": stat,
        }

    @staticmethod
    def _date_only(value) -> str:
        if not value:
            return ""
        try:
            return pd.Timestamp(value).isoformat()[:10]
        except Exception:
            return str(value)[:10]
