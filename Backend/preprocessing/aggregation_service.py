from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datasets.mask_info import match_umm_var_variable, resolve_mask_info
from datasets.qa_flags import (
    QA_CF_DETERMINISTIC,
    QA_INFERRED,
    QA_NOT_APPLIED,
    QA_VERIFIED,
    resolve_qa_info,
)
from datasets.registry import known_quality_flag_vars, load_registry
from earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
from services import variable_choice_registry
from utils.geo_utils import identify_time


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
        umm_var_facts: Any = None,
        keep_time: bool = False,
        handle: str | None = None,
        qa_good_tokens: list[str] | None = None,
    ) -> AggregatedResult:
        if stat not in self._STAT_FUNCS:
            raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(self._STAT_FUNCS)}")

        da = self.to_dataarray(data, variable=variable, handle=handle)

        yaml_info = col_info or self._collection_info(collection_id, variable)
        umm_var_variable = match_umm_var_variable(umm_var_facts, variable or da.name)
        resolved_col_info, masking_provenance = resolve_mask_info(
            yaml_info=yaml_info, umm_var_variable=umm_var_variable, cf_attrs=da.attrs,
        )

        # T25 Phase 3: three-tier QA masking (datasets/qa_flags.py) -- a
        # pinned collections.yaml rule, else the sibling flag variable's own
        # CF flag_values/flag_meanings parsed deterministically (falling
        # back to the agent's proposal for ambiguous tokens), else no mask.
        # Always merged into the same masking-provenance disclosure so a
        # caller never has to guess whether QA masking ran silently either.
        qf_source = data if isinstance(data, xr.Dataset) else None
        qf_var, flag_attrs = self._resolve_qa_flag_var(data, da, yaml_info)
        qa_col_info, qa_provenance = resolve_qa_info(
            yaml_info=yaml_info,
            flag_attrs=flag_attrs,
            proposed_good_tokens=qa_good_tokens,
            short_name=yaml_info.get("short_name"),
        )
        if qf_var:
            resolved_col_info["quality_flag_var"] = qf_var
        resolved_col_info.update(qa_col_info)

        # Honesty guard (review #1): resolve_qa_info decides *which* flag values
        # count as good, but apply_quality_mask only actually runs the mask when
        # the flag variable's data is reachable -- i.e. a Dataset carrying
        # ``qf_var`` was passed as ``data``. Every current tool path passes an
        # already-extracted DataArray, so ``qf_source`` is None and no QA mask is
        # applied; stamping "verified"/"cf-deterministic"/"inferred" then would
        # disclose a mask that never ran. Downgrade to an explicit not-applied
        # status so the provenance never claims more than happened. (Restoring
        # the mask on the tool paths is tracked separately -- it needs the
        # opened Dataset threaded through, not just the science DataArray.)
        qa_will_apply = (
            qf_source is not None
            and resolved_col_info.get("quality_flag_var") in getattr(qf_source, "data_vars", {})
            and ("qa_good_values" in resolved_col_info or "qa_bad_values" in resolved_col_info)
        )
        if not qa_will_apply and qa_provenance.get("qa_status") in (
            QA_VERIFIED,
            QA_CF_DETERMINISTIC,
            QA_INFERRED,
        ):
            qa_provenance = {
                "qa_status": QA_NOT_APPLIED,
                "qa_source": qa_provenance.get("qa_source", "none"),
                "qa_note": "quality-flag data not present in the opened view; mask not applied",
            }
        masking_provenance.update(qa_provenance)

        da = self.apply_quality_mask(
            da,
            qf_source,
            resolved_col_info,
            variable=variable,
        )

        # T25: identified by CF metadata (standard_name/axis/datetime dtype),
        # not the literal name "time" -- so a MERRA-2-style `valid_time` dim
        # is still the one transparent auto-reduction, instead of surviving
        # into _normalize_to_2d as an unrecognized extra dimension.
        time_dim = identify_time(da)
        if time_dim is None or time_dim not in da.dims:
            reduced = da
            valid_indices = [0]
        else:
            valid_indices = self._valid_time_indices(da, time_dim)
            if not valid_indices:
                reduced = da.isel({time_dim: slice(0, 0)}).mean(dim=time_dim, skipna=True)
            else:
                valid_da = da.isel({time_dim: valid_indices})
                if keep_time and valid_da.sizes.get(time_dim, 0) == 1:
                    reduced = valid_da
                else:
                    reduced = valid_da.reduce(self._STAT_FUNCS[stat], dim=time_dim)

        result_ds = reduced.to_dataset(name=da.name or variable or "value")
        result_ds.attrs.update(getattr(data, "attrs", {}))
        result_ds.attrs["n_granules"] = len(valid_indices)
        result_ds.attrs["cadence"] = self._cadence(data, collection_id, variable, col_info)

        meta = self._build_meta(
            data, len(valid_indices), self._cadence(data, collection_id, variable, col_info), stat, valid_indices, time_dim,
        )
        meta["masking"] = masking_provenance

        return AggregatedResult(ds=result_ds, meta=meta)

    def to_dataarray(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        variable: str | None = None,
        handle: str | None = None,
        collection_id: str | None = None,
        col_info: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Resolve ``data`` to a single science-variable DataArray.

        Resolution never invents a scientific choice (T25): explicit
        ``variable`` -> the choice recorded for ``handle`` at retrieval time
        (services.variable_choice_registry) -> the file's only data
        variable -> its only *science* variable once QA-flag vars are set
        aside -> a structured, candidate-listing error. The previous
        ``next(iter(data.data_vars))`` silent-first-variable fallback is
        deleted, not softened -- a multi-science-variable file with no choice
        made anywhere in that chain must refuse, not guess.

        ``variable`` and the recorded choice are matched by exact name or by
        bare leaf name: registry variable lists and recorded choices are HDF
        group-qualified (``product/vertical_column_troposphere``), while
        open_handle merges those groups down to the bare leaf
        (``vertical_column_troposphere``) that actually appears in
        ``data_vars``.

        ``collection_id``/``col_info`` are accepted for call-site
        compatibility (aggregate() forwards its own kwargs) but no longer
        participate in variable-name resolution; they remain masking-only
        concerns handled by ``resolve_mask_info``.
        """
        if isinstance(data, xr.DataArray):
            return data
        if not data.data_vars:
            raise RuntimeError("Dataset has no data variables.")

        data_vars = list(data.data_vars)
        name = self._match_var(variable, data_vars)
        if name is None and handle:
            name = self._match_var(variable_choice_registry.get(handle), data_vars)
        if name is None:
            if len(data_vars) == 1:
                name = data_vars[0]
            else:
                # A QA flag is never a science-variable candidate (T25): a
                # TEMPO science+main_data_quality_flag pair still resolves to
                # the single science variable without a spurious refusal.
                science_vars = self._science_vars(data, data_vars)
                if len(science_vars) == 1:
                    name = science_vars[0]
                else:
                    raise self._ambiguous_variable_error(data, science_vars or data_vars)

        da = data[name]
        if variable:
            da.name = variable
        return da

    @staticmethod
    def _match_var(requested: str | None, data_vars: list[str]) -> str | None:
        """The ``data_vars`` entry matching ``requested`` by exact name or by
        bare leaf name (so a group-qualified ``product/foo`` choice resolves
        to the merged ``foo``), or None when ``requested`` is falsy/absent."""
        if not requested:
            return None
        if requested in data_vars:
            return requested
        leaf = requested.rsplit("/", 1)[-1]
        return leaf if leaf in data_vars else None

    @staticmethod
    def _science_vars(data: xr.Dataset, data_vars: list[str]) -> list[str]:
        """``data_vars`` with QA-flag variables removed. A var is a flag if
        its bare leaf name is a pinned ``quality_flag_var`` in the registry,
        or it carries CF ``flag_values`` and ``flag_meanings`` attrs -- the
        same signal ``_resolve_qa_flag_var`` uses to find the sibling flag."""
        flag_names = known_quality_flag_vars()
        science = []
        for name in data_vars:
            attrs = data[name].attrs
            is_flag = name.rsplit("/", 1)[-1] in flag_names or (
                "flag_values" in attrs and "flag_meanings" in attrs
            )
            if not is_flag:
                science.append(name)
        return science

    def _ambiguous_variable_error(self, data: xr.Dataset, data_vars: list[str]) -> MCPToolError:
        candidates = []
        for name in data_vars:
            attrs = data[name].attrs
            label = attrs.get("long_name") or attrs.get("standard_name")
            candidates.append(f"{name} ({label})" if label else name)
        return MCPToolError(
            CATEGORY_VARIABLE_CHOICE_REQUIRED,
            f"This file has {len(data_vars)} science variables and no variable was chosen: "
            f"{', '.join(candidates)}. Specify which one to analyze.",
            suggestion=f"Pass variable=<name> from: {', '.join(data_vars)}.",
        )

    def apply_quality_mask(
        self,
        da: xr.DataArray,
        ds: xr.Dataset | None = None,
        col_info: dict[str, Any] | None = None,
        *,
        apply_quality_flag: bool = True,
        variable: str | None = None,
        umm_var_facts: Any = None,
    ) -> xr.DataArray:
        col_info = col_info or {}
        if umm_var_facts is not None:
            umm_var_variable = match_umm_var_variable(umm_var_facts, variable or da.name)
            col_info, _ = resolve_mask_info(yaml_info=col_info, umm_var_variable=umm_var_variable, cf_attrs=da.attrs)
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
            good_values = col_info.get("qa_good_values")
            bad_values = col_info.get("qa_bad_values")
            if good_values is not None:
                da = da.where(qf.isin(good_values))
            elif bad_values is not None:
                da = da.where(~qf.isin(bad_values))
        return da

    def _resolve_qa_flag_var(
        self, data: xr.Dataset | xr.DataArray, da: xr.DataArray, yaml_info: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any]]:
        """Locate the sibling QA-flag variable and its CF attrs, never
        guessing between ambiguous candidates (T25 doctrine): a pinned
        collections.yaml name -> the CF ``ancillary_variables`` attribute on
        the science variable (the real CF convention for exactly this) ->
        the single sibling data var carrying both ``flag_values`` and
        ``flag_meanings``, if there is exactly one. Anything else (no
        candidate, or more than one with no way to choose) resolves to no
        flag var at all -- Tier 3, not a guess.
        """
        ds = data if isinstance(data, xr.Dataset) else None

        qf_var = yaml_info.get("quality_flag_var")
        if qf_var:
            if ds is not None and qf_var in ds.data_vars:
                return qf_var, dict(ds[qf_var].attrs)
            return qf_var, {}

        if ds is None:
            return None, {}

        ancillary = da.attrs.get("ancillary_variables")
        if ancillary:
            for candidate in str(ancillary).split():
                if candidate in ds.data_vars:
                    return candidate, dict(ds[candidate].attrs)

        candidates = [
            name for name, var in ds.data_vars.items()
            if name != da.name and "flag_values" in var.attrs and "flag_meanings" in var.attrs
        ]
        if len(candidates) == 1:
            return candidates[0], dict(ds[candidates[0]].attrs)
        return None, {}

    def compute_values_stat(self, values: np.ndarray, stat: str) -> float:
        if stat not in self._STAT_FUNCS:
            raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(self._STAT_FUNCS)}")
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            raise ValueError("No finite values available for statistic.")
        return float(self._STAT_FUNCS[stat](valid))

    def _valid_time_indices(self, da: xr.DataArray, time_dim: str) -> list[int]:
        indices = []
        for i in range(da.sizes[time_dim]):
            if bool(np.isfinite(da.isel({time_dim: i}).values).any()):
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
        time_dim: str | None = None,
    ) -> dict[str, Any]:
        time_dim = time_dim or identify_time(data)
        times = []
        if time_dim and time_dim in getattr(data, "coords", {}):
            all_times = [str(t) for t in data[time_dim].values]
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
