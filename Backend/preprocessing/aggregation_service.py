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
        source_ds: xr.Dataset | None = None,
    ) -> AggregatedResult:
        if stat not in self._STAT_FUNCS:
            raise ValueError(f"Unsupported aggregation stat '{stat}'. Valid: {sorted(self._STAT_FUNCS)}")

        da = self.to_dataarray(data, variable=variable, handle=handle)

        # ``data`` itself carries the sibling QA-flag variable when a caller
        # still passes a full Dataset (every existing unit test); otherwise
        # ``source_ds`` is the tool layer's separately-threaded opened
        # Dataset for an already-extracted/cropped ``data`` DataArray (T25
        # masking-execution fix -- every real tool path takes this branch).
        qf_source = data if isinstance(data, xr.Dataset) else source_ds
        da, masking_provenance = self.resolve_and_mask(
            da,
            variable=variable,
            col_info=col_info,
            collection_id=collection_id,
            umm_var_facts=umm_var_facts,
            qa_good_tokens=qa_good_tokens,
            source_ds=qf_source,
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

    def timeseries_aggregation_meta(
        self,
        data: xr.Dataset | xr.DataArray,
        valid_indices: list[int],
        stat: str,
        time_dim: str | None = None,
        *,
        collection_id: str | None = None,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The same aggregation_label/granule_dates/n_granules/cadence summary
        ``aggregate()`` builds, for a caller that keeps every timestep instead
        of reducing over time (T32: conduct_temporal_statistic masks each
        step via ``resolve_and_mask`` directly, never calling ``aggregate()``,
        so its timeseries charts got no Granules/cadence block). ``valid_indices``
        is the caller's own record of which timesteps survived masking --
        the same shape ``aggregate()`` derives internally via
        ``_valid_time_indices``.
        """
        cadence = self._cadence(data, collection_id, variable, col_info)
        return self._build_meta(data, len(valid_indices), cadence, stat, valid_indices, time_dim)

    def resolve_and_mask(
        self,
        da: xr.DataArray,
        *,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
        collection_id: str | None = None,
        umm_var_facts: Any = None,
        qa_good_tokens: list[str] | None = None,
        source_ds: xr.Dataset | None = None,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Resolve fill/valid-range/QA masking facts (T25's collections.yaml
        -> UMM-Var -> CF-attrs precedence, plus the three-tier QA-flag
        doctrine) and apply them to ``da``, honestly. Shared by aggregate()
        (which reduces the result over time afterwards) and
        conduct_temporal_statistic (which masks every time step the same way
        but keeps them all, never reducing) -- one masking-resolution path,
        not a hand-rolled second copy.

        ``source_ds`` is the Dataset carrying ``da``'s sibling QA-flag
        variable -- either the full Dataset a caller passed as ``data``
        (existing unit tests), or the tool layer's separately opened Dataset
        when ``da`` is already an extracted/cropped DataArray (T25 masking-
        execution fix). ``da`` and ``source_ds``'s coordinates only need to
        share the same labeling convention (e.g. both longitude-normalized
        the same way) -- xarray aligns a cropped ``da`` against a
        full-grid ``source_ds`` via its default inner join, no explicit
        cropping of ``source_ds`` required.

        Returns ``(masked_da, masking_provenance)``.
        """
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
        qf_var, flag_attrs = self._resolve_qa_flag_var(source_ds, da, yaml_info)
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
        # ``qf_var`` was supplied as ``source_ds``. Stamping "verified"/
        # "cf-deterministic"/"inferred" when it isn't would disclose a mask
        # that never ran. Downgrade to an explicit not-applied status so the
        # provenance never claims more than happened.
        qa_will_apply = (
            source_ds is not None
            and resolved_col_info.get("quality_flag_var") in getattr(source_ds, "data_vars", {})
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
            source_ds,
            resolved_col_info,
            variable=variable,
        )
        return da, masking_provenance

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
        variable -> the collection's pinned ``primary_var`` (collections.yaml,
        matched via the file's ``short_name`` attr -- a curated choice, not a
        guess) -> its only *science* variable once QA-flag vars are set aside
        -> a structured, candidate-listing error. The previous
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
                # A collection's pinned collections.yaml primary_var, matched
                # via the file's short_name global attr, is a curated human
                # choice -- not the deleted next-first-var guess -- so a
                # registered multi-variable file (e.g. AER_DBDT AOD's 10
                # science vars, primary COMBINE_AOD_550_AVG) resolves to its
                # intended variable instead of a spurious refusal.
                name = self._registry_primary_var(data, data_vars)
            if name is None:
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

    def _registry_primary_var(self, data: xr.Dataset, data_vars: list[str]) -> str | None:
        """The collection's pinned ``primary_var`` (collections.yaml), matched
        via the file's ``short_name`` global attr, if it names one of
        ``data_vars`` (by exact or bare-leaf name, like every other tier). A
        pinned primary_var is a curated human choice, not a guess, so honoring
        it here does not reopen the deleted silent-first-variable behavior --
        an unregistered file, or one whose primary_var isn't present, still
        falls through to the science-var / refusal tiers below."""
        from datasets.mask_info import col_info_for_short_name, short_name_from_attrs

        short_name = short_name_from_attrs(getattr(data, "attrs", None))
        if not short_name:
            return None
        primary = col_info_for_short_name(str(short_name).upper()).get("primary_var")
        return self._match_var(primary, data_vars) if primary else None

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
            da = da.where(~self._fill_match(da, actual_fill))
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
                # Symmetric with the good_values path: a pixel whose flag is
                # absent (NaN/fill -> unknown quality, e.g. OMI_HCHO's
                # uncomputed-quality) is dropped, not silently kept as good.
                # ``isin`` already excludes NaN on the good path; mirror that
                # here rather than ``~isin`` alone, which counts every
                # unknown-flag pixel as good.
                da = da.where(qf.notnull() & ~qf.isin(bad_values))
        return da

    @staticmethod
    def _fill_match(da: xr.DataArray, fill: Any) -> xr.DataArray:
        """Boolean mask of cells equal to the fill value.

        Integer-valued fills (the common satellite case: -1, 0, -9999, 255)
        are exact sentinels -> compare exactly. The old
        ``atol=abs(fill)*1e-3`` band collapsed to atol=0 for a 0 fill (fine
        by accident) but, worse, wrongly masked legitimate values *near* a
        small fill (e.g. 49.99 against a 50 fill), and the widened UMM-Var
        fill tier makes 0-valued fills reachable. Exact equality is correct
        and never nukes a whole variable through a degenerate tolerance. A
        genuine non-integer float fill (rare) keeps a fixed relative+absolute
        tolerance for float-storage drift, independent of the fill magnitude.
        """
        fill_f = float(fill)
        if fill_f.is_integer():
            return da == fill
        return np.isclose(da, fill_f, rtol=1e-6, atol=1e-9)

    def _resolve_qa_flag_var(
        self, ds: xr.Dataset | None, da: xr.DataArray, yaml_info: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any]]:
        """Locate the sibling QA-flag variable and its CF attrs, never
        guessing between ambiguous candidates (T25 doctrine): a pinned
        collections.yaml name -> the CF ``ancillary_variables`` attribute on
        the science variable (the real CF convention for exactly this) ->
        the single sibling data var carrying both ``flag_values`` and
        ``flag_meanings``, if there is exactly one. Anything else (no
        candidate, or more than one with no way to choose) resolves to no
        flag var at all -- Tier 3, not a guess.

        ``ds`` is whatever Dataset the caller has the flag variable's data
        reachable through (see ``resolve_and_mask``'s ``source_ds``) -- None
        when no Dataset is available at all.
        """
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
