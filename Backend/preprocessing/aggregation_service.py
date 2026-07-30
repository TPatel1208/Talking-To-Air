from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datasets.mask_info import SOURCE_CF_ATTRS, match_umm_var_variable, resolve_mask_info
from datasets.qa_flags import (
    QA_CF_DETERMINISTIC,
    QA_INFERRED,
    QA_NOT_APPLIED,
    QA_VERIFIED,
    resolve_qa_info,
)
from datasets.registry import load_registry
from earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
from preprocessing.variable_resolver import Resolution, resolve
from services import variable_choice_registry
from utils.geo_utils import identify_time
from utils.phase_timing import phase_timer

logger = logging.getLogger(__name__)

# Key under which to_dataarray stashes the VariableResolver's facts on the
# returned DataArray's attrs, so aggregate()/the tool layer can lift the
# disclosure + chosen-variable provenance into the answer without any change
# to to_dataarray's signature (the single chokepoint all 8 tools funnel
# through). Set via assign_attrs, so the source Dataset's variable attrs are
# never mutated.
VARIABLE_RESOLUTION_ATTR = "_variable_resolution"


@dataclass(frozen=True)
class AggregatedResult:
    ds: xr.Dataset
    meta: dict[str, Any]


# Global-attribute spellings that carry a granule's temporal coverage when the
# file has no time coordinate at all (e.g. HAQ_TROPOMI_NO2_GLOBAL_M_L3 monthly
# means: dims are lat/lon only, the month lives in RangeBeginningDate/
# RangeEndingDate). ACDD first, then the ECS/CMR pair every GES DISC product
# stamps. Each entry is (date_key, optional_time_key).
_TEMPORAL_START_ATTRS = (("time_coverage_start", None), ("RangeBeginningDate", "RangeBeginningTime"))
_TEMPORAL_END_ATTRS = (("time_coverage_end", None), ("RangeEndingDate", "RangeEndingTime"))


def _attr_timestamp(attrs: dict, candidates: tuple) -> str:
    for date_key, time_key in candidates:
        date = attrs.get(date_key)
        if not date:
            continue
        date = str(date).strip()
        time = str(attrs.get(time_key) or "").strip() if time_key else ""
        if time and "T" not in date:
            return f"{date}T{time.rstrip('Z')}"
        return date
    return ""


def attrs_time_range(attrs: dict | None) -> tuple[str, str]:
    """Granule temporal coverage from global attributes -- the fallback for
    files whose date exists only as metadata, never as a time coordinate."""
    attrs = attrs or {}
    return _attr_timestamp(attrs, _TEMPORAL_START_ATTRS), _attr_timestamp(attrs, _TEMPORAL_END_ATTRS)


def fill_match(values: Any, fill: Any) -> Any:
    """Boolean mask of cells equal to the fill value — the ONE definition of
    fill matching, shared by the science-variable masking
    (``apply_quality_mask``) and the companion-evidence valid mask
    (``plot_tools._band_valid_mask``), so a tolerance fix can never silently
    diverge between the two. Works on ``xr.DataArray`` and ``np.ndarray``
    alike (pure elementwise ops).

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
        return values == fill_f
    return np.isclose(values, fill_f, rtol=1e-6, atol=1e-9)


def _decode_encoding(da: xr.DataArray, source_ds: xr.Dataset | None = None) -> dict:
    """The scale/offset encoding xarray recorded when it decoded ``da``.

    Prefers the *unmasked* source Dataset's copy of the variable: in-place ops
    the tool layer runs before masking — ``.where()`` (geometry mask), ``.sel()``
    (crop) — strip ``.encoding`` off the working array, but the opened Dataset
    (``source_ds``) still carries it. Falls back to the array's own encoding
    (compare's aligned slices reach masking without a geometry ``.where``, so
    theirs survives)."""
    if source_ds is not None and hasattr(source_ds, "data_vars"):
        var = source_ds.data_vars.get(da.name)
        enc = getattr(var, "encoding", None) or {}
        if "scale_factor" in enc or "add_offset" in enc:
            return enc
    return getattr(da, "encoding", None) or {}


def _scale_cf_bounds_to_physical(
    da: xr.DataArray, valid_min: Any, valid_max: Any, source_ds: xr.Dataset | None = None,
) -> tuple[Any, Any]:
    """Convert a variable's CF ``valid_min``/``valid_max`` from PACKED to
    physical units when xarray has scale/offset-decoded the data.

    CF publishes ``valid_range``/``valid_min``/``valid_max`` in the file's
    stored (packed) integer units. xarray's default ``mask_and_scale`` decode
    turns the DATA into physical units and moves ``scale_factor``/``add_offset``
    out of ``.attrs`` into ``.encoding`` — but it leaves the valid-range attrs
    packed. Masking the decoded field against a packed bound (``da <= 30000``
    on a field that now runs ~0–6e17) silently wipes the entire real variable.

    Scaling is keyed off ``.encoding`` presence, not ``.attrs``: encoding
    carries scale/offset iff xarray actually decoded the data, so an undecoded
    (still-packed) array — whose bounds are already commensurable — is left
    untouched. A no-op when the variable was never scale/offset-decoded, so
    registry/UMM-Var bounds (already physical) never reach this path.
    """
    encoding = _decode_encoding(da, source_ds)
    scale = encoding.get("scale_factor")
    offset = encoding.get("add_offset")
    if scale is None and offset is None:
        return valid_min, valid_max
    s = float(scale) if scale is not None else 1.0
    o = float(offset) if offset is not None else 0.0
    lo = valid_min * s + o if valid_min is not None else None
    hi = valid_max * s + o if valid_max is not None else None
    # A negative scale_factor flips the ordering — a packed lower bound becomes
    # the physical upper bound. Reorder so valid_min <= valid_max downstream.
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _sample_std(a: Any, **kwargs: Any) -> Any:
    """Sample standard deviation (ddof=1), NaN-aware. Over a handful of
    granules the values are a *sample* of the field's variability, not the
    whole population — population std (ddof=0) understates it (~22% low at
    n=3). A single sample (n=1) has no spread to estimate at all: the honest
    answer is NaN, so the degenerate-slice RuntimeWarning is suppressed rather
    than papered over with a fabricated 0.0. Signature mirrors ``np.nanstd``
    (accepts ``axis=`` from ``xarray.DataArray.reduce``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, ddof=1, **kwargs)


def area_weighted_mean(da: xr.DataArray) -> float:
    """Cosine-latitude area-weighted mean over a (lat, lon) field — the ONE
    definition of a regional mean, shared by the statistics tool and the
    comparison stats so they can never disagree.

    On a regular lat/lon grid, cell area shrinks by cos(latitude) toward the
    poles, so an unweighted mean over grid cells over-weights high latitudes
    — a few percent for a CONUS-scale region, badly wrong for continental or
    global ones. Falls back to the unweighted mean when no latitude
    dimension is identifiable (point data, an already-flattened array).
    Raises ``ValueError`` when no finite values remain, mirroring
    ``compute_values_stat``.
    """
    from utils.geo_utils import find_lat_coord

    values = np.asarray(da.values, dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("No finite values available for statistic.")

    lat_name = find_lat_coord(da)
    if lat_name is None or lat_name not in da.dims:
        return float(np.nanmean(values[np.isfinite(values)]))

    # float64 weights regardless of how the granule stores its latitude axis:
    # real products publish float32 lat, and accumulating float32 weights makes
    # the answer depend on how many zero-weight (masked-out) cells happen to be
    # summed -- the same region answered ~5e-7 differently from a continental
    # granule than from a tight crop of it.
    weights = np.cos(np.deg2rad(da[lat_name].astype("float64")))
    result = float(da.weighted(weights).mean(skipna=True))
    if not np.isfinite(result):
        raise ValueError("No finite values available for statistic.")
    return result


def flag_pass_condition(qf: xr.DataArray, good_values: Any = None, bad_values: Any = None) -> xr.DataArray:
    """Boolean condition of flag cells passing QA — the ONE good/bad flag
    doctrine (T25), shared by ``apply_quality_mask`` and the evidence
    pass-rate fact (``plot_tools._qa_pass_rate_fact``) so the reported pass
    rate can never quietly disagree with the mask actually applied.

    With ``good_values``, membership passes (``isin`` already excludes an
    absent/NaN flag). With only ``bad_values``, a pixel passes only when its
    flag is present AND not bad — an unknown-quality pixel (NaN/fill flag,
    e.g. OMI_HCHO's uncomputed-quality) is never counted as good.
    """
    if good_values is not None:
        return qf.isin(good_values)
    return qf.notnull() & ~qf.isin(bad_values)


# P1: cap on how many candidate variable names an ambiguous-variable refusal
# renders. Keeps the error (and its re-feed cost to the model) bounded no matter
# how wide the file; see _ambiguous_variable_error.
_MAX_LISTED_CANDIDATES = 20


class VariableChoiceRequired(Exception):
    """T49 deterministic short-circuit. When ``to_dataarray`` reaches its
    low-confidence tail -- a genuinely ambiguous file the T48 resolver won't
    guess -- it raises THIS instead of a bare ``MCPToolError``, so the choice
    reaches the researcher as a clickable picker built deterministically from
    the resolver's own candidates, never narrated (and possibly truncated) by
    the LLM.

    It carries two things:
    - ``resolution``: the resolver's full ``Resolution`` (every ranked
      candidate), from which the tool layer builds the uncapped
      ``VariableChoice`` picker and emits it out-of-band (emit_variable_choice).
    - ``mcp_error``: the SAME P1-bounded ``MCPToolError`` the low tier used to
      raise -- the tool returns this compact, capped payload to the model as the
      sub-task's terminal tool result, so the model can write a one-line "I've
      shown you a picker" without ever seeing (or re-feeding) the full list.
    """

    def __init__(self, resolution: Resolution, mcp_error: MCPToolError):
        super().__init__(mcp_error.message)
        self.resolution = resolution
        self.mcp_error = mcp_error


def _cell_count(data: xr.Dataset | xr.DataArray) -> int:
    """Cells entering an aggregation, for the phase-timing size context (T51):
    duration alone can't tell an I/O-bound phase from a CPU-bound one. Nothing
    to report for an unexpected shape -- this is telemetry, never a failure."""
    try:
        if isinstance(data, xr.Dataset):
            return int(sum(int(var.size) for var in data.data_vars.values()))
        return int(data.size)
    except Exception:  # pragma: no cover -- defensive
        return 0


class AggregationService:
    """Single entry point for satellite data validity filtering and reductions."""

    _STAT_FUNCS = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": _sample_std,
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

        # T51: one phase for the whole variable-resolve -> QA-mask -> temporal-
        # reduce chain. It's the compute that a lazily-opened bundle actually
        # pays for -- the reduction is what forces the dask graph, so this
        # phase's duration is where a slow multi-granule open shows up.
        with phase_timer("aggregate", stat=stat, cells_in=_cell_count(data)):
            return self._aggregate(
                data,
                collection_id,
                stat,
                variable=variable,
                col_info=col_info,
                umm_var_facts=umm_var_facts,
                keep_time=keep_time,
                handle=handle,
                qa_good_tokens=qa_good_tokens,
                source_ds=source_ds,
            )

    def _aggregate(
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
        da = self.to_dataarray(data, variable=variable, handle=handle)

        # T48: when to_dataarray resolved the variable itself (a wide,
        # unregistered, multi-product file), it stashed the chosen name + a
        # disclosure on the array attrs. Capture it now, before masking's
        # ``.where`` strips attrs, so the disclosure can ride out in meta to the
        # tool layer's provenance and the chat answer.
        variable_resolution = da.attrs.get(VARIABLE_RESOLUTION_ATTR)

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
        cadence = self._cadence(data, collection_id, variable, col_info)
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
                elif stat == "mean":
                    # A temporal mean is "the average over the period", so each
                    # cadence bucket must count equally -- not each granule
                    # (Finding #11). Clustered sampling (20 granules on two days
                    # plus one on a third) otherwise over-weights the dense days.
                    # Evenly-cadenced series have one granule per bucket, so the
                    # weights are uniform and the result is unchanged.
                    reduced = self._cadence_weighted_mean(valid_da, time_dim, cadence)
                else:
                    reduced = valid_da.reduce(self._STAT_FUNCS[stat], dim=time_dim)

        result_ds = reduced.to_dataset(name=da.name or variable or "value")
        result_ds.attrs.update(getattr(data, "attrs", {}))
        result_ds.attrs["n_granules"] = len(valid_indices)
        result_ds.attrs["cadence"] = cadence

        # Global attrs are the temporal-fallback source for time-coordinate-less
        # granules; the tool layer's opened Dataset (source_ds) carries them
        # when ``data`` is an already-extracted DataArray (variable attrs only).
        source_attrs: dict[str, Any] = {}
        for attr_source in (source_ds, data):
            source_attrs.update(getattr(attr_source, "attrs", None) or {})
        meta = self._build_meta(
            data, len(valid_indices), cadence, stat, valid_indices, time_dim,
            source_attrs=source_attrs,
        )
        meta["masking"] = masking_provenance
        if variable_resolution:
            meta["variable_resolution"] = variable_resolution

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
        return self._build_meta(
            data, len(valid_indices), cadence, stat, valid_indices, time_dim,
            source_attrs=getattr(data, "attrs", None),
        )

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

        # CF valid-range bounds are packed; the decoded data is physical. When
        # the CF-attrs tier supplied the bounds, scale them to physical before
        # apply_quality_mask compares them against the decoded field (registry/
        # UMM-Var tiers are already physical, so only the cf_attrs tier is
        # rescaled).
        if masking_provenance.get("valid_range_source") == SOURCE_CF_ATTRS:
            scaled_min, scaled_max = _scale_cf_bounds_to_physical(
                da, resolved_col_info.get("valid_min"), resolved_col_info.get("valid_max"),
                source_ds=source_ds,
            )
            if scaled_min is not None:
                resolved_col_info["valid_min"] = scaled_min
            if scaled_max is not None:
                resolved_col_info["valid_max"] = scaled_max

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

        The exact intent/curation tiers come first and are unchanged: explicit
        ``variable`` -> the choice recorded for ``handle`` at retrieval time
        (services.variable_choice_registry) -> the file's only data
        variable -> the collection's pinned ``primary_var`` (collections.yaml,
        matched via the file's ``short_name`` attr -- a curated choice, not a
        guess). The previous ``next(iter(data.data_vars))`` silent-first-variable
        fallback stays deleted.

        When all of those miss on a multi-variable file, the VariableResolver
        (T48) takes over: it classifies each variable (implementation plumbing
        vs distinct product), scores the weak signals, drops empties, ranks, and
        either auto-picks a populated science field -- stashing its facts and a
        disclosure on the returned array's ``_variable_resolution`` attrs -- or,
        for a genuinely low-confidence file, returns no name and this raises the
        P1-bounded, ranked candidate error so the researcher chooses. It never
        invents a scientific choice (T25): a genuine sensor/algorithm fork is
        disclosed, not silently guessed.

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
        resolution: Resolution | None = None
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
                # T48: the VariableResolver replaces the old
                # _science_vars->refuse tail. It classifies (implementation vs
                # distinct product), scores, drops empties, ranks, and decides
                # -- auto-picking a populated science field (with a disclosure
                # when the choice is a genuine sensor/algorithm fork) instead of
                # dumping 432 opaque candidates. A genuinely low-confidence file
                # still refuses, now with the resolver's RANKED, bounded list.
                resolution = resolve(data, requested=variable)
                self._log_resolution(resolution)
                if resolution.name is None:
                    # T49: hand the low-confidence refusal to the researcher as
                    # a deterministic picker, not LLM prose. The signal carries
                    # the full Resolution (for the uncapped out-of-band picker)
                    # plus the P1-bounded MCPToolError (the compact terminal
                    # tool result the model still needs).
                    ranked = [c.name for c in resolution.candidates] or data_vars
                    raise VariableChoiceRequired(
                        resolution, self._ambiguous_variable_error(data, ranked),
                    )
                name = resolution.name

        da = data[name]
        if variable:
            da.name = variable
        if resolution is not None:
            # assign_attrs returns a fresh DataArray (copied attrs) -- the
            # source Dataset's variable attrs are never mutated.
            da = da.assign_attrs(**{VARIABLE_RESOLUTION_ATTR: self._resolution_facts(resolution, data)})
        return da

    @staticmethod
    def _resolution_facts(resolution: Resolution, ds: xr.Dataset) -> dict[str, Any]:
        """The VariableResolver's decision, flattened for provenance/logging:
        the chosen variable, both decision axes, the ranked alternatives, the
        reasons behind the pick, and the deterministic disclosure line (T48).
        Carried out of to_dataarray on the returned array's attrs.

        For a MEDIUM-confidence auto-pick (T49), the full deterministic picker
        is stashed too -- the answer is delivered with its chart, but the
        dispatch layer lifts this into ``AgentResult.variable_choice`` so the
        researcher can override the guess in one click. Built here (where the
        Dataset is in hand for per-candidate units); the auto-send prompts are
        left blank for run_satellite to fill from the original request."""
        chosen = next((c for c in resolution.candidates if c.name == resolution.name), None)
        facts: dict[str, Any] = {
            "chosen": resolution.name,
            "resolution_confidence": resolution.resolution_confidence,
            "scientific_ambiguity": resolution.scientific_ambiguity,
            "disclosure": resolution.disclosure,
            "reasons": list(chosen.reasons) if chosen else [],
            "alternatives": [
                c.name for c in resolution.candidates if c.name != resolution.name
            ][:_MAX_LISTED_CANDIDATES],
        }
        if resolution.resolution_confidence == "medium":
            from preprocessing.variable_choice_builder import build_variable_choice

            facts["variable_choice"] = build_variable_choice(resolution, ds).model_dump()
        return facts

    @staticmethod
    def _log_resolution(resolution: Resolution) -> None:
        """Operator diagnosability (user story 6): a wrong pick must be
        traceable from logs -- the chosen name, both axes, and the ranked
        candidate scores."""
        logger.info(
            "variable_resolution",
            extra={
                "_event": "variable_resolution",
                "_chosen": resolution.name,
                "_confidence": resolution.resolution_confidence,
                "_ambiguity": resolution.scientific_ambiguity,
                "_candidates": [
                    {"name": c.name, "category": c.category, "score": c.score,
                     "valid_fraction": c.valid_fraction}
                    for c in resolution.candidates[:_MAX_LISTED_CANDIDATES]
                ],
            },
        )

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

    def _ambiguous_variable_error(self, data: xr.Dataset, data_vars: list[str]) -> MCPToolError:
        # P1: the candidate list MUST be bounded in size. A Yori-aggregated L3
        # whose groups all share Mean/Standard_Deviation/Pixel_Counts/
        # Histogram_Counts leaves (AERDA_D3_VIIRS_MODIS,
        # dataset_a88593edb7246c9b) merges to 432 data_vars -- listing every
        # one produced a ~20 KB error that, re-fed to the model on every
        # stat/plot call, drove a context explosion and whole-turn timeout
        # (blank plot, no statistics; live 2026-07-19). Naming a capped sample
        # plus the true total turns that O(N) blowup into O(1) while staying
        # honest about how many candidates exist. WHICH candidates surface is
        # now the VariableResolver's job (T48): ``data_vars`` arrives already
        # ranked (distinct products first, empties dropped), so the capped
        # sample is the most relevant few. This tier only bounds the size.
        total = len(data_vars)
        shown = data_vars[:_MAX_LISTED_CANDIDATES]
        candidates = []
        for name in shown:
            attrs = data[name].attrs
            label = attrs.get("long_name") or attrs.get("standard_name")
            candidates.append(f"{name} ({label})" if label else name)
        hidden = total - len(shown)
        more = f" (and {hidden} more)" if hidden else ""
        return MCPToolError(
            CATEGORY_VARIABLE_CHOICE_REQUIRED,
            f"This file has {total} science variables and no variable was chosen. "
            f"Showing {len(shown)} of {total}: {', '.join(candidates)}{more}. "
            f"Specify which one to analyze.",
            suggestion=f"Pass variable=<name>, e.g. one of: {', '.join(shown)}{more}.",
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
        valid_min = col_info.get("valid_min")
        valid_max = col_info.get("valid_max")
        if valid_min is None and valid_max is None:
            # No resolved (physical) bound from col_info — fall back to the
            # file's own CF attrs: valid_min/valid_max, or the combined
            # ``valid_range: [min, max]`` spelling plenty of NASA products
            # publish instead (neither applied by xarray on decode). These are
            # in PACKED units, so scale them to physical to match the decoded
            # field before comparison (see _scale_cf_bounds_to_physical) —
            # otherwise a scaled product's whole field is wiped. col_info
            # bounds are already physical and are used verbatim above.
            from datasets.mask_info import _split_valid_range_attr

            attr_min = da.attrs.get("valid_min")
            attr_max = da.attrs.get("valid_max")
            if attr_min is None and attr_max is None:
                attr_min, attr_max = _split_valid_range_attr(da.attrs.get("valid_range"))
            valid_min, valid_max = _scale_cf_bounds_to_physical(da, attr_min, attr_max, source_ds=ds)

        if actual_fill is not None:
            da = da.where(~fill_match(da, actual_fill))
        if valid_min is not None:
            da = da.where(da >= valid_min)
        if valid_max is not None:
            da = da.where(da <= valid_max)

        qf_var = col_info.get("quality_flag_var")
        if apply_quality_flag and ds is not None and qf_var and qf_var in ds.data_vars:
            qf = ds[qf_var]
            good_values = col_info.get("qa_good_values")
            bad_values = col_info.get("qa_bad_values")
            if good_values is not None or bad_values is not None:
                qf = self._decode_flag_fill(qf)
                da = da.where(flag_pass_condition(qf, good_values, bad_values))
        return da

    @staticmethod
    def _decode_flag_fill(qf: xr.DataArray) -> xr.DataArray:
        """Null the QA-flag variable's OWN fill/out-of-range sentinels before
        the good/bad pass test. A flag stored as an undecoded integer sentinel
        (255, -1) that xarray never turned to NaN would otherwise satisfy
        ``flag_pass_condition``'s ``notnull() & ~isin(bad_values)`` and let its
        science pixel through as a real 'good' observation. Resolves the flag's
        own ``_FillValue``/``valid_range`` from its CF attrs (the same
        ``resolve_mask_info`` discipline the science variable gets) and masks
        those cells to NaN, so an unknown-quality pixel reads as absent, not
        good. A no-op when the flag declares no fill/valid bounds."""
        resolved, _ = resolve_mask_info(cf_attrs=dict(qf.attrs))
        fill = resolved.get("fill_value")
        valid_min = resolved.get("valid_min")
        valid_max = resolved.get("valid_max")
        if fill is not None:
            qf = qf.where(~fill_match(qf, fill))
        if valid_min is not None:
            qf = qf.where(qf >= valid_min)
        if valid_max is not None:
            qf = qf.where(qf <= valid_max)
        return qf

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

    # Cadence -> the pandas offset a timestamp is floored to when grouping
    # granules into the buckets a temporal mean must weight equally (#11).
    _CADENCE_FLOOR = {"hourly": "h", "daily": "D"}

    def _cadence_weighted_mean(self, da: xr.DataArray, time_dim: str, cadence: str) -> xr.DataArray:
        """Per-pixel temporal mean that weights each cadence bucket equally,
        not each granule (Finding #11) -- so 'average over the period' is not
        biased toward days sampled more densely than the cadence.

        Each timestep's weight is ``1 / (granules in its cadence bucket)``, so
        every bucket contributes weight 1 to the mean regardless of how many
        granules landed in it. Buckets come from flooring the timestamp to the
        product's cadence (hour/day/month). An unknown or unbucketable cadence
        can't define a period, so it falls back to the plain unweighted mean --
        as does an evenly-cadenced series, where every bucket holds exactly one
        granule and the weights are already uniform.
        """
        if cadence not in ("hourly", "daily", "monthly"):
            return da.reduce(np.nanmean, dim=time_dim)
        try:
            stamps = pd.to_datetime(np.asarray(da[time_dim].values))
            if cadence == "monthly":
                buckets = stamps.to_period("M")
            else:
                buckets = stamps.floor(self._CADENCE_FLOOR[cadence])
            counts = pd.Series(1, index=buckets).groupby(level=0).transform("size")
        except Exception:
            # Non-datetime or otherwise unbucketable time axis -- never fatal.
            return da.reduce(np.nanmean, dim=time_dim)
        weights = xr.DataArray(
            1.0 / np.asarray(counts.values, dtype=float),
            dims=[time_dim],
            coords={time_dim: da[time_dim]},
        )
        return da.weighted(weights).mean(dim=time_dim, skipna=True)

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
        # "unknown", never "daily": an off-registry product's cadence is a
        # fact this backend does not have, and defaulting it stamped a false
        # provenance claim ("12 daily granules" on a year of monthly means)
        # plus wrong period labels ("Annual"/"Daily" inference below) on
        # exactly the datasets the universal pipeline exists to welcome.
        return str(info.get("cadence", "unknown"))

    def _build_meta(
        self,
        data: xr.Dataset | xr.DataArray,
        n_granules: int,
        cadence: str,
        stat: str,
        valid_indices: list[int],
        time_dim: str | None = None,
        source_attrs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        time_dim = time_dim or identify_time(data)
        times = []
        if time_dim and time_dim in getattr(data, "coords", {}):
            all_times = [str(t) for t in np.atleast_1d(data[time_dim].values)]
            times = [all_times[i] for i in valid_indices if i < len(all_times)]

        start = self._date_only(times[0]) if times else ""
        end = self._date_only(times[-1]) if times else ""
        if not times:
            # No time coordinate on the data at all (e.g. a monthly L3 mean
            # whose dims are lat/lon only): the granule's coverage lives in
            # its global attrs (ACDD time_coverage_* / ECS RangeBeginning*).
            # Fall back there rather than reporting "no dates" for data that
            # plainly has one -- the granule list is the coverage start (the
            # granule's own date), disclosed only for the single-granule case
            # where that is exact.
            attr_start, attr_end = attrs_time_range(source_attrs)
            start = self._date_only(attr_start)
            end = self._date_only(attr_end)
            if n_granules <= 1 and start:
                times = [start]
        # An unknown cadence is disclosed as plain "granules" — never dressed
        # up as a frequency word the data never claimed.
        cadence_label = {"hourly": "hourly", "daily": "daily", "monthly": "monthly"}.get(cadence, cadence)
        if cadence == "unknown":
            granule_str = f"{n_granules} granule{'s' if n_granules != 1 else ''}"
        else:
            granule_str = f"{n_granules} {cadence_label} granule{'s' if n_granules != 1 else ''}"

        # A frequency period word ("Annual"/"Daily") is inferred from the
        # granule *count*, but the count only implies a span under the assumed
        # cadence -- reprocessed/overlapping granules break that assumption
        # (Finding #14). Require the actual date span to match before applying
        # the word, so 12 monthly granules clustered inside one month don't
        # read "Annual", nor 10 hourly granules across a week read "Daily".
        span_days = self._span_days(start, end)
        if n_granules <= 1:
            period = "Single Snapshot"
        elif cadence == "monthly" and n_granules == 12 and span_days is not None and span_days >= 300:
            period = "Annual"
        elif cadence == "hourly" and n_granules >= 10 and span_days is not None and span_days <= 1:
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
            # Explicit range facts so provenance/query builders don't have to
            # re-derive them from a DataArray whose time dim the aggregation
            # itself already reduced away (plot_tools passes the *reduced*
            # array downstream).
            "start_date": start,
            "end_date": end,
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

    @staticmethod
    def _span_days(start: str, end: str) -> float | None:
        """The calendar-day span between two ``_date_only`` strings, or None
        when either is missing/unparseable (so a period-word guard falls back
        to the explicit date range rather than a frequency word it can't
        justify)."""
        if not start or not end:
            return None
        try:
            return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0
        except Exception:
            return None
