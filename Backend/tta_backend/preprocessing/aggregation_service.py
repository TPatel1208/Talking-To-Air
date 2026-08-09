from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.datasets.mask_info import SOURCE_CF_ATTRS, match_umm_var_variable, resolve_mask_info
from tta_backend.datasets.qa_flags import (
    QA_CF_DETERMINISTIC,
    QA_INFERRED,
    QA_NOT_APPLIED,
    QA_NO_FLAG_VARIABLE,
    QA_VERIFIED,
    resolve_qa_info,
)
from tta_backend.datasets.registry import load_registry
from tta_backend.earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
from tta_backend.preprocessing.variable_resolver import Resolution, resolve
from tta_backend.services import variable_choice_registry
from tta_backend.utils.geo_utils import identify_time
from tta_backend.utils.phase_timing import phase_timer

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


class MaskedField(NamedTuple):
    """What ``_resolve_and_mask`` hands back to callers INSIDE this module.

    ``counts`` is the raw QA counter block ``_apply_quality_mask`` recorded,
    empty when QA masking never ran. It is deliberately absent from the public
    ``resolve_and_mask`` interface: ``aggregate`` is its only consumer -- it
    reuses the per-timestep reduction rather than force a second I/O pass over
    a lazily-opened bundle (T55) -- and every fact an outside caller needs is
    already summarized into ``provenance``. Handing it out publicly would widen
    the interface for a reuse channel nobody outside can act on.
    """

    data: xr.DataArray
    provenance: dict[str, Any]
    counts: dict[str, Any]


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
    (``apply_quality_mask``) and the companion-evidence band masking
    (``plot_tools._band_time_mean``), so a tolerance fix can never silently
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


def cos_lat_weights(da: xr.DataArray) -> xr.DataArray | None:
    """Cos(latitude) cell weights broadcastable over ``da``, or ``None`` when
    no latitude dimension is identifiable (point data, an already-flattened
    array). The ONE weight definition — the area-weighted regional mean and
    the area-weighted QA pass rate (T55) sit in the same stat grid, so they
    must be weighted by the same thing or they quietly describe different
    fields.

    float64 regardless of how the granule stores its latitude axis: real
    products publish float32 lat, and accumulating float32 weights makes the
    answer depend on how many zero-weight (masked-out) cells happen to be
    summed -- the same region answered ~5e-7 differently from a continental
    granule than from a tight crop of it.
    """
    from tta_backend.utils.geo_utils import find_lat_coord

    lat_name = find_lat_coord(da)
    if lat_name is None or lat_name not in da.dims:
        return None
    return np.cos(np.deg2rad(da[lat_name].astype("float64")))


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
    values = np.asarray(da.values, dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("No finite values available for statistic.")

    weights = cos_lat_weights(da)
    if weights is None:
        return float(np.nanmean(values[np.isfinite(values)]))

    result = float(da.weighted(weights).mean(skipna=True))
    if not np.isfinite(result):
        raise ValueError("No finite values available for statistic.")
    return result


def flag_pass_condition(qf: xr.DataArray, good_values: Any = None, bad_values: Any = None) -> xr.DataArray:
    """Boolean condition of flag cells passing QA — the ONE good/bad flag
    doctrine (T25). ``apply_quality_mask`` both masks with this condition and
    counts the reported pass rate from it (``_count_qa_pixels``, T55), so the
    number on screen is derived from the same boolean that gutted the data and
    is structurally incapable of disagreeing with it. There is no sibling
    implementation left to keep in step: the evidence path's second copy was
    retired when this became the single source.

    With ``good_values``, membership passes (``isin`` already excludes an
    absent/NaN flag). With only ``bad_values``, a pixel passes only when its
    flag is present AND not bad — an unknown-quality pixel (NaN/fill flag,
    e.g. OMI_HCHO's uncomputed-quality) is never counted as good.
    """
    if good_values is not None:
        return qf.isin(good_values)
    return qf.notnull() & ~qf.isin(bad_values)


def _count_qa_pixels(da: xr.DataArray, qf: xr.DataArray, condition: xr.DataArray) -> dict[str, Any]:
    """Count the QA outcome of the pixels ``apply_quality_mask`` is about to
    mask, from the SAME boolean ``condition`` it masks with (T55).

    ``checked`` is ``da.notnull()`` measured *after* fill/valid-range masking
    and, on both real tool paths, after the geometry crop -- so the
    denominator is "pixels in the AOI that had a retrievable value at all",
    never the raw grid size. A fill pixel is not a QA failure and must not be
    counted as one.
    """
    with phase_timer("qa_pass_rate", cells_in=_cell_count(da)):
        checked = da.notnull()
        # ``np.isfinite``, deliberately not ``notnull``: this feeds the
        # valid-timestep scan below, whose definition of "this timestep
        # survived masking" excludes +-inf as well as NaN. It has to be
        # exactly that boolean to be allowed to replace it.
        finite = np.isfinite(da)
        checked, finite, qf, condition = xr.align(
            checked, finite, qf, condition, join="inner",
        )
        # The extent every counter below is reduced over, read off the aligned
        # array rather than declared by a caller -- so the disclosure cannot
        # claim a region the reductions did not actually cover. A pass rate is
        # uninterpretable without it: "4 checked pixels" reads the same whether
        # that was a whole grid or a single monitor cell, and a point series
        # masked before it was narrowed used to report its entire grid here.
        counted_extent = {str(dim): int(size) for dim, size in checked.sizes.items()}
        passing = checked & condition
        # A pixel whose own flag ``_decode_flag_fill`` nulled lands in the
        # failing bucket -- correct masking (unknown quality reads as absent,
        # not good), but in the *report* it would collapse "failed QA" into "QA
        # unknown". Count it separately so the disclosure keeps them apart.
        flag_missing = checked & qf.isnull()
        # Cos(latitude) area-weighted, the SAME weights ``area_weighted_mean``
        # uses: an unweighted pixel ratio sitting in the same stat grid as an
        # area-weighted mean over-counts shrunken poleward cells (Finding #13).
        # The raw integer counts are kept alongside -- they are the honest "how
        # many observations" fact the disclosure text needs.
        weights = cos_lat_weights(checked)
        if weights is None:
            weights = 1.0
        checked_area, passing_area = checked * weights, passing * weights

        # ONE compute for every reduction. Each of these walks the same
        # fill/valid-masked graph, and a compute per counter would re-read the
        # bundle per counter.
        #
        # MEASURED, not assumed (T51's discipline), and re-measured after the
        # first estimate proved optimistic. Counting a lazily-opened 3-granule
        # bundle through a full ``aggregate()`` with a dask.delayed load
        # counter showed the source read THREE times, not twice:
        #   1. this compute,
        #   2. ``_valid_time_indices``' per-timestep ``.values`` scan, and
        #   3. the temporal reduction, forced downstream at ``.values``.
        # Pass 2 is now folded into pass 1 -- the ``valid_by_time`` reduction
        # below is the same boolean that scan was computing, so it rides this
        # graph walk for free and ``_valid_time_indices`` consumes the answer
        # instead of re-deriving it. Verified at 2 reads per granule, down
        # from 3, by ``test_the_bundle_is_read_once_per_pass_...``.
        #
        # Two passes is the FLOOR here, not a compromise pending more work: the
        # reduction graph cannot be built until the valid timesteps are known
        # (``_cadence_weighted_mean`` weights each cadence bucket by how many
        # VALID granules landed in it, and ``_build_meta`` reports their dates),
        # so scan-then-reduce is a genuine data dependency. Fusing this compute
        # into the reduction instead would hit the same floor from the other
        # side while also forcing ``aggregate`` to return an eager result.
        # Persisting the masked array would collapse the passes but trades I/O
        # for RAM on exactly the multi-granule bundles that have OOM'd this
        # backend before, so it is deliberately not done here.
        reductions = {
            "checked": checked.sum(),
            "passing": passing.sum(),
            "flag_missing": flag_missing.sum(),
            "checked_area": checked_area.sum(),
            "passing_area": passing_area.sum(),
        }
        # Per-timestep rates, reduced over the spatial dims of the SAME
        # pre/post arrays, so a day the mask gutted stays visible instead of
        # being averaged into one cumulative fraction.
        time_dim = identify_time(checked)
        if time_dim is not None and time_dim in checked.dims and time_dim in checked.coords:
            spatial = [d for d in checked.dims if d != time_dim]
            reductions["checked_area_by_time"] = checked_area.sum(spatial)
            reductions["passing_area_by_time"] = passing_area.sum(spatial)
            # ``_aggregate``'s valid-timestep scan, folded in here. A timestep
            # survives masking iff some pixel is finite AFTER the QA mask, and
            # ``da.where(condition)`` makes that exactly ``finite & condition``
            # -- so this is not an approximation of the scan, it is the scan,
            # moved onto a graph walk that was happening anyway.
            reductions["valid_by_time"] = (finite & condition).any(spatial)
        else:
            time_dim = None
        reduced = xr.Dataset(reductions).compute()

    counts = {
        "checked": int(reduced["checked"]),
        "passing": int(reduced["passing"]),
        "flag_missing": int(reduced["flag_missing"]),
        "counted_extent": counted_extent,
    }
    area_checked = float(reduced["checked_area"])
    if area_checked > 0:
        counts["pass_rate"] = float(reduced["passing_area"]) / area_checked
    if time_dim is not None:
        counts["pass_rate_by_time"] = _by_time_rates(
            reduced["passing_area_by_time"], reduced["checked_area_by_time"],
        )
        counts["times"] = [
            pd.Timestamp(t).isoformat() for t in np.asarray(reduced[time_dim].values)
        ]
        # Handed back for ``_valid_time_indices`` to consume instead of
        # re-scanning. Carries the dimension it was reduced over so the
        # consumer can refuse a mismatch rather than trust positions blindly.
        counts["valid_time_dim"] = time_dim
        counts["valid_time_flags"] = [
            bool(v) for v in np.asarray(reduced["valid_by_time"].values)
        ]
    return counts


def _by_time_rates(passing_area: xr.DataArray, checked_area: xr.DataArray) -> list[float | None]:
    """Per-timestep pass rates, ``None`` for a timestep with nothing
    retrievable to check -- the same absent-not-zero honesty the cumulative
    rate gets, and it keeps the series index-aligned with its timestamps."""
    rates: list[float | None] = []
    for numerator, denominator in zip(
        np.asarray(passing_area.values, dtype="float64"),
        np.asarray(checked_area.values, dtype="float64"),
    ):
        rates.append(round(float(numerator) / float(denominator), 6) if denominator > 0 else None)
    return rates


# What the reported pass rate's denominator actually is, in one self-describing
# line, so the number explains itself in CSV/metadata export and not only in
# JSX -- and so nobody reads it as interchangeable with "Valid values %", which
# answers the different question "did we get data at all".
QA_PASS_RATE_BASIS = (
    "cos(latitude)-weighted fraction of pixels in the analyzed region that had a "
    "retrievable value (after fill/valid-range masking) and passed the QA flag"
)


def _qa_pass_rate_provenance(counts: dict[str, Any]) -> dict[str, Any]:
    """The realized-QA disclosure keys for ``masking_provenance``, from the
    counters ``apply_quality_mask`` recorded (T55).

    Empty when QA masking never ran: the keys are *absent*, not ``null``,
    matching the existing downgrade-to-not-applied honesty guard, so
    "QA didn't run" stays distinguishable from a real 0% pass rate.
    ``checked == 0`` is a different, real state -- a fully fill- or
    cloud-covered scene -- and reports ``qa_checked_pixels: 0`` with no rate,
    so the UI can say "no retrievable pixels to check" rather than the wrong
    "Not applied".
    """
    if not counts:
        return {}
    provenance = {
        "qa_checked_pixels": counts["checked"],
        "qa_passing_pixels": counts["passing"],
        "qa_flag_missing_pixels": counts["flag_missing"],
        "qa_counted_extent": counts["counted_extent"],
        "qa_pass_rate_basis": QA_PASS_RATE_BASIS,
    }
    if "pass_rate" in counts:
        provenance["qa_pass_rate"] = round(counts["pass_rate"], 6)
    if "pass_rate_by_time" in counts:
        provenance["qa_pass_rate_by_time"] = counts["pass_rate_by_time"]
        provenance["qa_pass_rate_times"] = counts["times"]
    return provenance


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
        # The private form, for its third field: the counters come back
        # alongside the provenance so the valid-timestep scan below can reuse
        # the per-timestep reduction they already computed, instead of
        # re-reading the bundle to derive the same booleans. This is the only
        # caller that needs them, which is why they are not on the public
        # interface.
        masked = self._resolve_and_mask(
            da,
            variable=variable,
            col_info=col_info,
            collection_id=collection_id,
            umm_var_facts=umm_var_facts,
            qa_good_tokens=qa_good_tokens,
            source_ds=qf_source,
        )
        da, masking_provenance = masked.data, masked.provenance

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
            valid_indices = self._valid_time_indices(da, time_dim, masked.counts)
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
        """Mask ``da`` honestly and disclose exactly what was masked.

        See ``_resolve_and_mask`` for the resolution doctrine. This is the
        interface every caller outside this module uses: the masked array and
        its complete masking provenance, nothing else to learn. The QA counters
        the mask recorded stay behind the seam -- ``aggregate`` is their only
        consumer and it calls the private form directly.

        Returns ``(masked_da, masking_provenance)``.
        """
        masked = self._resolve_and_mask(
            da,
            variable=variable,
            col_info=col_info,
            collection_id=collection_id,
            umm_var_facts=umm_var_facts,
            qa_good_tokens=qa_good_tokens,
            source_ds=source_ds,
        )
        return masked.data, masked.provenance

    def _resolve_and_mask(
        self,
        da: xr.DataArray,
        *,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
        collection_id: str | None = None,
        umm_var_facts: Any = None,
        qa_good_tokens: list[str] | None = None,
        source_ds: xr.Dataset | None = None,
    ) -> MaskedField:
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

        Returns a ``MaskedField``: the masked array, its complete masking
        provenance, and the raw QA counters this call computed. The counters
        ride the return value rather than a caller-supplied out-dict so the
        reuse channel is visible in the type -- ``aggregate`` consumes them for
        its valid-timestep scan instead of forcing a second I/O pass over the
        bundle, and no other caller has to know they exist.
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
        # Distinguish "no flag to interpret" from "flag we could not interpret".
        # ``_resolve_qa_flag_var`` returns a pinned name even when that band is
        # absent from the opened view, so a None here means we looked at a real
        # Dataset and it publishes no quality flag at all -- nothing to pin,
        # nothing to fix. Only claimable when a Dataset was actually available.
        if (
            source_ds is not None
            and qf_var is None
            and qa_provenance.get("qa_status") == QA_NOT_APPLIED
        ):
            qa_provenance = {**qa_provenance, "qa_status": QA_NO_FLAG_VARIABLE}

        masking_provenance.update(qa_provenance)

        # T55: count the QA outcome where the mask is actually applied and fold
        # it into the same provenance the disclosure already travels in. Every
        # caller narrows to its analyzed region -- an AOI crop on the plot/stat
        # paths, the monitor cell on the validation path -- *before* masking, so
        # the counters mean the same thing on a heatmap, a timeseries, and a
        # point series. ``qa_counted_extent`` states which of those it was.
        da, counts = self._apply_quality_mask(
            da, source_ds, resolved_col_info, variable=variable,
        )
        masking_provenance.update(_qa_pass_rate_provenance(counts))
        return MaskedField(da, masking_provenance, counts)

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
            from tta_backend.preprocessing.variable_choice_builder import build_variable_choice

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
        from tta_backend.datasets.mask_info import col_info_for_short_name, short_name_from_attrs

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

    def _apply_quality_mask(
        self,
        da: xr.DataArray,
        ds: xr.Dataset | None = None,
        col_info: dict[str, Any] | None = None,
        *,
        apply_quality_flag: bool = True,
        variable: str | None = None,
        umm_var_facts: Any = None,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Apply fill / valid-range / QA-flag masking and report what the QA
        pass counted.

        Internal to this module: ``_resolve_and_mask`` is the only caller, and
        it is what resolves the masking facts this consumes. The returned
        counters are empty when QA-flag masking did not run at all -- absent,
        never zeroed, so "no QA mask" stays distinguishable from "nothing
        passed".
        """
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
            from tta_backend.datasets.mask_info import _split_valid_range_attr

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

        counts: dict[str, Any] = {}
        qf_var = col_info.get("quality_flag_var")
        if apply_quality_flag and ds is not None and qf_var and qf_var in ds.data_vars:
            qf = ds[qf_var]
            good_values = col_info.get("qa_good_values")
            bad_values = col_info.get("qa_bad_values")
            if good_values is not None or bad_values is not None:
                qf = self._decode_flag_fill(qf)
                condition = flag_pass_condition(qf, good_values, bad_values)
                counts = _count_qa_pixels(da, qf, condition)
                da = da.where(condition)
        return da, counts

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

    def qa_flag_variable(
        self,
        ds: xr.Dataset | None,
        da: xr.DataArray,
        col_info: dict[str, Any] | None = None,
    ) -> str | None:
        """The name of ``da``'s sibling QA-flag variable, or ``None``.

        The identification half of the masking doctrine, for callers who need
        to know *which* variable carries quality without masking anything --
        plot_tools excludes it from the evidence band loop, because T55 already
        reports its pass rate once, from the mask itself, and a second
        computation there could only disagree.

        Answers only with a name the caller can use: a flag pinned in
        ``col_info`` but absent from the opened view is not reachable, so it is
        not an answer. Total by construction -- an unreadable or malformed view
        resolves to "no flag identified" rather than raising, since this
        question is always best-effort and never worth failing a chart over.
        """
        try:
            qf_var, _ = self._resolve_qa_flag_var(ds, da, col_info or {})
        except Exception:  # pragma: no cover -- defensive, see docstring
            return None
        if qf_var and qf_var in getattr(ds, "data_vars", {}):
            return qf_var
        return None

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

    def _valid_time_indices(
        self,
        da: xr.DataArray,
        time_dim: str,
        qa_pixel_counts: dict[str, Any] | None = None,
    ) -> list[int]:
        """Which timesteps still hold a finite value after masking.

        Prefers the ``valid_by_time`` reduction ``_count_qa_pixels`` already
        computed on the same graph (see its comment): re-deriving it here is a
        second full I/O pass over a lazily-opened bundle for a boolean that has
        already been paid for. Falls back to computing it directly when QA
        masking did not run -- and then as ONE reduction rather than the
        previous ``.values`` per timestep, which forced a separate graph walk
        for every granule.
        """
        flags = self._fused_valid_flags(da, time_dim, qa_pixel_counts)
        if flags is None:
            spatial = [d for d in da.dims if d != time_dim]
            flags = np.atleast_1d(np.asarray(np.isfinite(da).any(spatial).values))
        return [i for i, is_valid in enumerate(flags) if bool(is_valid)]

    @staticmethod
    def _fused_valid_flags(
        da: xr.DataArray, time_dim: str, qa_pixel_counts: dict[str, Any] | None,
    ) -> list[bool] | None:
        """The precomputed per-timestep validity flags, or ``None`` when they
        cannot be trusted to index ``da``'s time axis.

        The counters are reduced over the inner-aligned arrays, which is the
        same axis ``da.where(condition)`` produced -- but positional indices
        are only safe if that holds, so a different dimension name or length
        refuses the shortcut and re-scans instead of silently dropping the
        wrong granules.
        """
        if not qa_pixel_counts:
            return None
        flags = qa_pixel_counts.get("valid_time_flags")
        if flags is None or qa_pixel_counts.get("valid_time_dim") != time_dim:
            return None
        if len(flags) != da.sizes.get(time_dim):
            return None
        return flags

    def temporal_mean(self, da: xr.DataArray, time_dim: str, cadence: str) -> xr.DataArray:
        """Collapse ``time_dim`` to "the average over the period".

        The public form of the cadence-bucket weighting below, for a caller
        that has already reduced space itself and only needs time collapsed --
        the vertical profile, whose reduction runs space-then-time (T56 D4) and
        so arrives here holding a (timestep x layer) matrix rather than a field.
        Without this the profile would have to reach into a private method or,
        worse, spell its own ``mean(dim=time)`` and quietly reintroduce the
        clustered-sampling bias Finding #11 removed.
        """
        return self._cadence_weighted_mean(da, time_dim, cadence)

    def cadence_for(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        collection_id: str | None = None,
        variable: str | None = None,
        col_info: dict[str, Any] | None = None,
    ) -> str:
        """The product's publishing cadence, or ``"unknown"``. Public because
        ``temporal_mean`` needs one and only this module knows how it is
        resolved (data attrs -> col_info -> registry)."""
        return self._cadence(data, collection_id, variable, col_info)

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
