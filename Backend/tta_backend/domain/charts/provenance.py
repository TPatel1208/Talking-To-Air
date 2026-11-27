"""The provenance, query and export blocks a chart payload carries, and its companion evidence.

``requested_scope`` (the scope a retrieval was asked for) is per-request
state, so the caller looks it up and passes it in.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from tta_backend.datasets.mask_info import resolve_mask_info
from tta_backend.datasets.variable_roles import classify_inventory, related_variables
from tta_backend.domain.aggregation.aggregation_service import AggregationService, area_weighted_mean, fill_match
from tta_backend.domain.earthdata.opening import normalize_longitudes
from tta_backend.domain.geo import find_lat_coord, find_lon_coord, identify_time
from tta_backend.domain.masking.geometry import geometry_mask, sel_bounds
from tta_backend.infrastructure.phase_timing import phase_timer

_aggregation_service = AggregationService()


def time_range(da: xr.DataArray, agg_meta: dict[str, Any] | None = None) -> tuple[str, str]:
    """First and last time of ``da``'s time coordinate, else the aggregation
    meta's dates (a reduced field has lost its time axis; some granules never had one)."""
    if "time" in da.coords:
        times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
        if times:
            return times[0], times[-1]
    if agg_meta:
        start = agg_meta.get("start_date") or ""
        end = agg_meta.get("end_date") or ""
        dates = agg_meta.get("granule_dates") or []
        return start or (dates[0] if dates else ""), end or (dates[-1] if dates else "")
    return "", ""


def query_definition(
    da: xr.DataArray, region: dict[str, Any] | None, aggregation: str,
    chart_parameters: dict[str, Any] | None = None, agg_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_date, end_date = time_range(da, agg_meta)
    query: dict[str, Any] = {
        "dataset": da.name or "",
        "start_date": start_date,
        "end_date": end_date,
        "bbox": list(region["bounds"]) if region else None,
        "aggregation": aggregation,
    }
    if chart_parameters:
        query["chart_parameters"] = chart_parameters
    return {k: v for k, v in query.items() if v not in (None, "", [])}


def dataset_facts(col_info: dict[str, Any] | None) -> dict[str, Any]:
    """Registry facts about the collection the plotted variable came from."""
    col_info = col_info or {}
    provider = col_info.get("provider") or ""
    instrument = col_info.get("instrument") or ""
    return {
        "dataset": col_info.get("short_name") or "",
        "dataset_description": col_info.get("description") or "",
        "dataset_version": col_info.get("version") or "",
        "collection_id": col_info.get("collection_id") or "",
        "provider": provider,
        "instrument": instrument,
        "source": " — ".join(part for part in (provider, instrument) if part),
        "maturity": col_info.get("maturity") or "unknown",
        "maturity_note": col_info.get("maturity_note") or "",
    }


def variable_definition(da: xr.DataArray, col_info: dict[str, Any] | None) -> dict[str, Any]:
    """long_name, units, valid range and fill facts for the plotted variable."""
    col_info = col_info or {}
    valid_min = col_info.get("valid_min")
    valid_max = col_info.get("valid_max")
    fill_value = col_info.get("fill_value")
    has_range = valid_min is not None or valid_max is not None
    has_fill = fill_value is not None
    if has_fill and has_range:
        mask_note = "fill values and a valid range are defined"
    elif has_fill:
        mask_note = "fill values are defined, no valid range"
    elif has_range:
        mask_note = "a valid range is defined, no fill values"
    else:
        mask_note = "no fill/range metadata"
    return {
        "long_name": da.attrs.get("long_name", ""),
        "units": da.attrs.get("units", ""),
        "advisory_notes": [],
        "valid_ranges": {"min": valid_min, "max": valid_max} if has_range else {},
        "fill_value": fill_value,
        "mask_note": mask_note,
    }


def qa_methodology(col_info: dict[str, Any] | None) -> dict[str, Any]:
    """The registry's quality-flag rule, as general methodology."""
    col_info = col_info or {}
    methodology = {
        "quality_flag_var": col_info.get("quality_flag_var"),
        "qa_good_values": col_info.get("qa_good_values"),
        "qa_bad_values": col_info.get("qa_bad_values"),
    }
    return {k: v for k, v in methodology.items() if v is not None}


def inventory_records(ds: xr.Dataset) -> list[dict[str, Any]]:
    """An opened Dataset's bands as ``classify_inventory`` records, each with its ``group_path``."""
    return [
        {
            "name": name,
            "group": var.attrs.get("group_path"),
            "standard_name": var.attrs.get("standard_name"),
            "long_name": var.attrs.get("long_name"),
            "units": var.attrs.get("units"),
        }
        for name, var in ds.data_vars.items()
    ]


def related_variables_view(da: xr.DataArray, col_info: dict[str, Any] | None, ds: xr.Dataset | None = None) -> dict[str, Any]:
    """The plotted variable's role and its quality, uncertainty and context
    siblings, from the opened Dataset's bands when given, else the registry's list."""
    col_info = col_info or {}
    variables = col_info.get("variables")
    with phase_timer("related_variables", from_dataset=ds is not None) as timing:
        if ds is not None and getattr(ds, "data_vars", None):
            variables = inventory_records(ds)
        timing["variables"] = len(variables or [])
        return related_variables(
            variables,
            groups=col_info.get("groups"),
            primary_var=col_info.get("primary_var"),
            quality_flag_var=col_info.get("quality_flag_var"),
            plotted_variable=str(da.name or ""),
        )


def evi_leaf(name: object) -> str:
    """The bare, lower-cased leaf of a possibly group-qualified variable name."""
    return str(name or "").rsplit("/", 1)[-1].lower()


def crop_band_to_region(band: xr.DataArray, region: dict[str, Any]) -> tuple[xr.DataArray | None, int]:
    """A companion band masked to the region's geometry and cropped to its
    bounds, and the number of grid cells inside the region (counted on an
    all-ones twin, so the band's own fill does not shrink it). ``(None, 0)``
    when the band has no lat/lon grid."""
    lon_coord = find_lon_coord(band)
    if lon_coord:
        band = normalize_longitudes(band, lon_coord)
    lat_coord = find_lat_coord(band)
    if lat_coord is None or lon_coord is None:
        return None, 0

    mask = geometry_mask(band, region["geometry"])
    cropped = sel_bounds(band.where(mask), lat_coord, lon_coord, region["bounds"])

    footprint = sel_bounds(xr.ones_like(band).where(mask), lat_coord, lon_coord, region["bounds"])
    in_region = int(np.isfinite(np.asarray(footprint.values)).sum())
    return cropped, in_region


def band_time_mean(band: xr.DataArray, resolved: dict[str, Any]) -> xr.DataArray:
    """Mask a band's fill and out-of-range cells, then take its per-pixel
    mean over time, skipping NaN; unchanged when it has no time dimension."""
    fill = resolved.get("fill_value")
    valid_min = resolved.get("valid_min")
    valid_max = resolved.get("valid_max")
    if fill is not None:
        try:
            band = band.where(~fill_match(band, fill))
        except (TypeError, ValueError):
            pass
    if valid_min is not None:
        band = band.where(band >= valid_min)
    if valid_max is not None:
        band = band.where(band <= valid_max)
    time_dim = identify_time(band)
    if time_dim is not None and time_dim in band.dims:
        band = band.mean(dim=time_dim, skipna=True)
    return band


def band_mean_fact(
    band: xr.DataArray, leaf: str, role: str, region: dict[str, Any] | None, *, pct_of_science: float | None = None,
) -> dict[str, Any] | None:
    """The area-weighted mean of a context or uncertainty band over the
    region, in its own units, with its coverage; ``pct_of_science`` adds the
    band as a fraction of the science mean."""
    resolved, _ = resolve_mask_info(cf_attrs=dict(band.attrs))
    band = band_time_mean(band, resolved)
    if region is None:
        return None
    cropped, in_region = crop_band_to_region(band, region)
    if cropped is None or in_region == 0:
        return None
    vals = np.asarray(cropped.values, dtype="float64")
    valid = np.isfinite(vals)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return None
    mean_val = area_weighted_mean(cropped)
    fact = {
        "name": leaf,
        "role": role,
        "stat": "mean",
        "value": round(mean_val, 6),
        "units": band.attrs.get("units", "") or "",
        "coverage": round(valid_count / in_region, 4),
    }
    # A zero science mean leaves the ratio undefined, so it is left out.
    if pct_of_science is not None and pct_of_science != 0:
        fact["pct_of_science"] = round(abs(mean_val / pct_of_science), 4)
    return fact


class DeferredScienceMean:
    """The plotted variable's area-weighted mean, computed once on first use
    (only an uncertainty band reads it). A lazy array is materialized once
    first; None means no finite data or a failure."""

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da
        self._value: float | None = None
        self.computed = False

    def __call__(self) -> float | None:
        if not self.computed:
            self.computed = True
            self._value = self._compute()
        return self._value

    def _compute(self) -> float | None:
        try:
            da = self._da
            if getattr(da, "chunks", None) is not None:
                da = da.compute()
            return area_weighted_mean(da)
        except Exception:
            return None


def evidence(ds: xr.Dataset | None, da: xr.DataArray, col_info: dict[str, Any] | None, region: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Facts from the quality and context bands beside the plotted variable in
    the opened Dataset: ``{name, role, stat, value, units, coverage}``, with
    ``pct_of_science`` on uncertainty bands. Only bands already in ``ds`` are
    read; any failure gives fewer facts, never an error."""
    if ds is None or not hasattr(ds, "data_vars") or region is None:
        return []
    with phase_timer("evidence", science_cells=int(getattr(da, "size", 0))) as timing:
        facts = evidence_facts(ds, da, col_info, region, timing)
    return facts


def evidence_facts(
    ds: xr.Dataset, da: xr.DataArray, col_info: dict[str, Any] | None, region: dict[str, Any] | None, timing: dict[str, Any],
) -> list[dict[str, Any]]:
    col_info = col_info or {}
    facts: list[dict[str, Any]] = []
    science_leaf = evi_leaf(da.name)
    science_mean = DeferredScienceMean(da)

    # The quality flag's pass rate is reported once, as masking provenance, so
    # its band is left out here.
    qf_var = _aggregation_service.qa_flag_variable(ds, da, col_info)
    qf_leaf = evi_leaf(qf_var) if qf_var else None

    try:
        classified = classify_inventory(
            inventory_records(ds),
            primary_var=col_info.get("primary_var"),
            quality_flag_var=col_info.get("quality_flag_var"),
        )
    except Exception:
        return facts

    timing["bands_classified"] = len(classified)
    bands_read = 0
    for entry in classified:
        role, confidence, name = entry["role"], entry["confidence"], entry["name"]
        leaf, norm = entry["leaf"], evi_leaf(entry["leaf"])
        if confidence not in ("high", "medium"):
            continue
        if role not in ("quality", "context"):
            continue
        if norm == science_leaf or norm == qf_leaf:
            continue
        if name not in ds.data_vars:
            continue
        is_uncertainty = "uncertainty" in norm
        # Quality bands other than uncertainty have no defined summary.
        if role == "quality" and not is_uncertainty:
            continue
        bands_read += 1
        timing["band_cells"] = timing.get("band_cells", 0) + int(getattr(ds[name], "size", 0))
        try:
            fact = band_mean_fact(
                ds[name], leaf, role, region,
                pct_of_science=science_mean() if is_uncertainty else None,
            )
        except Exception:
            fact = None
        if fact:
            facts.append(fact)

    timing["bands_read"] = bands_read
    timing["facts"] = len(facts)
    timing["science_mean_computed"] = science_mean.computed
    return facts


def merged_multi_provenance(panels: list[dict[str, Any]]) -> dict[str, Any]:
    """A multi-panel payload's provenance: the first panel's, with every
    panel's region name, and without the per-panel evidence and related variables."""
    merged = dict(panels[0].get("provenance", {}))
    merged.pop("evidence", None)
    merged.pop("related_variables", None)
    merged["region_name"] = ", ".join(
        panel.get("provenance", {}).get("region_name", "") for panel in panels
    )
    merged["aggregation"] = "single snapshot comparison"
    return merged


def delivered_scope(region_name: str, start_date: str, end_date: str, agg_meta: dict[str, Any] | None) -> dict[str, Any]:
    """The region, dates and cadence the data delivered."""
    return {
        "region_name": region_name,
        "start_date": start_date,
        "end_date": end_date,
        "cadence": (agg_meta or {}).get("cadence"),
    }


def provenance(
    handles: list[str], da: xr.DataArray, region_name: str, aggregation: str,
    agg_meta: dict[str, Any] | None = None, col_info: dict[str, Any] | None = None,
    ds: xr.Dataset | None = None, region: dict[str, Any] | None = None,
    *, requested_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_date, end_date = time_range(da, agg_meta)
    block = {
        "variable": da.name or "",
        "start_date": start_date,
        "end_date": end_date,
        "region_name": region_name,
        "delivered_scope": delivered_scope(region_name, start_date, end_date, agg_meta),
        "requested_scope": requested_scope,
        "region_type": (region or {}).get("region_type"),
        "region_origin": (region or {}).get("region_origin"),
        "display_name": (region or {}).get("display_name") or region_name,
        "aggregation": aggregation,
        "units": da.attrs.get("units", ""),
        "source_handles": list(handles),
        **dataset_facts(col_info),
        "variable_definition": variable_definition(da, col_info),
        "qa_methodology": qa_methodology(col_info),
        "related_variables": related_variables_view(da, col_info, ds=ds),
        "evidence": evidence(ds, da, col_info, region),
    }
    if agg_meta:
        block["aggregation"] = agg_meta["aggregation_label"]
        block["n_granules"] = agg_meta["n_granules"]
        block["cadence"] = agg_meta["cadence"]
        block["granule_dates"] = agg_meta["granule_dates"]
        if agg_meta.get("masking"):
            block["masking"] = agg_meta["masking"]
        if agg_meta.get("variable_resolution"):
            block["variable_resolution"] = agg_meta["variable_resolution"]
        if agg_meta.get("level_resolution"):
            block["level_resolution"] = agg_meta["level_resolution"]
    return block


def attach_reproducibility(
    payload: dict[str, Any],
    handles: list[str],
    da: xr.DataArray,
    region_name: str,
    aggregation: str,
    chart_parameters: dict[str, Any] | None = None,
    agg_meta: dict[str, Any] | None = None,
    region: dict[str, Any] | None = None,
    col_info: dict[str, Any] | None = None,
    ds: xr.Dataset | None = None,
    *,
    requested_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the provenance, query, export and ``metadata.source_handles`` blocks to ``payload``."""
    aggregation_label = agg_meta["aggregation_label"] if agg_meta else aggregation
    with phase_timer("provenance", science_cells=int(getattr(da, "size", 0))):
        payload["provenance"] = provenance(
            handles, da, region_name, aggregation_label, agg_meta, col_info, ds=ds, region=region,
            requested_scope=requested_scope,
        )
    payload["query"] = query_definition(da, region, aggregation_label, chart_parameters, agg_meta)
    payload["export"] = {
        "type": payload.get("type"),
        "variable": da.name or "",
        "units": da.attrs.get("units", ""),
        "region_name": region_name,
        "aggregation": aggregation_label,
        "aggregation_meta": agg_meta or payload.get("aggregation_meta") or {},
        "chart_parameters": chart_parameters or {},
        "source_handles": list(handles),
    }
    if agg_meta:
        payload["query"]["aggregation"] = agg_meta["aggregation_label"]
    payload.setdefault("metadata", {})
    payload["metadata"]["source_handles"] = list(handles)
    return payload
