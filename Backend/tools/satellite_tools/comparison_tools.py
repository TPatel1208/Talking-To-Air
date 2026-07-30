"""
comparison_tools.py
--------------------
Region/period comparison tools (PRD T08).

Compares two already-retrieved handles of the same variable — two AOIs over
one period (mode="region") or one AOI over two periods (mode="period") — and
renders the result as a `comparison` artifact (T06). Region mode never
differences (the two domains aren't spatially comparable cell-by-cell): it
renders side-by-side panels on a shared color scale plus per-region stats.
Period mode grid-aligns both retrievals via the MCP's `align` transform,
differences B minus A cell-by-cell (cells missing on either side excluded
from both the map and the stats), and reports a diverging difference map
plus anomaly statistics.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional, TypeVar

import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool

from config.workflow_stages import STAGE_RENDER
from datasets.mask_info import col_info_for_short_name, short_name_from_attrs
from earthdata_mcp.results import CATEGORY_PROVIDER_UNAVAILABLE, MCPToolError, parse_tool_result
from preprocessing.aggregation_service import (
    AggregationService,
    VariableChoiceRequired,
    area_weighted_mean,
)
from preprocessing.variable_choice_builder import emit_variable_choice_payload
from services.open_handle import OpenHandleError, open_handle
from tools.satellite_tools.plot_tools import (
    _da_to_heatmap_payload,
    _delivered_scope,
    _normalize_longitudes,
    _percentile_bounds,
    _requested_scope,
    _save_chart,
    _time_range as _pt_time_range,
)
from utils.geo_utils import find_lat_coord, find_lon_coord
from utils.plotting import _normalize_to_2d
from utils.streaming import emit_status

logger = logging.getLogger(__name__)

_aggregation_service = AggregationService()

# Ride a compare through a transient MCP outage instead of failing on the first
# blip. compare is the heaviest MCP caller (open A + open B + align + open
# aligned), so it disproportionately lands in the earthdata-mcp server's
# crash-restart windows (~10-15s, diagnosed 2026-07-20) — the single biggest
# source of its "temporary service interruption" flakiness. Five tries with
# exponential backoff span ~20s of retrying, covering an observed restart with
# margin; the backoff sleeps also let call_tool's circuit-breaker recovery
# cooldown elapse so a half-open probe can re-close the breaker mid-compare.
_TRANSIENT_RETRY_ATTEMPTS = 5
_TRANSIENT_RETRY_BASE_DELAY_SECONDS = 1.5
_TRANSIENT_RETRY_MAX_DELAY_SECONDS = 10.0

_T = TypeVar("_T")


async def _retry_transient(op: Callable[[], Awaitable[_T]], *, label: str) -> _T:
    """Await ``op()``, retrying only a transient ``provider_unavailable``
    MCPToolError (a transport blip / the MCP server restarting) up to
    ``_TRANSIENT_RETRY_ATTEMPTS`` times with exponential backoff. Any other
    MCPToolError category (user_input / contract / too_large / …) is a real,
    non-retryable answer and re-raises immediately, as does the last transient
    attempt. Mirrors await_retrieval's poll-retry loop (services.retrieval_composites)."""
    delay = _TRANSIENT_RETRY_BASE_DELAY_SECONDS
    for attempt in range(_TRANSIENT_RETRY_ATTEMPTS):
        try:
            return await op()
        except MCPToolError as exc:
            if exc.category != CATEGORY_PROVIDER_UNAVAILABLE or attempt == _TRANSIENT_RETRY_ATTEMPTS - 1:
                raise
            logger.warning(
                "compare_transient_retry",
                extra={
                    "_event": "compare_transient_retry",
                    "_label": label,
                    "_attempt": attempt + 1,
                    "_attempts": _TRANSIENT_RETRY_ATTEMPTS,
                },
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _TRANSIENT_RETRY_MAX_DELAY_SECONDS)
    raise AssertionError("unreachable: retry loop exited without returning or raising")

# Percent change is suppressed when period A's mean is smaller than this
# fraction of A's own 2–98th-percentile magnitude range — i.e. the baseline is
# too small, relative to the data's own scale, to normalize by without the
# percentage exploding. A fraction of the field's own spread (not an absolute
# constant) keeps the guard scale-free across variables of wildly different
# magnitudes.
_PERCENT_CHANGE_FLOOR_FRACTION = 0.05


def _difference(da_a, da_b):
    """period B minus period A, cell-by-cell.

    xarray subtraction already propagates NaN — a cell missing (fill-masked)
    on either side becomes NaN in the difference, i.e. excluded from both
    the rendered map and any stat computed over it, never a fabricated 0.
    """
    return da_b - da_a


def _source_labels(aligned) -> list[str] | None:
    """String values of the ``source`` coordinate, if it carries them — the
    handles the MCP stamped for each aligned slice. None when there is no
    ``source`` coordinate to read labels from."""
    if "source" not in getattr(aligned, "coords", {}):
        return None
    return [str(v) for v in np.atleast_1d(aligned["source"].values)]


def _aligned_source_order(aligned, handle_a, handle_b):
    """The (A, B) selectors for pulling the two sources out of the aligned
    cube. Prefers *label* selection when the ``source`` coordinate stamps both
    handles (an MCP-side reorder then can't flip which slice is A); falls back
    to positional order otherwise. Each selector is ``("sel", handle)`` or
    ``("isel", index)``, applied by ``_apply_source_selector`` — shared by the
    science DataArray split and the sibling-Dataset split so their per-source
    pairing can never diverge."""
    labels = _source_labels(aligned)
    if handle_a and handle_b and labels is not None and handle_a in labels and handle_b in labels:
        return ("sel", handle_a), ("sel", handle_b)
    return ("isel", 0), ("isel", 1)


def _apply_source_selector(obj, selector):
    kind, key = selector
    return obj.sel(source=key) if kind == "sel" else obj.isel(source=key)


def _split_aligned(aligned, handle_a: str | None = None, handle_b: str | None = None):
    """Split the MCP `align` transform's output into its two source arrays.

    ``align(source_handles=[a, b])`` grid-aligns >=2 gridded inputs into one
    cube_ handle with a leading ``source`` dimension of length 2. Which slice
    is A vs B is resolved by *label* when the aligned cube stamps the source
    handles on the ``source`` coordinate (harmony-retrieval-mcp PRD-023) —
    passing ``handle_a``/``handle_b`` then selects by name, so an MCP-side
    reorder of the ``source`` dim cannot silently flip every difference sign
    (the worst wrong answer: the map still looks right).

    Without stamped labels (or without the handles) it falls back to positional
    order — ``isel(source=0)`` is A, ``isel(source=1)`` is B — which relies on
    the MCP preserving input order in ``source``. That assumption is pinned by
    a contract test in the harmony-retrieval-mcp repo
    (``test_align_preserves_source_order``); this docstring is its TTA-side
    anchor. Until PRD-023 stamps labels, positional order is the live path.
    """
    if "source" not in aligned.dims:
        raise ValueError(f"Aligned result has no 'source' dimension: dims={list(aligned.dims)}")
    n_sources = aligned.sizes["source"]
    if n_sources != 2:
        raise ValueError(f"Expected an aligned result with 2 sources, found {n_sources}.")
    sel_a, sel_b = _aligned_source_order(aligned, handle_a, handle_b)
    return _apply_source_selector(aligned, sel_a), _apply_source_selector(aligned, sel_b)


def _percent_change_floor(a_paired_vals) -> float | None:
    """The smallest period-A mean magnitude a percent change may be normalized
    by: ``_PERCENT_CHANGE_FLOOR_FRACTION`` of |A|'s own 2–98th-percentile
    range over the paired-valid cells. Returns None when A has no valid cells
    (nothing to scale against)."""
    valid = np.abs(a_paired_vals[np.isfinite(a_paired_vals)])
    if valid.size == 0:
        return None
    spread = float(np.percentile(valid, 98) - np.percentile(valid, 2))
    return _PERCENT_CHANGE_FLOOR_FRACTION * spread


def _anomaly_stats(da_a, da_b, diff, threshold: float | None) -> dict:
    """Mean difference, percent change (relative to period A's mean), and
    optionally the area exceeding a threshold change magnitude.

    All computed only over cells valid on both sides (``diff`` already
    carries NaN wherever either side was missing — see ``_difference``).
    Means are cos(latitude) area-weighted (``area_weighted_mean``) — on a
    lat/lon grid an unweighted cell mean over-weights high latitudes, and a
    percent change built from two such means inherits the bias.

    Percent change is withheld (``percent_change=None`` with a
    ``percent_change_note`` explaining why) when period A's mean is too small,
    relative to the field's own dynamic range, to normalize by — see
    ``_percent_change_floor``. Near-zero baselines are routine for anomaly-like
    fields, and dividing by one yields an absurd "+48,000%" with full
    confidence. The absolute ``mean_difference`` is always reported regardless.
    """
    diff_vals = np.asarray(diff.values, dtype=float)
    valid_diff = diff_vals[np.isfinite(diff_vals)]

    # Period A restricted to the same paired (valid-on-both-sides) cells,
    # kept as a DataArray so the weighted mean still sees the latitude axis.
    a_paired = da_a.where(diff.notnull())
    a_paired_vals = np.asarray(a_paired.values, dtype=float)

    mean_difference = area_weighted_mean(diff) if valid_diff.size else None
    mean_a = area_weighted_mean(a_paired) if np.isfinite(a_paired_vals).any() else None

    percent_change = None
    percent_change_note = None
    if mean_difference is not None and mean_a is not None:
        floor = _percent_change_floor(a_paired_vals)
        if mean_a == 0.0 or (floor is not None and abs(mean_a) < floor):
            percent_change_note = (
                "Percent change withheld — baseline too small to normalize by: period A's "
                f"area-weighted mean ({mean_a:.3g}) is under {_PERCENT_CHANGE_FLOOR_FRACTION:.0%} of "
                "the field's own 2–98th-percentile range, where a percentage would explode. "
                "The absolute mean difference is reported instead."
            )
        else:
            percent_change = (mean_difference / mean_a) * 100.0

    stats = {
        "n_cells": int(valid_diff.size),
        "mean_difference": mean_difference,
        "percent_change": percent_change,
    }
    if percent_change_note is not None:
        stats["percent_change_note"] = percent_change_note

    if threshold is not None:
        exceeding = valid_diff[np.abs(valid_diff) >= threshold]
        stats["area_exceeding_threshold"] = {
            "threshold": threshold,
            "n_cells": int(exceeding.size),
            "fraction": (exceeding.size / valid_diff.size) if valid_diff.size else 0.0,
        }

    return stats


def _region_stats(da, *, basis: str | None = None) -> dict | None:
    """Basic descriptive stats over da's valid cells, or None if none are
    valid. The mean is cos(latitude) area-weighted (``area_weighted_mean``);
    median/max/min are order statistics, which weighting doesn't move enough
    to justify a weighted-quantile dependency.

    ``basis`` names the field these stats summarize (Finding #12). Compare
    always time-means each side, so max/min are the extremes of the
    *period-mean* map — below the instantaneous peak/trough. Disclosing the
    basis lets the label read "Max (period-mean)" instead of a bare "Max" that
    invites reading it as the true peak. Omitted (no key) when unnamed, so the
    stats shape is unchanged for callers summarizing a plain snapshot."""
    values = np.asarray(da.values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    stats = {
        "mean": area_weighted_mean(da),
        "median": float(np.median(valid)),
        "max": float(np.max(valid)),
        "min": float(np.min(valid)),
        "n_pixels": int(valid.size),
    }
    if basis:
        stats["basis"] = basis
    return stats


def _empty_overlap_error(da, label: str) -> str | None:
    """Reject a side with no valid data at all — no overlap with its requested window."""
    values = np.asarray(da.values, dtype=float)
    if not np.isfinite(values).any():
        return f"No valid data found for {label} — its retrieval has no overlap with the requested window."
    return None


def _time_range(da) -> tuple[str, str] | tuple[None, None]:
    if "time" not in da.coords:
        return None, None
    times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
    if not times:
        return None, None
    return times[0], times[-1]


def _disjoint_periods_error(da_a, da_b) -> str | None:
    """Region mode compares two AOIs over what should be the same period —
    reject if their time windows don't even overlap (a plain-language guard,
    not a proxy for period-mode's own aligned differencing)."""
    start_a, end_a = _time_range(da_a)
    start_b, end_b = _time_range(da_b)
    if start_a is None or start_b is None:
        return None
    if end_a < start_b or end_b < start_a:
        return (
            f"The two retrievals cover disjoint time periods ({start_a[:10]}..{end_a[:10]} vs "
            f"{start_b[:10]}..{end_b[:10]}) — region mode compares two areas over the same "
            "period; use mode=\"period\" to compare two periods instead."
        )
    return None


def _variable_mismatch_error(da_a, da_b) -> str | None:
    """Return an error message if da_a/da_b are different variables, else None."""
    name_a = (da_a.name or "").strip()
    name_b = (da_b.name or "").strip()
    if name_a and name_b and name_a != name_b:
        return (
            f"Cannot compare different variables: '{name_a}' (A) vs '{name_b}' (B). "
            "compare only supports comparing the same variable across two regions or periods."
        )
    return None


def _units_mismatch_error(da_a, da_b) -> str | None:
    """Return an error message if da_a/da_b carry different (non-empty) units,
    else None.

    ``compare`` differences raw numbers cell-by-cell; two products both naming
    their column ``vertical_column_troposphere`` but storing it in molec/cm^2
    vs mol/m^2 would otherwise be differenced as if commensurable, under a
    single (wrong) unit label. No unit conversion is attempted here (out of
    scope) -- naming both units is the actionable answer, mirroring
    ``_variable_mismatch_error``.

    Absent units on ONLY ONE side is the more dangerous case, not a safe one:
    the difference would still be differenced and stamped with the side that
    *does* declare units (see the caller's ``units = da_a.attrs...`` label),
    presenting an unverifiable-commensurability comparison as a real number
    under a confident label -- so it is a hard stop too. Only when *neither*
    side declares units is there nothing to mislabel (same variable, no unit
    label emitted), and the comparison proceeds."""
    units_a = str(da_a.attrs.get("units", "") or "").strip()
    units_b = str(da_b.attrs.get("units", "") or "").strip()
    if units_a and units_b:
        if units_a != units_b:
            return (
                f"Cannot compare values in different units: '{units_a}' (A) vs '{units_b}' (B). "
                "compare does not convert units — retrieve both sides in the same units to difference them."
            )
        return None
    if units_a or units_b:
        declared, missing_side = (units_a, "B") if units_a else (units_b, "A")
        return (
            f"Cannot compare: one side declares units ('{declared}') and the other ({missing_side}) "
            "has none, so the two cannot be confirmed commensurable before differencing. "
            "Retrieve both sides with matching units to difference them."
        )
    return None


def _mask_col_info(da, ds=None) -> dict:
    """See plot_tools._mask_col_info: collection identity is dataset-level,
    so ``ds.attrs`` is checked before the per-variable ``da.attrs`` (T25
    masking-execution fix)."""
    short_name = (
        short_name_from_attrs(ds.attrs if ds is not None else None)
        or short_name_from_attrs(da.attrs)
        or da.name
        or ""
    )
    return col_info_for_short_name(str(short_name).upper())


def _prepare_2d(da, variable_name: str, source_ds=None):
    """Apply quality masking and collapse to a single 2-D (lat, lon) snapshot
    (time-mean, matching plot_singular's default) so every side of a
    comparison renders as one map.

    Returns ``(reduced_2d, agg_meta)``. The aggregation meta carries the
    masking provenance ``aggregate()`` produced — including any QA-flag mask it
    applied — so the comparison builders can *disclose* that masking instead of
    silently swallowing it (T25/T46: compare must honor the same masking and
    scope-echo doctrines the heatmap/timeseries paths do)."""
    col_info = _mask_col_info(da, source_ds)
    aggregation = _aggregation_service.aggregate(
        da, variable=variable_name, stat="mean", col_info=col_info, source_ds=source_ds,
    )
    reduced = next(iter(aggregation.ds.data_vars.values()))
    return _normalize_to_2d(reduced), aggregation.meta


def _compare_side_provenance(handle: str, da_2d, agg_meta: dict, variable_name: str, units: str) -> dict:
    """Masking + T46 scope-echo provenance for one side of a comparison,
    mirroring plot_tools._provenance's disclosure facts.

    Carries the QA-flag ``masking`` ``aggregate()`` produced, the ``delivered_scope``
    the data actually covers, and the ``requested_scope`` a composite recorded
    for this handle (when one was recorded). The delivered scope's region name
    is set from the recorded request rather than the panel label — compare never
    re-geocodes, so the delivered region IS the requested one, and stamping the
    period/panel label there would fire a spurious region-substitution note."""
    requested = _requested_scope([handle])
    start_date, end_date = _pt_time_range(da_2d, agg_meta)
    region_name = str((requested or {}).get("location") or "")
    provenance = {
        "variable": variable_name,
        "units": units,
        "source_handles": [handle],
        "masking": agg_meta.get("masking"),
        "delivered_scope": _delivered_scope(region_name, start_date, end_date, agg_meta),
        "requested_scope": requested,
    }
    if agg_meta.get("variable_resolution"):
        # T48: if the resolver auto-picked this side's variable, disclose it --
        # same channel and note as the heatmap path (one resolver, one
        # disclosure, no drift between tools).
        provenance["variable_resolution"] = agg_meta["variable_resolution"]
    return provenance


def _bbox_from_da(da) -> list[float]:
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    lats = np.asarray(da[lat_coord].values, dtype=float)
    lons = np.asarray(da[lon_coord].values, dtype=float)
    return [float(np.nanmin(lons)), float(np.nanmin(lats)), float(np.nanmax(lons)), float(np.nanmax(lats))]


# How each comparison scale was derived, disclosed to the frontend legend so
# its saturation warning (T43) fires for compare maps exactly as it does for
# single maps. A shared scale is a plain 2/98 percentile clip of the combined
# panels; a diverging scale clips at the diff's 98th-percentile magnitude.
_SHARED_SCALE_DISCLOSURE = {"method": "percentile", "p": [2, 98]}
_DIVERGING_SCALE_DISCLOSURE = {"method": "percentile_magnitude", "p": 98}


def _shared_bounds(da_a, da_b) -> tuple[float, float]:
    combined = np.concatenate([
        np.asarray(da_a.values, dtype=float).ravel(),
        np.asarray(da_b.values, dtype=float).ravel(),
    ])
    return _percentile_bounds(combined)


def _diverging_bounds(diff_da) -> tuple[float, float]:
    """Symmetric, zero-centered scale sized to the diff's 98th-percentile magnitude."""
    vals = np.asarray(diff_da.values, dtype=float)
    valid = vals[np.isfinite(vals)]
    if valid.size == 0:
        return -1.0, 1.0
    max_abs = float(np.percentile(np.abs(valid), 98))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


async def _open_and_prepare_side(
    handle: str, mcp_tools: dict[str, BaseTool], variable: Optional[str], label: str
):
    """Open one side of a compare (export -> open -> normalize -> select
    variable) and return ``(ds, da, error_json, emit_side_effect)`` — exactly
    one of ``(ds, da)`` or ``error_json`` is populated, never both.

    Both sides run concurrently via asyncio.gather (opening A and B are
    independent MCP round trips, so running them together halves the
    wall-clock wait on the slower side instead of paying for both in
    sequence) — but only ONE side's error is ever returned to the caller
    (make_compare checks err_a before err_b). An ambiguous-variable side's
    UI side effect (emit_variable_choice_payload/emit_status) is therefore
    NOT fired here: firing it unconditionally would broadcast a picker for
    a side whose result the caller may end up discarding entirely (e.g. A
    fails outright while B is merely ambiguous — the caller returns A's
    error, but B's picker would already be on the wire with nothing to
    follow up on). Instead the side effect is returned as a zero-arg
    callable the caller invokes only for whichever side's error actually
    gets returned."""
    try:
        ds = await _retry_transient(lambda: open_handle(handle, mcp_tools), label=f"open {label}")
        # Normalize longitude on the whole opened Dataset before extraction,
        # so the science DataArray and its sibling QA-flag variable (still
        # reachable through ds) share one coordinate convention (T25
        # masking-execution fix; see plot_tools).
        lon_coord = find_lon_coord(ds)
        if lon_coord:
            ds = _normalize_longitudes(ds, lon_coord)
        da = _aggregation_service.to_dataarray(ds, handle=handle, variable=variable)
        return ds, da, None, None
    except VariableChoiceRequired as e:
        # T49: this side of the compare is ambiguous — deliver a
        # deterministic picker (multi-part partial delivery is per-side: this
        # side yields a picker instead of a panel; a resolvable side still
        # renders) — but only if this side's error is the one actually
        # returned; see the emit_side_effect docstring note above.
        def _emit_side_effect(ds=ds, e=e) -> None:
            emit_variable_choice_payload(e.resolution, ds)
            emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)

        return None, None, json.dumps({"error": e.mcp_error.to_dict()}), _emit_side_effect
    except MCPToolError as e:
        return None, None, json.dumps({"error": e.to_dict()}), None
    except OpenHandleError as e:
        return (
            None,
            None,
            json.dumps({"error": f"Failed to open handle '{handle}' ({label}): {e}"}),
            None,
        )


def _region_panel(da, handle: str, title: str, variable_name: str, units: str, vmin: float, vmax: float) -> dict:
    panel = _da_to_heatmap_payload(
        da, title, variable_name, units, render_overlay=True, value_range=(vmin, vmax),
        scale_disclosure=_SHARED_SCALE_DISCLOSURE,
    )
    panel["bounds"] = _bbox_from_da(da)
    panel["metadata"] = {"source_handles": [handle]}
    return panel


def make_compare(mcp_tools: dict[str, BaseTool]):
    @tool
    async def compare(
        handle_a: str,
        handle_b: str,
        mode: str,
        label_a: str = "A",
        label_b: str = "B",
        threshold: Optional[float] = None,
        variable: Optional[str] = None,
    ) -> str:
        """
        Compare two retrievals of the same variable — either two regions
        over one period (mode="region") or one region across two periods
        (mode="period") — and render the result as a comparison artifact.

        region mode: side-by-side panels on a shared color scale plus
        per-region summary statistics. Never differences — the two domains
        aren't spatially comparable cell-by-cell.
        period mode: grid-aligns both retrievals via the MCP's `align`
        transform, then differences period B minus period A cell-by-cell
        (cells missing on either side excluded from both the map and the
        stats), rendering a diverging, zero-centered difference map plus
        mean difference / percent change / area exceeding a threshold change.

        Both retrievals must be the same variable, and (region mode) must
        cover overlapping time windows — mismatches are rejected with a
        plain explanation rather than silently attempted.

        Args:
            handle_a: obs_/cube_ handle for the first region or period.
            handle_b: obs_/cube_ handle for the second region or period.
            mode: "region" (two AOIs, same period) or "period" (one AOI, two periods).
            label_a: panel/legend label for handle_a, e.g. a region or period name.
            label_b: panel/legend label for handle_b.
            threshold: period mode only — absolute change magnitude defining
                a "significant" change, for the area-exceeding-threshold stat.
            variable: Science variable to use for both handles, for a
                multi-variable file with no variable chosen at retrieval time.

        Returns:
            JSON string — comparison chart payload (frontend-renderable) with
            an embedded artifact ref and summary statistics.
        """
        if mode not in ("region", "period"):
            return json.dumps({"error": f"Unknown mode '{mode}'. Use 'region' or 'period'."})

        # Open both sides concurrently — independent MCP round trips (export +
        # download/open), so gathering them halves the wall-clock wait on the
        # slower side instead of paying for both one after another.
        (ds_a, da_a, err_a, emit_a), (ds_b, da_b, err_b, emit_b) = await asyncio.gather(
            _open_and_prepare_side(handle_a, mcp_tools, variable, "A"),
            _open_and_prepare_side(handle_b, mcp_tools, variable, "B"),
        )
        # Only the winning side's deferred UI side effect (if any) fires —
        # the discarded side's ambiguous-variable picker must never reach
        # the client when its result isn't the one being returned.
        if err_a:
            if emit_a:
                emit_a()
            return err_a
        if err_b:
            if emit_b:
                emit_b()
            return err_b

        mismatch = _variable_mismatch_error(da_a, da_b)
        if mismatch:
            return json.dumps({"error": mismatch})

        units_mismatch = _units_mismatch_error(da_a, da_b)
        if units_mismatch:
            return json.dumps({"error": units_mismatch})

        empty_a = _empty_overlap_error(da_a, f"{label_a} ({handle_a})")
        if empty_a:
            return json.dumps({"error": empty_a})
        empty_b = _empty_overlap_error(da_b, f"{label_b} ({handle_b})")
        if empty_b:
            return json.dumps({"error": empty_b})

        variable_name = da_a.name or ""
        units = da_a.attrs.get("units", "")

        emit_status("Building comparison...", stage=STAGE_RENDER)

        if mode == "region":
            disjoint = _disjoint_periods_error(da_a, da_b)
            if disjoint:
                return json.dumps({"error": disjoint})
            # CPU-bound mask -> aggregate -> payload chain (T16), run off
            # the event loop.
            return await asyncio.to_thread(
                _build_region_comparison, handle_a, handle_b, da_a, da_b, label_a, label_b, variable_name, units,
                ds_a, ds_b,
            )

        try:
            align_raw = await _retry_transient(
                lambda: mcp_tools["align"].ainvoke({"source_handles": [handle_a, handle_b]}),
                label="align",
            )
            align_result = parse_tool_result(align_raw)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        aligned_handle = align_result.get("handle")
        if align_result.get("status") == "error" or not aligned_handle:
            return json.dumps({
                "error": align_result.get("message")
                or "Failed to grid-align the two retrievals onto a common grid."
            })

        try:
            aligned_ds = await _retry_transient(
                lambda: open_handle(aligned_handle, mcp_tools), label="open aligned"
            )
            aligned_lon_coord = find_lon_coord(aligned_ds)
            if aligned_lon_coord:
                aligned_ds = _normalize_longitudes(aligned_ds, aligned_lon_coord)
            # variable_name is already resolved from A/B above -- pass it
            # explicitly so the aligned (multi-source) dataset doesn't hit
            # its own ambiguous-variable error for a choice already made.
            aligned_da = _aggregation_service.to_dataarray(aligned_ds, handle=aligned_handle, variable=variable_name)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open the aligned result '{aligned_handle}': {e}"})

        try:
            aligned_a, aligned_b = _split_aligned(aligned_da, handle_a, handle_b)
        except ValueError as e:
            return json.dumps({"error": f"Grid alignment produced an unusable result: {e}"})

        # Split the whole aligned Dataset per-source too (mirroring
        # _split_aligned's own isel), so each side's QA-flag sibling -- if
        # `align` preserved one -- stays reachable through the matching
        # source_ds instead of one carrying an extra 'source' dim the
        # already-sliced science DataArray doesn't have (T25 masking-
        # execution fix).
        if "source" in aligned_ds.dims:
            sel_a, sel_b = _aligned_source_order(aligned_da, handle_a, handle_b)
            aligned_ds_a = _apply_source_selector(aligned_ds, sel_a)
            aligned_ds_b = _apply_source_selector(aligned_ds, sel_b)
        else:
            aligned_ds_a = aligned_ds_b = aligned_ds

        # CPU-bound mask -> aggregate -> payload chain (T16), run off the
        # event loop.
        return await asyncio.to_thread(
            _build_period_comparison, handle_a, handle_b, aligned_handle, aligned_a, aligned_b,
            label_a, label_b, variable_name, units, threshold, aligned_ds_a, aligned_ds_b,
        )

    return compare


def _build_region_comparison(handle_a, handle_b, da_a, da_b, label_a, label_b, variable_name, units, ds_a=None, ds_b=None) -> str:
    da_a_2d, meta_a = _prepare_2d(da_a, variable_name, source_ds=ds_a)
    da_b_2d, meta_b = _prepare_2d(da_b, variable_name, source_ds=ds_b)
    vmin, vmax = _shared_bounds(da_a_2d, da_b_2d)

    panel_a = _region_panel(da_a_2d, handle_a, label_a, variable_name, units, vmin, vmax)
    panel_b = _region_panel(da_b_2d, handle_b, label_b, variable_name, units, vmin, vmax)
    # Disclose each side's own masking + scope echo (T25/T46) on its panel, so a
    # per-panel view stays honest; the top-level provenance mirrors side A for
    # the dispatch-layer scope note (which reads a single chart.provenance).
    prov_a = _compare_side_provenance(handle_a, da_a_2d, meta_a, variable_name, units)
    prov_b = _compare_side_provenance(handle_b, da_b_2d, meta_b, variable_name, units)
    panel_a["provenance"] = prov_a
    panel_b["provenance"] = prov_b
    # Both sides are period-mean maps (``_prepare_2d`` time-means), so disclose
    # that basis on the extremes (Finding #12): a compare "Max" is the max of
    # the averaged field, not the instantaneous peak.
    stats_a = _region_stats(da_a_2d, basis="period-mean")
    stats_b = _region_stats(da_b_2d, basis="period-mean")

    payload = {
        "type": "heatmap_multi",
        "mode": "n-panel",
        "title": f"{variable_name}: {label_a} vs {label_b}",
        "variable": variable_name,
        "units": units,
        "panels": [panel_a, panel_b],
        "stats": {label_a: stats_a, label_b: stats_b},
        "provenance": prov_a,
        "metadata": {"source_handles": [handle_a, handle_b]},
    }
    return _save_chart(payload, f"{variable_name}_{label_a}_vs_{label_b}")


def _build_period_comparison(
    handle_a, handle_b, aligned_handle, aligned_a, aligned_b, label_a, label_b, variable_name, units, threshold,
    aligned_ds_a=None, aligned_ds_b=None,
) -> str:
    da_a_2d, meta_a = _prepare_2d(aligned_a, variable_name, source_ds=aligned_ds_a)
    da_b_2d, meta_b = _prepare_2d(aligned_b, variable_name, source_ds=aligned_ds_b)
    diff = _difference(da_a_2d, da_b_2d)
    stats = _anomaly_stats(da_a_2d, da_b_2d, diff, threshold)

    vmin, vmax = _diverging_bounds(diff)
    diff_payload = _da_to_heatmap_payload(
        diff, f"{variable_name}: {label_b} - {label_a}", variable_name, units, diverging=True,
        render_overlay=True, value_range=(vmin, vmax), scale_disclosure=_DIVERGING_SCALE_DISCLOSURE,
    )
    diff_payload["bounds"] = _bbox_from_da(diff)

    # Disclose the masking aggregate() applied to each side, and the T46 scope
    # echo, on the compare payload — the difference silently masked QA-flagged
    # cells on both sides with zero provenance before. Both sides go through the
    # same aligned product, so side A's masking is representative at top level;
    # each period keeps its own on its panel.
    prov_a = _compare_side_provenance(handle_a, da_a_2d, meta_a, variable_name, units)
    prov_b = _compare_side_provenance(handle_b, da_b_2d, meta_b, variable_name, units)

    payload = {
        "type": "heatmap_multi",
        "mode": "difference",
        "title": f"{variable_name}: {label_b} vs {label_a} (difference)",
        "variable": variable_name,
        "units": units,
        "panels": [
            {"title": label_a, "provenance": prov_a, "metadata": {"source_handles": [handle_a]}},
            {"title": label_b, "provenance": prov_b, "metadata": {"source_handles": [handle_b]}},
        ],
        "difference": diff_payload,
        "stats": stats,
        "provenance": prov_a,
        "sign_convention": f"{label_b} minus {label_a}",
        "metadata": {"source_handles": [handle_a, handle_b, aligned_handle]},
    }
    return _save_chart(payload, f"{variable_name}_{label_b}_minus_{label_a}")
