"""
preprocessing/level_resolver.py
=================================
Turning "300 hPa" into a layer index, and saying how much of an approximation
that was.

``dimension``/``dimension_value`` already mean two things depending on metadata
the caller cannot see -- a physical value on a coordinate-bearing dimension, an
integer index on a coordinate-less one. This module exists so that asking for a
physical level is a *third* parameter rather than a third meaning of the second
(T58 D1): a unit in the request cannot be misread, and a bare number is refused
rather than assumed to be whichever axis happened to be first.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from tta_backend.utils.geo_utils import vertical_axis_kind


@dataclass(frozen=True)
class LevelRequest:
    """A physical level someone asked for: how much, of what, in which units."""

    value: float
    kind: str
    units: str


_LEVEL_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z/^0-9_-]+)\s*$")


def parse_level(text: str, available: dict | None = None) -> LevelRequest:
    """Parse ``"500 hPa"`` / ``"26 km"`` into a request, or refuse.

    Strict on purpose (D11). A bare ``"500"`` is not an under-specified request
    that can be helpfully completed -- on a product publishing both pressure
    and altitude it is genuinely two different questions, and picking one is
    how the T56 routing bug happened. ``available`` (``{kind: units}`` read off
    the granule) lets the refusal answer with the product's OWN vocabulary
    instead of a generic list of units nobody published.
    """
    raw = str(text).strip()
    match = _LEVEL_PATTERN.match(raw)
    if match is None:
        raise _unparseable_level_error(raw, available)
    value, units = match.group(1), match.group(2)
    kind = vertical_axis_kind(_UnitsOnly(units))
    if kind is None:
        raise _unparseable_level_error(raw, available)
    return LevelRequest(value=float(value), kind=kind, units=units)


def _vocabulary(available: dict | None) -> str:
    if not available:
        return "Specify a physical level with its units, such as `500 hPa` or `10 km`."
    published = ", ".join(f"{kind} in {units}" for kind, units in sorted(available.items()))
    example = next(f"`500 {u}`" if k == "pressure" else f"`10 {u}`" for k, u in sorted(available.items()))
    return f"This dataset publishes {published}. Specify a physical level such as {example}."


@dataclass(frozen=True)
class LevelResolution:
    """A resolved layer, and everything a reader needs to judge it.

    Two INDEPENDENT axes of wrongness are disclosed, because either can be
    perfect while the other is poor (finding 4):

    * *agreement* -- ``dominant_fraction``/``runner_up``/``margin``: how much of
      the analyzed region would have picked this same layer pixel by pixel.
    * *level error* -- ``resolved_level``/``level_error``: how far the layer we
      landed on actually sits from what was asked. Over New Jersey a 300 hPa
      request lands 40 hPa away at 83% agreement, and an 850 hPa request lands
      46 hPa away at 100%.

    Together they answer the question a researcher actually has: *would this map
    have looked different if it had been done properly?*
    """

    index: int
    #: What to hand the EXISTING selection seam as ``dimension_value``. That
    #: seam reads a coordinate-less dimension by position and a
    #: coordinate-bearing one by value (``_select_dim_positional`` vs
    #: ``_select_dim_nearest``), so this is the index for the first and the
    #: coordinate value for the second. Passing an index to a dimension that
    #: has a coordinate would select the level whose VALUE is 21, not layer 21.
    selector_value: float
    kind: str
    units: str
    requested: float
    resolved_level: float
    level_error: float
    #: ``level_error`` as a fraction of the distance to the midpoint between
    #: this layer and its neighbour on the requested side. Scale-free, so it
    #: reads the same on a product with 1 hPa layers and one with 200 hPa
    #: layers, and it answers the question the raw error cannot: *could this
    #: product have done better?* Near 1 means the request fell almost exactly
    #: between two layers. Disclosed rather than gated -- see the module note on
    #: why no threshold is defensible here.
    level_error_fraction_of_gap: float
    #: max-minus-min of the resolved layer's own level across the analyzed
    #: region. Dominance near 100% bounds this only RELATIVE to the layer gap:
    #: on a product with 100 hPa gaps a layer ranging 260-340 hPa still scores
    #: ~100%, and "270.9 hPa, 100% agreement" would hide it entirely.
    resolved_level_spread: float
    dominant_fraction: float
    runner_up: int | None
    runner_up_fraction: float
    margin: float
    n_pixels: int
    #: Fraction of the analyzed region's columns that had no usable vertical
    #: coordinate and therefore did not vote. Without it, "100% of 4 analyzed
    #: pixels" sits beside a map built from forty.
    excluded_fraction: float
    axis_variable: str


def resolve_level(narrowed, dim: str, level: str, dataset=None) -> LevelResolution:
    """Resolve ``level`` to a single index along ``dim`` on ``narrowed``.

    ``narrowed`` is the region-narrowed, PRE-aggregation array. That is not a
    convenience: the vertical axes ride the time dimension, so the temporal
    reduction destroys them, and by the time the existing selection seam runs
    there is nothing left to match a height against (T58 finding 1). The index
    this returns then goes through that unchanged selection path (D7).

    ``dataset`` is the opened Dataset the array came out of, so a product that
    publishes its vertical axis as an ordinary DATA VARIABLE rather than a CF
    auxiliary coordinate is still resolvable. Without it this saw only
    ``narrowed.coords`` and told a researcher "this product does not publish an
    altitude axis" about a product that plainly does -- and that
    ``plot_vertical_profile``, reading the same granule, charts happily.
    """
    from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

    axes = _axis_candidates(narrowed, dim, dataset)
    available = {kind: str(var.attrs.get("units", "")).strip() for kind, (_, var) in axes.items()}
    request = parse_level(level, available=available)
    if request.kind not in axes:
        raise _axis_not_published_error(request, available)
    axis_name, axis = axes[request.kind]
    axis_units = str(axis.attrs.get("units", "")).strip()
    if not axis_units:
        # The companion door to the kPa bug. Falling back to the REQUEST's units
        # here made the conversion an identity, so a Pa-valued axis declaring
        # only ``standard_name: air_pressure`` answered "500 hPa" with the
        # 470 Pa layer at 100% disclosed agreement. An axis that does not say
        # what scale it is on cannot be matched against without guessing it.
        raise _axis_units_unknown_error(request, axis_name)

    # Both sides converted into the AXIS's units before anything is compared.
    # Comparing raw is the failure ``_select_dim_nearest`` already refuses to
    # make silently: 50000 (Pa) against an axis running 0.17..896 (hPa) snaps to
    # the deepest layer and looks entirely plausible.
    requested = _convert(request.value, request.units, axis_units, request.kind)

    # THE ANALYZED REGION, not its bounding box. ``mask_data_by_geometry`` ends
    # in ``.where(mask)``, which masks the data and leaves auxiliary coordinates
    # untouched -- so the axis still spans the whole cropped box. Restricting it
    # to the cells where the science variable survived does both halves at once:
    # it drops out-of-region cells, and it drops in-region cells the map renders
    # as no-data, which have no business voting on what the map depicts.
    region = _in_region_mask(narrowed, axis, dim)
    axis = axis.where(region)

    regional = reduce_keeping_axes(axis, keep=(dim,), stat="mean")
    values = np.asarray(regional.values, dtype="float64")
    if not np.isfinite(values).any():
        # BEFORE nanargmin, which raises a bare ValueError on an all-NaN slice
        # -- and a ValueError is not an MCPToolError, so the tool layer's
        # handler misses it and the researcher gets a 500 instead of an answer.
        raise _no_usable_column_error(request)
    _refuse_if_outside_the_axis(requested, values, axis_units, request)
    index = int(np.nanargmin(np.abs(values - requested)))

    agreement = _per_pixel_agreement(axis, dim, requested)
    dominant = agreement.pop("dominant")
    _refuse_if_not_honestly_one_layer(request, requested, axis_units, index, dominant, agreement, values)
    coordinate = narrowed[dim] if dim in getattr(narrowed, "coords", ()) else None
    return LevelResolution(
        index=index,
        selector_value=_selector_value(coordinate, index, request),
        kind=request.kind,
        units=axis_units,
        requested=requested,
        resolved_level=float(values[index]),
        level_error=float(abs(values[index] - requested)),
        level_error_fraction_of_gap=_error_fraction_of_gap(values, index, requested),
        resolved_level_spread=_layer_spread(axis, dim, index),
        axis_variable=str(axis_name),
        **agreement,
    )


def _axis_candidates(narrowed, dim: str, dataset) -> dict:
    """``{kind: (name, DataArray)}`` for every physical vertical axis spanning
    ``dim``, from the narrowed array's coordinates first and the opened
    Dataset's data variables second.

    Coordinates win because that is where these actually live on a CF-compliant
    granule: attached to the science variable, already carried through the same
    narrowing, co-located pixel-for-pixel with nothing to re-align. A Dataset
    copy is on the full granule grid, so it is aligned to the narrowed array
    here rather than read as-is -- reading it straight would report an axis
    averaged over a continent as if it described the region.
    """
    from tta_backend.utils.geo_utils import vertical_axes_for_dim

    found = {kind: (name, narrowed[name]) for kind, name in vertical_axes_for_dim(narrowed, dim).items()}
    if dataset is None or getattr(dataset, "data_vars", None) is None:
        return found
    for kind, name in vertical_axes_for_dim(dataset, dim).items():
        if kind in found or name not in dataset.data_vars or name == narrowed.name:
            continue
        try:
            aligned = dataset[name].broadcast_like(narrowed).sel(
                {d: narrowed[d] for d in narrowed.dims if d in dataset[name].coords},
            )
        except Exception:  # noqa: BLE001 — an axis we cannot align is an axis we do not offer
            continue
        found[kind] = (name, aligned)
    return found


def _in_region_mask(narrowed, axis, dim: str):
    """Where the science variable actually has a value, broadcast over the
    axis. ``narrowed`` may carry dimensions the axis does not (or vice versa),
    so this aligns rather than assuming a shared shape."""
    present = np.isfinite(narrowed)
    if not set(axis.dims) - set(narrowed.dims):
        return present.broadcast_like(axis)
    # An axis with extra dimensions: a cell counts if the science variable has
    # a value anywhere along the dimensions they do not share.
    return present.any(dim=[d for d in narrowed.dims if d not in axis.dims]).broadcast_like(axis)


def _selector_value(coordinate, index: int, request):
    """What the existing selection seam should be handed for the resolved layer.

    A coordinate with duplicate or non-monotonic values cannot be selected by
    value at all -- xarray's ``.sel(method="nearest")`` raises ``InvalidIndexError``
    / ``ValueError`` there, neither of which is an ``MCPToolError``, so it would
    escape the tool layer's handler as a 500. Refuse with the reason instead.
    """
    if coordinate is None:
        return index
    values = np.asarray(coordinate.values).ravel()
    if len(np.unique(values)) != len(values):
        raise _unselectable_dimension_error(request, "has repeated coordinate values")
    diffs = np.diff(values)
    if len(diffs) and not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise _unselectable_dimension_error(request, "has non-monotonic coordinate values")
    return float(values[index])


def _error_fraction_of_gap(values, index: int, requested: float) -> float:
    """How far the request sits toward the midpoint between the resolved layer
    and its neighbour ON THE REQUESTED SIDE, as a fraction. 0 is exactly on the
    layer, 1 is exactly halfway to the next one.

    The neighbour has to be the one the request is heading toward: layer spacing
    is wildly asymmetric on a log-pressure grid, and taking the nearer of the
    two neighbours reports fractions above 1 for a request that is genuinely
    nearest the layer it picked.
    """
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < 2:
        return 0.0
    order = finite[np.argsort(values[finite])]
    position = int(np.flatnonzero(order == index)[0])
    step = 1 if requested > values[index] else -1
    neighbour = position + step
    if not (0 <= neighbour < order.size):
        neighbour = position - step  # edge layer: use the only neighbour there is
    if not (0 <= neighbour < order.size):
        return 0.0
    half_gap = abs(values[order[neighbour]] - values[index]) / 2.0
    if half_gap == 0:
        return 0.0
    return min(1.0, float(abs(values[index] - requested) / half_gap))


def _layer_spread(axis, dim: str, index: int) -> float:
    """max-minus-min of one layer's level across the analyzed region."""
    layer = axis.isel({dim: index})
    finite = np.asarray(layer.values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(finite.max() - finite.min())


# Conversion factors into each kind's canonical unit (hPa, km), keyed by the
# same lowercase spellings ``geo_utils`` recognizes -- one vocabulary, so a unit
# the parser accepts is always a unit the resolver can convert.
_TO_CANONICAL = {
    "pressure": {
        "hpa": 1.0, "hectopascal": 1.0, "hectopascals": 1.0,
        "mb": 1.0, "mbar": 1.0, "millibar": 1.0, "millibars": 1.0,
        "pa": 0.01, "pascal": 0.01, "pascals": 0.01,
        "kpa": 10.0, "kilopascal": 10.0, "kilopascals": 10.0,
        "atm": 1013.25,
    },
    "altitude": {
        "km": 1.0, "kilometer": 1.0, "kilometers": 1.0,
        "kilometre": 1.0, "kilometres": 1.0,
        "m": 0.001, "metre": 0.001, "metres": 0.001, "meter": 0.001, "meters": 0.001,
    },
}


def _convert(value: float, from_units: str, to_units: str, kind: str) -> float:
    """``value`` expressed in ``to_units``, or a refusal when the axis's scale
    is unknown.

    Refusing rather than assuming parity is the whole point.
    ``vertical_axis_kind`` classifies on ``standard_name`` FIRST and only falls
    back to units, so a product declaring ``standard_name: air_pressure`` is
    accepted as a pressure axis whatever its units say -- kPa, for instance.
    Comparing an hPa request to a kPa axis unconverted resolved 50 hPa to
    500 hPa and disclosed it as 100% unanimous: an ordinary-looking map of the
    wrong altitude, which is the exact failure this feature exists to
    eliminate. ``from_units`` is always convertible; the parser only accepts
    units in this same vocabulary.
    """
    table = _TO_CANONICAL[kind]
    source = table.get(from_units.strip().lower())
    target = table.get(to_units.strip().lower())
    if source is None or target is None:
        raise _unconvertible_axis_error(from_units, to_units, kind)
    return value * source / target


def _unconvertible_axis_error(from_units: str, to_units: str, kind: str):
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    known = ", ".join(sorted(_TO_CANONICAL[kind]))
    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"This product's {kind} axis is published in {to_units!r}, which this backend "
        f"cannot convert {from_units!r} into -- so a level cannot be matched against it "
        "without guessing the scale, and guessing it wrong returns an ordinary-looking "
        "map of the wrong altitude.",
        suggestion=(
            f"Select a layer directly with 'dimension'/'dimension_value'. Convertible "
            f"{kind} units are: {known}."
        ),
    )


def _refuse_if_outside_the_axis(requested: float, values, axis_units: str, request) -> None:
    """Refuse a level the product's axis does not span, instead of snapping to
    its top or bottom layer.

    Same contract ``_select_dim_nearest`` already keeps for coordinate-bearing
    dimensions, and for the same reason: an edge snap turns a units mismatch or
    a wrong-product request into an ordinary-looking map of the wrong altitude.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    # Tolerant at the endpoints. The bound is the cos(latitude)-WEIGHTED mean of
    # the axis, so even a column holding one value everywhere comes back a few
    # ulps off that value -- and asking for exactly the level the product
    # publishes at its top layer would be refused as out of range. Caught by the
    # round-trip check, which is what it is for. The tolerance is relative, so
    # it never widens the range enough to admit a real units mismatch (those are
    # off by factors of 100 or 1000).
    if _within(requested, low, high):
        return
    raise MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"{request.value:g} {request.units} is outside the vertical range this product "
        f"covers over the analyzed region ({low:.4g} to {high:.4g} {axis_units}) -- "
        "refusing to snap to the nearest edge layer, which would return an "
        "ordinary-looking map of a level that was never requested.",
        suggestion=f"Ask for a level between {low:.4g} and {high:.4g} {axis_units}.",
    )


def _within(value: float, low: float, high: float) -> bool:
    if low <= value <= high:
        return True
    return math.isclose(value, low, rel_tol=1e-9) or math.isclose(value, high, rel_tol=1e-9)


def _axis_units_unknown_error(request, axis_name: str):
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"This product's {request.kind} axis ({axis_name}) publishes no units, so "
        f"{request.value:g} {request.units} cannot be matched against it without "
        "guessing what scale its numbers are on -- and guessing wrong returns an "
        "ordinary-looking map of the wrong altitude.",
        suggestion="Select a layer directly with 'dimension'/'dimension_value'.",
    )


def _no_usable_column_error(request):
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"No pixel in the analyzed region has a usable vertical coordinate, so "
        f"{request.value:g} {request.units} cannot be resolved to a layer.",
        suggestion="Try a region or time window with fuller retrieval coverage.",
    )


def _unselectable_dimension_error(request, problem: str):
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"This product's vertical dimension {problem}, so the layer matching "
        f"{request.value:g} {request.units} cannot be selected by coordinate value.",
        suggestion="Select a layer directly with 'dimension'/'dimension_value' by index.",
    )


def _axis_not_published_error(request, available: dict):
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    return MCPToolError(
        CATEGORY_DIMENSION_CHOICE_REQUIRED,
        f"This product does not publish a {request.kind} axis, so "
        f"{request.value:g} {request.units} cannot be resolved against it.",
        suggestion=_vocabulary(available),
    )


def _per_pixel_agreement(axis, dim: str, requested: float) -> dict:
    """How much of the analyzed region would independently pick the same layer.

    Each pixel column resolves the request against its OWN vertical coordinate;
    the disclosure is the distribution of those answers. A column with any
    non-finite level is excluded rather than partially compared -- a partial
    column would silently make the layers it does have look more dominant.

    A "column" is one (timestep, cell), not one cell: the vertical grid moves
    between timesteps, the plotted map is a mean over all of them, and a single
    layer index is applied to every one. So each timestep's opinion counts.

    AREA-weighted, by the same ``cos_lat_weights`` the regional mean and the QA
    pass rate use. That is not a refinement -- it is the only way this fraction
    can describe the same field the regional-mean axis does. Counting cells
    instead would let half the CELLS be a quarter of the AREA and still report
    "50%", and the number sits next to a regional mean that weighted them
    differently.
    """
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    ordered = axis.transpose(dim, ...)
    columns = np.asarray(ordered.values, dtype="float64").reshape(ordered.sizes[dim], -1)
    # A column votes if it has ANY finite layer, not if it has EVERY one. The
    # all-or-nothing filter meant a single all-fill layer -- routine in profile
    # retrievals, where the top layer fails at high solar zenith angle and the
    # bottom sits below terrain -- refused every level request for the whole
    # product, reporting "no pixel has a complete vertical coordinate" when 23
    # of 24 layers were perfect. The regional mean never had this problem
    # (skipna), so the two halves of the refusal rule also disagreed about which
    # pixels they were describing.
    usable_mask = np.isfinite(columns).any(axis=0)
    usable = columns[:, usable_mask]
    n_region = int(columns.shape[1])
    if usable.shape[1] == 0:
        return {
            "dominant": None, "dominant_fraction": 0.0, "runner_up": None,
            "runner_up_fraction": 0.0, "margin": 0.0, "n_pixels": 0,
            "excluded_fraction": 1.0 if n_region else 0.0,
        }
    weights = _column_weights(ordered, dim, cos_lat_weights(ordered))[usable_mask]
    # nanargmin over each column's own finite layers: a column missing the top
    # layer still has an opinion about 500 hPa.
    distance = np.abs(usable - requested)
    distance[~np.isfinite(distance)] = np.inf
    choices = np.argmin(distance, axis=0)
    counts = np.bincount(choices, weights=weights, minlength=usable.shape[0])
    total = float(counts.sum())
    # Ties broken toward the lower index (``kind="stable"``), so a 50/50 split
    # names one layer reproducibly instead of by array-order accident -- and it
    # is refused below either way.
    # Descending by weight, ties broken toward the LOWER layer index. Sorting
    # the negated counts gets that directly; reversing a stable ascending sort
    # reverses the tie order too, which the previous comment here claimed it
    # did not. Only affects which layer a refusal names, but a comment that
    # says the opposite of the code is worse than no comment.
    order = np.argsort(-counts, kind="stable")
    runner_up = int(order[1]) if counts.size > 1 and counts[order[1]] else None
    runner_up_fraction = float(counts[order[1]] / total) if runner_up is not None else 0.0
    dominant_fraction = float(counts[order[0]] / total)
    return {
        "dominant": int(order[0]),
        "dominant_fraction": dominant_fraction,
        "runner_up": runner_up,
        "runner_up_fraction": runner_up_fraction,
        "margin": dominant_fraction - runner_up_fraction,
        # A COUNT, deliberately, even though the fractions are area-weighted:
        # it is the sample SIZE behind them, and a weighted "n" is not a number
        # anyone can interpret ("83% of 1,847.3 pixels"). It must therefore
        # never be rendered as the denominator of an area percentage -- "91.3%
        # of 40 analyzed pixels" is false when 40 equal-count cells span a 10:1
        # range of area. The disclosure says "of the analyzed area" and reports
        # the count separately as a sample size.
        "n_pixels": int(usable.shape[1]),
        "excluded_fraction": float(1.0 - usable.shape[1] / n_region) if n_region else 0.0,
    }


def _column_weights(ordered, dim: str, lat_weights) -> np.ndarray:
    """One weight per flattened non-vertical column, aligned with the same
    ``reshape(n_layers, -1)`` the values went through.

    Broadcast through xarray rather than by hand: the vertical axis's dim order
    varies by product, and a hand-rolled ``repeat``/``tile`` would silently
    align weights to the wrong dimension on the first product whose dims come
    back in another order.
    """
    template = ordered.isel({dim: 0}, drop=True)
    if lat_weights is None:
        return np.ones(int(np.prod(template.shape)) or 1, dtype="float64")
    broadcast = lat_weights.broadcast_like(template).transpose(*template.dims)
    return np.asarray(broadcast.values, dtype="float64").ravel()


# Below a MAJORITY, most of the analyzed region resolves to some layer other
# than the one returned, and labelling the result with the requested level is a
# claim about data it does not show. Derived, not chosen: the Phase 1 spike's
# lowest observed dominance on a real product and a modest AOI was 83.1%, so
# this refuses nothing measured to be sound, while a 51/49 coin flip -- which
# grilling named as the case that must not resolve -- falls below it. A 2/3
# threshold was proposed and rejected: nothing makes 66.7% safe and 66.6% not.
_MIN_DOMINANT_FRACTION = 0.5


def _refuse_if_not_honestly_one_layer(
    request, requested: float, axis_units: str, index, dominant, agreement, values,
) -> None:
    """Refuse when a single index cannot honestly stand for the request (D6)."""
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError

    fraction = agreement["dominant_fraction"]
    n_pixels = agreement["n_pixels"]
    if not n_pixels:
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"No pixel in the analyzed region has a complete vertical coordinate, "
            f"so {request.value:g} {request.units} cannot be resolved to a layer.",
            suggestion="Try a region or time window with fuller retrieval coverage.",
        )
    if fraction <= _MIN_DOMINANT_FRACTION:
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"{request.value:g} {request.units} does not resolve to one layer over this "
            f"region: the most common answer is layer {dominant}, but only "
            f"{fraction * 100:.1f}% of the {n_pixels:,} analyzed pixels agree with it "
            f"(runner-up layer {agreement['runner_up']} at "
            f"{agreement['runner_up_fraction'] * 100:.1f}%). A map labelled "
            f"'{request.value:g} {request.units}' would describe a level most of the "
            "region is not on.",
            suggestion=(
                "Narrow the region so its vertical grid is more uniform, or select a "
                "layer directly with the 'dimension'/'dimension_value' parameters."
            ),
        )
    runner_up = agreement["runner_up"]
    if dominant is not None and runner_up is not None and abs(dominant - runner_up) > 1:
        # Threshold-free, and it catches what the majority floor cannot: a
        # region containing two unrelated vertical structures. A 60/40 split
        # clears the floor comfortably, but if those two groups are fourteen
        # layers apart then no single index describes the region. On every
        # measurement the spike took, the runner-up was the NEIGHBOURING layer;
        # a distant one is evidence, not noise.
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"{request.value:g} {request.units} resolves to two layers that are far "
            f"apart over this region: {fraction * 100:.1f}% of the analyzed area is on "
            f"layer {dominant} and {agreement['runner_up_fraction'] * 100:.1f}% on layer "
            f"{runner_up}, {abs(dominant - runner_up)} layers apart. That is two "
            "different vertical structures in one region, and one layer index cannot "
            "stand for both.",
            suggestion=(
                "Narrow the region so its vertical grid is more uniform, or select a "
                "layer directly with the 'dimension'/'dimension_value' parameters."
            ),
        )
    if dominant is not None and dominant != index:
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"{request.value:g} {request.units} is ambiguous over this region: the "
            f"regional-mean vertical axis resolves it to layer {index} "
            f"({values[index]:.4g} {axis_units}), but {fraction * 100:.1f}% of the "
            f"{n_pixels:,} analyzed pixels resolve it to layer {dominant} instead. "
            "One index cannot stand for both.",
            suggestion=(
                "Narrow the region so its vertical grid is more uniform, or select a "
                "layer directly with the 'dimension'/'dimension_value' parameters."
            ),
        )


def _unparseable_level_error(raw: str, available: dict | None):
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"{raw!r} is not a physical level. A level must carry the units the "
        "atmosphere has, because the units are what say which vertical axis "
        "was meant -- a bare number does not.",
        suggestion=_vocabulary(available),
    )


class _UnitsOnly:
    """A units string in the shape ``vertical_axis_kind`` reads.

    The request's unit is classified by the SAME function that classifies a
    granule's axis (``geo_utils.vertical_axis_kind``) rather than by a second
    vocabulary. Two lists would drift, and the drift would be silent in the
    worst direction: a unit the request accepts but no axis is ever classified
    as would resolve against nothing, or -- worse -- against the wrong axis.
    """

    def __init__(self, units: str):
        self.attrs = {"units": units}
