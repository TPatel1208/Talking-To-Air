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
    dominant_fraction: float
    runner_up: int | None
    runner_up_fraction: float
    margin: float
    n_pixels: int
    axis_variable: str


def resolve_level(narrowed, dim: str, level: str) -> LevelResolution:
    """Resolve ``level`` to a single index along ``dim`` on ``narrowed``.

    ``narrowed`` is the region-narrowed, PRE-aggregation array. That is not a
    convenience: the vertical axes ride the time dimension, so the temporal
    reduction destroys them, and by the time the existing selection seam runs
    there is nothing left to match a height against (T58 finding 1). The index
    this returns then goes through that unchanged selection path (D7).
    """
    from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes
    from tta_backend.utils.geo_utils import vertical_axes_for_dim

    axes = vertical_axes_for_dim(narrowed, dim)
    available = {kind: str(narrowed[name].attrs.get("units", "")) for kind, name in axes.items()}
    request = parse_level(level, available=available)
    if request.kind not in axes:
        raise _axis_not_published_error(request, available)
    axis_name = axes[request.kind]
    axis = narrowed[axis_name]
    axis_units = str(axis.attrs.get("units", "")) or request.units

    # Both sides converted into the AXIS's units before anything is compared.
    # Comparing raw is the failure ``_select_dim_nearest`` already refuses to
    # make silently: 50000 (Pa) against an axis running 0.17..896 (hPa) snaps to
    # the deepest layer and looks entirely plausible.
    requested = _convert(request.value, request.units, axis_units, request.kind)

    regional = reduce_keeping_axes(axis, keep=(dim,), stat="mean")
    values = np.asarray(regional.values, dtype="float64")
    _refuse_if_outside_the_axis(requested, values, axis_units, request)
    index = int(np.nanargmin(np.abs(values - requested)))

    agreement = _per_pixel_agreement(axis, dim, requested)
    dominant = agreement.pop("dominant")
    _refuse_if_not_honestly_one_layer(request, requested, axis_units, index, dominant, agreement, values)
    coordinate = narrowed[dim] if dim in getattr(narrowed, "coords", ()) else None
    return LevelResolution(
        index=index,
        selector_value=float(np.asarray(coordinate.values).ravel()[index]) if coordinate is not None else index,
        kind=request.kind,
        units=axis_units,
        requested=requested,
        resolved_level=float(values[index]),
        level_error=float(abs(values[index] - requested)),
        axis_variable=str(axis_name),
        **agreement,
    )


# Conversion factors into each kind's canonical unit (hPa, km), keyed by the
# same lowercase spellings ``geo_utils`` recognizes -- one vocabulary, so a unit
# the parser accepts is always a unit the resolver can convert.
_TO_CANONICAL = {
    "pressure": {
        "hpa": 1.0, "mb": 1.0, "mbar": 1.0, "millibar": 1.0, "millibars": 1.0,
        "pa": 0.01, "pascal": 0.01, "atm": 1013.25,
    },
    "altitude": {
        "km": 1.0, "kilometer": 1.0, "kilometers": 1.0, "kilometre": 1.0,
        "m": 0.001, "metre": 0.001, "meter": 0.001, "meters": 0.001,
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
    complete = np.isfinite(columns).all(axis=0)
    usable = columns[:, complete]
    if usable.shape[1] == 0:
        return {
            "dominant": None, "dominant_fraction": 0.0, "runner_up": None,
            "runner_up_fraction": 0.0, "margin": 0.0, "n_pixels": 0,
        }
    weights = _column_weights(ordered, dim, cos_lat_weights(ordered))[complete]
    choices = np.argmin(np.abs(usable - requested), axis=0)
    counts = np.bincount(choices, weights=weights, minlength=usable.shape[0])
    total = float(counts.sum())
    # Ties broken toward the lower index (``kind="stable"``), so a 50/50 split
    # names one layer reproducibly instead of by array-order accident -- and it
    # is refused below either way.
    order = np.argsort(counts, kind="stable")[::-1]
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
        # it is the sample size behind them, and a weighted "n" is not a number
        # anyone can interpret ("83% of 1,847.3 pixels").
        "n_pixels": int(usable.shape[1]),
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
