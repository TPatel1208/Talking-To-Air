"""
preprocessing/frame_stack.py
============================
The bucketed reduction: one field per interval of the requested span, instead
of one field for the whole of it (T59 Phase 3).

``aggregate()`` collapses the time axis to a single 2-D field. This module
keeps that axis, grouped into cadence buckets, so a week of TEMPO NO2 can be
scrubbed rather than re-retrieved a day at a time.

Why this is not ``reduce_keeping_axes``
--------------------------------------
``reduce_keeping_axes`` reduces every dimension EXCEPT ``keep``. Asking it for
frames would mean ``keep=("time", "lat", "lon")``, which collapses nothing at
all: the two operations frames actually need -- group time into buckets, then
block-mean lat/lon -- are both additions to it rather than uses of it, and
teaching one function to sometimes-group-and-sometimes-not is how a shared
reduction acquires two behaviours behind one name.

It IS the right home for the per-frame REGIONAL statistics, and this module
calls it for exactly that (``keep=("bucket",)``). That is the reuse worth
having: the cos(latitude)-weighted-mean / unweighted-everything-else split
lives in one place, so a frame's mean cannot drift from the map's.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord

# D5's cell ceiling. A CEILING, not a target: ``k`` below is an integer, so a
# real grid lands at 70-96% of it (14,124 measured on a 535x658 regional grid,
# 19,107 on the full 2950x5771 TEMPO domain). ``FrameStack.cells_per_frame``
# carries what was realized, never this number.
FRAME_CELL_CEILING = 20_000

# D7's gate. A month of hourly granules is a 700-stop slider no human can use;
# above this the cadence buckets are grouped and the scrubber becomes D14's
# second tier.
MAX_FRAMES = 60

# D3's backstop, on SPATIAL EXTENT -- the term D7 does not bound. Phase 3 §7
# measured a production-shaped ``build_frame_stack`` being OOM-killed by the
# kernel on the full TEMPO domain in the 3.9 GB container, while Phase 2's bare
# ``groupby().mean().coarsen().mean()`` survives the same bundle at 1,363 MB.
# So the reduction's own consumers are what push it over, and a frame-count
# gate lets a full-domain request straight through to a reduction that dies.
#
# The number is the largest extent MEASURED to complete, rounded up, not a
# guess with a safety factor:
#
#   535 x 658   =   352 k cells, 54 hourly buckets ->   385 MB   completed
#   1250 x 3000 =  3.75 M cells, 54 hourly buckets -> 1,342 MB   completed
#   2950 x 5771 = 17.0 M cells, 54 hourly buckets -> OOM-KILLED
#
# Two cheaper-reduction levers exist and NEITHER is measured: rechunking the
# grouped intermediate spatially before the fused consumers (each chunk is a
# full 2950x5771x8 = 136 MB slab today), and unfusing the pooled histogram back
# into its own pass. Either could raise this constant -- with a number attached.
# Until one of them is measured, a full-domain scrub is refused out loud rather
# than attempted and killed.
MAX_FRAME_NATIVE_CELLS = 4_000_000

# D3's other backstop, on SPAN, sitting above D14's coarsening rather than
# instead of it. Coarsening is what fits a long span into the frame budget and
# Phase 3 §2 measured it as the common case (a 2.2-day TEMPO retrieval is
# already 54 hourly frames), but it has no ceiling of its own: it will fold a
# decade of hourly buckets into 60 stops and label each one a scrub position.
#
# The limit is expressed as buckets PER FRAME rather than as a duration,
# because that is the quantity that decides whether a stop is still an interval
# a reader asked for. At 24 an hourly scrub reaches 60 days with frames never
# coarser than a day, and a daily scrub reaches ~4 years with frames never
# coarser than 24 days.
MAX_BUCKETS_PER_FRAME = 24
MAX_SPAN_BUCKETS = MAX_FRAMES * MAX_BUCKETS_PER_FRAME

#: D14's two tiers, named so a payload consumer can branch on them.
#: ``cadence`` -- the frames ARE the cadence buckets and the period map is
#: derived from them exactly. ``coarsened`` -- the frames are a DIFFERENT
#: temporal aggregation, and ``FrameStack.delta`` quantifies the relationship.
CADENCE_TIER = "cadence"
COARSENED_TIER = "coarsened"

# D9's clip, and how many bins the pooled distribution is accumulated in. A
# quantile is not a streaming reduction, so an exact one over a lazily-opened
# multi-granule stack means either holding N x native in memory (the thing the
# Architectural Constraint forbids) or walking the graph twice. A histogram is
# the third option: bounded memory, one walk, and at 2^16 bins the quantization
# is ~1.5e-5 of the field's range -- some five orders of magnitude finer than
# TEMPO's own retrieval uncertainty, and far below anything a colour ramp can
# show.
POOLED_CLIP = (2.0, 98.0)
_POOLED_BINS = 1 << 16

# What the scrubber's colour scale actually is, in one self-describing line, so
# the number explains itself in the payload and in methods.md rather than only
# in JSX -- the same reason ``QA_PASS_RATE_BASIS`` exists.
POOLED_SCALE_BASIS = (
    "2nd-98th percentile pooled across every frame and the period mean, at native "
    "resolution before the block mean; the scrubber's scale only, the Map tab is "
    "untouched"
)

# Cadence -> the pandas offset a timestamp is floored to. The same bucketing
# ``_cadence_weighted_mean`` does, so a frame and the map agree about which
# granules belong to which interval.
_CADENCE_FLOOR = {"hourly": "h", "daily": "D"}
# Cadence -> the offset the FULL axis steps by. Distinct from the floor because
# a month is not a fixed duration: "MS" walks month starts, "D" would not.
_CADENCE_STEP = {"hourly": "h", "daily": "D", "monthly": "MS"}

BUCKETABLE_CADENCES = tuple(_CADENCE_STEP)

#: D6a decision 1: the statistics a stack can carry as a FIELD -- one value per
#: pixel per frame, ``groupby(bucket).max(time)`` block-maxed onto the frame
#: grid.
#:
#: **This is NOT ``_FRAME_STATS``, and the two must never be merged.** That
#: tuple, further down, is the per-frame REGIONAL SCALAR: one number for the
#: whole region, which ``Frame.statistics["max"]`` has carried since Phase 3 and
#: which already renders in the readout. This one is a PLANE. They share three
#: words and nothing else -- a chart can show the max plane while every frame's
#: scalar max is unchanged, and the scalar max of the max plane is not even the
#: same number as the scalar max of the mean plane. Naming them alike is how a
#: quantity acquires two accounts of itself, which is the failure
#: ``DELTA_BASIS``' own docstring exists to prevent, wearing a third shoe.
#:
#: ``"mean"`` is in this tuple because it is a legal request, not because it
#: lands in ``FrameStack.planes``: the mean plane is the one this module has
#: always shipped and it stays in the top-level fields (D6a decision 5 -- "the
#: mean entry keeps its exact current shape, URL and cost").
PLANE_STATISTICS = ("mean", "max", "min")

# Grid-neutral dim names a non-mean plane wears inside the fused dataset.
# Names rather than the real lat/lon dims because that is the whole point of
# ``_bare_grid``: no coordinate, nothing to align on, nothing to outer-join.
_PLANE_LAT_DIM = "plane_lat"
_PLANE_LON_DIM = "plane_lon"


@dataclass(frozen=True)
class FrameRefusal:
    """Why a request may not be framed.

    ``reason`` is machine-readable and ``detail`` is the sentence a reader
    gets. The two are separate because the caller decides which refusals are
    worth telling anyone about, and that policy is the TOOL's, not the
    reduction's: a single-granule map was never going to have a time axis to
    browse, while a week of full-domain TEMPO is a request that could have been
    framed and was not.
    """

    reason: str
    detail: str


def frame_gate(
    da: xr.DataArray,
    *,
    time_dim: str | None,
    cadence: str,
    span: tuple[str, str] | None = None,
    max_frames: int = MAX_FRAMES,
) -> FrameRefusal | None:
    """Whether ``da`` may be handed to :func:`build_frame_stack`. ``None`` = yes.

    The gate lives here rather than in the tool because every limit it enforces
    is a fact about the reduction, measured on the reduction. What the tool owns
    is what to SAY about a refusal.
    """
    # ``time_dim in da.coords`` is not belt-and-braces on top of ``in da.dims``:
    # a dimension can exist with no coordinate on it -- TEMPO's O3PROF ``layer``
    # does exactly that -- and without the check the axis is built from
    # positions rather than timestamps, which floors to 1970 and labels the
    # scrubber with intervals that never happened. ``_apply_quality_mask``
    # guards its own per-timestep counters the same way, for the same reason.
    if (
        time_dim is None
        or time_dim not in da.dims
        or time_dim not in da.coords
        or da.sizes[time_dim] < 2
    ):
        return FrameRefusal(
            "no_time_axis",
            "This map has no browsable time axis: it was built from a single "
            "granule, or its time dimension carries no timestamps.",
        )
    if cadence not in BUCKETABLE_CADENCES:
        return FrameRefusal(
            "cadence_unknown",
            f"This product's cadence is {cadence!r}, so a frame axis has no "
            "interval width to use; frames need one of "
            f"{', '.join(BUCKETABLE_CADENCES)}.",
        )

    # A frame is one 2-D field per interval, and the blob's ``(1 + N, ny, nx)``
    # layout says so. Anything else -- a vertical axis the caller has not
    # selected a layer from, above all -- would reduce to a stack that reshapes
    # cleanly and renders the wrong plane.
    spatial = [dim for dim in da.dims if dim != time_dim]
    try:
        grid = _spatial_dims(da, spatial)
    except ValueError as exc:
        return FrameRefusal("no_spatial_grid", str(exc))
    extra = [dim for dim in spatial if dim not in grid]
    if extra:
        return FrameRefusal(
            "extra_dimensions",
            "A frame axis needs one 2-D field per interval, and this one still "
            f"carries {', '.join(extra)}.",
        )

    # Measured on the NARROWED field, the one that actually reaches the
    # reduction -- the same place ``_profile_scale_guard`` takes its reading,
    # and for the same reason: narrowing is what makes the ordinary regional
    # case small.
    cells = _native_cells(da, time_dim)
    if cells > MAX_FRAME_NATIVE_CELLS:
        return FrameRefusal(
            "extent_too_large",
            f"A frame axis over this region would reduce {cells:,} cells per "
            f"interval, above the {MAX_FRAME_NATIVE_CELLS:,}-cell limit a frame "
            "stack is built within.",
        )

    n_buckets = len(_axis_starts(_span_of(da, time_dim, span), cadence))
    if n_buckets < 2:
        return FrameRefusal(
            "single_interval",
            f"Every granule here falls inside one {cadence} interval, which is "
            "one stop -- there is nothing to scrub between.",
        )
    if n_buckets > max_frames * MAX_BUCKETS_PER_FRAME:
        return FrameRefusal(
            "span_too_long",
            f"This span covers {n_buckets:,} {cadence} intervals, above the "
            f"{max_frames * MAX_BUCKETS_PER_FRAME:,} a frame axis is built "
            f"within -- past it each of the {max_frames} frames would average "
            f"more than {MAX_BUCKETS_PER_FRAME} intervals together.",
        )
    return None


def _native_cells(da: xr.DataArray, time_dim: str) -> int:
    """Cells in ONE interval's field: everything but the time axis.

    Per interval rather than in total, because that is the shape the grouped
    intermediate holds -- one native plane per bucket, and the plane is what
    the fused consumers each fan out from.
    """
    cells = 1
    for dim, size in da.sizes.items():
        if dim != time_dim:
            cells *= int(size)
    return cells


def _span_of(
    da: xr.DataArray, time_dim: str, span: tuple[str, str] | None,
) -> tuple[str, str]:
    """The interval the frame axis covers: what the caller asked for, or the
    extent of ``da``'s own time axis when it did not say.

    Shared by the gate and the reduction on purpose. A gate that counted
    buckets over a different span than the one the stack is built on would
    admit exactly the requests it exists to stop.
    """
    if span is not None:
        return span
    stamps = pd.to_datetime(np.asarray(da[time_dim].values))
    return stamps.min().isoformat(), stamps.max().isoformat()


@dataclass(frozen=True)
class Frame:
    """One stop on the scrubber, and what it has to disclose about itself.

    A 90%-masked bucket renders as a clean low uniform field, indistinguishable
    from a calm day to someone scrubbing for an event, so the interval's own
    coverage travels with it rather than being inferable from the picture
    (D10).
    """

    t_start: str
    t_end: str
    n_granules: int
    #: cos(latitude)-weighted fraction of the ANALYZED REGION that held a value
    #: in this interval, measured on the native field. Never derived from
    #: ``FrameStack.values`` -- a block is finite if any cell in it is, so the
    #: frame grid reports a sparse hour as almost fully covered (D10).
    valid_fraction: float
    #: This interval's QA pass rate, rolled up by AREA SUMS across its
    #: timesteps (finding 12). ``None`` when nothing in the interval was
    #: checkable, or when QA masking never ran -- absent, so that "QA didn't
    #: run" stays distinguishable from a real 0%.
    qa_pass_rate: float | None
    #: ``count``/``mean``/``min``/``max`` for this interval, reduced from the
    #: NATIVE field (D5a). ``{"count": 0}`` alone when nothing survived --
    #: absent, not a maximum of zero. Unrounded: formatting is the payload's
    #: business, and rounding here would make the D14 tier-1 identity between
    #: the frames and the period map uncheckable.
    statistics: dict


@dataclass(frozen=True)
class StatisticPlane:
    """One additional statistic's own pair of rendered planes (D6a).

    The mean is NOT one of these. It lives in ``FrameStack``'s own top-level
    fields where it always has, so a caller that never asks for a second
    statistic sees a stack byte-identical to the one this module shipped before
    max existed -- decision 5's "the mean entry keeps its exact current shape,
    URL and cost", read at the Python level and not only at the wire level.

    Everything here is the same KIND of quantity as its top-level counterpart
    and none of it is the same NUMBER: a block max is not a block mean, and the
    agreement figure beside it answers a different question with a different
    across-frame reduction (see ``_frame_grid_delta``'s ``combine``).
    """

    values: np.ndarray
    period_values: np.ndarray
    #: The same quantity ``FrameStack.frame_grid_delta`` carries, reduced
    #: across frames by THIS plane's own statistic (see ``_frame_grid_delta``'s
    #: ``combine``). D6a decision 8 expects an exact ``0.0`` here on every
    #: selection plane in both tiers, and Phase 11 G5 measured exactly that on
    #: two live bundles. It is computed rather than asserted because a promised
    #: zero and a measured one are different claims -- the same reason
    #: ``frame_grid_delta`` exists at all.
    frame_grid_delta: dict
    #: THIS plane's own pooled 2-98 clip (D9), never the mean plane's: one
    #: colour has to mean one value at every stop of a scrub, and a max plane
    #: rendered against the mean's clip saturates at exactly the stops someone
    #: switched to it to see.
    value_range: tuple[float, float] | None
    #: D6a decision 9, on the ``"max"`` plane only: how much ground a rendered
    #: peak claims (see ``_extent_overstatement``). ``None`` on every other
    #: plane, and on a chart that never coarsened.
    extent_overstatement: dict | None = None


@dataclass(frozen=True)
class FrameStack:
    """The frame axis and the float32 array the frontend renders.

    ``values`` is a RENDERING artifact and nothing is derived from it (D5a).
    Every number a reader is shown comes off the native field before the block
    mean, because at k=8 a frame has already lost 16% of its own p98 and 70%
    of its max.
    """

    frames: list[Frame]
    values: np.ndarray
    #: The period mean on the SAME frame grid by the SAME method -- "frame 0"
    #: (D5). Parking the scrubber here reproduces the Map tab exactly, and in
    #: the cadence tier a reader can check D4's identity by averaging the stack.
    period_values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    coarsen_k: tuple[int, int]
    #: The REALIZED cells per frame, never the ceiling that was asked for.
    cells_per_frame: int
    cadence: str
    #: How many cadence buckets one frame holds. 1 in the cadence tier.
    buckets_per_frame: int
    tier: str
    #: D16, present only in the coarsened tier: ``headline`` (the weighted
    #: relative disagreement between the frames' own period mean and the map's)
    #: and ``max_abs`` (the worst pixel, in the field's physical units).
    #: ``None`` in the cadence tier -- the relationship there is identity, and
    #: saying so is different from measuring it and finding nothing.
    delta: dict | None
    #: The same metric applied to ``values`` and ``period_values`` themselves:
    #: what a reader gets by averaging the blob against the plane beside it.
    #: Present in BOTH tiers, because the block mean disagrees with the
    #: across-frame mean under partial coverage whatever the temporal tier is
    #: -- 1.876% on a real regional TEMPO chart whose native delta is
    #: 0.000002%. Never interchangeable with ``delta``; neither bounds the
    #: other.
    frame_grid_delta: dict
    #: The scrubber's pooled 2-98 clip (D9), or ``None`` when nothing survived.
    #: The Map tab's own range is untouched: deriving it from the stack would
    #: make a map's colours depend on whether frames happened to be built.
    value_range: tuple[float, float] | None
    #: D6a's additional planes, keyed by statistic -- ``"max"`` and ``"min"``,
    #: never ``"mean"``. Empty unless the caller asked for them, which is what
    #: keeps every existing caller's stack unchanged rather than merely
    #: compatible.
    planes: dict[str, StatisticPlane] = dataclass_field(default_factory=dict)


def coarsen_factors(n_lat: int, n_lon: int, target_cells: int) -> tuple[int, int]:
    """Block size landing at or under ``target_cells`` output cells.

    One shared factor per axis derived from the area ratio, so blocks stay
    square-ish in index space and a plume is not smeared preferentially along
    one of them.
    """
    total = n_lat * n_lon
    if total <= target_cells:
        return 1, 1
    k = max(1, int(np.ceil((total / target_cells) ** 0.5)))
    return k, k


def _bucket_starts(stamps: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    """Each timestamp floored to the start of its cadence bucket."""
    if cadence == "monthly":
        return stamps.to_period("M").to_timestamp()
    return stamps.floor(_CADENCE_FLOOR[cadence])


def _axis_starts(span: tuple[str, str], cadence: str) -> pd.DatetimeIndex:
    """Every bucket start in the REQUESTED span, whether or not a granule
    landed in it.

    This is finding 3, and it is the whole reason the axis is built from the
    request rather than from the data: ``_valid_time_indices`` drops timesteps
    that did not survive masking before the reduction runs, so a ``groupby``
    over the survivors emits only buckets that still hold a granule. An hour
    the mask emptied, and an hour nothing was retrieved for, would both
    disappear -- and the scrubber would silently renumber every stop after
    them, which is worse than showing nothing, because it looks right.
    """
    start, end = (pd.Timestamp(value) for value in span)
    first = _bucket_starts(pd.DatetimeIndex([start]), cadence)[0]
    return pd.date_range(first, pd.Timestamp(end), freq=_CADENCE_STEP[cadence])


def _bucket_end(start: pd.Timestamp, cadence: str) -> pd.Timestamp:
    """The exclusive end of the bucket beginning at ``start``."""
    return start + pd.tseries.frequencies.to_offset(_CADENCE_STEP[cadence])


def build_frame_stack(
    da: xr.DataArray,
    *,
    time_dim: str,
    cadence: str,
    span: tuple[str, str] | None = None,
    target_cells: int = FRAME_CELL_CEILING,
    max_frames: int = MAX_FRAMES,
    region_area: float | None = None,
    qa_counts: dict | None = None,
    value_bracket: tuple[float, float] | None = None,
    statistics: tuple[str, ...] = ("mean",),
) -> FrameStack:
    """Reduce ``da`` to one field per interval of the requested span.

    ``span`` is the interval the user asked for. It defaults to the extent of
    ``da``'s own time axis, which is right for a caller that has not narrowed
    the request -- but a caller that HAS should pass it, because a bucket at
    either end that no granule reached is a fact about the retrieval and
    belongs on the axis.

    The reduction is composed as ONE lazy graph --
    ``groupby(bucket).mean().coarsen(k, boundary="pad").mean()`` -- and computed
    once. That composition is not a stylistic preference: measured on a
    36-granule full-domain TEMPO open (2950x5771 = 17M cells) it stays lazy at
    32,042 tasks and materializes (32, 99, 193) = 4.9 MB, against the 2,179 MB
    the grouped-then-coarsened intermediate would need. Do not rewrite it as a
    chunked or hand-rolled reduction; that adds real complexity to solve a
    measured non-problem.

    ``statistics`` names the PLANES to build (D6a). It defaults to the mean
    alone, which is this module's whole history: the mean's planes stay in the
    top-level fields under their own names, and anything else asked for lands
    in ``FrameStack.planes``. Extra statistics join the same fused graph rather
    than getting one each -- see ``_plane_terms``.
    """
    # An unknown or unbucketable cadence cannot define a period, and it is not
    # a corner case: a Harmony bundle member carries no ``short_name``, so an
    # unregistered product resolves to "unknown" -- the same condition under
    # which ``_cadence_weighted_mean`` degrades to a plain unweighted mean.
    # Inventing a bucket width here would label a scrubber with intervals the
    # product never published. The caller's gate turns this into a refusal.
    if cadence not in BUCKETABLE_CADENCES:
        raise ValueError(
            f"Cannot build a frame axis at cadence {cadence!r}; frames need one of "
            f"{', '.join(BUCKETABLE_CADENCES)}."
        )
    unknown = [name for name in statistics if name not in PLANE_STATISTICS]
    if unknown:
        raise ValueError(
            f"Cannot build a frame plane for {', '.join(unknown)}; planes are one "
            f"of {', '.join(PLANE_STATISTICS)}."
        )

    stamps = pd.to_datetime(np.asarray(da[time_dim].values))
    axis = _axis_starts(_span_of(da, time_dim, span), cadence)
    axis_labels = [start.isoformat() for start in axis]

    grouper = xr.DataArray(
        np.array([start.isoformat() for start in _bucket_starts(stamps, cadence)]),
        dims=[time_dim],
        coords={time_dim: da[time_dim]},
    ).rename("bucket")

    spatial = [dim for dim in da.dims if dim != time_dim]
    lat_dim, lon_dim = _spatial_dims(da, spatial)
    # "Contributed a value somewhere", the same boolean ``_valid_time_indices``
    # keeps a timestep on: a granule the mask emptied is not a granule this
    # frame was built from, and reporting it as one would make an empty frame
    # look like a measured absence of pollution.
    contributed = np.isfinite(da).any(spatial)

    # Per CADENCE bucket, on the full requested axis: the reindex is what puts
    # the emptied and the never-retrieved intervals back, as NaN.
    buckets = da.groupby(grouper).mean(dim=time_dim, skipna=True).reindex(
        bucket=axis_labels,
    )
    bucket_granules = contributed.groupby(grouper).sum(dim=time_dim).reindex(
        bucket=axis_labels,
    ).fillna(0.0)

    # The period map, DERIVED from the buckets (D4). Each cadence bucket counts
    # once, whatever survived in it -- which is what ``_cadence_weighted_mean``
    # now computes too, so frame 0 and the Map tab cannot disagree.
    period_native = buckets.mean(dim="bucket", skipna=True)

    group = _frame_group_size(len(axis), max_frames)
    frames_native, frame_granules = _group_buckets(buckets, bucket_granules, group)

    k_lat, k_lon = coarsen_factors(da.sizes[lat_dim], da.sizes[lon_dim], target_cells)
    coarse = _block_reduce(frames_native, lat_dim, lon_dim, k_lat, k_lon)
    period_coarse = _block_reduce(period_native, lat_dim, lon_dim, k_lat, k_lon)
    edges = _pooled_bin_edges(da, value_bracket, qa_counts)
    extra_planes = tuple(name for name in statistics if name != "mean")

    # ONE compute for the whole stack: the frames the user renders, and every
    # number the user is shown, walk the same graph once. Every additional
    # plane joins THIS dataset for the same reason -- a statistic with its own
    # compute is a second I/O pass over a lazily-opened bundle.
    computed = xr.Dataset({
        "values": coarse,
        "period_values": period_coarse,
        "n_granules": frame_granules,
        "observed_area": _observed_area(frames_native, (lat_dim, lon_dim)),
        "pooled_histogram": (
            _histogram(frames_native, edges) + _histogram(period_native, edges)
        ),
        **_native_statistics(frames_native, (lat_dim, lon_dim)),
        **_delta_terms(frames_native, period_native, group),
        **_plane_terms(
            da, grouper=grouper, time_dim=time_dim, axis_labels=axis_labels,
            group=group, lat_dim=lat_dim, lon_dim=lon_dim,
            k_lat=k_lat, k_lon=k_lon, edges=edges, statistics=extra_planes,
        ),
    }).compute()

    n_frames = frames_native.sizes["bucket"]
    intervals = _frame_intervals(axis, cadence, group, n_frames)
    counts = np.nan_to_num(np.asarray(computed["n_granules"].values), nan=0.0)
    values = computed["values"].transpose("bucket", lat_dim, lon_dim)
    statistics = _statistics_per_frame(computed, n_frames)
    coverage = _valid_fractions(computed, da, (lat_dim, lon_dim), region_area)
    qa_rates = _qa_pass_rates(qa_counts, axis_labels, cadence, group, n_frames)

    # The one place float64 becomes float32, deliberately and at the edge: the
    # whole pipeline above is float64, and this halves what the blob stores
    # while staying ~5 orders of magnitude finer than TEMPO's own retrieval
    # uncertainty. Narrowed HERE rather than in the constructor so the
    # shipped-array delta below can be measured on the bytes that actually
    # ship rather than on their float64 ancestors.
    shipped = np.asarray(values.values, dtype="float32")
    shipped_period = np.asarray(
        computed["period_values"].transpose(lat_dim, lon_dim).values, dtype="float32",
    )
    # The ONE weight definition, the same one ``_delta_terms`` uses natively --
    # a second cos(lat) here would make the two deltas incomparable by
    # construction. Imported locally for the reason every other call site in
    # this module does: ``aggregation_service`` reaches back this way.
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    frame_weights = cos_lat_weights(values)
    weights = None if frame_weights is None else np.asarray(frame_weights.values)

    return FrameStack(
        frames=[
            Frame(
                t_start=start.isoformat(),
                t_end=end.isoformat(),
                n_granules=int(count),
                valid_fraction=fraction,
                qa_pass_rate=qa_rate,
                statistics=stats,
            )
            for (start, end), count, fraction, qa_rate, stats in zip(
                intervals, counts, coverage, qa_rates, statistics,
            )
        ],
        values=shipped,
        period_values=shipped_period,
        lats=np.asarray(values[lat_dim].values),
        lons=np.asarray(values[lon_dim].values),
        coarsen_k=(k_lat, k_lon),
        cells_per_frame=int(values.sizes[lat_dim] * values.sizes[lon_dim]),
        cadence=cadence,
        buckets_per_frame=group,
        tier=CADENCE_TIER if group == 1 else COARSENED_TIER,
        delta=_delta(computed, group),
        frame_grid_delta=_frame_grid_delta(shipped, shipped_period, weights),
        value_range=_pooled_range(computed["pooled_histogram"].values, edges),
        planes=_planes(
            computed, extra_planes, weights, (k_lat, k_lon), edges,
        ),
    )


def _planes(
    computed: xr.Dataset,
    statistics: tuple[str, ...],
    weights: np.ndarray | None,
    k: tuple[int, int],
    edges: np.ndarray,
) -> dict[str, StatisticPlane]:
    """The computed non-mean planes, narrowed to the bytes that ship.

    float32 at the same edge and for the same reason the mean's planes are
    narrowed there -- except that for a selection statistic the narrowing costs
    nothing even in principle: ``max`` picks a value some native cell held
    rather than computing one, and rounding a selection to float32 rounds the
    same number the mean plane's own cells were rounded from. Where the mean
    carries Phase 3's 0.000002% storage noise, this carries none.
    """
    planes: dict[str, StatisticPlane] = {}
    for name in statistics:
        values = computed[f"plane_{name}_values"].transpose(
            "bucket", _PLANE_LAT_DIM, _PLANE_LON_DIM,
        )
        period = computed[f"plane_{name}_period_values"].transpose(
            _PLANE_LAT_DIM, _PLANE_LON_DIM,
        )
        shipped = np.asarray(values.values, dtype="float32")
        shipped_period = np.asarray(period.values, dtype="float32")
        planes[name] = StatisticPlane(
            values=shipped,
            period_values=shipped_period,
            frame_grid_delta=_frame_grid_delta(
                shipped, shipped_period, weights, combine=name,
            ),
            value_range=_pooled_range(
                computed[f"plane_{name}_histogram"].values, edges,
            ),
            extent_overstatement=(
                _extent_overstatement(computed, k) if name == "max" else None
            ),
        )
    return planes


def _extent_overstatement(computed: xr.Dataset, k: tuple[int, int]) -> dict | None:
    """D6a decision 9's disclosure: how much ground a rendered peak claims.

    ``headline`` pools every finite block of every frame into ONE ratio of
    weighted sums; ``worst_frame`` is the single frame that stretched a peak
    furthest, disclosed beside the pooled figure rather than folded into it,
    the same way ``delta`` carries a headline and a worst pixel. ``ceiling`` is
    k^2 -- the value the pooled figure would take if exactly one native cell
    reached each block's max and no block ran short of real cells, which Phase
    11 measured as very nearly the case on both live bundles (24.70x and 24.75x
    against a ceiling of 25).

    It gates nothing, for the reason ``_delta`` gates nothing: whether painting
    a peak across 25 cells' worth of ground invalidates a particular reading is
    the reader's judgment, and the honest thing is to hand them the number.
    """
    if "overstatement_painted_area" not in computed:
        return None
    painted = np.atleast_1d(
        np.asarray(computed["overstatement_painted_area"].values, dtype="float64"),
    )
    peak = np.atleast_1d(
        np.asarray(computed["overstatement_peak_area"].values, dtype="float64"),
    )
    measurable = np.isfinite(painted) & np.isfinite(peak) & (peak > 0)
    if not measurable.any():
        return None
    per_frame = painted[measurable] / peak[measurable]
    return {
        "headline": float(painted[measurable].sum() / peak[measurable].sum()),
        "worst_frame": float(per_frame.max()),
        "ceiling": int(k[0] * k[1]),
        "basis": EXTENT_OVERSTATEMENT_BASIS,
    }


# The third basis string in this module, and the only one that is not about
# values agreeing. ``DELTA_BASIS`` and ``FRAME_GRID_DELTA_BASIS`` both ask
# whether two ways of reducing the same field give the same number; this asks
# how much GROUND a rendered pixel claims for a value one cell of it held. A
# reader handed the wrong one of the three has been told something false about
# a real measurement, which is why there are three of them and not one.
EXTENT_OVERSTATEMENT_BASIS = (
    "cos(latitude)-weighted area every finite block of the stored max planes paints, "
    "over the same weighted area of the native cells that actually held each block's "
    "own maximum, pooled across every frame at native resolution"
)


def _bracket_from_counts(qa_counts: dict | None) -> tuple[float, float] | None:
    """The value range ``apply_quality_mask`` already measured, if it ran.

    Read straight off the counters rather than asked for separately: the real
    plot path hands this module ``MaskedField.counts`` and nothing else, and a
    range it had to remember to pass would be forgotten once and then double
    the I/O of every scrubbed plot with nothing saying so.
    """
    if not qa_counts:
        return None
    low, high = qa_counts.get("value_min"), qa_counts.get("value_max")
    if low is None or high is None:
        return None
    return float(low), float(high)


def _pooled_bin_edges(
    da: xr.DataArray,
    value_bracket: tuple[float, float] | None,
    qa_counts: dict | None,
) -> np.ndarray:
    """Bin edges spanning every value the pooled distribution can hold.

    A bucket mean is an average of ``da``'s own values, so ``da``'s range
    brackets the whole stack AND the period map. ``value_bracket`` lets a
    caller hand over a range some earlier reduction already paid for --
    ``apply_quality_mask``'s counters carry one, measured on the same graph
    walk it was doing anyway. The shortcut is exactly the shape
    ``_fused_valid_flags`` uses, and for the same reason: re-deriving it here
    is a second full I/O pass over a lazily-opened bundle for a number that has
    already been bought.

    A bracket that turns out to be too narrow saturates rather than silently
    drops cells out of the pool (see ``_histogram``), so a caller cannot make
    the scale describe a subset of the data by passing a bad range -- only a
    coarser one.
    """
    if value_bracket is None:
        value_bracket = _bracket_from_counts(qa_counts)
    if value_bracket is not None:
        low, high = (float(value) for value in value_bracket)
    else:
        extremes = xr.Dataset(
            {"low": da.min(skipna=True), "high": da.max(skipna=True)},
        ).compute()
        low, high = float(extremes["low"]), float(extremes["high"])
    if not (np.isfinite(low) and np.isfinite(high)):
        return np.array([0.0, 1.0])
    if high <= low:
        # A constant field still has a percentile; give it a bin to land in.
        span = abs(low) if low else 1.0
        low, high = low - span * 0.5, high + span * 0.5
    return np.linspace(low, high, _POOLED_BINS + 1)


def _histogram(field: xr.DataArray, edges: np.ndarray) -> xr.DataArray:
    """The pooled distribution of ``field``, accumulated lazily.

    Clipped into the outer bins first: a value outside the edges would fall
    through every bin and leave the pool entirely, which would make the clip
    describe a subset of the data rather than all of it. Saturating is the
    honest failure here -- it can only coarsen the tails, never hide them.
    NaN compares false against every edge and drops out, which is what absent
    means.
    """
    clipped = field.clip(float(edges[0]), float(edges[-1]))
    if clipped.chunks is not None:
        import dask.array as dask_array

        counts, _ = dask_array.histogram(clipped.data, bins=edges)
    else:
        counts, _ = np.histogram(np.asarray(clipped.values), bins=edges)
    return xr.DataArray(counts, dims=["pooled_bin"])


def _pooled_range(
    histogram: np.ndarray, edges: np.ndarray,
) -> tuple[float, float] | None:
    """D9's 2-98 clip, read off the pooled histogram's CDF.

    NOT ``min(vmins)/max(vmaxs)`` over per-frame clips, which is what the
    frontend's ``computeSharedColorScale`` does for comparison panels: that
    union is a union of tails, and over 60 frames it is whatever the noisiest
    bucket did. Pooling is the whole point -- one colour has to mean one value
    at every stop of the scrub.
    """
    counts = np.asarray(histogram, dtype="float64")
    total = counts.sum()
    if total <= 0:
        return None
    cumulative = np.cumsum(counts) / total
    bounds = []
    for percentile in POOLED_CLIP:
        target = percentile / 100.0
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, counts.size - 1)
        below = cumulative[index - 1] if index else 0.0
        share = counts[index] / total
        # Linear within the bin the CDF crosses, so a wide bin does not quantize
        # the answer to its own edge.
        offset = (target - below) / share if share > 0 else 0.0
        bounds.append(
            float(edges[index] + min(max(offset, 0.0), 1.0) * (edges[index + 1] - edges[index]))
        )
    return bounds[0], bounds[1]


def _frame_group_size(n_buckets: int, max_frames: int) -> int:
    """How many cadence buckets go into one frame.

    1 wherever the native cadence fits the budget -- D14's tier one, where the
    frames ARE the cadence buckets and the period map is derived from them
    exactly. Above it the frames become a different temporal aggregation, and
    ``_delta`` is what keeps that honest rather than silent.
    """
    if max_frames < 1 or n_buckets <= max_frames:
        return 1
    return int(np.ceil(n_buckets / max_frames))


def _group_buckets(
    buckets: xr.DataArray, bucket_granules: xr.DataArray, group: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Fold ``group`` consecutive cadence buckets into one frame.

    A frame is the mean of its cadence buckets' means, NOT the mean of their
    granules: the tier that exists to fit long spans would otherwise
    reintroduce granule weighting exactly where sampling is most uneven, which
    is the bias D4 was fixed to remove.

    The field's own reducer is the mean and the granule counts' is the sum,
    which is the one thing about this fold that does NOT follow the plane's
    statistic: a max plane's frame still holds however many granules its
    cadence buckets held.
    """
    return (
        _grouped_by(buckets, group, "mean"),
        _grouped_by(bucket_granules, group, "sum"),
    )


def _grouped_by(buckets: xr.DataArray, group: int, how: str) -> xr.DataArray:
    """Fold ``group`` consecutive cadence buckets into one frame, under ``how``.

    ``boundary="pad"``, so a span that does not divide evenly keeps its last,
    shorter frame instead of dropping it. The bucket labels are dropped first:
    they are strings, and a coarsen would try to reduce them.
    """
    if group == 1:
        return buckets
    labelless = buckets.drop_vars("bucket")
    coarsened = labelless.coarsen(bucket=group, boundary="pad")
    return getattr(coarsened, how)(skipna=True)


def _plane_terms(
    da: xr.DataArray,
    *,
    grouper: xr.DataArray,
    time_dim: str,
    axis_labels: list[str],
    group: int,
    lat_dim: str,
    lon_dim: str,
    k_lat: int,
    k_lon: int,
    edges: np.ndarray,
    statistics: tuple[str, ...],
) -> dict[str, xr.DataArray]:
    """The lazy terms for every plane beyond the mean, for the SAME dataset.

    Not their own ``.compute()``, and the read-counter test is what holds that
    open: dask fuses all three groupby-then-coarsen reductions off one read of
    each source chunk exactly as it already fuses ``period_values``,
    ``n_granules``, ``observed_area`` and ``_delta_terms``, so a second
    statistic is arithmetic on chunks that were being read anyway. Phase 11's
    G1 measured it both ways -- an exact 6-read parity on synthetic loads, and
    1.5-1.9x (never 3x) wall-clock and peak RSS on two live bundles.

    Every block-reduced term goes through ``_bare_grid`` before it lands in the
    dataset, for the reason that function's docstring gives at length: three
    differently-reduced arrays under one coordinate name is a silent outer
    join, not an error.
    """
    terms: dict[str, xr.DataArray] = {}
    for name in statistics:
        buckets = getattr(da.groupby(grouper), name)(
            dim=time_dim, skipna=True,
        ).reindex(bucket=axis_labels)
        # The period plane is DERIVED from the cadence buckets, exactly as the
        # mean's is (D4): each bucket counts once. For a selection statistic
        # that is a distinction without a difference -- the max of the bucket
        # maxima is the max of everything -- but deriving it the same way keeps
        # one rule rather than two, and keeps stop 0 reproducible from the
        # stack a reader downloaded (D6a decision 3).
        period_native = getattr(buckets, name)(dim="bucket", skipna=True)
        frames_native = _grouped_by(buckets, group, name)
        terms[f"plane_{name}_values"] = _bare_grid(
            _block_reduce(frames_native, lat_dim, lon_dim, k_lat, k_lon, name),
            lat_dim, lon_dim,
        )
        terms[f"plane_{name}_period_values"] = _bare_grid(
            _block_reduce(period_native, lat_dim, lon_dim, k_lat, k_lon, name),
            lat_dim, lon_dim,
        )
        # Its OWN pooled 2-98 clip (D9), over its own frames and its own period
        # plane. A max plane scrubbed against the mean plane's clip would
        # saturate at every stop with a peak in it, which is every stop anyone
        # switched to it to look at. Native resolution, before the block
        # reduction, exactly as the mean's is (D5a and ``POOLED_SCALE_BASIS``).
        terms[f"plane_{name}_histogram"] = (
            _histogram(frames_native, edges) + _histogram(period_native, edges)
        )
        if name == "max":
            terms.update(_overstatement_terms(
                frames_native, lat_dim, lon_dim, k_lat, k_lon,
            ))
    return terms


def _overstatement_terms(
    frames_native: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    k_lat: int,
    k_lon: int,
) -> dict[str, xr.DataArray]:
    """D6a decision 9's two lazy sums, per frame, at NATIVE resolution.

    A block max paints one cell's value across every cell its block covers, so
    the rendered peak claims k^2 cells' worth of ground for an observation that
    holds at one. The numerator is the area the block PAINTS -- every real
    native cell it covers, observed or not, because that is the ground the
    pixel colours -- and the denominator is the area that actually held the
    block's own maximum. Both cos(latitude)-weighted, both summed before
    anything is divided (D16: a ratio of weighted sums, never a mean of
    per-block ratios, never signed).

    ``coarsen(...).construct(...)`` rather than a reduction, because the
    numerator needs each native cell compared against ITS OWN block's max --
    the block max has to come back down to native resolution to be compared,
    and construct is the reshape that makes both live in one lazy expression.
    It is a view, not a materialization: the sums collapse it straight back
    down, so this rides the graph walk the frames were already paying for --
    which is what Phase 11 G4 means by measuring the overstatement "in the same
    walk that already materializes the native max field", and what the
    read-counter test holds open.

    Nothing is emitted at k=(1,1): every rendered pixel IS a native cell there,
    and a ``1.0x`` would read as a measurement of an extent that was never
    stretched.
    """
    if (k_lat, k_lon) == (1, 1):
        return {}
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    window = {
        lat_dim: ("plane_block_lat", "plane_cell_lat"),
        lon_dim: ("plane_block_lon", "plane_cell_lon"),
    }
    blocks = {lat_dim: k_lat, lon_dim: k_lon}
    fine = ["plane_block_lat", "plane_cell_lat", "plane_block_lon", "plane_cell_lon"]

    # The area of every REAL cell of the grid: ones over the spatial footprint,
    # weighted by cos(latitude). The pad ``construct`` adds is NaN here as well
    # as in the data, which is exactly how a padded cell is told apart from a
    # real cell that simply holds no observation.
    footprint = xr.ones_like(frames_native.isel({"bucket": 0}, drop=True))
    weights = cos_lat_weights(frames_native)
    if weights is not None:
        footprint = footprint * weights

    cells = frames_native.coarsen(blocks, boundary="pad").construct(window)
    areas = footprint.coarsen(blocks, boundary="pad").construct(window)

    block_max = cells.max(["plane_cell_lat", "plane_cell_lon"], skipna=True)
    # A block nothing was observed in paints nothing, so it belongs in neither
    # sum -- it renders as absent, not as an overstated peak.
    observed = np.isfinite(block_max)
    # NaN compares false against everything, so this already excludes both the
    # pad and the unobserved cells without a second mask.
    at_peak = cells == block_max
    return {
        "overstatement_painted_area": (
            areas.where(areas.notnull() & observed, 0.0).sum(fine)
        ),
        "overstatement_peak_area": (
            areas.where(at_peak, 0.0).sum(fine)
        ),
    }


def _frame_intervals(
    axis: pd.DatetimeIndex, cadence: str, group: int, n_frames: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Each frame's half-open interval, from the cadence buckets it holds.

    A stop labeled with one day's timestamp while showing three days averaged
    is a lie the picture itself cannot correct, so the interval is what travels
    rather than a single instant.
    """
    intervals = []
    for index in range(n_frames):
        first = axis[index * group]
        last = axis[min((index + 1) * group, len(axis)) - 1]
        intervals.append((first, _bucket_end(last, cadence)))
    return intervals


def _block_reduce(
    field: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    k_lat: int,
    k_lon: int,
    how: str = "mean",
) -> xr.DataArray:
    """The spatial reduction, composed onto the lazy graph rather than applied
    to a materialized result.

    ``boundary="pad"``, never ``"trim"``: trim silently clips up to k-1 rows
    and columns off the region edge -- 29 of them at the k the full TEMPO
    domain needs -- and the edge of a masked region is where the gradients are.
    The pad is NaN and ``skipna=True`` skips it, verified on real bundles and
    against a synthetic control; an all-NaN block correctly stays NaN rather
    than becoming 0.

    ``how`` is the block reducer, and it is the SAME reducer as the plane's
    temporal one on purpose (D6a decision 4): a max plane block-MAXES, because
    a block mean of a per-pixel max would render a smoothed field and defeat
    the product -- D5a measured the block mean keeping a p50 of 29.8% of a
    single-cell max at k=8, where Phase 11 measured block max keeping 100.0% at
    every percentile of every frame of both live bundles. The pad and all-NaN
    guarantees above were re-measured for ``max``/``min`` in that same gate
    (G3): the pad is skipped and an all-NaN block stays NaN rather than
    collapsing to the ``-inf``/``+inf`` identity element.
    """
    if (k_lat, k_lon) == (1, 1):
        return field
    coarsened = field.coarsen({lat_dim: k_lat, lon_dim: k_lon}, boundary="pad")
    return getattr(coarsened, how)(skipna=True)


def _bare_grid(field: xr.DataArray, lat_dim: str, lon_dim: str) -> xr.DataArray:
    """Strip a block-reduced array's lat/lon coordinate and rename its dims.

    ``coarsen(...).max()`` reduces the lat/lon COORDINATE with the same reducer
    as the data, so a block max's coarse latitude is numerically a different
    number from a block mean's or a block min's for the very same block. Packing
    those arrays into one ``xr.Dataset`` under the shared names
    ``latitude``/``longitude`` makes xarray align them on those coordinate
    VALUES -- an outer join onto the union of all three label sets, which
    silently reindexes EVERY variable in the dataset, including ones nobody
    touched, onto a mostly-NaN grid, and raises nothing. Phase 11's own probe
    script hit it and caught it only because a printed shape stopped matching
    the crop two lines above.

    The blocks are identical across statistics -- same ``k``, same padding --
    so there is one right answer for what to label them, and it is the mean
    plane's, which is already ``FrameStack.lats``/``lons``. Renaming to dims
    carrying no index at all sidesteps alignment entirely: plain positional
    stacking onto the grid the stack already publishes.
    """
    renamed = field.rename({lat_dim: _PLANE_LAT_DIM, lon_dim: _PLANE_LON_DIM})
    return renamed.drop_vars([_PLANE_LAT_DIM, _PLANE_LON_DIM], errors="ignore")


def _observed_area(native: xr.DataArray, spatial: tuple[str, str]) -> xr.DataArray:
    """Per bucket, the cos(latitude)-weighted area that held a value.

    Measured on the NATIVE field, which is the whole point: block-meaning a
    sparsely observed hour produces a frame that looks 99.6% covered when the
    truth is 94.7% (measured), and the inflation grows as the cell budget
    shrinks. Lazy -- it rides the frames' own graph walk.
    """
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    weights = cos_lat_weights(native)
    finite = np.isfinite(native)
    return (finite if weights is None else finite * weights).sum(list(spatial))


def _region_area(
    da: xr.DataArray, spatial: tuple[str, str], region_area: float | None,
) -> float:
    """The denominator: the cos(latitude)-weighted area of the ANALYZED REGION.

    ``region_area`` comes from the rasterized geometry mask, the only place the
    region's own footprint exists. Without it the fallback is the cropped
    grid's area -- the BOUNDING BOX, which is exact for a box-shaped region and
    is the same fallback ``_per_layer_valid_fraction`` takes. For anything else
    the box is too big and every frame under-reports its coverage: a complete
    retrieval over the continental US read 60% covered that way, because the
    Atlantic counted as missing observations.
    """
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    if region_area and region_area > 0:
        return float(region_area)
    lat_dim, lon_dim = spatial
    weights = cos_lat_weights(da)
    if weights is None:
        return float(da.sizes[lat_dim] * da.sizes[lon_dim])
    return float(weights.sum()) * float(da.sizes[lon_dim])


def _valid_fractions(
    computed: xr.Dataset,
    da: xr.DataArray,
    spatial: tuple[str, str],
    region_area: float | None,
) -> list[float]:
    """Observed area over region area, divided once, per frame."""
    total = _region_area(da, spatial, region_area)
    observed = np.nan_to_num(
        np.atleast_1d(np.asarray(computed["observed_area"].values, dtype="float64")),
        nan=0.0,
    )
    if total <= 0:
        return [0.0] * observed.size
    # Clamped: a recorded footprint is measured on the pre-QA grid, so a
    # rounding difference must never publish a coverage above 1.
    return [min(1.0, float(value) / total) for value in observed]


def _delta_terms(
    frames_native: xr.DataArray, period_native: xr.DataArray, group: int,
) -> dict[str, xr.DataArray]:
    """D16's three sums, lazy, at NATIVE resolution.

    ``F`` is the mean of the coarse frame means -- what a reader averaging the
    scrubber would get -- and ``M`` is the period map. In the cadence tier they
    are the same array and there is nothing to measure.

    Compared over the cells finite in BOTH, so a cell one form reaches and the
    other does not is excluded rather than counted as an infinite disagreement.
    Native, because D5a binds here too: measuring on the frame grid would
    compare two block means, and a +1.2 and a -1.2 in one block average away.
    """
    if group == 1:
        return {}
    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    frames_period = frames_native.mean(dim="bucket", skipna=True)
    both = np.isfinite(frames_period) & np.isfinite(period_native)
    difference = np.abs(frames_period - period_native).where(both)
    magnitude = np.abs(period_native).where(both)
    weights = cos_lat_weights(period_native)
    if weights is None:
        weights = 1.0
    return {
        "delta_numerator": (difference * weights).sum(),
        "delta_denominator": (magnitude * weights).sum(),
        "delta_max_abs": difference.max(),
    }


def _delta(computed: xr.Dataset, group: int) -> dict | None:
    """The D16 disclosure: a ratio of weighted sums, and the worst pixel.

    NEVER a mean of per-pixel ratios. Geophysical fields go near zero, so
    relative error explodes there -- a real TEMPO bundle reported 1017% at a
    pixel where the period map sat near zero, and a per-pixel-ratio headline
    would have been that pixel and nothing else.

    NEVER signed. Cancellation reports ~0 for a field off by +-10% in opposite
    directions, which is precisely what this metric exists to catch.

    It gates nothing: 0.4% is fine for one analysis and material for another,
    and that judgment is the reader's.
    """
    if group == 1:
        return None
    denominator = float(computed["delta_denominator"])
    numerator = float(computed["delta_numerator"])
    max_abs = float(computed["delta_max_abs"])
    return {
        "headline": numerator / denominator if denominator > 0 else None,
        "max_abs": max_abs if np.isfinite(max_abs) else None,
        "basis": DELTA_BASIS,
    }


# What the delta actually measures, in one self-describing line, so the number
# explains itself in the payload and in methods.md rather than only in JSX --
# the same reason ``QA_PASS_RATE_BASIS`` exists.
DELTA_BASIS = (
    "cos(latitude)-weighted sum of |frame-derived period mean - period map| over the "
    "same weighted sum of |period map|, at native resolution over cells finite in both"
)


def _frame_grid_delta(
    values: np.ndarray,
    period_values: np.ndarray,
    weights: np.ndarray | None,
    *,
    combine: str = "mean",
) -> dict:
    """D16's metric applied to the arrays that SHIP -- a second quantity beside
    ``_delta``, never a replacement for it.

    The two answer different questions and a reader has both. ``_delta`` asks
    whether the coarser temporal aggregation is a different measurement, and
    measures NATIVELY because a +1.2 and a -1.2 inside one block must not be
    allowed to cancel. This asks whether the two float32 planes in the same
    blob satisfy the identity the payload claims for them, and the only honest
    place to ask that is on the planes themselves.

    Neither bounds the other, which is why both are published rather than one
    standing in for the other: on a real 6-day TEMPO retrieval the native
    figure is 3.74% and this one is 3.28% -- the block mean partly CANCELS the
    time-coarsening term there, and compounds it elsewhere.

    Present in both tiers, unlike ``_delta``. In the cadence tier the temporal
    relationship really is identity and measuring it would find nothing; the
    block mean's disagreement is a separate fact, and at k=(1,1) a measured
    0.0% is the check performed rather than a claim about nothing.

    Plain numpy, after ``.compute()`` and after the float32 narrowing, on
    purpose: these arrays are at most 60 x 20,000 cells and already in hand, so
    this needs no lazy terms and cannot add a second graph walk. Accumulated in
    float64 from the float32 planes -- the float32 is what a reader downloads,
    but summing it in float32 would fold our own rounding into a number whose
    whole subject is somebody else's.

    ``combine`` is the ACROSS-FRAME reduction, and it has to be a parameter
    because the identity being checked is the plane's own. The mean plane
    claims its period plane is the average of its frames; the max plane claims
    the period plane is their maximum. Running the mean's combiner over a max
    plane would compute a real number that answers a question nobody asked --
    and it would be large, not zero, so it would look exactly like the max tier
    failing an identity it in fact satisfies. The mean path below is unchanged
    to the bit.
    """
    frames = np.asarray(values, dtype="float64")
    period = np.asarray(period_values, dtype="float64")

    # An explicit finite mask rather than ``nanmean``/``nanmax``: an all-empty
    # cell is the normal case on a swath-tiled product, not an exception worth
    # warning about, and it must land as NaN so the ``both`` mask below excludes
    # it. (``np.nanmax`` over an all-NaN column warns AND returns ``-inf``,
    # which is the same +-inf leak ``_block_reduce``'s pad guarantee exists to
    # prevent, arriving from the other direction.)
    finite = np.isfinite(frames)
    seen = finite.any(axis=0)
    if combine == "mean":
        contributing = finite.sum(axis=0)
        total = np.where(finite, frames, 0.0).sum(axis=0)
        frames_period = np.divide(
            total, contributing,
            out=np.full(period.shape, np.nan),
            where=contributing > 0,
        )
    else:
        # The identity element as the fill, so an absent cell can never win the
        # selection: -inf for a max, +inf for a min.
        identity = -np.inf if combine == "max" else np.inf
        selected = np.where(finite, frames, identity)
        picked = selected.max(axis=0) if combine == "max" else selected.min(axis=0)
        frames_period = np.where(seen, picked, np.nan)

    both = np.isfinite(frames_period) & np.isfinite(period)
    difference = np.where(both, np.abs(frames_period - period), 0.0)
    magnitude = np.where(both, np.abs(period), 0.0)
    column = (
        np.ones((period.shape[0], 1)) if weights is None
        else np.asarray(weights, dtype="float64").reshape(-1, 1)
    )

    denominator = float((magnitude * column).sum())
    numerator = float((difference * column).sum())
    worst = float(difference.max()) if both.any() else float("nan")
    return {
        "headline": numerator / denominator if denominator > 0 else None,
        "max_abs": worst if np.isfinite(worst) else None,
        "basis": PLANE_AGREEMENT_BASIS[combine],
    }


# The companion to ``DELTA_BASIS``, and it exists for the same reason: a
# quantity with two accounts of itself is how the screen and the document start
# disagreeing. The word doing the work is "stored" -- that is the entire
# difference from ``DELTA_BASIS``' "at native resolution", and it is what makes
# the two impossible to confuse in a payload that carries both.
#
# It says "stored frame grid", NOT "block-mean frame grid": a region under the
# cell ceiling does not coarsen, and naming the grid after a reduction that did
# not run would assert a mechanism a k=(1,1) chart does not have. Where the
# block mean IS real, ``_grid_rule`` prints the realized factors, which is the
# right home for a fact that is only sometimes true. One basis, accurate in
# both regimes, rather than a basis with two forms.
FRAME_GRID_DELTA_BASIS = (
    "cos(latitude)-weighted sum of |mean of the stored frame planes - the stored period "
    "plane| over the same weighted sum of |the stored period plane|, on the stored frame "
    "grid the browser downloads, over cells finite in both"
)


def _plane_agreement_basis(combine: str) -> str:
    """``FRAME_GRID_DELTA_BASIS`` for the plane reduced with ``combine``.

    One sentence per plane rather than one shared sentence, because the word
    that changes is the one carrying the whole claim. A reader shown "mean of
    the stored frame planes" beside a number produced by taking their MAXIMUM
    has been told something false about a real measurement -- and it is a
    particularly quiet falsehood here, because the max plane's number is zero
    under its own basis and would be a large non-zero under the mean's, so the
    printed value would not even hint at the mismatch.
    """
    return FRAME_GRID_DELTA_BASIS.replace(
        "mean of the stored frame planes", f"{combine} of the stored frame planes", 1,
    )


#: The basis string each plane's agreement figure carries, keyed by the
#: across-frame reduction that produced it. ``"mean"`` is ``FRAME_GRID_DELTA_BASIS``
#: itself, unchanged and identical -- a payload and a methods.md that already
#: quote it keep quoting the same bytes.
PLANE_AGREEMENT_BASIS = {
    combine: FRAME_GRID_DELTA_BASIS if combine == "mean" else _plane_agreement_basis(combine)
    for combine in PLANE_STATISTICS
}


def _qa_pass_rates(
    qa_counts: dict | None,
    axis_labels: list[str],
    cadence: str,
    group: int,
    n_frames: int,
) -> list[float | None]:
    """Per-bucket QA pass rates, from the per-timestep area counters.

    ``qa_counts`` is ``MaskedField.counts`` -- what ``apply_quality_mask``
    recorded on the graph walk it was doing anyway. The roll-up sums the
    cos(latitude)-weighted ``passing_area`` and ``checked_area`` over the
    bucket and divides ONCE. It deliberately does not touch
    ``pass_rate_by_time``: those ratios have already been divided, and
    averaging them weights a scan that clipped the region's corner exactly as
    heavily as one that covered all of it.

    Timesteps are matched to buckets by their own TIMESTAMP, never by position.
    The counters are reduced over the inner-aligned arrays while the field here
    may already have had its invalid timesteps dropped, so the two axes are not
    parallel -- and attributing one hour's quality to another is the kind of
    error that shows up as a plausible number.

    Every timestep counts, including ones the mask emptied: an hour QA rejected
    entirely is a blank frame whose 0% pass rate is the only thing separating it
    from an hour nothing was retrieved for.
    """
    if not qa_counts:
        return [None] * n_frames
    times = qa_counts.get("times")
    checked = qa_counts.get("checked_area_by_time")
    passing = qa_counts.get("passing_area_by_time")
    if times is None or checked is None or passing is None:
        return [None] * n_frames

    frame_of = {
        label: index // group for index, label in enumerate(axis_labels)
    }
    stamps = pd.to_datetime(np.asarray(times))
    labels = [start.isoformat() for start in _bucket_starts(stamps, cadence)]
    checked_sum = [0.0] * n_frames
    passing_sum = [0.0] * n_frames
    for label, checked_area, passing_area in zip(labels, checked, passing):
        index = frame_of.get(label)
        if index is None or index >= n_frames:
            continue
        checked_sum[index] += float(checked_area)
        passing_sum[index] += float(passing_area)

    return [
        passing_sum[index] / checked_sum[index] if checked_sum[index] > 0 else None
        for index in range(n_frames)
    ]


#: The per-frame statistics, keyed as ``_field_statistics`` keys them so the
#: scrubber's statistics block and the map's are the same shape (D11: the chart
#: describes what the slider points at, and frame 0 reproduces today exactly).
_FRAME_STATS = ("mean", "min", "max")


def _native_statistics(
    native: xr.DataArray, spatial: tuple[str, str],
) -> dict[str, xr.DataArray]:
    """Per-bucket regional statistics, reduced from the NATIVE field.

    ``reduce_keeping_axes`` is the reuse this module exists to make: it owns
    the cos(latitude)-area-weighted-mean / unweighted-everything-else split
    that ``conduct_temporal_statistic`` established, so a frame's mean cannot
    drift from the map's by re-deriving it here. ``keep=("bucket",)`` is
    literally its job -- collapse the region, keep one value per position on
    the retained axis.

    Lazy: these ride the same graph walk the frames do.
    """
    from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

    stats = {
        f"stat_{name}": reduce_keeping_axes(native, keep=("bucket",), stat=name)
        for name in _FRAME_STATS
    }
    stats["stat_count"] = np.isfinite(native).sum(list(spatial))
    return stats


def _statistics_per_frame(computed: xr.Dataset, n_frames: int) -> list[dict]:
    """The computed reductions, per frame, absent where nothing survived."""
    columns = {
        name: np.atleast_1d(np.asarray(computed[f"stat_{name}"].values, dtype="float64"))
        for name in _FRAME_STATS
    }
    counts = np.nan_to_num(np.asarray(computed["stat_count"].values), nan=0.0)

    out: list[dict] = []
    for index in range(n_frames):
        count = int(counts[index])
        # A bucket with no surviving cell has no mean, no floor and no peak.
        # Emitting 0.0 for them would put a measurement where there is an
        # absence, which is the whole of D10.
        if count == 0:
            out.append({"count": 0})
            continue
        stats = {"count": count}
        for name, column in columns.items():
            value = float(column[index])
            if np.isfinite(value):
                stats[name] = value
        out.append(stats)
    return out


def _spatial_dims(da: xr.DataArray, spatial: list[str]) -> tuple[str, str]:
    """The (lat, lon) dimension names, by CF metadata rather than by position.

    T24's rule: a Harmony-reformatted grid can arrive ordered (time, lon, lat),
    and coarsening the wrong axis would blur the field along one direction
    while leaving the other native.
    """
    lat_dim, lon_dim = find_lat_coord(da), find_lon_coord(da)
    if lat_dim in spatial and lon_dim in spatial:
        return lat_dim, lon_dim
    raise ValueError(
        "A frame stack needs an identifiable latitude and longitude dimension; "
        f"got dims {tuple(da.dims)}."
    )
