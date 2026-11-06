"""
services/methods_export_service.py
====================================
Assembles the Markdown a paper's methods section needs, deterministically,
from an artifact's own lineage and citations (provenance_service's output) —
never from the chat transcript, and never through an LLM, so the same
session always yields the same text. An optional LLM polish pass may later
rewrite the prose, but these structured facts are what get re-validated
against afterward.

Consumes the shapes provenance_service now produces from the real MCP:
lineage nodes ``{handle, kind, events?: [{event_type, detail, created_at}]}``
and citations ``{handle, doi, collection_citations: [...]}`` (CMR UMM-C
records verbatim). Every field access is defensive — an unexpected shape
must degrade to a blander line, never a KeyError that 500s the endpoint
(QA 2026-07-17 blocker).
"""
from __future__ import annotations

from typing import Any


def build_methods_markdown(
    artifact_title: str,
    aoi_description: str,
    time_window: str,
    lineage: dict[str, Any],
    citations: list[dict[str, Any]],
    maturity: str | None = None,
    maturity_note: str | None = None,
    chart: dict[str, Any] | None = None,
) -> str:
    nodes = lineage.get("nodes") or []

    lines = [
        f"## Methods — {artifact_title}",
        "",
        f"Data were retrieved for the area of interest **{aoi_description}** over "
        f"the period **{time_window}**.",
        "",
        "### Datasets",
        "",
    ]
    for node in nodes:
        if node.get("kind") == "dataset":
            citation = next(
                (c for c in citations if c.get("handle") == node.get("handle")),
                None,
            )
            title = _citation_title(citation) or node.get("handle", "unknown dataset")
            doi_suffix = f" (doi: {citation['doi']})" if citation and citation.get("doi") else ""
            lines.append(f"- {title}{doi_suffix}")
    lines += ["", "### Processing chain", ""]
    for index, node in enumerate(nodes, start=1):
        step = f"{index}. **{node.get('handle', 'unknown')}** ({node.get('kind', 'step')})"
        step_text = _step_text(node)
        if step_text:
            step += f" — {step_text}"
        lines.append(step)

    retrieval_dates = _retrieval_dates(nodes)
    lines += ["", "### Retrieval dates", ""]
    for date in retrieval_dates:
        lines.append(f"- {date}")

    lines += _frames_section(chart or {})

    # Above the references because it qualifies what the reader will cite.
    # Omitted when unknown: an "unknown" heading would read as a finding.
    if maturity and maturity != "unknown":
        lines += ["", "### Data maturity", "", f"- Product maturity: **{maturity}**"]
        if maturity_note:
            lines.append(f"- {maturity_note}")

    lines += ["", "### References", ""]
    for index, citation in enumerate(citations, start=1):
        lines.append(f"{index}. {_citation_text(citation)}")

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Temporal-frames disclosure
#
# A chart row holds two frame blocks: the reproducibility recipe
# (``payload["export"]["frames"]["spec"]``) and the render block
# (``payload["frames"]``, written by ``frame_store._axis_block``).
#
# One precedence rule, applied only in ``_recipe``: the spec wins for every
# field it carries, and the render block fills in shared fields when there's
# no spec. Measured facts (per-frame counts, the delta) come only from the
# render block. They're never copied into the spec, so each number has one
# source.
# --------------------------------------------------------------------------

#: Fields the recipe and the artifact both carry, and which therefore need the
#: precedence rule. The spec's own extras (``boundary``, ``statistic``,
#: ``span``, ``n_frames``) simply go missing on a row without one — a line that
#: cannot be stated is dropped, never guessed.
_SHARED_RECIPE_FIELDS = (
    "cadence", "tier", "buckets_per_frame", "coarsen_k", "cells_per_frame",
)


def _recipe(chart: dict[str, Any], render: dict[str, Any]) -> dict[str, Any]:
    merged = {
        key: render[key] for key in _SHARED_RECIPE_FIELDS if render.get(key) is not None
    }
    merged.update(
        {key: value for key, value in _frames_spec(chart).items() if value is not None}
    )
    return merged


def _frames_section(chart: dict[str, Any]) -> list[str]:
    """The bucketing disclosure, or nothing.

    Three states, the same three the scrubber has:

    * **frames present** — the full disclosure.
    * **refused** — which refusal, in the backend's own words.
    * **neither** — silence. Frames don't change the map, so their absence
      says nothing about the figure, and "temporal frames: none" would read
      as a finding.
    """
    render = chart.get("frames")
    if not isinstance(render, dict) or not render.get("frames"):
        return _refusal_section(chart)

    recipe = _recipe(chart, render)
    cadence = str(recipe.get("cadence") or "unknown")
    n_frames = _as_int(recipe.get("n_frames")) or len(render.get("frames") or [])
    # A frame is one cadence interval in tier one and several in tier two, so
    # the cadence qualifies the frames ONLY in tier one. "12 hourly frames"
    # over five-hour frames would contradict the weighting bullet beneath it.
    coarsened = _is_coarsened(recipe)
    unit = "frames" if coarsened else f"{cadence} frames"
    lines = ["", "### Temporal frames", ""]
    lines.append(
        f"This figure also carries a browsable time axis of **{n_frames:,} "
        f"{unit}**{_span_phrase(recipe)}."
    )
    lines.append("")
    lines.append(_weighting_rule(recipe, cadence, coarsened))
    grid = _grid_rule(recipe)
    if grid:
        lines.append(grid)
    lines.append(_gap_rule(render.get("frames") or [], coarsened))
    # Exports are always native resolution, never the frame grid. The label
    # comes from the row; if the row doesn't say, don't invent it.
    label = _export_label(chart)
    if label:
        lines.append(
            f"- Exports and downloads of this figure are the **{label}** at "
            "native resolution — never the frame grid, and never an individual "
            "frame."
        )
    agreement = _delta_section(render, recipe, cadence, chart.get("units"))
    lines += agreement
    lines += _planes_section(
        render, recipe, cadence, chart.get("units"), bool(agreement),
    )
    return lines


#: The nouns the extra planes are named by, in the backend's own order
#: (``PLANE_STATISTICS`` minus the mean, which is not a ``planes`` key). Nouns
#: rather than the payload's keys because this document is prose: "browsed as a
#: per-interval max" reads as jargon for a field name where "maximum" reads as
#: the quantity it is.
_SELECTION_NOUNS = {"max": "maximum", "min": "minimum"}

#: What one frame of each plane holds, in the same voice ``_weighting_rule``
#: describes the mean's in.
_SELECTION_VERBS = {"max": "highest", "min": "lowest"}


def _planes_section(
    render: dict[str, Any], recipe: dict[str, Any], cadence: str, units: Any,
    agreement_stated: bool,
) -> list[str]:
    """What else this figure's time axis can be browsed as (e.g. max/min).

    Planes are browsable, never downloadable.

    With ``agreement_stated``, the lead says the agreement figures above are
    the mean's. On screen the toggle replaces them, but here both sections
    appear, so a reader could otherwise take them for this plane's.
    """
    selections = [
        name for name in _browsable_statistics(render, recipe)
        if name in _SELECTION_NOUNS
    ]
    if not selections:
        return _plane_refusal_section(render)

    nouns = " and ".join(f"**{_SELECTION_NOUNS[name]}**" for name in selections)
    lead = (
        f"This figure's time axis can also be browsed as a per-interval {nouns}. "
        "They are viewing modes: exports and downloads are unchanged."
    )
    if agreement_stated:
        lead += (
            " The frame–map agreement figures above are the period mean's; "
            "each plane's own is stated with it."
        )
    lines = ["", "### Browsable statistics", "", lead, ""]
    planes = render.get("planes")
    planes = planes if isinstance(planes, dict) else {}
    for name in selections:
        plane = planes.get(name)
        lines += _plane_lines(
            name, plane if isinstance(plane, dict) else {}, recipe, cadence, units,
        )
    return lines


def _plane_refusal_section(render: dict[str, Any]) -> list[str]:
    """Why this figure has a scrubber and no extra statistics — or silence.

    * **planes present** — the disclosure above.
    * **refused** — this, with the gate's own sentence, so a missing toggle
      always comes with a reason.
    * **neither** (no ``planes`` key, e.g. older charts) — silence.
    """
    refusal = render.get("planes_unavailable")
    if not isinstance(refusal, dict) or not refusal.get("reason"):
        return []
    sentence = "This figure's time axis is browsable as a period mean only."
    detail = str(refusal.get("detail") or "").strip()
    if detail:
        sentence = f"{sentence} {detail}"
    return ["", "### Browsable statistics", "", sentence]


def _plane_lines(
    name: str, plane: dict[str, Any], recipe: dict[str, Any], cadence: str,
    units: Any,
) -> list[str]:
    """One plane: what a frame of it holds, and whether it satisfies the
    identity the payload claims for it.

    Based on the payload's measured figure, never the operation's name. A
    non-zero figure for max/min is reported as a defect, since selection
    can't disagree with itself.
    """
    noun = _SELECTION_NOUNS[name]
    figure = _delta_figure(plane.get("frame_grid_delta"), units)
    lead = f"- **{noun.capitalize()}**: {_selection_frame_rule(name, recipe, cadence)}"
    if figure is None:
        # Unmeasured is not the same as agreeing (e.g. older charts).
        return [
            lead
            + f" Whether taking the {noun} of the stored frame planes "
            "reproduces the stored period plane was not measured for this "
            "figure."
        ]
    if figure[0] == 0.0:
        # Stated, not measured — and stated as an exact identity, because that
        # is what a zero here means. ``_pct`` would render it "under 0.1%",
        # which reads as a residual too small to resolve rather than as an
        # equality that holds.
        lead += (
            f" Taking the {noun} of the stored frame planes reproduces the "
            f"stored period plane exactly — a {noun} selects one of the values "
            "it is given rather than combining them, so there is no "
            "disagreement here to measure."
        )
    else:
        lead += (
            f" Taking the {noun} of the stored frame planes misses the stored "
            f"period plane by **{_pct(figure[0])}**{figure[1]} — which a {noun} "
            "is not able to do, so read it as a defect in this figure rather "
            "than as a caveat on it."
        )
    lines = [lead]
    # The account of what was compared, quoted rather than paraphrased, in both
    # branches: an identity claimed without saying what it was checked against
    # is the same unanchored assertion ``_figure_lines`` refuses for a figure.
    if figure[2]:
        lines.append(f"- Basis: {figure[2]}.")
    return lines + _overstatement_lines(plane)


def _overstatement_lines(plane: dict[str, Any]) -> list[str]:
    """How much ground a rendered peak claims, versus what was measured.

    ``ceiling`` (k²) is NOT an upper bound: the sums are cos(latitude)-
    weighted, so the figure can exceed k². Never say "up to k×" and don't
    render ``ceiling``; state only this chart's measured figure.

    Only the max plane has this; others return nothing (not a zero).
    """
    measured = plane.get("extent_overstatement")
    if not isinstance(measured, dict):
        return []
    headline = _as_float(measured.get("headline"))
    if headline is None:
        return []
    lines = [
        "- Every cell of a block is drawn at that block's highest value, so "
        f"the ground shown at peak level is about **{headline:.1f}×** the "
        "ground the peak was actually measured over."
    ]
    worst = _as_float(measured.get("worst_frame"))
    if worst is not None:
        # Beside the pooled figure rather than folded into it, the same way
        # ``delta`` carries a headline and a worst pixel.
        lines[-1] += (
            f" Its single worst frame stretched one further, to {worst:.1f}×."
        )
    basis = measured.get("basis")
    if basis:
        lines.append(f"- Basis: {basis}.")
    return lines


def _selection_frame_rule(name: str, recipe: dict[str, Any], cadence: str) -> str:
    """What one frame of a selection plane holds, per tier.

    The noun trap this project has now hit three times. Tier one: a frame IS a
    cadence interval, so "the highest value one hourly interval holds" is
    exact. Tier two: a frame's value is the highest across
    ``buckets_per_frame`` intervals, and calling that "the maximum of each
    interval" is wrong in the direction that makes the plane sound more precise
    than it is — the same direction ``_gap_rule``'s "13 of the 48 intervals"
    was wrong in.
    """
    verb = _SELECTION_VERBS[name]
    if _is_coarsened(recipe):
        return (
            f"each frame is the {verb} value across the "
            f"{_count_phrase(recipe)}{cadence} intervals it spans, at each cell."
        )
    return (
        f"each frame is the {verb} value one {cadence} interval of this "
        "product holds at each cell."
    )


def _browsable_statistics(
    render: dict[str, Any], recipe: dict[str, Any],
) -> list[str]:
    """Which statistics the axis actually offers, from what LANDED.

    The recipe's list first — it is ``_scrubbable_statistics``' own output,
    derived at plot time from the keys that got a ``_key`` — and the render
    block's own planes as the fallback for a row written without one, read off
    ``url`` under the same rule the frontend uses: a plane whose write failed,
    or one evicted since, has a block and no url, and naming it here would
    describe a mode the reader cannot open.
    """
    stated = recipe.get("statistics")
    if isinstance(stated, (list, tuple)):
        return [str(name) for name in stated]
    planes = render.get("planes")
    if not isinstance(planes, dict):
        return []
    return [
        name for name in planes
        if isinstance(planes[name], dict) and planes[name].get("url")
    ]


def _export_label(chart: dict[str, Any]) -> str | None:
    label = _export_frames(chart).get("exports")
    return str(label) if label else None


# Above this, the delta's lead sentence goes bold; below it, it's stated
# without comment (whether it matters is the reader's call).
#
# Must match the ``high`` edge in ``frameDelta.js:severityOf``; a test pins
# them together.
_DELTA_HIGH = 0.10


def _delta_section(
    render: dict[str, Any], recipe: dict[str, Any], cadence: str, units: Any,
) -> list[str]:
    """The two disagreements, as one section — or nothing at all.

    Two different questions, so both are stated:

    * ``delta``: are the frames a different temporal aggregation from the
      map? Measured at native resolution so errors can't cancel per block.
    * ``frame_grid_delta``: is the map the average of the shipped frame
      arrays? Measured on the arrays.

    Neither bounds the other. In the cadence tier ``delta`` is absent (a
    frame is a bucket), but the frames were still block-meaned for display,
    so ``frame_grid_delta`` can be non-zero.

    Silent when nothing was measured and the tier is exact. Placed above the
    References because it qualifies what the reader will cite.
    """
    coarsened = _is_coarsened(recipe)
    native = _delta_figure(render.get("delta"), units)
    shipped = _delta_figure(render.get("frame_grid_delta"), units)
    if not coarsened and shipped is None:
        return []

    lead = _agreement_lead(recipe, cadence, coarsened)
    if coarsened and native is None:
        # Coarsened, but the temporal disagreement was not measured. Saying so
        # is the honest state; omitting it would let a reader assume the frames
        # and the map are the same aggregation.
        lead += (
            " The size of that disagreement was not measured for this figure."
        )

    loudest = max(
        (figure[0] for figure in (native, shipped) if figure is not None),
        default=None,
    )
    if loudest is not None and loudest >= _DELTA_HIGH:
        lead = f"{_loud_lead(loudest, coarsened)} {lead}"

    lines = ["", "### Frame–map agreement", "", lead, ""]
    if native is not None:
        lines += _figure_lines(f"Disagreement: **{_pct(native[0])}**", native)
    if shipped is not None:
        lines += _figure_lines(
            "Averaging the frame planes and comparing them to the period plane "
            f"beside them: **{_pct(shipped[0])}**",
            shipped,
        )
    if native is not None and shipped is not None:
        # Otherwise a reader assumes the larger number contains the smaller.
        lines.append(
            "- The two are measured on different arrays and neither bounds the "
            "other: the block mean can cancel part of the coarsening as easily "
            "as compound it."
        )
    return lines


def _delta_figure(
    delta: Any, units: Any,
) -> tuple[float, str, str | None] | None:
    """One measured disagreement as ``(headline, largest-clause, basis)``, or
    ``None`` when it was not measured. Shared by both so the two figures cannot
    acquire different formatting and read as different kinds of quantity."""
    delta = delta if isinstance(delta, dict) else {}
    headline = _as_float(delta.get("headline"))
    if headline is None:
        return None
    max_abs = _as_float(delta.get("max_abs"))
    largest = ""
    if max_abs is not None:
        unit_suffix = f" {units}" if units else ""
        largest = f", largest absolute difference {max_abs:.3e}{unit_suffix}"
    basis = delta.get("basis")
    return headline, largest, str(basis) if basis else None


def _figure_lines(label: str, figure: tuple[float, str, str | None]) -> list[str]:
    """A figure and, directly beneath it, the account of what it measured.

    Quoted, never paraphrased, and never shared between the two: restating a
    basis in different words creates a second definition of one quantity, and
    reusing one basis across two quantities tells the reader something false
    about a real measurement. Both are the failure ``DELTA_BASIS`` exists to
    prevent.
    """
    _headline, largest, basis = figure
    lines = [f"- {label}{largest}."]
    if basis:
        lines.append(f"- Basis: {basis}.")
    return lines


# Must match the floor in ``frameDelta.js:formatPct``;
# ``SharedDeltaThresholdTests`` reads the JS source and compares.
_DELTA_FLOOR = 0.001
_DELTA_FLOOR_TEXT = "under 0.1%"


def _pct(headline: float) -> str:
    """One decimal place, or an honest floor.

    Below 0.1% it's float32 noise, and "0.0%" would claim false precision.
    """
    if headline < _DELTA_FLOOR:
        return _DELTA_FLOOR_TEXT
    return f"{headline * 100:.1f}%"


def _agreement_lead(recipe: dict[str, Any], cadence: str, coarsened: bool) -> str:
    """What kind of relationship the frames have to the map, per tier — and,
    in tier one, whether there is a block mean at all.

    The k=(1,1) branch is not a corner case. A New Jersey retrieval is 10,624
    native cells, under the 20,000-cell ceiling, so it does not coarsen — and
    small regions are most regions. Telling that reader their frames were
    "block-meaned to the rendering resolution" describes a mechanism their
    figure does not have, which is the exact failure this section was added to
    remove. Found on live data, in this function's first draft.
    """
    if coarsened:
        return (
            f"Each frame averages {_count_phrase(recipe) or 'several '}{cadence} "
            "intervals, so the frames are a coarser temporal aggregation than "
            "the map, which is computed independently at native resolution."
        )
    same = (
        f"Each frame is one {cadence} interval and the period map is their "
        "equally-weighted mean, so the frames and the map are the same "
        "temporal aggregation."
    )
    if not _is_downsampled(recipe):
        return (
            f"{same} They are also on the same grid — these frames are not "
            "downsampled — so the residual below is float32 storage precision: "
            "the frames were narrowed individually and the map was narrowed "
            "after averaging."
        )
    return (
        f"{same} They are not on the same grid: the frame planes are "
        "block-meaned to the rendering resolution and the map is not, and a "
        "block mean does not commute with an average over intervals wherever "
        "coverage is uneven."
    )


def _is_downsampled(recipe: dict[str, Any]) -> bool:
    """Whether a block mean actually ran. Absent ``coarsen_k`` reads as
    downsampled: a row that cannot say is a row whose frames may well have
    been, and claiming the stronger "same grid" property for it would be
    inventing the thing this branch exists to stop inventing."""
    factors = recipe.get("coarsen_k")
    if not isinstance(factors, (list, tuple)) or not factors:
        return True
    return any((_as_int(k) or 1) > 1 for k in factors)


def _loud_lead(headline: float, coarsened: bool) -> str:
    """The bolded escalation, keyed to the LARGEST disclosed disagreement.

    One rule over everything disclosed, rather than one per figure. Keying it
    to the native delta alone would leave the cadence tier — which has no
    native delta at all — unable to raise its voice, and that is precisely the
    tier where a double-digit array disagreement is most surprising, because
    the temporal relationship there is an exact identity.
    """
    if coarsened:
        return (
            "**The frames and the period map are answering materially "
            f"different questions: they disagree by {_pct(headline)} of the "
            "map's own magnitude.**"
        )
    return (
        "**Averaging this figure's frame planes does not reproduce the period "
        f"plane: they differ by {_pct(headline)} of the map's own magnitude.**"
    )


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _gap_rule(frames: list[Any], coarsened: bool) -> str:
    """How many frames hold nothing, and — kept apart — which kind of nothing.

    Counted in FRAMES, and named as intervals only in tier one where the two
    are the same thing. Each entry here is one frame; in tier two a frame spans
    several cadence intervals, so calling three empty frames "three empty
    hours" would understate the gap by the coarsening factor.

    ``n_granules == 0`` with a ``qa_pass_rate`` is "observed, and quality
    screening rejected all of it". ``n_granules == 0`` with none is "nothing
    was retrieved". They are different measurements about different problems,
    and one tally that merges them reports a QA failure as a coverage failure.
    Emptiness is read off the surviving granule count, the same way
    ``frameAxis.js`` reads it for the screen.
    """
    total = len(frames)
    rejected = retrieved_none = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        count = _as_int(frame.get("n_granules"))
        if count is None or count > 0:
            continue
        if frame.get("qa_pass_rate") is None:
            retrieved_none += 1
        else:
            rejected += 1

    noun, plural = ("frame", "frames") if coarsened else ("interval", "intervals")
    empty = rejected + retrieved_none
    if not empty:
        return (
            f"- Empty {plural}: none; all {total:,} hold at least one "
            "contributing granule."
        )
    kinds = []
    if rejected:
        kinds.append(
            f"{rejected:,} observed but rejected entirely by quality screening"
        )
    if retrieved_none:
        kinds.append(f"{retrieved_none:,} with nothing retrieved")
    return (
        f"- Empty {plural}: {empty:,} of {total:,} — {', and '.join(kinds)}. "
        f"An empty {noun} contributes nothing to the period mean and is not "
        "interpolated."
    )


def _refusal_section(chart: dict[str, Any]) -> list[str]:
    """Which refusal, in the gate's own words — or silence.

    ``_DISCLOSED_FRAME_REFUSALS`` are the three worth saying out loud, and only
    one of them (``extent_too_large``) is fixed by narrowing the map. The
    detail sentence is quoted rather than paraphrased, for ``DELTA_BASIS``'
    reason: a second account of why the axis is missing is a second account.
    The rest — a single granule, one interval, an unselected vertical axis —
    withhold nothing the request implied and stay silent here too.
    """
    disclosure = chart.get("frames_unavailable")
    if not isinstance(disclosure, dict) or not disclosure.get("reason"):
        return []
    sentence = "No frame-by-frame time axis was built for this figure."
    detail = str(disclosure.get("detail") or "").strip()
    if detail:
        sentence = f"{sentence} {detail}"
    return ["", "### Temporal frames", "", sentence]


def _is_coarsened(recipe: dict[str, Any]) -> bool:
    """Tier two, by either witness: the tier itself, or a frame that spans more
    than one cadence bucket.

    Two witnesses on purpose. What this decides — whether a frame may be called
    an "interval" — is a category error when it is wrong, understating a gap by
    the coarsening factor, so it must not turn on one field having survived.
    """
    if str(recipe.get("tier") or "") == "coarsened":
        return True
    return (_as_int(recipe.get("buckets_per_frame")) or 1) > 1


def _weighting_rule(recipe: dict[str, Any], cadence: str, coarsened: bool) -> str:
    """How equal weighting applies in this tier.

    Tier one: a frame is a cadence bucket, and the map is the mean of the
    frames. Tier two: a frame is the mean of its buckets' means, not of their
    granules, so long spans don't reintroduce granule weighting.
    """
    if not coarsened:
        # Say "intervals at native resolution", not "frames": averaging the
        # rendered frames won't exactly reproduce the map (see _delta_section).
        return (
            f"- Each frame is one {cadence} interval of this product, and the "
            "period map is the equally-weighted mean of those intervals at "
            "native resolution: every interval counts once, however many "
            "granules it holds."
        )
    return (
        f"- Each frame is the mean of its {_count_phrase(recipe)}{cadence} "
        "intervals' means, not of their granules: every interval counts once, "
        "however many granules it holds."
    )


def _count_phrase(recipe: dict[str, Any]) -> str:
    """``"5 "`` when the coarsening factor is known, ``""`` when it is not — a
    row that lost the field must read "its hourly intervals' means", never
    "its 0 hourly intervals' means"."""
    per_frame = _as_int(recipe.get("buckets_per_frame")) or 0
    return f"{per_frame:,} " if per_frame > 1 else ""


def _grid_rule(recipe: dict[str, Any]) -> str | None:
    """The block-mean grid and the cell count actually realized.

    Never the 20,000-cell ceiling: integer coarsening quantizes, so a real grid
    lands at 70-96% of what was asked for, and the recipe carries what landed.
    """
    cells = _as_int(recipe.get("cells_per_frame"))
    if not cells:
        return None
    factors = [str(k) for k in (recipe.get("coarsen_k") or []) if _as_int(k)]
    block = f" — the mean over {' × '.join(factors)} native blocks" if factors else ""
    boundary = recipe.get("boundary")
    if block and boundary:
        block += f", boundary `{boundary}`"
    return (
        f"- Frames are rendered on a block-mean grid of {cells:,} cells per "
        f"frame{block + ' —' if block else ''} which is a rendering "
        "resolution, not a scientific product."
    )


def _export_frames(chart: dict[str, Any]) -> dict[str, Any]:
    """``payload["export"]["frames"]`` — the recipe block — or an empty one."""
    export = chart.get("export")
    frames = export.get("frames") if isinstance(export, dict) else None
    return frames if isinstance(frames, dict) else {}


def _frames_spec(chart: dict[str, Any]) -> dict[str, Any]:
    spec = _export_frames(chart).get("spec")
    return spec if isinstance(spec, dict) else {}


def _span_phrase(recipe: dict[str, Any]) -> str:
    span = recipe.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2 and all(span):
        return f", spanning {span[0]} to {span[1]}"
    return ""


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _step_text(node: dict[str, Any]) -> str:
    events = node.get("events") or []
    return "; ".join(_event_text(event) for event in events)


def _event_text(event: dict[str, Any]) -> str:
    detail = event.get("detail") or {}
    detail_parts = ", ".join(f"{key} {value}" for key, value in detail.items())
    when = event.get("created_at") or "time unknown"
    suffix = f", {detail_parts}" if detail_parts else ""
    return f"{event.get('event_type', 'event')} ({when}{suffix})"


def _retrieval_dates(nodes: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for node in nodes:
        for event in node.get("events") or []:
            if event.get("event_type") == "materialized":
                date = str(event.get("created_at") or "")[:10]
                if date and date not in dates:
                    dates.append(date)
    return dates


def _citation_title(citation: dict[str, Any] | None) -> str | None:
    """The dataset's own title, from the first CMR CollectionCitations entry."""
    if not citation:
        return None
    for entry in citation.get("collection_citations") or []:
        title = entry.get("Title")
        if title:
            return str(title)
    return None


def _citation_text(citation: dict[str, Any]) -> str:
    """One reference line from a CMR citation record: the formal
    CollectionCitations fields when published, else the DOI, else the handle."""
    for entry in citation.get("collection_citations") or []:
        parts = [
            str(entry[key])
            for key in ("Creator", "Title", "Publisher", "ReleaseDate")
            if entry.get(key)
        ]
        if parts:
            text = ", ".join(parts)
            if citation.get("doi"):
                text += f", doi:{citation['doi']}"
            return text
    if citation.get("doi"):
        return f"doi:{citation['doi']}"
    return str(citation.get("handle", ""))
