"""
services/methods_export_service.py
====================================
T10: assembles the Markdown a paper's methods section needs, deterministically,
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

    # T57: stated ABOVE the references, because it qualifies everything the
    # reader is about to cite. Emitted only when the maturity is actually
    # known: printing a "Data maturity: unknown" heading would read as a
    # finding, and an unstated maturity must never become a reassuring one.
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
# T59 Phase 7 — the temporal-frames disclosure (D12).
#
# Two blocks exist in a chart's jsonb row and they could in principle disagree:
# ``payload["export"]["frames"]["spec"]`` is the reproducibility recipe written
# beside ``source_handles``/``variable``/``region_name``, and
# ``payload["frames"]`` is the artifact ``frame_store._axis_block`` wrote.
#
# ONE precedence rule, applied ONCE in ``_recipe`` rather than per call site:
# **the spec is authoritative for every field it carries**, the render block
# stands in for the fields it shares when a row has no spec, and the two are
# never consulted separately for the same statement. The measured facts — the
# per-frame counts the gap tally is derived from, and the D16 delta — come from
# the render block alone, because only it has them.
#
# They are NOT copied into the spec at plot time. Both blocks are written
# together by ``_attach_frames``, in the same branch of the same write, so
# there is no state where one exists without the other; duplicating the delta
# into the recipe would put one number in two places in one document, which is
# the disagreement this rule exists to prevent.
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
    * **neither** — silence. Frames are additive (D15): the map is identical
      whether or not one was built, so their absence says nothing about the
      figure the reader is citing. Printing "temporal frames: none" would read
      as a finding about a feature the request never implied — the same reason
      T57 omits an unknown maturity rather than printing one.
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
    # D12, for the reader who is about to put the download in a paper: the
    # frame grid is a rendering resolution and export never touches it. The
    # label is taken from the row rather than restated here — a row that does
    # not say what its export is must not have the claim invented for it.
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
    """What else this figure's time axis can be browsed as (D6a).

    Its own section, for ``### Frame–map agreement``'s reason: it carries a
    measured number that qualifies what the reader is about to cite. The lead
    keeps D12 unambiguous in the same breath — a plane is browsable and is not
    downloadable, in any phase.

    ``agreement_stated`` closes a hazard this document has and the screen does
    not. On screen the toggle REPLACES the delta disclosure — ``resolveFrameDelta``
    branches on the statistic before it touches either of the mean's figures —
    so the mean's 1.876% is never on the page beside a max plane. Here both
    sections are always present, and a figure whose subject is unstated three
    lines above bullets about a different statistic is a figure a reader can
    take for that statistic's. The clause lives in this section rather than
    that one so the figures above keep their single account of themselves.
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

    Three states here too, and the middle one did not exist before Phase 14:

    * **planes present** — the disclosure above.
    * **refused** — this, with the gate's own sentence. ``_refusal_section``'s
      posture one level in: what is missing is the toggle rather than the axis,
      and a reader who finds no mode and no reason is exactly the person the
      disclosed refusals exist to keep out of that position.
    * **neither** — silence, and a document byte-identical to what it was.
      Phase 13 omitted ``planes`` rather than emitting it empty precisely so
      this state stays distinguishable from the one above, and every chart
      plotted before Phase 14 is in it.
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

    **The identity is read off the payload's own figure, never off the
    operation's name.** A maximum is associative in theory, and printing
    "exactly" on that ground would be the same class of falsehood Phase 8
    caught one file along — true of the mechanism, false of the arrays. The
    exact branch is what the backend measures today (Phase 11 G5: ``0.0`` max
    abs diff on both identities, both live bundles); the other exists so it
    cannot silently become wrong, and it names the figure a defect because for
    this statistic that is what it would be.
    """
    noun = _SELECTION_NOUNS[name]
    figure = _delta_figure(plane.get("frame_grid_delta"), units)
    lead = f"- **{noun.capitalize()}**: {_selection_frame_rule(name, recipe, cadence)}"
    if figure is None:
        # Unmeasured is not the same as agreeing — ``_delta_section``'s rule,
        # one plane in. Every row written before Phase 12 is in this state.
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
    """D6a decision 9 / Phase 11 G4: how much ground a rendered peak claims.

    **Phase 14's finding binds this wording.** ``ceiling`` is k² and is NOT an
    upper bound — both sums in ``_overstatement_terms`` are cos(latitude)-
    weighted per cell, so a block whose max sits on its lowest-weighted row
    contributes more than k², and the pooled figure straddles k² rather than
    approaching it from below (measured 4.0000014 against a ceiling of 4).
    Nothing here says "up to k×" or "at most 25×", and ``ceiling`` is
    deliberately not rendered at all. What the quantity supports is this
    chart's own measured figure.

    ``None`` on every plane but max, and this is silence rather than a zero: a
    block minimum paints a trough and decision 9 measures the peak.
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


# Where a disagreement stops being a caveat and starts being a finding, and the
# ONLY threshold this document has: below it the delta is still stated, in its
# own section, without editorial — "0.4% is fine for one analysis and material
# for another, and that judgment is the reader's" (D16). Above it the lead
# sentence goes bold, which is the only hierarchy Markdown offers.
#
# The value is ``frameDelta.js:severityOf``'s ``high`` edge, and Phase 3's two
# real measurements fall on either side of it: 5.4% is a caveat a reader should
# carry, 22.3% means the scrubber and the map are answering materially
# different questions. The screen resolves three bands and the document two, so
# only this edge is shared — pinned by a test naming the JS source, because two
# thresholds for one number is how the screen and the document start
# disagreeing.
_DELTA_HIGH = 0.10


def _delta_section(
    render: dict[str, Any], recipe: dict[str, Any], cadence: str, units: Any,
) -> list[str]:
    """The two disagreements, as one section — or nothing at all.

    **Two, because there are two questions and a reader has both.** ``delta``
    (D16) asks whether the frames are a different temporal aggregation from the
    map, and answers it at native resolution so a +1.2 and a -1.2 inside one
    block cannot cancel. ``frame_grid_delta`` asks whether the two arrays in
    the payload are each other's average, and answers it on the arrays. Phase 8
    measured 3.74% and 3.28% for one chart: the block mean partly cancelled the
    coarsening term there, so neither number bounds the other and publishing
    only one leaves the reader with the wrong bound in an unknown direction.

    Emitted in BOTH tiers now. In the cadence tier ``delta`` is rightly absent
    — a frame IS a cadence bucket and measuring their difference would find
    nothing — but the frames were still block-meaned to the rendering grid and
    the map was not, so a figure with no temporal disagreement can and does
    have an array disagreement. That is the gap Phase 8 found: the tier with
    nothing to disclose was the tier stating the flattest identity.

    Silent when nothing at all was measured and the tier is exact, which is
    every row written before this was measured. A section built from what was
    measured rather than from what would be nice to say.

    The T57 maturity block's placement, for the T57 reason: above the
    References, because it qualifies everything the reader is about to cite.
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
        # Phase 8 §2, in one line, because a reader handed two numbers will
        # otherwise assume the larger one contains the smaller.
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


# The second number this document shares with the screen, after
# ``_DELTA_HIGH`` — and it is here as a named constant for the same reason that
# one is. Phase 8 measured what an unpinned shared value is worth: editing the
# JS edge to 0.15 left every test in both languages green, because a comment
# naming the other file is not a check. ``frameDelta.js:formatPct`` draws the
# floor in the same place, and ``SharedDeltaThresholdTests`` reads it out of
# the JS source and compares.
_DELTA_FLOOR = 0.001
_DELTA_FLOOR_TEXT = "under 0.1%"


def _pct(headline: float) -> str:
    """One decimal place, or an honest floor.

    Under 0.1% the document cannot resolve the number it would be printing:
    an undownsampled chart's arrays agree to float32 storage noise (Phase 3's
    0.000002%), and rendering that as "0.0%" claims a precision this document
    does not have while reading like a measurement that found nothing.
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
    """How many frames hold nothing, and — kept apart — which kind of nothing
    (D10).

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
    """D4's equal weighting, at whichever level this tier applies it.

    Tier one: a frame IS a cadence bucket, and the map is the mean of the
    frames. Tier two: a frame is the mean of its buckets' MEANS, not of their
    granules — D4's rule one level up, so the tier that exists to fit long
    spans does not reintroduce granule weighting where sampling is most uneven.
    """
    if not coarsened:
        # "of those INTERVALS at NATIVE RESOLUTION", not "of the frames". A
        # frame is an array on the rendering grid; an interval is a span of
        # time. The equal weighting is a fact about the second, and stating it
        # about the first invited a reader to check it by averaging their
        # download — which Phase 8 did, and got 1.876% instead of nothing.
        # ``_delta_section`` now reports that figure; this sentence goes back
        # to describing the reduction, which is what it is true of.
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
