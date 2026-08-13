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
    lines += _delta_section(render, recipe, cadence, chart.get("units"))
    return lines


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
    """What kind of relationship the frames have to the map, per tier."""
    if coarsened:
        return (
            f"Each frame averages {_count_phrase(recipe) or 'several '}{cadence} "
            "intervals, so the frames are a coarser temporal aggregation than "
            "the map, which is computed independently at native resolution."
        )
    return (
        f"Each frame is one {cadence} interval and the period map is their "
        "equally-weighted mean, so the frames and the map are the same "
        "temporal aggregation. They are not on the same grid: the frame planes "
        "are block-meaned to the rendering resolution and the map is not, and "
        "a block mean does not commute with an average over intervals wherever "
        "coverage is uneven."
    )


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
