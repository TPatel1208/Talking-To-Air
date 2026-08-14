"""
tests/test_methods_export_service.py
======================================
T10: golden tests for the deterministic methods-text generator. Canned
lineage/citation fixtures (the same shapes provenance_service produces from
the REAL MCP — nodes with prefix-derived kinds, events keyed
``event_type``/``detail``/``created_at``, citations as CMR records) go in;
the exact expected Markdown comes out — no LLM in the loop, so the same
session always yields the same text.
"""
import unittest


def _frames_chart(*, render: dict | None = None, spec: dict | None = None, **extra) -> dict:
    """A chart payload shaped exactly as ``_attach_frames`` writes it.

    Two blocks, written together in one jsonb row: ``payload["frames"]`` is
    ``frame_store._axis_block``'s output (the artifact), and
    ``payload["export"]["frames"]`` is the reproducibility recipe beside
    ``source_handles``/``variable``/``region_name``.
    """
    chart: dict = {"units": "molecules/cm^2", **extra}
    if render is not None:
        chart["frames"] = render
    if spec is not None:
        chart["export"] = {"frames": {"spec": spec, "exports": "period aggregate"}}
    return chart


def _cadence_tier_chart(**overrides) -> dict:
    """Tier one: a frame IS one of the product's own cadence buckets, and the
    period map is their average. Twelve hourly frames, all observed.

    ``coarsen_k`` is 5x5, deliberately — the regime a real regional retrieval
    lands in (Phase 8: 352,181 native cells), and the regime in which "the map
    is the mean of the frames" stops being true of the arrays themselves.
    """
    from tta_backend.preprocessing.frame_stack import FRAME_GRID_DELTA_BASIS

    render = {
        "cadence": "hourly",
        "tier": "cadence",
        "buckets_per_frame": 1,
        "coarsen_k": [5, 5],
        "cells_per_frame": 19107,
        "shape": [13, 99, 193],
        "delta": None,
        "frames": [
            {
                "t_start": f"2025-06-14T{hour:02d}:00:00",
                "t_end": f"2025-06-14T{hour + 1:02d}:00:00",
                "n_granules": 3,
                "valid_fraction": 0.938,
                "qa_pass_rate": 0.35,
                "statistics": {"count": 14124},
            }
            for hour in range(12)
        ],
        "pipeline_version": "t59-1",
    }
    spec = {
        "cadence": "hourly",
        "tier": "cadence",
        "n_frames": 12,
        "buckets_per_frame": 1,
        "max_frames": 60,
        "target_cells": 20000,
        "cells_per_frame": 19107,
        "coarsen_k": [5, 5],
        "boundary": "pad",
        "statistic": "mean",
        "dtype": "float32",
        "span": ["2025-06-14T00:00:00", "2025-06-14T12:00:00"],
        "pipeline_version": "t59-1",
    }
    # Phase 8's measurement on `map_2ea3dd7b34cf`: a cadence-tier chart has no
    # `delta` — its frames ARE its cadence buckets — and still misses the
    # period plane by 1.876% on the arrays it hands the browser, because the
    # block mean and the across-frame mean do not commute under partial
    # coverage. The tier with nothing to disclose was the tier stating the
    # flattest identity.
    render.setdefault("frame_grid_delta", {
        "headline": 0.018760,
        "max_abs": 2.72e15,
        "basis": FRAME_GRID_DELTA_BASIS,
    })
    render.update(overrides.pop("render", {}))
    spec.update(overrides.pop("spec", {}))
    return _frames_chart(render=render, spec=spec, **overrides)


def _coarsened_tier_chart(
    *, headline, max_abs, buckets_per_frame=5,
    grid_headline=0.0328, grid_max_abs=1.953e15, **overrides,
) -> dict:
    """Tier two: a frame averages several cadence buckets, so the frames are a
    DIFFERENT temporal aggregation from the map, which D14 computes
    independently. Phase 3 measured that disagreement at 22.3% and 5.4% on two
    real TEMPO bundles, and the 60-frame budget is reached at ~2.5 days — so
    this is the common case, not the edge.

    The two defaults for the shipped-array figure are Phase 8's measurement on
    `map_1531e35a0e18`, where the disclosed 3.74% sat beside arrays that
    disagreed by 3.28%: the block mean partly CANCELLED the coarsening term
    there rather than compounding it, so neither number bounds the other and
    the smaller one was the one on screen.
    """
    from tta_backend.preprocessing.frame_stack import (
        DELTA_BASIS, FRAME_GRID_DELTA_BASIS,
    )

    render = {
        "tier": "coarsened",
        "buckets_per_frame": buckets_per_frame,
        "delta": {"headline": headline, "max_abs": max_abs, "basis": DELTA_BASIS},
        "frame_grid_delta": {
            "headline": grid_headline,
            "max_abs": grid_max_abs,
            "basis": FRAME_GRID_DELTA_BASIS,
        },
        **overrides.pop("render", {}),
    }
    spec = {
        "tier": "coarsened",
        "buckets_per_frame": buckets_per_frame,
        **overrides.pop("spec", {}),
    }
    return _cadence_tier_chart(render=render, spec=spec, **overrides)


def _planes_chart(*, statistics=("mean", "max", "min"), planes=None, **overrides) -> dict:
    """A chart carrying D6a's extra planes, shaped as Phase 13/14 write them.

    Two blocks again, and they carry DIFFERENT plane facts on purpose. The
    spec's ``statistics`` is the recipe's list of what the axis can be browsed
    as (``_scrubbable_statistics``, derived from what landed); the per-plane
    disclosures — ``frame_grid_delta``, ``extent_overstatement``, ``value_range``
    — live only on the render block, the way the mean's delta already does.

    The defaults are Phase 11 G4/G5's live measurements: both selection
    identities bit-exact (``0.0``), and the max plane's spatial overstatement at
    24.7x pooled with a worst frame of 24.9x.
    """
    from tta_backend.preprocessing.frame_stack import (
        EXTENT_OVERSTATEMENT_BASIS, PLANE_AGREEMENT_BASIS,
    )

    def _plane(name: str, *, overstatement=None) -> dict:
        return {
            "url": f"/chart/map_1531e35a0e18/frames.{name}.f32.gz",
            "value_range": [5.7e14, 3.3e15],
            "frame_grid_delta": {
                "headline": 0.0,
                "max_abs": 0.0,
                "basis": PLANE_AGREEMENT_BASIS[name],
            },
            "extent_overstatement": overstatement,
        }

    if planes is None:
        planes = {
            "max": _plane("max", overstatement={
                "headline": 24.7,
                "worst_frame": 24.9,
                "ceiling": 25,
                "basis": EXTENT_OVERSTATEMENT_BASIS,
            }),
            "min": _plane("min"),
        }
    render = {"planes": planes, **overrides.pop("render", {})}
    spec = {"statistics": list(statistics), **overrides.pop("spec", {})}
    return _cadence_tier_chart(render=render, spec=spec, **overrides)


def _section(markdown: str, heading: str) -> str:
    """The body of one ``### heading`` section, up to the next heading."""
    body = markdown.split(f"### {heading}\n\n", 1)[1]
    return body.split("\n### ", 1)[0].rstrip("\n")


_LINEAGE = {"nodes": [{"handle": "dataset_tempo_no2", "kind": "dataset"}]}
_CITATIONS = [{"handle": "dataset_tempo_no2", "doi": "10.5067/TEMPO/NO2/L3"}]


def _markdown_for(chart: dict) -> str:
    from tta_backend.services.methods_export_service import build_methods_markdown

    return build_methods_markdown(
        artifact_title="TEMPO NO2 over New Jersey",
        aoi_description="New Jersey",
        time_window="2025-06-14",
        lineage=_LINEAGE,
        citations=_CITATIONS,
        chart=chart,
    )


class TemporalFramesDisclosureTests(unittest.TestCase):
    """T59 D12: the bucketing disclosure is mandatory, and it lands here rather
    than only in JSX for the same reason ``QA_PASS_RATE_BASIS`` exists — the
    reader of this document cannot see what was on screen."""

    def test_a_cadence_tier_chart_states_its_bucketing_grid_and_equal_weighting(self):
        markdown = _markdown_for(_cadence_tier_chart())

        self.assertEqual(
            _section(markdown, "Temporal frames"),
            "\n".join([
                "This figure also carries a browsable time axis of **12 hourly "
                "frames**, spanning 2025-06-14T00:00:00 to 2025-06-14T12:00:00.",
                "",
                # "of those intervals at native resolution", not "of the
                # frames". A frame is an array on the rendering grid and an
                # interval is a span of time; the equal weighting is a fact
                # about the second, and stating it about the first told the
                # reader they could check it by averaging a download. Phase 8
                # did exactly that and got 1.876%.
                "- Each frame is one hourly interval of this product, and the period "
                "map is the equally-weighted mean of those intervals at native "
                "resolution: every interval counts once, however many granules it "
                "holds.",
                "- Frames are rendered on a block-mean grid of 19,107 cells per "
                "frame — the mean over 5 × 5 native blocks, boundary `pad` — which "
                "is a rendering resolution, not a scientific product.",
                "- Empty intervals: none; all 12 hold at least one contributing "
                "granule.",
                "- Exports and downloads of this figure are the **period "
                "aggregate** at native resolution — never the frame grid, and "
                "never an individual frame.",
            ]),
        )

    def test_a_coarsened_frame_is_never_labelled_with_the_cadence_it_spans(self):
        # "12 hourly frames" is true in tier one, where a frame IS an hourly
        # interval, and false in tier two, where each frame averages five of
        # them — it would contradict the weighting bullet directly beneath it.
        markdown = _markdown_for(
            _coarsened_tier_chart(headline=0.0543, max_abs=3.39e15)
        )

        self.assertIn(
            "This figure also carries a browsable time axis of **12 frames**, "
            "spanning 2025-06-14T00:00:00 to 2025-06-14T12:00:00.",
            markdown,
        )
        self.assertNotIn("12 hourly frames", markdown)

    def test_a_coarsened_tier_counts_empty_frames_not_empty_intervals(self):
        # Each entry in the render block is one FRAME, and in tier two a frame
        # spans several cadence intervals. Calling the tally "intervals" there
        # would report 3 empty frames as 3 empty hours when they are 15.
        frames = [
            {
                "n_granules": 0 if i < 3 else 4,
                "qa_pass_rate": (0.0 if i < 1 else None) if i < 3 else 0.35,
            }
            for i in range(12)
        ]
        markdown = _markdown_for(
            _coarsened_tier_chart(
                headline=0.0543, max_abs=3.39e15, render={"frames": frames},
            )
        )

        self.assertIn(
            "- Empty frames: 3 of 12 — 1 observed but rejected entirely by "
            "quality screening, and 2 with nothing retrieved. An empty frame "
            "contributes nothing to the period mean and is not interpolated.",
            markdown,
        )
        self.assertNotIn("Empty intervals", markdown)

    def test_the_coarsened_tier_alone_is_enough_to_stop_calling_frames_intervals(self):
        # The noun must not depend on buckets_per_frame surviving: a coarsened
        # row that lost it would otherwise relapse into "hourly frames" and
        # "empty intervals", which is the same category error one field away.
        chart = _coarsened_tier_chart(headline=0.0543, max_abs=3.39e15)
        del chart["frames"]["buckets_per_frame"]
        del chart["export"]["frames"]["spec"]["buckets_per_frame"]

        markdown = _markdown_for(chart)

        self.assertIn("browsable time axis of **12 frames**", markdown)
        self.assertIn("- Empty frames: none;", markdown)
        self.assertNotIn("hourly frames", markdown)
        # And the count it lost is omitted, not printed as a quantity: "its 0
        # hourly intervals" and "averages 0 hourly intervals" are both false.
        self.assertIn("Each frame is the mean of its hourly intervals' means", markdown)
        self.assertIn("Each frame averages several hourly intervals", markdown)
        self.assertNotIn(" 0 hourly", markdown)

    def test_the_two_empty_interval_states_are_counted_apart_never_merged(self):
        # Phase 3 §4, measured: TEMPO only scans in daylight, so 28 of 54 hourly
        # buckets on a padded 2.2-day span hold no contributing granule. The two
        # empty states are different measurements and D10 requires them
        # distinguishable — ``qa_pass_rate=0.0`` is "observed, QA rejected all of
        # it"; ``qa_pass_rate=None`` is "nothing was retrieved". A single "28
        # gaps" tally would report a QA problem and a coverage problem as one
        # number.
        frames = [
            {
                "t_start": f"2025-06-14T{hour:02d}:00:00",
                "n_granules": 0 if hour < 28 else 2,
                "valid_fraction": 0.0 if hour < 28 else 0.9,
                # Three observed-then-rejected, twenty-five never retrieved.
                "qa_pass_rate": (0.0 if hour < 3 else None) if hour < 28 else 0.35,
                "statistics": {"count": 0 if hour < 28 else 14124},
            }
            for hour in range(54)
        ]
        chart = _cadence_tier_chart(
            render={"frames": frames}, spec={"n_frames": 54},
        )

        self.assertIn(
            "- Empty intervals: 28 of 54 — 3 observed but rejected entirely by "
            "quality screening, and 25 with nothing retrieved. An empty interval "
            "contributes nothing to the period mean and is not interpolated.",
            _markdown_for(chart),
        )

    def test_only_the_empty_state_that_occurred_is_named(self):
        # A chart whose gaps are all one kind must not imply the other happened.
        frames = [
            {
                "n_granules": 0 if hour < 4 else 2,
                "valid_fraction": 0.0 if hour < 4 else 0.9,
                "qa_pass_rate": None if hour < 4 else 0.35,
                "statistics": {},
            }
            for hour in range(12)
        ]
        markdown = _markdown_for(
            _cadence_tier_chart(render={"frames": frames}, spec={"n_frames": 12})
        )

        self.assertIn(
            "- Empty intervals: 4 of 12 — 4 with nothing retrieved. An empty "
            "interval contributes nothing to the period mean and is not "
            "interpolated.",
            markdown,
        )
        self.assertNotIn("quality screening", markdown)


class BrowsableStatisticsTests(unittest.TestCase):
    """T59 Phase 16 / D6a decisions 8 and 9: the planes leave the screen.

    A ``methods.md`` goes into supplementary material and is read by someone
    who was not sitting at the scrubber. Phase 15 gave the max plane four
    sentences on screen; a reader who browsed it, screenshotted it and cited
    the figure has none of them here.

    Its own section rather than bullets under ``### Temporal frames``, for the
    reason ``### Frame–map agreement`` has one: it carries a measured number
    that qualifies what the reader is about to cite. The lead sentence does the
    work of keeping it separate from the artifact — D12 is unchanged and the
    export bullet above stays exactly where it is.
    """

    def test_a_chart_with_planes_names_what_its_axis_can_be_browsed_as(self):
        markdown = _markdown_for(_planes_chart())

        self.assertIn(
            "This figure's time axis can also be browsed as a per-interval "
            "**maximum** and **minimum**. They are viewing modes: exports and "
            "downloads are unchanged.",
            _section(markdown, "Browsable statistics"),
        )

    def test_a_chart_that_never_had_planes_says_nothing_at_all(self):
        """Phase 13's reason for omitting the key rather than emitting it
        empty, arriving at the last consumer.

        Every chart plotted before Phase 14 is in this state, and this document
        is generated, archived and never diffed by a human — so a row written a
        year ago must render exactly as it rendered then. "Browsable
        statistics: none" would read as a finding about a feature the request
        never implied, which is ``_frames_section``'s third state and T57's
        omitted maturity, one level in.
        """
        markdown = _markdown_for(_cadence_tier_chart())

        self.assertNotIn("Browsable statistics", markdown)
        self.assertNotIn("maximum", markdown)
        # And the section it would have followed is untouched.
        self.assertIn("### Frame–map agreement", markdown)

    def test_a_row_whose_recipe_lists_only_the_mean_offers_no_selection_modes(self):
        # ``_scrubbable_statistics`` returns ["mean"] for a chart whose planes
        # never landed. A list is not a promise that it holds extras.
        markdown = _markdown_for(_planes_chart(statistics=("mean",), planes={}))

        self.assertNotIn("Browsable statistics", markdown)

    def test_each_plane_states_its_identity_and_prints_no_percentage_for_it(self):
        """D6a decision 8: one sentence stating the identity, and an EXACT
        equality rather than a tolerance in disguise.

        Both of the render block's figures belong to the mean — ``delta`` is
        its temporal disagreement and ``frame_grid_delta`` its shipped-array
        one — and a selection plane has neither. Passing the mean's 1.876%
        through here would attribute a mean-only disagreement to a plane that
        does not have it; passing the plane's own ``0.0`` through ``_pct``
        would print "under 0.1%", which D6a names as exactly the failure
        ``_DELTA_FLOOR`` exists to prevent.
        """
        section = _section(_markdown_for(_planes_chart()), "Browsable statistics")

        self.assertIn(
            "Taking the maximum of the stored frame planes reproduces the "
            "stored period plane exactly — a maximum selects one of the values "
            "it is given rather than combining them, so there is no "
            "disagreement here to measure.",
            section,
        )
        self.assertIn(
            "Taking the minimum of the stored frame planes reproduces the "
            "stored period plane exactly",
            section,
        )
        # No figure, of any formatting, for either plane's agreement. "0.0%"
        # claims a precision, "under 0.1%" claims a residual, and this is an
        # identity that holds exactly.
        self.assertNotIn("under 0.1%", section)
        self.assertNotIn("0.0%", section)

    def test_the_identity_is_asserted_from_the_payload_never_from_the_name(self):
        """A plane that ever ships a real disagreement reports the figure and
        says to read it as a defect.

        Claiming "exactly" because a maximum is associative *in theory* would
        be the same class of falsehood Phase 8 caught in ``frameDelta.js``:
        true of the mechanism, false of the arrays. The screen's
        ``selectionDelta`` branches on the payload for this reason, and this
        document must branch on the same fact rather than on the operation's
        name.
        """
        from tta_backend.preprocessing.frame_stack import PLANE_AGREEMENT_BASIS

        chart = _planes_chart()
        chart["frames"]["planes"]["max"]["frame_grid_delta"] = {
            "headline": 0.042, "max_abs": 8.1e14,
            "basis": PLANE_AGREEMENT_BASIS["max"],
        }
        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn(
            "Taking the maximum of the stored frame planes misses the stored "
            "period plane by **4.2%**, largest absolute difference 8.100e+14 "
            "molecules/cm^2 — which a maximum is not able to do, so read it as "
            "a defect in this figure rather than as a caveat on it.",
            section,
        )
        # The min plane, unchanged, still states its identity beside it.
        self.assertIn(
            "Taking the minimum of the stored frame planes reproduces the "
            "stored period plane exactly",
            section,
        )

    def test_each_plane_carries_its_own_basis_and_never_the_means(self):
        """``_figure_lines``' rule, applied to a claim rather than a figure.

        ``PLANE_AGREEMENT_BASIS`` is one sentence per plane because the word
        that changes carries the whole claim: a reader shown "mean of the
        stored frame planes" beside an identity produced by taking their
        MAXIMUM has been told something false about a real measurement — and
        quietly, because the max plane's number is zero under its own basis and
        would be large under the mean's, so the printed value would not even
        hint at the mismatch.
        """
        from tta_backend.preprocessing.frame_stack import (
            FRAME_GRID_DELTA_BASIS, PLANE_AGREEMENT_BASIS,
        )

        section = _section(_markdown_for(_planes_chart()), "Browsable statistics")

        self.assertIn(f"- Basis: {PLANE_AGREEMENT_BASIS['max']}.", section)
        self.assertIn(f"- Basis: {PLANE_AGREEMENT_BASIS['min']}.", section)
        # The mean's own basis belongs to the section above and is not reused
        # here — three planes under one account of themselves is the failure
        # the three separate basis strings exist to prevent.
        self.assertNotIn(f"- Basis: {FRAME_GRID_DELTA_BASIS}.", section)
        self.assertIn(
            f"- Basis: {FRAME_GRID_DELTA_BASIS}.",
            _section(_markdown_for(_planes_chart()), "Frame–map agreement"),
        )

    def test_a_plane_whose_identity_was_never_measured_says_so(self):
        """``_delta_section``'s rule, one plane in: unmeasured is not the same
        as agreeing, and a section built from what was measured rather than
        from what would be nice to say cannot claim "exactly" for a row that
        never checked."""
        chart = _planes_chart()
        chart["frames"]["planes"]["max"]["frame_grid_delta"] = None

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn(
            "Whether taking the maximum of the stored frame planes reproduces "
            "the stored period plane was not measured for this figure.",
            section,
        )
        self.assertNotIn("reproduces the stored period plane exactly — a maximum", section)

    def test_the_max_plane_discloses_the_ground_a_rendered_peak_claims(self):
        """Phase 11 G4, asked for in ``methods.md`` by name.

        With both of the max tier's deltas structurally zero, this one number
        carries all the weight the mean tier spreads over two — and it is the
        one a reader who screenshotted a peak into a slide has no other way to
        learn. ``EXTENT_OVERSTATEMENT_BASIS`` is the third basis string in
        ``frame_stack`` and until now had never been printed anywhere.
        """
        from tta_backend.preprocessing.frame_stack import EXTENT_OVERSTATEMENT_BASIS

        section = _section(_markdown_for(_planes_chart()), "Browsable statistics")

        self.assertIn(
            "- Every cell of a block is drawn at that block's highest value, "
            "so the ground shown at peak level is about **24.7×** the ground "
            "the peak was actually measured over. Its single worst frame "
            "stretched one further, to 24.9×.",
            section,
        )
        self.assertIn(f"- Basis: {EXTENT_OVERSTATEMENT_BASIS}.", section)

    def test_the_overstatement_is_this_chart_s_own_measurement_not_a_constant(self):
        # Phase 14's k=(2,2) chart measured 4.0000014 — a different figure from
        # Phase 11's ≈24.7×, and the document must be the chart's own.
        chart = _planes_chart()
        chart["frames"]["planes"]["max"]["extent_overstatement"].update(
            {"headline": 4.0000014, "worst_frame": 4.31, "ceiling": 4},
        )

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn("about **4.0×** the ground the peak was actually", section)
        self.assertIn("stretched one further, to 4.3×.", section)
        self.assertNotIn("24.7", section)

    def test_the_min_plane_is_given_no_overstatement_it_does_not_have(self):
        # ``extent_overstatement`` is None on every plane but max, by
        # construction: a block minimum paints a trough, not a peak, and D6a
        # decision 9 measures the peak. Silence, not a zero.
        section = _section(_markdown_for(_planes_chart()), "Browsable statistics")

        self.assertEqual(section.count("Every cell of a block is drawn"), 1)
        self.assertNotIn("lowest value, so the ground shown", section)

    def test_the_document_never_claims_the_overstatement_has_a_bound(self):
        """Phase 14's finding, addressed to this phase by name.

        ``ceiling`` is k² and is NOT an upper bound: both sums in
        ``_overstatement_terms`` are cos(latitude)-weighted per cell, so a
        block whose max sits on its lowest-weighted row contributes more than
        k², and the pooled figure straddles k² rather than approaching it from
        below. The fixture here is Phase 14's own straddling measurement —
        4.0000014 against a ceiling of 4 — so a document that rendered the
        ceiling would be printing a number this chart has already exceeded.
        """
        chart = _planes_chart()
        chart["frames"]["planes"]["max"]["extent_overstatement"].update(
            {"headline": 4.0000014, "worst_frame": 4.31, "ceiling": 4},
        )

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertNotIn("up to", section)
        self.assertNotIn("at most", section)
        self.assertNotIn("ceiling", section)
        # The ceiling itself, which would render as "4×" beside the measured
        # "4.0×" and the worst frame's "4.3×".
        self.assertNotIn("4×", section)


    def test_a_coarsened_frame_holds_the_peak_across_its_intervals_not_of_one(self):
        """The noun trap this project has now hit twice (``_gap_rule``,
        ``_weighting_rule``), arriving at a third sentence.

        In tier two a frame's value is the maximum across
        ``buckets_per_frame`` cadence intervals, not the maximum of one — and
        the wrong noun here is wrong in the direction that makes the plane
        sound more precise than it is, which is the same direction "13 of the
        48 intervals" was wrong in.
        """
        chart = _planes_chart(spec={"tier": "coarsened", "buckets_per_frame": 5},
                              render={"tier": "coarsened", "buckets_per_frame": 5})

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn(
            "- **Maximum**: each frame is the highest value across the 5 hourly "
            "intervals it spans, at each cell.",
            section,
        )
        self.assertNotIn("one hourly interval", section)

    def test_a_coarsened_row_that_lost_its_bucket_count_omits_it_rather_than_prints_zero(self):
        # ``_count_phrase``'s rule: "across the hourly intervals it spans",
        # never "across the 0 hourly intervals it spans".
        chart = _planes_chart(spec={"tier": "coarsened"}, render={"tier": "coarsened"})
        del chart["frames"]["buckets_per_frame"]

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn(
            "each frame is the highest value across the hourly intervals it "
            "spans, at each cell.",
            section,
        )
        self.assertNotIn(" 0 hourly", section)

    def test_a_chart_above_the_plane_ceiling_says_so_in_the_gates_own_words(self):
        """The middle state, which did not exist before Phase 14.

        ``_refusal_section``'s posture, one level in: this figure HAS a
        scrubber, so what is missing is the extra statistics rather than the
        axis, and a reader who finds neither a mode nor a reason is exactly the
        person ``_DISCLOSED_FRAME_REFUSALS`` exists to keep out of that
        position. Quoted, never paraphrased — a second account of why the
        planes are missing is a second account.
        """
        detail = (
            "A frame axis over this region reduces 4,200,000 cells per "
            "interval, above the 1,000,000-cell limit the additional max and "
            "min planes are built within; the mean scrubber is unaffected."
        )
        chart = _planes_chart(statistics=("mean",), planes={}, render={
            "planes_unavailable": {
                "reason": "plane_extent_too_large", "detail": detail,
            },
        })

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertEqual(
            section,
            "This figure's time axis is browsable as a period mean only. " + detail,
        )

    def test_a_plane_refusal_carrying_no_detail_still_names_what_is_missing(self):
        chart = _planes_chart(statistics=("mean",), planes={}, render={
            "planes_unavailable": {"reason": "plane_extent_too_large"},
        })

        self.assertEqual(
            _section(_markdown_for(chart), "Browsable statistics"),
            "This figure's time axis is browsable as a period mean only.",
        )


    def test_the_agreement_figures_above_are_scoped_to_the_mean_they_belong_to(self):
        """A hazard the document has and the screen does not.

        On screen the toggle REPLACES the delta disclosure: in max mode
        ``resolveFrameDelta`` branches before it touches either figure, so the
        mean's 1.876% is never on the page beside a max plane. This document
        has no modes — both sections are always present, and the agreement
        figures sit three lines above bullets about a different statistic with
        nothing saying whose they are.

        One clause, in this section rather than in that one: the figures above
        keep their single account of themselves, and the reader who arrives
        here after browsing a peak is told which statistic they described.
        """
        section = _section(_markdown_for(_planes_chart()), "Browsable statistics")

        self.assertIn(
            "The frame–map agreement figures above are the period mean's; each "
            "plane's own is stated with it.",
            section,
        )

    def test_a_row_with_no_agreement_section_is_not_told_to_scope_one(self):
        # Nothing above to misattribute, so the clause would be pointing at a
        # section that is not there — the same reason ``_delta_section`` stays
        # silent on a row that measured nothing.
        chart = _planes_chart(render={"frame_grid_delta": None})

        markdown = _markdown_for(chart)

        self.assertNotIn("Frame–map agreement", markdown)
        self.assertNotIn("figures above", markdown)
        # ...and the section itself is otherwise unchanged.
        self.assertIn("per-interval **maximum** and **minimum**", markdown)

    def test_the_statistics_are_the_ones_that_landed_not_the_ones_asked_for(self):
        """The same rule three consumers now share.

        ``store_frame_stack`` degrades one statistic at a time, so a plane
        whose write failed — or one evicted since — has a block entry and no
        key. The recipe's ``statistics`` is ``_scrubbable_statistics``' own
        output, derived from what landed, and naming a mode from the presence
        of a block would describe a toggle the reader watches 404.
        """
        chart = _planes_chart(statistics=("mean", "max"))

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn("per-interval **maximum**.", section)
        self.assertNotIn("minimum", section)

    def test_a_row_with_no_recipe_falls_back_to_the_planes_that_have_urls(self):
        # A row written without a spec block still has to disclose what it
        # holds, and the render block's urls are the same witness under the
        # same rule — the fallback ``_recipe``'s precedence already applies to
        # every field the two share.
        chart = _planes_chart()
        del chart["export"]["frames"]["spec"]
        del chart["frames"]["planes"]["min"]["url"]

        section = _section(_markdown_for(chart), "Browsable statistics")

        self.assertIn("per-interval **maximum**.", section)
        self.assertNotIn("minimum", section)

    def test_the_export_sentence_survives_verbatim_and_stays_unambiguous(self):
        """D12, and the one thing tension 1 was not allowed to blur.

        A plane is browsable and is not downloadable, in any phase. The export
        bullet keeps its exact wording and its exact place in ``### Temporal
        frames``, and the new section states the same fact in its own lead
        rather than qualifying that one.
        """
        markdown = _markdown_for(_planes_chart())

        self.assertIn(
            "- Exports and downloads of this figure are the **period "
            "aggregate** at native resolution — never the frame grid, and "
            "never an individual frame.",
            _section(markdown, "Temporal frames"),
        )
        self.assertEqual(markdown.count("Exports and downloads of this figure"), 1)
        self.assertLess(
            markdown.index("### Temporal frames"),
            markdown.index("### Browsable statistics"),
        )
        self.assertLess(
            markdown.index("### Browsable statistics"),
            markdown.index("### References"),
        )

    def test_the_whole_section_reads_as_one_document_on_a_real_shaped_chart(self):
        """The golden, for ``TemporalFramesDisclosureTests``' reason: every
        line of this is prose a stranger reads a year from now, and a
        by-fragment test lets two of them drift into contradicting each
        other."""
        from tta_backend.preprocessing.frame_stack import (
            EXTENT_OVERSTATEMENT_BASIS, PLANE_AGREEMENT_BASIS,
        )

        markdown = _markdown_for(_planes_chart())

        self.assertEqual(
            _section(markdown, "Browsable statistics"),
            "\n".join([
                "This figure's time axis can also be browsed as a per-interval "
                "**maximum** and **minimum**. They are viewing modes: exports "
                "and downloads are unchanged. The frame–map agreement figures "
                "above are the period mean's; each plane's own is stated with "
                "it.",
                "",
                "- **Maximum**: each frame is the highest value one hourly "
                "interval of this product holds at each cell. Taking the "
                "maximum of the stored frame planes reproduces the stored "
                "period plane exactly — a maximum selects one of the values it "
                "is given rather than combining them, so there is no "
                "disagreement here to measure.",
                f"- Basis: {PLANE_AGREEMENT_BASIS['max']}.",
                "- Every cell of a block is drawn at that block's highest "
                "value, so the ground shown at peak level is about **24.7×** "
                "the ground the peak was actually measured over. Its single "
                "worst frame stretched one further, to 24.9×.",
                f"- Basis: {EXTENT_OVERSTATEMENT_BASIS}.",
                "- **Minimum**: each frame is the lowest value one hourly "
                "interval of this product holds at each cell. Taking the "
                "minimum of the stored frame planes reproduces the stored "
                "period plane exactly — a minimum selects one of the values it "
                "is given rather than combining them, so there is no "
                "disagreement here to measure.",
                f"- Basis: {PLANE_AGREEMENT_BASIS['min']}.",
            ]),
        )

    def test_this_sections_vocabulary_covers_every_statistic_the_backend_builds(self):
        """The fourth pairing of this kind, and the reason it is a test rather
        than a comment.

        ``_SELECTION_NOUNS`` is not merely a label table: the identity sentence
        beside each noun asserts that the statistic SELECTS one of the values
        it is given, which is true of a maximum and a minimum and false of a
        mean, a median over an even count, or a percentile. So a statistic
        added to ``PLANE_STATISTICS`` cannot be given prose here by default —
        and it must not be dropped from the document in silence either, which
        is what an unrecognized key would otherwise be.

        Failing here is the intended outcome of adding one: the new plane needs
        a sentence someone has decided is true of it.
        """
        from tta_backend.preprocessing.frame_stack import PLANE_STATISTICS
        from tta_backend.services.methods_export_service import (
            _SELECTION_NOUNS, _SELECTION_VERBS,
        )

        self.assertEqual(
            {"mean", *_SELECTION_NOUNS},
            set(PLANE_STATISTICS),
            "A statistic was added to PLANE_STATISTICS without prose in "
            "methods_export_service. Its identity sentence claims the "
            "statistic selects one of its inputs — decide whether that is true "
            "of the new one rather than letting it drop out of the document.",
        )
        self.assertEqual(set(_SELECTION_NOUNS), set(_SELECTION_VERBS))

    def test_malformed_plane_blocks_degrade_to_blander_lines_never_a_crash(self):
        # ``api.py``'s broad except turns any surprise here into a generic 500
        # (the QA 2026-07-17 blocker), so a shape bug becomes invisible rather
        # than loud. Every one of these is a row a degraded write can produce.
        for planes in (
            "not a dict",
            {"max": "not a dict"},
            {"max": {}},
            {"max": {"frame_grid_delta": "no"}},
            {"max": {"extent_overstatement": "no"}},
            {"max": {"extent_overstatement": {"headline": float("nan")}}},
            {"max": {"extent_overstatement": {"headline": 24.7, "worst_frame": "?"}}},
            {"unknown_statistic": {"url": "/x", "frame_grid_delta": None}},
        ):
            with self.subTest(planes=planes):
                markdown = _markdown_for(
                    _planes_chart(planes=planes, statistics=("mean", "max", "min"))
                )
                self.assertIn("### References", markdown)
                self.assertNotIn("None", markdown)


class SharedPlaneClaimTests(unittest.TestCase):
    """T59 Phase 16 tension 2: two languages now have to agree about a
    SENTENCE, and every tool this repo has for that problem takes a number.

    ``_DELTA_HIGH`` ↔ ``severityOf``'s high edge and ``_DELTA_FLOOR`` ↔
    ``formatPct``'s floor are constants, and a constant can be read out of the
    other language and compared. Prose cannot: a test asserting the document
    and the screen hold equal strings would be a test forbidding the document
    from reading like a document, and ``_figure_lines``' own rule — a basis is
    quoted, never paraphrased — is about a *quoted* string, which is exactly
    what these sentences are not.

    **So what is pinned here is the CLAIM inside each sentence, not its
    wording**, from both sides:

    * neither language may render a percentage for a selection plane's
      agreement, because the identity holds exactly;
    * neither may claim a bound on the overstatement (Phase 14);
    * both must take the overstatement from the payload's measured figure,
      which the document's own two-fixture test proves for this side.

    Read through the same bind mount ``SharedDeltaThresholdTests`` documents
    and asserts — ``frameStatistic.js`` is inside the mount that already
    serves ``frameDelta.js``, so no compose change was needed. Each pattern
    below was verified to BITE by mutating the JS source in memory and
    confirming the assertion fails; a contract test that cannot fail is worse
    than none, and this repo has measured that exact thing happening.
    """

    def _js_source(self, filename: str) -> str:
        import os

        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../Frontend/src/utils", filename)
        )
        self.assertTrue(
            os.path.isfile(path),
            f"{filename} not found at {path} — the shared plane claims cannot "
            "be checked, which is the same as not checking them.",
        )
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _function_body(self, source: str, name: str) -> str:
        """One exported function's source, so a pattern found anywhere in the
        file cannot stand in for one found in the function that makes the
        claim. ``frameDelta.js`` legitimately says "up to" about a worst
        PIXEL; this is about a bound on the overstatement."""
        start = source.index(f"export function {name}(")
        rest = source[start:]
        end = rest.find("\nexport ", 1)
        return rest if end == -1 else rest[:end]

    def test_neither_language_claims_a_bound_on_the_overstatement(self):
        note = self._function_body(
            self._js_source("frameStatistic.js"), "extentOverstatementNote",
        )

        for forbidden in ("up to", "at most", "ceiling"):
            with self.subTest(forbidden=forbidden):
                # The comment block above the function names both phrases in
                # order to forbid them, so only the rendered string is checked.
                rendered = note.split("return (", 1)[1]
                self.assertNotIn(forbidden, rendered)

    def test_the_screen_takes_the_overstatement_from_the_payload_too(self):
        # The document's version of this claim is asserted by feeding two
        # fixtures and getting two numbers. The screen's cannot be run from
        # here, so what is checked is that the figure comes off the payload at
        # all rather than out of a literal.
        note = self._function_body(
            self._js_source("frameStatistic.js"), "extentOverstatementNote",
        )

        self.assertIn("extent_overstatement", note)
        self.assertIn("headline.toFixed(1)", note)
        self.assertNotIn("24.7", note)

    def test_the_screens_identity_sentence_prints_no_percentage_either(self):
        """``_pct`` and ``formatPct`` draw the same floor, so a selection
        plane's ``0.0`` would print "under 0.1%" in both languages — D6a's own
        example of the failure ``_DELTA_FLOOR`` exists to prevent. This
        document's side is asserted above; this is the screen's.
        """
        source = self._js_source("frameDelta.js")
        exact = next(
            line for line in source.splitlines()
            if "reproduces the period plane at stop 0 exactly" in line
        )

        self.assertNotIn("pct", exact)
        self.assertNotIn("formatPct", exact)

    def test_the_compose_file_mounts_the_js_sources_into_the_test_service(self):
        # The same assertion ``SharedDeltaThresholdTests`` makes, made again
        # for the second file that now depends on the mount: without it, every
        # check in this class fails on a missing file rather than on a real
        # drift, and a reader of that failure has to work out which it is.
        import os

        import yaml

        compose_path = "/compose/docker-compose.yml"
        if not os.path.exists(compose_path):
            self.skipTest(
                f"{compose_path} not mounted (run via docker compose backend-test)"
            )
        with open(compose_path, "r", encoding="utf-8") as handle:
            compose = yaml.safe_load(handle)

        volumes = compose["services"]["backend-test"].get("volumes") or []
        self.assertIn(
            "./Frontend/src/utils:/Frontend/src/utils:ro",
            [str(entry) for entry in volumes],
        )


class SharedDeltaThresholdTests(unittest.TestCase):
    """T59 Phase 8. One number, two languages: ``_DELTA_HIGH`` here and
    ``severityOf``'s ``high`` edge in ``frameDelta.js``.

    Until Phase 8 the only thing joining them was a comment naming the other
    file, and a comment is not a check — measured 2026-08-11 by editing the JS
    edge to 0.15 and watching all 339 frontend tests and all 24 tests in this
    file stay green. The screen resolves three severity bands and this document
    two, so only the ``high`` edge is shared; that edge is what this reads out
    of the JS source and compares.

    The JS file lives outside ``./Backend``, this image's build context, so
    ``Frontend/src/utils`` is bind-mounted into the ``backend-test`` service at
    the path the repo-relative lookup already resolves to — the ``.coveragerc``
    / ``docker-compose.yml`` / ``init_agent_charts.sql`` precedent, and the same
    resolution ``test_jobs_service.FinishedRowStatusContractTests`` uses for
    ``jobCard.js``. One mount serves both; a skip when it is absent would let
    that mount disappear silently, so the mount itself is asserted below,
    against the compose file that declares it.
    """

    #: Resolved relative to this file, so the same expression works on the host
    #: and — via the bind mount — inside the test container.
    REPO_PATH = "../../Frontend/src/utils/frameDelta.js"
    COMPOSE_PATH = "/compose/docker-compose.yml"

    def _js_source(self) -> str:
        import os

        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), self.REPO_PATH)
        )
        self.assertTrue(
            os.path.isfile(path),
            f"frameDelta.js not found at {path} — the shared delta threshold "
            "cannot be checked, which is the same as not checking it.",
        )
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def test_the_high_edge_is_the_same_number_the_screen_uses(self):
        import re

        from tta_backend.services.methods_export_service import _DELTA_HIGH

        source = self._js_source()
        match = re.search(
            r"if\s*\(\s*headline\s*>=\s*([0-9.]+)\s*\)\s*return\s*'high'", source,
        )
        self.assertIsNotNone(
            match,
            "severityOf's 'high' edge could not be found in frameDelta.js. If "
            "the function was renamed or restructured, this check must be "
            "re-pointed rather than deleted — the two constants are still two.",
        )
        self.assertEqual(
            float(match.group(1)),
            _DELTA_HIGH,
            "methods_export_service._DELTA_HIGH and frameDelta.js:severityOf's "
            "'high' edge have drifted apart. They are the one threshold the "
            "document and the screen share; two values means a delta the "
            "scrubber calls high that the methods document states plainly.",
        )

    def test_the_reporting_floor_is_the_same_number_the_screen_uses(self):
        """Phase 9's second shared constant, pinned the moment it appeared.

        Both languages refuse to print a percentage below 0.1% — an
        undownsampled chart's arrays agree to float32 storage noise, and "0.0%"
        overstates that while reading like a measurement that found nothing.
        Two floors would put "0.0%" in the document beside "under 0.1%" on the
        screen for one chart. Less severe than a drifted ``_DELTA_HIGH``, and
        exactly the same failure: the repo now has the idiom, so the second
        instance is pinned rather than commented.
        """
        import re

        from tta_backend.services.methods_export_service import (
            _DELTA_FLOOR, _DELTA_FLOOR_TEXT,
        )

        source = self._js_source()
        match = re.search(
            r"headline\s*>=\s*([0-9.]+)\s*\?\s*`\$\{\(headline", source,
        )
        self.assertIsNotNone(
            match,
            "formatPct's floor could not be found in frameDelta.js. If the "
            "function was renamed or restructured, re-point this check rather "
            "than delete it — the two constants are still two.",
        )
        self.assertEqual(float(match.group(1)), _DELTA_FLOOR)
        self.assertIn(f"'{_DELTA_FLOOR_TEXT}'", source)

    def test_the_compose_file_mounts_the_js_sources_into_the_test_service(self):
        # Without the mount, every Python-reads-JS check in this repo fails on a
        # missing file rather than on a real drift, and a reader of that failure
        # has to work out which of the two it is. That is not hypothetical:
        # FinishedRowStatusContractTests was written against jobCard.js and the
        # mount was never added, so it had never once run in the container.
        import os

        import yaml

        if not os.path.exists(self.COMPOSE_PATH):
            self.skipTest(
                f"{self.COMPOSE_PATH} not mounted (run via docker compose backend-test)"
            )
        with open(self.COMPOSE_PATH, "r", encoding="utf-8") as handle:
            compose = yaml.safe_load(handle)

        volumes = compose["services"]["backend-test"].get("volumes") or []
        self.assertIn(
            "./Frontend/src/utils:/Frontend/src/utils:ro",
            [str(entry) for entry in volumes],
        )


class FrameMapAgreementTests(unittest.TestCase):
    """T59 D16 / Risk 4. Markdown has no colour and no border, so the only
    hierarchy available is placement and weight: the delta gets its own section
    above the References — the T57 maturity block's treatment, because it
    likewise qualifies everything the reader is about to cite — and a
    double-digit disagreement additionally gets a bolded lead sentence, so it
    cannot render as the same quiet footnote a sub-1% one does."""

    def test_a_double_digit_delta_leads_the_section_in_bold_above_the_references(self):
        # job_52a95bb4cb79e2ee, measured 2026-08-10: headline 22.27%,
        # max|F-M| 9.58e16 molecules/cm², at 5 hourly buckets per frame.
        from tta_backend.preprocessing.frame_stack import (
            DELTA_BASIS, FRAME_GRID_DELTA_BASIS,
        )

        markdown = _markdown_for(
            _coarsened_tier_chart(headline=0.2227, max_abs=9.58e16)
        )

        self.assertEqual(
            _section(markdown, "Frame–map agreement"),
            "\n".join([
                "**The frames and the period map are answering materially "
                "different questions: they disagree by 22.3% of the map's own "
                "magnitude.** Each frame averages 5 hourly intervals, so the "
                "frames are a coarser temporal aggregation than the map, which "
                "is computed independently at native resolution.",
                "",
                "- Disagreement: **22.3%**, largest absolute difference "
                "9.580e+16 molecules/cm^2.",
                f"- Basis: {DELTA_BASIS}.",
                "- Averaging the frame planes and comparing them to the period "
                "plane beside them: **3.3%**, largest absolute difference "
                "1.953e+15 molecules/cm^2.",
                f"- Basis: {FRAME_GRID_DELTA_BASIS}.",
                # Phase 8 §2, in one line, because a reader given two numbers
                # will otherwise assume the larger one contains the smaller.
                "- The two are measured on different arrays and neither bounds "
                "the other: the block mean can cancel part of the coarsening as "
                "easily as compound it.",
            ]),
        )
        self.assertLess(
            markdown.index("### Frame–map agreement"),
            markdown.index("### References"),
        )

    def test_the_shipped_figure_is_never_labelled_with_the_native_basis(self):
        """Phase 8 §2's mirror gap: tier two disclosed 3.74% under a basis
        reading "at native resolution" while the arrays on screen disagreed by
        3.28%. Two quantities under one account of themselves is the same
        failure ``DELTA_BASIS`` was written to prevent, wearing the other
        shoe."""
        from tta_backend.preprocessing.frame_stack import (
            DELTA_BASIS, FRAME_GRID_DELTA_BASIS,
        )

        markdown = _markdown_for(
            _coarsened_tier_chart(headline=0.0374, max_abs=3.127e15)
        )
        section = _section(markdown, "Frame–map agreement")

        native, shipped = section.index("**3.7%**"), section.index("**3.3%**")
        self.assertLess(native, shipped)
        # Each figure's basis is the line under that figure, and they are not
        # the same line.
        self.assertLess(native, section.index(DELTA_BASIS))
        self.assertLess(section.index(DELTA_BASIS), shipped)
        self.assertLess(shipped, section.index(FRAME_GRID_DELTA_BASIS))

    def test_a_single_digit_delta_is_stated_plainly_without_the_bolded_lead(self):
        # job_c1122dfd051c15ee, measured 2026-08-10: 5.43% at 4 buckets/frame.
        # Still its own section — D16 gates nothing, and whether 5.4% matters is
        # the reader's judgment, not this document's.
        markdown = _markdown_for(
            _coarsened_tier_chart(
                headline=0.0543, max_abs=3.39e15, buckets_per_frame=4,
            )
        )

        self.assertEqual(
            _section(markdown, "Frame–map agreement").split("\n")[0],
            "Each frame averages 4 hourly intervals, so the frames are a "
            "coarser temporal aggregation than the map, which is computed "
            "independently at native resolution.",
        )
        self.assertIn(
            "- Disagreement: **5.4%**, largest absolute difference "
            "3.390e+15 molecules/cm^2.",
            markdown,
        )

    def test_the_bolded_lead_turns_on_at_the_same_edge_the_screen_uses(self):
        # frameDelta.js:severityOf returns 'high' at `headline >= 0.1`. The
        # screen resolves three bands and this document two, so this is the one
        # edge they share — and it is shared by value, in two places, which is
        # what this test exists to catch.
        bolded = "**The frames and the period map are answering"

        self.assertIn(
            bolded,
            _markdown_for(_coarsened_tier_chart(headline=0.10, max_abs=1.0)),
        )
        self.assertNotIn(
            bolded,
            _markdown_for(_coarsened_tier_chart(headline=0.0999, max_abs=1.0)),
        )

    def test_the_loud_treatment_follows_the_larger_disagreement_whichever_it_is(self):
        """One escalation rule, over everything disclosed. Keying the bold off
        the native figure alone would leave the tier that has no native figure
        unable to raise its voice at all — and tier one is exactly where a
        double-digit block-mean disagreement would be most surprising, because
        the temporal aggregation there is an exact identity."""
        markdown = _markdown_for(
            _cadence_tier_chart(
                render={"frame_grid_delta": {
                    "headline": 0.14, "max_abs": 9.0e15, "basis": "…",
                }},
            )
        )

        self.assertIn(
            "**Averaging this figure's frame planes does not reproduce the "
            "period plane: they differ by 14.0% of the map's own magnitude.**",
            markdown,
        )

    def test_a_coarsened_row_below_the_edge_natively_still_shouts_for_its_arrays(self):
        """The two figures are independent, so the escalation cannot be read
        off either one alone. Phase 8 measured them going in opposite
        directions on the same chart."""
        bolded = "**The frames and the period map are answering"

        self.assertIn(
            bolded,
            _markdown_for(
                _coarsened_tier_chart(
                    headline=0.02, max_abs=1.0, grid_headline=0.31,
                    grid_max_abs=4.0e15,
                )
            ),
        )

    def test_the_cadence_tier_reports_what_averaging_its_own_download_gives(self):
        """The Phase 8 finding, in the document that stated it flattest.

        A cadence-tier frame IS a cadence bucket, so there is no temporal
        disagreement and ``delta`` is rightly absent — but the frames a reader
        can download have been block-meaned to the rendering grid and the map
        has not, and those two reductions do not commute where coverage is
        uneven. On the chart this fixture is taken from that is 1.9%, with a
        worst pixel most of the way across the map's own colour ramp.
        """
        from tta_backend.preprocessing.frame_stack import FRAME_GRID_DELTA_BASIS

        markdown = _markdown_for(_cadence_tier_chart())

        self.assertEqual(
            _section(markdown, "Frame–map agreement"),
            "\n".join([
                "Each frame is one hourly interval and the period map is their "
                "equally-weighted mean, so the frames and the map are the same "
                "temporal aggregation. They are not on the same grid: the frame "
                "planes are block-meaned to the rendering resolution and the map "
                "is not, and a block mean does not commute with an average over "
                "intervals wherever coverage is uneven.",
                "",
                "- Averaging the frame planes and comparing them to the period "
                "plane beside them: **1.9%**, largest absolute difference "
                "2.720e+15 molecules/cm^2.",
                f"- Basis: {FRAME_GRID_DELTA_BASIS}.",
            ]),
        )
        # The temporal identity is still stated, and still true — of the
        # intervals, at the resolution it is computed at.
        self.assertIn(
            "the period map is the equally-weighted mean of those intervals at "
            "native resolution",
            markdown,
        )

    def test_an_undownsampled_chart_reports_the_floor_rather_than_a_bare_zero(self):
        """At k=(1,1) the block mean is the identity function and the two
        arrays agree to float32 storage noise — Phase 3's 0.000002%. Printing
        "0.0%" would overstate that to a precision the document does not have,
        and printing nothing would drop the one tier where the check passes."""
        markdown = _markdown_for(
            _cadence_tier_chart(
                render={
                    "coarsen_k": [1, 1],
                    "frame_grid_delta": {
                        "headline": 2e-8, "max_abs": 1.5e8, "basis": "…",
                    },
                },
                spec={"coarsen_k": [1, 1]},
            )
        )

        self.assertIn(
            "- Averaging the frame planes and comparing them to the period plane "
            "beside them: **under 0.1%**",
            markdown,
        )

    def test_an_undownsampled_chart_is_never_told_its_frames_were_block_meaned(self):
        """Found on live data 2026-08-13, in a sentence written by this phase.

        A New Jersey retrieval is 10,624 native cells — under the 20,000-cell
        ceiling — so `coarsen_k` is (1,1) and there is **no block mean at all**.
        The tier-one lead nonetheless said "They are not on the same grid: the
        frame planes are block-meaned to the rendering resolution and the map
        is not", which is a mechanism that does not exist for this figure.
        Small regions are not a corner case; they are most of them.

        Exactly the failure this phase was written to remove — a sentence
        stating something about the arrays that the arrays do not satisfy — so
        it is pinned rather than patched.
        """
        # BOTH blocks, because `_attach_frames` writes them together and the
        # precedence rule lets the spec win — a fixture that moves only the
        # render block is a row that cannot occur, and it silently kept the
        # 5×5 recipe this test is about.
        markdown = _markdown_for(
            _cadence_tier_chart(
                render={
                    "coarsen_k": [1, 1],
                    "frame_grid_delta": {
                        "headline": 2.2915e-8, "max_abs": 6.6076e8, "basis": "…",
                    },
                },
                spec={"coarsen_k": [1, 1]},
            )
        )
        section = _section(markdown, "Frame–map agreement")

        self.assertNotIn("block-mean", section)
        self.assertNotIn("not on the same grid", section)
        self.assertIn("these frames are not downsampled", section)
        # And the residual is named for what it actually is at k=(1,1): the one
        # remaining difference is that the frames were narrowed to float32
        # individually and the map was narrowed after averaging.
        self.assertIn("float32 storage precision", section)

    def test_a_row_predating_the_shipped_array_measurement_stays_silent(self):
        """A chart written before Phase 9 has no such measurement, and the
        section is built from what was measured rather than from what would be
        nice to say. The scoped weighting sentence carries that row on its own:
        it is true at native resolution whether or not anyone checked the
        arrays."""
        markdown = _markdown_for(
            _cadence_tier_chart(render={"frame_grid_delta": None})
        )

        self.assertNotIn("Frame–map agreement", markdown)
        self.assertIn("equally-weighted mean of those intervals", markdown)

    def test_a_coarsened_tier_with_no_measured_delta_says_so(self):
        # Coarsened but unmeasured is not the same as agreeing. Omitting the
        # section entirely would let a reader assume the frames and the map are
        # the same aggregation.
        markdown = _markdown_for(_coarsened_tier_chart(headline=None, max_abs=None))

        self.assertIn(
            "The size of that disagreement was not measured for this figure.",
            markdown,
        )
        self.assertNotIn("Disagreement: **", markdown)
        # ...and the figure that WAS measured is still reported. One missing
        # measurement must not suppress the other, or a row that lost the
        # native term would silently lose the arrays term with it.
        self.assertIn(
            "- Averaging the frame planes and comparing them to the period plane "
            "beside them: **3.3%**",
            markdown,
        )
        # The "neither bounds the other" line needs two numbers to be about.
        self.assertNotIn("neither bounds the other", markdown)


class ExportLabelTests(unittest.TestCase):
    """T59 D12: export is unchanged and labeled "period aggregate". The label
    already ships in the jsonb row for a programmatic consumer; this document
    is where the human reader — who cannot see the screen and is about to put
    the download in a paper — learns which field they actually have."""

    def test_the_export_label_is_taken_from_the_row_not_restated(self):
        markdown = _markdown_for(_cadence_tier_chart())

        self.assertIn(
            "- Exports and downloads of this figure are the **period "
            "aggregate** at native resolution — never the frame grid, and "
            "never an individual frame.",
            markdown,
        )

    def test_a_row_that_does_not_state_its_export_label_is_not_given_one(self):
        # The bullet asserts what export DOES. A row that does not say must not
        # have the claim invented on its behalf.
        chart = _cadence_tier_chart()
        del chart["export"]["frames"]["exports"]

        markdown = _markdown_for(chart)

        self.assertNotIn("Exports and downloads", markdown)
        self.assertIn("### Temporal frames", markdown)


class FramesRefusalTests(unittest.TestCase):
    """Three refusals are disclosed rather than swallowed, because only one of
    them is fixed by narrowing the map. ``cadence_unknown`` earns it most: an
    unregistered product's cadence is also the condition under which
    ``_cadence_weighted_mean`` degrades to a plain unweighted mean, so it is a
    fact about the map itself, not only about the missing scrubber."""

    def _refused(self, reason: str, detail: str) -> str:
        disclosure = {"reason": reason, "detail": detail}
        return _markdown_for({
            "units": "molecules/cm^2",
            "frames_unavailable": disclosure,
            "export": {"frames": {"unavailable": disclosure}},
        })

    def test_an_unknown_cadence_refusal_is_quoted_not_paraphrased(self):
        # Verbatim from frame_gate: paraphrasing a refusal here would create a
        # second account of why the axis is missing.
        detail = (
            "This product's cadence is 'unknown', so a frame axis has no "
            "interval width to use; frames need one of hourly, daily, monthly."
        )

        self.assertEqual(
            _section(self._refused("cadence_unknown", detail), "Temporal frames"),
            "No frame-by-frame time axis was built for this figure. " + detail,
        )

    def test_a_span_refusal_is_disclosed_with_its_own_detail(self):
        detail = (
            "This span covers 9,000 hourly intervals, above the 1,440 a frame "
            "axis is built within -- past it each of the 60 frames would "
            "average more than 24 intervals together."
        )

        self.assertIn(detail, self._refused("span_too_long", detail))

    def test_an_extent_refusal_is_disclosed_with_its_own_detail(self):
        detail = (
            "A frame axis over this region would reduce 17,024,450 cells per "
            "interval, above the 4,000,000-cell limit a frame stack is built "
            "within."
        )

        self.assertIn(detail, self._refused("extent_too_large", detail))

    def test_a_refusal_carrying_no_detail_still_names_that_none_was_built(self):
        self.assertEqual(
            _section(self._refused("span_too_long", ""), "Temporal frames"),
            "No frame-by-frame time axis was built for this figure.",
        )


class FramesShapeDefensivenessTests(unittest.TestCase):
    """``api.py``'s broad except converts any surprise in here into a generic
    contract error (the QA 2026-07-17 blocker), which means a shape bug becomes
    an invisible 500 rather than a loud one. So the shapes are tested."""

    def test_a_chart_with_no_frames_and_no_refusal_says_nothing_at_all(self):
        # The third state, and the one the frontend also keeps silent — but for
        # a reason of its own: frames are additive (D15), so the map is
        # identical whether one was built or not, and their absence says
        # nothing about the figure being cited. "Temporal frames: none" would
        # read as a finding, the way T57's omitted maturity would.
        markdown = _markdown_for({"units": "molecules/cm^2"})

        self.assertNotIn("Temporal frames", markdown)
        self.assertNotIn("Frame–map agreement", markdown)

    def test_no_chart_at_all_is_the_pre_t59_call_and_is_unchanged(self):
        from tta_backend.services.methods_export_service import build_methods_markdown

        markdown = build_methods_markdown(
            artifact_title="No chart", aoi_description="somewhere",
            time_window="unknown", lineage=_LINEAGE, citations=_CITATIONS,
        )

        self.assertNotIn("Temporal frames", markdown)
        self.assertIn("### References", markdown)

    def test_a_row_with_no_recipe_block_still_discloses_what_the_artifact_knows(self):
        # The spec is authoritative for every field it carries — but a row that
        # has none must fall back to the render block for the fields that DO
        # live in both, uniformly, rather than losing whole lines. Only the
        # spec-only fields (boundary, span) go missing.
        chart = _cadence_tier_chart()
        del chart["export"]["frames"]["spec"]

        markdown = _markdown_for(chart)

        self.assertIn("browsable time axis of **12 hourly frames**.", markdown)
        self.assertIn(
            "- Frames are rendered on a block-mean grid of 19,107 cells per "
            "frame — the mean over 5 × 5 native blocks — which is a rendering "
            "resolution, not a scientific product.",
            markdown,
        )
        self.assertNotIn("boundary", markdown)
        self.assertNotIn("spanning", markdown)

    def test_malformed_frames_blocks_degrade_to_blander_lines_never_a_crash(self):
        from tta_backend.services.methods_export_service import build_methods_markdown

        for chart in (
            {"frames": "not a dict"},
            {"frames": {"frames": [None, 7, {}]}},
            {"frames": {"frames": [{}]}, "export": "not a dict"},
            {"frames": {"frames": [{}]}, "export": {"frames": {"spec": []}}},
            {
                "frames": {"frames": [{}], "tier": "coarsened", "delta": "no"},
                "export": {"frames": {"spec": {"coarsen_k": [None, "x"]}}},
            },
            {
                "frames": {
                    "frames": [{"n_granules": "?", "qa_pass_rate": "?"}],
                    "tier": "coarsened",
                    "delta": {"headline": float("nan"), "max_abs": None},
                },
            },
            {"frames_unavailable": {"reason": "span_too_long"}},
            {"frames_unavailable": "not a dict"},
        ):
            with self.subTest(chart=chart):
                markdown = build_methods_markdown(
                    artifact_title="Odd frames", aoi_description="somewhere",
                    time_window="unknown", lineage=_LINEAGE,
                    citations=_CITATIONS, chart=chart,
                )
                self.assertIn("## Methods — Odd frames", markdown)
                self.assertIn("### References", markdown)


class BuildMethodsMarkdownTests(unittest.TestCase):
    def test_a_single_dataset_map_artifact_renders_the_full_golden_methods_text(self):
        from tta_backend.services.methods_export_service import build_methods_markdown

        lineage = {
            "nodes": [
                {"handle": "dataset_tempo_no2", "kind": "dataset", "depth": 1},
                {"handle": "aoi_nj", "kind": "aoi", "depth": 1},
                {
                    "handle": "obs_1",
                    "kind": "observation",
                    "events": [
                        {"event_type": "routed", "detail": {"provider": "GES_DISC"}, "created_at": "2026-07-01T00:00:00Z"},
                        {"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:12:00Z"},
                    ],
                },
            ]
        }
        citations = [
            {
                "handle": "dataset_tempo_no2",
                "concept_id": "C2930725014-LARC_CLOUD",
                "doi": "10.5067/TEMPO/NO2/L3",
                "doi_authority": "https://doi.org",
                "collection_citations": [{
                    "Creator": "NASA/LARC/SD/ASDC",
                    "Title": "TEMPO NO2 L3",
                    "Publisher": "NASA ASDC",
                }],
                "reference_citation_count": 12,
            }
        ]

        markdown = build_methods_markdown(
            artifact_title="TEMPO NO2 over New Jersey",
            aoi_description="New Jersey",
            time_window="2026-06-01/2026-06-30",
            lineage=lineage,
            citations=citations,
        )

        self.assertEqual(
            markdown,
            "\n".join([
                "## Methods — TEMPO NO2 over New Jersey",
                "",
                "Data were retrieved for the area of interest **New Jersey** over "
                "the period **2026-06-01/2026-06-30**.",
                "",
                "### Datasets",
                "",
                "- TEMPO NO2 L3 (doi: 10.5067/TEMPO/NO2/L3)",
                "",
                "### Processing chain",
                "",
                "1. **dataset_tempo_no2** (dataset)",
                "2. **aoi_nj** (aoi)",
                "3. **obs_1** (observation) — routed (2026-07-01T00:00:00Z, provider GES_DISC); "
                "materialized (2026-07-01T00:12:00Z)",
                "",
                "### Retrieval dates",
                "",
                "- 2026-07-01",
                "",
                "### References",
                "",
                "1. NASA/LARC/SD/ASDC, TEMPO NO2 L3, NASA ASDC, doi:10.5067/TEMPO/NO2/L3",
                "",
            ]),
        )

    def test_a_comparisons_aligned_intermediate_appears_in_the_processing_chain(self):
        # T10 story 7: a T08 comparison's resampling step must be visible in
        # the method, not hidden — it renders as its own numbered step.
        from tta_backend.services.methods_export_service import build_methods_markdown

        lineage = {
            "nodes": [
                {"handle": "dataset_a", "kind": "dataset", "depth": 2},
                {"handle": "dataset_b", "kind": "dataset", "depth": 2},
                {
                    "handle": "cube_aligned_1",
                    "kind": "cube",
                    "events": [{"event_type": "created", "detail": {"operation": "align"}, "created_at": "2026-07-01T00:10:00Z"}],
                },
            ]
        }
        citations = [
            {"handle": "dataset_a", "doi": "10.5067/A", "collection_citations": [{"Title": "Dataset A"}]},
            {"handle": "dataset_b", "doi": "10.5067/B", "collection_citations": [{"Title": "Dataset B"}]},
        ]

        markdown = build_methods_markdown(
            artifact_title="A vs B comparison",
            aoi_description="New Jersey",
            time_window="2026-06-01/2026-06-30",
            lineage=lineage,
            citations=citations,
        )

        processing_section = markdown.split("### Processing chain\n\n")[1].split("\n\n")[0]
        self.assertEqual(
            processing_section,
            "1. **dataset_a** (dataset)\n"
            "2. **dataset_b** (dataset)\n"
            "3. **cube_aligned_1** (cube) — created (2026-07-01T00:10:00Z, operation align)",
        )
        self.assertIn("- Dataset A (doi: 10.5067/A)", markdown)
        self.assertIn("- Dataset B (doi: 10.5067/B)", markdown)
        self.assertIn("1. Dataset A, doi:10.5067/A", markdown)
        self.assertIn("2. Dataset B, doi:10.5067/B", markdown)

    def test_shape_surprises_degrade_to_blander_lines_never_a_crash(self):
        # QA 2026-07-17: the endpoint 500'd on a KeyError. The generator must
        # tolerate missing keys anywhere in lineage/citations — an unknown
        # future event shape yields duller text, not an exception.
        from tta_backend.services.methods_export_service import build_methods_markdown

        lineage = {
            "nodes": [
                {"handle": "dataset_x", "kind": "dataset"},
                {"handle": "obs_2", "kind": "observation", "events": [{}, {"event_type": "materialized"}]},
                {},
            ]
        }
        citations = [{}]

        markdown = build_methods_markdown(
            artifact_title="Odd shapes",
            aoi_description="somewhere",
            time_window="unknown",
            lineage=lineage,
            citations=citations,
        )

        self.assertIn("## Methods — Odd shapes", markdown)
        self.assertIn("event (time unknown)", markdown)
        self.assertIn("materialized (time unknown)", markdown)


if __name__ == "__main__":
    unittest.main()
