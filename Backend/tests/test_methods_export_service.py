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
    period map is their average. Twelve hourly frames, all observed."""
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
    render.update(overrides.pop("render", {}))
    spec.update(overrides.pop("spec", {}))
    return _frames_chart(render=render, spec=spec, **overrides)


def _coarsened_tier_chart(*, headline, max_abs, buckets_per_frame=5, **overrides) -> dict:
    """Tier two: a frame averages several cadence buckets, so the frames are a
    DIFFERENT temporal aggregation from the map, which D14 computes
    independently. Phase 3 measured that disagreement at 22.3% and 5.4% on two
    real TEMPO bundles, and the 60-frame budget is reached at ~2.5 days — so
    this is the common case, not the edge."""
    from tta_backend.preprocessing.frame_stack import DELTA_BASIS

    render = {
        "tier": "coarsened",
        "buckets_per_frame": buckets_per_frame,
        "delta": {"headline": headline, "max_abs": max_abs, "basis": DELTA_BASIS},
        **overrides.pop("render", {}),
    }
    spec = {
        "tier": "coarsened",
        "buckets_per_frame": buckets_per_frame,
        **overrides.pop("spec", {}),
    }
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
                "- Each frame is one hourly interval of this product, and the period "
                "map is the equally-weighted mean of the frames: every interval "
                "counts once, however many granules it holds.",
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
        from tta_backend.preprocessing.frame_stack import DELTA_BASIS

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
            ]),
        )
        self.assertLess(
            markdown.index("### Frame–map agreement"),
            markdown.index("### References"),
        )

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

    def test_the_cadence_tier_states_the_identity_rather_than_a_measurement(self):
        # A frame IS a cadence bucket here and the map is their average, so
        # there is no disagreement to report. Printing "0%" would claim a
        # measurement that was never taken — and the identity itself is already
        # stated in the Temporal frames section.
        markdown = _markdown_for(_cadence_tier_chart())

        self.assertNotIn("Frame–map agreement", markdown)
        self.assertIn(
            "the period map is the equally-weighted mean of the frames", markdown,
        )

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
