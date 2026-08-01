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
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


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
