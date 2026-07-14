"""
tests/test_methods_export_service.py
======================================
T10: golden tests for the deterministic methods-text generator. Canned
lineage/citation fixtures (the same shapes provenance_service produces) go
in; the exact expected Markdown comes out — no LLM in the loop, so the same
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
        from services.methods_export_service import build_methods_markdown

        lineage = {
            "nodes": [
                {"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"},
                {"handle": "aoi_nj", "kind": "aoi", "description": "New Jersey"},
                {
                    "handle": "obs_1",
                    "kind": "observation",
                    "events": [
                        {"stage": "routed", "at": "2026-07-01T00:00:00Z", "provider": "GES_DISC"},
                        {"stage": "materialized", "at": "2026-07-01T00:12:00Z"},
                    ],
                },
            ]
        }
        citations = [
            {
                "dataset_handle": "dataset_tempo_no2",
                "doi": "10.5067/TEMPO/NO2/L3",
                "citation": "NASA, TEMPO NO2 Tropospheric Column L3, doi:10.5067/TEMPO/NO2/L3",
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
                "1. **dataset_tempo_no2** (dataset) — TEMPO NO2 L3",
                "2. **aoi_nj** (aoi) — New Jersey",
                "3. **obs_1** (observation) — routed (2026-07-01T00:00:00Z, provider GES_DISC); "
                "materialized (2026-07-01T00:12:00Z)",
                "",
                "### Retrieval dates",
                "",
                "- 2026-07-01",
                "",
                "### References",
                "",
                "1. NASA, TEMPO NO2 Tropospheric Column L3, doi:10.5067/TEMPO/NO2/L3",
                "",
            ]),
        )

    def test_a_comparisons_aligned_intermediate_appears_in_the_processing_chain(self):
        # T10 story 7: a T08 comparison's resampling step must be visible in
        # the method, not hidden — it renders as its own numbered step.
        from services.methods_export_service import build_methods_markdown

        lineage = {
            "nodes": [
                {"handle": "dataset_a", "kind": "dataset", "description": "Dataset A"},
                {"handle": "dataset_b", "kind": "dataset", "description": "Dataset B"},
                {
                    "handle": "aligned_1",
                    "kind": "aligned",
                    "events": [{"stage": "aligned", "at": "2026-07-01T00:10:00Z", "method": "outer"}],
                },
            ]
        }
        citations = [
            {"dataset_handle": "dataset_a", "doi": "10.5067/A", "citation": "Provider A dataset citation"},
            {"dataset_handle": "dataset_b", "doi": "10.5067/B", "citation": "Provider B dataset citation"},
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
            "1. **dataset_a** (dataset) — Dataset A\n"
            "2. **dataset_b** (dataset) — Dataset B\n"
            "3. **aligned_1** (aligned) — aligned (2026-07-01T00:10:00Z, method outer)",
        )
        self.assertIn("- Dataset A (doi: 10.5067/A)", markdown)
        self.assertIn("- Dataset B (doi: 10.5067/B)", markdown)
        self.assertIn("1. Provider A dataset citation", markdown)
        self.assertIn("2. Provider B dataset citation", markdown)


if __name__ == "__main__":
    unittest.main()
