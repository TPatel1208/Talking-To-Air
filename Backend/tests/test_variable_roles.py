"""
tests/test_variable_roles.py
============================
PRD T35 — variable-role taxonomy. Validated against the *real* product
inventories captured under ``tests/fixtures/variable_inventories/`` (task
zero: describe_dataset against every registered collection, plus a spot
granule open for MODIS AOD where UMM-Var is empty), not against the rules the
classifier was written from — so expanding beyond TEMPO can't silently
regress.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

from datasets.variable_roles import (  # noqa: E402
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ROLE_CONTEXT,
    ROLE_QUALITY,
    ROLE_RETRIEVAL_METADATA,
    ROLE_SCIENCE,
    ROLE_UNCLASSIFIED,
    classify_inventory,
    classify_variable,
    related_variables,
)

FIXTURE_DIR = os.path.join(BACKEND_DIR, "tests", "fixtures", "variable_inventories")


def _load_fixtures() -> list[dict]:
    fixtures = []
    for path in sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            fixtures.append(json.load(f))
    return fixtures


class GoldenInventoryTests(unittest.TestCase):
    """Table-driven: for every registered collection, the classified role of
    each real variable must match the ground-truth expectation recorded during
    task zero. The corpus grows each time a new product surprises the
    classifier."""

    def test_fixtures_exist_for_every_registered_collection(self):
        from datasets.registry import load_registry

        keys = {fx["registry_key"] for fx in _load_fixtures()}
        registered = set(load_registry().keys())
        self.assertEqual(
            registered - keys, set(),
            "every registered collection needs a captured ground-truth inventory fixture",
        )

    def test_every_variable_matches_its_ground_truth_role(self):
        fixtures = _load_fixtures()
        self.assertTrue(fixtures, "no inventory fixtures found")
        for fx in fixtures:
            classified = classify_inventory(
                fx["variables"],
                groups=fx.get("groups"),
                primary_var=fx.get("primary_var"),
                quality_flag_var=fx.get("quality_flag_var"),
            )
            by_name = {c["name"]: c["role"] for c in classified}
            for name, expected in fx["expected_roles"].items():
                with self.subTest(collection=fx["registry_key"], variable=name):
                    self.assertEqual(
                        by_name.get(name), expected,
                        f"{fx['registry_key']}:{name} expected {expected}, got {by_name.get(name)}",
                    )

    def test_modis_aod_has_science_but_no_invented_context_beyond_real_bands(self):
        # MODIS AOD's describe_dataset returns 0 UMM-Var records; the fixture is
        # a real granule open. It genuinely carries a Sensor_Zenith_Angle band
        # and coordinates (context) — the classifier must reflect that reality,
        # not invent extra atmospheric context, and never leave the AOD product
        # unclassified.
        fx = next(f for f in _load_fixtures() if f["registry_key"] == "MODIS_AOD_TERRA")
        classified = classify_inventory(
            fx["variables"], groups=fx.get("groups"),
            primary_var=fx.get("primary_var"), quality_flag_var=fx.get("quality_flag_var"),
        )
        by_name = {c["name"]: c["role"] for c in classified}
        self.assertEqual(by_name["COMBINE_AOD_550_AVG"], ROLE_SCIENCE)
        self.assertEqual(by_name["DT_AOD_550_AVG"], ROLE_SCIENCE)
        # No unclassified science-product masquerade, and no invented context.
        roles = set(by_name.values())
        self.assertNotIn(ROLE_RETRIEVAL_METADATA, roles)
        self.assertEqual(by_name["Sensor_Zenith_Angle"], ROLE_CONTEXT)


class CollisionAndNegativeTests(unittest.TestCase):
    """The sharp cases — where a science stem and an exception marker collide,
    or where science-as-default would wrongly fire."""

    def test_uncertainty_of_a_science_column_is_quality_not_science(self):
        role, _ = classify_variable("product/vertical_column_troposphere_uncertainty")
        self.assertEqual(role, ROLE_QUALITY)

    def test_stratosphere_column_is_science(self):
        role, _ = classify_variable("product/vertical_column_stratosphere", group="product")
        self.assertEqual(role, ROLE_SCIENCE)

    def test_cloud_screened_science_column_is_science_not_context(self):
        # 'CloudScreened' contains 'cloud' but the variable is a science column.
        role, _ = classify_variable(
            "HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2CloudScreened"
        )
        self.assertEqual(role, ROLE_SCIENCE)

    def test_amf_is_retrieval_metadata(self):
        self.assertEqual(classify_variable("support_data/amf_troposphere")[0], ROLE_RETRIEVAL_METADATA)
        self.assertEqual(classify_variable("support_data/fitted_slant_column")[0], ROLE_RETRIEVAL_METADATA)

    def test_below_cloud_partial_is_retrieval_metadata_not_context(self):
        # '*_below_cloud' contains 'cloud' but is a retrieval intermediate.
        role, _ = classify_variable("product/o3_below_cloud", group="product")
        self.assertEqual(role, ROLE_RETRIEVAL_METADATA)

    def test_cloud_fraction_band_is_context(self):
        self.assertEqual(classify_variable("product/radiative_cloud_frac", group="product")[0], ROLE_CONTEXT)
        self.assertEqual(classify_variable("product/uv_aerosol_index", group="product")[0], ROLE_CONTEXT)

    def test_algorithm_and_bookkeeping_fields_are_unclassified(self):
        # Science-as-default is gone: none of these carry positive science
        # evidence, so they must fall to unclassified, not a forced bucket.
        for name in ("processing_version", "scan_line", "orbit_number",
                     "detector_index", "weight", "sample_weight"):
            with self.subTest(variable=name):
                role, confidence = classify_variable(name)
                self.assertEqual(role, ROLE_UNCLASSIFIED)
                self.assertIsNone(confidence)


class GroupPriorTests(unittest.TestCase):
    """Group membership is a strong prior, applied ahead of name patterns."""

    def test_unrecognized_name_in_qa_statistics_is_quality(self):
        role, confidence = classify_variable("qa_statistics/some_unknown_diagnostic")
        self.assertEqual(role, ROLE_QUALITY)
        self.assertEqual(confidence, CONFIDENCE_HIGH)

    def test_geolocation_angle_is_context(self):
        role, confidence = classify_variable("geolocation/solar_zenith_angle")
        self.assertEqual(role, ROLE_CONTEXT)
        self.assertEqual(confidence, CONFIDENCE_HIGH)

    def test_bare_name_in_science_group_is_science(self):
        # A variable with no marker and no science stem, living in a science
        # group, becomes science via the group's positive evidence.
        role, confidence = classify_variable("product/fc", group="product")
        self.assertEqual(role, ROLE_SCIENCE)
        self.assertEqual(confidence, CONFIDENCE_HIGH)

    def test_marker_beats_science_group(self):
        # An uncertainty inside a science group is still quality — the marker is
        # checked before the science-group prior.
        role, _ = classify_variable("key_science_data/column_uncertainty", group="key_science_data")
        self.assertEqual(role, ROLE_QUALITY)

    def test_explicit_record_group_reaches_the_group_priors(self):
        """A post-open inventory (plot_tools' evidence path) carries bare leaf
        names — open_handle merges groups without prefixing — so group
        membership must be able to travel as an explicit per-record ``group``
        and still fire the group priors, exactly as a slash-qualified
        describe_dataset name does. TEMPO_NO2's real
        qa_statistics/max_vertical_column_stratosphere_sample would otherwise
        misread as science via the verticalcolumn stem."""
        out = classify_inventory([
            {"name": "max_vertical_column_stratosphere_sample", "group": "qa_statistics"},
        ])
        self.assertEqual(out[0]["role"], ROLE_QUALITY)
        self.assertEqual(out[0]["confidence"], CONFIDENCE_HIGH)
        self.assertEqual(out[0]["group"], "qa_statistics")


class ConfidenceTests(unittest.TestCase):
    """Confidence describes the decision: High (metadata/group), Medium
    (marker), Low (stem), None (unclassified)."""

    def test_standard_name_reports_high(self):
        role, confidence = classify_variable("Tropospheric_NO2",
                                             standard_name="troposphere_mole_content_of_nitrogen_dioxide")
        self.assertEqual(role, ROLE_SCIENCE)
        self.assertEqual(confidence, CONFIDENCE_HIGH)

    def test_marker_rule_reports_medium(self):
        _, confidence = classify_variable("support_data/surface_pressure")
        self.assertEqual(confidence, CONFIDENCE_MEDIUM)

    def test_science_stem_reports_low(self):
        role, confidence = classify_variable(
            "HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2Trop"
        )
        self.assertEqual(role, ROLE_SCIENCE)
        self.assertEqual(confidence, CONFIDENCE_LOW)

    def test_unmatched_reports_none_and_unclassified(self):
        role, confidence = classify_variable("orbit_number")
        self.assertEqual(role, ROLE_UNCLASSIFIED)
        self.assertIsNone(confidence)


class RelatedVariablesTests(unittest.TestCase):
    """The chart-page related-variables view — role + cheap companion links,
    built from the registry's curated variable subset (no MCP round trip)."""

    def test_tempo_o3_surfaces_real_context_bands(self):
        # TEMPO O3TOT's registry subset carries real context bands.
        rv = related_variables(
            [
                "product/column_amount_o3", "product/radiative_cloud_frac",
                "product/fc", "product/o3_below_cloud", "product/so2_index",
                "product/uv_aerosol_index",
            ],
            groups=["product"],
            primary_var="column_amount_o3",
            quality_flag_var=None,
            plotted_variable="column_amount_o3",
        )
        self.assertEqual(rv["role"], ROLE_SCIENCE)
        self.assertIn("radiative_cloud_frac", rv["context_siblings"])
        self.assertIn("uv_aerosol_index", rv["context_siblings"])
        # o3_below_cloud is a retrieval intermediate, not a context sibling.
        self.assertNotIn("o3_below_cloud", rv["context_siblings"])
        self.assertIsNone(rv["qa_sibling"])

    def test_qa_and_uncertainty_siblings_are_matched(self):
        rv = related_variables(
            [
                "product/vertical_column", "product/vertical_column_uncertainty",
                "product/main_data_quality_flag",
            ],
            groups=["product"],
            primary_var="vertical_column",
            quality_flag_var="main_data_quality_flag",
            plotted_variable="vertical_column",
        )
        self.assertEqual(rv["qa_sibling"], "main_data_quality_flag")
        self.assertEqual(rv["uncertainty_sibling"], "vertical_column_uncertainty")

    def test_precision_style_companion_is_the_uncertainty_sibling(self):
        """OMI_O3 carries its error companion as ColumnAmountO3Precision — the
        classifier already tags it quality; the sibling matcher must surface
        it rather than requiring the literal <stem>uncertainty spelling."""
        rv = related_variables(
            ["ColumnAmountO3", "ColumnAmountO3Precision", "RadiativeCloudFraction"],
            primary_var="ColumnAmountO3",
            plotted_variable="ColumnAmountO3",
        )
        self.assertEqual(rv["uncertainty_sibling"], "ColumnAmountO3Precision")

    def test_exact_uncertainty_spelling_is_preferred_over_other_suffixes(self):
        rv = related_variables(
            ["vertical_column", "vertical_column_std", "vertical_column_uncertainty"],
            primary_var="vertical_column",
            plotted_variable="vertical_column",
        )
        self.assertEqual(rv["uncertainty_sibling"], "vertical_column_uncertainty")

    def test_modis_aod_invents_no_context_siblings(self):
        # MODIS AOD's registry subset is empty — the panel must show the
        # plotted role and nothing spurious.
        rv = related_variables(
            [], groups=[], primary_var="COMBINE_AOD_550_AVG", quality_flag_var=None,
            plotted_variable="COMBINE_AOD_550_AVG",
        )
        self.assertEqual(rv["role"], ROLE_SCIENCE)
        self.assertEqual(rv["context_siblings"], [])
        self.assertIsNone(rv["uncertainty_sibling"])
        self.assertIsNone(rv["qa_sibling"])


class EnrichmentSeamTests(unittest.TestCase):
    """discovery_service.describe_dataset attaches the classified inventory
    additively, without disturbing the keys existing callers already read."""

    def _describe_result(self) -> dict:
        return {
            "handle": "dataset_abc",
            "concept_id": "C3685896625-LARC_CLOUD",  # TEMPO_O3TOT
            "metadata": {"short_name": "TEMPO_O3TOT_L3"},
            "variables": [
                {"name": "product/column_amount_o3", "long_name": "ozone", "units": "DU"},
                {"name": "product/radiative_cloud_frac", "long_name": "cloud frac", "units": None},
                {"name": "product/o3_below_cloud", "long_name": "below cloud", "units": "DU"},
                {"name": "qa_statistics/num_column_samples", "long_name": "n", "units": None},
                {"name": "weight", "long_name": "area weight", "units": "km^2"},
            ],
            "variable_count": 5,
        }

    def test_attach_inventory_is_additive_and_classifies(self):
        from services.discovery_service import _attach_inventory

        result = self._describe_result()
        original_keys = set(result.keys())
        enriched = _attach_inventory(result)

        self.assertTrue(original_keys.issubset(set(enriched.keys())))
        self.assertEqual(enriched["variable_count"], 5)
        self.assertIn("inventory", enriched)
        inv = enriched["inventory"]
        by_name = {v["name"]: v["role"] for v in inv["variables"]}
        self.assertEqual(by_name["product/column_amount_o3"], ROLE_SCIENCE)
        self.assertEqual(by_name["product/radiative_cloud_frac"], ROLE_CONTEXT)
        self.assertEqual(by_name["product/o3_below_cloud"], ROLE_RETRIEVAL_METADATA)
        self.assertEqual(by_name["qa_statistics/num_column_samples"], ROLE_QUALITY)
        self.assertEqual(by_name["weight"], ROLE_UNCLASSIFIED)
        self.assertEqual(inv["collection_key"], "TEMPO_O3TOT")
        self.assertEqual(inv["counts"][ROLE_SCIENCE], 1)
        self.assertEqual(inv["roles_present"][0], ROLE_SCIENCE)

    def test_attach_inventory_untouched_when_no_variables(self):
        from services.discovery_service import _attach_inventory

        result = {"handle": "dataset_x", "variables": []}
        enriched = _attach_inventory(result)
        self.assertNotIn("inventory", enriched)

    def test_concept_id_match_beats_an_earlier_short_name_match(self):
        """TEMPO_HCHO and TEMPO_HCHO_V03 share short_name TEMPO_HCHO_L3 with
        distinct collection_ids. A V03 describe result carrying its exact
        concept_id must resolve to TEMPO_HCHO_V03 even though TEMPO_HCHO
        iterates first in the registry and also matches on short_name —
        concept_id is the stronger identity and wins across ALL entries."""
        from services.discovery_service import _attach_inventory

        result = {
            "handle": "dataset_tempo_hcho_v03",
            "concept_id": "C2930761273-LARC_CLOUD",  # TEMPO_HCHO_V03
            "metadata": {"short_name": "TEMPO_HCHO_L3"},
            "variables": [{"name": "product/vertical_column"}],
        }
        enriched = _attach_inventory(result)
        self.assertEqual(enriched["inventory"]["collection_key"], "TEMPO_HCHO_V03")

    def test_malformed_variable_record_degrades_to_no_inventory(self):
        """A live MCP response with a non-string variable name must not become
        an unhandled 500 (api.py registers only the MCPToolError handler): the
        result degrades to its bare, inventory-less shape — the same honest
        degrade plot_tools' evidence path already applies (T18 doctrine)."""
        from services.discovery_service import _attach_inventory

        result = {"handle": "dataset_x", "variables": [{"name": 123}]}
        enriched = _attach_inventory(result)
        self.assertNotIn("inventory", enriched)


if __name__ == "__main__":
    unittest.main()
