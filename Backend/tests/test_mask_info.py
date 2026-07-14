import unittest


class OverrideForTests(unittest.TestCase):
    def test_returns_empty_dict_when_short_name_has_no_override(self):
        from datasets.mask_info import override_for

        self.assertEqual(override_for("TEMPO_NO2", overrides={}), {})

    def test_returns_empty_dict_when_short_name_is_none(self):
        from datasets.mask_info import override_for

        self.assertEqual(override_for(None), {})

    def test_returns_the_recorded_override_for_a_known_quirk(self):
        from datasets.mask_info import override_for

        overrides = {"TEMPO_NO2_QUIRK": {"fill_value": -1e30, "valid_min": 0.0, "valid_max": 1e16}}

        info = override_for("TEMPO_NO2_QUIRK", overrides=overrides)

        self.assertEqual(info, {"fill_value": -1e30, "valid_min": 0.0, "valid_max": 1e16})


class ColInfoForShortNameTests(unittest.TestCase):
    """T25 masking-execution fix: col_info_for_short_name is the seam that
    lets a tool's identity marker (an opened file's short_name attribute)
    actually reach collections.yaml's pinned qa_good_values/quality_flag_var
    -- the tool layer never has a collection_id, and the science variable
    name is not a registry key, so this short_name match is what makes a
    pinned Tier-1 QA rule reachable at all."""

    def test_returns_empty_dict_for_no_short_name(self):
        from datasets.mask_info import col_info_for_short_name

        self.assertEqual(col_info_for_short_name(None), {})
        self.assertEqual(col_info_for_short_name(""), {})

    def test_matches_a_registered_collection_by_short_name_case_insensitively(self):
        from datasets.mask_info import col_info_for_short_name

        info = col_info_for_short_name("tempo_no2_l3")

        self.assertEqual(info["quality_flag_var"], "main_data_quality_flag")
        self.assertEqual(info["qa_good_values"], [0])
        self.assertEqual(info["collection_id"], "C3685896708-LARC_CLOUD")

    def test_returns_empty_dict_for_an_unregistered_short_name(self):
        from datasets.mask_info import col_info_for_short_name

        self.assertEqual(col_info_for_short_name("SOME_UNKNOWN_COLLECTION"), {})

    def test_mask_overrides_take_precedence_over_the_registry_match(self):
        import datasets.mask_info as mask_info_module

        original = dict(mask_info_module.MASK_OVERRIDES)
        mask_info_module.MASK_OVERRIDES["TEMPO_NO2_L3"] = {"fill_value": -42.0}
        self.addCleanup(lambda: mask_info_module.MASK_OVERRIDES.clear() or mask_info_module.MASK_OVERRIDES.update(original))

        info = mask_info_module.col_info_for_short_name("TEMPO_NO2_L3")

        self.assertEqual(info["fill_value"], -42.0)
        # The registry's own fields (not overridden) still come through.
        self.assertEqual(info["qa_good_values"], [0])


class ShortNameFromAttrsTests(unittest.TestCase):
    """AOD misrouting follow-up (2026-07-12): real granules spell the identity
    marker differently. The AER_DBDT AOD export carries CF/ACDD ``ShortName``
    and no lowercase ``short_name`` at all, so a lowercase-only lookup silently
    failed to recognize a registered collection (missing both registry masking
    and the primary_var fallback)."""

    def test_reads_the_lowercase_short_name_spelling(self):
        from datasets.mask_info import short_name_from_attrs

        self.assertEqual(short_name_from_attrs({"short_name": "TEMPO_NO2_L3"}), "TEMPO_NO2_L3")

    def test_reads_the_cf_acdd_ShortName_spelling(self):
        from datasets.mask_info import short_name_from_attrs

        self.assertEqual(
            short_name_from_attrs({"ShortName": "AER_DBDT_D10KM_L3_MODIS_TERRA"}),
            "AER_DBDT_D10KM_L3_MODIS_TERRA",
        )

    def test_returns_none_when_no_identity_marker_is_present(self):
        from datasets.mask_info import short_name_from_attrs

        self.assertIsNone(short_name_from_attrs({"title": "some product", "platform": "TERRA"}))
        self.assertIsNone(short_name_from_attrs(None))
        self.assertIsNone(short_name_from_attrs({}))


class ResolveMaskInfoPrecedenceTests(unittest.TestCase):
    """T25 Phase 1: collections.yaml override -> UMM-Var facts -> CF file
    attrs -> mask nothing, with every tier's win recorded in provenance."""

    def test_cf_attrs_used_and_recorded_when_no_yaml_or_umm_var_info(self):
        from datasets.mask_info import resolve_mask_info

        resolved, provenance = resolve_mask_info(
            yaml_info=None,
            umm_var_variable=None,
            cf_attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 100.0, "units": "ppb"},
        )

        self.assertEqual(resolved, {"fill_value": -999.0, "valid_min": 0.0, "valid_max": 100.0, "units": "ppb"})
        self.assertEqual(provenance["fill_value_source"], "cf_attrs")
        self.assertEqual(provenance["valid_range_source"], "cf_attrs")
        self.assertTrue(provenance["applied"])

    def test_umm_var_facts_win_over_cf_attrs_when_no_yaml_override(self):
        from datasets.mask_info import resolve_mask_info

        resolved, provenance = resolve_mask_info(
            yaml_info=None,
            umm_var_variable={
                "name": "NO2_column",
                "units": "mol/m^2",
                "fill_values": [{"value": -9999.0, "context": "FillValue"}],
                "valid_ranges": [{"min": 0.0, "max": 1.0, "context": "valid_range"}],
            },
            cf_attrs={"_FillValue": -999.0, "valid_min": -10.0, "valid_max": 10.0, "units": "wrong"},
        )

        self.assertEqual(resolved["fill_value"], -9999.0)
        self.assertEqual(resolved["valid_min"], 0.0)
        self.assertEqual(resolved["valid_max"], 1.0)
        self.assertEqual(resolved["units"], "mol/m^2")
        self.assertEqual(provenance["fill_value_source"], "umm_var")
        self.assertEqual(provenance["valid_range_source"], "umm_var")
        self.assertTrue(provenance["applied"])

    def test_yaml_override_wins_over_umm_var_and_cf_attrs(self):
        from datasets.mask_info import resolve_mask_info

        resolved, provenance = resolve_mask_info(
            yaml_info={"fill_value": -1.0, "valid_min": 0.0, "valid_max": 500.0, "units": "molecules/cm^2"},
            umm_var_variable={
                "fill_values": [{"value": -9999.0}],
                "valid_ranges": [{"min": -10.0, "max": 10.0}],
                "units": "mol/m^2",
            },
            cf_attrs={"_FillValue": -999.0, "valid_min": -10.0, "valid_max": 10.0, "units": "wrong"},
        )

        self.assertEqual(resolved, {"fill_value": -1.0, "valid_min": 0.0, "valid_max": 500.0, "units": "molecules/cm^2"})
        self.assertEqual(provenance["fill_value_source"], "collections_yaml")
        self.assertEqual(provenance["valid_range_source"], "collections_yaml")
        self.assertTrue(provenance["applied"])

    def test_no_source_anywhere_discloses_none_and_unapplied(self):
        from datasets.mask_info import resolve_mask_info

        resolved, provenance = resolve_mask_info(yaml_info=None, umm_var_variable=None, cf_attrs=None)

        self.assertEqual(resolved, {})
        self.assertEqual(provenance["fill_value_source"], "none")
        self.assertEqual(provenance["valid_range_source"], "none")
        self.assertFalse(provenance["applied"])

    def test_umm_var_with_no_fill_or_range_falls_through_to_cf_attrs(self):
        from datasets.mask_info import resolve_mask_info

        resolved, provenance = resolve_mask_info(
            yaml_info=None,
            umm_var_variable={"name": "cloud_fraction", "fill_values": [], "valid_ranges": [], "units": "1"},
            cf_attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 1.0},
        )

        self.assertEqual(resolved["fill_value"], -999.0)
        self.assertEqual(resolved["valid_min"], 0.0)
        self.assertEqual(resolved["valid_max"], 1.0)
        self.assertEqual(provenance["fill_value_source"], "cf_attrs")
        self.assertEqual(provenance["valid_range_source"], "cf_attrs")
        # Units still resolve from UMM-Var even though fill/range fell through.
        self.assertEqual(resolved["units"], "1")


class MaskInfoAppliesToAggregationServiceTests(unittest.TestCase):
    def test_dataset_attrs_mask_a_synthetic_dataset_with_sentinel_fills(self):
        import numpy as np
        import xarray as xr

        from preprocessing.aggregation_service import AggregationService

        da = xr.DataArray(
            [[-999.0, 10.0], [600.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 500.0},
        )

        masked = AggregationService().apply_quality_mask(da, col_info={})

        values = masked.values
        self.assertTrue(np.isnan(values[0, 0]))  # sentinel fill
        self.assertTrue(np.isnan(values[1, 0]))  # out of valid range
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)

    def test_override_table_corrects_a_known_wrong_umm_var_record(self):
        import numpy as np
        import xarray as xr

        from datasets.mask_info import override_for
        from preprocessing.aggregation_service import AggregationService

        # The dataset's own attrs are wrong for this (fictional) quirky collection.
        da = xr.DataArray(
            [[-1.0, 10.0], [600.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -1.0, "valid_min": -1000.0, "valid_max": 1000.0},
        )
        overrides = {"QUIRKY_NO2": {"fill_value": -1.0, "valid_min": 0.0, "valid_max": 500.0}}

        col_info = override_for("QUIRKY_NO2", overrides=overrides)
        masked = AggregationService().apply_quality_mask(da, col_info=col_info)

        values = masked.values
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertTrue(np.isnan(values[1, 0]))
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)


if __name__ == "__main__":
    unittest.main()
