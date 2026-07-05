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
