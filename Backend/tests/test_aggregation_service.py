import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class AggregationServiceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr
        self.col_info = {
            "primary_var": "no2",
            "cadence": "daily",
            "fill_value": -999.0,
            "valid_min": 0.0,
            "valid_max": 100.0,
        }

    def test_aggregate_counts_only_valid_time_steps(self):
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "no2": (
                    ("time", "lat", "lon"),
                    self.np.array([
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[-999.0, -999.0], [-999.0, -999.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]),
                )
            },
            coords={
                "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            attrs={"cadence": "daily"},
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)
        da = result.ds["no2"]

        self.assertEqual(result.meta["n_granules"], 2)
        self.assertEqual(result.meta["granule_dates"], ["2024-01-01", "2024-01-03"])
        self.assertEqual(float(da.sel(lat=40.0, lon=-75.0)), 3.0)

    def test_all_stats_supported_and_invalid_stat_raises(self):
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([
                [[1.0, 3.0]],
                [[5.0, 7.0]],
            ]),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-01-01", "2024-01-02"], "lat": [40.0], "lon": [-75.0, -74.0]},
            name="no2",
        )
        service = AggregationService()

        self.assertEqual(float(service.aggregate(da, stat="max", col_info=self.col_info).ds["no2"].values[0, 1]), 7.0)
        self.assertEqual(float(service.aggregate(da, stat="min", col_info=self.col_info).ds["no2"].values[0, 1]), 3.0)
        self.assertEqual(float(service.aggregate(da, stat="median", col_info=self.col_info).ds["no2"].values[0, 0]), 3.0)
        self.assertAlmostEqual(float(service.aggregate(da, stat="std", col_info=self.col_info).ds["no2"].values[0, 0]), 2.0)
        with self.assertRaises(ValueError):
            service.aggregate(da, stat="mode", col_info=self.col_info)

    def test_apply_quality_mask_falls_back_to_dataset_attrs_when_col_info_empty(self):
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            [[-999.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 100.0},
        )

        masked = AggregationService().apply_quality_mask(da, col_info={})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))
        self.assertTrue(self.np.isnan(values[1, 0]))
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)

    def test_aggregate_meta_discloses_cf_attrs_masking_when_no_registry_or_umm_var_info(self):
        """T25 Phase 1: an unregistered collection with no UMM-Var facts still
        gets a masking-provenance disclosure — the CF-attrs tier — instead of
        aggregate() staying silent about where fill/valid came from."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-999.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="unregistered_var",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 500.0},
        )

        result = AggregationService().aggregate(da, variable="unregistered_var")

        self.assertIn("masking", result.meta)
        self.assertEqual(result.meta["masking"]["fill_value_source"], "cf_attrs")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "cf_attrs")
        self.assertTrue(result.meta["masking"]["applied"])

    def test_aggregate_uses_umm_var_facts_when_no_yaml_col_info_supplied(self):
        """describe_dataset's per-variable UMM-Var facts mask an unregistered
        collection correctly even though the file's own CF attrs are wrong."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-9999.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="tempo_so2",
            attrs={"_FillValue": -1.0, "valid_min": -1e6, "valid_max": 1e6},  # wrong per-file attrs
        )
        umm_var_facts = [
            {
                "name": "tempo_so2",
                "fill_values": [{"value": -9999.0}],
                "valid_ranges": [{"min": 0.0, "max": 500.0}],
                "units": "molecules/cm^2",
            }
        ]

        result = AggregationService().aggregate(da, variable="tempo_so2", umm_var_facts=umm_var_facts)

        values = result.ds["tempo_so2"].values
        self.assertTrue(self.np.isnan(values[0, 0]))  # masked by UMM-Var fill, not the wrong CF fill
        self.assertTrue(self.np.isnan(values[1, 0]))  # 600 > UMM-Var valid_max of 500
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(result.meta["masking"]["fill_value_source"], "umm_var")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "umm_var")

    def test_aggregate_col_info_override_still_wins_over_umm_var_facts(self):
        """Registry/quirk-ledger col_info stays the top precedence tier even
        when UMM-Var facts are also supplied."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-1.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="no2",
        )
        umm_var_facts = [{"name": "no2", "fill_values": [{"value": -9999.0}], "valid_ranges": [{"min": 0.0, "max": 5000.0}]}]

        result = AggregationService().aggregate(
            da, variable="no2", col_info=self.col_info, umm_var_facts=umm_var_facts,
        )

        self.assertEqual(result.meta["masking"]["fill_value_source"], "collections_yaml")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "collections_yaml")

    def test_apply_quality_mask_col_info_override_wins_over_dataset_attrs(self):
        from preprocessing.aggregation_service import AggregationService

        # Dataset's own attrs are wrong (a known CMR/UMM-Var quirk); the
        # override in col_info must take precedence.
        da = self.xr.DataArray(
            [[-1.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -1.0, "valid_min": -1000.0, "valid_max": 1000.0},
        )

        masked = AggregationService().apply_quality_mask(
            da, col_info={"fill_value": -1.0, "valid_min": 0.0, "valid_max": 100.0}
        )

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # fill
        self.assertTrue(self.np.isnan(values[1, 0]))  # 200 > override valid_max of 100
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)


if __name__ == "__main__":
    unittest.main()
