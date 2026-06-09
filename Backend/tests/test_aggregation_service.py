import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


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


if __name__ == "__main__":
    unittest.main()
