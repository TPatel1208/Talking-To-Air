"""
The comparison analysis in domain/analysis/comparison.py: differences, anomaly
statistics, aligned-cube splitting and region statistics, on synthetic
aligned/misaligned cube fixtures.
"""
import importlib.util
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain", "numpy", "pandas", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class DifferenceTests(unittest.TestCase):
    def test_difference_is_period_b_minus_period_a(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import difference as _difference

        da_a = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))
        da_b = xr.DataArray([[5.0, 5.0], [5.0, 5.0]], dims=("lat", "lon"))

        diff = _difference(da_a, da_b)

        self.assertEqual(diff.values.tolist(), [[4.0, 3.0], [2.0, 1.0]])

    def test_a_cell_missing_on_either_side_is_excluded_from_the_difference(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.analysis.comparison import difference as _difference

        da_a = xr.DataArray([1.0, np.nan, 3.0], dims=("x",))
        da_b = xr.DataArray([10.0, 20.0, np.nan], dims=("x",))

        diff = _difference(da_a, da_b)

        self.assertEqual(diff.values[0], 9.0)
        self.assertTrue(np.isnan(diff.values[1]))
        self.assertTrue(np.isnan(diff.values[2]))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class AnomalyStatsTests(unittest.TestCase):
    def test_mean_difference_and_percent_change_match_hand_computed_values(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import anomaly_stats as _anomaly_stats, difference as _difference

        # a: mean 2.0 -> b: mean 3.0. diff mean = 1.0, percent change = 50%.
        da_a = xr.DataArray([1.0, 2.0, 3.0], dims=("x",))
        da_b = xr.DataArray([2.0, 3.0, 4.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertEqual(stats["n_cells"], 3)
        self.assertAlmostEqual(stats["mean_difference"], 1.0)
        self.assertAlmostEqual(stats["percent_change"], 50.0)
        self.assertNotIn("area_exceeding_threshold", stats)

    def test_cells_missing_on_either_side_are_excluded_from_stats(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.analysis.comparison import anomaly_stats as _anomaly_stats, difference as _difference

        da_a = xr.DataArray([1.0, np.nan], dims=("x",))
        da_b = xr.DataArray([2.0, 5.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertEqual(stats["n_cells"], 1)
        self.assertAlmostEqual(stats["mean_difference"], 1.0)

    def test_area_exceeding_threshold_counts_cells_at_or_above_the_magnitude(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import anomaly_stats as _anomaly_stats, difference as _difference

        da_a = xr.DataArray([0.0, 0.0, 0.0, 0.0], dims=("x",))
        da_b = xr.DataArray([1.0, 2.0, 5.0, 10.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=5.0)

        self.assertEqual(stats["area_exceeding_threshold"]["n_cells"], 2)
        self.assertAlmostEqual(stats["area_exceeding_threshold"]["fraction"], 0.5)
        self.assertEqual(stats["area_exceeding_threshold"]["threshold"], 5.0)

    def test_percent_change_is_reported_when_the_baseline_clears_the_floor(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.analysis.comparison import (
            PERCENT_CHANGE_FLOOR_FRACTION as _PERCENT_CHANGE_FLOOR_FRACTION,
            anomaly_stats as _anomaly_stats,
            difference as _difference,
        )

        # A spans 1..101 (mean ~51); its own 2–98th-percentile range is wide,
        # so the floor is a few units and the sizeable mean clears it easily.
        a_vals = np.arange(1.0, 102.0)
        da_a = xr.DataArray(a_vals, dims=("x",))
        da_b = xr.DataArray(a_vals + a_vals, dims=("x",))  # b = 2a -> +100%
        diff = _difference(da_a, da_b)

        spread = float(np.percentile(a_vals, 98) - np.percentile(a_vals, 2))
        floor = _PERCENT_CHANGE_FLOOR_FRACTION * spread
        self.assertGreater(abs(a_vals.mean()), floor)  # baseline clears the floor

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertAlmostEqual(stats["percent_change"], 100.0)
        self.assertNotIn("percent_change_note", stats)

    def test_percent_change_is_withheld_with_a_note_when_the_baseline_is_below_the_floor(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.analysis.comparison import (
            PERCENT_CHANGE_FLOOR_FRACTION as _PERCENT_CHANGE_FLOOR_FRACTION,
            anomaly_stats as _anomaly_stats,
            difference as _difference,
        )

        # A is ~symmetric about zero (mean a hair above zero) but has a wide
        # magnitude range -- the anomaly-field case where a tiny baseline would
        # otherwise explode the percentage. b is a small constant shift, so the
        # mean difference is real and finite.
        a_vals = np.arange(-50.0, 51.0) + 1.0  # -49..51, mean = 1.0
        da_a = xr.DataArray(a_vals, dims=("x",))
        da_b = xr.DataArray(a_vals + 3.0, dims=("x",))
        diff = _difference(da_a, da_b)

        spread = float(np.percentile(np.abs(a_vals), 98) - np.percentile(np.abs(a_vals), 2))
        floor = _PERCENT_CHANGE_FLOOR_FRACTION * spread
        self.assertLess(abs(a_vals.mean()), floor)  # baseline is under the floor

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        # mean_difference is always kept; percent change is the thing withheld.
        self.assertAlmostEqual(stats["mean_difference"], 3.0)
        self.assertIsNone(stats["percent_change"])
        self.assertIn("percent_change_note", stats)
        self.assertIn("baseline too small", stats["percent_change_note"].lower())


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class SplitAlignedTests(unittest.TestCase):
    def test_splits_a_two_source_aligned_cube_into_its_two_arrays_in_order(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import split_aligned as _split_aligned

        da = xr.DataArray(
            [[[1.0, 2.0]], [[10.0, 20.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": [0, 1], "lat": [10.0], "lon": [30.0, 40.0]},
        )

        da_a, da_b = _split_aligned(da)

        self.assertEqual(da_a.values.tolist(), [[1.0, 2.0]])
        self.assertEqual(da_b.values.tolist(), [[10.0, 20.0]])

    def test_prefers_explicit_source_labels_over_position_so_a_reorder_cannot_flip_sign(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import split_aligned as _split_aligned

        # The MCP stamps the source handles as the `source` coordinate, but in
        # the *opposite* order from how they were passed. Label selection must
        # win: A is the array labeled obs_a regardless of its position.
        da = xr.DataArray(
            [[[10.0, 20.0]], [[1.0, 2.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": ["obs_b", "obs_a"], "lat": [10.0], "lon": [30.0, 40.0]},
        )

        da_a, da_b = _split_aligned(da, handle_a="obs_a", handle_b="obs_b")

        self.assertEqual(da_a.values.tolist(), [[1.0, 2.0]])   # the obs_a-labeled slice
        self.assertEqual(da_b.values.tolist(), [[10.0, 20.0]])  # the obs_b-labeled slice

    def test_falls_back_to_positional_order_when_sources_carry_no_handle_labels(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import split_aligned as _split_aligned

        # Integer source coords (no handle names) -> the MCP input-order
        # convention: position 0 is A, position 1 is B.
        da = xr.DataArray(
            [[[1.0, 2.0]], [[10.0, 20.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": [0, 1], "lat": [10.0], "lon": [30.0, 40.0]},
        )

        da_a, da_b = _split_aligned(da, handle_a="obs_a", handle_b="obs_b")

        self.assertEqual(da_a.values.tolist(), [[1.0, 2.0]])
        self.assertEqual(da_b.values.tolist(), [[10.0, 20.0]])

    def test_rejects_an_aligned_result_without_a_source_dimension(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import split_aligned as _split_aligned

        da = xr.DataArray([[1.0, 2.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [30.0, 40.0]})

        with self.assertRaises(ValueError):
            _split_aligned(da)

    def test_rejects_an_aligned_result_with_the_wrong_number_of_sources(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import split_aligned as _split_aligned

        da = xr.DataArray(
            [[[1.0]], [[2.0]], [[3.0]]],
            dims=("source", "lat", "lon"),
            coords={"source": [0, 1, 2], "lat": [10.0], "lon": [30.0]},
        )

        with self.assertRaises(ValueError):
            _split_aligned(da)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class RegionStatsTests(unittest.TestCase):
    def test_computes_basic_stats_over_valid_cells(self):
        import xarray as xr
        from tta_backend.domain.analysis.comparison import region_stats as _region_stats

        da = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))

        stats = _region_stats(da)

        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["n_pixels"], 4)

    def test_returns_none_when_no_valid_cells(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.analysis.comparison import region_stats as _region_stats

        da = xr.DataArray([np.nan, np.nan], dims=("x",))

        self.assertIsNone(_region_stats(da))

    def test_extremes_disclose_the_field_they_summarize(self):
        """Compare region max/min are order statistics of the
        *period-mean* field (compare always time-means each side), so they sit
        below the instantaneous peak. A bare "Max"/"Min" label invites reading
        them as the true peak/trough. When the caller names the field basis, the
        stats disclose it so the label can be qualified rather than bare."""
        import xarray as xr
        from tta_backend.domain.analysis.comparison import region_stats as _region_stats

        da = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))

        stats = _region_stats(da, basis="period-mean")

        self.assertEqual(stats["basis"], "period-mean")
        # The bare values are unchanged; only their provenance is disclosed.
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["min"], 1.0)

    def test_basis_is_omitted_when_the_caller_does_not_name_one(self):
        """Regression: the disclosure is additive — a caller that doesn't name
        a field basis (a plain snapshot) gets the original stats shape."""
        import xarray as xr
        from tta_backend.domain.analysis.comparison import region_stats as _region_stats

        stats = _region_stats(xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon")))

        self.assertNotIn("basis", stats)


if __name__ == "__main__":
    unittest.main()
