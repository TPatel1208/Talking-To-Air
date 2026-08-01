"""
Tests for tools/satellite_tools/comparison_tools.py (PRD T08 — region/period
comparison workflow).

Hermetic at the analysis-tool seam: synthetic aligned/misaligned cube
fixtures exercised through the module's own helpers (prior art:
test_validation_tools.py testing validation_tools' helpers directly).
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain", "numpy", "pandas", "xarray"]
FULL_TOOL_REQUIRED_MODULES = REQUIRED_MODULES + [
    "langchain_mcp_adapters", "fastmcp", "uvicorn", "zarr", "httpx",
]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class VariableMismatchTests(unittest.TestCase):
    def test_same_variable_name_is_not_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _variable_mismatch_error

        da_a = xr.DataArray([1.0], name="no2")
        da_b = xr.DataArray([2.0], name="no2")

        self.assertIsNone(_variable_mismatch_error(da_a, da_b))

    def test_different_variable_names_are_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _variable_mismatch_error

        da_a = xr.DataArray([1.0], name="no2")
        da_b = xr.DataArray([2.0], name="hcho")

        error = _variable_mismatch_error(da_a, da_b)

        self.assertIsNotNone(error)
        self.assertIn("no2", error)
        self.assertIn("hcho", error)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class UnitsMismatchTests(unittest.TestCase):
    def test_equal_units_are_not_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _units_mismatch_error

        da_a = xr.DataArray([1.0], name="no2", attrs={"units": "mol/m^2"})
        da_b = xr.DataArray([2.0], name="no2", attrs={"units": "mol/m^2"})

        self.assertIsNone(_units_mismatch_error(da_a, da_b))

    def test_absent_units_on_both_sides_are_not_a_mismatch(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _units_mismatch_error

        # Same variable, neither side publishes a units attr: no confident unit
        # label is stamped on the difference, so there is nothing to mislabel.
        da_a = xr.DataArray([1.0], name="no2")
        da_b = xr.DataArray([2.0], name="no2")

        self.assertIsNone(_units_mismatch_error(da_a, da_b))

    def test_units_on_only_one_side_is_rejected_naming_the_declared_unit(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _units_mismatch_error

        # One side declares units, the other has none: the difference would be
        # differenced and stamped with the declared label, presenting an
        # unverifiable-commensurability comparison as a real number.
        da_a = xr.DataArray([1.0], name="no2", attrs={"units": "mol/m^2"})
        da_b = xr.DataArray([2.0], name="no2")  # no units attr at all

        error = _units_mismatch_error(da_a, da_b)
        self.assertIsNotNone(error)
        self.assertIn("mol/m^2", error)

        # Symmetric: the missing side being A is just as unverifiable.
        error_swapped = _units_mismatch_error(da_b, da_a)
        self.assertIsNotNone(error_swapped)
        self.assertIn("mol/m^2", error_swapped)

    def test_different_units_are_rejected_naming_both(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _units_mismatch_error

        da_a = xr.DataArray([1.0], name="vertical_column_troposphere", attrs={"units": "molec/cm^2"})
        da_b = xr.DataArray([2.0], name="vertical_column_troposphere", attrs={"units": "mol/m^2"})

        error = _units_mismatch_error(da_a, da_b)

        self.assertIsNotNone(error)
        self.assertIn("molec/cm^2", error)
        self.assertIn("mol/m^2", error)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class DifferenceTests(unittest.TestCase):
    def test_difference_is_period_b_minus_period_a(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _difference

        da_a = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))
        da_b = xr.DataArray([[5.0, 5.0], [5.0, 5.0]], dims=("lat", "lon"))

        diff = _difference(da_a, da_b)

        self.assertEqual(diff.values.tolist(), [[4.0, 3.0], [2.0, 1.0]])

    def test_a_cell_missing_on_either_side_is_excluded_from_the_difference(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _difference

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
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

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
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

        da_a = xr.DataArray([1.0, np.nan], dims=("x",))
        da_b = xr.DataArray([2.0, 5.0], dims=("x",))
        diff = _difference(da_a, da_b)

        stats = _anomaly_stats(da_a, da_b, diff, threshold=None)

        self.assertEqual(stats["n_cells"], 1)
        self.assertAlmostEqual(stats["mean_difference"], 1.0)

    def test_area_exceeding_threshold_counts_cells_at_or_above_the_magnitude(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _anomaly_stats, _difference

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
        from tools.satellite_tools.comparison_tools import (
            _PERCENT_CHANGE_FLOOR_FRACTION,
            _anomaly_stats,
            _difference,
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
        from tools.satellite_tools.comparison_tools import (
            _PERCENT_CHANGE_FLOOR_FRACTION,
            _anomaly_stats,
            _difference,
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
        from tools.satellite_tools.comparison_tools import _split_aligned

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
        from tools.satellite_tools.comparison_tools import _split_aligned

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
        from tools.satellite_tools.comparison_tools import _split_aligned

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
        from tools.satellite_tools.comparison_tools import _split_aligned

        da = xr.DataArray([[1.0, 2.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [30.0, 40.0]})

        with self.assertRaises(ValueError):
            _split_aligned(da)

    def test_rejects_an_aligned_result_with_the_wrong_number_of_sources(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _split_aligned

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
        from tools.satellite_tools.comparison_tools import _region_stats

        da = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))

        stats = _region_stats(da)

        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["n_pixels"], 4)

    def test_returns_none_when_no_valid_cells(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _region_stats

        da = xr.DataArray([np.nan, np.nan], dims=("x",))

        self.assertIsNone(_region_stats(da))

    def test_extremes_disclose_the_field_they_summarize(self):
        """Finding #12: compare region max/min are order statistics of the
        *period-mean* field (compare always time-means each side), so they sit
        below the instantaneous peak. A bare "Max"/"Min" label invites reading
        them as the true peak/trough. When the caller names the field basis, the
        stats disclose it so the label can be qualified rather than bare."""
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _region_stats

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
        from tools.satellite_tools.comparison_tools import _region_stats

        stats = _region_stats(xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon")))

        self.assertNotIn("basis", stats)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class EmptyOverlapTests(unittest.TestCase):
    def test_returns_none_when_data_has_finite_values(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _empty_overlap_error

        da = xr.DataArray([1.0, 2.0], dims=("x",))

        self.assertIsNone(_empty_overlap_error(da, "A"))

    def test_returns_an_error_naming_the_side_when_all_values_are_missing(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _empty_overlap_error

        da = xr.DataArray([np.nan, np.nan], dims=("x",))

        error = _empty_overlap_error(da, "A")

        self.assertIsNotNone(error)
        self.assertIn("A", error)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "comparison-tool dependencies are not installed",
)
class DisjointPeriodsTests(unittest.TestCase):
    def test_overlapping_time_ranges_are_not_disjoint(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-01", periods=2)})
        da_b = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-02", periods=2)})

        self.assertIsNone(_disjoint_periods_error(da_a, da_b))

    def test_non_overlapping_time_ranges_are_rejected(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-01-01", periods=2)})
        da_b = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": pd.date_range("2024-06-01", periods=2)})

        self.assertIsNotNone(_disjoint_periods_error(da_a, da_b))

    def test_no_time_dimension_on_either_side_is_not_disjoint(self):
        import xarray as xr
        from tools.satellite_tools.comparison_tools import _disjoint_periods_error

        da_a = xr.DataArray([1.0, 2.0], dims=("x",))
        da_b = xr.DataArray([1.0, 2.0], dims=("x",))

        self.assertIsNone(_disjoint_periods_error(da_a, da_b))


_RENDER_REQUIRED_MODULES = REQUIRED_MODULES + [
    "shapely", "rasterio", "cartopy", "affine", "zarr", "fastmcp", "uvicorn",
    "langchain_mcp_adapters",
]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in _RENDER_REQUIRED_MODULES),
    "comparison scale-disclosure test dependencies are not installed",
)
class ComparisonScaleDisclosureTests(unittest.TestCase):
    """Issue #3: comparison panels use percentile clips (2/98 shared, 98th-pct
    magnitude diverging) but passed them via ``value_range``, which stamped
    scale={"method":"explicit"} -- so the legend's saturation warning the rest
    of the app makes never fired for compare maps. The clip must be disclosed."""

    def _2d(self, values):
        import numpy as np
        import xarray as xr

        arr = np.asarray(values, dtype=float)
        n_lat, n_lon = arr.shape
        return xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={
                "lat": np.linspace(10, 20, n_lat),
                "lon": np.linspace(-100, -90, n_lon),
            },
            name="no2",
            attrs={"units": "mol/m^2"},
        )

    def test_region_panels_disclose_the_shared_2_98_percentile_clip(self):
        from unittest.mock import patch
        from tools.satellite_tools.comparison_tools import _build_region_comparison

        da_a = self._2d([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        da_b = self._2d([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_region_comparison("h_a", "h_b", da_a, da_b, "A", "B", "no2", "mol/m^2")

        for panel in emitted["payload"]["panels"]:
            self.assertEqual(panel["scale"], {"method": "percentile", "p": [2, 98]})

    def test_period_difference_discloses_the_diverging_magnitude_percentile_clip(self):
        from unittest.mock import patch
        from tools.satellite_tools.comparison_tools import _build_period_comparison

        aligned_a = self._2d([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        aligned_b = self._2d([[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_period_comparison(
                "h_a", "h_b", "h_aligned", aligned_a, aligned_b, "A", "B", "no2", "mol/m^2", None,
            )

        self.assertEqual(
            emitted["payload"]["difference"]["scale"],
            {"method": "percentile_magnitude", "p": 98},
        )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in _RENDER_REQUIRED_MODULES),
    "comparison disclosure test dependencies are not installed",
)
class ComparisonDisclosureTests(unittest.TestCase):
    """Finding #4: compare silently applied QA-flag masking (via
    _prepare_2d -> aggregate) yet attached neither masking provenance nor the
    T46 scope echo the heatmap/timeseries paths always attach. Both disclosure
    doctrines must reach the compare payload — masking so the frontend's
    resolveMasking finds it, scope so subagent_dispatch's scope note can fire."""

    def _2d(self, values, name="no2"):
        import numpy as np
        import xarray as xr

        arr = np.asarray(values, dtype=float)
        n_lat, n_lon = arr.shape
        return xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, n_lat), "lon": np.linspace(-100, -90, n_lon)},
            name=name,
            attrs={"units": "mol/m^2"},
        )

    def _record_scope(self, handle, scope):
        from services import scope_registry

        scope_registry.record_pending(handle, scope)
        scope_registry.finalize(handle, handle)
        self.addCleanup(lambda: scope_registry._scopes.pop(handle, None))

    def test_region_comparison_discloses_qa_masking_provenance(self):
        from tools.satellite_tools.comparison_tools import _build_region_comparison

        da_a = self._2d([[1.0, 2.0], [3.0, 4.0]])
        da_b = self._2d([[2.0, 4.0], [6.0, 8.0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_region_comparison("h_a", "h_b", da_a, da_b, "A", "B", "no2", "mol/m^2")

        # The masking that aggregate() applied is disclosed on the payload where
        # the frontend's resolveMasking (chart.provenance?.masking) can find it,
        # instead of running silently.
        masking = emitted["payload"]["provenance"]["masking"]
        self.assertTrue(masking.get("qa_status"))
        self.assertIn("valid_range_source", masking)

    def test_period_difference_discloses_qa_masking_provenance(self):
        from tools.satellite_tools.comparison_tools import _build_period_comparison

        aligned_a = self._2d([[1.0, 2.0], [3.0, 4.0]])
        aligned_b = self._2d([[2.0, 4.0], [6.0, 8.0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_period_comparison(
                "h_a", "h_b", "h_aligned", aligned_a, aligned_b, "A", "B", "no2", "mol/m^2", None,
            )

        masking = emitted["payload"]["provenance"]["masking"]
        self.assertTrue(masking.get("qa_status"))

    def _flagged_side(self, values, flags):
        """A TEMPO_NO2-shaped Dataset whose ``short_name`` matches the pinned
        collections.yaml QA rule, so masking actually runs on this side."""
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {
                "no2": (
                    ("lat", "lon"), np.asarray(values, dtype=float), {"units": "mol/m^2"},
                ),
                "main_data_quality_flag": (
                    ("lat", "lon"), np.asarray(flags, dtype="int64"),
                ),
            },
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
            attrs={"short_name": "TEMPO_NO2_L3"},
        )

    def test_each_compare_side_carries_its_own_labelled_pass_rate(self):
        """T55 on compare: each side gets its own realized pass rate, attached
        to the panel that names it. Two pass rates with no way to tell which
        period or region each belongs to would be worse than showing none."""
        from tools.satellite_tools.comparison_tools import _build_region_comparison

        # Side A: 1 of 4 pixels passes. Side B: all 4 pass.
        ds_a = self._flagged_side([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [1, 1]])
        ds_b = self._flagged_side([[2.0, 4.0], [6.0, 8.0]], [[0, 0], [0, 0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_region_comparison(
                "h_a", "h_b", ds_a["no2"], ds_b["no2"], "June", "July", "no2", "mol/m^2",
                ds_a=ds_a, ds_b=ds_b,
            )

        panel_a, panel_b = emitted["payload"]["panels"]
        self.assertEqual(panel_a["title"], "June")
        self.assertEqual(panel_b["title"], "July")
        self.assertLess(panel_a["provenance"]["masking"]["qa_pass_rate"], 0.5)
        self.assertAlmostEqual(panel_b["provenance"]["masking"]["qa_pass_rate"], 1.0)
        self.assertEqual(panel_a["provenance"]["masking"]["qa_passing_pixels"], 1)
        self.assertEqual(panel_b["provenance"]["masking"]["qa_passing_pixels"], 4)

    def test_region_comparison_surfaces_scope_echo_so_the_dispatch_note_fires(self):
        import numpy as np
        import xarray as xr
        from config.error_templates import render_scope_note
        from tools.satellite_tools.comparison_tools import _build_region_comparison

        # Delivered data spans all of June; the recorded request was a single day
        # — the T46 substitution the compare surface previously hid entirely.
        da = xr.DataArray(
            np.ones((3, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={
                "time": ["2026-06-01", "2026-06-15", "2026-06-30"],
                "lat": [10.0, 20.0],
                "lon": [-100.0, -90.0],
            },
            name="no2",
            attrs={"units": "mol/m^2"},
        )
        self._record_scope("h_a", {"location": "California", "time_range": ["2026-06-15", "2026-06-15"]})

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_region_comparison("h_a", "h_b", da, da, "A", "B", "no2", "mol/m^2")

        prov = emitted["payload"]["provenance"]
        note = render_scope_note(prov.get("requested_scope"), prov.get("delivered_scope"))
        self.assertIsNotNone(note)
        self.assertIn("2026-06-15", note)  # the single day the user asked for
        self.assertIn("2026-06-01", note)  # the wider span actually delivered

    def test_period_comparison_surfaces_scope_echo(self):
        from config.error_templates import render_scope_note
        from tools.satellite_tools.comparison_tools import _build_period_comparison

        # Aligned slices carry their coverage as global attrs (the timeless-L3
        # fallback), spanning June; the recorded request was one day.
        aligned_a = self._2d([[1.0, 2.0], [3.0, 4.0]])
        aligned_a.attrs["time_coverage_start"] = "2026-06-01"
        aligned_a.attrs["time_coverage_end"] = "2026-06-30"
        aligned_b = self._2d([[2.0, 4.0], [6.0, 8.0]])
        self._record_scope("h_a", {"location": "Texas", "time_range": ["2026-06-15", "2026-06-15"]})

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_period_comparison(
                "h_a", "h_b", "h_aligned", aligned_a, aligned_b, "A", "B", "no2", "mol/m^2", None,
            )

        prov = emitted["payload"]["provenance"]
        note = render_scope_note(prov.get("requested_scope"), prov.get("delivered_scope"))
        self.assertIsNotNone(note)
        self.assertIn("2026-06-15", note)

    def test_no_scope_note_when_nothing_was_recorded_for_the_handles(self):
        from config.error_templates import render_scope_note
        from tools.satellite_tools.comparison_tools import _build_region_comparison

        da_a = self._2d([[1.0, 2.0], [3.0, 4.0]])
        da_b = self._2d([[2.0, 4.0], [6.0, 8.0]])

        emitted = {}
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            _build_region_comparison("h_a", "h_b", da_a, da_b, "A", "B", "no2", "mol/m^2")

        prov = emitted["payload"]["provenance"]
        # requested_scope is None (nothing recorded) -> the note stays silent,
        # never nagging on a comparison with no recorded request to check against.
        self.assertIsNone(prov.get("requested_scope"))
        self.assertIsNone(render_scope_note(prov.get("requested_scope"), prov.get("delivered_scope")))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in FULL_TOOL_REQUIRED_MODULES),
    "full compare tool test dependencies are not installed",
)
class CompareToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)
        self._align_handler = None

        async def _align(source_handles, method="outer", workspace_id="default"):
            return await self._align_handler(source_handles)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
            "align": _align,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def test_region_mode_produces_side_by_side_panels_on_a_shared_scale(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_a():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_b():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[10.0, 20.0], [30.0, 40.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_a", make_a)
        self.volume.add_zarr("obs_b", make_b)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await compare.ainvoke({
                "handle_a": "obs_a", "handle_b": "obs_b", "mode": "region",
                "label_a": "Newark", "label_b": "Philly",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)

        # (a) the full panel/stats detail still reaches the frontend pipeline,
        # out-of-band from the model-facing return value (T13).
        full = emitted["payload"]
        self.assertEqual(full["type"], "heatmap_multi")
        self.assertEqual(full["mode"], "n-panel")
        self.assertEqual(len(full["panels"]), 2)
        # Shared color scale across both panels (region mode never differences).
        self.assertEqual(full["panels"][0]["vmin"], full["panels"][1]["vmin"])
        self.assertEqual(full["panels"][0]["vmax"], full["panels"][1]["vmax"])
        # Means are cos(latitude) area-weighted: rows at lat=10 ([1,2]) and
        # lat=20 ([3,4]); Philly's fixture is 10x Newark's.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected_newark = ((1.0 + 2.0) * w10 + (3.0 + 4.0) * w20) / (2 * (w10 + w20))
        self.assertAlmostEqual(full["stats"]["Newark"]["mean"], expected_newark)
        self.assertAlmostEqual(full["stats"]["Philly"]["mean"], 10.0 * expected_newark)
        # Finding #12: each region's extremes disclose they summarize the
        # period-mean map, not the instantaneous peak.
        self.assertEqual(full["stats"]["Newark"]["basis"], "period-mean")
        self.assertEqual(full["stats"]["Philly"]["basis"], "period-mean")
        self.assertNotIn("difference", full)

        # T23: each panel gets its own rendered overlay, colorized against
        # the *shared* vmin/vmax (not each panel's own percentile bounds) --
        # the anti-drift guarantee between the map and its legend.
        chart_id = result["chart_id"]
        self.assertEqual(full["panels"][0]["overlay"]["url"], f"/chart/{chart_id}/overlay.png?panel=0")
        self.assertEqual(full["panels"][1]["overlay"]["url"], f"/chart/{chart_id}/overlay.png?panel=1")

        # (b) the model-facing result is the compact summary.
        self.assertEqual(result["render_type"], "heatmap_multi")
        for bulky_key in ("panels", "stats", "mode"):
            self.assertNotIn(bulky_key, result)

        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual(ref["metadata"]["mode"], "n-panel")
        self.assertEqual([p["handle"] for p in ref["metadata"]["panels"]], ["obs_a", "obs_b"])
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_a", "obs_b"])

    async def test_region_mode_discloses_masking_and_scope_echo_end_to_end(self):
        import numpy as np
        import xarray as xr
        from config.error_templates import render_scope_note
        from services import scope_registry
        from tools.satellite_tools import comparison_tools

        def make_ds():
            return xr.Dataset(
                {"no2": (
                    ("time", "lat", "lon"),
                    [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]],
                    {"units": "mol/m^2"},
                )},
                coords={
                    "time": np.array(["2026-06-01", "2026-06-30"], dtype="datetime64[ns]"),
                    "lat": [10.0, 20.0], "lon": [30.0, 40.0],
                },
            )

        self.volume.add_zarr("obs_a", make_ds)
        self.volume.add_zarr("obs_b", make_ds)
        # The researcher asked about a single day for side A; the retrieval
        # delivered the whole month — a T46 substitution compare used to hide.
        scope_registry.record_pending("obs_a", {"location": "Newark", "time_range": ["2026-06-15", "2026-06-15"]})
        scope_registry.finalize("obs_a", "obs_a")
        self.addCleanup(lambda: scope_registry._scopes.pop("obs_a", None))

        emitted = {}
        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            raw = await compare.ainvoke({
                "handle_a": "obs_a", "handle_b": "obs_b", "mode": "region",
                "label_a": "Newark", "label_b": "Philly",
            })
        self.assertNotIn("error", json.loads(raw))

        prov = emitted["payload"]["provenance"]
        # Masking is disclosed (never silent) ...
        self.assertTrue(prov["masking"].get("qa_status"))
        # ... and the scope substitution surfaces the way the dispatch layer reads it.
        note = render_scope_note(prov.get("requested_scope"), prov.get("delivered_scope"))
        self.assertIsNotNone(note)
        self.assertIn("2026-06-15", note)

    async def test_period_mode_differences_b_minus_a_via_mcp_align(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_a():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_b():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[2.0, 4.0], [6.0, 8.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_aligned():
            return xr.Dataset(
                {"no2": (
                    ("source", "lat", "lon"),
                    [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]],
                    {"units": "mol/m^2"},
                )},
                coords={"source": [0, 1], "lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_june25", make_a)
        self.volume.add_zarr("obs_june26", make_b)
        self.volume.add_zarr("cube_aligned1", make_aligned)

        async def _align(source_handles):
            self.assertEqual(source_handles, ["obs_june25", "obs_june26"])
            return {"handle": "cube_aligned1", "status": "ok", "alignment_report": {"method": "outer"}}

        self._align_handler = _align

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await compare.ainvoke({
                "handle_a": "obs_june25", "handle_b": "obs_june26", "mode": "period",
                "label_a": "June 2025", "label_b": "June 2026",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)

        # (a) the full difference grid/stats still reach the frontend pipeline.
        full = emitted["payload"]
        self.assertEqual(full["mode"], "difference")
        # b - a: [[1,2],[3,4]] doubled -> diff = a itself: [[1,2],[3,4]]
        self.assertEqual(full["difference"]["values"], [[1.0, 2.0], [3.0, 4.0]])
        # mean_difference is cos(latitude) area-weighted (lat rows 10/20);
        # percent change stays exactly 100% since b = 2a cell-for-cell.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected_mean_diff = ((1.0 + 2.0) * w10 + (3.0 + 4.0) * w20) / (2 * (w10 + w20))
        self.assertAlmostEqual(full["stats"]["mean_difference"], expected_mean_diff)
        self.assertAlmostEqual(full["stats"]["percent_change"], 100.0)
        # Diverging, zero-centered scale.
        self.assertAlmostEqual(full["difference"]["vmin"], -full["difference"]["vmax"])

        # T23: the difference panel gets a rendered overlay too.
        self.assertEqual(full["difference"]["overlay"]["url"], f"/chart/{result['chart_id']}/overlay.png")

        # (b) the model-facing result is the compact summary, using the
        # difference grid's own dimensions/value range (T13).
        self.assertEqual(result["render_type"], "heatmap_multi")
        self.assertEqual(result["grid_dims"], [2, 2])
        self.assertAlmostEqual(result["vmin"], full["difference"]["vmin"])
        self.assertAlmostEqual(result["vmax"], full["difference"]["vmax"])
        for bulky_key in ("panels", "stats", "mode", "difference"):
            self.assertNotIn(bulky_key, result)

        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual(ref["metadata"]["mode"], "difference")
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_june25", "obs_june26", "cube_aligned1"])

    async def test_mismatched_variables_are_rejected_with_a_plain_explanation(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_no2():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        def make_hcho():
            return xr.Dataset(
                {"hcho": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_no2", make_no2)
        self.volume.add_zarr("obs_hcho", make_hcho)

        compare = comparison_tools.make_compare(self.mcp_tools)
        raw = await compare.ainvoke({"handle_a": "obs_no2", "handle_b": "obs_hcho", "mode": "region"})
        result = json.loads(raw)

        self.assertIn("error", result)
        self.assertIn("no2", result["error"])
        self.assertIn("hcho", result["error"])

    async def test_compare_opens_both_sides_concurrently(self):
        """Opening handle A and handle B are independent MCP round trips
        (export -> download/open); compare must gather them instead of
        awaiting one after another, so the wall-clock wait is close to the
        slower side alone, not the sum of both."""
        import asyncio
        import time
        from unittest.mock import AsyncMock

        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_ds(value):
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[value, value], [value, value]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        async def slow_open(handle, tools):
            await asyncio.sleep(0.3)
            return make_ds(1.0 if handle == "obs_a" else 2.0)

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: None), \
             patch.object(comparison_tools, "open_handle", AsyncMock(side_effect=slow_open)):
            start = time.monotonic()
            raw = await compare.ainvoke({"handle_a": "obs_a", "handle_b": "obs_b", "mode": "region"})
            elapsed = time.monotonic() - start

        result = json.loads(raw)
        self.assertNotIn("error", result)
        # Sequential opens would take >=0.6s; concurrent should land near 0.3s.
        self.assertLess(elapsed, 0.5)

    async def test_compare_does_not_emit_a_picker_for_a_side_discarded_after_the_other_fails(self):
        """Both sides now run concurrently via asyncio.gather, so side B still
        runs to completion even when side A has already failed outright.
        Only A's error is returned (make_compare checks err_a first) — B's
        variable-choice picker (an out-of-band, irreversible emission) must
        NOT fire in that case, since the caller discards B's result
        entirely and the picker would have no matching returned state for a
        client to act on."""
        from unittest.mock import AsyncMock, patch

        import xarray as xr
        from earthdata_mcp.results import MCPToolError
        from preprocessing.aggregation_service import VariableChoiceRequired
        from services.open_handle import OpenHandleError
        from tools.satellite_tools import comparison_tools

        def make_ds():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        async def open_side(handle, tools):
            if handle == "obs_bad":
                raise OpenHandleError("simulated open failure")
            return make_ds()

        fake_resolution = object()
        fake_mcp_error = MCPToolError("contract", "ambiguous variable")

        def fake_to_dataarray(ds, *, handle=None, variable=None):
            if handle == "obs_ambiguous":
                raise VariableChoiceRequired(fake_resolution, fake_mcp_error)
            return ds["no2"]

        emitted = []

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: None), \
             patch.object(comparison_tools, "open_handle", AsyncMock(side_effect=open_side)), \
             patch.object(
                 comparison_tools._aggregation_service, "to_dataarray", side_effect=fake_to_dataarray
             ), \
             patch.object(
                 comparison_tools, "emit_variable_choice_payload",
                 lambda resolution, ds: emitted.append(resolution),
             ):
            raw = await compare.ainvoke({
                "handle_a": "obs_bad", "handle_b": "obs_ambiguous", "mode": "region",
            })

        result = json.loads(raw)
        self.assertIn("error", result)
        self.assertIn("obs_bad", result["error"])
        # B's picker must never have fired -- its result was discarded.
        self.assertEqual(emitted, [])

    async def test_compare_still_emits_a_picker_when_that_side_is_the_one_returned(self):
        """The deferred-emission fix must not suppress a picker that IS the
        winning (returned) error -- only a discarded side's picker should be
        skipped."""
        from unittest.mock import AsyncMock, patch

        import xarray as xr
        from earthdata_mcp.results import MCPToolError
        from preprocessing.aggregation_service import VariableChoiceRequired
        from tools.satellite_tools import comparison_tools

        def make_ds():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        async def open_side(handle, tools):
            return make_ds()

        fake_resolution = object()
        fake_mcp_error = MCPToolError("contract", "ambiguous variable")

        def fake_to_dataarray(ds, *, handle=None, variable=None):
            if handle == "obs_ambiguous":
                raise VariableChoiceRequired(fake_resolution, fake_mcp_error)
            return ds["no2"]

        emitted = []

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tools.satellite_tools.plot_tools.emit_chart", lambda p: None), \
             patch.object(comparison_tools, "open_handle", AsyncMock(side_effect=open_side)), \
             patch.object(
                 comparison_tools._aggregation_service, "to_dataarray", side_effect=fake_to_dataarray
             ), \
             patch.object(
                 comparison_tools, "emit_variable_choice_payload",
                 lambda resolution, ds: emitted.append(resolution),
             ):
            raw = await compare.ainvoke({
                "handle_a": "obs_ok", "handle_b": "obs_ambiguous", "mode": "region",
            })

        result = json.loads(raw)
        self.assertIn("error", result)
        self.assertEqual(emitted, [fake_resolution])

    async def test_an_unknown_mode_is_rejected(self):
        import xarray as xr
        from tools.satellite_tools import comparison_tools

        def make_ds():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_x", make_ds)
        self.volume.add_zarr("obs_y", make_ds)

        compare = comparison_tools.make_compare(self.mcp_tools)
        raw = await compare.ainvoke({"handle_a": "obs_x", "handle_b": "obs_y", "mode": "bogus"})
        result = json.loads(raw)

        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
