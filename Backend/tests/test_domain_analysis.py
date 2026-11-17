"""domain.analysis on small synthetic arrays: checks, differences, reductions, statistics, peaks, thresholds."""
import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.datasets import standards
from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean
from tta_backend.domain.analysis import checks, comparison, peaks, reduction, statistics, thresholds

LATS = [30.0, 60.0]
LONS = [-100.0, -99.0, -98.0]


def field(values, *, lats=LATS, lons=LONS, name="no2", units="ppb"):
    return xr.DataArray(np.asarray(values, dtype=float), dims=("lat", "lon"), coords={"lat": lats, "lon": lons}, name=name, attrs={"units": units})


def cube(values, times, *, lats=LATS, lons=LONS, name="no2"):
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(times).values, "lat": lats, "lon": lons}, name=name, attrs={"units": "ppb"},
    )


class CheckTests(unittest.TestCase):
    def test_variables_differ_only_when_both_are_named(self):
        self.assertIsNone(checks.variable_mismatch("no2", "no2"))
        self.assertIsNone(checks.variable_mismatch("no2", None))
        mismatch = checks.variable_mismatch("no2", " hcho ")
        self.assertEqual((mismatch.reason, mismatch.facts), ("variable_mismatch", {"a": "no2", "b": "hcho"}))

    def test_units_must_match_and_one_sided_units_are_a_mismatch(self):
        self.assertIsNone(checks.units_mismatch("ppb", "ppb"))
        self.assertIsNone(checks.units_mismatch("", None))
        self.assertEqual(checks.units_mismatch("ppb", "ppm").reason, "unit_mismatch")
        missing = checks.units_mismatch(None, "ppm")
        self.assertEqual((missing.reason, missing.facts), ("unit_missing", {"declared": "ppm", "missing_side": "A"}))

    def test_disjoint_periods_and_unknown_ranges(self):
        self.assertIsNone(checks.disjoint_periods(("2024-06-01", "2024-06-05"), ("2024-06-05", "2024-06-09")))
        self.assertIsNone(checks.disjoint_periods((None, None), ("2024-06-05", "2024-06-09")))
        found = checks.disjoint_periods(("2024-06-01T00", "2024-06-02T00"), ("2024-07-01T00", "2024-07-02T00"))
        self.assertEqual(found.facts, {"start_a": "2024-06-01", "end_a": "2024-06-02", "start_b": "2024-07-01", "end_b": "2024-07-02"})

    def test_grids_must_line_up_cell_for_cell(self):
        coarse_lat, coarse_lon = np.arange(30.0, 40.1, 1.0), np.arange(-100.0, -89.9, 1.0)
        fine_lat, fine_lon = np.arange(30.0, 40.01, 0.1), np.arange(-100.0, -89.99, 0.1)
        self.assertIsNone(checks.grid_mismatch(coarse_lat, coarse_lon, coarse_lat, coarse_lon))
        # A subset of the same lattice, and a hair of floating-point noise, still line up.
        self.assertIsNone(checks.grid_mismatch(coarse_lat, coarse_lon, coarse_lat[2:5] + 1e-9, coarse_lon[3:]))
        for lat_b, lon_b in ((fine_lat, fine_lon), (coarse_lat + 0.5, coarse_lon)):
            found = checks.grid_mismatch(coarse_lat, coarse_lon, lat_b, lon_b)
            self.assertEqual(found.reason, "grid_mismatch")
        found = checks.grid_mismatch(fine_lat, fine_lon, coarse_lat, coarse_lon)
        self.assertEqual(found.facts, {"spacing_a": "0.1 by 0.1 degree", "spacing_b": "1 by 1 degree"})
        # Grids that do not overlap are not a mismatch; there is just nothing to pair.
        self.assertIsNone(checks.grid_mismatch(coarse_lat, coarse_lon, coarse_lat, coarse_lon + 50.5))

    def test_time_range_and_valid_data(self):
        da = cube(np.ones((2, 2, 3)), ["2024-06-02", "2024-06-01"])
        start, end = checks.time_range(da)
        self.assertTrue(start.startswith("2024-06-01") and end.startswith("2024-06-02"))
        self.assertEqual(checks.time_range(field(np.ones((2, 3)))), (None, None))
        self.assertTrue(checks.has_valid_data(field([[np.nan, 1, np.nan], [np.nan] * 3])))
        self.assertFalse(checks.has_valid_data(field(np.full((2, 3), np.nan))))


class ComparisonTests(unittest.TestCase):
    def test_pair_cells_on_identical_grids_keeps_both(self):
        a, b = field(np.ones((2, 3))), field(np.full((2, 3), 2.0))
        pa, pb = comparison.pair_cells(a, b)
        np.testing.assert_array_equal(pb.values, b.values)
        self.assertEqual(comparison.shared_cells(a, b), 6)

    def test_pair_cells_takes_the_nearest_cell_within_half_a_spacing(self):
        a = field(np.ones((2, 3)))
        # Shifted by a hair, and one column further east than A reaches.
        b = field([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], lats=[30.0001, 60.0001], lons=[-99.0001, -98.0001, -97.0001])
        _, pb = comparison.pair_cells(a, b)
        np.testing.assert_array_equal(np.isnan(pb.values[:, 0]), [True, True])
        np.testing.assert_array_equal(pb.values[:, 1:], [[1.0, 2.0], [4.0, 5.0]])
        self.assertEqual(comparison.shared_cells(a, b), 4)

    def test_grids_with_no_cell_in_common_share_none(self):
        a = field(np.ones((2, 3)))
        b = field(np.ones((2, 3)), lons=[10.0, 11.0, 12.0])
        self.assertEqual(comparison.shared_cells(a, b), 0)

    def test_percent_change_withholds_cells_under_the_floor(self):
        a = xr.DataArray(np.array([0.0, 100.0, 200.0, 1.0]), dims=("x",))
        b = xr.DataArray(np.array([1.0, 150.0, 100.0, 2.0]), dims=("x",))
        percent, withheld = comparison.percent_change_field(a, b)
        # Floor is 5% of A's 2-98th percentile range (~9.6): 0 and 1 are under it.
        self.assertEqual(withheld, 2)
        np.testing.assert_allclose(percent.values[1:3], [50.0, -50.0])
        self.assertTrue(np.isnan(percent.values[[0, 3]]).all())

    def test_field_summary_names_the_extreme_cells(self):
        da = field([[1.0, np.nan, 3.0], [9.0, -2.0, 4.0]])
        summary = comparison.field_summary(da)
        self.assertEqual((summary["max"], summary["min"], summary["cells"]), (9.0, -2.0, 5))
        self.assertEqual(summary["max_location"], {"lat": 60.0, "lon": -100.0})
        self.assertEqual(summary["min_location"], {"lat": 60.0, "lon": -99.0})
        self.assertAlmostEqual(summary["mean"], area_weighted_mean(da))
        self.assertIsNone(comparison.field_summary(field(np.full((2, 3), np.nan))))


class ReductionTests(unittest.TestCase):
    def test_a_mean_weights_each_day_equally(self):
        # Three granules on day one (values 1) and one on day two (value 5).
        times = ["2024-06-01T10", "2024-06-01T11", "2024-06-01T12", "2024-06-02T10"]
        values = np.stack([np.full((2, 3), v) for v in (1.0, 1.0, 1.0, 5.0)])
        reduced, steps = reduction.reduce_over_time(cube(values, times), "mean", "daily")
        np.testing.assert_allclose(reduced.values, 3.0)
        self.assertEqual(steps, [0, 1, 2, 3])
        plain, _ = reduction.reduce_over_time(cube(values, times), "mean", "unknown")
        np.testing.assert_allclose(plain.values, 2.0)

    def test_empty_steps_are_skipped_and_other_stats_are_per_cell(self):
        values = np.stack([np.full((2, 3), np.nan), np.full((2, 3), 2.0), np.full((2, 3), 7.0)])
        reduced, steps = reduction.reduce_over_time(cube(values, ["2024-06-01", "2024-06-02", "2024-06-03"]), "max", "daily")
        self.assertEqual(steps, [1, 2])
        np.testing.assert_allclose(reduced.values, 7.0)
        self.assertEqual(reduced.attrs["units"], "ppb")

    def test_a_field_without_time_is_returned_as_is(self):
        da = field(np.ones((2, 3)))
        reduced, steps = reduction.reduce_over_time(da, "mean", "daily")
        self.assertIs(reduced, da)
        self.assertEqual(steps, [0])

    def test_an_unknown_statistic_is_refused(self):
        with self.assertRaises(ValueError):
            reduction.reduce_over_time(field(np.ones((2, 3))), "mode", "daily")

    def test_regional_series_is_area_weighted_and_drops_empty_steps(self):
        day_one = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        values = np.stack([day_one, np.full((2, 3), np.nan)])
        frame = reduction.regional_series(cube(values, ["2024-06-02", "2024-06-01"]), "mean")
        self.assertEqual(len(frame), 1)
        w30, w60 = math.cos(math.radians(30)), math.cos(math.radians(60))
        self.assertAlmostEqual(frame["value"].iloc[0], (w30 * 1 + w60 * 3) / (w30 + w60))
        self.assertEqual(frame["time"].iloc[0], pd.Timestamp("2024-06-02"))


class StatisticTests(unittest.TestCase):
    def setUp(self):
        values = np.stack([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[3.0, 2.0, 1.0], [6.0, 5.0, 10.0]]])
        self.da = cube(values, ["2024-06-01", "2024-06-02"])

    def test_max_is_over_every_observation_and_mean_is_area_weighted(self):
        top = statistics.grid_statistic(self.da, "max", "daily")
        self.assertEqual(top["value"], 10.0)
        self.assertIn("every valid observation", top["basis"])
        mean = statistics.grid_statistic(self.da, "mean", "daily")
        mean_field = self.da.mean("time")
        self.assertAlmostEqual(mean["value"], area_weighted_mean(mean_field))
        self.assertEqual(mean["count"], 6)
        self.assertAlmostEqual(mean["spread"], float(np.std(mean_field.values, ddof=1)))
        self.assertEqual(mean["spread_basis"], statistics.GRID_SPREAD_BASIS)
        self.assertNotIn("std", mean)

    def test_percentile_of_the_temporal_mean_field(self):
        result = statistics.grid_statistic(self.da, "percentile", "daily", 50.0)
        self.assertAlmostEqual(result["value"], float(np.percentile(self.da.mean("time").values, 50)))
        self.assertIn("50th percentile", result["basis"])

    def test_no_valid_cell_gives_none(self):
        self.assertIsNone(statistics.grid_statistic(self.da.where(False), "mean", "daily"))
        self.assertIsNone(statistics.series_statistic(np.array([np.nan]), "mean"))

    def test_series_statistic_and_a_single_value_has_no_spread(self):
        result = statistics.series_statistic(np.array([1.0, np.nan, 3.0]), "mean")
        self.assertEqual((result["value"], result["count"]), (2.0, 2))
        self.assertAlmostEqual(result["spread"], math.sqrt(2))
        self.assertEqual(result["spread_basis"], statistics.SERIES_SPREAD_BASIS)
        self.assertIsNone(statistics.series_statistic(np.array([4.0]), "median")["spread"])

    def test_basis_statistic_matches_the_stat_tool_bases(self):
        mean_field = field([[1.0, np.nan, 3.0], [5.0, 7.0, 9.0]])
        value, basis = statistics.basis_statistic("median", mean_field)
        self.assertEqual(value, 5.0)
        self.assertEqual(basis, "median of the temporal-mean field (per-cell mean over time first)")
        value, _ = statistics.basis_statistic("min", mean_field, field(np.full((2, 3), -1.0)))
        self.assertEqual(value, -1.0)
        with self.assertRaises(ValueError):
            statistics.basis_statistic("max", mean_field)

    def test_peak_cell(self):
        peak = statistics.peak_cell(field([[1.0, 9.0, np.nan], [2.0, 3.0, 4.0]]))
        self.assertEqual(peak, statistics.Peak(value=9.0, lat=30.0, lon=-99.0))
        self.assertIsNone(statistics.peak_cell(field(np.full((2, 3), np.nan))))
        with self.assertRaises(statistics.LatLonNotFound):
            statistics.peak_cell(xr.DataArray(np.ones((2, 2)), dims=("a", "b")))

    def test_only_a_failed_coordinate_read_becomes_peak_coordinates_unreadable(self):
        da = field([[1.0, 9.0, 2.0], [2.0, 3.0, 4.0]])
        # A peak index past the latitude axis fails when its coordinate is read.
        with mock.patch.object(np, "unravel_index", return_value=(5, 0)):
            with self.assertRaises(statistics.PeakCoordinatesUnreadable) as caught:
                statistics.peak_cell(da)
        self.assertIn("out of bounds", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, IndexError)
        # The same error anywhere else in the search propagates unchanged.
        with mock.patch.object(np, "nanargmax", side_effect=IndexError("not a coordinate read")):
            with self.assertRaises(IndexError):
                statistics.peak_cell(da)


class PeakTests(unittest.TestCase):
    def test_daily_peaks_of_a_grid_name_the_time_and_cell(self):
        values = np.stack([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0], [4.0, 50.0, 6.0]],
            [[7.0, np.nan, 1.0], [1.0, 1.0, 1.0]],
            np.full((2, 3), np.nan),
        ])
        times = ["2024-06-01T10", "2024-06-01T14", "2024-06-02T10", "2024-06-03T10"]
        table = peaks.daily_peaks_grid(cube(values, times))
        self.assertEqual(table["date"].tolist(), ["2024-06-01", "2024-06-02"])
        first = table.iloc[0]
        self.assertEqual((first["value"], first["lat"], first["lon"]), (50.0, 60.0, -99.0))
        self.assertTrue(first["time"].startswith("2024-06-01T14"))
        self.assertEqual(table.iloc[1]["value"], 7.0)

    def test_daily_peaks_of_a_series_name_the_line(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2024-06-01T10Z", "2024-06-01T11Z", "2024-06-02T10Z", "2024-06-02T12Z"]),
            "value": [1.0, 4.0, np.nan, 2.0],
            "series": ["a", "b", "a", "a"],
        })
        table = peaks.daily_peaks_series(frame)
        self.assertEqual(table[["date", "value", "series"]].values.tolist(), [["2024-06-01", 4.0, "b"], ["2024-06-02", 2.0, "a"]])
        self.assertTrue(peaks.daily_peaks_series(frame.iloc[0:0]).empty)


class ThresholdTests(unittest.TestCase):
    def test_grid_exceedances_report_area_fraction_and_days(self):
        day_one = [[1.0, 6.0, 1.0], [1.0, 1.0, 1.0]]
        day_two = [[1.0, 1.0, 1.0], [7.0, np.nan, 1.0]]
        day_three = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        table, overall = thresholds.grid_exceedances(cube([day_one, day_two, day_three], ["2024-06-01", "2024-06-02", "2024-06-03"]), 5.0)
        self.assertEqual(table["cells_above"].tolist(), [1, 1, 0])
        self.assertEqual(table["valid_cells"].tolist(), [6, 5, 6])
        w30, w60 = math.cos(math.radians(30)), math.cos(math.radians(60))
        self.assertAlmostEqual(table["area_fraction_above"].iloc[0], w30 / (3 * w30 + 3 * w60))
        self.assertEqual((table["max"].iloc[1], table["max_lat"].iloc[1], table["max_lon"].iloc[1]), (7.0, 60.0, -100.0))
        self.assertEqual((overall["days_with_data"], overall["days_above"], overall["cells_ever_above"]), (3, 2, 2))
        self.assertAlmostEqual(overall["area_fraction_ever_above"], (w30 + w60) / (3 * w30 + 3 * w60))

    def test_a_value_equal_to_the_level_does_not_exceed_and_a_timeless_grid_is_one_row(self):
        table, overall = thresholds.grid_exceedances(field([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]), 5.0)
        self.assertEqual(len(table), 1)
        self.assertIsNone(table["date"].iloc[0])
        self.assertEqual(overall["days_above"], 0)

    def test_series_exceedances(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2024-06-01T10Z", "2024-06-01T11Z", "2024-06-02T10Z", "2024-06-03T10Z"]),
            "value": [0.08, 0.05, 0.09, np.nan],
            "series": ["a", "b", "b", "a"],
        })
        table, overall = thresholds.series_exceedances(frame, 0.07)
        self.assertEqual(table["date"].tolist(), ["2024-06-01", "2024-06-02"])
        self.assertEqual(overall, {"values_checked": 3, "values_above": 2, "days_with_data": 2, "days_above": 2, "lines_above": 2})

    def test_against_a_standard_values_are_truncated_as_the_aqi_truncates_them(self):
        self.assertEqual(standards.truncate_to_reporting_precision("o3", 0.0705), 0.070)
        self.assertEqual(standards.truncate_to_reporting_precision("o3", 0.0710), 0.071)
        self.assertEqual(standards.truncate_to_reporting_precision("pm25", 35.49), 35.4)
        self.assertTrue(math.isnan(standards.truncate_to_reporting_precision("o3", float("nan"))))
        with self.assertRaises(ValueError):
            standards.truncate_to_reporting_precision("lead", 1.0)
        # 0.0705 ppm is Moderate, so it is not above the 0.070 level either.
        self.assertEqual(standards.aqi_category("o3", 0.0705).category, "Moderate")
        np.testing.assert_array_equal(thresholds.truncate_for_standard("o3", [0.0705, np.nan, 0.0719]), [0.070, np.nan, 0.071])

        frame = pd.DataFrame({
            "time": pd.to_datetime(["2024-06-01T10Z", "2024-06-02T10Z"]), "value": [0.0705, 0.0712], "series": ["a", "a"],
        })
        truncated, overall = thresholds.series_exceedances(frame, 0.070, truncate_for="o3")
        self.assertEqual((truncated["value"].tolist(), overall["values_above"]), ([0.0712], 1))
        raw, _ = thresholds.series_exceedances(frame, 0.070)
        self.assertEqual(raw["value"].tolist(), [0.0705, 0.0712])

        grid = cube([[[0.0705, 0.06, 0.06], [0.06, 0.06, 0.06]]], ["2024-06-01"])
        table, overall = thresholds.grid_exceedances(grid, 0.070, truncate_for="o3")
        self.assertEqual((overall["days_above"], table["max"].iloc[0]), (0, 0.0705))
        self.assertEqual(thresholds.grid_exceedances(grid, 0.070)[1]["days_above"], 1)

    def test_naaqs_for_picks_or_lists_the_standards(self):
        self.assertEqual(thresholds.naaqs_for("o3", None).level, 0.070)
        self.assertEqual(thresholds.naaqs_for("pm25", None), ["annual", "24-hour"])
        self.assertEqual(thresholds.naaqs_for("pm25", "24-hour").level, 35)
        with self.assertRaises(thresholds.UnknownStandard):
            thresholds.naaqs_for("pm25", "1-hour")
        with self.assertRaises(thresholds.UnknownStandard):
            thresholds.naaqs_for("lead", None)


if __name__ == "__main__":
    unittest.main()


class SearchHitTests(unittest.TestCase):
    def test_a_hit_is_read_from_its_summary_as_the_service_sends_it(self):
        from tta_backend.domain.earthdata.retrieval import dataset_handle_from_search, search_hits

        reply = {"datasets": [
            {"handle": "dataset_a", "summary": {"concept_id": "C1-OTHER", "short_name": "OTHER"}},
            {"handle": "dataset_b", "summary": {
                "concept_id": "C3685896708-LARC_CLOUD", "short_name": "TEMPO_NO2_L3", "entry_title": "TEMPO NO2",
            }},
        ], "count": 2}
        self.assertEqual(dataset_handle_from_search(reply, "C3685896708-LARC_CLOUD"), "dataset_b")
        self.assertIsNone(dataset_handle_from_search(reply, "C9-MISSING"))
        self.assertEqual(search_hits(reply)[1]["short_name"], "TEMPO_NO2_L3")

    def test_a_size_estimate_is_read_from_its_megabytes(self):
        from tta_backend.domain.earthdata.retrieval import estimated_bytes

        self.assertEqual(estimated_bytes({"total_granules": 4, "total_size_mb": 2.0}), 2 * 1024 * 1024)
        # Granules to fetch but no size reported: unknown, so the user is asked.
        self.assertIsNone(estimated_bytes({"total_granules": 4, "total_size_mb": 0.0}))
        self.assertEqual(estimated_bytes({"total_granules": 0, "total_size_mb": 0.0}), 0)
        self.assertIsNone(estimated_bytes({"warning": "no estimate"}))
