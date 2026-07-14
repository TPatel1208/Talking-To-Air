"""
Tests for tools/satellite_tools/validation_tools.py (PRD T07 — satellite<->ground
validation workflow).

Hermetic at the analysis-tool seam: synthetic cube fixtures with known values
at known coordinates, exercised through the module's own helpers (prior art:
test_satellite_plot_payload.py testing _da_to_heatmap_payload directly).
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

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
    "satellite validation dependencies are not installed",
)
class NearestCellExtractionTests(unittest.TestCase):
    def test_extracts_value_at_exact_grid_point(self):
        import xarray as xr
        from tools.satellite_tools.validation_tools import _nearest_cell_series

        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        value = _nearest_cell_series(da, lat=20.0, lon=30.0)

        self.assertEqual(float(value), 3.0)

    def test_extracts_nearest_value_for_off_grid_point(self):
        import xarray as xr
        from tools.satellite_tools.validation_tools import _nearest_cell_series

        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        # (18.0, 32.0) is closer to grid point (20.0, 30.0) -> 3.0
        value = _nearest_cell_series(da, lat=18.0, lon=32.0)

        self.assertEqual(float(value), 3.0)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite validation dependencies are not installed",
)
class MonitorSeriesExtractionTests(unittest.TestCase):
    def test_excludes_fill_values_and_counts_coverage(self):
        import numpy as np
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.validation_tools import _extract_monitor_series

        times = pd.date_range("2024-01-01", periods=3, freq="D")
        values = np.array([
            [[1.0, 2.0]],
            [[-9999.0, 2.0]],  # fill at (lat=10, lon=30) on day 2
            [[3.0, 2.0]],
        ])
        da = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": [10.0], "lon": [30.0, 40.0]},
            attrs={"_FillValue": -9999.0},
        )

        times_out, values_out, coverage = _extract_monitor_series(da, lat=10.0, lon=30.0)

        self.assertEqual(values_out, [1.0, 3.0])
        self.assertEqual(len(times_out), 2)
        self.assertEqual(coverage["n_total"], 3)
        self.assertEqual(coverage["n_valid"], 2)
        self.assertEqual(coverage["n_excluded"], 1)
        self.assertAlmostEqual(coverage["coverage_fraction"], 2 / 3)

    def test_excludes_values_outside_valid_range(self):
        import numpy as np
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools.validation_tools import _extract_monitor_series

        times = pd.date_range("2024-01-01", periods=2, freq="D")
        values = np.array([[[500.0]], [[5.0]]])
        da = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": [10.0], "lon": [30.0]},
            attrs={"valid_min": 0.0, "valid_max": 100.0},
        )

        times_out, values_out, coverage = _extract_monitor_series(da, lat=10.0, lon=30.0)

        self.assertEqual(values_out, [5.0])
        self.assertEqual(coverage["n_total"], 2)
        self.assertEqual(coverage["n_valid"], 1)
        self.assertEqual(coverage["n_excluded"], 1)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite validation dependencies are not installed",
)
class DailyPairingTests(unittest.TestCase):
    def test_aggregates_hourly_satellite_values_to_daily_and_pairs_by_date(self):
        from tools.satellite_tools.validation_tools import _pair_daily

        times = [
            "2024-01-01T00:00:00", "2024-01-01T12:00:00",  # day 1: mean = 2.0
            "2024-01-02T00:00:00",                          # day 2: mean = 10.0
        ]
        values = [1.0, 3.0, 10.0]
        ground_daily = {"2024-01-01": 5.0, "2024-01-02": 20.0, "2024-01-03": 99.0}

        paired = _pair_daily(times, values, ground_daily)

        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0], {"date": "2024-01-01", "satellite": 2.0, "ground": 5.0})
        self.assertEqual(paired[1], {"date": "2024-01-02", "satellite": 10.0, "ground": 20.0})

    def test_dates_without_a_ground_reading_are_dropped(self):
        from tools.satellite_tools.validation_tools import _pair_daily

        times = ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        values = [1.0, 2.0]
        ground_daily = {"2024-01-01": 5.0}  # no reading for 2024-01-02

        paired = _pair_daily(times, values, ground_daily)

        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["date"], "2024-01-01")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite validation dependencies are not installed",
)
class CorrelationStatsTests(unittest.TestCase):
    def test_perfectly_correlated_series_gives_r_of_one(self):
        from tools.satellite_tools.validation_tools import _correlation_stats

        paired = [
            {"date": "2024-01-01", "satellite": 1.0, "ground": 2.0},
            {"date": "2024-01-02", "satellite": 2.0, "ground": 4.0},
            {"date": "2024-01-03", "satellite": 3.0, "ground": 6.0},
            {"date": "2024-01-04", "satellite": 4.0, "ground": 8.0},
        ]

        stats = _correlation_stats(paired, total_ground_days=4)

        self.assertAlmostEqual(stats["r"], 1.0)
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["coverage_fraction"], 1.0)

    def test_anti_correlated_series_gives_r_of_negative_one(self):
        from tools.satellite_tools.validation_tools import _correlation_stats

        paired = [
            {"date": "2024-01-01", "satellite": 1.0, "ground": 8.0},
            {"date": "2024-01-02", "satellite": 2.0, "ground": 6.0},
            {"date": "2024-01-03", "satellite": 3.0, "ground": 4.0},
            {"date": "2024-01-04", "satellite": 4.0, "ground": 2.0},
        ]

        stats = _correlation_stats(paired)

        self.assertAlmostEqual(stats["r"], -1.0)

    def test_coverage_fraction_relative_to_total_ground_days(self):
        from tools.satellite_tools.validation_tools import _correlation_stats

        paired = [
            {"date": "2024-01-01", "satellite": 1.0, "ground": 2.0},
            {"date": "2024-01-02", "satellite": 2.0, "ground": 3.0},
            {"date": "2024-01-03", "satellite": 3.0, "ground": 4.0},
        ]

        stats = _correlation_stats(paired, total_ground_days=5)

        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["coverage_fraction"], 0.6)

    def test_fewer_than_two_points_yields_none_correlation(self):
        from tools.satellite_tools.validation_tools import _correlation_stats

        stats = _correlation_stats([{"date": "2024-01-01", "satellite": 1.0, "ground": 2.0}])

        self.assertIsNone(stats["r"])
        self.assertEqual(stats["n"], 1)

    def test_pooled_stats_concatenate_across_monitors(self):
        from tools.satellite_tools.validation_tools import _correlation_stats

        monitor_a = [
            {"date": "2024-01-01", "satellite": 1.0, "ground": 2.0},
            {"date": "2024-01-02", "satellite": 2.0, "ground": 4.0},
        ]
        monitor_b = [
            {"date": "2024-01-01", "satellite": 3.0, "ground": 6.0},
            {"date": "2024-01-02", "satellite": 4.0, "ground": 8.0},
        ]

        pooled = _correlation_stats(monitor_a + monitor_b)

        self.assertEqual(pooled["n"], 4)
        self.assertAlmostEqual(pooled["r"], 1.0)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite validation dependencies are not installed",
)
class ExceedanceDaysHelperTests(unittest.TestCase):
    def test_hard_threshold_flags_only_days_above_it(self):
        from tools.satellite_tools.validation_tools import _exceedance_days

        records = [
            {"date_local": "2024-01-01", "first_max_value": "2.0"},
            {"date_local": "2024-01-02", "first_max_value": "5.0"},
            {"date_local": "2024-01-03", "first_max_value": "1.0"},
        ]

        flagged = _exceedance_days(records, "first_max_value", hard_threshold=3.0, percentile_threshold=None)

        self.assertEqual(flagged, {"2024-01-02"})

    def test_percentile_threshold_flags_top_fraction(self):
        from tools.satellite_tools.validation_tools import _exceedance_days

        records = [
            {"date_local": "2024-01-01", "first_max_value": "1.0"},
            {"date_local": "2024-01-02", "first_max_value": "2.0"},
            {"date_local": "2024-01-03", "first_max_value": "3.0"},
            {"date_local": "2024-01-04", "first_max_value": "4.0"},
        ]

        flagged = _exceedance_days(records, "first_max_value", hard_threshold=None, percentile_threshold=75.0)

        self.assertEqual(flagged, {"2024-01-04"})

    def test_records_with_missing_values_are_ignored(self):
        from tools.satellite_tools.validation_tools import _exceedance_days

        records = [
            {"date_local": "2024-01-01", "first_max_value": None},
            {"date_local": "2024-01-02", "first_max_value": "9.0"},
        ]

        flagged = _exceedance_days(records, "first_max_value", hard_threshold=3.0, percentile_threshold=None)

        self.assertEqual(flagged, {"2024-01-02"})


def _fake_aqs_get(monitors_body, daily_body):
    """Stub for epa_aqs_tools._aqs_get, routed by endpoint.

    Patched at this level (rather than httpx.AsyncClient) because the
    fake-MCP seam's own transport also uses the process-global httpx module —
    patching httpx.AsyncClient there would break the real MCP session too.
    """

    async def _aqs_get(endpoint, params):
        if endpoint == "monitors/byBox":
            return {"Header": [{"status": "success"}], "Data": monitors_body}
        if endpoint == "dailyData/byBox":
            return {"Header": [{"status": "success"}], "Data": daily_body}
        return {"Header": [{"status": "success"}], "Data": []}

    return _aqs_get


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in FULL_TOOL_REQUIRED_MODULES),
    "full validate_against_ground tool test dependencies are not installed",
)
class ValidateAgainstGroundToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def test_validate_against_ground_pairs_and_scores_a_single_monitor(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools import validation_tools

        def make_cube():
            times = pd.date_range("2024-01-01", periods=3, freq="D")
            return xr.Dataset(
                {"no2": (
                    ("time", "lat", "lon"),
                    [[[1.0, 100.0], [100.0, 100.0]],
                     [[2.0, 100.0], [100.0, 100.0]],
                     [[3.0, 100.0], [100.0, 100.0]]],
                    {"units": "mol/m^2"},
                )},
                coords={"time": times, "lat": [40.0, 41.0], "lon": [-74.0, -73.0]},
            )

        self.volume.add_zarr("cube_1", make_cube)

        monitors_body = [{
            "latitude": "40.0",
            "longitude": "-74.0",
            "state_code": "34",
            "county_code": "017",
            "site_number": "0006",
            "local_site_name": "Newark Firehouse",
        }]
        daily_body = [
            {
                "date_local": "2024-01-01", "arithmetic_mean": "2.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
                "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
            },
            {
                "date_local": "2024-01-02", "arithmetic_mean": "4.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
                "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
            },
            {
                "date_local": "2024-01-03", "arithmetic_mean": "6.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
                "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
            },
        ]

        region = {
            "bounds": (-75.0, 39.0, -73.0, 41.0),  # (minx, miny, maxx, maxy)
            "geometry": None,
            "name": "New Jersey",
        }

        with patch.object(validation_tools._resolver, "aresolve_location", AsyncMock(return_value=region)), \
             patch(
                 "tools.ground_sensor_tools.epa_aqs_tools._aqs_get",
                 AsyncMock(side_effect=_fake_aqs_get(monitors_body, daily_body)),
             ):
            tool = validation_tools.make_validate_against_ground(self.mcp_tools)
            raw = await tool.ainvoke({
                "handle": "cube_1",
                "location": "New Jersey",
                "param_code": "42602",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        result = json.loads(raw)

        self.assertEqual(result["monitors_matched"], 1)
        self.assertEqual(result["satellite_units"], "mol/m^2")
        self.assertEqual(result["pairing"], "daily")

        monitor_result = result["monitors"][0]
        self.assertEqual(monitor_result["station_id"], "34-017-0006")
        self.assertEqual(monitor_result["ground_units"], "ppb")
        self.assertAlmostEqual(monitor_result["stats"]["r"], 1.0)
        self.assertEqual(monitor_result["stats"]["n"], 3)
        self.assertEqual(monitor_result["coverage"]["n_excluded"], 0)

        pooled = result["pooled_stats"]
        self.assertAlmostEqual(pooled["r"], 1.0)
        self.assertEqual(pooled["n"], 3)

        self.assertEqual(len(result["_artifact_refs"]), 1)
        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "timeseries")
        series = ref["metadata"]["series"]
        self.assertEqual(len(series), 2)
        kinds = {s["source_kind"] for s in series}
        self.assertEqual(kinds, {"satellite", "ground"})
        ground_series = next(s for s in series if s["source_kind"] == "ground")
        self.assertEqual(ground_series["station_id"], "34-017-0006")
        self.assertEqual(ref["metadata"]["source_handles"], ["cube_1"])

        # Provenance: cube handle, monitor id, pairing params traceable end-to-end.
        self.assertEqual(monitor_result["source_handles"], ["cube_1"])
        self.assertIn("34-017-0006", result["monitor_ids"])

    async def test_exceedance_overlay_marks_the_right_days_on_the_satellite_series(self):
        import pandas as pd
        import xarray as xr
        from tools.satellite_tools import validation_tools

        def make_cube():
            times = pd.date_range("2024-01-01", periods=3, freq="D")
            return xr.Dataset(
                {"no2": (
                    ("time", "lat", "lon"),
                    [[[10.0, 100.0], [100.0, 100.0]],
                     [[20.0, 100.0], [100.0, 100.0]],
                     [[30.0, 100.0], [100.0, 100.0]]],
                    {"units": "mol/m^2"},
                )},
                coords={"time": times, "lat": [40.0, 41.0], "lon": [-74.0, -73.0]},
            )

        self.volume.add_zarr("cube_2", make_cube)

        monitors_body = [{
            "latitude": "40.0",
            "longitude": "-74.0",
            "state_code": "34",
            "county_code": "017",
            "site_number": "0006",
            "local_site_name": "Newark Firehouse",
        }]
        daily_body = [
            {
                "date_local": "2024-01-01", "first_max_value": "2.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "pollutant_standard": "NO2 1-hour 2010",
            },
            {
                "date_local": "2024-01-02", "first_max_value": "150.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "pollutant_standard": "NO2 1-hour 2010",
            },
            {
                "date_local": "2024-01-03", "first_max_value": "1.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "pollutant_standard": "NO2 1-hour 2010",
            },
        ]

        region = {
            "bounds": (-75.0, 39.0, -73.0, 41.0),
            "geometry": None,
            "name": "New Jersey",
        }

        with patch.object(validation_tools._resolver, "aresolve_location", AsyncMock(return_value=region)), \
             patch(
                 "tools.ground_sensor_tools.epa_aqs_tools._aqs_get",
                 AsyncMock(side_effect=_fake_aqs_get(monitors_body, daily_body)),
             ):
            tool = validation_tools.make_exceedance_overlay(self.mcp_tools)
            raw = await tool.ainvoke({"handle": "cube_2", "location": "New Jersey", "param_code": "42602"})

        result = json.loads(raw)

        self.assertEqual(result["monitors_matched"], 1)
        self.assertEqual(result["measurement_field"], "first_max_value")
        self.assertEqual(result["pollutant_standard"], "NO2 1-hour 2010")

        monitor_result = result["monitors"][0]
        self.assertEqual(monitor_result["station_id"], "34-017-0006")
        # Only 2024-01-02 (150.0) exceeds the NO2 regulatory limit (100.0 ppb).
        self.assertEqual(monitor_result["exceedance_dates"], ["2024-01-02"])
        self.assertEqual(monitor_result["source_handles"], ["cube_2"])

        self.assertEqual(len(result["_artifact_refs"]), 1)
        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "timeseries")
        self.assertEqual(ref["metadata"]["source_handles"], ["cube_2"])


if __name__ == "__main__":
    unittest.main()
