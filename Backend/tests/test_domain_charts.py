"""The chart payload builders in domain.charts, on small synthetic data."""
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.domain.charts import colormaps, heatmap, monitor_map, profile, provenance, summary, table, timeseries


def field(n_lat: int = 4, n_lon: int = 5, *, lon0: float = -75.0, seed: int = 1) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(5.0, 1.0, (n_lat, n_lon)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(39.0, 41.0, n_lat), "lon": np.linspace(lon0, lon0 + 2.0, n_lon)},
        name="NO2",
        attrs={"units": "mol/m^2"},
    )


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


class HeatmapPayloadTests(unittest.TestCase):
    def test_the_payload_has_the_fields_the_map_renders(self):
        payload = heatmap.heatmap_payload(field(), "NO2 over NJ", "NO2", "mol/m^2")
        for key in ("type", "title", "variable", "units", "lats", "lons", "values", "statistics",
                    "vmin", "vmax", "scale", "colormap", "overlay"):
            self.assertIn(key, payload)
        self.assertEqual(payload["type"], "heatmap")
        self.assertEqual((len(payload["values"]), len(payload["values"][0])), (4, 5))
        self.assertEqual(payload["scale"], {"method": "percentile", "p": [2, 98]})
        self.assertEqual(set(payload["overlay"]), {"bounds"})

    def test_the_overlay_bounds_are_pixel_edges(self):
        payload = heatmap.heatmap_payload(field(), "t", "NO2", "u")
        west, south, east, north = payload["overlay"]["bounds"]
        self.assertAlmostEqual(west, -75.25)
        self.assertAlmostEqual(east, -72.75)
        self.assertAlmostEqual(south, 39.0 - 1 / 3)
        self.assertAlmostEqual(north, 41.0 + 1 / 3)

    def test_0_to_360_longitudes_are_normalised(self):
        payload = heatmap.heatmap_payload(field(lon0=285.0), "t", "NO2", "u")
        self.assertTrue(all(-180 <= lon <= 180 for lon in payload["lons"]))

    def test_statistics_come_from_the_full_field_before_thinning(self):
        da = field(200, 200)
        da[150, 150] = 1000.0
        payload = heatmap.heatmap_payload(da, "t", "NO2", "u")
        self.assertLessEqual(len(payload["lats"]) * len(payload["lons"]), heatmap.MAX_GRID_CELLS)
        self.assertEqual(payload["statistics"]["max"], 1000.0)
        self.assertEqual(payload["statistics"]["count"], 40_000)

    def test_an_overlay_renderer_gets_the_full_resolution_field_and_the_scale(self):
        seen = {}

        def renderer(lats, lons, arr, lut, vmin, vmax):
            seen.update(shape=arr.shape, vmin=vmin, vmax=vmax, lut=len(lut))
            return {"_path": "x.png"}

        payload = heatmap.heatmap_payload(field(200, 200), "t", "NO2", "u", overlay_renderer=renderer)
        self.assertEqual(seen["shape"], (200, 200))
        self.assertEqual(payload["overlay"]["_path"], "x.png")
        self.assertAlmostEqual(seen["vmin"], payload["vmin"], places=4)

    def test_an_imposed_range_is_disclosed_as_given(self):
        payload = heatmap.heatmap_payload(
            field(), "t", "NO2", "u", value_range=(-2.0, 2.0), scale_disclosure={"method": "shared"},
        )
        self.assertEqual((payload["vmin"], payload["vmax"]), (-2.0, 2.0))
        self.assertEqual(payload["scale"], {"method": "shared"})
        explicit = heatmap.heatmap_payload(field(), "t", "NO2", "u", value_range=(0.0, 1.0))
        self.assertEqual(explicit["scale"], {"method": "explicit"})

    def test_the_area_weighted_mean_weights_by_cos_latitude(self):
        arr = np.array([[1.0, 1.0], [3.0, 3.0]])
        lats = np.array([0.0, 60.0])
        mean = heatmap.area_weighted_mean(arr, lats, np.isfinite(arr))
        self.assertAlmostEqual(mean, (1.0 * 1.0 + 3.0 * 0.5) / 1.5)

    def test_percentile_bounds_of_a_constant_or_empty_field(self):
        self.assertEqual(heatmap.percentile_bounds(np.array([np.nan])), (0.0, 1.0))
        self.assertEqual(heatmap.percentile_bounds(np.full(4, 2.0)), (1.98, 2.02))

    def test_the_comparison_tool_uses_these_diverging_bounds(self):
        from tta_backend.tools.satellite_tools import comparison_tools

        self.assertIs(comparison_tools.diverging_bounds, heatmap.diverging_bounds)
        self.assertIs(comparison_tools.DIVERGING_SCALE_DISCLOSURE, heatmap.DIVERGING_SCALE_DISCLOSURE)
        self.assertFalse(hasattr(comparison_tools, "_diverging_bounds"))

    def test_diverging_bounds_are_centred_on_zero(self):
        low, high = heatmap.diverging_bounds(np.array([-1.0, 0.5, 4.0, np.nan]))
        self.assertEqual(low, -high)
        self.assertGreater(high, 0)
        self.assertEqual(heatmap.diverging_bounds(np.zeros(3)), (-1.0, 1.0))
        diverging = heatmap.heatmap_payload(field(), "t", "NO2", "u", diverging=True)
        self.assertEqual(diverging["colormap"]["name"], "RdBu_r")

    def test_the_time_mean_field_skips_empty_timesteps(self):
        times = np.array(["2024-06-01T12", "2024-06-02T12", "2024-06-03T12"], dtype="datetime64[ns]")
        data = np.stack([np.full((2, 2), 1.0), np.full((2, 2), np.nan), np.full((2, 2), 3.0)])
        da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]})
        mean, valid = heatmap.time_mean_field(da, "time", "daily")
        self.assertEqual(valid, [0, 2])
        self.assertTrue(np.allclose(mean.values, 2.0))
        empty, none = heatmap.time_mean_field(da.isel(time=[1]), "time", "daily")
        self.assertEqual(none, [])
        self.assertTrue(np.isnan(empty.values).all())


class OldToolPathTests(unittest.TestCase):
    """plot_tools builds its payloads through these functions, unchanged."""

    def test_the_plot_tools_heatmap_is_the_domain_heatmap_with_its_colormap(self):
        from tta_backend.tools.satellite_tools import plot_tools
        from tta_backend.utils.colormaps import resolve

        da = field(30, 40)
        old = plot_tools._da_to_heatmap_payload(da, "t", "NO2", "u", diverging=True, value_range=(-1.0, 1.0))
        new = heatmap.heatmap_payload(
            da, "t", "NO2", "u", diverging=True, value_range=(-1.0, 1.0), colormap=resolve("NO2", diverging=True),
        )
        self.assertEqual(canonical(old), canonical(new))

    def test_the_plot_tools_overlay_is_rendered_through_its_store(self):
        from tta_backend.tools.satellite_tools import plot_tools

        with mock.patch.object(plot_tools, "_render_and_store_overlay", return_value={"_path": "p"}) as render:
            payload = plot_tools._da_to_heatmap_payload(field(), "t", "NO2", "u", render_overlay=True)
        render.assert_called_once()
        self.assertEqual(payload["overlay"]["_path"], "p")

    def test_reproducibility_matches_with_the_requested_scope_looked_up_by_the_tool(self):
        from tta_backend.tools.satellite_tools import plot_tools

        da = field()
        meta = {"aggregation_label": "L", "n_granules": 2, "cadence": "daily", "granule_dates": ["a", "b"]}
        args = (["obs_1"], da, "R", "agg", {"c": 1}, meta, {"bounds": [-75, 39, -73, 41]}, {"short_name": "S"})
        with mock.patch.object(plot_tools, "_requested_scope", return_value={"location": "NJ"}):
            old = plot_tools._attach_reproducibility({"type": "heatmap"}, *args)
        new = provenance.attach_reproducibility({"type": "heatmap"}, *args, requested_scope={"location": "NJ"})
        self.assertEqual(canonical(old), canonical(new))
        self.assertEqual(new["provenance"]["requested_scope"], {"location": "NJ"})
        self.assertEqual(new["metadata"]["source_handles"], ["obs_1"])

    def test_plot_tools_uses_the_domain_longitude_normaliser(self):
        from tta_backend.domain.earthdata.opening import normalize_longitudes
        from tta_backend.tools.satellite_tools import plot_tools

        self.assertIs(plot_tools._normalize_longitudes, normalize_longitudes)

    def test_the_utils_colormaps_are_the_domain_ones(self):
        from tta_backend.utils import colormaps as app_colormaps

        self.assertIs(app_colormaps.resolve, colormaps.resolve)
        self.assertIs(app_colormaps.by_name, colormaps.by_name)
        self.assertIs(app_colormaps.ColormapResolution, colormaps.ColormapResolution)


class SummaryTests(unittest.TestCase):
    def test_a_heatmap_summary_has_dims_range_and_handles(self):
        payload = heatmap.heatmap_payload(field(), "t", "NO2", "u")
        payload["metadata"] = {"source_handles": ["grid_x"]}
        found = summary.chart_model_summary(payload)
        self.assertEqual(found["grid_dims"], [4, 5])
        self.assertEqual(found["source_handles"], ["grid_x"])
        self.assertNotIn("values", found)

    def test_only_landed_planes_are_scrubbable(self):
        frames = {"frames": [1, 2], "_key": "k", "planes": {"max": {"_key": "m"}, "min": {}}}
        self.assertEqual(summary.scrubbable_statistics(frames), ["mean", "max"])


class TimeseriesTests(unittest.TestCase):
    def test_points_drop_nan_and_sort_by_time(self):
        times = np.array(["2024-06-03", "2024-06-01", "2024-06-02"], dtype="datetime64[ns]")
        found = timeseries.timeseries_points(np.array([3.0, 1.0, np.nan]), times)
        self.assertEqual(found, (["2024-06-01T00:00:00", "2024-06-03T00:00:00"], [1.0, 3.0], [1, 0]))
        self.assertEqual(timeseries.timeseries_points(np.array([np.nan]), times[:1]), ([], [], []))

    def line(self, label: str, units: str | None) -> timeseries.Line:
        return timeseries.Line(label=label, units=units, times=["t1"], values=[1.0], source={"handle": label})

    def test_lines_in_one_unit_share_an_axis(self):
        payload = timeseries.multi_series_payload(
            [self.line("a", "ppb"), self.line("b", "ppb")], title="T", variable="NO2", stat="value",
        )
        self.assertEqual(payload["axes"], [{"id": "y", "units": "ppb"}])
        self.assertEqual([s["axis"] for s in payload["series"]], ["y", "y"])
        self.assertEqual((payload["times"], payload["values"]), (["t1"], [1.0]))

    def test_a_second_unit_gets_the_second_axis_and_a_third_is_refused(self):
        payload = timeseries.multi_series_payload(
            [self.line("a", "ppb"), self.line("b", "molecules/cm^2")], title="T", variable="", stat="value",
        )
        self.assertEqual([s["axis"] for s in payload["series"]], ["y", "y2"])
        with self.assertRaises(ValueError):
            timeseries.multi_series_payload(
                [self.line("a", "ppb"), self.line("b", "ppm"), self.line("c", None)], title="T", variable="", stat="value",
            )

    def test_line_points_from_a_frame(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2024-06-02", "2024-06-01", "2024-06-03"], utc=True),
            "value": [2.0, 1.0, np.nan],
        })
        times, values = timeseries.line_points(frame)
        self.assertEqual(values, [1.0, 2.0])
        self.assertEqual(times[0], "2024-06-01T00:00:00+00:00")


PRESSURES = [0.2, 2.0, 30.0, 130.0, 500.0, 900.0]


def layered(values=None, *, times=("2024-06-01T12", "2024-06-01T13")) -> xr.DataArray:
    shape = (len(times), 2, 3, len(PRESSURES))
    data = np.broadcast_to(np.asarray(values if values is not None else [1.0, 2, 3, 4, 5, 6]), shape).copy()
    pressure = np.broadcast_to(np.asarray(PRESSURES), shape).copy()
    dims = ("time", "lat", "lon", "layer")
    return xr.DataArray(
        data, dims=dims,
        coords={
            "time": np.array(list(times), dtype="datetime64[ns]"), "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0],
            "pressure": (dims, pressure, {"units": "hPa", "standard_name": "air_pressure"}),
        },
        name="o3", attrs={"units": "DU"},
    )


class ProfileTests(unittest.TestCase):
    def test_the_vertical_dimension_is_the_one_left(self):
        self.assertEqual(profile.vertical_dim(layered(), "time"), ("layer", ["layer"]))
        flat = layered().isel(layer=0)
        self.assertEqual(profile.vertical_dim(flat, "time"), (None, []))

    def test_a_profile_has_one_value_per_layer_against_pressure(self):
        found = profile.reduce_profile(layered(), vertical_dim="layer", time_dim="time", cadence="hourly", region_cells=6)
        assert found is not None
        self.assertEqual(found.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(found.default_axis, "pressure")
        self.assertEqual(found.vertical["pressure"]["values"], PRESSURES)
        self.assertEqual(found.vertical["pressure"]["spread"], [0.0] * 6)
        self.assertEqual(found.layer_order, "top_down")
        self.assertEqual(found.valid_fraction, [1.0] * 6)
        self.assertEqual(found.valid_indices, [0, 1])

    def test_an_empty_field_has_no_profile(self):
        empty = layered(values=[np.nan] * 6)
        self.assertIsNone(profile.reduce_profile(empty, vertical_dim="layer", time_dim="time", cadence="hourly"))

    def test_the_layer_order_is_measured_off_the_axis(self):
        self.assertEqual(profile.layer_order([900.0, 1.0], "pressure"), "bottom_up")
        self.assertEqual(profile.layer_order([1.0, 60.0], "altitude"), "bottom_up")
        self.assertEqual(profile.layer_order([5.0], "pressure"), "unknown")

    def test_a_profile_survives_the_table_round_trip(self):
        found = profile.reduce_profile(layered(), vertical_dim="layer", time_dim="time", cadence="hourly")
        assert found is not None
        frame = profile.profile_table(found)
        self.assertEqual(list(frame.columns), ["layer", "value", "valid_fraction", "pressure", "pressure_spread", "pressure_unit"])
        back = profile.profile_from_table(frame)
        self.assertEqual(back.values, found.values)
        self.assertEqual(back.vertical["pressure"]["units"], "hPa")
        self.assertEqual(back.layer_order, "top_down")
        with self.assertRaises(ValueError):
            profile.profile_from_table(pd.DataFrame({"a": [1]}))

    def test_the_payload_has_the_profile_chart_fields(self):
        found = profile.reduce_profile(layered(), vertical_dim="layer", time_dim="time", cadence="hourly")
        assert found is not None
        payload = profile.profile_payload(found, title="T", variable="o3", units="DU", masking={}, agg_meta={"a": 1})
        for key in ("type", "values", "layers", "vertical", "default_axis", "layer_order", "valid_fraction", "aggregation_meta"):
            self.assertIn(key, payload)
        self.assertNotIn("aggregation_meta", profile.profile_payload(found, title="T", variable="o3", units="DU", masking={}))
        self.assertEqual(set(profile.profile_export(payload)), {"layers", "values", "valid_fraction", "vertical", "default_axis", "layer_order"})

    def test_an_oversized_profile_is_named(self):
        with mock.patch.object(profile, "MAX_PROFILE_CELLS", 10):
            self.assertIn("cell limit", profile.profile_scale_guard(layered()) or "")
        self.assertIsNone(profile.profile_scale_guard(layered()))


class MonitorColouringTests(unittest.TestCase):
    def test_ozone_categories_at_their_boundaries(self):
        cases = {
            0.054: "Good", 0.0549: "Good", 0.055: "Moderate", 0.070: "Moderate", 0.071: "Unhealthy for Sensitive Groups",
            0.200: "Very Unhealthy",
        }
        for value, category in cases.items():
            with self.subTest(value=value):
                self.assertEqual(monitor_map.classify("o3", value, "8-hour")["aqi_category"], category)

    def test_a_value_beyond_the_table_is_labelled_not_given_the_top_category(self):
        found = monitor_map.classify("o3", 0.201, "8-hour")
        self.assertEqual(found["aqi_category"], monitor_map.BEYOND_LABEL)
        self.assertIsNone(found["aqi_low"])
        self.assertEqual(monitor_map.classify("pm25", 325.5, "24-hour")["aqi_category"], monitor_map.BEYOND_LABEL)
        self.assertEqual(monitor_map.classify("pm25", 325.4, "24-hour")["aqi_category"], "Hazardous")

    def test_a_value_below_a_partial_table_is_labelled(self):
        self.assertEqual(monitor_map.classify("o3", 0.1, "1-hour")["aqi_category"], monitor_map.BELOW_LABEL)

    def test_colours_are_the_categories_not_the_value_range(self):
        frame = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"], "lat": [40.0, 41.0], "lon": [-75.0, -74.0], "value": [0.050, 0.052]})
        rows = monitor_map.monitor_rows(frame, pollutant="o3", period="8-hour", unit="ppm")
        self.assertEqual({row["color"] for row in rows}, {monitor_map.AQI_COLORS["Good"]})
        self.assertEqual(rows[0]["latitude"], 40.0)
        self.assertEqual(rows[0]["standard_level"], 0.07)
        self.assertFalse(rows[0]["above_standard_level"])

    def test_above_the_standard_level_agrees_with_the_aqi_category(self):
        frame = pd.DataFrame({
            "id": ["a", "b"], "name": ["A", "B"], "lat": [40.0, 41.0], "lon": [-75.0, -74.0], "value": [0.0705, 0.0712],
        })
        rows = monitor_map.monitor_rows(frame, pollutant="o3", period="8-hour", unit="ppm")
        self.assertEqual([r["aqi_category"] for r in rows], ["Moderate", "Unhealthy for Sensitive Groups"])
        self.assertEqual([r["above_standard_level"] for r in rows], [False, True])
        self.assertEqual(rows[0]["value"], 0.0705)

    def test_a_site_table_is_reduced_to_one_point_per_site(self):
        frame = pd.DataFrame({
            "site_id": ["s1", "s1", "s2", "s3"], "name": ["One", "One", "Two", "Three"],
            "lat": [40.0, 40.0, 41.0, 39.0], "lon": [-75.0, -75.0, -74.0, -74.5],
            "value": [0.05, 0.07, 0.04, np.nan], "unit": ["ppm"] * 4,
        })
        top = monitor_map.one_point_per_site(frame, "max")
        self.assertEqual(top["id"].tolist(), ["s1", "s2", "s3"])
        self.assertEqual(top["value"].tolist()[:2], [0.07, 0.04])
        self.assertTrue(np.isnan(top["value"].iloc[2]))
        self.assertEqual(top["rows"].tolist(), [2, 1, 1])
        self.assertAlmostEqual(monitor_map.one_point_per_site(frame, "mean")["value"].iloc[0], 0.06)
        self.assertEqual(monitor_map.missing_site_columns(frame.drop(columns=["unit", "lat"])), ["lat", "unit"])
        self.assertEqual(monitor_map.site_units(frame.assign(unit=["ppm", "ppb", None, "ppm"])), ["ppm", "ppb"])
        with self.assertRaises(ValueError):
            monitor_map.one_point_per_site(frame, "median")

    def test_a_missing_value_has_no_category(self):
        self.assertEqual(monitor_map.classify("o3", None, "8-hour")["aqi_category"], monitor_map.NO_VALUE_LABEL)
        self.assertEqual(monitor_map.classify("o3", float("nan"), "8-hour")["aqi_category"], monitor_map.NO_VALUE_LABEL)

    def test_the_pollutant_is_read_from_the_quantity(self):
        self.assertEqual(monitor_map.pollutant_for("Ozone daily maximum"), "o3")
        self.assertEqual(monitor_map.pollutant_for("PM2.5 concentration"), "pm25")
        self.assertIsNone(monitor_map.pollutant_for("Smoke"))

    def test_the_legend_covers_the_table_and_beyond(self):
        entries = monitor_map.legend("o3", "8-hour")
        self.assertEqual(entries[0]["category"], "Good")
        self.assertEqual(entries[-1]["category"], monitor_map.BEYOND_LABEL)
        self.assertEqual(monitor_map.legend("o3", "1-hour")[0]["category"], monitor_map.BELOW_LABEL)


class TableRowsTests(unittest.TestCase):
    def test_rows_are_capped_and_plain_json(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"], utc=True),
            "value": [1.0, np.nan, 3.0],
            "n": np.array([1, 2, 3], dtype="int64"),
        })
        columns, rows = table.table_rows(frame, 2)
        self.assertEqual(columns, ["time", "value", "n"])
        self.assertEqual(rows, [
            {"time": "2024-06-01T00:00:00+00:00", "value": 1.0, "n": 1},
            {"time": "2024-06-02T00:00:00+00:00", "value": None, "n": 2},
        ])
        json.dumps(rows)
        with self.assertRaises(ValueError):
            table.table_rows(frame, 0)

    def test_cells_of_other_types(self):
        self.assertEqual(table.json_cell(np.datetime64("2024-06-01")), "2024-06-01T00:00:00")
        self.assertIsNone(table.json_cell(np.datetime64("NaT", "ns")))
        self.assertIsNone(table.json_cell(pd.NA))
        self.assertEqual(table.json_cell(np.float32(1.5)), 1.5)
        self.assertEqual(table.json_cell(True), True)


if __name__ == "__main__":
    unittest.main()
