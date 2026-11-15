"""The render capabilities on the framework fakes: payload shapes, storage, events and refusals."""
import datetime
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.domain.charts import monitor_map as colouring
from tta_backend.harness.capabilities import CapabilityRegistry, invoke
from tta_backend.harness.capabilities.render import maps, monitor_map, profiles, tables, timeseries
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.handles import Descriptor, HandleRecord
from tta_backend.harness.handles.model import HandleKind, Period
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import MemoryArtifacts, RecordingEvents, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)
NJ = {"name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4], "area": "polygons_" + "0" * 32}
LATS = np.arange(39.0, 41.01, 0.5)
LONS = np.arange(-75.5, -73.99, 0.5)


def grid_dataset(*, times: int = 0, value: float = 2.0, extra_dim: bool = False, name: str = "vertical_column_troposphere") -> xr.Dataset:
    rng = np.random.default_rng(7)
    dims: tuple[str, ...] = ("lat", "lon")
    coords: dict[str, Any] = {"lat": LATS, "lon": LONS}
    shape: tuple[int, ...] = (LATS.size, LONS.size)
    if times:
        dims = ("time", *dims)
        coords["time"] = pd.date_range("2024-06-01T15:00", periods=times, freq="D").to_numpy()
        shape = (times, *shape)
    if extra_dim:
        dims = (*dims, "level")
        coords["level"] = [1, 2]
        shape = (*shape, 2)
    data = value + rng.normal(0.0, 0.1, shape)
    return xr.Dataset({name: (dims, data, {"units": "molecules/cm^2"})}, coords=coords)


def points(values: list[float] | None = None, n: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame({
        "id": [f"34-001-000{i}" for i in range(n)],
        "name": [f"Site {i}" for i in range(n)],
        "lat": np.linspace(39.5, 40.5, n),
        "lon": np.linspace(-75.0, -74.2, n),
    })
    if values is not None:
        frame["value"] = values
    return frame


def series(lines: dict[str, list[float]], start: str = "2024-06-01") -> pd.DataFrame:
    rows = []
    for name, values in lines.items():
        for i, value in enumerate(values):
            rows.append({"time": pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=i), "value": value, "series": name})
    return pd.DataFrame(rows)


class RenderCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        self.events = RecordingEvents()
        self.artifacts = MemoryArtifacts()
        context = make_run_context(stores=self.stores, events=self.events, artifacts=self.artifacts)
        self.env: RunEnv = await make_run_env(self._tmp.name, context=context)

    async def seed(self, kind: HandleKind, data: Any, **descriptor: Any) -> HandleRecord:
        descriptor.setdefault("summary", f"A {kind}")
        return await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind=kind, data=data,
            descriptor=Descriptor(**descriptor), operation="test.seed", created_at=NOW,
        )

    async def seed_grid(self, ds: xr.Dataset | None = None, **descriptor: Any) -> HandleRecord:
        descriptor.setdefault("product", "TEMPO_NO2")
        descriptor.setdefault("unit", "molecules/cm^2")
        descriptor.setdefault("region", NJ)
        descriptor.setdefault("masking", {"variable": "vertical_column_troposphere", "qa_status": "verified", "region_cells": 20})
        descriptor.setdefault("period", Period(start="2024-06-01", end="2024-06-03"))
        return await self.seed("grid", ds if ds is not None else grid_dataset(), **descriptor)

    async def seed_points(self, frame: pd.DataFrame, **descriptor: Any) -> HandleRecord:
        descriptor.setdefault("quantity", "Ozone daily maximum 8-hour average")
        descriptor.setdefault("unit", "ppm")
        descriptor.setdefault("product", "EPA AQS")
        descriptor.setdefault("region", NJ)
        return await self.seed("points", frame, **descriptor)

    async def seed_series(self, frame: pd.DataFrame, **descriptor: Any) -> HandleRecord:
        descriptor.setdefault("quantity", "NO2 concentration")
        descriptor.setdefault("unit", "ppb")
        descriptor.setdefault("product", "EPA AQS")
        descriptor.setdefault("region", NJ)
        return await self.seed("series", frame, **descriptor)

    async def failure(self, capability: Any, params: dict[str, Any]) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, capability, params)
        return caught.exception

    async def entries(self) -> list[Any]:
        return await self.stores.traces.entries("user-a", self.env.run_id)

    def assert_stored_and_emitted(self, result: Any) -> dict[str, Any]:
        [chart_id] = result.charts
        stored = self.artifacts.stored[chart_id]
        [emitted] = self.events.charts
        self.assertEqual(emitted["chart_id"], chart_id)
        self.assertEqual({k: v for k, v in emitted.items() if k != "chart_id"}, stored)
        return stored


class GroupTests(unittest.TestCase):
    def test_every_render_capability_registers_under_the_one_group(self):
        registry = CapabilityRegistry()
        registry.add_group(GROUP)
        for module in (maps, monitor_map, timeseries, tables, profiles):
            self.assertIs(module.GROUP, GROUP)
            for capability in module.CAPABILITIES:
                registry.register(capability)
                self.assertIsNone(capability.output_kind)
        self.assertEqual(
            [c.name for c in registry.in_group("render")],
            ["render.map", "render.monitor_map", "render.profile", "render.table", "render.timeseries"],
        )
        self.assertTrue(GROUP.notes)


class MapTests(RenderCase):
    async def test_a_grid_is_stored_and_emitted_as_a_heatmap(self):
        grid = await self.seed_grid()
        result = await invoke(self.env, maps.render_map, {"grid": grid.id})

        self.assertEqual(result.status, "done")
        self.assertIsNone(result.handle)
        payload = self.assert_stored_and_emitted(result)
        for key in ("type", "title", "variable", "units", "lats", "lons", "values", "statistics", "vmin", "vmax",
                    "scale", "colormap", "overlay", "bounds", "aggregation_meta", "is_aggregated", "provenance",
                    "query", "export", "metadata"):
            self.assertIn(key, payload)
        self.assertEqual(payload["type"], "heatmap")
        self.assertEqual(payload["title"], "vertical_column_troposphere over New Jersey")
        self.assertEqual(payload["units"], "molecules/cm^2")
        self.assertFalse(payload["is_aggregated"])
        self.assertEqual(payload["bounds"], NJ["bounds"])
        self.assertEqual(payload["metadata"]["source_handles"], [grid.id])
        self.assertEqual(payload["provenance"]["source_handles"], [grid.id])
        self.assertEqual(payload["export"]["source_handles"], [grid.id])
        self.assertEqual(payload["provenance"]["masking"]["qa_status"], "verified")
        self.assertEqual(payload["provenance"]["dataset"], "TEMPO_NO2_L3")
        self.assertNotIn("url", payload["overlay"])
        self.assertNotIn("frames", payload)
        self.assertEqual(result.summary["render_type"], "heatmap")
        self.assertEqual(result.summary["grid_dims"], [LATS.size, LONS.size])

    async def test_the_step_is_traced_with_its_chart(self):
        grid = await self.seed_grid()
        result = await invoke(self.env, maps.render_map, {"grid": grid.id})
        [entry] = await self.entries()
        self.assertEqual((entry.capability, entry.status), ("render.map", "done"))
        self.assertEqual(entry.params["grid"], grid.id)
        self.assertEqual(entry.result["charts"], list(result.charts))
        self.assertEqual(entry.result["method"], result.method)
        self.assertIn("was drawn as a map", result.method or "")

    async def test_a_time_axis_is_shown_as_the_time_mean_and_says_so(self):
        grid = await self.seed_grid(grid_dataset(times=3))
        result = await invoke(self.env, maps.render_map, {"grid": grid.id})
        payload = self.assert_stored_and_emitted(result)
        self.assertTrue(payload["is_aggregated"])
        self.assertEqual(payload["aggregation_meta"]["n_granules"], 3)
        self.assertIn("Mean", payload["aggregation_meta"]["aggregation_label"])
        self.assertEqual(payload["provenance"]["n_granules"], 3)
        self.assertTrue(result.values["time_mean"])
        self.assertTrue(any("time mean" in d for d in result.disclosures))
        self.assertIn(payload["aggregation_meta"]["title_suffix"], payload["title"])

    async def test_points_and_polygons_are_drawn_over_the_map(self):
        grid = await self.seed_grid()
        monitors = await self.seed_points(points([0.05, None, 0.07]))
        smoke = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"density": "light"},
             "geometry": {"type": "Polygon", "coordinates": [[[-75, 39], [-74, 39], [-74, 40], [-75, 39]]]}},
        ]}
        plume = await self.seed("polygons", smoke)
        result = await invoke(self.env, maps.render_map, {"grid": grid.id, "points": monitors.id, "polygons": [plume.id]})
        payload = self.assert_stored_and_emitted(result)
        overlay_points = payload["overlays"]["points"]["points"]
        self.assertEqual([p["id"] for p in overlay_points], ["34-001-0000", "34-001-0001", "34-001-0002"])
        self.assertEqual(overlay_points[0]["value"], 0.05)
        self.assertIsNone(overlay_points[1]["value"])
        self.assertEqual(payload["overlays"]["polygons"][0]["handle"], plume.id)
        self.assertEqual(payload["overlays"]["polygons"][0]["geojson"]["features"][0]["properties"]["density"], "light")
        self.assertEqual(payload["metadata"]["source_handles"], [grid.id, monitors.id, plume.id])
        self.assertEqual((result.values["overlay_points"], result.values["overlay_polygons"]), (3, 1))
        # The disclosures say what the chart data carries, not that it is drawn.
        self.assertIn("The chart data carries all 3 points as an overlay (overlays.points).", result.disclosures)
        self.assertIn("The chart data carries 1 polygon feature(s) as an overlay (overlays.polygons).", result.disclosures)
        self.assertFalse(any("drawn" in d for d in result.disclosures))

    async def test_a_diverging_map_centres_on_zero(self):
        ds = grid_dataset(value=0.0)
        grid = await self.seed_grid(ds)
        result = await invoke(self.env, maps.render_map, {"grid": grid.id, "style": "diverging"})
        payload = self.assert_stored_and_emitted(result)
        self.assertEqual(payload["vmin"], -payload["vmax"])
        self.assertEqual(payload["colormap"]["name"], "RdBu_r")
        self.assertEqual(payload["scale"]["method"], "percentile_magnitude")
        sequential = await invoke(self.env, maps.render_map, {"grid": grid.id})
        self.assertEqual(self.artifacts.stored[sequential.charts[0]]["scale"]["method"], "percentile")

    async def test_a_grid_with_a_level_dimension_is_refused(self):
        grid = await self.seed_grid(grid_dataset(extra_dim=True))
        failure = await self.failure(maps.render_map, {"grid": grid.id})
        self.assertEqual(failure.category, FailureCategory.UNSUPPORTED_SHAPE)
        self.assertEqual(failure.result, {"refusal": "wrong_dims"})
        self.assertEqual(self.events.charts, [])

    async def test_a_grid_with_no_valid_values_is_insufficient_evidence(self):
        ds = grid_dataset()
        ds["vertical_column_troposphere"][:] = np.nan
        grid = await self.seed_grid(ds)
        result = await invoke(self.env, maps.render_map, {"grid": grid.id})
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(self.artifacts.stored, {})

    async def test_wrong_handle_kinds_are_refused(self):
        line = await self.seed_series(series({"a": [1.0]}))
        failure = await self.failure(maps.render_map, {"grid": line.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        grid = await self.seed_grid()
        failure = await self.failure(maps.render_map, {"grid": grid.id, "points": grid.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class MonitorMapTests(RenderCase):
    async def test_each_point_is_coloured_by_its_aqi_category(self):
        # 0.054 and 0.055 straddle Good/Moderate; 0.201 is beyond the 8-hour table.
        monitors = await self.seed_points(points([0.054, 0.055, 0.201]))
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": monitors.id})

        [table_id] = result.charts
        stored = self.artifacts.stored[table_id]
        rows = stored["rows"]
        self.assertEqual([r["aqi_category"] for r in rows], ["Good", "Moderate", colouring.BEYOND_LABEL])
        self.assertEqual(rows[0]["color"], colouring.AQI_COLORS["Good"])
        self.assertEqual(rows[1]["color"], colouring.AQI_COLORS["Moderate"])
        self.assertEqual(rows[2]["color"], colouring.BEYOND_COLOR)
        self.assertIsNone(rows[2]["aqi_low"])
        self.assertEqual([r["above_standard_level"] for r in rows], [False, False, True])
        self.assertEqual((rows[0]["latitude"], rows[0]["longitude"]), (39.5, -75.0))
        self.assertTrue(stored["metadata"]["has_coordinates"])
        self.assertEqual(stored["metadata"]["colouring"]["method"], "aqi_category")
        self.assertEqual(stored["metadata"]["source_handles"], [monitors.id])
        self.assertEqual(self.events.charts, [])
        self.assertEqual(result.summary["by_category"], {"Good": 1, "Moderate": 1, colouring.BEYOND_LABEL: 1})
        self.assertTrue(any("not a colour from its place among the values" in d for d in result.disclosures))
        self.assertTrue(any("above the AQI table" in d for d in result.disclosures))
        [entry] = await self.entries()
        self.assertEqual((entry.capability, entry.status), ("render.monitor_map", "done"))

    async def test_close_values_far_apart_in_rank_share_a_colour(self):
        monitors = await self.seed_points(points([0.030, 0.031, 0.032]))
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": monitors.id})
        rows = self.artifacts.stored[result.charts[0]]["rows"]
        self.assertEqual({r["color"] for r in rows}, {colouring.AQI_COLORS["Good"]})

    async def test_values_in_ppb_are_converted_for_ozone(self):
        monitors = await self.seed_points(points([54.0, 71.0, 120.0]), unit="ppb")
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": monitors.id})
        rows = self.artifacts.stored[result.charts[0]]["rows"]
        self.assertEqual([r["aqi_category"] for r in rows], ["Good", "Unhealthy for Sensitive Groups", "Very Unhealthy"])
        self.assertEqual(rows[0]["units"], "ppm")
        self.assertTrue(any("converted from ppb to ppm" in d for d in result.disclosures))

    async def test_an_explicit_pollutant_and_period(self):
        monitors = await self.seed_points(points([9.0, 9.1, 400.0]), quantity="Monitor readings", unit="ug/m3")
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": monitors.id, "pollutant": "pm25"})
        rows = self.artifacts.stored[result.charts[0]]["rows"]
        self.assertEqual([r["aqi_category"] for r in rows], ["Good", "Moderate", colouring.BEYOND_LABEL])
        self.assertEqual(result.summary["averaging_period"], "24-hour")
        below = await self.seed_points(points([0.1, 0.13, 0.5]))
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": below.id, "averaging_period": "1-hour"})
        rows = self.artifacts.stored[result.charts[0]]["rows"]
        self.assertEqual([r["aqi_category"] for r in rows], [colouring.BELOW_LABEL, "Unhealthy for Sensitive Groups", "Hazardous"])

    async def test_refusals_name_their_reason(self):
        no_values = await self.seed_points(points())
        failure = await self.failure(monitor_map.render_monitor_map, {"data": no_values.id})
        self.assertEqual(failure.result, {"refusal": "no_values"})

        unknown = await self.seed_points(points([1.0, 2.0, 3.0]), quantity="Something")
        failure = await self.failure(monitor_map.render_monitor_map, {"data": unknown.id})
        self.assertEqual(failure.result, {"refusal": "unknown_pollutant"})

        wrong_unit = await self.seed_points(points([1.0, 2.0, 3.0]), unit="ug/m3")
        failure = await self.failure(monitor_map.render_monitor_map, {"data": wrong_unit.id})
        self.assertEqual(failure.result, {"refusal": "unit_mismatch"})
        self.assertIn("ug/m3", failure.reason)

        failure = await self.failure(monitor_map.render_monitor_map, {"data": wrong_unit.id, "pollutant": "o3", "averaging_period": "annual"})
        self.assertEqual(failure.result, {"refusal": "no_breakpoints"})

        line = await self.seed_series(series({"a": [1.0]}))
        failure = await self.failure(monitor_map.render_monitor_map, {"data": line.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)

    def site_table(self, values: list[float], units: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame({
            "site_id": ["34-001-0001", "34-001-0001", "34-003-0002"][: len(values)],
            "name": ["Site 1", "Site 1", "Site 2"][: len(values)],
            "lat": [39.5, 39.5, 40.5][: len(values)], "lon": [-75.0, -75.0, -74.2][: len(values)],
            "date": ["2024-06-01", "2024-06-02", "2024-06-01"][: len(values)],
            "value": values, "unit": units or ["ppm"] * len(values),
        })

    async def test_a_monitor_table_is_reduced_to_one_value_per_site(self):
        table = await self.seed(
            "table", self.site_table([0.050, 0.0712, 0.060]),
            quantity="Ozone daily maximum 8-hour average", unit="ppm", product="EPA AQS", region=NJ,
        )
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": table.id})
        stored = self.artifacts.stored[result.charts[0]]
        rows = stored["rows"]
        self.assertEqual([(r["site_id"], r["value"]) for r in rows], [("34-001-0001", 0.0712), ("34-003-0002", 0.060)])
        self.assertEqual([r["aqi_category"] for r in rows], ["Unhealthy for Sensitive Groups", "Moderate"])
        self.assertEqual(stored["metadata"]["colouring"]["site_value"], {"source": "table", "stat": "max", "max_rows_per_site": 2})
        self.assertIn("(each site's max)", stored["title"])
        self.assertIn("Each monitor's value is the max of its rows in the table (up to 2 per site).", result.disclosures)
        self.assertEqual(result.summary["site_value"], "max")

        mean = await invoke(self.env, monitor_map.render_monitor_map, {"data": table.id, "stat": "mean"})
        rows = self.artifacts.stored[mean.charts[0]]["rows"]
        self.assertAlmostEqual(rows[0]["value"], (0.050 + 0.0712) / 2)
        self.assertTrue(any("is the mean of its rows" in d for d in mean.disclosures))

    async def test_a_table_without_the_site_columns_or_in_two_units_is_refused(self):
        partial = await self.seed("table", self.site_table([0.05]).drop(columns=["unit", "lat"]), quantity="Ozone")
        failure = await self.failure(monitor_map.render_monitor_map, {"data": partial.id})
        self.assertEqual(failure.result, {"refusal": "missing_columns"})
        self.assertIn("lacks the columns lat, unit", failure.reason)
        mixed = await self.seed("table", self.site_table([0.05, 50.0], ["ppm", "ppb"]), quantity="Ozone")
        failure = await self.failure(monitor_map.render_monitor_map, {"data": mixed.id})
        self.assertEqual(failure.result, {"refusal": "mixed_units"})

    async def test_above_the_standard_is_truncated_like_the_category_and_disclosed(self):
        monitors = await self.seed_points(points([0.0705, 0.0712, 0.05]))
        result = await invoke(self.env, monitor_map.render_monitor_map, {"data": monitors.id})
        rows = self.artifacts.stored[result.charts[0]]["rows"]
        self.assertEqual([r["aqi_category"] for r in rows], ["Moderate", "Unhealthy for Sensitive Groups", "Good"])
        self.assertEqual([r["above_standard_level"] for r in rows], [False, True, False])
        self.assertTrue(any("truncated to 3 decimal(s) (ppm) before comparing with the NAAQS level" in d for d in result.disclosures))
        self.assertFalse(any("coloured" in d for d in result.disclosures))


class TimeseriesTests(RenderCase):
    async def test_several_series_share_one_chart(self):
        monitors = await self.seed_series(series({"34-001-0001": [10.0, 12.0], "34-001-0002": [8.0, np.nan, 9.0]}))
        satellite = await self.seed_series(
            series({"New Jersey": [11.0, 11.5]}), product="TEMPO_NO2", quantity="NO2 concentration",
        )
        result = await invoke(self.env, timeseries.render_timeseries, {"series": [monitors.id, satellite.id]})
        payload = self.assert_stored_and_emitted(result)

        self.assertEqual(payload["type"], "timeseries")
        labels = [s["label"] for s in payload["series"]]
        self.assertEqual(labels, ["34-001-0001 · EPA AQS", "34-001-0002 · EPA AQS", "New Jersey · TEMPO_NO2"])
        self.assertEqual(payload["series"][1]["values"], [8.0, 9.0])
        self.assertEqual(payload["series"][2]["handle"], satellite.id)
        self.assertEqual(payload["axes"], [{"id": "y", "units": "ppb"}])
        self.assertEqual((payload["times"], payload["values"]), (payload["series"][0]["times"], [10.0, 12.0]))
        self.assertEqual(payload["metadata"]["source_handles"], [monitors.id, satellite.id])
        self.assertEqual(result.values, {"lines": 3, "axes": 1})
        self.assertIn(
            "The chart data holds 3 lines (series); its times and values fields hold the first, 34-001-0001 · EPA AQS.",
            result.disclosures,
        )
        self.assertEqual(result.method, "The time-series chart data holds 3 line(s).")

    async def test_a_second_unit_gets_a_second_axis(self):
        monitor = await self.seed_series(series({"m": [10.0]}))
        column = await self.seed_series(series({"NJ": [2e15]}), unit="molecules/cm^2", product="TEMPO_NO2")
        result = await invoke(self.env, timeseries.render_timeseries, {"series": [monitor.id, column.id]})
        payload = self.assert_stored_and_emitted(result)
        self.assertEqual(payload["axes"], [{"id": "y", "units": "ppb"}, {"id": "y2", "units": "molecules/cm^2"}])
        self.assertEqual([s["axis"] for s in payload["series"]], ["y", "y2"])
        self.assertTrue(any("separate y axes" in d for d in result.disclosures))

    async def test_a_third_unit_is_refused(self):
        handles = [
            (await self.seed_series(series({"a": [1.0]}), unit=unit)).id for unit in ("ppb", "ppm", "ug/m3")
        ]
        failure = await self.failure(timeseries.render_timeseries, {"series": handles})
        self.assertEqual(failure.result, {"refusal": "too_many_units"})
        self.assertEqual(self.events.charts, [])

    async def test_too_many_lines_are_refused(self):
        many = await self.seed_series(series({f"s{i}": [1.0] for i in range(timeseries.MAX_LINES + 1)}))
        failure = await self.failure(timeseries.render_timeseries, {"series": [many.id]})
        self.assertEqual(failure.result, {"refusal": "too_many_lines"})

    async def test_empty_lines_are_disclosed_and_all_empty_is_insufficient(self):
        mixed = await self.seed_series(series({"a": [1.0], "b": [np.nan]}))
        result = await invoke(self.env, timeseries.render_timeseries, {"series": [mixed.id]})
        self.assertEqual(result.values["lines"], 1)
        self.assertTrue(any("no valid values" in d for d in result.disclosures))
        empty = await self.seed_series(series({"b": [np.nan]}))
        result = await invoke(self.env, timeseries.render_timeseries, {"series": [empty.id]})
        self.assertEqual(result.status, "insufficient_evidence")

    async def test_a_grid_is_not_a_series(self):
        grid = await self.seed_grid()
        failure = await self.failure(timeseries.render_timeseries, {"series": [grid.id]})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class TableTests(RenderCase):
    async def test_a_table_is_stored_with_a_row_cap_and_a_disclosure(self):
        frame = pd.DataFrame({"date": pd.date_range("2024-06-01", periods=30).date, "value": np.arange(30.0)})
        record = await self.seed("table", frame, summary="Exceedance days", product="EPA AQS")
        result = await invoke(self.env, tables.render_table, {"table": record.id, "max_rows": 10})

        [table_id] = result.charts
        stored = self.artifacts.stored[table_id]
        self.assertEqual(stored["title"], "Exceedance days")
        self.assertEqual(stored["columns"], ["date", "value"])
        self.assertEqual(len(stored["rows"]), 10)
        self.assertEqual(stored["rows"][0], {"date": "2024-06-01", "value": 0.0})
        self.assertEqual(stored["metadata"]["row_count"], 30)
        self.assertTrue(stored["metadata"]["truncated"])
        self.assertFalse(stored["metadata"]["has_coordinates"])
        self.assertEqual(result.disclosures, ("The table shows the first 10 of 30 rows.",))
        self.assertEqual(result.method, "The table was shown, 10 of 30 rows.")
        self.assertEqual(self.events.charts, [])
        [entry] = await self.entries()
        self.assertEqual(entry.result["charts"], [table_id])

    async def test_a_short_table_is_shown_whole(self):
        record = await self.seed("table", pd.DataFrame({"latitude": [40.0], "longitude": [-74.0]}))
        result = await invoke(self.env, tables.render_table, {"table": record.id})
        stored = self.artifacts.stored[result.charts[0]]
        self.assertFalse(stored["metadata"]["truncated"])
        self.assertTrue(stored["metadata"]["has_coordinates"])
        self.assertEqual(result.disclosures, ())

    async def test_the_cap_is_bounded_and_kinds_are_checked(self):
        record = await self.seed("table", pd.DataFrame({"a": [1]}))
        failure = await self.failure(tables.render_table, {"table": record.id, "max_rows": tables.MAX_ROWS + 1})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        grid = await self.seed_grid()
        failure = await self.failure(tables.render_table, {"table": grid.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


if __name__ == "__main__":
    unittest.main()
