"""The operations capabilities on the framework fakes: results, refusals, descriptors, lineage, trace, chaining."""
import datetime
import json
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tests.test_harness_grids import NJ, granule
from tta_backend.datasets.qa_flags import QA_VERIFIED
from tta_backend.datasets.registry import load_registry
from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean
from tta_backend.harness.capabilities import CapabilityRegistry, invoke
from tta_backend.harness.capabilities.operations import compare, events, group, pair, reduce
from tta_backend.harness.capabilities.shapes import series_frame
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, HandleRecord, ServicePayload
from tta_backend.harness.handles.model import Period
from tta_backend.harness.handles.payloads import Scalar
from tta_backend.harness.testing import FakePlaces, make_run_context, make_run_env
from tta_backend.harness.trace import MAX_JSON_BYTES, FailureCategory, StepFailure

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)
LATS = [40.0, 40.5]
LONS = [-75.0, -74.5, -74.0]
REGION = {"name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4]}
MASKING = {"variable": "no2", "qa_status": QA_VERIFIED, "qa_pass_rate": 0.9, "region_cells": 6}


def cube(values: Any, days: list[str], *, name: str = "no2", units: str | None = "ppb", lats: Any = LATS, lons: Any = LONS) -> xr.Dataset:
    attrs = {"units": units} if units else {}
    return xr.Dataset(
        {name: (("time", "lat", "lon"), np.asarray(values, dtype=float), attrs)},
        coords={"time": pd.to_datetime(days).values, "lat": lats, "lon": lons},
    )


def flat(values: Any, *, name: str = "no2", units: str = "ppb", lats: Any = LATS, lons: Any = LONS) -> xr.Dataset:
    return xr.Dataset({name: (("lat", "lon"), np.asarray(values, dtype=float), {"units": units})}, coords={"lat": lats, "lon": lons})


class OperationsCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.files: dict[str, xr.Dataset] = {}

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.files[record.id]

        self.env = await make_run_env(
            self._tmp.name, context=make_run_context(), open_retrieval=open_retrieval,
            places=FakePlaces({"new jersey": NJ}),
        )

    async def grid(
        self, data: xr.Dataset, *, quantity: str | None = "no2_tropospheric_column", unit: str | None = "ppb",
        product: str | None = "TEMPO_NO2", period: tuple[str, str] = ("2024-06-01", "2024-06-03"),
        masking: dict[str, Any] | None = None, region: Any = None,
    ) -> str:
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="grid", data=data,
            descriptor=Descriptor(
                summary="NO2 over New Jersey", quantity=quantity, unit=unit, product=product,
                region=REGION if region is None else region, period=Period(start=period[0], end=period[1]),
                masking=MASKING if masking is None else masking,
            ),
            operation="sources.satellite.get_data", created_at=NOW,
        )
        return record.id

    async def series(self, frame: pd.DataFrame, *, unit: str = "ppm") -> str:
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="series", data=series_frame(frame),
            descriptor=Descriptor(
                summary="Ozone readings", quantity="ozone concentration", unit=unit, product="EPA AQS",
                region=REGION, period=Period(start="2024-06-01", end="2024-06-03"), masking={"certified": True},
            ),
            operation="sources.monitors.readings", created_at=NOW,
        )
        return record.id

    async def record(self, handle: str) -> HandleRecord:
        return await self.env.resolver.get("user-a", handle)

    async def opened(self, handle: str) -> Any:
        return await self.env.resolver.open("user-a", handle)

    async def refusal(self, capability: Any, params: dict[str, Any]) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, capability, params)
        return caught.exception

    async def entries(self) -> list[Any]:
        return list(await self.env.context.storage.traces.entries("user-a", self.env.run_id))

    def assert_bounded(self, result: Any) -> None:
        self.assertLessEqual(len(json.dumps(result.recorded())), MAX_JSON_BYTES)


THREE_DAYS = ["2024-06-01", "2024-06-02", "2024-06-03"]


def ramp(offset: float = 0.0) -> np.ndarray:
    base = np.arange(6, dtype=float).reshape(2, 3)
    return np.stack([base + offset, base + offset + 3.0, base + offset + 6.0])


class AggregateTimeTests(OperationsCase):
    async def test_the_mean_over_time_is_minted_with_the_input_descriptor_and_lineage(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        result = await invoke(self.env, reduce.aggregate_time, {"data": source}, step="mean")

        self.assertEqual((result.status, result.kind, result.unit, result.product), ("done", "grid", "ppb", "TEMPO_NO2"))
        stored = await self.opened(result.handle)
        np.testing.assert_allclose(stored["no2"].values, ramp().mean(axis=0))
        self.assertEqual(stored["no2"].attrs["units"], "ppb")
        self.assertEqual((result.summary["time_steps"], result.summary["max"]), (3, 8.0))
        self.assertEqual(result.summary["max_location"], {"lat": 40.5, "lon": -74.0})
        self.assertEqual(result.method, "Each grid cell was reduced to its mean over 3 valid time steps (each hourly period weighted equally).")

        record = await self.record(result.handle)
        self.assertEqual(record.parents, (source,))
        self.assertEqual(record.operation, "operations.aggregate_time")
        self.assertEqual(record.parameters, {"data": source, "stat": "mean"})
        d = record.descriptor
        self.assertEqual((d.quantity, d.unit, d.product, d.region), ("no2_tropospheric_column", "ppb", "TEMPO_NO2", REGION))
        self.assertEqual(d.period, Period(start="2024-06-01", end="2024-06-03"))
        self.assertEqual(d.masking, MASKING)

        [entry] = await self.entries()
        self.assertEqual((entry.capability, entry.status, entry.step), ("operations.aggregate_time", "done", "mean"))
        self.assertEqual(entry.result["handle"], result.handle)
        self.assert_bounded(result)

    async def test_max_takes_each_cell_s_largest_value(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        result = await invoke(self.env, reduce.aggregate_time, {"data": source, "stat": "max"})
        np.testing.assert_allclose((await self.opened(result.handle))["no2"].values, ramp()[2])
        self.assertIn("per cell over every valid time step", result.method)

    async def test_a_grid_without_a_time_axis_is_refused(self):
        source = await self.grid(flat(np.ones((2, 3))))
        failure = await self.refusal(reduce.aggregate_time, {"data": source})
        self.assertEqual((failure.category, failure.result), (FailureCategory.BAD_INPUT, {"refusal": "no_time_axis"}))
        self.assertEqual(failure.reason, "operations.aggregate_time: data has no time axis to work over.")
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.BAD_INPUT))

    async def test_a_grid_with_a_level_dimension_is_refused_as_an_unsupported_shape(self):
        data = xr.Dataset(
            {"o3": (("time", "layer", "lat", "lon"), np.ones((1, 2, 2, 3)), {"units": "ppb"})},
            coords={"time": pd.to_datetime(["2024-06-01"]).values, "layer": [0, 1], "lat": LATS, "lon": LONS},
        )
        failure = await self.refusal(reduce.aggregate_time, {"data": await self.grid(data)})
        self.assertEqual((failure.category, failure.result["refusal"]), (FailureCategory.UNSUPPORTED_SHAPE, "unsupported_dims"))
        self.assertIn("layer", failure.reason)

    async def test_no_valid_value_is_insufficient_evidence(self):
        source = await self.grid(cube(np.full((3, 2, 3), np.nan), THREE_DAYS))
        result = await invoke(self.env, reduce.aggregate_time, {"data": source})
        self.assertEqual((result.status, result.handle, result.values["situation"]), ("insufficient_evidence", None, "too_little_valid"))
        self.assertIn("no valid values after masking", result.reason)
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.NO_DATA))

    async def test_an_unknown_statistic_is_invalid_input(self):
        failure = await self.refusal(reduce.aggregate_time, {"data": await self.grid(cube(ramp(), THREE_DAYS)), "stat": "mode"})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertIn("stat", failure.reason)

    async def test_a_series_is_not_accepted(self):
        handle = await self.series(pd.DataFrame({"time": ["2024-06-01T00Z"], "value": [1.0], "series": ["a"]}))
        failure = await self.refusal(reduce.aggregate_time, {"data": handle})
        self.assertIn("must be a handle of kind grid", failure.reason)


class RegionalSeriesTests(OperationsCase):
    async def test_the_area_weighted_mean_at_each_step(self):
        values = ramp()
        values[1] = np.nan
        source = await self.grid(cube(values, THREE_DAYS))
        result = await invoke(self.env, reduce.regional_series_capability, {"data": source})

        self.assertEqual((result.status, result.kind), ("done", "series"))
        frame = await self.opened(result.handle)
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["series"]), {"New Jersey"})
        expected = area_weighted_mean(cube(ramp(), THREE_DAYS)["no2"].isel(time=0))
        self.assertAlmostEqual(frame["value"].iloc[0], expected)
        self.assertEqual((result.summary["points"], result.summary["time_steps_total"]), (2, 3))
        self.assertEqual(result.method, "The mean over New Jersey was taken at each of 2 time steps (cos(latitude) area-weighted).")
        record = await self.record(result.handle)
        self.assertEqual((record.parents, record.descriptor.masking, record.descriptor.unit), ((source,), MASKING, "ppb"))

    async def test_a_grid_without_a_time_axis_is_refused(self):
        failure = await self.refusal(reduce.regional_series_capability, {"data": await self.grid(flat(np.ones((2, 3))))})
        self.assertEqual(failure.result, {"refusal": "no_time_axis"})


class StatisticTests(OperationsCase):
    async def test_a_grid_mean_is_a_scalar_with_its_spread_and_count(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        result = await invoke(self.env, reduce.statistic, {"data": source})

        self.assertEqual((result.status, result.kind, result.unit), ("done", "scalar", "ppb"))
        mean_field = cube(ramp(), THREE_DAYS)["no2"].mean("time")
        self.assertAlmostEqual(result.summary["value"], area_weighted_mean(mean_field))
        self.assertEqual(result.summary["count"], 6)
        self.assertAlmostEqual(result.summary["spread"], float(np.std(mean_field.values, ddof=1)))
        self.assertEqual(result.summary["spread_basis"], "sample standard deviation across the valid cells of the temporal-mean field")
        self.assertNotIn("std", result.summary)
        scalar = await self.opened(result.handle)
        self.assertIsInstance(scalar, Scalar)
        self.assertEqual((scalar.value, scalar.unit), (result.summary["value"], "ppb"))
        self.assertTrue(result.method.startswith("The mean was computed as the cos(latitude) area-weighted"))
        self.assertIn("is not an uncertainty of the mean", result.method)
        record = await self.record(result.handle)
        self.assertEqual(record.descriptor.masking, MASKING)

    async def test_the_spread_is_not_reported_as_the_scalar_s_uncertainty(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        for params in ({"stat": "max"}, {"stat": "percentile", "percentile": 90}, {"stat": "mean"}):
            with self.subTest(**params):
                result = await invoke(self.env, reduce.statistic, {"data": source, **params})
                scalar = await self.opened(result.handle)
                self.assertIsNone(scalar.uncertainty)
                record = await self.record(result.handle)
                spread = record.parameters["spread"]
                self.assertEqual((spread["value"], spread["count"]), (result.summary["spread"], 6))
                self.assertEqual(spread["basis"], result.summary["spread_basis"])
                self.assertIn("not an uncertainty", record.descriptor.summary)
                self.assertIn(result.summary["spread_basis"], record.descriptor.summary)

    async def test_max_is_over_every_observation_and_a_percentile_needs_its_rank(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        top = await invoke(self.env, reduce.statistic, {"data": source, "stat": "max"})
        self.assertEqual(top.summary["value"], 11.0)
        p90 = await invoke(self.env, reduce.statistic, {"data": source, "stat": "percentile", "percentile": 90})
        self.assertAlmostEqual(p90.summary["value"], float(np.percentile(ramp().mean(axis=0), 90)))
        failure = await self.refusal(reduce.statistic, {"data": source, "stat": "percentile"})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)

    async def test_a_series_line_and_a_missing_line(self):
        frame = pd.DataFrame({
            "time": ["2024-06-01T10Z", "2024-06-01T11Z", "2024-06-02T10Z"],
            "value": [0.04, 0.06, 0.10], "series": ["a", "a", "b"],
        })
        handle = await self.series(frame)
        result = await invoke(self.env, reduce.statistic, {"data": handle, "stat": "max", "line": "a"})
        self.assertEqual((result.summary["value"], result.summary["count"], result.unit), (0.06, 2, "ppm"))
        self.assertEqual((await self.record(result.handle)).descriptor.masking, {"certified": True})
        failure = await self.refusal(reduce.statistic, {"data": handle, "line": "zzz"})
        self.assertEqual(failure.result, {"refusal": "line_not_found"})

    async def test_nothing_valid_is_insufficient_evidence(self):
        source = await self.grid(cube(np.full((3, 2, 3), np.nan), THREE_DAYS))
        result = await invoke(self.env, reduce.statistic, {"data": source})
        self.assertEqual(result.status, "insufficient_evidence")


class DifferenceTests(OperationsCase):
    async def test_b_minus_a_with_its_summary(self):
        a = await self.grid(cube(ramp(), THREE_DAYS), period=("2024-06-01", "2024-06-03"))
        b = await self.grid(cube(ramp(10.0), ["2024-07-01", "2024-07-02", "2024-07-03"]), period=("2024-07-01", "2024-07-03"))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b, "threshold": 5.0})

        self.assertEqual((result.status, result.kind, result.unit), ("done", "grid", "ppb"))
        np.testing.assert_allclose((await self.opened(result.handle))["no2"].values, np.full((2, 3), 10.0))
        s = result.summary
        self.assertAlmostEqual(s["mean"], 10.0)
        self.assertEqual((s["min"], s["max"], s["paired_cells"]), (10.0, 10.0, 6))
        self.assertEqual(s["max_location"], {"lat": 40.0, "lon": -75.0})
        self.assertAlmostEqual(s["mean_difference"], 10.0)
        self.assertAlmostEqual(s["percent_change"], 10.0 / area_weighted_mean(cube(ramp(), THREE_DAYS)["no2"].mean("time")) * 100)
        self.assertEqual(s["area_exceeding_threshold"]["n_cells"], 6)
        self.assertEqual(result.read("summary.max_location"), {"lat": 40.0, "lon": -75.0})
        self.assertEqual((s["paired_fraction"], s["cells_valid_in_either"]), (1.0, 6))
        self.assertIn("over 6 cells valid on both, 100% of the 6 cells valid in either", result.method)
        self.assertEqual(result.disclosures, ())
        self.assertNotIn("products", s)

        record = await self.record(result.handle)
        self.assertEqual(record.parents, (a, b))
        d = record.descriptor
        self.assertEqual((d.quantity, d.unit, d.product, d.region, d.period), ("no2_tropospheric_column difference", "ppb", "TEMPO_NO2", REGION, None))
        self.assertEqual(d.masking["qa_status"], QA_VERIFIED)
        self.assertEqual(d.masking["region_cells"], 6)
        self.assertEqual(d.masking["inputs"], [MASKING, MASKING])
        self.assertEqual(result.values["period_b"], {"start": "2024-07-01", "end": "2024-07-03"})
        self.assert_bounded(result)

    async def test_percent_change_is_in_percent(self):
        a = await self.grid(cube(np.full((3, 2, 3), 10.0), THREE_DAYS))
        b = await self.grid(cube(np.full((3, 2, 3), 15.0), THREE_DAYS))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b, "change": "percent"})
        self.assertEqual((result.unit, result.summary["mean"], result.summary["withheld_cells"]), ("%", 50.0, 0))
        self.assertEqual((await self.record(result.handle)).descriptor.quantity, "no2_tropospheric_column percent change")

    async def test_cells_missing_on_either_side_are_left_out(self):
        values = ramp()
        values[:, 0, 0] = np.nan
        a = await self.grid(cube(values, THREE_DAYS))
        b = await self.grid(cube(ramp(1.0), THREE_DAYS))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b})
        self.assertEqual(result.summary["paired_cells"], 5)
        self.assertTrue(np.isnan((await self.opened(result.handle))["no2"].values[0, 0]))

    async def test_mismatched_units_are_refused(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(), THREE_DAYS, units="ppm"), unit="ppm")
        failure = await self.refusal(compare.difference, {"a": a, "b": b})
        self.assertEqual((failure.category, failure.result), (FailureCategory.BAD_INPUT, {"refusal": "unit_mismatch"}))
        self.assertEqual(
            failure.reason,
            "operations.difference: the inputs are in different units (ppb and ppm); units are not converted, so they cannot be compared.",
        )

    async def test_units_on_one_side_only_are_refused(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(), THREE_DAYS, units=None), unit=None)
        failure = await self.refusal(compare.difference, {"a": a, "b": b})
        self.assertEqual(failure.result, {"refusal": "unit_missing"})
        self.assertIn("input B declares none", failure.reason)

    async def test_different_quantities_or_variables_are_refused(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(), THREE_DAYS), quantity="hcho_total_column")
        failure = await self.refusal(compare.difference, {"a": a, "b": b})
        self.assertEqual(failure.result, {"refusal": "variable_mismatch"})
        self.assertIn("no2_tropospheric_column and hcho_total_column", failure.reason)

        c = await self.grid(cube(ramp(), THREE_DAYS, name="hcho"), quantity=None)
        d = await self.grid(cube(ramp(), THREE_DAYS), quantity=None)
        self.assertEqual((await self.refusal(compare.difference, {"a": d, "b": c})).result, {"refusal": "variable_mismatch"})

    async def test_disjoint_periods_are_refused_only_when_one_period_is_asked_for(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(), ["2024-07-01", "2024-07-02", "2024-07-03"]), period=("2024-07-01", "2024-07-03"))
        failure = await self.refusal(compare.difference, {"a": a, "b": b, "over": "one_period"})
        self.assertEqual(failure.result, {"refusal": "disjoint_periods"})
        self.assertIn("2024-06-01 to 2024-06-03 and 2024-07-01 to 2024-07-03", failure.reason)
        self.assertEqual((await invoke(self.env, compare.difference, {"a": a, "b": b})).status, "done")

    async def test_grids_of_different_resolution_or_offset_are_refused(self):
        coarse_lat, coarse_lon = [40.0, 41.0], [-75.0, -74.0, -73.0]
        fine_lat, fine_lon = list(np.arange(40.0, 41.01, 0.25)), list(np.arange(-75.0, -72.99, 0.25))
        coarse = cube(np.ones((3, 2, 3)), THREE_DAYS, lats=coarse_lat, lons=coarse_lon)
        fine = cube(np.ones((3, len(fine_lat), len(fine_lon))), THREE_DAYS, lats=fine_lat, lons=fine_lon)
        offset = cube(np.ones((3, 2, 3)), THREE_DAYS, lats=coarse_lat, lons=[-74.5, -73.5, -72.5])
        for a_data, b_data in ((coarse, fine), (fine, coarse), (coarse, offset)):
            with self.subTest(a=a_data.sizes["lon"], b=b_data.sizes["lon"], b_lon=float(b_data["lon"][0])):
                a, b = await self.grid(a_data), await self.grid(b_data, product="TROPOMI_NO2")
                failure = await self.refusal(compare.difference, {"a": a, "b": b})
                self.assertEqual((failure.category, failure.result), (FailureCategory.BAD_INPUT, {"refusal": "grid_mismatch"}))
                self.assertIn("are not regridded", failure.reason)
        failure = await self.refusal(compare.difference, {"a": await self.grid(coarse), "b": await self.grid(fine)})
        self.assertIn("a has 1 by 1 degree cells, b has 0.25 by 0.25 degree", failure.reason)

    async def test_two_products_on_the_same_grid_are_compared_and_named(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(1.0), THREE_DAYS), product="TROPOMI_NO2")
        result = await invoke(self.env, compare.difference, {"a": a, "b": b})
        self.assertEqual(result.status, "done")
        self.assertEqual(result.summary["products"], {"a": "TEMPO_NO2", "b": "TROPOMI_NO2"})
        [disclosure] = result.disclosures
        self.assertIn("Two products are compared: a is TEMPO_NO2 and b is TROPOMI_NO2", disclosure)
        self.assertIsNone((await self.record(result.handle)).descriptor.product)

    async def test_too_few_paired_cells_is_insufficient_evidence(self):
        # Same lattice; b reaches two columns past a, so a third of the valid cells pair.
        a = await self.grid(cube(np.ones((3, 2, 1)), THREE_DAYS, lons=[-74.0]))
        b = await self.grid(cube(np.ones((3, 2, 3)), THREE_DAYS))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b})
        self.assertEqual((result.status, result.values["situation"]), ("insufficient_evidence", "too_little_valid"))
        self.assertEqual(
            result.reason,
            "Too little valid data: only 2 of the 6 cells valid in either grid are valid in both (33%); "
            "at least 50% is needed to compare them.",
        )
        # Above the floor it is compared over the overlap, and the fraction is reported.
        c = await self.grid(cube(np.ones((3, 2, 3)), THREE_DAYS))
        d = await self.grid(cube(np.ones((3, 2, 2)), THREE_DAYS, lons=[-75.0, -74.5]))
        partial = await invoke(self.env, compare.difference, {"a": c, "b": d})
        s = partial.summary
        self.assertEqual((partial.status, s["paired_cells"], s["cells_valid_in_either"], s["paired_fraction"]), ("done", 4, 6, 0.667))
        self.assertIn("67% of the 6 cells valid in either", partial.method)

    async def test_two_different_periods_are_both_recorded_on_the_handle(self):
        july = ["2024-07-01", "2024-07-02", "2024-07-03"]
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(1.0), july), period=("2024-07-01", "2024-07-03"))
        record = await self.record((await invoke(self.env, compare.difference, {"a": a, "b": b})).handle)
        self.assertIsNone(record.descriptor.period)
        self.assertEqual(record.parameters["periods"], {
            "a": {"start": "2024-06-01", "end": "2024-06-03"}, "b": {"start": "2024-07-01", "end": "2024-07-03"},
        })
        self.assertIn("(a covers 2024-06-01 to 2024-06-03; b covers 2024-07-01 to 2024-07-03)", record.descriptor.summary)
        # One shared period stays the descriptor's period, with nothing added.
        c = await self.grid(cube(ramp(1.0), THREE_DAYS))
        same = await self.record((await invoke(self.env, compare.difference, {"a": a, "b": c})).handle)
        self.assertEqual(same.descriptor.period, Period(start="2024-06-01", end="2024-06-03"))
        self.assertNotIn("periods", same.parameters)
        self.assertNotIn("covers", same.descriptor.summary)

    async def test_grids_with_no_cell_in_common_are_refused_as_empty_overlap(self):
        a = await self.grid(cube(ramp(), THREE_DAYS))
        b = await self.grid(cube(ramp(), THREE_DAYS, lons=[-70.0, -69.5, -69.0]))
        failure = await self.refusal(compare.difference, {"a": a, "b": b})
        self.assertEqual(failure.result, {"refusal": "empty_overlap"})

    async def test_no_cell_valid_on_both_sides_is_insufficient_evidence(self):
        left, right = ramp(), ramp()
        left[:, :, :2] = np.nan
        right[:, :, 2] = np.nan
        a, b = await self.grid(cube(left, THREE_DAYS)), await self.grid(cube(right, THREE_DAYS))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b})
        self.assertEqual((result.status, result.values["situation"]), ("insufficient_evidence", "too_little_valid"))

    async def test_a_percent_change_withheld_everywhere_is_insufficient_evidence(self):
        a = await self.grid(cube(np.zeros((3, 2, 3)), THREE_DAYS))
        b = await self.grid(cube(np.ones((3, 2, 3)), THREE_DAYS))
        result = await invoke(self.env, compare.difference, {"a": a, "b": b, "change": "percent"})
        self.assertEqual((result.status, result.values["situation"]), ("insufficient_evidence", "baseline_too_small"))


class DailyPeakTests(OperationsCase):
    async def test_each_day_s_peak_of_a_grid(self):
        values = ramp()
        values[1, 0, 1] = 99.0
        source = await self.grid(cube(values, ["2024-06-01T10", "2024-06-01T15", "2024-06-02T10"]))
        result = await invoke(self.env, events.daily_peak, {"data": source})

        self.assertEqual((result.status, result.kind), ("done", "table"))
        table = await self.opened(result.handle)
        self.assertEqual(table["date"].tolist(), ["2024-06-01", "2024-06-02"])
        self.assertEqual(table["value"].tolist(), [99.0, 11.0])
        s = result.summary
        self.assertEqual((s["days"], s["max"], s["max_date"], s["max_location"]), (2, 99.0, "2024-06-01", {"lat": 40.0, "lon": -74.5}))
        self.assertTrue(s["max_time"].startswith("2024-06-01T15"))
        self.assertEqual(result.method, "Each day's highest value was found over every valid grid cell and time step, for 2 days.")
        self.assertEqual((await self.record(result.handle)).descriptor.masking, MASKING)

    async def test_each_day_s_peak_of_a_series_names_the_line(self):
        frame = pd.DataFrame({
            "time": ["2024-06-01T10Z", "2024-06-01T11Z", "2024-06-02T10Z"], "value": [0.04, 0.06, 0.10],
            "series": ["a", "b", "a"],
        })
        result = await invoke(self.env, events.daily_peak, {"data": await self.series(frame)})
        self.assertEqual(result.summary["max_location"], "a")
        self.assertEqual(result.values["dates"], ["2024-06-01", "2024-06-02"])

    async def test_a_grid_without_time_is_refused(self):
        failure = await self.refusal(events.daily_peak, {"data": await self.grid(flat(np.ones((2, 3))))})
        self.assertEqual(failure.result, {"refusal": "no_time_axis"})

    async def test_many_days_stay_within_the_result_bound(self):
        days = [d.date().isoformat() for d in pd.date_range("2024-01-01", periods=90)]
        source = await self.grid(cube(np.random.default_rng(1).random((90, 2, 3)), days))
        result = await invoke(self.env, events.daily_peak, {"data": source})
        self.assertEqual((result.summary["days"], len(result.values["dates"]), result.values["dates_truncated"]), (90, 31, True))
        self.assertEqual(len(await self.opened(result.handle)), 90)
        self.assert_bounded(result)


class ThresholdTests(OperationsCase):
    async def test_a_grid_against_a_given_level_reports_area_and_days(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        result = await invoke(self.env, events.threshold, {"data": source, "level": 7.5})

        table = await self.opened(result.handle)
        self.assertEqual(table["cells_above"].tolist(), [0, 1, 4])
        s = result.summary
        self.assertEqual((s["days_above"], s["days_with_data"], s["cells_ever_above"], s["level"], s["unit"]), (2, 3, 4, 7.5, "ppb"))
        self.assertGreater(s["area_fraction_ever_above"], 0.5)
        self.assertEqual(result.values["standard"], "a threshold of 7.5 ppb")
        self.assertEqual(result.values["days"], ["2024-06-02", "2024-06-03"])
        self.assertEqual(
            result.method,
            "Values were compared with a threshold of 7.5 ppb (compared as given, without truncation), a value exceeding "
            "it when above it; 2 of 3 days with data had a value above it.",
        )
        self.assertFalse(s["truncated"])

    async def test_a_series_against_the_ozone_standard_names_it(self):
        frame = pd.DataFrame({
            "time": ["2024-06-01T10Z", "2024-06-02T10Z", "2024-06-03T10Z"], "value": [0.065, 0.075, 0.070],
            "series": ["34-001-0001"] * 3,
        })
        result = await invoke(self.env, events.threshold, {"data": await self.series(frame), "pollutant": "o3"})
        self.assertEqual((result.summary["values_above"], result.summary["days_above"], result.summary["level"]), (1, 1, 0.07))
        self.assertEqual(result.values["standard"], "the o3 8-hour NAAQS level of 0.07 ppm")
        self.assertTrue(any("not a violation" in d for d in result.disclosures))
        self.assertEqual((await self.opened(result.handle))["value"].tolist(), [0.075])

    async def test_against_a_standard_values_are_truncated_as_for_the_aqi(self):
        frame = pd.DataFrame({
            "time": ["2024-06-01T10Z", "2024-06-02T10Z"], "value": [0.0705, 0.0712], "series": ["34-001-0001"] * 2,
        })
        handle = await self.series(frame)
        result = await invoke(self.env, events.threshold, {"data": handle, "pollutant": "o3"})
        # 0.0705 ppm is Moderate on the AQI, so it is not above the 0.070 level either.
        self.assertEqual((result.summary["values_above"], result.summary["truncated"]), (1, True))
        self.assertEqual((await self.opened(result.handle))["value"].tolist(), [0.0712])
        self.assertIn("truncated to 3 decimal(s) (ppm) before comparing, as for the AQI category", result.method)
        self.assertTrue(any("only once truncated" in d for d in result.disclosures))
        # The same values against the level given as a number are compared raw.
        raw = await invoke(self.env, events.threshold, {"data": handle, "level": 0.070})
        self.assertEqual((raw.summary["values_above"], raw.summary["truncated"]), (2, False))

    async def test_a_standard_in_another_unit_is_refused(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        failure = await self.refusal(events.threshold, {"data": source, "pollutant": "o3"})
        self.assertEqual(failure.result, {"refusal": "standard_unit_mismatch"})
        self.assertIn("ppb and the o3 standard in ppm", failure.reason)

    async def test_a_pollutant_with_several_standards_without_one_is_bad_input(self):
        frame = pd.DataFrame({"time": ["2024-06-01T10Z"], "value": [40.0], "series": ["a"]})
        handle = await self.series(frame, unit="ug/m3")
        missing = await self.refusal(events.threshold, {"data": handle, "pollutant": "pm25"})
        self.assertEqual(missing.category, FailureCategory.BAD_INPUT)
        self.assertEqual(
            missing.reason,
            "operations.threshold needs a standard for pm25; call it again with averaging_period one of: 24-hour, annual.",
        )
        description = events.ThresholdParams.model_fields["averaging_period"].description or ""
        self.assertIn("one of: 1-hour, 8-hour, 24-hour, annual.", description)
        done = await invoke(self.env, events.threshold, {"data": handle, "pollutant": "pm25", "averaging_period": "24-hour"})
        self.assertEqual(done.summary["values_above"], 1)
        unknown = await self.refusal(events.threshold, {"data": handle, "pollutant": "pm25", "averaging_period": "1-hour"})
        self.assertEqual(unknown.result, {"refusal": "unknown_standard"})
        self.assertEqual(
            unknown.reason,
            "operations.threshold: the standards file has no pm25 standard for 1-hour; "
            "call it again with averaging_period one of: 24-hour, annual.",
        )

    async def test_a_wrong_period_lists_only_that_pollutants_periods(self):
        frame = pd.DataFrame({"time": ["2024-06-01T10Z"], "value": [0.075], "series": ["a"]})
        handle = await self.series(frame)
        wrong = await self.refusal(events.threshold, {"data": handle, "pollutant": "o3", "averaging_period": "annual"})
        self.assertEqual(wrong.result, {"refusal": "unknown_standard"})
        self.assertTrue(wrong.reason.endswith("call it again with averaging_period one of: 8-hour."), wrong.reason)

    async def test_a_level_and_a_pollutant_together_are_invalid(self):
        source = await self.grid(cube(ramp(), THREE_DAYS))
        failure = await self.refusal(events.threshold, {"data": source, "level": 1.0, "pollutant": "o3"})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class ChainingTests(OperationsCase):
    async def test_derived_grids_feed_further_operations_with_lineage(self):
        june = await self.grid(cube(ramp(), THREE_DAYS))
        july = await self.grid(cube(ramp(2.0), ["2024-07-01", "2024-07-02", "2024-07-03"]), period=("2024-07-01", "2024-07-03"))
        mean_june = await invoke(self.env, reduce.aggregate_time, {"data": june}, step="june")
        mean_july = await invoke(self.env, reduce.aggregate_time, {"data": july}, step="july")
        diff = await invoke(self.env, compare.difference, {"a": mean_june.handle, "b": mean_july.handle}, step="diff")
        self.assertAlmostEqual(diff.summary["mean"], 2.0)
        stat = await invoke(self.env, reduce.statistic, {"data": diff.handle, "stat": "max"}, step="max")
        self.assertAlmostEqual(stat.summary["value"], 2.0)

        diff_record = await self.record(diff.handle)
        self.assertEqual(diff_record.parents, (mean_june.handle, mean_july.handle))
        self.assertEqual(diff_record.descriptor.masking["region_cells"], 6)
        self.assertEqual((await self.record(stat.handle)).parents, (diff.handle,))
        self.assertEqual([e.step for e in await self.entries()], ["june", "july", "diff", "max"])
        # A reduced grid has no time axis left to reduce.
        failure = await self.refusal(reduce.regional_series_capability, {"data": mean_june.handle})
        self.assertEqual(failure.result, {"refusal": "no_time_axis"})

    async def test_a_retrieval_grid_s_masking_provenance_is_carried_through(self):
        area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle
        area_record = await self.record(area)
        flags = np.zeros((1, 8, 7), dtype="int16")
        flags[0, 3, 2] = 1
        self.files["obs_tempo"] = granule(flags=flags, value=3.0e15)
        cfg = load_registry()["TEMPO_NO2"]
        await self.env.context.storage.handles.record_retrieval(HandleRecord(
            id="obs_tempo", kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
            descriptor=Descriptor(
                summary="TEMPO NO2 over New Jersey", quantity=cfg.quantity, product="TEMPO_NO2", unit=cfg.units,
                region={**area_record.descriptor.region, "area": area},
                masking={"variable": "vertical_column_troposphere"},
            ),
            created_at=NOW, payload=ServicePayload(),
        ))
        reduced = await invoke(self.env, reduce.aggregate_time, {"data": "obs_tempo"})
        masking = (await self.record(reduced.handle)).descriptor.masking
        self.assertEqual((masking["qa_status"], masking["quality_flag"]), (QA_VERIFIED, "main_data_quality_flag"))
        self.assertLess(masking["qa_pass_rate"], 1.0)
        self.assertGreater(masking["region_cells"], 0)
        self.assertEqual(reduced.summary["max"], 3.0e15)

        diff = await invoke(self.env, compare.difference, {"a": reduced.handle, "b": "obs_tempo"})
        self.assertEqual(diff.summary["max"], 0.0)
        self.assertEqual(diff.summary["paired_cells"], reduced.summary["cells"])
        self.assertEqual((await self.record(diff.handle)).descriptor.masking["qa_status"], QA_VERIFIED)


class GroupTests(unittest.TestCase):
    def test_every_module_shares_the_one_group_and_registers(self):
        registry = CapabilityRegistry()
        registry.add_group(group.GROUP)
        for module in (reduce, compare, events, pair):
            self.assertIs(module.GROUP, group.GROUP)
            for capability in module.CAPABILITIES:
                registry.register(capability)
                self.assertFalse(capability.retrieves)
                self.assertIsNone(capability.connector)
                self.assertTrue(capability.method.strip())
        self.assertEqual(
            {c.name for c in registry.in_group("operations")},
            {"operations.aggregate_time", "operations.regional_series", "operations.statistic",
             "operations.difference", "operations.daily_peak", "operations.threshold", "operations.pair_with_monitors"},
        )
        self.assertTrue(group.GROUP.notes.strip())


if __name__ == "__main__":
    unittest.main()
