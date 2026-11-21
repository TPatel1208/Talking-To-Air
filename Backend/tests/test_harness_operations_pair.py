"""operations.pair_with_monitors: nearest cell, masking, daily pairing, correlation, refusals, descriptor and lineage."""
import unittest
from typing import Any

import numpy as np
import pandas as pd

from tests.test_harness_operations import NOW, REGION, OperationsCase, cube
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.operations import pair
from tta_backend.harness.capabilities.shapes import points_frame, series_frame
from tta_backend.harness.handles import Descriptor
from tta_backend.harness.handles.model import Period
from tta_backend.harness.trace import FailureCategory

DAYS = [f"2024-01-0{n}" for n in range(1, 7)]
LATS = [40.60, 40.70, 40.80]
LONS = [-74.30, -74.20, -74.10]
COLUMN = "molecules/cm^2"
# Monitor A sits in the cell (40.70, -74.20); monitor B in (40.80, -74.10).
SITE_A = {"id": "34-017-0006", "name": "Newark Firehouse", "lat": 40.703, "lon": -74.198}
SITE_B = {"id": "34-013-0011", "name": "Bayonne", "lat": 40.798, "lon": -74.103}


def satellite(a_values: list[float], b_values: list[float] | None = None) -> Any:
    """Every cell reads 100; the cell holding monitor A reads ``a_values``, and B's ``b_values`` when given."""
    values = np.full((len(DAYS), len(LATS), len(LONS)), 100.0)
    values[:, 1, 1] = a_values
    if b_values is not None:
        values[:, 2, 2] = b_values
    return cube(values, DAYS, units=COLUMN, lats=LATS, lons=LONS)


def readings_frame(lines: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = [
        {"time": f"{day}T00:00:00Z", "value": value, "series": monitor}
        for monitor, days in lines.items() for day, value in days.items()
    ]
    return pd.DataFrame(rows)


class PairCase(OperationsCase):
    async def readings(self, lines: dict[str, dict[str, float]], *, quantity: str = "nitrogen dioxide concentration") -> str:
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="series", data=series_frame(readings_frame(lines)),
            descriptor=Descriptor(
                summary="EPA AQS daily NO2", quantity=quantity, unit="ppb", product="EPA AQS",
                region=REGION, period=Period(start="2024-01-01", end="2024-01-06"),
            ),
            operation="sources.monitors.readings", created_at=NOW,
        )
        return record.id

    async def monitors(self, sites: list[dict[str, Any]]) -> str:
        frame = points_frame(pd.DataFrame(sites))
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="points", data=frame,
            descriptor=Descriptor(summary="monitors", quantity="nitrogen dioxide monitor locations", region=REGION),
            operation="sources.monitors.find", created_at=NOW,
        )
        return record.id

    async def pair(self, grid: str, readings: str, monitors: str) -> Any:
        return await invoke(self.env, pair.pair_with_monitors, {"grid": grid, "readings": readings, "monitors": monitors}, step="pair")


class PairTests(PairCase):
    async def test_a_monitor_is_paired_with_its_nearest_cell_by_day(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: 10.0 * (i + 1) for i, day in enumerate(DAYS[:5])}})
        sites = await self.monitors([SITE_A])

        result = await self.pair(grid, ground, sites)

        self.assertEqual((result.status, result.kind, result.unit), ("done", "series", COLUMN))
        frame = await self.opened(result.handle)
        self.assertEqual(list(frame["series"]), [SITE_A["id"]] * 5)
        self.assertEqual(list(frame["value"]), [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual([t.strftime("%Y-%m-%d") for t in frame["time"]], DAYS[:5])
        self.assertEqual((result.summary["paired_days"], result.summary["pairs"]), (5, 5))
        self.assertAlmostEqual(result.summary["pooled_r"], 1.0)
        [monitor] = result.summary["per_monitor"]
        self.assertEqual((monitor["id"], monitor["n"], monitor["ground_days"]), (SITE_A["id"], 5, 5))
        self.assertAlmostEqual(monitor["r"], 1.0)

    async def test_each_monitor_reads_its_own_cell_and_the_pool_covers_both(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]), unit=COLUMN)
        ground = await self.readings({
            SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)},
            SITE_B["id"]: {day: float(i) for i, day in enumerate(DAYS)},
        })
        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        per = {m["id"]: m for m in result.summary["per_monitor"]}
        self.assertAlmostEqual(per[SITE_A["id"]]["r"], 1.0)
        self.assertAlmostEqual(per[SITE_B["id"]]["r"], -1.0)
        self.assertEqual((result.summary["pairs"], result.summary["paired_days"], result.summary["monitors_paired"]), (12, 6, 2))
        self.assertAlmostEqual(result.summary["pooled_r"], float(np.corrcoef([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], list(range(6)) * 2)[0, 1]))

    async def test_a_masked_value_is_not_paired(self):
        grid = await self.grid(satellite([1, np.nan, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        frame = await self.opened(result.handle)
        self.assertEqual([t.strftime("%Y-%m-%d") for t in frame["time"]], ["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"])
        self.assertEqual(result.summary["paired_days"], 5)

    async def test_a_day_on_one_side_only_is_left_out(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {"2024-01-02": 5.0, "2024-01-04": 7.0, "2024-01-19": 8.0}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertEqual(list((await self.opened(result.handle))["value"]), [2.0, 4.0])
        [monitor] = result.summary["per_monitor"]
        self.assertEqual((monitor["n"], monitor["ground_days"]), (2, 3))

    async def test_a_monitor_with_no_pair_is_counted_and_named_while_the_rest_are_paired(self):
        values = satellite([1, 2, 3, 4, 5, 6], [np.nan] * 6)
        grid = await self.grid(values, unit=COLUMN)
        ground = await self.readings({
            SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)},
            SITE_B["id"]: {day: float(i) for i, day in enumerate(DAYS)},
        })

        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        self.assertEqual((result.summary["monitors"], result.summary["monitors_paired"], result.summary["monitors_without_pairs"]), (2, 1, 1))
        per = {m["id"]: m for m in result.summary["per_monitor"]}
        self.assertEqual((per[SITE_B["id"]]["n"], per[SITE_B["id"]]["reason"]), (0, "no_paired_day"))
        self.assertTrue(any(SITE_B["id"] in d and "no paired day" in d for d in result.disclosures))
        self.assertEqual(set((await self.opened(result.handle))["series"]), {SITE_A["id"]})

    async def test_a_monitor_outside_the_grid_or_without_a_location_pairs_nothing(self):
        far = {"id": "06-037-0001", "name": "Los Angeles", "lat": 34.0, "lon": -118.2}
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        days = {day: float(i) for i, day in enumerate(DAYS)}
        ground = await self.readings({SITE_A["id"]: days, far["id"]: days, "99-999-9999": days})

        result = await self.pair(grid, ground, await self.monitors([SITE_A, far]))

        per = {m["id"]: m.get("reason") for m in result.summary["per_monitor"]}
        self.assertEqual(per, {SITE_A["id"]: None, far["id"]: "outside_grid", "99-999-9999": "no_location"})
        self.assertEqual(result.summary["monitors_without_pairs"], 2)

    async def test_no_pair_anywhere_is_insufficient_evidence(self):
        grid = await self.grid(satellite([np.nan] * 6), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: 1.0 for day in DAYS}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertEqual((result.status, result.handle, result.values["situation"]), ("insufficient_evidence", None, "no_pairs"))
        self.assertIn("across 1 monitors", result.reason)

    async def test_a_constant_side_leaves_r_undefined_but_keeps_the_pair_count(self):
        grid = await self.grid(satellite([5] * 6), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertEqual((result.summary["pooled_r"], result.summary["pairs"]), (None, 6))
        self.assertIn("r not defined over 6 days", " ".join(result.disclosures))

    async def test_sub_daily_satellite_values_are_averaged_to_the_day(self):
        times = pd.to_datetime(["2024-01-01T15:00", "2024-01-01T19:00", "2024-01-02T16:00"])
        values = np.full((3, 3, 3), 100.0)
        values[:, 1, 1] = [2.0, 4.0, 9.0]
        grid = await self.grid(
            cube(values, [str(t) for t in times], units=COLUMN, lats=LATS, lons=LONS), unit=COLUMN,
        )
        ground = await self.readings({SITE_A["id"]: {"2024-01-01": 1.0, "2024-01-02": 2.0}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertEqual(list((await self.opened(result.handle))["value"]), [3.0, 9.0])


WEST = {"id": "06-037-1103", "name": "Los Angeles", "lat": 34.07, "lon": -118.23}


class PairMinimumTests(PairCase):
    async def test_fewer_than_five_paired_days_give_no_r_but_keep_the_counts(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS[:4])}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        [monitor] = result.summary["per_monitor"]
        self.assertEqual((monitor["r"], monitor["n"], result.summary["pooled_r"], result.summary["pairs"]), (None, 4, None, 4))
        text = " ".join(result.disclosures)
        self.assertIn("needs at least 5 paired days", text)
        self.assertIn(f"monitor {SITE_A['id']}", text)
        self.assertIn("the pooled result", text)

    async def test_exactly_five_gives_r_and_no_disclosure(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS[:5])}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertAlmostEqual(result.summary["pooled_r"], 1.0)
        self.assertNotIn("needs at least", " ".join(result.disclosures))

    async def test_a_short_monitor_is_named_while_the_pool_still_has_r(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]), unit=COLUMN)
        ground = await self.readings({
            SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)},
            SITE_B["id"]: {day: float(i) for i, day in enumerate(DAYS[:2])},
        })

        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        per = {m["id"]: m for m in result.summary["per_monitor"]}
        self.assertIsNone(per[SITE_B["id"]]["r"])
        self.assertIsNotNone(per[SITE_A["id"]]["r"])
        self.assertIsNotNone(result.summary["pooled_r"])
        text = " ".join(result.disclosures)
        self.assertIn(f"monitor {SITE_B['id']}", text)
        self.assertNotIn("the pooled result", text)


class PairNoReadingsTests(PairCase):
    async def test_a_found_monitor_with_no_readings_is_counted_and_named(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        self.assertEqual((result.summary["monitors"], result.summary["monitors_without_pairs"]), (2, 1))
        per = {m["id"]: m for m in result.summary["per_monitor"]}
        self.assertEqual((per[SITE_B["id"]]["reason"], per[SITE_B["id"]]["n"]), ("no_readings", 0))
        self.assertTrue(any(SITE_B["id"] in d and "no paired day" in d for d in result.disclosures))


class PairLocalDayTests(PairCase):
    async def test_a_western_monitor_s_evening_scans_fall_on_its_local_day(self):
        # Los Angeles standard time is UTC-8: 23:00 UTC on the 1st is 15:00 on the 1st, and 01:00 UTC on the 2nd is 17:00 on the 1st.
        stamps = [
            "2024-01-01T23:00", "2024-01-02T01:00", "2024-01-02T23:00", "2024-01-03T01:00", "2024-01-03T23:00",
            "2024-01-04T01:00", "2024-01-04T23:00", "2024-01-05T01:00", "2024-01-05T23:00", "2024-01-06T01:00",
        ]
        values = np.full((len(stamps), 3, 3), 100.0)
        values[:, 1, 1] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        grid = await self.grid(cube(values, stamps, units=COLUMN, lats=[33.9, 34.07, 34.2], lons=[-118.4, -118.23, -118.1]), unit=COLUMN)
        ground = await self.readings({WEST["id"]: {f"2024-01-0{n}": float(n) for n in range(1, 6)}})

        result = await self.pair(grid, ground, await self.monitors([WEST]))

        frame = await self.opened(result.handle)
        self.assertEqual([t.strftime("%Y-%m-%d") for t in frame["time"]], [f"2024-01-0{n}" for n in range(1, 6)])
        # Each local day holds the 23:00 scan of one UTC date and the 01:00 scan of the next.
        self.assertEqual(list(frame["value"]), [15.0, 35.0, 55.0, 75.0, 95.0])
        self.assertEqual(result.summary["pairs"], 5)

    async def test_values_stamped_at_midnight_are_dates_and_are_not_shifted(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})

        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        frame = await self.opened(result.handle)
        self.assertEqual([t.strftime("%Y-%m-%d") for t in frame["time"]], DAYS)


class PairCountTests(PairCase):
    async def test_paired_days_are_distinct_days_and_pairs_are_monitor_days(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]), unit=COLUMN)
        days = {day: float(i) for i, day in enumerate(DAYS)}
        ground = await self.readings({SITE_A["id"]: days, SITE_B["id"]: days})

        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        self.assertEqual((result.summary["paired_days"], result.summary["pairs"]), (6, 12))
        self.assertIn("12 monitor-days over 6 days", result.method)

    async def test_monitors_sharing_a_nearest_cell_are_named(self):
        twin = {"id": "34-017-0099", "name": "Twin", "lat": 40.701, "lon": -74.199}
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        days = {day: float(i) for i, day in enumerate(DAYS)}
        ground = await self.readings({SITE_A["id"]: days, twin["id"]: days, SITE_B["id"]: days})

        result = await self.pair(grid, ground, await self.monitors([SITE_A, twin, SITE_B]))

        shared = [d for d in result.disclosures if "share one nearest satellite cell" in d]
        self.assertEqual(len(shared), 1)
        self.assertIn(SITE_A["id"], shared[0])
        self.assertIn(twin["id"], shared[0])
        self.assertNotIn(SITE_B["id"], shared[0])
        self.assertIn("count once for each of them in the pooled correlation", shared[0])

    async def test_separate_cells_give_no_shared_cell_disclosure(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]), unit=COLUMN)
        days = {day: float(i) for i, day in enumerate(DAYS)}
        ground = await self.readings({SITE_A["id"]: days, SITE_B["id"]: days})

        result = await self.pair(grid, ground, await self.monitors([SITE_A, SITE_B]))

        self.assertFalse(any("share one nearest" in d for d in result.disclosures))


class PairRecordTests(PairCase):
    async def test_the_handle_carries_the_grid_unit_masking_lineage_and_the_unit_disclosure(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})
        sites = await self.monitors([SITE_A])

        result = await self.pair(grid, ground, sites)

        record = await self.record(result.handle)
        self.assertEqual(set(record.parents), {grid, ground, sites})
        self.assertEqual(record.operation, "operations.pair_with_monitors")
        d = record.descriptor
        self.assertEqual((d.unit, d.quantity, d.product), (COLUMN, "no2_tropospheric_column", "TEMPO_NO2"))
        self.assertEqual(d.region, REGION)
        self.assertEqual(d.masking["qa_status"], (await self.record(grid)).descriptor.masking["qa_status"])
        first = result.disclosures[0]
        self.assertIn(f"tropospheric column ({COLUMN})", first)
        self.assertIn("surface concentrations (ppb)", first)
        self.assertIn("pattern over time, not in level", first)
        self.assertIn("no bias is given", first)
        self.assertIn(f"{SITE_A['id']} r 1.00 over 6 days", " ".join(result.disclosures))
        self.assertEqual(
            result.method,
            "Each monitor's nearest grid cell was paired with its daily readings by local day: 6 monitor-days over 6 days, "
            "at 1 of 1 monitors.",
        )
        self.assert_bounded(result)
        [entry] = await self.entries()
        self.assertEqual((entry.capability, entry.status, entry.step), ("operations.pair_with_monitors", "done", "pair"))

    async def test_the_output_is_a_series_the_chart_and_the_statistic_take(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {day: float(i) for i, day in enumerate(DAYS)}})
        result = await self.pair(grid, ground, await self.monitors([SITE_A]))

        self.assertTrue(pair.pair_with_monitors.output_kind == "series")
        from tta_backend.harness.capabilities.operations import reduce

        mean = await invoke(self.env, reduce.statistic, {"data": result.handle, "stat": "mean"})
        self.assertEqual(mean.summary["value"], 3.5)


class PairRefusalTests(PairCase):
    async def test_a_quantity_and_pollutant_that_are_not_an_accepted_pair_are_refused_naming_the_pairs(self):
        cases = {
            "wrong pollutant": ("no2_tropospheric_column", "ozone concentration"),
            "wrong quantity": ("o3_total_column", "nitrogen dioxide concentration"),
            "no quantity": (None, "nitrogen dioxide concentration"),
        }
        for label, (quantity, pollutant) in cases.items():
            with self.subTest(label):
                grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN, quantity=quantity)
                ground = await self.readings({SITE_A["id"]: {"2024-01-01": 1.0}}, quantity=pollutant)
                failure = await self.refusal(
                    pair.pair_with_monitors, {"grid": grid, "readings": ground, "monitors": await self.monitors([SITE_A])},
                )
                self.assertEqual((failure.category, failure.result), (FailureCategory.BAD_INPUT, {"refusal": "unsupported_pair"}))
                self.assertIn("the accepted pairs are no2_tropospheric_column with no2", failure.reason)

    async def test_a_grid_without_a_time_axis_is_refused(self):
        from tests.test_harness_operations import flat

        grid = await self.grid(flat(np.ones((3, 3)), lats=LATS, lons=LONS), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {"2024-01-01": 1.0}})
        failure = await self.refusal(
            pair.pair_with_monitors, {"grid": grid, "readings": ground, "monitors": await self.monitors([SITE_A])},
        )
        self.assertEqual(failure.result, {"refusal": "no_time_axis"})

    async def test_a_handle_of_the_wrong_kind_is_refused(self):
        grid = await self.grid(satellite([1, 2, 3, 4, 5, 6]), unit=COLUMN)
        ground = await self.readings({SITE_A["id"]: {"2024-01-01": 1.0}})
        failure = await self.refusal(pair.pair_with_monitors, {"grid": ground, "readings": ground, "monitors": grid})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertIn("must be a handle of kind", failure.reason)


if __name__ == "__main__":
    unittest.main()
