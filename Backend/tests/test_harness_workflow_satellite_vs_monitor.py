"""The satellite-vs-monitor workflow run end to end in memory on a fake earthdata service and AQS."""
import unittest
from typing import Any

import numpy as np
import xarray as xr

from tests.test_harness_grids import granule
from tests.test_harness_sources_monitors import daily_row, monitor_row
from tests.test_harness_sources_satellite import NO2
from tests.test_harness_workflow_packs_run import (
    GET_DATA,
    SITE_A,
    SITE_B,
    TIMESERIES,
    PackRun,
    step,
)
from tests.test_harness_workflow_runner import AREA_ENTRY
from tta_backend.harness.results import Mode

JUNE = {"start": "2024-06-01", "end": "2024-06-06"}
NO2_STANDARD = "NO2 1-hour 2010"
DAYS = [f"2024-06-0{n}" for n in range(1, 7)]
# The satellite fake reads 2.0e15, 2.1e15 ... on the six days, in every cell.
SATELLITE = [2.0e15 + n * 1e14 for n in range(6)]
GROUND_A = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
GROUND_B = [5.0, 25.0, 15.0, 45.0, 35.0, 55.0]


def six_days(first: str) -> xr.Dataset:
    return xr.concat([granule(day=day, value=2.0e15 + n * 1e14) for n, day in enumerate(DAYS)], dim="time")


def no2_row(site: dict[str, Any], day: str, value: float) -> dict[str, Any]:
    return daily_row(site, day, value, standard=NO2_STANDARD, units_of_measure="Parts per billion", sample_duration="1 HOUR")


class SatelliteVsMonitorRuns(PackRun):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        days = DAYS
        self.service.make_file = six_days
        self.aqs.rows = {
            "monitors": [monitor_row(SITE_A), monitor_row(SITE_B)],
            "dailyData": [
                *(no2_row(SITE_A, day, value) for day, value in zip(days, GROUND_A, strict=True)),
                *(no2_row(SITE_B, day, value) for day, value in zip(days, GROUND_B, strict=True)),
            ],
        }
        self.inputs = {"quantity": NO2, "area": "New Jersey", "dates": JUNE}

    async def run_pairing(self, inputs: dict[str, Any] | None = None):
        result = await self.f.start(inputs or self.inputs, name="satellite-vs-monitor")
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.mode, Mode.draft("satellite-vs-monitor", 1))
        return result

    def finding(self, result, name: str) -> Any:
        return next(f.value for f in result.findings if f.name == name)

    async def test_the_steps_run_once_each_with_one_satellite_retrieval(self):
        result = await self.run_pairing()
        entries = await self.f.entries(result.trace_id)

        self.assertEqual(entries[0], AREA_ENTRY)
        done = sorted(e[2] for e in entries if e[3] == "done")
        self.assertEqual(
            done, sorted(["input.area", "find", "data", "readings", "pair", "figure_series", "monitor_values", "figure_map"]),
        )
        self.assertIn(step(GET_DATA, "data", "waiting"), entries)
        # One gridded retrieval serves both monitors.
        self.assertEqual(self.service.jobs, {"job_1": "obs_1"})
        self.assertEqual(self.service.series, None)

    async def test_the_findings_are_the_pooled_correlation_and_the_paired_days(self):
        result = await self.run_pairing()

        self.assertEqual(
            [f.name for f in result.findings],
            ["monitors_found", "monitors_paired", "monitors_without_pairs", "paired_days", "pairs", "correlation_r"],
        )
        self.assertEqual(self.finding(result, "monitors_found"), 2)
        self.assertEqual(self.finding(result, "monitors_paired"), 2)
        self.assertEqual(self.finding(result, "monitors_without_pairs"), 0)
        self.assertEqual(self.finding(result, "paired_days"), 6)
        self.assertEqual(self.finding(result, "pairs"), 12)
        expected = float(np.corrcoef(SATELLITE * 2, GROUND_A + GROUND_B)[0, 1])
        self.assertAlmostEqual(self.finding(result, "correlation_r"), expected)
        self.assertNotIn("bias", " ".join(f.name for f in result.findings))

    async def test_the_answer_says_a_column_is_compared_with_surface_readings(self):
        result = await self.run_pairing()

        text = " ".join(result.disclosures)
        self.assertIn("tropospheric column (molecules/cm^2)", text)
        self.assertIn("surface concentrations (ppb)", text)
        self.assertIn("not in level", text)
        self.assertIn("Per monitor, correlation r over its paired days", text)

    async def test_it_draws_a_time_series_of_both_sides_and_a_monitor_map(self):
        result = await self.run_pairing()

        self.assert_charts_stored(result, 2)
        entries = await self.f.entries(result.trace_id)
        self.assertIn(step(TIMESERIES, "figure_series"), entries)
        self.assertIn(step("render.monitor_map", "figure_map"), entries)

    async def test_the_handles_returned_are_the_pairs_the_readings_and_the_monitors(self):
        result = await self.run_pairing()

        kinds = sorted(h.split("_")[0] for h in result.handles)
        self.assertEqual(kinds, ["points", "series", "series"])

    async def test_a_named_product_is_the_one_used(self):
        result = await self.run_pairing({**self.inputs, "named": "TEMPO_NO2"})

        self.assertTrue(any(d.startswith("Product: TEMPO_NO2") for d in result.disclosures))

    async def test_a_monthly_product_cannot_be_paired_by_day(self):
        result = await self.f.start({**self.inputs, "named": "TROPOMI_NO2"}, name="satellite-vs-monitor")

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("so daily values are not available", result.reason or "")
        self.assertEqual(self.service.jobs, {})

    async def test_a_quantity_the_pairing_does_not_accept_gets_no_retrieval(self):
        result = await self.f.start({**self.inputs, "quantity": "o3_total_column"}, name="satellite-vs-monitor")

        self.assertNotEqual(result.status, "done")
        self.assertEqual(self.service.jobs, {})

    async def test_a_monitor_without_ground_readings_is_counted_with_its_reason(self):
        self.aqs.rows["dailyData"] = [r for r in self.aqs.rows["dailyData"] if r["site_number"] != SITE_B["site_number"]]

        result = await self.run_pairing()

        self.assertEqual(self.finding(result, "monitors_found"), 2)
        self.assertEqual(self.finding(result, "monitors_without_pairs"), 1)
        self.assertEqual(self.finding(result, "paired_days"), 6)
        self.assertEqual(self.finding(result, "pairs"), 6)
        self.assertAlmostEqual(self.finding(result, "correlation_r"), 1.0)
        self.assertTrue(any("no paired day" in d and "34-007-0003" in d for d in result.disclosures))

    async def test_a_short_overlap_gives_no_correlation_finding(self):
        self.aqs.rows["dailyData"] = [r for r in self.aqs.rows["dailyData"] if r["date_local"] <= "2024-06-03" and r["site_number"] == SITE_A["site_number"]]

        result = await self.run_pairing()

        self.assertNotIn("correlation_r", [f.name for f in result.findings])
        self.assertEqual(self.finding(result, "pairs"), 3)
        self.assertTrue(any("needs at least 5 paired days" in d for d in result.disclosures))


if __name__ == "__main__":
    unittest.main()
