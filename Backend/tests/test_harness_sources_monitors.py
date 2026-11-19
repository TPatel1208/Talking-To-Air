"""The ground monitor capabilities on the framework fakes, against a fake EPA AQS."""
import dataclasses
import datetime
import json
import tempfile
import typing
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.domain.monitors.pollutants import POLLUTANTS, Pollutant
from tta_backend.harness.capabilities import CapabilityRegistry, invoke
from tta_backend.harness.capabilities.sources import monitors
from tta_backend.harness.handles import Descriptor, HandleRecord
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.testing import FixedClock, StaticCredentials, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

EMAIL, KEY = "researcher@lab.edu", "aqs-k3y-value"
SECRET = json.dumps({"email": EMAIL, "key": KEY})
NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

# Two sites near downtown Los Angeles, one in Pasadena; site A has two ozone instruments.
SITE_A = {"state_code": "06", "county_code": "037", "site_number": "1103", "latitude": 34.0669, "longitude": -118.2268,
          "local_site_name": "Los Angeles-N. Main Street"}
SITE_B = {"state_code": "06", "county_code": "037", "site_number": "2005", "latitude": 34.1326, "longitude": -118.1272,
          "local_site_name": "Pasadena"}
SITE_FAR = {"state_code": "06", "county_code": "071", "site_number": "0001", "latitude": 34.9, "longitude": -117.0,
            "local_site_name": "Desert"}


def polygon(corners: list[tuple[float, float]]) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[list(c) for c in [*corners, corners[0]]]]}


# Inside the default area's box: holds SITE_A, leaves out SITE_B.
TRIANGLE = polygon([(-118.3, 34.0), (-118.1, 34.0), (-118.3, 34.2)])


def monitor_row(site: dict[str, Any], poc: int = 1) -> dict[str, Any]:
    return {**site, "poc": poc, "parameter_name": "Ozone", "city_name": "LA", "county_name": "Los Angeles",
            "state_name": "California", "open_date": "1990-01-01", "close_date": None}


def sample_row(site: dict[str, Any], day: str, hour: int, value: float, poc: int = 1) -> dict[str, Any]:
    return {**site, "poc": poc, "date_local": day, "time_local": f"{hour:02d}:00", "date_gmt": day,
            "time_gmt": f"{(hour + 8) % 24:02d}:00", "sample_measurement": value, "units_of_measure": "Parts per million",
            "sample_duration": "1 HOUR", "qualifier": None}


def daily_row(site: dict[str, Any], day: str, value: float, *, standard: str = "Ozone 8-hour 2015", poc: int = 1,
              **extra: Any) -> dict[str, Any]:
    return {**site, "poc": poc, "date_local": day, "pollutant_standard": standard, "first_max_value": value,
            "arithmetic_mean": value / 2, "units_of_measure": "Parts per million",
            "sample_duration": "8-HR RUN AVG BEGIN HOUR", "aqi": round(value * 1000), "observation_count": 24, **extra}


def quarterly_row(site: dict[str, Any], year: int, quarter: int, value: float) -> dict[str, Any]:
    return {**site, "poc": 1, "year": str(year), "quarter": str(quarter), "pollutant_standard": "Ozone 8-hour 2015",
            "arithmetic_mean": value, "maximum_value": value * 2, "minimum_value": value / 2,
            "units_of_measure": "Parts per million", "certification_indicator": "Certified"}


class FakeAqs:
    """Answers AQS endpoints from canned rows, filtered the way EPA filters them."""

    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None, *, status: int = 200) -> None:
        self.rows = rows or {}
        self.status = status
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        q = request.url.params
        if q.get("key") != KEY or q.get("email") != EMAIL:
            return httpx.Response(400, json={"Header": [{"status": "Failed", "url": str(request.url),
                                                         "error": ["Email and/or key are invalid."]}]})
        if self.status >= 400:
            return httpx.Response(self.status, json={"Header": [{"status": "Failed", "error": ["down"]}]})
        prefix, how = request.url.path.removeprefix("/data/api/").split("/")
        rows = [r for r in self.rows.get(prefix, []) if self._matches(r, how, q)]
        status = "Success" if rows else "No data matched your selection"
        return httpx.Response(200, json={"Header": [{"status": status}], "Data": rows})

    @staticmethod
    def _matches(row: dict[str, Any], how: str, q: httpx.QueryParams) -> bool:
        if how == "bySite" and (row["state_code"], row["county_code"], row["site_number"]) != (q["state"], q["county"], q["site"]):
            return False
        if how == "byBox" and not (float(q["minlat"]) <= row["latitude"] <= float(q["maxlat"])
                                   and float(q["minlon"]) <= row["longitude"] <= float(q["maxlon"])):
            return False
        day = row.get("date_local")
        if day is not None and not q["bdate"] <= day.replace("-", "") <= q["edate"]:
            return False
        if "year" in row and not q["bdate"][:4] <= row["year"] <= q["edate"][:4]:
            return False
        return True

    def endpoints(self) -> list[str]:
        return [r.url.path.removeprefix("/data/api/") for r in self.requests]


def default_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "monitors": [monitor_row(SITE_FAR), monitor_row(SITE_B), monitor_row(SITE_A, 1), monitor_row(SITE_A, 2)],
        "sampleData": [
            sample_row(SITE_A, "2024-06-01", 12, 0.050), sample_row(SITE_A, "2024-06-01", 13, 0.060),
            sample_row(SITE_A, "2024-06-01", 12, 0.090, poc=2),
            sample_row(SITE_B, "2024-06-01", 12, 0.040),
        ],
        "dailyData": [
            daily_row(SITE_A, "2024-06-01", 0.075), daily_row(SITE_A, "2024-06-02", 0.065),
            daily_row(SITE_A, "2024-06-01", 0.050, standard="Ozone 1-hour 1979"),
            daily_row(SITE_B, "2024-06-01", 0.072), daily_row(SITE_B, "2024-06-02", 0.070),
        ],
        "quarterlyData": [quarterly_row(SITE_A, 2024, q, 0.03 + q / 100) for q in (1, 2, 3, 4)],
        "annualData": [{**quarterly_row(SITE_A, 2024, 0, 0.04), "quarter": None}],
    }


class _MonitorCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        aqs_client.reset_request_state()
        self.addCleanup(aqs_client.reset_request_state)
        self.aqs = FakeAqs(default_rows())
        real_fetcher = aqs_client.fetcher
        for p in (
            patch.object(aqs_client, "fetcher", lambda credential: real_fetcher(credential, transport=self.aqs.transport)),
            patch.object(aqs_client.asyncio, "sleep", AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.stores = memory_stores()
        self.env = await self.make_env(StaticCredentials({"epa_aqs": SECRET}))

    async def make_env(self, credentials: Any, *, stores: Stores | None = None, now: datetime.datetime = NOW) -> RunEnv:
        context = make_run_context(stores=stores or self.stores, credentials=credentials, clock=FixedClock(now))
        return await make_run_env(self._tmp.name, context=context)

    async def area(self, env: RunEnv | None = None, bounds: Any = (-118.3, 34.0, -118.1, 34.2),
                   geometry: dict[str, Any] | None = None) -> HandleRecord:
        """An area handle; its footprint defaults to its bounding box."""
        env = env or self.env
        if geometry is None and len(bounds) == 4:
            west, south, east, north = bounds
            geometry = polygon([(west, south), (east, south), (east, north), (west, north)])
        features = [] if geometry is None else [{"type": "Feature", "geometry": geometry, "properties": {}}]
        return await env.resolver.mint(
            user_id=env.context.identity.user_id, conversation_id="conv-1", kind="polygons",
            data={"type": "FeatureCollection", "features": features},
            descriptor=Descriptor(summary="Los Angeles", region={"name": "Los Angeles", "bounds": list(bounds)}),
            operation="sources.areas.resolve", created_at=NOW,
        )

    async def triangle(self) -> HandleRecord:
        """An area whose bounding box holds sites A and B but whose footprint holds only A."""
        return await self.area(geometry=TRIANGLE)

    async def failure(self, capability: Any, params: dict[str, Any], env: RunEnv | None = None) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(env or self.env, capability, params)
        return caught.exception

    async def entries(self, env: RunEnv | None = None) -> list[Any]:
        env = env or self.env
        return await self.stores.traces.entries(env.context.identity.user_id, env.run_id)


class FindTests(_MonitorCase):
    async def test_monitors_near_a_point_are_points_nearest_first_one_per_site(self):
        result = await invoke(self.env, monitors.find, {"pollutant": "o3", "lat": 34.07, "lon": -118.22,
                                                        "start": "2024-06-01", "end": "2024-06-30"})

        frame = await self.env.resolver.open("user-a", result.handle, "points")
        self.assertEqual(result.kind, "points")
        self.assertEqual(list(frame["id"]), ["06-037-1103", "06-037-2005", "06-071-0001"])
        self.assertEqual(list(frame.columns[:4]), ["id", "name", "lat", "lon"])
        self.assertEqual((frame["pollutant"][0], frame["param_code"][0], frame["pocs"][0]), ("o3", "44201", "1,2"))
        self.assertEqual(result.values["monitor_ids"], list(frame["id"]))
        self.assertEqual(result.summary["monitors"], 3)
        self.assertEqual(result.summary["period"], {"start": "2024-06-01", "end": "2024-06-30"})
        self.assertLess(result.summary["nearest_km"], 1.0)
        self.assertIn("parameter 44201", result.disclosures[0])
        self.assertEqual(result.product, "EPA AQS")
        self.assertIn("3 were found", result.method)
        record = await self.stores.handles.get("user-a", result.handle)
        # Made from coordinates alone, so it has no parents.
        self.assertEqual((record.origin, record.parents, record.operation), ("derived", (), "sources.monitors.find"))
        self.assertEqual(record.descriptor.period.start, "2024-06-01")
        sent = self.aqs.requests[0].url.params
        self.assertEqual((sent["param"], sent["bdate"], sent["edate"]), ("44201", "20240601", "20240630"))

    async def test_monitors_near_an_area_handle_say_which_lie_outside_it(self):
        area = await self.area()

        result = await invoke(self.env, monitors.find, {"pollutant": "o3", "area": area.id, "limit": 3,
                                                        "start": "2024-06-01", "end": "2024-06-30"})

        frame = await self.env.resolver.open("user-a", result.handle, "points")
        self.assertEqual(list(frame["inside_area"]), [True, True, False])
        self.assertEqual(result.summary["inside"], 2)
        self.assertIn("1 of the 3 monitors lie outside the area's boundary; they are listed as nearby monitors "
                      "with inside_area false.", result.disclosures)
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (area.id,))
        self.assertEqual(record.descriptor.region, {"name": "Los Angeles", "bounds": [-118.3, 34.0, -118.1, 34.2],
                                                    "area": area.id, "wording": None})
        box = self.aqs.requests[0].url.params
        self.assertTrue(float(box["minlon"]) <= -118.3 and float(box["maxlon"]) >= -118.1)

    async def test_a_monitor_in_the_bounding_box_but_outside_the_footprint_is_not_inside_the_area(self):
        area = await self.triangle()

        result = await invoke(self.env, monitors.find, {"pollutant": "o3", "area": area.id, "limit": 3,
                                                        "start": "2024-06-01", "end": "2024-06-30"})

        frame = await self.env.resolver.open("user-a", result.handle, "points")
        self.assertEqual(dict(zip(frame["id"], frame["inside_area"], strict=True)),
                         {"06-037-1103": True, "06-037-2005": False, "06-071-0001": False})
        self.assertEqual(result.summary["inside"], 1)

    async def test_no_monitors_is_insufficient_evidence_with_fixed_wording(self):
        self.aqs.rows["monitors"] = []

        result = await invoke(self.env, monitors.find, {"pollutant": "no2", "lat": 10.0, "lon": 10.0,
                                                        "start": "2024-06-01", "end": "2024-06-30"})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "EPA AQS returned no active Nitrogen dioxide monitors for the point or within "
                                        "5 degrees of it for 2024-06-01 to 2024-06-30.")
        [entry] = await self.entries()
        self.assertEqual(entry.failure_category, FailureCategory.NO_DATA)

    async def test_an_area_without_bounds_or_a_footprint_is_bad_input(self):
        for area in (await self.area(bounds=(1, 2)), await self.area(geometry={"type": "Polygon", "coordinates": []})):
            with self.subTest(area=area.id):
                failure = await self.failure(monitors.find, {"pollutant": "o3", "area": area.id,
                                                             "start": "2024-06-01", "end": "2024-06-02"})

                self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(self.aqs.requests, [])

    async def test_a_point_and_an_area_together_are_refused(self):
        area = await self.area()

        failure = await self.failure(monitors.find, {"pollutant": "o3", "area": area.id, "lat": 1.0, "lon": 1.0,
                                                     "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class ReadingsTests(_MonitorCase):
    async def test_hourly_readings_for_several_monitors_are_one_line_each_in_utc(self):
        result = await invoke(self.env, monitors.readings, {
            "pollutant": "o3", "monitors": ["06-037-1103", "06-037-2005"], "frequency": "hourly",
            "start": "2024-06-01", "end": "2024-06-01",
        })

        frame = await self.env.resolver.open("user-a", result.handle, "series")
        self.assertEqual(result.kind, "series")
        self.assertEqual(sorted(set(frame["series"])), ["06-037-1103", "06-037-2005"])
        site_a = frame[frame["series"] == "06-037-1103"]
        # POC 1 has the most values, so POC 2's reading at the same hour is not mixed in.
        self.assertEqual(list(site_a["value"]), [0.05, 0.06])
        self.assertEqual(str(site_a["time"].iloc[0]), "2024-06-01 20:00:00+00:00")
        self.assertEqual(result.unit, "ppm")
        self.assertEqual(result.summary["monitors"], 2)
        self.assertEqual(result.summary["values"], 3)
        self.assertEqual((result.summary["min"], result.summary["max"]), (0.04, 0.06))
        self.assertAlmostEqual(result.summary["mean"], 0.05)
        self.assertIn("Monitor 06-037-1103 reports more than one instrument; its values use POC 1 (1 HOUR).", result.disclosures)
        self.assertEqual(self.aqs.endpoints(), ["sampleData/bySite", "sampleData/bySite"])
        record = await self.stores.handles.get("user-a", result.handle)
        # Made from literal monitor ids alone, so it has no parents.
        self.assertEqual((record.parents, record.operation), ((), "sources.monitors.readings"))

    async def test_daily_readings_from_a_points_handle_use_the_standard_statistic(self):
        found = await invoke(self.env, monitors.find, {"pollutant": "o3", "lat": 34.07, "lon": -118.22, "limit": 2,
                                                       "start": "2024-06-01", "end": "2024-06-02"})

        result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "points": found.handle,
                                                            "start": "2024-06-01", "end": "2024-06-02"})

        frame = await self.env.resolver.open("user-a", result.handle, "series")
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(frame[frame["series"] == "06-037-1103"]["value"]), [0.075, 0.065])
        self.assertTrue(any("daily maximum 8-hour average" in d and "Ozone 8-hour 2015" in d for d in result.disclosures))
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (found.handle,))

    async def test_the_daily_row_including_event_data_is_used_whatever_order_aqs_sends(self):
        for first in ("Events Excluded", "Events Included"):
            with self.subTest(first=first):
                aqs_client.reset_request_state()
                rows = {"Events Excluded": daily_row(SITE_A, "2024-06-01", 0.060, event_type="Events Excluded"),
                        "Events Included": daily_row(SITE_A, "2024-06-01", 0.081, event_type="Events Included")}
                second = next(k for k in rows if k != first)
                self.aqs.rows["dailyData"] = [rows[first], rows[second],
                                              daily_row(SITE_A, "2024-06-02", 0.050, event_type="No Events")]

                result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                    "start": "2024-06-01", "end": "2024-06-02"})

                frame = await self.env.resolver.open("user-a", result.handle, "series")
                self.assertEqual(list(frame["value"]), [0.081, 0.05])
                self.assertTrue(any("'No Events', or 'Events Included'" in d for d in result.disclosures))
                self.assertIn("1 monitor-days have different values with and without exceptional-event data "
                              "(such as wildfire smoke): 06-037-1103 2024-06-01.", result.disclosures)

    async def test_a_monitor_day_with_only_an_excluded_row_uses_it_and_says_so(self):
        self.aqs.rows["dailyData"] = [daily_row(SITE_A, "2024-06-01", 0.060, event_type="Events Excluded")]

        result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                            "start": "2024-06-01", "end": "2024-06-01"})

        self.assertIn("1 monitor-days had no row including event data, so their 'Events Excluded' row is used.",
                      result.disclosures)

    async def test_a_monitor_without_readings_is_named(self):
        result = await invoke(self.env, monitors.readings, {
            "pollutant": "o3", "monitors": ["06-037-1103", "06-071-0001"], "start": "2024-06-01", "end": "2024-06-02",
        })

        self.assertIn("No daily readings for monitors: 06-071-0001.", result.disclosures)
        self.assertEqual(result.values["monitor_ids"], ["06-037-1103"])

    async def test_no_readings_is_insufficient_evidence(self):
        result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                            "start": "2023-01-01", "end": "2023-01-05"})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "EPA AQS returned no daily Ozone readings for the requested monitors for "
                                        "2023-01-01 to 2023-01-05.")

    async def test_no_recent_readings_says_they_may_not_be_published(self):
        result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                            "start": "2025-12-20", "end": "2025-12-21"})

        self.assertIn("may not be published yet", result.reason)

    async def test_recent_data_is_disclosed_as_preliminary(self):
        self.aqs.rows["dailyData"].append(daily_row(SITE_A, "2025-06-01", 0.07))

        result = await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                            "start": "2025-06-01", "end": "2025-06-01"})

        self.assertIn("Data from 2025 onward is preliminary: not yet certified, and it may change.", result.disclosures)

    async def test_a_period_across_years_is_asked_for_one_year_at_a_time(self):
        await invoke(self.env, monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                   "start": "2023-12-30", "end": "2024-06-02"})

        dates = [(r.url.params["bdate"], r.url.params["edate"]) for r in self.aqs.requests]
        self.assertEqual(dates, [("20231230", "20231231"), ("20240101", "20240602")])

    async def test_too_many_requests_or_too_long_an_hourly_period_is_too_large(self):
        ids = [f"06-037-{n:04d}" for n in range(1, 8)]
        failures = [
            await self.failure(monitors.readings, {"pollutant": "o3", "monitors": ids, "start": "2023-06-01", "end": "2024-06-01"}),
            await self.failure(monitors.readings, {"pollutant": "o3", "monitors": ids[:1], "frequency": "hourly",
                                                   "start": "2024-06-01", "end": "2024-07-15"}),
        ]

        self.assertEqual([f.category for f in failures], [FailureCategory.REQUEST_TOO_LARGE] * 2)
        self.assertEqual(self.aqs.requests, [])

    async def test_a_malformed_monitor_id_is_refused_before_any_request(self):
        failure = await self.failure(monitors.readings, {"pollutant": "o3", "monitors": ["site"],
                                                         "start": "2024-06-01", "end": "2024-06-01"})

        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(self.aqs.requests, [])


class SummaryTests(_MonitorCase):
    async def test_daily_summary_is_a_flat_table_for_the_default_standard(self):
        result = await invoke(self.env, monitors.daily_summary, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                 "start": "2024-06-01", "end": "2024-06-02"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(result.kind, "table")
        self.assertEqual(list(frame["period"]), ["2024-06-01", "2024-06-02"])
        self.assertEqual(list(frame["max"]), [0.075, 0.065])
        self.assertEqual(set(frame["pollutant_standard"]), {"Ozone 8-hour 2015"})
        self.assertIn("peak_value", frame.columns)
        # The monitor map's columns, value being the standard's daily statistic.
        self.assertEqual(list(frame["value"]), [0.075, 0.065])
        self.assertEqual(set(frame["unit"]), {"ppm"})
        self.assertEqual(list(frame["name"]), ["Los Angeles-N. Main Street"] * 2)
        self.assertEqual((frame["lat"][0], frame["lon"][0]), (34.0669, -118.2268))
        self.assertIn("daily maximum 8-hour average", result.values["value"])
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertIn("daily maximum 8-hour average", record.descriptor.quantity)
        self.assertIn("value is the daily maximum 8-hour average", record.descriptor.summary)
        self.assertEqual(result.unit, "ppm")
        self.assertEqual((result.summary["rows"], result.summary["monitors"], result.summary["max"]), (2, 1, 0.075))
        self.assertEqual(result.values["standard"], "Ozone 8-hour 2015")
        self.assertIn("Rows are AQS daily statistics for the Ozone 8-hour 2015 standard.", result.disclosures)
        self.assertIn("Ozone 8-hour 2015", result.method)

    async def test_quarterly_summary_keeps_quarters_overlapping_the_period(self):
        result = await invoke(self.env, monitors.quarterly_summary, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                     "start": "2024-05-01", "end": "2024-08-31"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(list(frame["period"]), ["2024-Q2", "2024-Q3"])
        self.assertEqual(list(frame["min"]), [0.025, 0.03])
        self.assertEqual(list(frame["value"]), list(frame["mean"]))
        self.assertTrue({"site_id", "name", "lat", "lon", "value", "unit"} <= set(frame.columns))
        self.assertIn("Quarterly rows cover whole quarters that overlap the period.", result.disclosures)
        self.assertFalse(any("preliminary" in d or "certified" in d for d in result.disclosures))

    async def test_annual_summary_over_an_area_asks_by_box(self):
        area = await self.area()

        result = await invoke(self.env, monitors.annual_summary, {"pollutant": "o3", "area": area.id,
                                                                  "start": "2024-01-01", "end": "2024-12-31"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(list(frame["site_id"]), ["06-037-1103"])
        self.assertEqual(self.aqs.endpoints(), ["monitors/byBox", "annualData/byBox"])
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual((record.parents, record.operation), ((area.id,), "sources.monitors.annual_summary"))

    async def test_an_area_summary_leaves_out_monitors_outside_the_footprint(self):
        area = await self.triangle()

        result = await invoke(self.env, monitors.daily_summary, {"pollutant": "o3", "area": area.id,
                                                                 "start": "2024-06-01", "end": "2024-06-02"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(set(frame["site_id"]), {"06-037-1103"})
        self.assertIn("1 monitors in the area's bounding box but outside its boundary were left out.", result.disclosures)

    async def test_an_area_too_large_is_refused_before_any_data_request(self):
        state = await self.area(bounds=(-125.0, 32.0, -114.0, 42.0))

        failure = await self.failure(monitors.daily_summary, {"pollutant": "o3", "area": state.id,
                                                              "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertEqual(failure.reason, "sources.monitors.daily_summary covers too large an area for EPA AQS; "
                                         "narrow the area or name the monitors.")
        self.assertEqual(self.aqs.requests, [])

    async def test_an_area_with_too_many_monitors_is_refused_after_listing_them(self):
        many = [monitor_row({**SITE_A, "site_number": f"{n:04d}"}) for n in range(monitors.MAX_AREA_MONITOR_YEARS + 1)]
        self.aqs.rows["monitors"] = many
        area = await self.area()

        failure = await self.failure(monitors.exceedance_days, {"pollutant": "o3", "area": area.id,
                                                                "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)
        self.assertIn("narrow the area or name the monitors", failure.reason)
        self.assertEqual(self.aqs.endpoints(), ["monitors/byBox"])

    async def test_an_area_without_monitors_inside_it_is_insufficient_evidence_without_a_data_request(self):
        self.aqs.rows["monitors"] = [monitor_row(SITE_B)]
        area = await self.triangle()

        result = await invoke(self.env, monitors.annual_summary, {"pollutant": "o3", "area": area.id,
                                                                  "start": "2024-01-01", "end": "2024-12-31"})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(self.aqs.endpoints(), ["monitors/byBox"])

    async def test_an_unknown_averaging_period_is_bad_input(self):
        failure = await self.failure(monitors.daily_summary, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                              "averaging": "annual", "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)

    async def test_no_summaries_is_insufficient_evidence(self):
        result = await invoke(self.env, monitors.annual_summary, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                  "start": "2020-01-01", "end": "2020-12-31"})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "EPA AQS returned no annual summaries of Ozone for the requested monitors for "
                                        "2020-01-01 to 2020-12-31.")


class ExceedanceTests(_MonitorCase):
    async def test_days_above_the_naaqs_level_are_listed_and_the_standard_named(self):
        result = await invoke(self.env, monitors.exceedance_days, {
            "pollutant": "o3", "monitors": ["06-037-1103", "06-037-2005"], "start": "2024-06-01", "end": "2024-06-02",
        })

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(list(zip(frame["date"], frame["site_id"], strict=True)),
                         [("2024-06-01", "06-037-1103"), ("2024-06-01", "06-037-2005")])
        self.assertEqual(list(frame["level"]), [0.07, 0.07])
        self.assertTrue({"site_id", "name", "lat", "lon", "value", "unit"} <= set(frame.columns))
        self.assertEqual(list(frame["name"]), ["Los Angeles-N. Main Street", "Pasadena"])
        self.assertIn("truncated to 3 decimals", result.values["value"])
        self.assertEqual(result.unit, "ppm")
        self.assertEqual(result.summary["exceedance_days"], 1)
        self.assertEqual(result.summary["monitor_days_checked"], 4)
        self.assertEqual(result.summary["max"], 0.075)
        self.assertEqual(result.values["standard"], "Ozone 8-hour NAAQS (Ozone 8-hour 2015)")
        self.assertEqual((result.values["level"], result.values["days"]), (0.07, ["2024-06-01"]))
        self.assertTrue(any(d.startswith("Applied the Ozone 8-hour NAAQS of 0.07 ppm") for d in result.disclosures))
        self.assertIn("level of 0.07 ppm; 1 days exceeded it", result.method)
        self.assertTrue(any(d.startswith("Daily values were truncated to the standard's reporting precision (3 decimals")
                            for d in result.disclosures))

    async def test_a_value_above_the_level_only_past_its_reporting_precision_does_not_exceed(self):
        self.aqs.rows["dailyData"] = [daily_row(SITE_A, "2024-06-01", 0.0705), daily_row(SITE_A, "2024-06-02", 0.0712)]

        result = await invoke(self.env, monitors.exceedance_days, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                   "start": "2024-06-01", "end": "2024-06-02"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(result.values["days"], ["2024-06-02"])
        self.assertEqual((list(frame["value"]), list(frame["measured"])), ([0.071], [0.0712]))

    async def test_exceedances_use_the_row_including_event_data(self):
        self.aqs.rows["dailyData"] = [
            daily_row(SITE_A, "2024-06-01", 0.060, event_type="Events Excluded"),
            daily_row(SITE_A, "2024-06-01", 0.085, event_type="Events Included"),
        ]

        result = await invoke(self.env, monitors.exceedance_days, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                   "start": "2024-06-01", "end": "2024-06-01"})

        self.assertEqual((result.values["days"], result.summary["monitor_days_checked"]), (["2024-06-01"], 1))

    async def test_area_exceedance_days_leave_out_monitors_outside_the_footprint(self):
        area = await self.triangle()

        result = await invoke(self.env, monitors.exceedance_days, {"pollutant": "o3", "area": area.id,
                                                                   "start": "2024-06-01", "end": "2024-06-02"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(list(frame["site_id"]), ["06-037-1103"])
        self.assertEqual((result.summary["monitors"], result.summary["monitor_days_checked"]), (1, 2))
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (area.id,))

    async def test_no_exceedances_is_an_empty_table_not_missing_evidence(self):
        self.aqs.rows["dailyData"] = [daily_row(SITE_A, "2024-06-01", 0.03)]

        result = await invoke(self.env, monitors.exceedance_days, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                   "start": "2024-06-01", "end": "2024-06-01"})

        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual((result.status, len(frame), result.summary["exceedance_days"]), ("done", 0, 0))
        self.assertIsNone(result.summary["max"])

    async def test_no_daily_values_is_insufficient_evidence(self):
        result = await invoke(self.env, monitors.exceedance_days, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                   "start": "2022-06-01", "end": "2022-06-03"})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "EPA AQS returned no daily Ozone values for the requested monitors for "
                                        "2022-06-01 to 2022-06-03.")

    async def test_an_annual_standard_has_no_exceedance_days(self):
        failure = await self.failure(monitors.exceedance_days, {"pollutant": "pm25", "monitors": ["06-037-1103"],
                                                                "averaging": "annual", "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


class CredentialTests(_MonitorCase):
    async def test_a_missing_connector_fails_before_any_request(self):
        env = await self.make_env(StaticCredentials())

        failure = await self.failure(monitors.find, {"pollutant": "o3", "lat": 34.0, "lon": -118.0,
                                                     "start": "2024-06-01", "end": "2024-06-02"}, env)

        self.assertEqual(failure.category, FailureCategory.CREDENTIAL_INVALID)
        self.assertEqual(failure.reason, "sources.monitors.find needs the epa_aqs connector, which is not connected.")
        self.assertEqual(self.aqs.requests, [])

    async def test_a_rejected_or_malformed_credential_asks_to_reconnect(self):
        for secret in (json.dumps({"email": EMAIL, "key": "wrong-key"}), "not-a-pair"):
            with self.subTest(secret=secret):
                env = await self.make_env(StaticCredentials({"epa_aqs": secret}))

                failure = await self.failure(monitors.readings, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                                 "start": "2024-06-01", "end": "2024-06-02"}, env)

                self.assertEqual(failure.category, FailureCategory.CREDENTIAL_INVALID)
                self.assertEqual(failure.reason, "The epa_aqs credential is invalid. Reconnect it in Settings -> Connectors.")

    async def test_epa_being_down_is_a_temporary_source_failure(self):
        self.aqs.status = 503

        failure = await self.failure(monitors.find, {"pollutant": "o3", "lat": 34.0, "lon": -118.0,
                                                     "start": "2024-06-01", "end": "2024-06-02"})

        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)

    async def test_there_is_no_fallback_credential(self):
        self.assertTrue(all(c.connector == "epa_aqs" for c in monitors.CAPABILITIES))
        env = await self.make_env(StaticCredentials({"earthdata": "edl-token"}))

        await self.failure(monitors.daily_summary, {"pollutant": "o3", "monitors": ["06-037-1103"],
                                                    "start": "2024-06-01", "end": "2024-06-02"}, env)

        self.assertEqual(self.aqs.requests, [])

    async def test_the_secret_never_reaches_params_results_records_or_errors(self):
        area = await self.area()
        found = await invoke(self.env, monitors.find, {"pollutant": "o3", "area": area.id,
                                                       "start": "2024-06-01", "end": "2024-06-02"})
        results = [found]
        for capability in monitors.CAPABILITIES[1:]:
            results.append(await invoke(self.env, capability, {"pollutant": "o3", "points": found.handle,
                                                               "start": "2024-06-01", "end": "2024-06-02"}))
        bad_env = await self.make_env(StaticCredentials({"epa_aqs": json.dumps({"email": EMAIL, "key": "wrong-key"})}))
        failure = await self.failure(monitors.find, {"pollutant": "o3", "lat": 34.0, "lon": -118.0,
                                                     "start": "2024-06-01", "end": "2024-06-02"}, bad_env)

        records = [await self.stores.handles.get("user-a", r.handle) for r in results]
        entries = await self.entries() + await self.entries(bad_env)
        text = json.dumps(default=str, obj=[
            [r.recorded() for r in results],
            [r.model_dump(mode="json") for r in records],
            [dataclasses.asdict(e) for e in entries],
            failure.reason,
            [str(failure.result)],
        ])
        self.assertTrue(all(r.status == "done" for r in results))
        for secret in (EMAIL, KEY, "wrong-key"):
            self.assertNotIn(secret, text)


class DeclarationTests(unittest.TestCase):
    def test_the_group_and_its_capabilities_register(self):
        registry = CapabilityRegistry()
        registry.add_group(monitors.GROUP)
        for capability in monitors.CAPABILITIES:
            registry.register(capability)

        names = [c.name for c in registry.in_group("sources.monitors")]
        self.assertEqual(names, sorted([
            "sources.monitors.find", "sources.monitors.readings", "sources.monitors.daily_summary",
            "sources.monitors.quarterly_summary", "sources.monitors.annual_summary", "sources.monitors.exceedance_days",
        ]))
        self.assertTrue(monitors.GROUP.notes.strip())
        self.assertEqual({c.output_kind for c in monitors.CAPABILITIES}, {"points", "series", "table"})
        self.assertEqual(monitors.find.schema()["properties"]["area"]["x-handle-kinds"], ["polygons"])


class PollutantTests(_MonitorCase):
    async def test_a_pollutant_outside_the_supported_set_is_bad_input_before_any_request(self):
        by_monitor = {"monitors": ["06-037-1103"], "start": "2024-06-01", "end": "2024-06-02"}
        near_a_point = {"lat": 34.07, "lon": -118.22, "start": "2024-06-01", "end": "2024-06-02"}
        for capability, params in (
            (monitors.find, near_a_point),
            (monitors.readings, by_monitor),
            (monitors.daily_summary, by_monitor),
            (monitors.quarterly_summary, by_monitor),
            (monitors.annual_summary, by_monitor),
            (monitors.exceedance_days, by_monitor),
        ):
            for pollutant in ("lead", "co2", "ozone", ""):
                with self.subTest(capability=capability.name, pollutant=pollutant):
                    failure = await self.failure(capability, {"pollutant": pollutant, **params})
                    self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(self.aqs.requests, [])

    def test_every_pollutant_a_capability_accepts_has_its_aqs_definition(self):
        self.assertEqual(set(typing.get_args(Pollutant)), set(POLLUTANTS))
        for capability in monitors.CAPABILITIES:
            with self.subTest(capability=capability.name):
                self.assertEqual(set(capability.schema()["properties"]["pollutant"]["enum"]), set(POLLUTANTS))


if __name__ == "__main__":
    unittest.main()
