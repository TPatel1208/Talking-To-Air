"""The EPA AQS domain client and calculations, against a fake HTTP transport."""
import datetime
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tta_backend.datasets.standards import load_standards
from tta_backend.domain.monitors import aqs, parsing, queries
from tta_backend.domain.monitors.pollutants import POLLUTANTS, convert

CREDENTIAL = aqs.AqsCredential(email="me@lab.edu", key="my-key")
SETTLED = {"param": "88101", "bdate": "20200101", "edate": "20200107"}


class FakeAqs:
    """An httpx MockTransport answering like EPA: rows, a rejected pair, or an error status."""

    def __init__(self, rows=None, *, status=200, header=None, rejected_keys=()):
        self.rows = rows if rows is not None else [{"row": 1}]
        self.status = status
        self.header = header
        self.rejected_keys = set(rejected_keys)
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.params.get("key") in self.rejected_keys:
            return httpx.Response(400, json={"Header": [{"status": "Failed", "url": str(request.url),
                                                         "error": ["Email and/or key are invalid."]}]})
        if self.status >= 400:
            return httpx.Response(self.status, json={"Header": [{"status": "Failed", "url": str(request.url),
                                                                 "error": ["Service down for my-key"]}]})
        header = self.header or {"status": "Success"}
        return httpx.Response(200, json={"Header": [header], "Data": self.rows})


class _Reset(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        aqs.reset_request_state()
        self.addCleanup(aqs.reset_request_state)
        sleep = patch.object(aqs.asyncio, "sleep", AsyncMock())
        sleep.start()
        self.addCleanup(sleep.stop)


class RequestTests(_Reset):
    async def test_a_request_carries_the_given_credential_and_returns_the_body(self):
        fake = FakeAqs()

        data = await aqs.request("dailyData/bySite", dict(SETTLED), CREDENTIAL, transport=fake.transport)

        sent = fake.requests[0].url
        self.assertEqual(sent.path, "/data/api/dailyData/bySite")
        self.assertEqual((sent.params["email"], sent.params["key"], sent.params["param"]), ("me@lab.edu", "my-key", "88101"))
        self.assertEqual(data["Data"], [{"row": 1}])

    async def test_a_rejected_pair_raises_the_credential_category(self):
        fake = FakeAqs(rejected_keys={"my-key"})

        with self.assertRaises(aqs.AqsCredentialsRejected) as caught:
            await aqs.request("list/states", {}, CREDENTIAL, transport=fake.transport)

        self.assertEqual(caught.exception.category, "token_invalid")

    async def test_a_server_error_is_temporary_and_its_text_is_scrubbed_of_the_key(self):
        fake = FakeAqs(status=503)

        with self.assertRaises(aqs.AqsRequestFailed) as caught:
            await aqs.request("list/states", {}, CREDENTIAL, transport=fake.transport)

        self.assertEqual(caught.exception.category, "provider_unavailable")
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertNotIn("my-key", str(caught.exception))

    async def test_a_failed_header_is_not_mistaken_for_an_empty_result(self):
        fake = FakeAqs(header={"status": "Failed", "error": ["bad param"]})

        with self.assertRaises(aqs.AqsRequestFailed) as caught:
            await aqs.request("list/states", {}, CREDENTIAL, transport=fake.transport)

        self.assertEqual(caught.exception.category, "contract")

    async def test_no_data_matched_is_an_empty_result(self):
        fake = FakeAqs(rows=[], header={"status": "No data matched your selection"})

        data = await aqs.request("list/states", {}, CREDENTIAL, transport=fake.transport)

        self.assertEqual(data["Data"], [])

    async def test_each_credential_paces_on_its_own_limiter(self):
        fake = FakeAqs()

        await aqs.request("list/states", {}, CREDENTIAL, transport=fake.transport)
        await aqs.request("list/states", {}, aqs.AqsCredential("b@lab.edu", "other-key"), transport=fake.transport)

        self.assertEqual(set(aqs._rate_limiters), {aqs.credential_digest("my-key"), aqs.credential_digest("other-key")})
        self.assertNotIn("my-key", json.dumps(list(aqs._rate_limiters)))

    def test_a_credential_repr_hides_the_key(self):
        self.assertNotIn("my-key", repr(CREDENTIAL))
        self.assertIn("me@lab.edu", repr(CREDENTIAL))

    def test_the_limiter_map_is_bounded_and_keeps_live_limiters_apart(self):
        clock = [1000.0]
        with patch.object(aqs, "_now", lambda: clock[0]), patch.object(aqs, "_MAX_RATE_LIMITERS", 3):
            busy = aqs._limiter_for("busy-key")
            busy.reserve()
            clock[0] += 61.0
            for key in ("a", "b"):
                aqs._limiter_for(key).reserve()
            busy_again = aqs._limiter_for("busy-key")
            busy_again.reserve()
            # Full: "a" and "b" are still pacing, "busy-key" was used last; "a" is the least recent.
            aqs._limiter_for("c")
            self.assertEqual(len(aqs._rate_limiters), 3)
            self.assertIs(aqs._limiter_for("busy-key"), busy_again)
            self.assertIsNot(aqs._limiter_for("b"), aqs._limiter_for("c"))
            clock[0] += 61.0
            # Every limiter is idle, so making room drops them all and no pacing state is lost.
            aqs._limiter_for("d")
            self.assertEqual(set(aqs._rate_limiters), {aqs.credential_digest("d")})

    async def test_validation_reports_an_unreachable_epa_as_unavailable(self):
        def refuse(request):
            raise httpx.ConnectError("boom")

        with self.assertRaises(aqs.AqsUnavailable):
            await aqs.validate_credentials(CREDENTIAL, transport=httpx.MockTransport(refuse))


class FetcherTests(_Reset):
    async def test_a_repeated_request_is_served_from_the_cache(self):
        fake = FakeAqs()
        fetch = aqs.fetcher(CREDENTIAL, transport=fake.transport)

        first = await fetch("dailyData/bySite", dict(SETTLED))
        second = await fetch("dailyData/bySite", dict(SETTLED))

        self.assertEqual(len(fake.requests), 1)
        self.assertIs(first, second)

    async def test_the_cache_is_shared_across_credentials(self):
        fake = FakeAqs()

        await aqs.fetcher(CREDENTIAL, transport=fake.transport)("dailyData/bySite", dict(SETTLED))
        await aqs.fetcher(aqs.AqsCredential("b@lab.edu", "k2"), transport=fake.transport)("dailyData/bySite", dict(SETTLED))

        self.assertEqual(len(fake.requests), 1)

    async def test_a_transport_failure_is_reported_as_unavailable(self):
        def refuse(request):
            raise httpx.ConnectError("boom")

        with self.assertRaises(aqs.AqsUnavailable) as caught:
            await aqs.fetcher(CREDENTIAL, transport=httpx.MockTransport(refuse))("list/states", {})

        self.assertEqual(caught.exception.category, "provider_unavailable")


class CredentialFromSecretTests(unittest.TestCase):
    def test_the_connector_secret_is_an_email_and_key_pair(self):
        credential = aqs.credential_from_secret(json.dumps({"email": "me@lab.edu", "key": "my-key"}))

        self.assertEqual(credential, CREDENTIAL)

    def test_a_malformed_secret_is_a_rejected_credential_without_echoing_it(self):
        for secret in ("not json", json.dumps({"email": "me@lab.edu"}), json.dumps({"email": "", "key": "k"}), "[]"):
            with self.subTest(secret=secret):
                with self.assertRaises(aqs.AqsCredentialsRejected) as caught:
                    aqs.credential_from_secret(secret)
                self.assertNotIn(secret, str(caught.exception))


class QueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_monitor_search_widens_until_enough_are_found(self):
        boxes = []

        async def fetch(endpoint, params):
            boxes.append(params["maxlat"] - params["minlat"])
            return {"Data": [{"id": i} for i in range(len(boxes))]}

        found = await queries.fetch_active_monitors(fetch, [40.0, 41.0, -75.0, -74.0], "44201", "20240101", "20240131", k=3)

        self.assertEqual(len(found), 3)
        self.assertEqual(boxes, [1.0, 2.0, 4.0])

    async def test_a_summary_keeps_one_standard_and_an_empty_one_raises_no_data(self):
        rows = [{"pollutant_standard": "Ozone 8-hour 2015", "v": 1}, {"pollutant_standard": "Ozone 1-hour 1979", "v": 2}]

        async def fetch(endpoint, params):
            self.assertEqual(endpoint, "dailyData/bySite")
            self.assertEqual((params["state"], params["county"], params["site"]), ("06", "037", "1103"))
            return {"Data": rows}

        kept, endpoint, _ = await queries.fetch_summary(
            fetch, "dailyData", "44201", datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), "20200101", "20200102",
            None, None, "06-037-1103", None, None, None, None, None, None, None, "Ozone 8-hour 2015",
        )
        self.assertEqual(kept, [rows[0]])
        with self.assertRaises(aqs.AqsNoDataError) as caught:
            await queries.fetch_summary(
                fetch, "dailyData", "44201", datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), "20200101", "20200102",
                None, None, "06-037-1103", None, None, None, None, None, None, None, "CO 8-hour 1971",
            )
        self.assertEqual(caught.exception.category, "no_data")
        self.assertNotIn("bySite", str(caught.exception))


class ParsingTests(unittest.TestCase):
    def test_periods_are_cut_at_calendar_years(self):
        spans = parsing.year_spans(datetime.date(2023, 11, 1), datetime.date(2025, 2, 1))

        self.assertEqual(spans, [
            (datetime.date(2023, 11, 1), datetime.date(2023, 12, 31)),
            (datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)),
            (datetime.date(2025, 1, 1), datetime.date(2025, 2, 1)),
        ])

    def test_recency_follows_the_given_day(self):
        today = datetime.date(2026, 1, 1)

        self.assertTrue(parsing.is_recent(datetime.date(2025, 12, 1), today=today))
        self.assertFalse(parsing.is_recent(datetime.date(2025, 6, 1), today=today))
        self.assertIn("has not published", parsing.no_data_message("daily summary", "1", today, today, today=today))

    def test_nearest_monitors_come_first(self):
        monitors = [{"latitude": 41.0, "longitude": -74.0}, {"latitude": 40.01, "longitude": -74.0}]

        nearest = parsing.nearest_k(monitors, 40.0, -74.0, 1)

        self.assertEqual(nearest[0]["latitude"], 40.01)
        self.assertAlmostEqual(nearest[0]["_dist"], 0.69, places=2)

    def test_exceedances_are_strictly_above_the_threshold_and_sorted_by_date(self):
        records = [
            {"date_local": "2024-06-02", "first_max_value": 0.071, "state_code": "06", "county_code": "037", "site_number": "1103"},
            {"date_local": "2024-06-01", "first_max_value": 0.080, "state_code": "06", "county_code": "037", "site_number": "1103"},
            {"date_local": "2024-06-03", "first_max_value": 0.070, "state_code": "06", "county_code": "037", "site_number": "1103"},
        ]

        body, cutoff = parsing.flag_exceedances(records, "first_max_value", 0.070, None)

        self.assertEqual([row["date"] for row in body], ["2024-06-01", "2024-06-02"])
        self.assertEqual(body[0]["site_id"], "06-037-1103")
        self.assertIsNone(cutoff)

    def test_sample_rows_are_sorted_by_site_and_time(self):
        rows = parsing.sample_rows([
            {"state_code": "06", "county_code": "037", "site_number": "2", "date_local": "2024-06-01", "time_local": "05:00", "sample_measurement": 3},
            {"state_code": "06", "county_code": "037", "site_number": "1", "date_local": "2024-06-01", "time_local": "07:00", "sample_measurement": 2},
        ])

        self.assertEqual([(r["site_id"], r["hour"]) for r in rows], [("06-037-1", 7), ("06-037-2", 5)])

    def test_aqs_units_map_to_standard_unit_codes(self):
        self.assertEqual(parsing.unit_code("Parts per million"), "ppm")
        self.assertEqual(parsing.unit_code("Micrograms/cubic meter (LC)"), "ug/m3")
        self.assertEqual(parsing.unit_code("Degrees Fahrenheit"), "Degrees Fahrenheit")
        self.assertIsNone(parsing.unit_code(None))


class PollutantTests(unittest.TestCase):
    def test_every_naaqs_in_the_standards_file_has_an_aqs_standard(self):
        standards = load_standards()
        self.assertEqual(set(POLLUTANTS), set(standards))
        for key, pollutant in POLLUTANTS.items():
            with self.subTest(pollutant=key):
                periods = {entry.averaging_period for entry in standards[key].naaqs}
                self.assertEqual({s.averaging_period for s in pollutant.standards}, periods)

    def test_the_short_term_standard_is_the_default(self):
        self.assertEqual(POLLUTANTS["pm25"].standard().averaging_period, "24-hour")
        self.assertEqual(POLLUTANTS["o3"].naaqs("8-hour").level, 0.070)

    def test_only_mixing_ratios_convert(self):
        self.assertEqual(convert(0.07, "ppm", "ppb"), 70.0)
        self.assertEqual(convert(70.0, "ppb", "ppm"), 0.07)
        self.assertIsNone(convert(1.0, "ug/m3", "ppb"))


if __name__ == "__main__":
    unittest.main()
