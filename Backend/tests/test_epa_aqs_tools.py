"""
Regression tests for EPA AQS tool normalisation and monitor-selection logic.

These tests cover the class of bugs where leading zeros are stripped from
EPA site/county/state codes, causing bySite queries to return
"No data matched your selection" even for valid monitors.
"""
import asyncio
import importlib.util
import os
import sys
import unittest
import unittest.mock
from unittest.mock import MagicMock

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_epa_module():
    """Load epa_aqs_tools with all external dependencies stubbed."""
    fake_settings = MagicMock(aqs_api_email="test@test.com", aqs_api_key="testkey")
    fake_artifact_ref = MagicMock(id="ref-id", type="table")
    fake_artifact_ref.model_dump.return_value = {"id": "ref-id", "type": "table"}
    fake_artifact_store = MagicMock()
    fake_artifact_store.put_table.return_value = fake_artifact_ref

    stubs = {
        "langchain": MagicMock(),
        "langchain.tools": MagicMock(tool=lambda f: f),
        "tta_backend.config": MagicMock(),
        "tta_backend.config.settings": MagicMock(get_settings=MagicMock(return_value=fake_settings)),
        "tta_backend.services": MagicMock(),
        "tta_backend.services.artifact_store": MagicMock(artifact_store=fake_artifact_store),
        "tta_backend.utils": MagicMock(),
        "tta_backend.utils.plotting": MagicMock(get_geocoding_service=MagicMock()),
    }

    prev = {k: sys.modules.pop(k, None) for k in stubs}
    # Also evict any previously-cached copy of the module under test.
    cached_key = next(
        (k for k in list(sys.modules) if "epa_aqs_tools" in k), None
    )
    cached_mod = sys.modules.pop(cached_key, None) if cached_key else None

    sys.modules.update(stubs)
    try:
        path = os.path.join(
            BACKEND_DIR, "tta_backend", "tools", "ground_sensor_tools", "epa_aqs_tools.py"
        )
        spec = importlib.util.spec_from_file_location("epa_aqs_tools_isolated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in prev.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if cached_key and cached_mod is not None:
            sys.modules[cached_key] = cached_mod

    return mod


_epa = _load_epa_module()
_normalise_numeric_filter = _epa._normalise_numeric_filter
_normalise_site_filter = _epa._normalise_site_filter
_resolve_filter = _epa._resolve_filter
_aggregate_summary_records = _epa._aggregate_summary_records


class NormaliseNumericFilterTests(unittest.TestCase):
    """Unit tests for _normalise_numeric_filter."""

    def test_valid_string_returned_unchanged_without_min_width(self):
        self.assertEqual(_normalise_numeric_filter("x", "42"), "42")

    def test_integer_input_converted_to_string(self):
        self.assertEqual(_normalise_numeric_filter("x", 42), "42")

    def test_zero_padding_applied_when_min_width_set(self):
        self.assertEqual(_normalise_numeric_filter("site", "52", min_width=4), "0052")

    def test_already_padded_value_unchanged(self):
        self.assertEqual(_normalise_numeric_filter("site", "0052", min_width=4), "0052")

    def test_integer_zero_padded_correctly(self):
        # LLM may emit site_number=52 as a JSON integer.
        self.assertEqual(_normalise_numeric_filter("site", 52, min_width=4), "0052")

    def test_placeholder_raises_value_error(self):
        for placeholder in ("site", "site_id", "n/a", "unknown", "??", ""):
            with self.subTest(placeholder=placeholder):
                with self.assertRaises(ValueError):
                    _normalise_numeric_filter("x", placeholder)

    def test_non_digit_raises_value_error(self):
        with self.assertRaises(ValueError):
            _normalise_numeric_filter("x", "abc")

    def test_whitespace_stripped_before_validation(self):
        self.assertEqual(_normalise_numeric_filter("x", "  42  ", min_width=4), "0042")


class NormaliseSiteFilterTests(unittest.TestCase):
    """
    Regression tests for the leading-zero bug in _normalise_site_filter.

    The EPA AQS API requires zero-padded codes:
      state  → 2 digits   ("5"  → "05")
      county → 3 digits   ("1"  → "001")
      site   → 4 digits   ("52" → "0052")

    The LLM may pass any of these as bare integers or short strings,
    so the normaliser must always pad to the required width.
    """

    def test_integer_site_number_zero_padded(self):
        _, _, site = _normalise_site_filter("35", "001", 52)
        self.assertEqual(site, "0052")

    def test_string_site_without_leading_zeros_padded(self):
        _, _, site = _normalise_site_filter("35", "001", "52")
        self.assertEqual(site, "0052")

    def test_county_code_zero_padded_to_three_digits(self):
        _, county, _ = _normalise_site_filter("35", "1", "0052")
        self.assertEqual(county, "001")

    def test_state_code_zero_padded_to_two_digits(self):
        state, _, _ = _normalise_site_filter("5", "001", "0052")
        self.assertEqual(state, "05")

    def test_all_codes_padded_together(self):
        state, county, site = _normalise_site_filter("5", "1", 7)
        self.assertEqual(state, "05")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0007")

    def test_already_padded_values_unchanged(self):
        state, county, site = _normalise_site_filter("35", "001", "0052")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_split_and_padded(self):
        # station_id format "35-1-52" → split into parts and zero-pad each
        state, county, site = _normalise_site_filter(None, None, "35-1-52")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_already_padded(self):
        state, county, site = _normalise_site_filter(None, None, "35-001-0052")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_wrong_part_count_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter(None, None, "35-001")

    def test_placeholder_site_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter("35", "001", "site_number")

    def test_non_numeric_site_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter("35", "001", "abc")


class ResolveFilterTests(unittest.TestCase):
    """Tests for _resolve_filter zero-padding in byState/byCounty branches."""

    def test_bystate_pads_short_state_code(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", None, None, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/byState")
        self.assertEqual(params["state"], "05")

    def test_bycounty_pads_state_and_county(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", "1", None, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/byCounty")
        self.assertEqual(params["state"], "05")
        self.assertEqual(params["county"], "001")

    def test_bysite_pads_all_three_codes(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", "1", 52, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/bySite")
        self.assertEqual(params["state"], "05")
        self.assertEqual(params["county"], "001")
        self.assertEqual(params["site"], "0052")

    def test_bysite_via_compound_station_id(self):
        endpoint, params = _resolve_filter(
            "dailyData", None, None, "35-1-52", None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/bySite")
        self.assertEqual(params["state"], "35")
        self.assertEqual(params["county"], "001")
        self.assertEqual(params["site"], "0052")

    def test_missing_all_filters_raises(self):
        with self.assertRaises(ValueError):
            _resolve_filter(
                "dailyData", None, None, None, None, None, None, None, None
            )


class AggregateSummaryMinMaxTests(unittest.TestCase):
    """min/max must come from real API statistics, never be fabricated.

    Regression for the ground-monitor table showing min == max == mean for
    every row: the EPA AQS dailyData endpoint returns no minimum_value or
    maximum_value fields, and the aggregator silently substituted the
    arithmetic_mean for both — fabricating stats. dailyData DOES supply
    first_max_value (the highest sample of the day per AQS docs), which is a
    genuine max; there is no per-day minimum at all, so min must be null.
    """

    @staticmethod
    def _daily_record(**overrides):
        # Shaped like a real dailyData response row: note there is NO
        # minimum_value / maximum_value field on this endpoint.
        record = {
            "state_code": "34",
            "county_code": "019",
            "site_number": "0007",
            "date_local": "2024-01-01",
            "arithmetic_mean": 8.970833,
            "first_max_value": 21.3,
            "first_max_hour": 8,
            "units_of_measure": "ppb",
            "sample_duration": "1 HOUR",
            "pollutant_standard": "NO2 1-hour 2010",
            "observation_count": 24,
            "observation_percent": 100,
            "local_site_name": "Downtown",
        }
        record.update(overrides)
        return record

    def test_daily_min_is_null_and_max_uses_first_max_value(self):
        body, _, _ = _aggregate_summary_records([self._daily_record()], "daily")
        row = body[0]
        self.assertEqual(row["mean"], 8.970833)
        self.assertEqual(row["observation_count"], 24)
        # The API supplies no per-day minimum: never echo the mean as min.
        self.assertIsNone(row["min"])
        # first_max_value IS the day's true sample maximum.
        self.assertEqual(row["max"], 21.3)

    def test_min_max_taken_from_api_fields_when_present(self):
        record = self._daily_record(minimum_value=2.1, maximum_value=19.4)
        body, _, _ = _aggregate_summary_records([record], "daily")
        row = body[0]
        self.assertEqual(row["min"], 2.1)
        self.assertEqual(row["max"], 19.4)

    def test_min_and_max_are_null_when_only_a_mean_is_supplied(self):
        record = self._daily_record()
        del record["first_max_value"]
        del record["first_max_hour"]
        body, _, _ = _aggregate_summary_records([record], "daily")
        row = body[0]
        self.assertEqual(row["mean"], 8.970833)
        self.assertIsNone(row["min"])
        self.assertIsNone(row["max"])


class NoDataMessageTests(unittest.IsolatedAsyncioTestCase):
    """Empty-result errors must read as plain language for a researcher.

    Regression for the live 2026-07-16 chat leak: asking for "NO2 in Newark
    yesterday" (a date EPA hadn't published yet) surfaced the internal dump
    "No dailyData data found for param 42602 between 2026-07-15 and
    2026-07-15 using dailyData/bySite with {'state': '34', 'county': '013',
    'site': '0017'}" verbatim. The message must explain EPA's ~2-month
    publication lag instead, and never include endpoint/params internals.
    """

    def test_recent_window_explains_the_publication_lag(self):
        from datetime import date, timedelta

        yesterday = date.today() - timedelta(days=1)
        message = _epa._no_data_message("daily summary", "42602", yesterday, yesterday)

        self.assertIn("has not published", message)
        self.assertIn("two months", message)
        self.assertIn(yesterday.isoformat(), message)

    def test_old_window_reads_as_plain_no_data_guidance(self):
        from datetime import date

        message = _epa._no_data_message(
            "daily summary", "42602", date(2020, 1, 1), date(2020, 1, 31)
        )

        self.assertIn("No EPA daily summary measurements", message)
        self.assertIn("2020-01-01", message)
        self.assertNotIn("has not published", message)

    async def test_fetch_summary_empty_result_never_leaks_endpoint_or_params(self):
        from datetime import date, timedelta
        from unittest.mock import AsyncMock, patch

        yesterday = date.today() - timedelta(days=1)

        with patch.object(_epa, "_aqs_get", AsyncMock(return_value={"Data": []})):
            with self.assertRaises(RuntimeError) as ctx:
                await _epa._fetch_summary(
                    "dailyData", "42602",
                    yesterday, yesterday,
                    yesterday.strftime("%Y%m%d"), yesterday.strftime("%Y%m%d"),
                    "34", "013", "0017",
                    None, None, None, None, None,
                    None, None, None,
                )

        text = str(ctx.exception)
        self.assertNotIn("dailyData/bySite", text)
        self.assertNotIn("{'state'", text)
        self.assertNotIn("param 42602", text)
        self.assertIn("two months", text)

    async def test_fetch_summary_empty_result_for_an_old_range_stays_plain_language(self):
        from datetime import date
        from unittest.mock import AsyncMock, patch

        bdate, edate = date(2020, 1, 1), date(2020, 1, 31)

        with patch.object(_epa, "_aqs_get", AsyncMock(return_value={"Data": []})):
            with self.assertRaises(RuntimeError) as ctx:
                await _epa._fetch_summary(
                    "dailyData", "42602",
                    bdate, edate,
                    bdate.strftime("%Y%m%d"), edate.strftime("%Y%m%d"),
                    "34", "013", "0017",
                    None, None, None, None, None,
                    None, None, None,
                )

        text = str(ctx.exception)
        self.assertNotIn("dailyData/bySite", text)
        self.assertNotIn("{'state'", text)
        self.assertIn("daily summary", text)


# ---------------------------------------------------------------------------
# Request cache + EPA rate limiting
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.url = "https://aqs.epa.gov/data/api/fake"

    def json(self):
        return self._payload


class _RecordingHttp:
    """Stands in for httpx.AsyncClient, recording every outbound EPA call."""

    def __init__(self, payload=None):
        self.calls = []
        self.payload = payload or {"Header": [{"status": "Success"}], "Data": [{"row": 1}]}

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return _FakeResponse(self.payload)


class AqsRequestCacheTests(unittest.IsolatedAsyncioTestCase):
    """The AQS request cache exists to keep EPA calls off the wire.

    EPA disables accounts that exceed 10 requests/minute without notice, and
    the credential is shared app-wide, so every avoidable duplicate matters.
    """

    def setUp(self):
        _epa._reset_aqs_request_state()

    def tearDown(self):
        _epa._reset_aqs_request_state()

    async def test_identical_requests_from_separate_tasks_fetch_once(self):
        """Regression: the original ContextVar cache was written inside the
        task asyncio.gather creates for each tool call, so it never
        propagated back and every call re-fetched. Live logs showed identical
        monitors/byBox params hitting EPA twice in a single turn."""
        http = _RecordingHttp()
        params = {"param": "88101", "bdate": "20200101", "edate": "20200107"}

        with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
            await asyncio.gather(_epa._aqs_get("dailyData/bySite", dict(params)))
            await asyncio.gather(_epa._aqs_get("dailyData/bySite", dict(params)))

        self.assertEqual(len(http.calls), 1)

    async def test_different_params_are_fetched_separately(self):
        http = _RecordingHttp()

        with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
            await _epa._aqs_get("dailyData/bySite", {"param": "88101", "edate": "20200107"})
            await _epa._aqs_get("dailyData/bySite", {"param": "42602", "edate": "20200107"})

        self.assertEqual(len(http.calls), 2)

    async def test_a_second_credential_reuses_the_first_researchers_result(self):
        """AQS measurements are public: the same query returns the same bytes
        whoever's key fetched it. Keying the cache on the credential would
        make every researcher re-pay a rate-limit slot for identical data,
        and would silently undo this cache once per-user keys land."""
        http = _RecordingHttp()
        params = {"param": "88101", "bdate": "20200101", "edate": "20200107"}

        with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
            with unittest.mock.patch.object(_epa, "AQS_EMAIL", "first@example.com"):
                await _epa._aqs_get("dailyData/bySite", dict(params))
            with unittest.mock.patch.object(_epa, "AQS_EMAIL", "second@example.com"):
                await _epa._aqs_get("dailyData/bySite", dict(params))

        self.assertEqual(len(http.calls), 1)

    async def test_an_unpublished_range_is_refetched_once_its_entry_goes_stale(self):
        """A range EPA has not published yet returns empty now and real rows
        in a couple of months. Caching that emptiness on the same terms as
        settled data would keep answering "no data" long after the
        measurements landed."""
        from datetime import date

        http = _RecordingHttp()
        today = date.today().strftime("%Y%m%d")
        clock = [1000.0]

        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
                await _epa._aqs_get("dailyData/bySite", {"param": "88101", "edate": today})
                clock[0] += 3600.0
                await _epa._aqs_get("dailyData/bySite", {"param": "88101", "edate": today})

        self.assertEqual(len(http.calls), 2)

    async def test_settled_measurements_stay_cached_across_turns(self):
        """Anything outside EPA's publication lag is final, so a later turn
        — or a different researcher — must not spend a rate-limit slot
        re-fetching it."""
        http = _RecordingHttp()
        clock = [1000.0]

        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
                await _epa._aqs_get("dailyData/bySite", {"param": "88101", "edate": "20200107"})
                clock[0] += 6 * 60 * 60
                await _epa._aqs_get("dailyData/bySite", {"param": "88101", "edate": "20200107"})

        self.assertEqual(len(http.calls), 1)

    async def test_cache_hits_do_not_spend_the_epa_request_budget(self):
        """The cache exists to protect the rate limit, so a served-from-cache
        answer must not reserve a slot. If it did, twenty repeats of one
        question would exhaust the minute as surely as twenty real calls."""
        http = _RecordingHttp()
        params = {"param": "88101", "bdate": "20200101", "edate": "20200107"}
        clock = [1000.0]

        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
                for _ in range(20):
                    await _epa._aqs_get("dailyData/bySite", dict(params))
            clock[0] += 2.0
            wait_for_a_fresh_request = _epa._limiter_for(_epa.AQS_KEY).reserve()

        self.assertEqual(len(http.calls), 1)
        self.assertEqual(wait_for_a_fresh_request, 0.0)

    async def test_the_cache_does_not_grow_without_bound(self):
        """Entries live for days and AQS payloads are whole result sets, so
        an unevicted entry per distinct query is a slow leak in a process
        that already has a history of memory pressure."""
        http = _RecordingHttp()

        # These are all cache misses, so the limiter legitimately paces them.
        # This test is about the cache's ceiling, not pacing, so the waiting
        # is stubbed out rather than actually served.
        with unittest.mock.patch.object(_epa.asyncio, "sleep", unittest.mock.AsyncMock()):
            with unittest.mock.patch.object(_epa.httpx, "AsyncClient", http):
                for i in range(_epa._MAX_CACHE_ENTRIES + 25):
                    await _epa._aqs_get("dailyData/bySite", {"param": "88101", "site": str(i)})

        self.assertLessEqual(len(_epa._response_cache), _epa._MAX_CACHE_ENTRIES)


class AqsRateLimitTests(unittest.TestCase):
    """EPA allows 10 requests/minute per credential and disables accounts that
    exceed it without notice, so pacing is mandatory. It must not, however,
    tax the ordinary case: one researcher asking one question makes only a
    handful of calls, spaced by the agent's own thinking time.
    """

    def setUp(self):
        _epa._reset_aqs_request_state()

    def tearDown(self):
        _epa._reset_aqs_request_state()

    def test_a_single_researchers_calls_are_not_delayed(self):
        clock = [1000.0]
        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            limiter = _epa._limiter_for("shared-key")
            waits = []
            for _ in range(4):
                waits.append(limiter.reserve())
                clock[0] += 2.0  # the agent's own model latency between calls

        self.assertEqual(waits, [0.0, 0.0, 0.0, 0.0])

    def test_the_request_past_the_cap_waits_for_a_slot_to_free(self):
        """Nine simultaneous requests fill EPA's minute; the tenth cannot go
        out until the first has aged out of the window."""
        clock = [1000.0]
        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            limiter = _epa._limiter_for("shared-key")
            waits = [limiter.reserve() for _ in range(10)]

        within_cap = waits[: _epa._AQS_MAX_REQUESTS_PER_WINDOW]
        past_cap = waits[_epa._AQS_MAX_REQUESTS_PER_WINDOW]

        self.assertTrue(all(w < _epa._AQS_WINDOW_SECONDS for w in within_cap), within_cap)
        self.assertEqual(past_cap, _epa._AQS_WINDOW_SECONDS)

    def test_slots_free_again_once_the_window_slides_past_them(self):
        clock = [1000.0]
        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            limiter = _epa._limiter_for("shared-key")
            for _ in range(_epa._AQS_MAX_REQUESTS_PER_WINDOW):
                limiter.reserve()
            clock[0] += _epa._AQS_WINDOW_SECONDS + 1.0
            wait_after_window = limiter.reserve()

        self.assertEqual(wait_after_window, 0.0)

    def test_one_credentials_exhausted_window_does_not_delay_another(self):
        """EPA's cap is per credential. If a researcher brings their own key,
        it has to buy them their own budget — queueing them behind everyone
        on the shared key would make bringing a key pointless."""
        clock = [1000.0]
        with unittest.mock.patch.object(_epa, "_now", lambda: clock[0]):
            shared = _epa._limiter_for("shared-key")
            for _ in range(_epa._AQS_MAX_REQUESTS_PER_WINDOW + 1):
                shared.reserve()
            own_key_wait = _epa._limiter_for("researchers-own-key").reserve()

        self.assertEqual(own_key_wait, 0.0)


class SummaryFailureClassificationTests(unittest.IsolatedAsyncioTestCase):
    """"EPA has no rows" and "EPA did not answer" must not share a status.

    The summary tools return a structured dict instead of raising, because a
    raised RuntimeError escapes ToolNode and kills the whole ground turn. But
    _aqs_get raises RuntimeError for transport and API failures too, so a
    catch-all handler reported an AQS outage or a rejected credential as
    ``no_data`` -- and the agent, reading a successful empty result, told the
    researcher no monitoring data exists. That is a scientific claim the
    service outage is no evidence for.
    """

    async def _daily_summary_header(self, error):
        from unittest.mock import AsyncMock, patch

        with patch.object(_epa, "_fetch_summary", AsyncMock(side_effect=error)):
            result = await _epa.get_daily_summary(
                param_code="42602", bdate="20200101", edate="20200107", state_code="34",
            )
        return result["Header"][0]

    async def test_an_empty_result_set_is_reported_as_no_data(self):
        header = await self._daily_summary_header(
            _epa.AqsNoDataError("No EPA daily summary measurements for 2020-01-01 to 2020-01-07.")
        )

        self.assertEqual(header["status"], "no_data")
        self.assertIn("No EPA daily summary measurements", header["note"])

    async def test_an_upstream_failure_is_not_reported_as_no_data(self):
        """The exact shape _aqs_get raises for an HTTP error. Reported as
        no_data, this became "there are no NO2 measurements there" in the
        answer -- indistinguishable from a real empty result."""
        header = await self._daily_summary_header(
            RuntimeError("AQS HTTP 503 on dailyData/byState: service unavailable")
        )

        self.assertNotEqual(header["status"], "no_data")
        self.assertEqual(header["status"], "upstream_error")
        self.assertIn("503", header["note"])

    async def test_a_rejected_credential_is_not_reported_as_no_data(self):
        header = await self._daily_summary_header(
            RuntimeError("EPA AQS request failed: invalid email/key combination")
        )

        self.assertEqual(header["status"], "upstream_error")

    async def test_the_no_data_type_still_satisfies_broad_runtime_error_handlers(self):
        """validation_tools' ground fetches catch RuntimeError to fall back on
        either kind, so the new type must stay a subclass rather than becoming
        a sibling."""
        self.assertTrue(issubclass(_epa.AqsNoDataError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
