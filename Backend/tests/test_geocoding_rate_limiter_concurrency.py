"""T45: GeocodingService.last_request is read-modify-write from both the
sync (geocode) and async (ageocode) paths with no coordination. Two parallel
calls can both read the same stale last_request, both compute "no throttle
needed", and both fire within Nominatim's 1 rps window at once — violating
its usage policy, which 403s on abuse.
"""
import asyncio
import importlib.util
import time
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio"]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_payload(name: str) -> list[dict]:
    return [{
        "lat": "1.0",
        "lon": "2.0",
        "display_name": name,
        "geojson": None,
        "boundingbox": ["0", "1", "0", "1"],
    }]


class _TimingFakeAsyncClient:
    call_times: list[float] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        type(self).call_times.append(time.monotonic())
        return _FakeResponse(_fake_payload(params["q"]))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "geocoding rate-limiter test dependencies are not installed",
)
class GeocodingRateLimiterConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_concurrent_ageocode_calls_are_throttled_a_full_second_apart(self):
        from tta_backend.utils.plotting import GeocodingService

        service = GeocodingService()
        # Prime last_request as if a request "just" fired: both concurrent
        # calls below then take the sleep-then-fire branch together. Without
        # coordination, both compute the same wait, both wake at nearly the
        # same instant, and neither re-checks last_request before firing —
        # the actual failure mode (parallel calls can both pass the 1 rps
        # check).
        service.last_request = time.time() - 0.1
        _TimingFakeAsyncClient.call_times = []

        with patch("tta_backend.utils.plotting.httpx.AsyncClient", _TimingFakeAsyncClient):
            await asyncio.gather(
                service.ageocode("T45 Concurrency Test Locale A"),
                service.ageocode("T45 Concurrency Test Locale B"),
            )

        self.assertEqual(len(_TimingFakeAsyncClient.call_times), 2)
        gap = abs(_TimingFakeAsyncClient.call_times[1] - _TimingFakeAsyncClient.call_times[0])
        # asyncio.sleep() can return a fraction of a millisecond early
        # (timer/scheduling jitter, not a throttle violation) -- a small
        # tolerance avoids flaking on that while still catching the actual
        # bug (both calls firing together, gap ~= 0).
        self.assertGreaterEqual(gap, 0.99)

    async def test_a_concurrent_sync_and_async_call_share_one_throttle_timestamp(self):
        """A sync geocode() call (run in a worker thread, as export_service
        does) racing a concurrent ageocode() call must still observe the
        1 rps gap — the two paths coordinate through the same last_request
        bookkeeping, not two independent clocks that can both fire at once."""
        import requests

        from tta_backend.utils.plotting import GeocodingService

        service = GeocodingService()
        # Same priming as the async/async test: forces both paths through
        # the sleep-then-fire branch so their ~0.9s in-flight windows
        # genuinely overlap, instead of relying on unpredictable thread
        # scheduling to interleave two near-instant fake calls.
        service.last_request = time.time() - 0.1
        sync_call_times: list[float] = []

        class _FakeSyncResponse:
            def json(self):
                return _fake_payload("T45 Sync Locale")

        def fake_requests_get(url, params=None, headers=None, timeout=None):
            sync_call_times.append(time.monotonic())
            return _FakeSyncResponse()

        _TimingFakeAsyncClient.call_times = []

        with patch.object(requests, "get", fake_requests_get), \
             patch("tta_backend.utils.plotting.httpx.AsyncClient", _TimingFakeAsyncClient):
            await asyncio.gather(
                asyncio.to_thread(service.geocode, "T45 Sync Locale"),
                service.ageocode("T45 Concurrency Test Locale C"),
            )

        self.assertEqual(len(sync_call_times), 1)
        self.assertEqual(len(_TimingFakeAsyncClient.call_times), 1)
        gap = abs(_TimingFakeAsyncClient.call_times[0] - sync_call_times[0])
        # asyncio.sleep() can return a fraction of a millisecond early
        # (timer/scheduling jitter, not a throttle violation) -- a small
        # tolerance avoids flaking on that while still catching the actual
        # bug (both calls firing together, gap ~= 0).
        self.assertGreaterEqual(gap, 0.99)


if __name__ == "__main__":
    unittest.main()
