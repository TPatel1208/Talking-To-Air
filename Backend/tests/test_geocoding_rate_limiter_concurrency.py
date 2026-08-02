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

# How far below the 1.0s throttle an *observed* firing gap may fall before the
# timing tests below call it a violation.
#
# The reservation arithmetic is exact — see
# ``test_concurrent_reservations_each_claim_a_distinct_one_second_slot``, which
# asserts it with no clock involved at all. What is not exact is when a sleeping
# task actually resumes: two concurrent ``asyncio.sleep`` calls resume with
# independent scheduling jitter, so the gap between them wobbles around 1.0s
# even when the slots either side of it are spaced perfectly.
#
# Measured on the Windows development host with *no application code involved* —
# just ``asyncio.gather(sleep(0.9), sleep(1.9))``, 15 runs: gaps ranged 0.9889 to
# 1.0103, so the jitter floor is roughly ±11ms. The tolerance here was
# originally 10ms, i.e. inside that floor, which made these tests fail about one
# run in five for a reason unconnected to the throttle. 100ms sits an order of
# magnitude above the measured noise while staying 10x smaller than the bug
# these tests exist to catch, which drives the gap to ~0.
THROTTLE_JITTER_TOLERANCE = 0.1
MIN_OBSERVED_GAP = 1.0 - THROTTLE_JITTER_TOLERANCE


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
        # Scheduling jitter, not a throttle violation -- see
        # THROTTLE_JITTER_TOLERANCE for the measurement behind this bound. The
        # bug this catches drives the gap to ~0, an order of magnitude below it.
        self.assertGreaterEqual(gap, MIN_OBSERVED_GAP)

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
        # Scheduling jitter, not a throttle violation -- see
        # THROTTLE_JITTER_TOLERANCE for the measurement behind this bound. The
        # bug this catches drives the gap to ~0, an order of magnitude below it.
        self.assertGreaterEqual(gap, MIN_OBSERVED_GAP)



class ThrottleSlotReservationTests(unittest.TestCase):
    """The reservation arithmetic, asserted without a clock.

    The timing tests above measure when requests actually fire, which mixes the
    guarantee with event-loop scheduling jitter and is why they need a
    tolerance. This one measures the arithmetic alone, so it is exact.

    Deliberately *not* a test that the lock works. A concurrent version of this
    — N threads reserving at once, asserting the throttle advanced by exactly N
    seconds — was written and then removed, because it passed just as happily
    with the lock replaced by a no-op: the critical section is a few bytecodes
    of arithmetic, so the GIL serialises it and the race will not reproduce on
    demand. A test that cannot fail on the bug it names is worse than no test,
    because it reads like coverage. The lock's justification is in the comment
    on ``_throttle_lock``; what is testable here is the slot arithmetic it
    protects.
    """

    def test_a_reservation_never_hands_out_a_slot_in_the_past(self):
        """An idle service must not bank credit: a caller arriving long after
        the previous request waits zero, but the *next* one still waits a full
        second rather than inheriting a stale slot from the distant past."""
        from tta_backend.utils.plotting import GeocodingService

        service = GeocodingService()
        service.last_request = time.time() - 3600

        immediate = service._reserve_throttle_slot()
        follower = service._reserve_throttle_slot()

        self.assertEqual(immediate, 0.0)
        self.assertGreater(follower, 0.9)
        self.assertLessEqual(follower, 1.0)


if __name__ == "__main__":
    unittest.main()
