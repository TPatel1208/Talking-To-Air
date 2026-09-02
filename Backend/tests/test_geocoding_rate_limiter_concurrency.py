"""T45: GeocodingService.last_request is read-modify-write from both the
sync (geocode) and async (ageocode) paths with no coordination. Two parallel
calls can both read the same stale last_request, both compute "no throttle
needed", and both fire within Nominatim's 1 rps window at once — violating
its usage policy, which 403s on abuse.
"""
import asyncio
import contextlib
import importlib.util
import threading
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio"]

# The guarantee under test: consecutive Nominatim requests are at least this
# far apart, however many callers race for the throttle.
MIN_THROTTLE_GAP = 1.0

# Every instant these tests compare is *computed* from the virtual clock below
# rather than read off a wall clock, so the only inexactness left is IEEE-754
# double representation (~1e-10 at the clock's magnitude). This slop sits four
# orders above that, and six orders smaller than the ~1.0s shortfall the bug
# produces, so it cannot launder a real throttle violation.
#
# It is deliberately not a scheduling-jitter allowance. This file used to carry
# one of those -- 100ms, calibrated over 15 runs of a bare
# ``asyncio.gather(sleep(0.9), sleep(1.9))`` on an idle host, where the jitter
# floor was +/-11ms. It still failed (observed gap 0.797s, ~18x that floor)
# during a 22-minute suite on a host that was swapping, because a tolerance
# measured on an idle machine does not describe a loaded one -- and widening it
# again would only move the next failure while weakening a real concurrency
# guard. The fix was to stop measuring the scheduler: the throttle is
# arithmetic, so assert the arithmetic.
FLOAT_SLOP = 1e-6


class _ModuleShim:
    """A stand-in for a module, with some attributes overridden.

    Patching ``plotting.time``/``plotting.asyncio`` with one of these confines
    the fake clock to the module under test, instead of reaching into the real
    ``time``/``asyncio``/``httpx`` modules, where every other test would see it.
    """

    def __init__(self, real, **overrides):
        self._real = real
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _VirtualClock:
    """Wall time replaced by arithmetic.

    Time advances only when someone sleeps, and only for them: ``sleep``/
    ``asleep`` never actually wait, they just move the calling task's (or
    thread's) own virtual clock forward, and ``asleep`` yields once to the
    event loop so concurrent callers still interleave the way they do in
    production. ``_reserve_throttle_slot``'s arithmetic therefore comes out
    identical on every run.

    A request's *firing instant* is its caller's virtual time when it issues:
    the moment that request would reach Nominatim, which is precisely what the
    1 rps policy constrains. Because it is derived from the waits the code
    chose, rather than sampled whenever the loop got round to resuming a
    sleeping task, it carries no scheduling jitter -- and a test that spends no
    real time asleep also stops costing the suite ~3s per case.
    """

    def __init__(self, now: float = 1_000_000.0):
        self.now = now
        self._slept: dict[object, float] = {}
        self.fired_at: list[tuple[str, float]] = []

    @staticmethod
    def _caller():
        """Whoever is sleeping: the running task, or the thread when there is
        no loop -- geocode() is sync and runs in a worker thread."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return task if task is not None else threading.current_thread()

    def time(self) -> float:
        """Now, as the *calling* caller sees it: the clock's start plus
        whatever this caller has already slept.

        Per-caller rather than one frozen global instant, because the code
        under test reads this same clock. A caller that sleeps and then reads
        the time again must see its own sleep, or the harness and the code
        disagree about when that caller is -- which shows up as fictitious
        firing instants the moment any caller makes two throttled calls, in
        the direction that widens gaps and so hides violations.
        """
        return self.now + self._slept.get(self._caller(), 0.0)

    def sleep(self, duration: float) -> None:
        """Stands in for ``time.sleep`` on the sync path."""
        self._record_sleep(duration)

    async def asleep(self, duration: float) -> None:
        """Stands in for ``asyncio.sleep`` on the async path."""
        self._record_sleep(duration)
        # Yield, so a concurrent caller gets to run -- but do not wait.
        await asyncio.sleep(0)

    def _record_sleep(self, duration: float) -> None:
        caller = self._caller()
        self._slept[caller] = self._slept.get(caller, 0.0) + duration

    def record_fire(self, label: str) -> None:
        """A request reaches the network at its caller's current virtual time."""
        self.fired_at.append((label, self.time()))

    def observed_gap(self) -> float:
        """Interval between the two recorded firings, whichever fired first."""
        (_, first), (_, second) = self.fired_at
        return abs(second - first)


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


def _timing_client(clock: _VirtualClock):
    """An ``httpx.AsyncClient`` stand-in that stamps each request with the
    virtual instant it would have reached the network."""

    class _TimingFakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None, headers=None):
            clock.record_fire(params["q"])
            return _FakeResponse(_fake_payload(params["q"]))

    return _TimingFakeAsyncClient


@contextlib.contextmanager
def _virtual_time(clock: _VirtualClock, sync_get=None):
    """Run plotting's geocoding paths against ``clock`` instead of wall time,
    with their HTTP seams faked so each request records when it fired."""
    from tta_backend.utils import plotting

    patches = [
        patch.object(plotting, "time",
                     _ModuleShim(plotting.time, time=clock.time, sleep=clock.sleep)),
        patch.object(plotting, "asyncio",
                     _ModuleShim(plotting.asyncio, sleep=clock.asleep)),
        patch.object(plotting, "httpx",
                     _ModuleShim(plotting.httpx, AsyncClient=_timing_client(clock))),
    ]
    if sync_get is not None:
        patches.append(patch.object(plotting, "requests",
                                    _ModuleShim(plotting.requests, get=sync_get)))
    with contextlib.ExitStack() as stack:
        for cm in patches:
            stack.enter_context(cm)
        yield


@contextlib.contextmanager
def _counting_reservations(service, reservations: list[float]):
    """Count trips through the one shared reservation point.

    The gap assertions alone cannot tell a whole fix from a half one: if only
    one of the two paths keeps its own read-modify-write, whether the gap still
    lands at 1.0s depends on which path happens to run first. Requiring that
    *both* callers reserved pins the contract the module docstring states --
    one shared timestamp, not two independent clocks -- without a race.
    """
    reserve = service._reserve_throttle_slot

    def counted():
        wait = reserve()
        reservations.append(wait)
        return wait

    with patch.object(service, "_reserve_throttle_slot", counted):
        yield


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "geocoding rate-limiter test dependencies are not installed",
)
class GeocodingRateLimiterConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """When the real geocode()/ageocode() paths race, the requests they issue
    land in distinct 1-second slots.

    ``ThrottleSlotReservationTests`` below pins the same arithmetic by calling
    ``_reserve_throttle_slot`` directly, and passes untouched when a caller
    stops reserving or stops waiting for its slot -- which is exactly the bug
    named in the module docstring. These two tests are what covers the wiring.
    """

    async def test_two_concurrent_ageocode_calls_are_throttled_a_full_second_apart(self):
        from tta_backend.utils.plotting import GeocodingService

        clock = _VirtualClock()
        service = GeocodingService()
        # Prime last_request as if a request "just" fired: both concurrent
        # calls below then take the sleep-then-fire branch together. Without
        # coordination, both compute the same wait, both aim at the same
        # instant, and neither re-checks last_request before firing -- the
        # actual failure mode (parallel calls can both pass the 1 rps check).
        service.last_request = clock.time() - 0.1
        reservations: list[float] = []

        with _virtual_time(clock), _counting_reservations(service, reservations):
            await asyncio.gather(
                service.ageocode("T45 Concurrency Test Locale A"),
                service.ageocode("T45 Concurrency Test Locale B"),
            )

        self.assertEqual(
            sorted(label for label, _ in clock.fired_at),
            ["T45 Concurrency Test Locale A", "T45 Concurrency Test Locale B"],
        )
        self.assertGreaterEqual(clock.observed_gap(), MIN_THROTTLE_GAP - FLOAT_SLOP)
        self.assertEqual(len(reservations), 2)

    async def test_a_concurrent_sync_and_async_call_share_one_throttle_timestamp(self):
        """A sync geocode() call (run in a worker thread, as export_service
        does) racing a concurrent ageocode() call must still observe the
        1 rps gap -- the two paths coordinate through the same last_request
        bookkeeping, not two independent clocks that can both fire at once."""
        from tta_backend.utils.plotting import GeocodingService

        clock = _VirtualClock()
        service = GeocodingService()
        # Same priming as the async/async test: forces both paths through the
        # sleep-then-fire branch, so each one has a wait to get wrong.
        service.last_request = clock.time() - 0.1
        reservations: list[float] = []

        class _FakeSyncResponse:
            def json(self):
                return _fake_payload("T45 Sync Locale")

        def fake_requests_get(url, params=None, headers=None, timeout=None):
            clock.record_fire(params["q"])
            return _FakeSyncResponse()

        with _virtual_time(clock, sync_get=fake_requests_get), \
                _counting_reservations(service, reservations):
            await asyncio.gather(
                asyncio.to_thread(service.geocode, "T45 Sync Locale"),
                service.ageocode("T45 Concurrency Test Locale C"),
            )

        # One request from each path, in whichever order they raced.
        self.assertEqual(
            sorted(label for label, _ in clock.fired_at),
            ["T45 Concurrency Test Locale C", "T45 Sync Locale"],
        )
        self.assertGreaterEqual(clock.observed_gap(), MIN_THROTTLE_GAP - FLOAT_SLOP)
        self.assertEqual(len(reservations), 2)


class ThrottleSlotReservationTests(unittest.TestCase):
    """The reservation arithmetic, asserted on its own.

    The tests above drive the real geocode()/ageocode() paths, and so cover the
    wiring: that each caller claims a slot from the shared reservation point
    and then actually waits for it. This one calls ``_reserve_throttle_slot``
    directly, so it says nothing about either caller -- verified by mutation:
    revert both paths to their own unsynchronised read-modify-write and this
    test still passes while both tests above fail. Neither half is redundant
    with the other.

    Deliberately *not* a test that the lock works. A concurrent version of this
    -- N threads reserving at once, asserting the throttle advanced by exactly N
    seconds -- was written and then removed, because it passed just as happily
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

        clock = _VirtualClock()
        service = GeocodingService()
        service.last_request = clock.time() - 3600

        # Frozen time makes both waits exact rather than "about a second":
        # nothing here should vary with how long the two calls take to run.
        with _virtual_time(clock):
            immediate = service._reserve_throttle_slot()
            follower = service._reserve_throttle_slot()

        self.assertEqual(immediate, 0.0)
        self.assertEqual(follower, MIN_THROTTLE_GAP)


if __name__ == "__main__":
    unittest.main()
