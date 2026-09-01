"""Telemetry for admission control.

N is the least-measured constant in this system. Every other memory number
here was measured on real data -- frame_stack's 1,342 MB at 3.75M cells, the
32 MiB chunk sweep, the 204 MiB warm floor -- but N itself is *extrapolated*:
derived by arithmetic from an 8 GiB ceiling that no measurement has yet been
taken at. These metrics are what let it stop being extrapolated.

The four questions a dashboard has to answer, and the series that answer them:

* "Is the limit ever reached?"  -- ``admission_in_flight`` against the limit.
* "Is anyone waiting, and how long?" -- ``admission_queued`` and
  ``admission_wait_seconds``. A queue that is never non-empty means N is
  generous; waits measured in minutes mean the container needs memory rather
  than the queue needing depth.
* "Are we refusing work?" -- ``admission_shed_total``, split by surface,
  because a shed export is a retry and a shed chat turn is a lost turn.
* "What does the process actually weigh while N are resident?" --
  ``admission_rss_bytes``.

That last one is why this is not simply left to ``refresh_process_gauges``.
That gauge samples RSS *at scrape time*, so a reduction that climbs to 1.3 GB
and releases between two scrapes is invisible -- and those are precisely the
peaks that cause the OOM. Sampling inside the section catches them.

The sample is taken at section exit and is a *lower bound* on the section's
true peak, which is stated plainly rather than dressed up: catching the real
peak would need a sampling thread or tracemalloc, and neither is worth putting
on this path. It is a close lower bound for the reason the reserve is set above
the resting floor -- CPython does not promptly return freed memory to the OS,
so RSS after a large reduction still reflects most of what that reduction
took. Deliberately *current* RSS rather than a monotonic high-water mark, for
the reason ``_current_process_rss_bytes`` already documents: peak-only readers
climb forever and hide the plateau-then-shrink shape T45 added them to expose.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_isolation import ProcessCacheIsolation  # noqa: E402


def _gauge(metric) -> float:
    return metric.collect()[0].samples[0].value


def _counter_total(metric, **labels) -> float:
    """A labelled counter's current total, tolerating a series that does not
    exist yet -- an un-incremented label has no sample, and a test that read it
    as zero-by-assumption would pass against a counter nobody ever wired."""
    for sample in metric.collect()[0].samples:
        if sample.name.endswith("_total") and all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    return 0.0


def _histogram_count(metric) -> float:
    for sample in metric.collect()[0].samples:
        if sample.name.endswith("_count"):
            return sample.value
    return 0.0


class _Occupant:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self) -> None:
        from tta_backend.services import admission

        async with admission.admit():
            self.entered.set()
            await self.release.wait()


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


class AdmissionPublishesItsOccupancyTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """The gauges track what is actually inside the section."""

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "2"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_the_in_flight_gauge_follows_the_section(self):
        from tta_backend.utils.metrics import ADMISSION_IN_FLIGHT

        occupant = _Occupant()
        task = asyncio.create_task(occupant.run())
        self.addCleanup(task.cancel)
        await _settle()

        self.assertEqual(
            _gauge(ADMISSION_IN_FLIGHT), 1.0,
            "the in-flight gauge did not follow a caller into the heavy "
            "section, so a dashboard cannot tell a busy backend from an idle "
            "one -- which is the first question asked when N is being tuned.",
        )

        occupant.release.set()
        await task
        self.assertEqual(
            _gauge(ADMISSION_IN_FLIGHT), 0.0,
            "the in-flight gauge did not come back down. A gauge that only "
            "rises reads as a permanently saturated backend and would make "
            "every later reading meaningless.",
        )

    async def test_the_queue_gauge_follows_the_waiters(self):
        from tta_backend.utils.metrics import ADMISSION_QUEUED

        occupants = [_Occupant() for _ in range(4)]  # 2 admitted, 2 waiting
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        self.assertEqual(
            _gauge(ADMISSION_QUEUED), 2.0,
            "the queue-depth gauge does not report waiting callers, so the "
            "difference between 'slow' and 'queued behind other work' is "
            "invisible exactly when someone is trying to explain a slow turn.",
        )


class AdmissionRecordsWhatItRefusedTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """Sheds are counted, and counted separately per surface."""

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "1"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_a_shed_is_counted(self):
        from tta_backend.utils.metrics import ADMISSION_SHED_TOTAL

        before = _counter_total(ADMISSION_SHED_TOTAL, surface="unspecified")

        occupants = [_Occupant() for _ in range(4)]  # 1 admitted + 3 queued == capacity
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        with self.assertRaises(self.admission.AdmissionOverloaded):
            async with self.admission.admit():
                pass

        self.assertEqual(
            _counter_total(ADMISSION_SHED_TOTAL, surface="unspecified") - before, 1.0,
            "a refused request was not counted. Shedding is the one admission "
            "outcome a user actually feels, and an uncounted refusal means the "
            "first report of it is a complaint rather than a graph.",
        )

    async def test_the_surface_is_recorded_with_the_shed(self):
        """A shed export is a retry; a shed chat turn is a lost turn that has
        already spent its LLM tokens. One number covering both cannot tell an
        operator which is happening, and they carry very different urgency."""
        from tta_backend.utils.metrics import ADMISSION_SHED_TOTAL

        before = _counter_total(ADMISSION_SHED_TOTAL, surface="chat")

        occupants = [_Occupant() for _ in range(4)]
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        with self.assertRaises(self.admission.AdmissionOverloaded):
            async with self.admission.admit(surface="chat"):
                pass

        self.assertEqual(
            _counter_total(ADMISSION_SHED_TOTAL, surface="chat") - before, 1.0,
            "the shed was not attributed to the surface that suffered it.",
        )


class AdmissionMeasuresWaitingAndWeightTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """The two observations that let N be re-derived from production."""

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "1"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_every_admission_observes_its_wait(self):
        from tta_backend.utils.metrics import ADMISSION_WAIT_SECONDS

        before = _histogram_count(ADMISSION_WAIT_SECONDS)
        async with self.admission.admit():
            pass

        self.assertEqual(
            _histogram_count(ADMISSION_WAIT_SECONDS) - before, 1.0,
            "an admission recorded no wait observation. The uncontended case "
            "must be observed too -- a histogram fed only when callers queue "
            "reports a backend that is always congested.",
        )

    async def test_rss_is_sampled_inside_the_section(self):
        """Not at scrape time. A reduction that climbs to 1.3 GB and releases
        between two Prometheus scrapes is invisible to the process gauge, and
        those are exactly the peaks that cause the OOM."""
        from tta_backend.utils.metrics import ADMISSION_RSS_BYTES

        before = _histogram_count(ADMISSION_RSS_BYTES)
        async with self.admission.admit():
            pass

        self.assertEqual(
            _histogram_count(ADMISSION_RSS_BYTES) - before, 1.0,
            "no RSS sample was taken for a completed heavy section, so there "
            "is no way to answer what the process weighs while N are resident "
            "-- which is the measurement that would let N stop being an "
            "extrapolation from a container nobody has run it in.",
        )


class TelemetryNeverBreaksAdmissionTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """Measuring the gate must not be able to break the gate.

    Admission is load-bearing for memory safety; its metrics are not load-
    bearing for anything. A metrics backend that raises -- a platform whose RSS
    reader is unavailable, a collector mid-reconfiguration -- must cost an
    observation, never a permit. The inverse (a raising gauge that strands a
    permit) would take capacity away permanently and hang the backend, which
    is a far worse outcome than a missing series.
    """

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "1"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_a_raising_metrics_backend_still_admits_and_releases(self):
        # Patched at the *metrics* module, not at admission's own wrappers.
        # Those wrappers are where the guard lives, so replacing them would
        # remove the protection this test exists to check and the test would
        # pass or fail for reasons unrelated to the contract.
        boom = unittest.mock.Mock(side_effect=RuntimeError("collector unavailable"))
        with unittest.mock.patch.multiple(
            "tta_backend.utils.metrics",
            set_admission_occupancy=boom,
            observe_admission_wait=boom,
            observe_admission_rss=boom,
            current_process_rss_bytes=boom,
        ):
            async with self.admission.admit():
                pass

        self.assertTrue(boom.called, "the failing metrics path was never reached")

        self.assertEqual(
            self.admission.in_flight(), 0,
            "a failing metrics call leaked a permit. Telemetry is best-effort "
            "and admission is not: losing an observation is a gap in a graph, "
            "while losing a permit removes capacity for the life of the "
            "process and eventually hangs every heavy request.",
        )
        async with self.admission.admit():
            pass


if __name__ == "__main__":
    unittest.main()
