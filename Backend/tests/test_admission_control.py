"""Admission control for the memory wall.

Every per-request gate in this backend is measured and honest -- ``frame_stack``
refuses above 4,000,000 native cells because 17M cells was OOM-killed and 3.75M
completed at 1,342 MB. But each of those gates bounds *one* request, and nothing
bounded how many ran at once: the heavy reducers all hop to a worker thread via
``asyncio.to_thread``, so they genuinely overlap, and the only ceiling was the
default executor's ``min(32, cpu_count + 4)`` -- a number derived from CPU count,
which has nothing to do with memory.

Measured 2026-09-01 on the live container: the warm process floor is 204 MiB
(346 MiB high-water), and ``memory.max`` reads ``max`` -- no limit at all, so an
overshoot is the kernel killing PID 1 and severing every in-flight SSE stream,
including turns that were nearly finished. Users have already reported that as
"network error".

What these tests pin
--------------------
* Concurrency is bounded, and the bound is *derived* from the declared memory
  ceiling rather than guessed -- so raising ``mem_limit`` raises throughput and
  lowering it lowers admission, instead of the two drifting apart silently.
* A permit is returned on **every** exit path. This is the failure mode that
  matters most here: a leaked permit does not corrupt anything, it *hangs* --
  in production forever, and in a ~29-minute suite it hangs somewhere nobody can
  attribute. Hence the exception and cancellation cases below, and hence
  admission state riding the runner-independent isolation policy in
  ``cache_isolation`` rather than a pytest-only fixture (``unittest`` never
  loads a conftest -- this repo has already paid for that once).
* Overload sheds instead of queueing without limit. An unbounded queue turns an
  OOM into a slow-motion timeout in which every caller waits, spends its LLM
  tokens, and *still* loses the turn at ``CHAT_TURN_TIMEOUT_SECONDS``.

Every assertion here has been shown to bite by mutation (2026-09-01): breaking
the queue cap, the depth multiple, the derivation floor, the explicit override,
the permit release, the queued counter, or the isolation hook each fails a named
test. Two of those mutations originally *hung* rather than failed, which is why
the admit-side assertions go through :func:`_admit_now` -- see its docstring.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_isolation import ProcessCacheIsolation, clear_process_caches  # noqa: E402


async def _settle() -> None:
    """Let every runnable task reach its next await point.

    The event loop is single-threaded and these tests never touch a real thread
    pool, so yielding a bounded number of times is deterministic -- there is no
    wall-clock race to lose. A ``sleep(delay)`` here would be the flaky version:
    it would pass on a fast machine and fail on a loaded one, and this suite
    runs on both.
    """
    for _ in range(50):
        await asyncio.sleep(0)


async def _admit_now(admission, timeout: float = 5.0) -> None:
    """Enter and leave the heavy section, failing fast if admission blocks.

    Every assertion that a caller *can* be admitted goes through here rather
    than through a bare ``async with``, because the two most dangerous defects
    in this module do not raise -- they hang. Proven by mutation: making the
    queue cap never trip, and never releasing a permit, each turned a should-be
    failing test into a run that never terminated. A hang is strictly worse than
    a failure here; it reports nothing, names no test, and does it inside a
    suite that takes around 29 minutes to reach the end.

    The timeout is generous on purpose. Admission is pure asyncio with no I/O,
    so a healthy acquire completes in microseconds -- five seconds is not a
    performance assertion that a loaded machine could trip, it is the boundary
    between "slow" and "never".
    """

    async def _once() -> None:
        async with admission.admit():
            return

    await asyncio.wait_for(_once(), timeout)


class _Occupant:
    """A caller that holds a permit until told to let go.

    Deliberately not a bare coroutine: the tests need to distinguish *entered
    the section* from *scheduled but still waiting*, which is the entire
    distinction admission control exists to create.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failed: BaseException | None = None

    async def run(self) -> None:
        from tta_backend.services import admission

        try:
            async with admission.admit():
                self.entered.set()
                await self.release.wait()
        except BaseException as exc:  # noqa: BLE001 - recorded, re-raised below
            self.failed = exc
            raise


class AdmissionBoundsConcurrencyTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """At most N callers are inside the heavy section at once."""

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "2"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_only_the_limit_runs_concurrently(self):
        occupants = [_Occupant() for _ in range(3)]
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])

        await _settle()

        self.assertEqual(
            self.admission.in_flight(), 2,
            "three callers arrived at a limit of two, but the number inside the "
            "heavy section is not two -- concurrency is not actually bounded.",
        )
        self.assertTrue(occupants[0].entered.is_set())
        self.assertTrue(occupants[1].entered.is_set())
        self.assertFalse(
            occupants[2].entered.is_set(),
            "the third caller entered the heavy section while two were already "
            "inside it. This is the OOM path: N concurrent reductions each hold "
            "their own intermediates, so admitting one too many is what takes "
            "the container over its ceiling.",
        )

    async def test_a_released_permit_admits_the_next_waiter(self):
        occupants = [_Occupant() for _ in range(3)]
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        occupants[0].release.set()
        await _settle()

        self.assertTrue(
            occupants[2].entered.is_set(),
            "a permit was released and the waiting caller was not admitted -- "
            "the queue does not drain, so load that arrives during a busy "
            "period never recovers.",
        )
        self.assertEqual(self.admission.in_flight(), 2)

    async def test_waiting_callers_are_reported_as_queued(self):
        """The queue depth has to be observable, or the ``status`` event that
        tells a user "queued, not stalled" has nothing to report and the wait is
        indistinguishable from the freeze this work exists to remove."""
        occupants = [_Occupant() for _ in range(4)]
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        self.assertEqual(self.admission.in_flight(), 2)
        self.assertEqual(self.admission.queued(), 2)


class PermitsAreReturnedOnEveryPathTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """A permit that is not returned does not corrupt -- it hangs.

    Which is why these cases exist separately from the happy path: a leak is
    invisible until the Nth one, and then the symptom is a backend that accepts
    requests and never answers any of them.
    """

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "1"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_an_exception_inside_the_section_returns_its_permit(self):
        with self.assertRaises(ValueError):
            async with self.admission.admit():
                raise ValueError("the reduction failed")

        self.assertEqual(
            self.admission.in_flight(), 0,
            "an exception inside the heavy section leaked its permit. Reductions "
            "raise routinely -- a refused extent, an unreadable granule -- so a "
            "leak here would drain the pool during ordinary operation.",
        )
        await _admit_now(self.admission)

    async def test_cancellation_while_waiting_returns_no_permit(self):
        """A cancelled *waiter* never held a permit, and must not release one it
        does not own -- that would over-credit the pool and admit N+1."""
        holder = _Occupant()
        held = asyncio.create_task(holder.run())
        await _settle()

        waiter = _Occupant()
        pending = asyncio.create_task(waiter.run())
        await _settle()
        self.assertFalse(waiter.entered.is_set())

        pending.cancel()
        await _settle()

        self.assertEqual(
            self.admission.in_flight(), 1,
            "cancelling a queued caller changed the in-flight count. A waiter "
            "holds nothing, so its cancellation must not credit the pool.",
        )
        holder.release.set()
        await held
        self.assertEqual(self.admission.in_flight(), 0)

    async def test_cancellation_inside_the_section_returns_its_permit(self):
        occupant = _Occupant()
        task = asyncio.create_task(occupant.run())
        await _settle()
        self.assertTrue(occupant.entered.is_set())

        task.cancel()
        await _settle()

        self.assertEqual(
            self.admission.in_flight(), 0,
            "a caller cancelled while holding a permit did not return it. "
            "Client disconnects cancel in-flight work, so this is the ordinary "
            "case, not the exotic one.",
        )


class OverloadIsShedRatherThanQueuedForeverTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """Past a bounded queue, arrivals are refused rather than accumulated."""

    async def asyncSetUp(self) -> None:
        from tta_backend.services import admission

        self.admission = admission
        self.enterContext(unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "2"}))
        admission.reset_admission()
        self.addCleanup(admission.reset_admission)

    async def test_the_queue_is_bounded_relative_to_the_limit(self):
        self.assertEqual(
            self.admission.queue_capacity(), 6,
            "the queue capacity is not three times the admission limit. The "
            "ratio is the whole policy: too small and a normal burst is refused, "
            "too large and overload becomes a slow-motion timeout in which every "
            "caller pays for a turn it will not get.",
        )

    async def test_arrivals_past_the_queue_cap_are_shed(self):
        occupants = [_Occupant() for _ in range(8)]  # 2 in flight + 6 queued
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        self.assertEqual(self.admission.in_flight(), 2)
        self.assertEqual(self.admission.queued(), 6)

        with self.assertRaises(
            self.admission.AdmissionOverloaded,
            msg="the ninth caller was queued rather than shed. An unbounded "
                "queue converts an OOM into every caller timing out at "
                "CHAT_TURN_TIMEOUT_SECONDS after spending its tokens.",
        ):
            await _admit_now(self.admission)

    async def test_shedding_stops_once_the_queue_drains(self):
        occupants = [_Occupant() for _ in range(8)]
        tasks = [asyncio.create_task(o.run()) for o in occupants]
        self.addCleanup(lambda: [t.cancel() for t in tasks])
        await _settle()

        for occupant in occupants:
            occupant.release.set()
        await _settle()

        # Must not raise -- shedding is a state, not a latch.
        await _admit_now(self.admission)


class TheLimitIsDerivedFromTheDeclaredCeilingTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """N follows the memory ceiling, so the two cannot drift apart.

    The alternative -- a hardcoded N with the ceiling in prose -- means someone
    doubling the container's memory gets no extra throughput, and someone
    halving it gets an OOM the constant no longer protects against.
    """

    def _limit_for(self, ceiling_mb: str) -> int:
        from tta_backend.services import admission

        env = {"BACKEND_MEM_LIMIT_MB": ceiling_mb, "HEAVY_ADMISSION_LIMIT": "0"}
        with unittest.mock.patch.dict(os.environ, env):
            admission.reset_admission()
            try:
                return admission.heavy_limit()
            finally:
                admission.reset_admission()

    async def test_the_deployment_ceiling_admits_five(self):
        self.assertEqual(
            self._limit_for("8192"), 5,
            "at the 8 GiB deployment ceiling the limit is not 5. Derivation is "
            "floor((8192 - 450 reserve) / 1400 per request); the reserve is the "
            "measured 204 MiB warm floor plus margin for allocator retention, "
            "and 1400 MB is the largest reduction frame_stack admits (1,342 MB "
            "measured at 3.75M cells).",
        )

    async def test_a_small_dev_container_admits_one(self):
        self.assertEqual(
            self._limit_for("3072"), 1,
            "a 3 GiB dev container must serialise heavy work rather than admit "
            "two reductions that together exceed it.",
        )

    async def test_the_limit_never_falls_below_one(self):
        self.assertEqual(
            self._limit_for("256"), 1,
            "a ceiling below one request's cost yielded a limit of zero, which "
            "does not protect memory -- it deadlocks every heavy request "
            "forever. Refusing to serve is not the same as refusing to admit.",
        )

    async def test_an_explicit_limit_overrides_the_derivation(self):
        from tta_backend.services import admission

        env = {"BACKEND_MEM_LIMIT_MB": "8192", "HEAVY_ADMISSION_LIMIT": "3"}
        with unittest.mock.patch.dict(os.environ, env):
            admission.reset_admission()
            self.addCleanup(admission.reset_admission)
            self.assertEqual(
                admission.heavy_limit(), 3,
                "an explicit HEAVY_ADMISSION_LIMIT did not win over the derived "
                "value. This is the operator's escape hatch for a host whose "
                "real headroom differs from what compose declares.",
            )


class AdmissionStateIsIsolatedBetweenTestsTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    """Admission state is process-global, so it leaks between tests.

    Registered in ``cache_isolation`` rather than in ``conftest.py`` for the
    reason that module documents: ``unittest`` never loads a conftest, and CI
    runs ``unittest discover``. A cache that leaks serves a stale answer; a
    permit that leaks hangs the runner with no attribution at all.
    """

    async def test_clearing_process_caches_restores_every_permit(self):
        from tta_backend.services import admission

        with unittest.mock.patch.dict(os.environ, {"HEAVY_ADMISSION_LIMIT": "1"}):
            admission.reset_admission()
            self.addCleanup(admission.reset_admission)

            occupant = _Occupant()
            task = asyncio.create_task(occupant.run())
            self.addCleanup(task.cancel)
            await _settle()
            self.assertEqual(admission.in_flight(), 1)

            clear_process_caches()

            self.assertEqual(
                admission.in_flight(), 0,
                "clear_process_caches() did not reset admission state, so a test "
                "that leaves a permit held hangs every later test that needs "
                "one -- in a suite that takes ~29 minutes to reach them.",
            )


if __name__ == "__main__":
    unittest.main()
