"""harness.jobs: reusing a saved retrieval job across runs, and checking the jobs waiting runs are on."""
import asyncio
import datetime
import tempfile
import unittest
from contextlib import suppress
from typing import Any

from tests.test_harness_grids import NJ
from tests.test_harness_sources_satellite import NO2, SECRET, TODAY, FakeService
from tta_backend.harness.capabilities import begin, invoke
from tta_backend.harness.capabilities.failures import JobFailed, JobStillRunning
from tta_backend.harness.capabilities.sources import areas, satellite
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness import jobs
from tta_backend.harness.job_store import InMemoryRetrievalJobStore, SavedJob
from tta_backend.harness.jobs import ReusingEarthdata, check_waiting, waiting_jobs
from tta_backend.harness.results import Mode
from tta_backend.harness.run_context import EarthdataSession, RunContext, RunEnv, request_fingerprint
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import (
    FakeEarthdata,
    FakePlaces,
    FakeToolError,
    FixedClock,
    StaticCredentials,
    make_run_context,
    make_run_env,
)
from tta_backend.harness.trace import FailureCategory, RunStatus, StepFailure, TraceEntry

NOW = datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.UTC)
REQUEST = {"dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "2024-06-01"}


class Clock:
    def __init__(self, now: datetime.datetime = NOW) -> None:
        self.current = now

    def today(self) -> datetime.date:
        return self.current.date()

    def now(self) -> datetime.datetime:
        return self.current


class Service:
    """A scripted earthdata session: submits mint job_1, job_2, ...; statuses are set per job.

    A status not set is running. ``gate``, when set, holds every submit until it opens.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.statuses: dict[str, dict[str, Any] | Exception] = {}
        self.submitted = 0
        self.gate: asyncio.Event | None = None
        self.in_flight = 0
        self.most_in_flight = 0

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        self.calls.append((tool, dict(arguments), idempotency_key))
        if tool in ("retrieve_subset", "retrieve_timeseries"):
            if self.gate is not None:
                await self.gate.wait()
            self.submitted += 1
            return {"job_handle": f"job_{self.submitted}", "message": "submitted"}
        if tool == "get_retrieval_status":
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
            try:
                await asyncio.sleep(0)
                status = self.statuses.get(arguments["job_handle"], {"status": "running"})
                if isinstance(status, Exception):
                    raise status
                return {"job_handle": arguments["job_handle"], **status}
            finally:
                self.in_flight -= 1
        return {"tool": tool}

    def tools(self) -> list[str]:
        return [tool for tool, _, _ in self.calls]


class ReuseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = Service()
        self.store = InMemoryRetrievalJobStore()
        self.clock = Clock()
        self.earthdata = self.wrap()

    def wrap(self, user_id: str = "user-a", inner: EarthdataSession | None = None) -> ReusingEarthdata:
        return ReusingEarthdata(inner or self.service, self.store, user_id, self.clock)

    async def submit(self, key: str = "key-1", tool: str = "retrieve_subset", **over: Any) -> dict[str, Any]:
        return await self.earthdata.call(tool, {**REQUEST, **over}, idempotency_key=key)

    async def saved(self, tool: str = "retrieve_subset", user_id: str = "user-a") -> SavedJob | None:
        return await self.store.find(user_id, request_fingerprint(tool, REQUEST))

    async def test_a_submit_passes_its_key_through_and_saves_the_job(self):
        answer = await self.submit()
        self.assertEqual(answer, {"job_handle": "job_1", "message": "submitted"})
        self.assertEqual(self.service.calls, [("retrieve_subset", REQUEST, "key-1")])
        self.assertEqual(await self.saved(), SavedJob(
            fingerprint=request_fingerprint("retrieve_subset", REQUEST), tool="retrieve_subset",
            job_handle="job_1", answer=answer, submitted=NOW,
        ))

    async def test_other_tools_pass_straight_through(self):
        for tool in ("check_coverage", "get_retrieval_status", "cancel_retrieval"):
            with self.subTest(tool):
                await self.earthdata.call(tool, {"job_handle": "job_9"}, idempotency_key="key-x")
                self.assertEqual(self.service.calls[-1], (tool, {"job_handle": "job_9"}, "key-x"))
        self.assertEqual(len(self.service.calls), 3)
        self.assertIsNone(await self.saved())

    async def test_a_saved_job_still_held_is_reused_under_another_key(self):
        first = await self.submit("run-1-key")
        for status in ({"status": "queued"}, {"status": "running"}, {"status": "ready", "obs_handle": "obs_1"}):
            with self.subTest(status):
                self.service.statuses["job_1"] = status
                self.clock.current = NOW + datetime.timedelta(hours=23, minutes=59)
                again = await self.submit("run-2-key")
                self.assertEqual(again, first)
        self.assertEqual(self.service.submitted, 1)
        self.assertEqual(self.service.tools(), ["retrieve_subset"] + ["get_retrieval_status"] * 3)
        status_call = self.service.calls[1]
        self.assertEqual(status_call, ("get_retrieval_status", {"job_handle": "job_1"}, None))

    async def test_each_submit_tool_and_request_is_saved_apart(self):
        await self.submit(tool="retrieve_subset")
        await self.submit(tool="retrieve_timeseries")
        await self.submit(time_range="2024-06-02")
        self.assertEqual(self.service.submitted, 3)
        timeseries = await self.saved("retrieve_timeseries")
        assert timeseries is not None
        self.assertEqual((timeseries.tool, timeseries.job_handle), ("retrieve_timeseries", "job_2"))

    async def test_an_old_job_is_not_asked_about_and_the_request_is_submitted(self):
        await self.submit()
        self.clock.current = NOW + datetime.timedelta(hours=24)
        answer = await self.submit("key-2")
        self.assertEqual(answer["job_handle"], "job_2")
        self.assertNotIn("get_retrieval_status", self.service.tools())
        saved = await self.saved()
        assert saved is not None
        self.assertEqual((saved.job_handle, saved.submitted), ("job_2", self.clock.current))

    async def test_a_shorter_max_age_is_kept(self):
        earthdata = ReusingEarthdata(self.service, self.store, "user-a", self.clock, max_age=datetime.timedelta(minutes=5))
        await earthdata.call("retrieve_subset", REQUEST, idempotency_key="k1")
        self.clock.current = NOW + datetime.timedelta(minutes=5)
        await earthdata.call("retrieve_subset", REQUEST, idempotency_key="k2")
        self.assertEqual(self.service.submitted, 2)

    async def test_a_job_the_provider_will_not_finish_is_forgotten_and_the_request_submitted(self):
        dead: list[dict[str, Any] | Exception] = [
            {"status": "failed"},
            {"status": "cancelled"},
            {"status": "expired"},
            {"status": "not_found"},
            {"status": "running", "message": "The job is paused by the provider"},
            FakeToolError("not_found"),
        ]
        await self.submit("key-0")
        for index, status in enumerate(dead, start=1):
            with self.subTest(status):
                self.service.statuses[f"job_{index}"] = status
                answer = await self.submit(f"key-{index}")
                self.assertEqual(answer["job_handle"], f"job_{index + 1}")
                saved = await self.saved()
                assert saved is not None
                self.assertEqual(saved.job_handle, f"job_{index + 1}")
        self.assertEqual(self.service.submitted, len(dead) + 1)
        self.assertEqual([key for tool, _, key in self.service.calls if tool == "retrieve_subset"],
                         [f"key-{i}" for i in range(len(dead) + 1)])

    async def test_a_status_outage_returns_the_saved_answer_without_submitting(self):
        first = await self.submit()
        self.service.statuses["job_1"] = FakeToolError("provider_unavailable")
        self.assertEqual(await self.submit("key-2"), first)
        self.assertEqual(self.service.submitted, 1)
        self.assertIsNotNone(await self.saved())

    async def test_another_classified_status_error_is_raised_and_the_job_kept(self):
        await self.submit()
        self.service.statuses["job_1"] = FakeToolError("token_expired")
        with self.assertRaises(FakeToolError) as caught:
            await self.submit("key-2")
        self.assertEqual(caught.exception.category, "token_expired")
        self.assertEqual(self.service.submitted, 1)
        self.assertIsNotNone(await self.saved())

    async def test_an_answer_without_a_job_handle_is_not_saved(self):
        class NoHandle(Service):
            async def call(self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None):
                await super().call(tool, arguments, idempotency_key=idempotency_key)
                return {"message": "accepted"}

        earthdata = self.wrap(inner=NoHandle())
        self.assertEqual(await earthdata.call("retrieve_subset", REQUEST, idempotency_key="k"), {"message": "accepted"})
        self.assertIsNone(await self.saved())

    async def test_another_users_saved_job_is_not_reused(self):
        await self.submit()
        other = self.wrap("user-b")
        answer = await other.call("retrieve_subset", REQUEST, idempotency_key="key-b")
        self.assertEqual(answer["job_handle"], "job_2")
        self.assertNotIn("get_retrieval_status", self.service.tools())
        saved_a, saved_b = await self.saved(), await self.saved(user_id="user-b")
        assert saved_a is not None and saved_b is not None
        self.assertEqual((saved_a.job_handle, saved_b.job_handle), ("job_1", "job_2"))

    async def test_two_identical_submits_at_once_submit_once(self):
        self.service.gate = asyncio.Event()
        both = asyncio.gather(self.submit("key-1"), self.submit("key-2"))
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(self.service.tools(), ["retrieve_subset"])
        self.service.gate.set()
        first, second = await both
        self.assertEqual(first, second)
        self.assertEqual(self.service.submitted, 1)
        self.assertEqual(jobs._LOCKS, {})

    async def test_two_instances_submitting_the_same_request_at_once_submit_once(self):
        # As two runs do, each with its own wrapper over the shared store.
        self.service.gate = asyncio.Event()
        first, second = self.wrap(), self.wrap()
        both = asyncio.gather(
            first.call("retrieve_subset", REQUEST, idempotency_key="run-1"),
            second.call("retrieve_subset", REQUEST, idempotency_key="run-2"),
        )
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(self.service.tools(), ["retrieve_subset"])
        self.service.gate.set()
        one, two = await both
        self.assertEqual(one, two)
        self.assertEqual(self.service.submitted, 1)
        self.assertEqual(jobs._LOCKS, {})

    async def test_another_users_identical_request_does_not_wait(self):
        self.service.gate = asyncio.Event()
        both = asyncio.gather(
            self.wrap("user-a").call("retrieve_subset", REQUEST, idempotency_key="a"),
            self.wrap("user-b").call("retrieve_subset", REQUEST, idempotency_key="b"),
        )
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(self.service.tools(), ["retrieve_subset", "retrieve_subset"])
        self.service.gate.set()
        await both
        self.assertEqual(jobs._LOCKS, {})

    async def test_different_requests_at_once_do_not_wait_on_each_other(self):
        self.service.gate = asyncio.Event()
        both = asyncio.gather(self.submit("key-1"), self.submit("key-2", time_range="2024-06-02"))
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(self.service.tools(), ["retrieve_subset", "retrieve_subset"])
        self.service.gate.set()
        first, second = await both
        self.assertNotEqual(first, second)

    async def test_a_repeat_of_a_key_answered_with_the_dead_job_is_not_saved_again(self):
        answered: dict[str, dict[str, Any]] = {}

        class Remembering(Service):
            async def call(self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None):
                if idempotency_key in answered:
                    return answered[idempotency_key]
                result = await super().call(tool, arguments, idempotency_key=idempotency_key)
                if idempotency_key is not None:
                    answered[idempotency_key] = result
                return result

        inner = Remembering()
        earthdata = self.wrap(inner=inner)
        await earthdata.call("retrieve_subset", REQUEST, idempotency_key="key-1")
        inner.statuses["job_1"] = {"status": "failed"}
        again = await earthdata.call("retrieve_subset", REQUEST, idempotency_key="key-1")
        self.assertEqual(again["job_handle"], "job_1")
        self.assertIsNone(await self.saved())

    async def test_a_store_that_fails_does_not_stop_a_submit(self):
        class Broken(InMemoryRetrievalJobStore):
            async def find(self, user_id: str, fingerprint: str) -> SavedJob | None:
                raise OSError("down")

            async def save(self, user_id: str, job: SavedJob) -> None:
                raise OSError("down")

        earthdata = ReusingEarthdata(self.service, Broken(), "user-a", self.clock)
        with self.assertLogs("tta_backend.harness.jobs", "WARNING"):
            answer = await earthdata.call("retrieve_subset", REQUEST, idempotency_key="k")
        self.assertEqual(answer["job_handle"], "job_1")


# End to end on the satellite source.


class SatelliteReuseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service = FakeService()
        self.fake = self.service.earthdata()
        self.jobs = InMemoryRetrievalJobStore()
        self.earthdata = ReusingEarthdata(self.fake, self.jobs, "user-a", FixedClock(TODAY))
        self.context = make_run_context(
            credentials=StaticCredentials({"earthdata": SECRET}), clock=FixedClock(TODAY), stores=memory_stores(),
        )
        first = await self.make_env()
        self.area = (await invoke(first, areas.resolve, {"place": "New Jersey"})).handle

    async def make_env(self, mode: Mode | None = None) -> RunEnv:
        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.service.files[record.id]

        return await make_run_env(
            self._tmp.name, context=self.context, earthdata=self.earthdata, open_retrieval=open_retrieval,
            places=FakePlaces({"new jersey": NJ}), mode=mode,
        )

    def params(self) -> dict[str, Any]:
        return {"quantity": NO2, "area": self.area, "dates": "2024-06-01", "date_role": "subject"}

    def submits(self) -> int:
        return [tool for tool, _ in self.fake.calls].count("retrieve_subset")

    async def test_two_runs_asking_for_the_same_data_submit_once(self):
        first_run, second_run = await self.make_env(), await self.make_env(Mode.validated("ozone-day", 1))
        first = await invoke(first_run, satellite.get_data, self.params(), step="event")
        second = await invoke(second_run, satellite.get_data, self.params(), step="event")

        self.assertEqual((first.status, second.status), ("done", "done"))
        self.assertEqual((first.handle, second.handle), ("obs_1", "obs_1"))
        self.assertEqual(self.submits(), 1)
        entries = await self.context.storage.traces.entries("user-a", second_run.run_id)
        waiting = next(e for e in entries if e.status == "waiting")
        self.assertEqual(waiting.result["job"]["jobs"], [{"job": "job_1", "day": "2024-06-01"}])

    async def test_a_retry_after_a_failed_job_forgets_it_and_submits_a_new_one(self):
        ready = self.service.status

        def status(args: dict[str, Any]) -> dict[str, Any]:
            if args["job_handle"] == "job_1":
                return {"job_handle": "job_1", "status": "failed", "message": "provider says no"}
            return ready(args)

        self.fake.tools["get_retrieval_status"] = status
        env = await self.make_env()
        with self.assertRaises(JobFailed):
            await invoke(env, satellite.get_data, self.params(), step="event")

        # The step's retry: the same step again under the next attempt key.
        retried = await invoke(env, satellite.get_data, self.params(), step="event", attempt=1)
        self.assertEqual((retried.status, retried.handle), ("done", "obs_2"))
        self.assertEqual(self.submits(), 2)
        [saved] = self.jobs._jobs.values()
        self.assertEqual(saved.job_handle, "job_2")

        # A later run collects the new job.
        later = await invoke(await self.make_env(), satellite.get_data, self.params(), step="event")
        self.assertEqual(later.handle, "obs_2")
        self.assertEqual(self.submits(), 2)

    async def test_a_waiting_run_from_a_real_begin_is_checked(self):
        env = await self.make_env()
        await begin(env, satellite.get_data, self.params(), step="event")
        await self.context.storage.traces.finish_run("user-a", env.run_id, "waiting", TODAY)
        self.assertEqual(await check_waiting(self.context, self.fake), {env.run_id: "ready"})
        self.service.statuses = [{"status": "running"}]
        self.assertEqual(await check_waiting(self.context, self.fake), {env.run_id: "running"})


# check_waiting on recorded runs.

CAPABILITY = "sources.satellite.get_data"


def job_ref(*ids: str) -> dict[str, Any]:
    return {"product": "TEMPO_NO2", "jobs": [{"job": job_id, "day": None} for job_id in ids]}


async def begun(env: RunEnv, job: dict[str, Any]) -> None:
    async with env.recorder.step(kind="use", capability=CAPABILITY, params={"dates": "2024-06-01"}) as scope:
        scope.status = "waiting"
        scope.result = {"job": job}


async def completed(env: RunEnv, job: dict[str, Any], outcome: str = "done") -> None:
    """A complete entry for ``job``: done, still_running, dead, cancelled or transient."""
    params = {"dates": "2024-06-01", "_job": job}
    ids = [item["job"] for item in job["jobs"]]
    with suppress(StepFailure):
        async with env.recorder.step(kind="use", capability=CAPABILITY, params=params) as scope:
            if outcome == "still_running":
                raise JobStillRunning(CAPABILITY, job, ids)
            if outcome == "dead":
                raise JobFailed("failed at the provider", job=job, state="failed", job_ids=ids)
            if outcome == "cancelled":
                raise StepFailure(FailureCategory.BAD_INPUT, "cancelled", result={"job": job, "cancelled": ids})
            if outcome == "transient":
                raise StepFailure(FailureCategory.SOURCE_UNAVAILABLE, "the provider is down")
            scope.result = {"status": "done", "handle": "obs_1"}


class CheckWaitingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        self.context = make_run_context(stores=self.stores)
        self.service = Service()

    async def run_with(
        self,
        *steps: tuple[str, dict[str, Any]],
        status: RunStatus = "waiting",
        context: RunContext | None = None,
        mode: Mode | None = None,
    ) -> str:
        """A run holding ``steps``, each ("begin", job) or (outcome, job), finished as ``status``."""
        context = context or self.context
        env = await make_run_env(self._tmp.name, context=context, mode=mode)
        for kind, job in steps:
            if kind == "begin":
                await begun(env, job)
            else:
                await completed(env, job, kind)
        if status != "running":
            await context.storage.traces.finish_run(context.identity.user_id, env.run_id, status, NOW)
        return env.run_id

    async def check(self, **kw: Any) -> dict[str, str]:
        return dict(await check_waiting(self.context, self.service, **kw))

    async def test_each_run_state(self):
        self.service.statuses.update({
            "job_ready": {"status": "ready", "obs_handle": "obs_1"},
            "job_ready_2": {"status": "ready", "obs_handle": "obs_2"},
            "job_failed": {"status": "failed"},
            "job_cancelled": {"status": "cancelled"},
            "job_expired": {"status": "expired"},
            "job_paused": {"status": "running", "message": "paused at the provider"},
            "job_gone": FakeToolError("not_found"),
            "job_down": FakeToolError("provider_unavailable"),
        })
        runs = {
            "ready": await self.run_with(("begin", job_ref("job_ready", "job_ready_2")), mode=Mode.validated("w", 1)),
            "running": await self.run_with(("begin", job_ref("job_queued"))),
            "mixed_running": await self.run_with(("begin", job_ref("job_ready", "job_queued"))),
            "mixed_failed": await self.run_with(("begin", job_ref("job_ready", "job_queued", "job_failed"))),
            "cancelled": await self.run_with(("begin", job_ref("job_cancelled"))),
            "expired": await self.run_with(("begin", job_ref("job_expired"))),
            "paused": await self.run_with(("begin", job_ref("job_paused"))),
            "gone": await self.run_with(("begin", job_ref("job_gone"))),
            "down": await self.run_with(("begin", job_ref("job_down", "job_ready"))),
            "no_job": await self.run_with(),
            "completed": await self.run_with(("begin", job_ref("job_done")), ("done", job_ref("job_done"))),
        }
        expected = {
            "ready": "ready", "running": "running", "mixed_running": "running", "mixed_failed": "failed",
            "cancelled": "failed", "expired": "failed", "paused": "failed", "gone": "failed",
            "down": "unknown", "no_job": "unknown", "completed": "unknown",
        }
        self.assertEqual(await self.check(), {runs[name]: state for name, state in expected.items()})
        polled = [args["job_handle"] for tool, args, _ in self.service.calls]
        self.assertEqual(sorted(polled), sorted(set(polled)))
        self.assertNotIn("job_done", polled)
        self.assertTrue(all(tool == "get_retrieval_status" and key is None for tool, _, key in self.service.calls))

    async def test_which_jobs_a_run_still_waits_on(self):
        self.service.statuses.update({"job_a": {"status": "ready"}, "job_d": {"status": "ready"}})
        still = await self.run_with(("begin", job_ref("job_a")), ("still_running", job_ref("job_a")))
        retried = await self.run_with(
            ("begin", job_ref("job_b")), ("dead", job_ref("job_b")), ("begin", job_ref("job_c")),
        )
        transient = await self.run_with(("begin", job_ref("job_d")), ("transient", job_ref("job_d")))
        settled = await self.run_with(
            ("begin", job_ref("job_e")), ("cancelled", job_ref("job_e")),
            ("begin", job_ref("job_f")), ("dead", job_ref("job_f")),
        )
        self.assertEqual(await self.check(), {
            still: "ready", retried: "running", transient: "ready", settled: "unknown",
        })
        polled = sorted(args["job_handle"] for _, args, _ in self.service.calls)
        self.assertEqual(polled, ["job_a", "job_c", "job_d"])

    async def test_a_job_two_runs_share_is_asked_about_once(self):
        first = await self.run_with(("begin", job_ref("job_1")))
        second = await self.run_with(("begin", job_ref("job_1")))
        self.assertEqual(await self.check(), {first: "running", second: "running"})
        self.assertEqual(len(self.service.calls), 1)

    async def test_only_this_users_waiting_runs_in_this_conversation_count(self):
        mine = await self.run_with(("begin", job_ref("job_mine")))
        await self.run_with(("begin", job_ref("job_done_run")), status="done")
        await self.run_with(("begin", job_ref("job_asking")), status="needs_input")
        await self.run_with(("begin", job_ref("job_running")), status="running")
        other_user = make_run_context(user_id="user-b", stores=self.stores)
        theirs = await self.run_with(("begin", job_ref("job_theirs")), context=other_user)
        other_conversation = make_run_context(conversation_id="conv-2", stores=self.stores)
        await self.run_with(("begin", job_ref("job_elsewhere")), context=other_conversation)

        self.assertEqual(await self.check(), {mine: "running"})
        self.assertEqual([args["job_handle"] for _, args, _ in self.service.calls], ["job_mine"])
        self.assertEqual(await check_waiting(other_user, self.service), {theirs: "running"})

    async def test_no_waiting_runs_asks_nothing(self):
        self.assertEqual(await self.check(), {})
        self.assertEqual(self.service.calls, [])

    async def test_status_calls_are_bounded(self):
        for index in range(7):
            await self.run_with(("begin", job_ref(f"job_{index}")))
        await self.check(max_concurrent=2)
        self.assertEqual(len(self.service.calls), 7)
        self.assertEqual(self.service.most_in_flight, 2)
        with self.assertRaises(ValueError):
            await self.check(max_concurrent=0)

    async def test_an_unclassified_status_error_is_raised(self):
        self.service.statuses["job_1"] = RuntimeError("bug")
        await self.run_with(("begin", job_ref("job_1")))
        with self.assertRaises(RuntimeError):
            await self.check()

    def test_a_question_entry_is_not_a_job(self):
        # A begin that settled with a question records a waiting entry with a result, not a job.
        question = TraceEntry(
            seq=1, run_id="run_" + "0" * 32, kind="use", status="waiting", started=NOW, seconds=0.0,
            capability=CAPABILITY, result={"status": "needs_input", "question": {"kind": "pick_one"}},
        )
        self.assertEqual(waiting_jobs([question]), [])


class RunEnvWrappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_env_wraps_a_plain_session_once(self):
        with tempfile.TemporaryDirectory() as root:
            plain = FakeEarthdata()
            env = await make_run_env(root, earthdata=plain)
            self.assertIsInstance(env.earthdata, ReusingEarthdata)

            # An already wrapped session is used as it is; a second layer
            # would wait on the lock the first holds.
            wrapped = ReusingEarthdata(plain, InMemoryRetrievalJobStore(), "user-a", FixedClock(TODAY))
            again = await make_run_env(root, earthdata=wrapped)
            self.assertIs(again.earthdata, wrapped)


if __name__ == "__main__":
    unittest.main()
