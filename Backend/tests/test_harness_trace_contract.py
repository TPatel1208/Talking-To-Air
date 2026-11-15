"""One behaviour suite run against both trace stores, and the step recorder on each."""
import asyncio
import datetime
import hashlib
import unittest
from contextlib import AsyncExitStack

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tta_backend.harness.results import Mode, WorkflowRef
from tta_backend.harness.trace import (
    MAX_JSON_BYTES,
    FailureCategory,
    InMemoryTraceStore,
    InvalidTrace,
    NewEntry,
    PostgresTraceStore,
    Run,
    RunClosed,
    RunNotFound,
    StepFailure,
    StepRecorder,
    TraceError,
    ensure_tables,
)

T0 = datetime.datetime(2026, 11, 11, 14, 0, 0, tzinfo=datetime.UTC)
PAUSED = ("needs_input", "waiting")
TERMINAL = ("done", "failed", "cancelled", "insufficient_evidence")


def rid(tag: str) -> str:
    """A well-formed run id that is stable for a readable tag."""
    return "run_" + hashlib.md5(tag.encode()).hexdigest()


RUN_A = rid("a")


def at(minutes: int) -> datetime.datetime:
    return T0 + datetime.timedelta(minutes=minutes)


def run(run_id=RUN_A, user="alice", conv="c1", started=0, **over) -> Run:
    over.setdefault("mode", Mode.exploratory())
    return Run(id=run_id, user_id=user, conversation_id=conv, started=at(started), **over)


def entry(run_id=RUN_A, **over) -> NewEntry:
    base = {"run_id": run_id, "kind": "use", "capability": "ops.diff", "status": "done",
            "started": at(1), "seconds": 0.5, "params": {"a": "g1"}, "result": "g2"}
    return NewEntry(**{**base, **over})


class FakeTime:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return T0 + datetime.timedelta(seconds=self.t)

    def monotonic(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TraceStoreContract:
    """Mixed into a TestCase that provides ``self.store``."""

    store: InMemoryTraceStore | PostgresTraceStore

    async def test_run_round_trip(self):
        await self.store.start_run(run())
        got = await self.store.get_run("alice", RUN_A)
        self.assertEqual(got, run())
        self.assertEqual(got.status, "running")

    async def test_mode_with_children_round_trips(self):
        children = (WorkflowRef(name="event-vs-baseline", version=1),
                    WorkflowRef(name="map", version=4))
        mode = Mode.validated("ozone-event", 3, children)
        await self.store.start_run(run(mode=mode))
        got = await self.store.get_run("alice", RUN_A)
        self.assertEqual(got.mode, mode)
        self.assertEqual(got.mode.child_workflows, children)
        [listed] = await self.store.list_runs("alice", "c1")
        self.assertEqual(listed.mode, mode)

    async def test_finish_run(self):
        await self.store.start_run(run())
        done = await self.store.finish_run("alice", RUN_A, "done", at(5))
        self.assertEqual((done.status, done.ended), ("done", at(5)))
        self.assertEqual(await self.store.get_run("alice", RUN_A), done)

    async def test_cannot_finish_as_running(self):
        await self.store.start_run(run())
        with self.assertRaises(InvalidTrace):
            await self.store.finish_run("alice", RUN_A, "running", at(5))

    async def test_run_must_start_running(self):
        with self.assertRaises(InvalidTrace):
            await self.store.start_run(run(status="done"))

    async def test_finished_run_refuses_every_write(self):
        for status in TERMINAL:
            with self.subTest(status):
                run_id = rid(f"finished-{status}")
                await self.store.start_run(run(run_id))
                await self.store.append("alice", entry(run_id))
                await self.store.finish_run("alice", run_id, status, at(5))
                writes = {
                    "append": lambda r: self.store.append("alice", entry(r)),
                    "append reserved": lambda r: self.store.append("alice", entry(r), seq=1),
                    "reserve": lambda r: self.store.reserve_seq("alice", r),
                    "finish": lambda r: self.store.finish_run("alice", r, "failed", at(6)),
                    "resume": lambda r: self.store.resume_run("alice", r),
                }
                for name, write in writes.items():
                    with self.subTest(name), self.assertRaises(RunClosed):
                        await write(run_id)
                got = await self.store.get_run("alice", run_id)
                self.assertEqual((got.status, got.ended), (status, at(5)))
                self.assertEqual(len(await self.store.entries("alice", run_id)), 1)

    async def test_paused_run_takes_no_entries_until_resumed(self):
        for status in PAUSED:
            with self.subTest(status):
                run_id = rid(f"paused-{status}")
                await self.store.start_run(run(run_id))
                await self.store.append(
                    "alice", entry(run_id, kind="ask", capability=None, status="waiting")
                )
                paused = await self.store.finish_run("alice", run_id, status, at(5))
                self.assertEqual((paused.status, paused.ended), (status, at(5)))
                with self.assertRaises(RunClosed):
                    await self.store.append("alice", entry(run_id))
                with self.assertRaises(RunClosed):
                    await self.store.reserve_seq("alice", run_id)
                resumed = await self.store.resume_run("alice", run_id)
                self.assertEqual((resumed.status, resumed.ended), ("running", None))
                self.assertEqual(await self.store.get_run("alice", run_id), resumed)
                self.assertEqual((await self.store.append("alice", entry(run_id))).seq, 2)

    async def test_resume_is_only_from_a_pause(self):
        await self.store.start_run(run())
        with self.assertRaises(RunClosed):
            await self.store.resume_run("alice", RUN_A)
        for status in PAUSED:
            with self.subTest(status):
                await self.store.finish_run("alice", RUN_A, status, at(5))
                await self.store.resume_run("alice", RUN_A)
                with self.assertRaises(RunClosed):
                    await self.store.resume_run("alice", RUN_A)

    async def test_waiting_run_can_move_to_needs_input_or_finish(self):
        await self.store.start_run(run())
        await self.store.finish_run("alice", RUN_A, "waiting", at(5))
        done = await self.store.finish_run("alice", RUN_A, "done", at(6))
        self.assertEqual((done.status, done.ended), ("done", at(6)))

    async def test_paused_run_can_be_cancelled(self):
        await self.store.start_run(run())
        await self.store.finish_run("alice", RUN_A, "needs_input", at(5))
        cancelled = await self.store.finish_run("alice", RUN_A, "cancelled", at(9))
        self.assertEqual((cancelled.status, cancelled.ended), ("cancelled", at(9)))

    async def test_duplicate_run_id(self):
        await self.store.start_run(run())
        with self.assertRaises(TraceError):
            await self.store.start_run(run())

    async def test_same_run_id_under_two_users_is_two_runs(self):
        await self.store.start_run(run())
        await self.store.start_run(run(user="bob", conv="c9"))
        await self.store.append("alice", entry(step="alice-step"))
        bob_entry = await self.store.append("bob", entry(step="bob-step"))
        self.assertEqual(bob_entry.seq, 1)
        await self.store.finish_run("bob", RUN_A, "done", at(3))
        alice = await self.store.get_run("alice", RUN_A)
        bob = await self.store.get_run("bob", RUN_A)
        self.assertEqual((alice.status, alice.conversation_id), ("running", "c1"))
        self.assertEqual((bob.status, bob.conversation_id), ("done", "c9"))
        self.assertEqual([e.step for e in await self.store.entries("alice", RUN_A)], ["alice-step"])
        self.assertEqual([e.step for e in await self.store.entries("bob", RUN_A)], ["bob-step"])

    async def test_parent_run_must_exist_for_the_same_user(self):
        child = rid("child")
        with self.assertRaises(RunNotFound):
            await self.store.start_run(run(child, parent_run_id=RUN_A))
        await self.store.start_run(run())
        with self.assertRaises(RunNotFound):
            await self.store.start_run(run(child, user="bob", parent_run_id=RUN_A))
        await self.store.start_run(run(child, parent_run_id=RUN_A))
        self.assertEqual((await self.store.get_run("alice", child)).parent_run_id, RUN_A)

    async def test_entries_get_increasing_seq_and_read_in_order(self):
        await self.store.start_run(run())
        seqs = [(await self.store.append("alice", entry(step=f"s{i}"))).seq for i in range(3)]
        self.assertEqual(seqs, [1, 2, 3])
        got = await self.store.entries("alice", RUN_A)
        self.assertEqual([e.seq for e in got], seqs)
        self.assertEqual([e.step for e in got], ["s0", "s1", "s2"])

    async def test_entry_round_trips_every_field(self):
        await self.store.start_run(run())
        parent = await self.store.append("alice", entry(kind="workflow", capability="child"))
        sent = entry(step="diff", parent_seq=parent.seq, status="failed",
                     failure_category=FailureCategory.NO_DATA, reason="no pixels",
                     params={"n": 1, "xs": [1.5, None], "nested": {"k": "v"}},
                     result=["g1", "g2"], seconds=1.25)
        saved = await self.store.append("alice", sent)
        [_, got] = await self.store.entries("alice", RUN_A)
        self.assertEqual(got, saved)
        self.assertEqual((got.step, got.parent_seq, got.failure_category, got.reason, got.result,
                          got.params, got.seconds, got.started, got.kind, got.capability),
                         ("diff", parent.seq, FailureCategory.NO_DATA, "no pixels", ["g1", "g2"],
                          sent.params, 1.25, sent.started, "use", "ops.diff"))

    async def test_numbers_compare_equal_after_a_round_trip(self):
        # JSONB may return 1e16 as an int and -0.0 as 0.0; values, not types, are pinned.
        await self.store.start_run(run())
        params = {"big": 1e16, "neg_zero": -0.0, "tenth": 0.1, "wide": 2**53 + 1, "neg": -7}
        await self.store.append("alice", entry(params=params, result=[1e16, -0.0]))
        [got] = await self.store.entries("alice", RUN_A)
        self.assertEqual(got.params, params)
        self.assertEqual(got.result, [1e16, 0.0])

    async def test_deep_nesting_within_the_byte_cap_round_trips(self):
        await self.store.start_run(run())
        deep: object = "x"
        for _ in range(MAX_JSON_BYTES // 2 - 10):
            deep = [deep]
        await self.store.append("alice", entry(result=deep))
        [got] = await self.store.entries("alice", RUN_A)
        self.assertEqual(got.result, deep)

    async def test_absent_result_and_step_stay_absent(self):
        await self.store.start_run(run())
        await self.store.append("alice", entry(kind="ask", capability=None, result=None))
        [got] = await self.store.entries("alice", RUN_A)
        self.assertIsNone(got.result)
        self.assertIsNone(got.step)
        self.assertIsNone(got.capability)

    async def test_nesting_and_parent_checks(self):
        await self.store.start_run(run())
        parent = await self.store.append("alice", entry(kind="workflow", capability="child"))
        child = await self.store.append("alice", entry(parent_seq=parent.seq))
        self.assertEqual(child.parent_seq, parent.seq)
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(parent_seq=99))

    async def test_parent_seq_must_come_before_the_entry(self):
        await self.store.start_run(run())
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(parent_seq=1))
        seq = await self.store.reserve_seq("alice", RUN_A)
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(parent_seq=seq), seq=seq)
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(parent_seq=seq + 1))
        self.assertEqual(await self.store.entries("alice", RUN_A), [])
        self.assertEqual((await self.store.append("alice", entry(parent_seq=seq))).seq, 2)

    async def test_refused_append_does_not_burn_a_seq(self):
        await self.store.start_run(run())
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(parent_seq=99))
        e = await self.store.append("alice", entry())
        self.assertEqual(e.seq, 1)

    async def test_nul_added_after_construction_is_refused_by_the_store(self):
        await self.store.start_run(run())
        sent = entry(params={"a": "g1"})
        sent.params["a"] = "g\x001"
        with self.assertRaisesRegex(InvalidTrace, "NUL"):
            await self.store.append("alice", sent)
        self.assertEqual(await self.store.entries("alice", RUN_A), [])

    async def test_reserved_seq_lets_children_nest_before_the_parent_is_written(self):
        await self.store.start_run(run())
        parent_seq = await self.store.reserve_seq("alice", RUN_A)
        child = await self.store.append("alice", entry(parent_seq=parent_seq))
        parent = await self.store.append("alice", entry(kind="workflow", capability="c"), seq=parent_seq)
        self.assertEqual((parent.seq, child.seq), (parent_seq, parent_seq + 1))
        got = await self.store.entries("alice", RUN_A)
        self.assertEqual([e.seq for e in got], [parent_seq, child.seq])

    async def test_reserved_seq_cannot_be_reused_or_invented(self):
        await self.store.start_run(run())
        seq = await self.store.reserve_seq("alice", RUN_A)
        await self.store.append("alice", entry(), seq=seq)
        with self.assertRaises(TraceError):
            await self.store.append("alice", entry(), seq=seq)
        with self.assertRaises(InvalidTrace):
            await self.store.append("alice", entry(), seq=50)

    async def test_unused_reservation_leaves_a_gap(self):
        await self.store.start_run(run())
        await self.store.reserve_seq("alice", RUN_A)
        e = await self.store.append("alice", entry())
        self.assertEqual(e.seq, 2)

    async def test_concurrent_appends_get_distinct_increasing_seqs(self):
        await self.store.start_run(run())
        saved = await asyncio.gather(
            *(self.store.append("alice", entry(step=f"s{i}")) for i in range(25))
        )
        seqs = [e.seq for e in saved]
        self.assertEqual(len(set(seqs)), 25)
        got = await self.store.entries("alice", RUN_A)
        self.assertEqual([e.seq for e in got], sorted(seqs))
        self.assertEqual({e.step for e in got}, {f"s{i}" for i in range(25)})

    async def test_sequences_are_per_run(self):
        run_b = rid("b")
        await self.store.start_run(run(RUN_A))
        await self.store.start_run(run(run_b))
        await self.store.append("alice", entry(RUN_A))
        first_b = await self.store.append("alice", entry(run_b))
        self.assertEqual(first_b.seq, 1)

    async def test_list_runs_is_per_conversation_and_ordered(self):
        late, early = rid("late"), rid("early")
        await self.store.start_run(run(late, started=9))
        await self.store.start_run(run(early, started=1))
        await self.store.start_run(run(rid("other"), conv="c2", started=2))
        got = await self.store.list_runs("alice", "c1")
        self.assertEqual([r.id for r in got], [early, late])
        self.assertEqual(await self.store.list_runs("alice", "nope"), [])

    async def test_other_users_run_is_indistinguishable_from_missing(self):
        missing_id = rid("missing")
        await self.store.start_run(run())
        await self.store.append("alice", entry())
        calls = {
            "get_run": lambda u, r: self.store.get_run(u, r),
            "entries": lambda u, r: self.store.entries(u, r),
            "finish": lambda u, r: self.store.finish_run(u, r, "done", at(2)),
            "resume": lambda u, r: self.store.resume_run(u, r),
            "reserve": lambda u, r: self.store.reserve_seq(u, r),
            "append": lambda u, r: self.store.append(u, entry(r)),
        }
        for name, call in calls.items():
            with self.subTest(name):
                with self.assertRaises(RunNotFound) as foreign:
                    await call("bob", RUN_A)
                with self.assertRaises(RunNotFound) as missing:
                    await call("bob", missing_id)
                self.assertEqual(str(foreign.exception).replace(RUN_A, "X"),
                                 str(missing.exception).replace(missing_id, "X"))
        self.assertEqual(await self.store.list_runs("bob", "c1"), [])
        self.assertEqual((await self.store.get_run("alice", RUN_A)).status, "running")
        self.assertEqual(len(await self.store.entries("alice", RUN_A)), 1)

    async def test_stored_data_is_not_aliased(self):
        await self.store.start_run(run())
        params = {"xs": ["g1"]}
        saved = await self.store.append("alice", entry(params=params))
        params["xs"].append("tampered")
        saved.params["xs"].append("tampered")
        [got] = await self.store.entries("alice", RUN_A)
        self.assertEqual(got.params, {"xs": ["g1"]})


class StepRecorderContract:
    """The recorder over ``self.store``; mixed in beside TraceStoreContract."""

    store: InMemoryTraceStore | PostgresTraceStore

    async def recorder(self) -> StepRecorder:
        self.clock = FakeTime()
        await self.store.start_run(run())
        return StepRecorder(self.store, "alice", RUN_A,
                            now=self.clock.now, monotonic=self.clock.monotonic)

    async def recorded(self):
        return await self.store.entries("alice", RUN_A)

    async def test_step_success_records_time_and_result(self):
        rec = await self.recorder()
        async with rec.step(kind="use", capability="ops.diff", params={"a": "g1"}, step="diff") as s:
            self.clock.advance(1.8)
            s.result = "grid_63"
        [e] = await self.recorded()
        self.assertEqual((e.status, e.result, e.seconds, e.started, e.step, e.params),
                         ("done", "grid_63", 1.8, T0, "diff", {"a": "g1"}))

    async def test_step_exception_is_recorded_by_type_name_only(self):
        rec = await self.recorder()
        with self.assertRaises(RuntimeError):
            async with rec.step(kind="use", capability="ops.diff"):
                self.clock.advance(0.5)
                raise RuntimeError("GET https://host/x?token=secret failed")
        [e] = await self.recorded()
        self.assertEqual((e.status, e.failure_category, e.seconds, e.reason),
                         ("failed", FailureCategory.INTERNAL_ERROR, 0.5, "RuntimeError"))

    async def test_step_failure_keeps_its_category_and_reason(self):
        rec = await self.recorder()
        with self.assertRaises(StepFailure):
            async with rec.step(kind="use", capability="ops.fetch"):
                raise StepFailure(FailureCategory.NO_DATA, "no valid pixels")
        [e] = await self.recorded()
        self.assertEqual((e.status, e.failure_category, e.reason),
                         ("failed", FailureCategory.NO_DATA, "no valid pixels"))

    async def test_step_failure_result_and_eviction_category_are_stored(self):
        rec = await self.recorder()
        with self.assertRaises(StepFailure):
            async with rec.step(kind="use", capability="ops.mask"):
                raise StepFailure(FailureCategory.INPUT_EVICTED, "data gone",
                                  result={"evicted": ["grid_1"]})
        [e] = await self.recorded()
        self.assertEqual((e.status, e.failure_category, e.result),
                         ("failed", FailureCategory.INPUT_EVICTED, {"evicted": ["grid_1"]}))

    async def test_step_failure_with_an_unstorable_result_becomes_an_internal_error(self):
        rec = await self.recorder()
        with self.assertRaises(StepFailure) as caught:
            async with rec.step(kind="use", capability="ops.mask"):
                raise StepFailure(FailureCategory.NO_DATA, "short", result="x" * (MAX_JSON_BYTES + 1))
        self.assertEqual(caught.exception.category, FailureCategory.INTERNAL_ERROR)
        [e] = await self.recorded()
        self.assertEqual((e.failure_category, e.reason, e.result),
                         (FailureCategory.INTERNAL_ERROR, "the step's outcome could not be recorded", None))

    async def test_long_step_failure_reason_is_truncated_not_lost(self):
        rec = await self.recorder()
        with self.assertRaises(StepFailure):
            async with rec.step(kind="use", capability="x"):
                raise StepFailure(FailureCategory.NO_DATA, "y" * 5000)
        [e] = await self.recorded()
        self.assertEqual(e.reason, "y" * 1000)

    async def test_step_status_can_be_set_by_the_step(self):
        rec = await self.recorder()
        async with rec.step(kind="ask") as s:
            s.status = "waiting"
            s.reason = "which area?"
        [e] = await self.recorded()
        self.assertEqual((e.status, e.reason), ("waiting", "which area?"))

    async def test_child_steps_nest_under_the_parent_step(self):
        rec = await self.recorder()
        async with rec.step(kind="workflow", capability="event-vs-baseline") as parent:
            async with rec.step(kind="use", capability="ops.diff", parent_seq=parent.seq) as child:
                child.result = "grid_1"
            with self.assertRaises(ValueError):
                async with rec.step(kind="use", capability="ops.mask", parent_seq=parent.seq):
                    raise ValueError("bad")
        es = await self.recorded()
        self.assertEqual([(e.seq, e.parent_seq, e.capability, e.status) for e in es],
                         [(1, None, "event-vs-baseline", "done"), (2, 1, "ops.diff", "done"),
                          (3, 1, "ops.mask", "failed")])

    async def test_bad_params_fail_before_the_body_runs(self):
        rec = await self.recorder()
        ran = False
        with self.assertRaises(InvalidTrace):
            async with rec.step(kind="use", capability="x", params={"rows": b"raw"}):
                ran = True
        self.assertFalse(ran)
        self.assertEqual(await self.recorded(), [])
        self.assertEqual((await self.store.append("alice", entry())).seq, 1)

    async def test_unstorable_outcome_is_replaced_by_a_failed_entry(self):
        outcomes = {
            "not JSON": {"result": b"raw"},
            "too large": {"result": "x" * (MAX_JSON_BYTES + 1)},
            "failed without a category": {"status": "failed"},
        }
        rec = await self.recorder()
        for name, outcome in outcomes.items():
            with self.subTest(name):
                with self.assertRaises(InvalidTrace):
                    async with rec.step(kind="use", capability="x", step=name) as s:
                        for attr, value in outcome.items():
                            setattr(s, attr, value)
        es = await self.recorded()
        self.assertEqual(
            [(e.step, e.status, e.failure_category, e.reason, e.result) for e in es],
            [(name, "failed", FailureCategory.INTERNAL_ERROR,
              "the step's outcome could not be recorded", None) for name in outcomes],
        )

    async def test_cancelled_step_is_recorded_and_cancellation_propagates(self):
        rec = await self.recorder()
        entered = asyncio.Event()

        async def body():
            async with rec.step(kind="use", capability="sources.fetch"):
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(body())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        [e] = await self.recorded()
        self.assertEqual((e.status, e.failure_category, e.reason),
                         ("failed", FailureCategory.CANCELLED, "the step was cancelled"))


class InMemoryTraceStoreTests(
    TraceStoreContract, StepRecorderContract, unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self):
        self.store = InMemoryTraceStore()


class PostgresTraceStoreTests(TraceStoreContract, StepRecorderContract, PostgresTestCase):
    async def asyncSetUp(self):
        self._stack = AsyncExitStack()
        self.connect = await self._stack.enter_async_context(isolated_schema())
        await ensure_tables(self.connect)
        self.store = PostgresTraceStore(self.connect)

    async def asyncTearDown(self):
        await self._stack.aclose()

    async def test_ensure_tables_is_safe_to_call_twice_and_keeps_data(self):
        await self.store.start_run(run())
        await ensure_tables(self.connect)
        await ensure_tables(self.connect)
        self.assertEqual((await self.store.get_run("alice", RUN_A)).id, RUN_A)

    async def test_tables_are_prefixed(self):
        async with self.connect() as conn:
            cur = await conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
            )
            names = {r[0] for r in await cur.fetchall()}
        self.assertEqual(names, {"harness_runs", "harness_trace_entries"})

    async def test_many_concurrent_appends_use_every_seq_once(self):
        await self.store.start_run(run())
        saved = await asyncio.gather(
            *(self.store.append("alice", entry(step=f"s{i}")) for i in range(60))
        )
        self.assertEqual(sorted(e.seq for e in saved), list(range(1, 61)))


if __name__ == "__main__":
    unittest.main()
