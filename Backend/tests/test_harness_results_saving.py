"""Workflow runs and explorations save every result they return, paused or finished, before the run is finished."""
import asyncio
import dataclasses
from typing import Any

from tests.test_harness_specialists import ExploreCase, call_on_area, final, on_last_scalar, task
from tests.test_harness_workflow_runner import FETCH, HOLD, LINEAR, SECRET, RunnerCase, make, spec
from tta_backend.harness.results import Result
from tta_backend.harness.results_store import InMemoryResultStore, ResultNotFound
from tta_backend.harness.storage import Stores
from tta_backend.harness.testing import make_run_context, tool_call
from tta_backend.harness.trace import RunClosed
from tta_backend.harness.workflows.runner import INTERNAL_REASON


class RecordingResults(InMemoryResultStore):
    """Results kept in memory, noting the run's trace status at each save."""

    def __init__(self, stores: Stores, *, broken: bool = False) -> None:
        super().__init__()
        self.stores = stores
        self.broken = broken
        self.seen: list[tuple[str, str]] = []

    async def put(self, user_id: str, result: Result) -> None:
        run = await self.stores.traces.get_run(user_id, result.trace_id)
        self.seen.append((result.status, run.status))
        if self.broken:
            raise RuntimeError("the results store is down " + SECRET)
        await super().put(user_id, result)


class WorkflowSavingTests(RunnerCase):
    def watched(self, *specs: dict[str, Any], broken: bool = False) -> tuple[Any, RecordingResults]:
        f = self.fixture(*specs)
        recording = RecordingResults(f.stores, broken=broken)
        f.stores.results = recording
        return f, recording

    async def saved(self, f: Any, run_id: str) -> Result:
        return await f.stores.results.get("user-a", run_id)

    async def test_a_finished_run_saves_its_result_before_the_run_is_finished(self):
        f, recording = self.watched(LINEAR)
        result = await f.start({"area": "NJ"})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(await self.saved(f, result.trace_id), result)
        self.assertEqual(recording.seen, [("done", "running")])

    async def test_a_waiting_run_is_saved_and_its_result_replaced_when_it_finishes(self):
        f, recording = self.watched(FETCH)
        self.script.statuses["job_1"] = ["running"]
        waiting = await f.start({})
        self.assertEqual(waiting.status, "waiting")
        self.assertEqual(await self.saved(f, waiting.trace_id), waiting)

        done = await f.runner.resume(await f.env(waiting.trace_id))
        self.assertEqual(done.status, "done", done.reason)
        self.assertEqual(await self.saved(f, waiting.trace_id), done)
        self.assertEqual(recording.seen, [("waiting", "running"), ("done", "running")])

    async def test_a_run_waiting_on_an_answer_is_saved_with_its_question(self):
        f, _ = self.watched(spec(
            [make("base"), {"id": "choose", "use": "ops.ask", "with": {"label": "choose"}}], {"data": ["$base"]},
            inputs={},
        ))
        result = await f.start({})
        self.assertEqual(result.status, "needs_input")
        saved = await self.saved(f, result.trace_id)
        self.assertEqual(saved, result)
        assert saved.question is not None
        self.assertEqual(saved.question["options"], ["surface", "column"])

    async def test_a_run_ending_before_any_step_is_saved(self):
        f, _ = self.watched(LINEAR)
        result = await f.start({"area": "Atlantis"})
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(await self.saved(f, result.trace_id), result)

    async def test_an_internal_error_saves_the_failed_result(self):
        f, recording = self.watched(LINEAR)

        async def broken(context: Any, run_id: str) -> Any:
            raise RuntimeError("no environment")

        result = await f.runner.start(f.context, broken, "fake", 1, {})
        self.assertEqual((result.status, result.reason), ("failed", INTERNAL_REASON))
        self.assertEqual(await self.saved(f, result.trace_id), result)
        self.assertEqual(recording.seen, [("failed", "running")])

    async def test_cancelling_a_waiting_run_saves_it_as_cancelled(self):
        f, _ = self.watched(FETCH)
        self.script.statuses["job_1"] = ["running"]
        waiting = await f.start({})
        await f.runner.cancel(await f.env(waiting.trace_id))
        saved = await self.saved(f, waiting.trace_id)
        self.assertEqual((saved.status, saved.reason), ("cancelled", "The run was cancelled."))
        self.assertEqual(saved.mode, waiting.mode)

    async def test_a_run_cancelled_while_it_runs_saves_the_cancelled_result_it_returns(self):
        f, _ = self.watched(HOLD)
        task_ = asyncio.create_task(f.start({}))
        await self.script.holding.wait()
        [run] = await f.stores.traces.list_runs("user-a", "conv-1")
        await f.runner.cancel(await f.env(run.id))
        self.script.release.set()
        result = await task_
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(await self.saved(f, run.id), result)

    async def test_a_cancelled_task_saves_a_cancelled_result(self):
        f, _ = self.watched(HOLD)
        task_ = asyncio.create_task(f.start({}))
        await self.script.holding.wait()
        [run] = await f.stores.traces.list_runs("user-a", "conv-1")
        task_.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task_
        self.assertEqual((await self.saved(f, run.id)).status, "cancelled")
        self.assertEqual(await f.run_status(run.id), "cancelled")

    async def test_a_cancel_that_loses_to_the_finish_leaves_the_finished_result(self):
        f, _ = self.watched(LINEAR)
        done = await f.start({"area": "NJ"})
        traces = f.stores.traces

        class Stale:
            """Reports the run as still running, as a read made just before it finished would."""

            def __getattr__(self, name: str) -> Any:
                return getattr(traces, name)

            async def get_run(self, user_id: str, run_id: str) -> Any:
                return dataclasses.replace(await traces.get_run(user_id, run_id), status="running")

        f.stores.traces = Stale()  # type: ignore[assignment]
        with self.assertRaises(RunClosed):
            await f.runner.cancel(await f.env(done.trace_id))
        f.stores.traces = traces
        self.assertEqual(await f.run_status(done.trace_id), "done")
        self.assertEqual(await self.saved(f, done.trace_id), done)

    async def test_a_cancel_landing_between_the_save_and_the_finish_leaves_a_cancelled_result(self):
        f, _ = self.watched(LINEAR)
        results = f.stores.results
        cancels: list[Any] = []

        class CancelFirst:
            """Runs a whole cancel, finish and save, just before the run's own done result is stored."""

            def __getattr__(self, name: str) -> Any:
                return getattr(results, name)

            async def put(self, user_id: str, result: Result) -> None:
                if result.status == "done" and not cancels:
                    cancels.append(await f.runner.cancel(await f.env(result.trace_id)))
                await results.put(user_id, result)

        f.stores.results = CancelFirst()  # type: ignore[assignment]
        result = await f.start({"area": "NJ"})
        [cancelled] = cancels
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual((result.status, result.reason), ("cancelled", "The run was cancelled."))
        self.assertEqual(await f.run_status(result.trace_id), "cancelled")
        saved = await results.get("user-a", result.trace_id)
        self.assertEqual(saved, result)
        self.assertEqual(saved.handles, result.handles)

    async def test_a_result_that_cannot_be_saved_is_logged_and_still_returned(self):
        f, _ = self.watched(LINEAR, broken=True)
        with self.assertLogs("tta_backend.harness.workflows.runner", level="ERROR") as logs:
            result = await f.start({"area": "NJ"})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(await f.run_status(result.trace_id), "done")
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn(SECRET, "\n".join(logs.output))

    async def test_results_are_saved_under_the_runs_user_only(self):
        f, _ = self.watched(LINEAR)
        result = await f.start({"area": "NJ"})
        with self.assertRaises(ResultNotFound):
            await f.stores.results.get("user-b", result.trace_id)


class ExplorationSavingTests(ExploreCase):
    def watch(self, *, broken: bool = False) -> RecordingResults:
        recording = RecordingResults(self.stores, broken=broken)
        self.stores.results = recording
        return recording

    async def test_a_finished_exploration_saves_its_result_before_the_run_is_finished(self):
        recording = self.watch()
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"), final("Mean is 2.5.", findings=on_last_scalar()),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(await self.stores.results.get("user-a", result.trace_id), result)
        self.assertEqual(recording.seen, [("done", "running")])

    async def test_a_waiting_exploration_is_saved(self):
        self.watch()
        [result] = await self.go(task(places=("NJ",)), script=[tool_call("sources__fake__fetch", {"label": "big"}), final()])
        self.assertEqual(result.status, "waiting")
        self.assertEqual(await self.stores.results.get("user-a", result.trace_id), result)

    async def test_an_exploration_asking_a_question_is_saved(self):
        self.watch()
        [result] = await self.go(task(places=("Atlantis",)), script=[final()])
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(await self.stores.results.get("user-a", result.trace_id), result)

    async def test_a_failed_exploration_is_saved(self):
        self.watch()
        [result] = await self.go(task(), models=None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(await self.stores.results.get("user-a", result.trace_id), result)

    async def test_each_task_saves_its_own_result(self):
        self.watch()
        results = await self.go(task("first"), task("second"), models=None)
        saved = await self.stores.results.for_runs("user-a", [r.trace_id for r in results])
        self.assertEqual([saved[r.trace_id] for r in results], results)

    async def test_a_result_that_cannot_be_saved_is_logged_and_still_returned(self):
        self.watch(broken=True)
        with self.assertLogs("tta_backend.harness.specialists.explore", level="ERROR") as logs:
            [result] = await self.go(task(places=("NJ",)), script=[final()])
        self.assertEqual(result.status, "done")
        self.assertEqual(await self.run_status(result), "done")
        self.assertNotIn(SECRET, "\n".join(logs.output))

    async def test_a_cancelled_exploration_saves_a_cancelled_result(self):
        self.watch()
        started = asyncio.Event()
        traces = self.stores.traces

        class Hold:
            def __getattr__(self, name: str) -> Any:
                return getattr(traces, name)

            async def append(self, *args: Any, **kwargs: Any) -> Any:
                # The first write waits to be cancelled; the ones that record the cancel go through.
                if not started.is_set():
                    started.set()
                    await asyncio.Event().wait()
                return await traces.append(*args, **kwargs)

        self.stores.traces = Hold()  # type: ignore[assignment]
        self.context = make_run_context(clock=self.clock, stores=self.stores)
        running = asyncio.ensure_future(self.go(task(places=("NJ",)), script=[final()]))
        await asyncio.wait_for(started.wait(), timeout=5)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        [run] = await traces.list_runs("user-a", "conv-1")
        self.assertEqual(run.status, "cancelled")
        self.assertEqual((await self.stores.results.get("user-a", run.id)).status, "cancelled")
