"""T64 D4/D5/D8/D9/D10: running one dispatch call's requests -- capped
concurrency, an all-or-nothing per-turn budget, dedupe by canonical request,
per-item failure isolation, results in request order."""
import asyncio
import unittest


def _sat(location, **extra):
    from tta_backend.models import SatelliteRequest

    return SatelliteRequest.model_validate(
        {"intent": "visualize", "figure_type": "map", "variable": "NO2", "location": location, **extra}
    )


def _ground(location):
    from tta_backend.models import GroundRequest

    return GroundRequest.model_validate({"intent": "readings", "pollutant": "NO2", "location": location})


class FakeRunner:
    """run_one stand-in: records every run and how many overlap."""

    def __init__(self, delays=None, fail=()):
        self.delays = delays or {}
        self.fail = set(fail)
        self.runs = []
        self.active = {"satellite": 0, "ground": 0}
        self.peak = {"satellite": 0, "ground": 0, "total": 0}

    async def __call__(self, request):
        from tta_backend.models import AgentResult

        self.runs.append(request.location)
        self.active[request.agent] += 1
        self.peak[request.agent] = max(self.peak[request.agent], self.active[request.agent])
        self.peak["total"] = max(self.peak["total"], sum(self.active.values()))
        try:
            await asyncio.sleep(self.delays.get(request.location, 0.01))
            if request.location in self.fail:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return AgentResult(text=f"answer for {request.location}")
        finally:
            self.active[request.agent] -= 1


class DispatchExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tta_backend.utils.streaming import _call_budget

        # One chat turn, as stream_response binds it.
        self._token = _call_budget.set({})

    async def asyncTearDown(self):
        from tta_backend.utils.streaming import _call_budget

        _call_budget.reset(self._token)

    async def test_results_come_back_in_request_order_whatever_finishes_first(self):
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner(delays={"New Jersey": 0.05, "New York": 0.0})

        batch = await execute_dispatch([_sat("New Jersey"), _sat("New York")], runner)

        self.assertEqual([i.index for i in batch.results], [0, 1])
        self.assertEqual([i.status for i in batch.results], ["ok", "ok"])
        self.assertEqual([i.result.text for i in batch.results], ["answer for New Jersey", "answer for New York"])
        self.assertEqual(runner.runs, ["New Jersey", "New York"])

    async def test_at_most_two_satellite_runs_overlap(self):
        """The satellite path is the memory-dangerous one (render-path OOM)."""
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner()

        batch = await execute_dispatch([_sat(f"region {i}") for i in range(6)], runner)

        self.assertEqual(runner.peak["satellite"], 2)
        self.assertEqual([i.status for i in batch.results], ["ok"] * 6)

    async def test_at_most_four_runs_overlap_across_both_agents(self):
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner()

        await execute_dispatch([_sat("NJ"), _sat("NY"), *(_ground(f"site {i}") for i in range(4))], runner)

        self.assertEqual(runner.peak["total"], 4)
        self.assertEqual(runner.peak["satellite"], 2)

    async def test_a_rate_limited_item_is_its_own_error_and_its_siblings_still_finish(self):
        """Phase 0 F7: on the free tier a 429 inside a batch is likely; it
        must cost that item, not the batch."""
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner(fail={"New York"})

        batch = await execute_dispatch([_sat("New Jersey"), _sat("New York"), _ground("Newark")], runner)

        self.assertEqual([i.status for i in batch.results], ["ok", "error", "ok"])
        failed = batch.results[1].result.text
        self.assertIn("rate", failed.lower())
        self.assertNotIn("RESOURCE_EXHAUSTED", failed)

    async def test_a_list_that_would_overrun_the_turns_budget_is_rejected_whole(self):
        """D8: six sub-agent runs per turn. A half-run list would hand the
        model a half-answer to reason about."""
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        await execute_dispatch([_sat(f"first {i}") for i in range(4)], runner)
        begin_model_step()

        second = await execute_dispatch([_ground(f"second {i}") for i in range(3)], runner)

        self.assertEqual([i.status for i in second.results], ["rejected"] * 3)
        self.assertIsNone(second.results[0].result)
        self.assertIn("6", second.results[0].reason)
        self.assertEqual(len(runner.runs), 4)

    async def test_a_third_dispatch_round_is_rejected_even_with_budget_left(self):
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        await execute_dispatch([_sat("NJ")], runner)
        begin_model_step()
        await execute_dispatch([_ground("Newark")], runner)
        begin_model_step()

        third = await execute_dispatch([_ground("Trenton")], runner)

        self.assertEqual([i.status for i in third.results], ["rejected"])
        self.assertIn("round", third.results[0].reason)
        self.assertEqual(runner.runs, ["NJ", "Newark"])

    async def test_a_request_repeated_in_one_list_runs_once(self):
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner()

        batch = await execute_dispatch([_sat("NJ"), _ground("Newark"), _sat("NJ")], runner)

        self.assertEqual(runner.runs, ["NJ", "Newark"])
        self.assertEqual([i.status for i in batch.results], ["ok", "ok", "duplicate"])
        self.assertEqual(batch.results[2].result.text, "answer for NJ")

    async def test_a_repeat_in_a_later_round_is_served_from_the_first_run_at_no_cost(self):
        """Replaces the old "[STOP] already called" prose: the model gets the
        answer it already had, so there is nothing to argue with. Rewording
        prior_findings does not make it a new request."""
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        await execute_dispatch([_sat(f"region {i}", prior_findings="peak at 40N") for i in range(6)], runner)
        begin_model_step()

        repeat = await execute_dispatch([_sat("region 0", prior_findings="the peak was near 40N")], runner)

        self.assertEqual([i.status for i in repeat.results], ["duplicate"])
        self.assertEqual(repeat.results[0].result.text, "answer for region 0")
        self.assertEqual(len(runner.runs), 6)

    async def test_two_dispatch_calls_in_one_model_message_share_one_run_of_a_common_request(self):
        """D5: ToolNode runs parallel tool calls as sibling Tasks, each with a
        copied context; the turn's state must still be one object."""
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner(delays={"NJ": 0.05})

        first, second = await asyncio.gather(
            asyncio.create_task(execute_dispatch([_sat("NJ")], runner)),
            asyncio.create_task(execute_dispatch([_sat("NJ"), _ground("Newark")], runner)),
        )

        self.assertEqual(sorted(runner.runs), ["NJ", "Newark"])
        self.assertEqual([i.status for i in first.results], ["ok"])
        self.assertEqual([i.status for i in second.results], ["duplicate", "ok"])
        self.assertEqual(second.results[0].result.text, "answer for NJ")


class DispatchLoggingTests(unittest.IsolatedAsyncioTestCase):
    """Live 2026-10-16: a second-round ground request made no AQS call, and
    nothing in the logs said whether it ran, was deduped or was rejected."""

    LOGGER = "tta_backend.services.dispatch_executor"

    async def asyncSetUp(self):
        from tta_backend.utils.streaming import _call_budget

        self._token = _call_budget.set({})

    async def asyncTearDown(self):
        from tta_backend.utils.streaming import _call_budget

        _call_budget.reset(self._token)

    @staticmethod
    def _events(captured, event):
        return [r for r in captured.records if getattr(r, "_event", None) == event]

    async def test_each_dispatch_call_logs_its_items_outcomes_and_the_turns_budget(self):
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner(fail={"NY"})
        with self.assertLogs(self.LOGGER, level="INFO") as captured:
            await execute_dispatch([_sat("NJ"), _sat("NY"), _sat("NJ")], runner)
            begin_model_step()
            await execute_dispatch([_ground("Newark"), _sat("NJ")], runner)

        first, second = self._events(captured, "dispatch_batch")
        self.assertEqual(first._statuses, ["ok", "error", "duplicate"])
        self.assertEqual(first._agents, ["satellite", "satellite", "satellite"])
        self.assertEqual((first._round, first._runs_used, first._max_runs), (1, 2, 6))
        self.assertEqual(second._statuses, ["ok", "duplicate"])
        self.assertEqual((second._round, second._runs_used, second._max_rounds), (2, 3, 2))

    async def test_each_run_logs_its_wait_for_a_slot_and_its_own_duration(self):
        from tta_backend.services.dispatch_executor import execute_dispatch

        runner = FakeRunner(delays={f"r{i}": 0.05 for i in range(3)})
        with self.assertLogs(self.LOGGER, level="INFO") as captured:
            await execute_dispatch([_sat("r0"), _sat("r1"), _sat("r2")], runner)

        items = self._events(captured, "dispatch_item")
        self.assertEqual(len(items), 3)
        self.assertTrue(all(i._agent == "satellite" and i._status == "ok" for i in items))
        self.assertTrue(all(i._run_seconds >= 0.04 for i in items))
        # Two satellite slots: the third run waited for one to free up.
        self.assertGreaterEqual(max(i._wait_seconds for i in items), 0.04)

    async def test_a_rejected_dispatch_is_logged_as_a_warning_with_its_reason(self):
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        await execute_dispatch([_sat("a")], runner)
        begin_model_step()
        await execute_dispatch([_sat("b")], runner)
        begin_model_step()
        with self.assertLogs(self.LOGGER, level="WARNING") as captured:
            await execute_dispatch([_sat("c")], runner)

        (batch,) = self._events(captured, "dispatch_batch")
        self.assertEqual(batch.levelname, "WARNING")
        self.assertEqual(batch._statuses, ["rejected"])
        self.assertIn("round limit", batch._reason)


class DispatchStepTests(unittest.IsolatedAsyncioTestCase):
    """A round is one supervisor model step, not one dispatch call; and the
    supervisor loses the dispatch tool once nothing more can come of it."""

    async def asyncSetUp(self):
        from tta_backend.utils.streaming import _call_budget

        self._token = _call_budget.set({})

    async def asyncTearDown(self):
        from tta_backend.utils.streaming import _call_budget

        _call_budget.reset(self._token)

    async def test_sibling_calls_from_one_model_step_share_one_round(self):
        """ToolNode runs two dispatch calls from one model message together;
        counting each as a round left no round for the dependent follow-up."""
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        begin_model_step()
        await asyncio.gather(
            execute_dispatch([_sat("NJ")], runner),
            execute_dispatch([_ground("Newark")], runner),
        )
        self.assertTrue(begin_model_step())

        follow_up = await execute_dispatch([_ground("Camden")], runner)

        self.assertEqual([i.status for i in follow_up.results], ["ok"])

    async def test_dispatch_closes_once_both_rounds_are_used(self):
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        self.assertTrue(begin_model_step())
        await execute_dispatch([_sat("NJ")], runner)
        self.assertTrue(begin_model_step())
        await execute_dispatch([_ground("Newark")], runner)

        self.assertFalse(begin_model_step())

    async def test_dispatch_closes_after_a_step_that_ran_nothing_new(self):
        """A duplicate-only or rejected call changes nothing, so a model that
        repeats it would loop until the recursion limit."""
        from tta_backend.services.dispatch_executor import begin_model_step, execute_dispatch

        runner = FakeRunner()
        begin_model_step()
        await execute_dispatch([_sat("NJ")], runner)
        self.assertTrue(begin_model_step())
        await execute_dispatch([_sat("NJ")], runner)

        self.assertFalse(begin_model_step())
        self.assertEqual(runner.runs, ["NJ"])

    async def test_a_step_without_a_dispatch_call_does_not_close_it(self):
        from tta_backend.services.dispatch_executor import begin_model_step

        self.assertTrue(begin_model_step())
        self.assertTrue(begin_model_step())


if __name__ == "__main__":
    unittest.main()
