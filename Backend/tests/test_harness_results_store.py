"""The results store in memory: upsert by run, isolation per user, reading several runs, every field kept."""
import unittest
from typing import Any

from tta_backend.harness.results import Finding, Gap, Mode, Result, WorkflowRef
from tta_backend.harness.results_store import InMemoryResultStore, ResultNotFound, ResultStore
from tta_backend.harness.trace import new_run_id


def full_result(run_id: str) -> Result:
    """A result with every field set away from its default."""
    return Result(
        status="needs_input",
        mode=Mode.validated("map", 2, children=(WorkflowRef(name="area_mean", version=1),)),
        findings=(Finding(
            name="mean", value=2.5e15, unit="molecules/cm^2", uncertainty={"sd": 1.0e14, "n": [1, 2]},
            region="New Jersey", period="2024-06-01 to 2024-06-03", source_handle="scalar_" + "a" * 32,
        ), Finding(name="days", value=3, source_handle="obs_1"), Finding(name="label", value="high", source_handle="obs_1")),
        handles=("obs_1", "grid_" + "b" * 32),
        charts=("chart-1", "chart-2"),
        disclosures=("Product: TEMPO_NO2, chosen by rule approved_preference.",),
        gaps=(Gap(step="figure", reason="The figure could not be drawn."),),
        trace_id=run_id,
        harness_release="test-release",
        reason="Step choose needs an answer before the workflow can continue.",
        question={"kind": "pick_one", "prompt": "Which level?", "options": ["surface", {"level": 2}], "items": []},
        interpretation="The column is higher in the north.",
        followups=("Compare with July?",),
    )


def done_result(run_id: str, **over: Any) -> Result:
    return Result(status="done", mode=Mode.exploratory(), trace_id=run_id, **over)


class ResultStoreContract:
    """Behaviour every results store shares; a subclass supplies ``store``."""

    store: ResultStore

    async def test_a_saved_result_reads_back_with_every_field(self):
        run_id = new_run_id()
        result = full_result(run_id)
        self.assertNotEqual(result.model_dump(), Result(status="needs_input", mode=result.mode, trace_id=run_id).model_dump())
        await self.store.put("user-a", result)
        self.assertEqual(await self.store.get("user-a", run_id), result)

    async def test_saving_again_replaces_the_result(self):
        run_id = new_run_id()
        await self.store.put("user-a", full_result(run_id))
        finished = done_result(run_id, handles=("obs_2",), interpretation="Finished.")
        await self.store.put("user-a", finished)
        self.assertEqual(await self.store.get("user-a", run_id), finished)
        self.assertEqual(await self.store.for_runs("user-a", [run_id]), {run_id: finished})

    async def test_a_missing_result_is_not_found(self):
        with self.assertRaises(ResultNotFound):
            await self.store.get("user-a", new_run_id())
        with self.assertRaises(ResultNotFound):
            await self.store.get("user-a", "not-a-run-id")

    async def test_another_users_result_is_exactly_a_missing_one(self):
        run_id = new_run_id()
        await self.store.put("user-a", done_result(run_id))
        with self.assertRaises(ResultNotFound) as caught:
            await self.store.get("user-b", run_id)
        with self.assertRaises(ResultNotFound) as missing:
            await self.store.get("user-b", new_run_id())
        self.assertEqual(type(caught.exception), type(missing.exception))
        self.assertEqual(await self.store.for_runs("user-b", [run_id]), {})

    async def test_two_users_keep_their_own_result_for_the_same_run_id(self):
        run_id = new_run_id()
        mine, theirs = done_result(run_id, charts=("chart-a",)), done_result(run_id, charts=("chart-b",))
        await self.store.put("user-a", mine)
        await self.store.put("user-b", theirs)
        self.assertEqual(await self.store.get("user-a", run_id), mine)
        self.assertEqual(await self.store.get("user-b", run_id), theirs)

    async def test_for_runs_returns_only_the_saved_ones(self):
        first, second, missing = new_run_id(), new_run_id(), new_run_id()
        await self.store.put("user-a", done_result(first))
        await self.store.put("user-a", full_result(second))
        found = await self.store.for_runs("user-a", [missing, second, first, second])
        self.assertEqual(set(found), {first, second})
        self.assertEqual(found[second], full_result(second))
        self.assertEqual(await self.store.for_runs("user-a", []), {})

    async def test_a_read_result_is_a_copy(self):
        run_id = new_run_id()
        await self.store.put("user-a", full_result(run_id))
        read = await self.store.get("user-a", run_id)
        assert read.question is not None
        read.question["prompt"] = "changed"
        self.assertEqual((await self.store.get("user-a", run_id)).question["prompt"], "Which level?")

    async def test_a_result_holding_nul_is_refused(self):
        with self.assertRaises(ValueError):
            await self.store.put("user-a", done_result(new_run_id(), interpretation="bad\x00text"))

    async def test_a_result_needs_a_user(self):
        with self.assertRaises(ValueError):
            await self.store.put("", done_result(new_run_id()))


class InMemoryResultStoreTests(ResultStoreContract, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = InMemoryResultStore()
