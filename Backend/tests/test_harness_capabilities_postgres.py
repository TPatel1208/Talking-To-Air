"""A capability call against the real Postgres handle registry and trace store."""
import tempfile

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_capabilities import (
    NOW,
    SECRET,
    count_rows,
    get_data,
    outcome_capability,
    scale,
    scripted_earthdata,
    seed_scalar,
    seed_table,
)
from tta_backend.harness.capabilities import CapabilityResult, invoke
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.testing import StaticCredentials, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure


class CapabilityPostgresTests(PostgresTestCase):
    async def test_invoke_mints_a_derived_handle_and_records_the_step(self):
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)
                env = await make_run_env(root, context=make_run_context(user_id="user-a", stores=stores))
                other = await make_run_env(root, context=make_run_context(user_id="user-b", stores=stores))
                seed = await seed_scalar(env, 2.0)
                theirs = await seed_scalar(other, 7.0)

                result = await invoke(env, scale, {"data": seed.id, "factor": 4}, step="scaled")
                with self.assertRaises(StepFailure) as caught:
                    await invoke(env, scale, {"data": theirs.id, "factor": 1})

                record = await stores.handles.get("user-a", result.handle)
                self.assertEqual((record.kind, record.parents, record.run_id), ("scalar", (seed.id,), env.run_id))
                self.assertEqual(record.payload.value, 8.0)
                self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)
                self.assertEqual(caught.exception.reason, "operations.scale: there is no handle for data.")

                entries = await stores.traces.entries("user-a", env.run_id)
                self.assertEqual([(e.kind, e.capability, e.status) for e in entries],
                                 [("use", "operations.scale", "done"), ("use", "operations.scale", "failed")])
                self.assertEqual(entries[0].result["handle"], result.handle)
                self.assertEqual(entries[0].params, {"data": seed.id, "factor": 4.0})
                self.assertEqual(entries[1].failure_category, FailureCategory.BAD_INPUT)

    async def test_failed_entries_keep_their_result_and_two_phase_calls_pair_up(self):
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)
                context = make_run_context(
                    user_id="user-a", stores=stores, credentials=StaticCredentials({"earthdata": SECRET})
                )
                env = await make_run_env(root, context=context, earthdata=scripted_earthdata())
                gone = await seed_table(env, 2)
                await env.resolver.evict("user-a", gone.id, NOW)
                stop = CapabilityResult(status="insufficient_evidence", reason="Too cloudy.", values={"days": 1})

                with self.assertRaises(StepFailure):
                    await invoke(env, count_rows, {"tables": [gone.id]})
                short = await invoke(env, outcome_capability("operations.short", stop), {})
                await invoke(env, get_data, {"variable": "no2"}, step="event")

                evicted, insufficient, waiting, done = await stores.traces.entries("user-a", env.run_id)
                self.assertEqual((evicted.failure_category, evicted.result),
                                 (FailureCategory.INPUT_EVICTED, {"evicted": [gone.id]}))
                self.assertEqual((insufficient.failure_category, insufficient.result),
                                 (FailureCategory.NO_DATA, short.recorded()))
                self.assertEqual((waiting.status, waiting.result), ("waiting", {"job": {"job_id": "job_1"}}))
                self.assertEqual((done.status, done.params["_job"]), ("done", {"job_id": "job_1"}))
