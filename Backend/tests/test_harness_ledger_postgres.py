"""The session ledger over the real Postgres stores: the same text as in memory, per user."""
import dataclasses

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_ledger import GROUND, MAP, LedgerCase
from tta_backend.harness.storage import ensure_tables, postgres_stores


class LedgerPostgresTests(PostgresTestCase, LedgerCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        schema = isolated_schema()
        connect = await schema.__aenter__()
        self.addAsyncCleanup(schema.__aexit__, None, None, None)
        await ensure_tables(connect)
        stores = postgres_stores(connect)
        self.f.stores = stores
        self.f.context = dataclasses.replace(self.f.context, storage=stores)

    async def test_the_ledger_reads_runs_results_and_handles_from_postgres(self):
        done = await self.start("map", MAP)
        self.assertEqual(done.status, "done", done.reason)
        ground = await self.start("monitor-readings", GROUND)
        self.assertEqual(ground.status, "done", ground.reason)
        waiting = await self.waiting_map()
        theirs = await self.start("map", MAP, context=self.context_for("user-b"))

        self.assertEqual(await self.f.stores.results.get("user-a", done.trace_id), done)
        ledger = await self.ledger(job_states={waiting.trace_id: "ready"})
        self.assertEqual([a.run_id for a in ledger.areas], [done.trace_id, ground.trace_id, waiting.trace_id])
        self.assertEqual([p.run_ids for p in ledger.products], [(done.trace_id,)])
        [monitors] = ledger.monitors
        self.assertEqual(monitors.run_id, ground.trace_id)
        self.assertEqual([c.ref for c in ledger.charts], [*done.charts, *ground.charts])
        [open_run] = ledger.open_runs
        self.assertEqual((open_run.run_id, open_run.reason), (waiting.trace_id, waiting.reason))

        text = await self.text(job_states={waiting.trace_id: "ready"})
        self.assertIn(f"continue it with resume_run({waiting.trace_id})", text)
        self.assertIn("- obs_1 grid: ", text)
        self.assertNotIn(theirs.trace_id, text)
        self.assertEqual(await self.text(job_states={waiting.trace_id: "ready"}), text)
