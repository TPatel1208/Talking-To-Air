"""One exploration task end to end against the real Postgres stores: the trace and the handles it leaves."""
import tempfile

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_specialists_catalogue import OBJECTIVE, CatalogueWorld, satellite_script
from tta_backend.harness.specialists import ExploreTask
from tta_backend.harness.storage import ensure_tables, postgres_stores


class ExplorePostgresTests(PostgresTestCase):
    async def test_an_analyse_task_records_its_run_entries_and_handles(self):
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)
                world = CatalogueWorld(root, stores)
                [result] = await world.explore(
                    ExploreTask(objective=OBJECTIVE, bundle="analyse", places=("New Jersey",)),
                    script=satellite_script(),
                )
                self.assertEqual(result.status, "done", result.reason)

                run = await stores.traces.get_run("user-a", result.trace_id)
                self.assertEqual((run.status, run.mode.kind), ("done", "exploratory"))
                entries = await stores.traces.entries("user-a", result.trace_id)
                [entry] = [e for e in entries if e.kind == "explore"]
                self.assertEqual(entry.params["inputs"], {"handles": [], "places": ["New Jersey"]})
                self.assertEqual(
                    {k: entry.result[k] for k in ("status", "model", "model_calls", "tool_calls")},
                    {"status": "done", "model": "fake:explore", "model_calls": 4, "tool_calls": 3},
                )
                children = [e for e in entries if e.kind == "use"]
                self.assertTrue(children and all(e.parent_seq == entry.seq for e in children))
                self.assertEqual([e.capability for e in children if e.status == "done"], [
                    "sources.areas.resolve", "sources.satellite.get_data", "operations.statistic", "render.map",
                ])

                area, grid, scalar = result.handles
                stat = await stores.handles.get("user-a", scalar)
                self.assertEqual((stat.run_id, stat.operation), (result.trace_id, "operations.statistic"))
                self.assertEqual(stat.parents, (grid,))
                self.assertEqual(result.findings[0].value, stat.payload.value)
                self.assertEqual((await stores.handles.get("user-a", area)).run_id, result.trace_id)
                # Another user sees none of it.
                self.assertEqual(await stores.traces.list_runs("user-b", "conv-1"), [])
