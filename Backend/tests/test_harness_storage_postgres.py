"""The harness's Postgres stores, created and used together on one database."""
import datetime

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tta_backend.harness.handles import Descriptor, HandleRecord, ServicePayload
from tta_backend.harness.results import Mode
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.trace import NewEntry, Run, new_run_id

NOW = datetime.datetime(2026, 1, 15, 12, tzinfo=datetime.UTC)


class HarnessStoresPostgresTests(PostgresTestCase):
    async def test_tables_are_created_once_and_serve_both_stores(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            await ensure_tables(connect)
            stores = postgres_stores(connect)

            handle = await stores.handles.record_retrieval(HandleRecord(
                id="obs_abc", kind="grid", origin="retrieval", user_id="user-a",
                conversation_id="conv-1", descriptor=Descriptor(summary="NO2 over New Jersey"),
                created_at=NOW, payload=ServicePayload(),
            ))
            run = Run(
                id=new_run_id(), user_id="user-a", conversation_id="conv-1",
                mode=Mode.exploratory(), status="running", started=NOW,
            )
            await stores.traces.start_run(run)
            await stores.traces.append("user-a", NewEntry(
                run_id=run.id, kind="use", capability="sources.satellite.get_data",
                params={"area": "aoi_nj"}, result=handle.id, status="done", started=NOW, seconds=1.5,
            ))

            self.assertEqual((await stores.handles.get("user-a", "obs_abc")).id, "obs_abc")
            entries = await stores.traces.entries("user-a", run.id)
            self.assertEqual([entry.result for entry in entries], ["obs_abc"])
