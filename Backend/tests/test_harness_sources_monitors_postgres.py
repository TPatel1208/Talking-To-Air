"""A ground monitor capability minting its handle in the real Postgres stores."""
import json
import tempfile
from unittest.mock import AsyncMock, patch

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_sources_monitors import EMAIL, KEY, SECRET, FakeAqs, default_rows
from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.sources import monitors
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.testing import StaticCredentials, make_run_context, make_run_env


class MonitorPostgresTests(PostgresTestCase):
    async def test_find_from_coordinates_has_no_parents_and_a_summary_from_its_points_lists_them(self):
        aqs_client.reset_request_state()
        self.addCleanup(aqs_client.reset_request_state)
        fake = FakeAqs(default_rows())
        real_fetcher = aqs_client.fetcher
        with tempfile.TemporaryDirectory() as root, \
                patch.object(aqs_client, "fetcher", lambda credential: real_fetcher(credential, transport=fake.transport)), \
                patch.object(aqs_client.asyncio, "sleep", AsyncMock()):
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)
                context = make_run_context(
                    user_id="user-a", stores=stores, credentials=StaticCredentials({"epa_aqs": SECRET})
                )
                env = await make_run_env(root, context=context)

                result = await invoke(env, monitors.find, {
                    "pollutant": "o3", "lat": 34.07, "lon": -118.22, "start": "2024-06-01", "end": "2024-06-30",
                }, step="monitors")

                record = await stores.handles.get("user-a", result.handle)
                self.assertEqual((record.kind, record.origin, record.parents, record.operation, record.run_id),
                                 ("points", "derived", (), "sources.monitors.find", env.run_id))
                self.assertEqual(record.descriptor.product, "EPA AQS")
                frame = await env.resolver.open("user-a", result.handle, "points")
                self.assertEqual(list(frame["id"]), result.values["monitor_ids"])

                [entry] = await stores.traces.entries("user-a", env.run_id)
                self.assertEqual((entry.capability, entry.status, entry.step), ("sources.monitors.find", "done", "monitors"))
                self.assertEqual(entry.result["handle"], result.handle)
                stored = json.dumps([record.model_dump(mode="json"), entry.params, entry.result], default=str)
                self.assertNotIn(KEY, stored)
                self.assertNotIn(EMAIL, stored)

                summary = await invoke(env, monitors.daily_summary, {
                    "pollutant": "o3", "points": result.handle, "start": "2024-06-01", "end": "2024-06-02",
                }, step="summary")

                made = await stores.handles.get("user-a", summary.handle)
                self.assertEqual((made.kind, made.parents), ("table", (result.handle,)))
                lineage = await stores.handles.get("user-a", made.parents[0])
                self.assertEqual(lineage.operation, "sources.monitors.find")
