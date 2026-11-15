"""sources.satellite.get_data against the real Postgres handle registry and trace store."""
import tempfile
from typing import Any

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_grids import NJ
from tests.test_harness_sources_satellite import NO2, SECRET, TODAY, FakeService
from tta_backend.datasets.qa_flags import QA_VERIFIED
from tta_backend.harness.capabilities import begin, complete, invoke
from tta_backend.harness.capabilities.sources import areas, satellite
from tta_backend.harness.handles import HandleError, HandleRecord
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.testing import FakePlaces, FixedClock, StaticCredentials, make_run_context, make_run_env


class SatellitePostgresTests(PostgresTestCase):
    async def test_begin_and_complete_record_the_job_and_the_retrieval_handle(self):
        service = FakeService()
        earthdata = service.earthdata()

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)

                async def env_for(user: str) -> Any:
                    context = make_run_context(
                        user_id=user, stores=stores, clock=FixedClock(TODAY),
                        credentials=StaticCredentials({"earthdata": SECRET}),
                    )
                    return await make_run_env(
                        root, context=context, earthdata=earthdata, open_retrieval=open_retrieval,
                        places=FakePlaces({"new jersey": NJ}),
                    )

                env = await env_for("user-a")
                area = (await invoke(env, areas.resolve, {"place": "New Jersey"}, step="area")).handle
                params = {"quantity": NO2, "area": area, "dates": "2024-06-01", "date_role": "subject"}

                job = await begin(env, satellite.get_data, params, step="event")
                # A resumed run completes from the saved job; nothing is submitted again.
                resumed = await env_for("user-a")
                result = await complete(resumed, satellite.get_data, params, job, step="event")

                self.assertEqual((result.status, result.handle, result.product), ("done", "obs_1", "TEMPO_NO2"))
                self.assertEqual([tool for tool, _ in earthdata.calls].count("retrieve_subset"), 1)
                record = await stores.handles.get("user-a", "obs_1")
                self.assertEqual((record.kind, record.origin, record.descriptor.product), ("grid", "retrieval", "TEMPO_NO2"))
                self.assertEqual(record.descriptor.region["area"], area)
                self.assertEqual(record.descriptor.masking["variable"], "vertical_column_troposphere")
                self.assertEqual(result.quality["qa_status"], QA_VERIFIED)

                entries = await stores.traces.entries("user-a", env.run_id)
                self.assertEqual([(e.capability, e.status) for e in entries],
                                 [("sources.areas.resolve", "done"), ("sources.satellite.get_data", "waiting")])
                self.assertEqual(entries[1].result["job"], job)
                [done] = await stores.traces.entries("user-a", resumed.run_id)
                self.assertEqual((done.status, done.params["_job"]), ("done", job))
                self.assertNotIn(SECRET, record.model_dump_json() + str(entries[1].result) + str(done.result))

                # Another user cannot resolve the recorded retrieval or the area.
                other = await env_for("user-b")
                with self.assertRaises(HandleError):
                    await stores.handles.get("user-b", "obs_1")
                with self.assertRaises(HandleError):
                    await other.resolver.get("user-b", area)
