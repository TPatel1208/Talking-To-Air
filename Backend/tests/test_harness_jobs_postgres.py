"""The Postgres retrieval job store: per-user rows, upsert, forget, and reuse through it."""
import datetime

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_jobs import NOW, REQUEST, Clock, Service
from tta_backend.harness.job_store import PostgresRetrievalJobStore, SavedJob, ensure_tables
from tta_backend.harness.jobs import ReusingEarthdata
from tta_backend.harness.run_context import request_fingerprint

FINGERPRINT = request_fingerprint("retrieve_subset", REQUEST)


def saved(job_handle: str = "job_1", **over: object) -> SavedJob:
    fields: dict[str, object] = {
        "fingerprint": FINGERPRINT, "tool": "retrieve_subset", "job_handle": job_handle,
        "answer": {"job_handle": job_handle, "message": "submitted"}, "submitted": NOW,
    }
    fields.update(over)
    return SavedJob(**fields)  # type: ignore[arg-type]


class PostgresRetrievalJobStoreTests(PostgresTestCase):
    async def test_a_saved_job_round_trips(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            await ensure_tables(connect)
            store = PostgresRetrievalJobStore(connect)
            self.assertIsNone(await store.find("user-a", FINGERPRINT))
            await store.save("user-a", saved())
            found = await store.find("user-a", FINGERPRINT)
            self.assertEqual(found, saved())
            assert found is not None
            self.assertEqual(found.submitted.utcoffset(), datetime.timedelta(0))

    async def test_saving_again_replaces_the_job(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            store = PostgresRetrievalJobStore(connect)
            await store.save("user-a", saved())
            later = saved("job_2", tool="retrieve_timeseries", submitted=NOW + datetime.timedelta(hours=1))
            await store.save("user-a", later)
            self.assertEqual(await store.find("user-a", FINGERPRINT), later)

    async def test_forget_removes_only_that_users_job(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            store = PostgresRetrievalJobStore(connect)
            await store.save("user-a", saved())
            await store.save("user-b", saved("job_b"))
            await store.forget("user-a", FINGERPRINT)
            await store.forget("user-a", "no-such-fingerprint")
            self.assertIsNone(await store.find("user-a", FINGERPRINT))
            self.assertEqual(await store.find("user-b", FINGERPRINT), saved("job_b"))

    async def test_another_users_job_is_missing(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            store = PostgresRetrievalJobStore(connect)
            await store.save("user-a", saved())
            self.assertIsNone(await store.find("user-b", FINGERPRINT))
            # Saving under user-b leaves user-a's row as it was.
            await store.save("user-b", saved("job_b"))
            self.assertEqual(await store.find("user-a", FINGERPRINT), saved())

    async def test_a_job_saved_by_one_session_is_reused_by_another(self):
        async with isolated_schema() as connect:
            await ensure_tables(connect)
            service = Service()
            first = ReusingEarthdata(service, PostgresRetrievalJobStore(connect), "user-a", Clock())
            second = ReusingEarthdata(service, PostgresRetrievalJobStore(connect), "user-a", Clock())
            answer = await first.call("retrieve_subset", REQUEST, idempotency_key="run-1")
            self.assertEqual(await second.call("retrieve_subset", REQUEST, idempotency_key="run-2"), answer)
            self.assertEqual(service.submitted, 1)
