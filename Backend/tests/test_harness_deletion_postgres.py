"""Deleting a conversation's harness records from the Postgres stores."""
import tempfile

from tests.harness_deletion_helpers import DeletionChecks
from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tta_backend.harness.storage import ensure_tables, postgres_stores


class PostgresDeletionTests(DeletionChecks, PostgresTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        connect = await self.enterAsyncContext(isolated_schema())
        await ensure_tables(connect)
        self.stores = postgres_stores(connect)

    async def test_the_conversations_runs_results_handles_payloads_and_its_own_jobs_go(self):
        await self.check_the_conversations_runs_results_handles_payloads_and_its_own_jobs_go()

    async def test_a_job_another_conversation_used_and_everything_else_stays(self):
        await self.check_a_job_another_conversation_used_and_everything_else_stays()

    async def test_deleting_again_does_nothing(self):
        await self.check_deleting_again_does_nothing()

    async def test_each_store_deletes_only_what_it_is_asked_for(self):
        built = await self.scenario()
        await self.stores.results.delete_runs("user-a", [])
        await self.stores.jobs.forget_jobs("user-a", [])
        entries = await self.stores.traces.waiting_uses_elsewhere("user-a", "conv-9")
        self.assertEqual([e.run_id for e in entries], [built.runs["a-other"]])
        self.assertEqual(await self.stores.handles.delete_conversation("user-a", "conv-1"), [])
        self.assertEqual(await self.stores.traces.delete_conversation("user-a", "conv-1"), [])
