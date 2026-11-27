"""Deleting a conversation's harness records from the in-memory stores."""
import tempfile
import unittest

from tests.harness_deletion_helpers import DeletionChecks
from tta_backend.harness.storage import memory_stores


class InMemoryDeletionTests(DeletionChecks, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.stores = memory_stores()

    async def test_the_conversations_runs_results_handles_payloads_and_its_own_jobs_go(self):
        await self.check_the_conversations_runs_results_handles_payloads_and_its_own_jobs_go()

    async def test_a_job_another_conversation_used_and_everything_else_stays(self):
        await self.check_a_job_another_conversation_used_and_everything_else_stays()

    async def test_deleting_again_does_nothing(self):
        await self.check_deleting_again_does_nothing()


if __name__ == "__main__":
    unittest.main()
