import unittest
from unittest.mock import patch


class FakeArtifactRepository:
    """In-memory stand-in for repositories.artifact_repository, keyed the
    same way the real Postgres table would be -- lets tests simulate a
    backend restart by handing a fresh ArtifactStore the same fake."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.swept_ttl_seconds = None

    async def save_artifact(self, artifact_id, user_id, thread_id, title, columns, rows, metadata):
        self.rows[artifact_id] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "title": title,
            "columns": list(columns),
            "rows": list(rows),
            "metadata": dict(metadata),
        }

    async def get_artifact(self, artifact_id):
        row = self.rows.get(artifact_id)
        return dict(row) if row is not None else None

    async def delete_artifacts_for_session(self, thread_id, user_id):
        self.rows = {
            aid: row for aid, row in self.rows.items()
            if not (row["thread_id"] == thread_id and row["user_id"] == user_id)
        }

    async def delete_expired_unclaimed(self, ttl_seconds):
        self.swept_ttl_seconds = ttl_seconds


class ClaimPersistsToRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_upserts_the_artifact_to_the_repository(self):
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()
        store = ArtifactStore()
        ref = store.put_table("Sample Table", ["date", "value"], [{"date": "2024-01-01", "value": 10}])

        with patch("services.artifact_store.artifact_repository", fake_repo):
            await store.claim(ref.id, "user-1", "thread-1")

        self.assertIn(ref.id, fake_repo.rows)
        row = fake_repo.rows[ref.id]
        self.assertEqual(row["user_id"], "user-1")
        self.assertEqual(row["thread_id"], "thread-1")
        self.assertEqual(row["rows"], [{"date": "2024-01-01", "value": 10}])

    async def test_claim_sweeps_expired_unclaimed_db_rows(self):
        """T39 user story #4: claim-time sweep of DB rows that were somehow
        left unclaimed past the TTL, so the table doesn't grow unboundedly
        from abandoned intermediate results."""
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()
        store = ArtifactStore(ttl_seconds=1800)
        ref = store.put_table("Sample Table", ["date", "value"], [{"date": "2024-01-01", "value": 10}])

        with patch("services.artifact_store.artifact_repository", fake_repo):
            await store.claim(ref.id, "user-1", "thread-1")

        self.assertEqual(fake_repo.swept_ttl_seconds, 1800)


class RestartSurvivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_page_and_csv_still_serve_after_a_simulated_restart(self):
        """Mint + claim on one ArtifactStore instance, then read from a
        *fresh* instance sharing only the fake repository -- this is exactly
        what a backend restart looks like: the in-memory dict is gone, only
        the durable row survives."""
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()

        with patch("services.artifact_store.artifact_repository", fake_repo):
            first_store = ArtifactStore()
            ref = first_store.put_table(
                "Sample Table", ["date", "value"], [{"date": "2024-01-01", "value": 10}],
            )
            await first_store.claim(ref.id, "user-1", "thread-1")

            second_store = ArtifactStore()
            page = await second_store.get_page(ref.id, "user-1")
            self.assertEqual(page["rows"], [{"date": "2024-01-01", "value": 10}])
            self.assertEqual(page["title"], "Sample Table")

            chunks = [chunk async for chunk in second_store.iter_csv_chunks(ref.id, "user-1")]
            csv_text = b"".join(chunks).decode("utf-8")
            self.assertIn("date,value", csv_text)
            self.assertIn("2024-01-01,10", csv_text)

    async def test_wrong_user_is_rejected_even_after_rehydration(self):
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()

        with patch("services.artifact_store.artifact_repository", fake_repo):
            first_store = ArtifactStore()
            ref = first_store.put_table("Sample Table", ["date", "value"], [{"date": "2024-01-01", "value": 10}])
            await first_store.claim(ref.id, "user-1", "thread-1")

            second_store = ArtifactStore()
            with self.assertRaises(KeyError):
                await second_store.get_page(ref.id, "user-2")

    async def test_unknown_artifact_raises_key_error(self):
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()
        store = ArtifactStore()

        with patch("services.artifact_store.artifact_repository", fake_repo):
            with self.assertRaises(KeyError):
                await store.get_page("tbl_missing", "user-1")


class UnclaimedArtifactExpiryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unclaimed_artifact_expires_after_its_ttl(self):
        import time
        from services.artifact_store import ArtifactStore

        fake_repo = FakeArtifactRepository()
        store = ArtifactStore(ttl_seconds=0)
        ref = store.put_table("Sample Table", ["date", "value"], [{"date": "2024-01-01", "value": 10}])
        time.sleep(0.01)

        with patch("services.artifact_store.artifact_repository", fake_repo):
            with self.assertRaises(KeyError):
                await store.get_page(ref.id, "user-1")


if __name__ == "__main__":
    unittest.main()
