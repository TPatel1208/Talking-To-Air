import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class DeleteSessionCascadeTests(unittest.IsolatedAsyncioTestCase):
    """T39: a claimed table artifact is durable in agent_artifacts now, so
    deleting a session must clean it up the same way it already cleans up
    agent_charts -- otherwise deleted sessions leave orphaned artifact rows
    behind forever."""

    async def test_delete_session_also_deletes_that_sessions_artifacts(self):
        from tta_backend.repositories.session_repository import SessionRepository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        async def fake_pg_connection(*args, **kwargs):
            return conn

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _pg_connection_cm(*args, **kwargs):
            yield conn

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock(return_value=[])) as delete_charts, \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()) as delete_artifacts, \
             patch("tta_backend.repositories.session_repository.pg_connection", _pg_connection_cm):
            deleted = await SessionRepository().delete_session("thread-1", "user-1")

        self.assertTrue(deleted)
        delete_charts.assert_awaited_once_with("thread-1", "user-1")
        delete_artifacts.assert_awaited_once_with("thread-1", "user-1")

    async def test_delete_session_skips_cascade_when_session_not_found(self):
        from tta_backend.repositories.session_repository import SessionRepository

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=False)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock()) as delete_charts, \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()) as delete_artifacts:
            deleted = await SessionRepository().delete_session("thread-1", "user-1")

        self.assertFalse(deleted)
        delete_charts.assert_not_awaited()
        delete_artifacts.assert_not_awaited()


class DeleteSessionOwnershipOrderingTests(unittest.IsolatedAsyncioTestCase):
    """The session_metadata row is the only record of who owns a thread, and
    deleting it was used as the ownership *check*. Because the four deletes
    share no transaction and each commits on its own, a failure anywhere in
    the cascade left the ownership record already gone and committed while
    the charts, artifacts and checkpoint rows survived. The retry then finds
    no row, reports "not found", and the orphans become permanently
    unreachable -- nothing can authorise deleting them any more.

    Ownership must be established by a read, and the record that proves it
    destroyed last.
    """

    @staticmethod
    def _pg_connection_cm(conn):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm(*args, **kwargs):
            yield conn

        return _cm

    async def test_a_failed_cascade_leaves_the_session_still_deletable(self):
        from tta_backend.repositories.session_repository import SessionRepository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=True)) as delete_metadata, \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock(side_effect=RuntimeError("connection reset"))), \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()), \
             patch("tta_backend.repositories.session_repository.pg_connection", self._pg_connection_cm(conn)):
            with self.assertRaises(RuntimeError):
                await SessionRepository().delete_session("thread-1", "user-1")

        delete_metadata.assert_not_awaited()

    async def test_the_ownership_row_is_deleted_last_but_is_still_deleted(self):
        """Counterweight to the test above: "never delete the metadata row"
        would also survive a failed cascade. Pin that the happy path still
        removes it, and removes it *after* the rows whose cleanup it
        authorises."""
        from tta_backend.repositories.session_repository import SessionRepository

        calls: list[str] = []

        conn = MagicMock()

        async def record_execute(sql, *args, **kwargs):
            calls.append("checkpoints")

        conn.execute = AsyncMock(side_effect=record_execute)
        conn.commit = AsyncMock()

        def record(name, result=True):
            async def _inner(*args, **kwargs):
                calls.append(name)
                return result
            return _inner

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock(side_effect=record("charts", []))), \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock(side_effect=record("artifacts"))), \
             patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(side_effect=record("metadata"))) as delete_metadata, \
             patch("tta_backend.repositories.session_repository.pg_connection", self._pg_connection_cm(conn)):
            deleted = await SessionRepository().delete_session("thread-1", "user-1")

        self.assertTrue(deleted)
        delete_metadata.assert_awaited_once_with("thread-1", "user-1")
        self.assertEqual(calls[-1], "metadata", f"metadata row must go last, got order: {calls}")
        self.assertIn("charts", calls[:-1])
        self.assertIn("artifacts", calls[:-1])
        self.assertIn("checkpoints", calls[:-1])

    async def test_checkpoints_are_deleted_before_charts(self):
        """Checkpoints still name the thread's charts. If the cascade stops
        after the charts are gone but before the checkpoints are, the thread
        stays loadable and points at charts that no longer exist."""
        from tta_backend.repositories.session_repository import SessionRepository

        calls: list[str] = []

        conn = MagicMock()

        async def record_execute(sql, *args, **kwargs):
            calls.append("checkpoints")

        conn.execute = AsyncMock(side_effect=record_execute)
        conn.commit = AsyncMock()

        async def record_charts(*args, **kwargs):
            calls.append("charts")
            return []

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock(side_effect=record_charts)), \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()), \
             patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.pg_connection", self._pg_connection_cm(conn)):
            await SessionRepository().delete_session("thread-1", "user-1")

        self.assertLess(calls.index("checkpoints"), calls.index("charts"), f"order: {calls}")


class DeleteSessionOverlayTests(unittest.IsolatedAsyncioTestCase):
    """Deleting a session removed its chart rows but not the overlay PNGs
    those rows named, so "delete my session" left the imagery on disk until
    the store's LRU cap happened to reach it. Measured on the live volume:
    62 of 70 PNGs referenced by no chart."""

    def setUp(self):
        import os
        import tempfile

        from tta_backend.config.settings import get_settings

        store = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(store.cleanup)
        env = patch.dict(os.environ, {"OVERLAY_STORE_DIR": os.path.join(store.name, "overlays")})
        env.start()
        self.addCleanup(env.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

    @staticmethod
    def _pg_connection_cm():
        from contextlib import asynccontextmanager

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        @asynccontextmanager
        async def _cm(*args, **kwargs):
            yield conn

        return _cm

    async def _delete(self, delete_charts):
        from tta_backend.repositories.session_repository import SessionRepository

        with patch("tta_backend.repositories.session_repository.session_belongs_to_user", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", delete_charts), \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()), \
             patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.pg_connection", self._pg_connection_cm()):
            return await SessionRepository().delete_session("thread-1", "user-1")

    async def test_deleting_a_session_deletes_the_overlays_its_charts_named(self):
        from tta_backend.services.overlay_store import read_overlay, write_overlay

        single = write_overlay(b"\x89PNG single")
        panels = [write_overlay(b"\x89PNG p0"), write_overlay(b"\x89PNG p1")]
        difference = write_overlay(b"\x89PNG diff")
        other_session = write_overlay(b"\x89PNG other")
        deleted_rows = [
            {"type": "heatmap", "overlay": {"_path": single}},
            {
                "type": "heatmap_multi",
                "panels": [{"overlay": {"_path": p}} for p in panels],
                "difference": {"overlay": {"_path": difference}},
            },
            {"type": "timeseries"},
        ]

        self.assertTrue(await self._delete(AsyncMock(return_value=deleted_rows)))

        for path in (single, *panels, difference):
            self.assertIsNone(read_overlay(path), f"{path} survived its session")
        self.assertEqual(read_overlay(other_session), b"\x89PNG other")

    async def test_a_failed_chart_delete_leaves_the_overlays(self):
        """The rows go first. If they survive, the charts are still served
        and must still find their overlays."""
        from tta_backend.services.overlay_store import read_overlay, write_overlay

        path = write_overlay(b"\x89PNG single")

        with self.assertRaises(RuntimeError):
            await self._delete(AsyncMock(side_effect=RuntimeError("connection reset")))

        self.assertEqual(read_overlay(path), b"\x89PNG single")

    async def test_a_failed_overlay_delete_does_not_fail_the_session_delete(self):
        """The rows are already gone, so a PNG left behind is an orphan the
        LRU cap reclaims -- not a reason to report the delete as failed and
        stop before the ownership row."""
        deleted_rows = [{"type": "heatmap", "overlay": {"_path": "/x/" + "a" * 32 + ".png"}}]

        with patch(
            "tta_backend.repositories.session_repository.overlay_store.delete_overlays",
            side_effect=PermissionError("read-only volume"),
        ):
            self.assertTrue(await self._delete(AsyncMock(return_value=deleted_rows)))


if __name__ == "__main__":
    unittest.main()
