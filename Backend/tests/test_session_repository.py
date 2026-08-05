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

        with patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=True)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock()) as delete_charts, \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()) as delete_artifacts, \
             patch("tta_backend.repositories.session_repository.pg_connection", _pg_connection_cm):
            deleted = await SessionRepository().delete_session("thread-1", "user-1")

        self.assertTrue(deleted)
        delete_charts.assert_awaited_once_with("thread-1", "user-1")
        delete_artifacts.assert_awaited_once_with("thread-1", "user-1")

    async def test_delete_session_skips_cascade_when_session_not_found(self):
        from tta_backend.repositories.session_repository import SessionRepository

        with patch("tta_backend.repositories.session_repository.delete_session_metadata", AsyncMock(return_value=False)), \
             patch("tta_backend.repositories.session_repository.delete_charts_for_session", AsyncMock()) as delete_charts, \
             patch("tta_backend.repositories.session_repository.delete_artifacts_for_session", AsyncMock()) as delete_artifacts:
            deleted = await SessionRepository().delete_session("thread-1", "user-1")

        self.assertFalse(deleted)
        delete_charts.assert_not_awaited()
        delete_artifacts.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
