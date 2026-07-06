import os
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _fake_pg_connection(conn):
    @asynccontextmanager
    async def _factory(*args, **kwargs):
        yield conn

    return _factory


class SaveChartIdSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_computes_a_stable_hash_id_when_payload_has_no_chart_id(self):
        from repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored = await chart_repository.save_chart("thread-1", {"type": "heatmap"}, "user-1")

        self.assertTrue(stored["chart_id"])
        # Same content + thread + user always yields the same id (dedup).
        with patch("repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored_again = await chart_repository.save_chart("thread-1", {"type": "heatmap"}, "user-1")
        self.assertEqual(stored["chart_id"], stored_again["chart_id"])

    async def test_honors_a_pre_set_chart_id_instead_of_recomputing(self):
        from repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored = await chart_repository.save_chart(
                "thread-1",
                {"type": "heatmap", "chart_id": "map_abc123"},
                "user-1",
            )

        self.assertEqual(stored["chart_id"], "map_abc123")


if __name__ == "__main__":
    unittest.main()
