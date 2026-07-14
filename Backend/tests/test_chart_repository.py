import os
import re
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

_SCHEMA_SQL = os.path.abspath(
    os.path.join(BACKEND_DIR, "..", "sql", "init_agent_charts.sql")
)


class SchemaContractTests(unittest.TestCase):
    """The `agent_charts.id` column must be TEXT, not UUID.

    save_chart honors caller-minted ids, and T06 artifact-typed plots mint
    prefixed, non-UUID ids (`map_...`, `cmp_...`, `ts_...`) that the frontend
    later cites verbatim to fetch the chart. A UUID column rejects those at
    INSERT ("invalid input syntax for type uuid"), silently breaking chart
    rendering. The unit suite mocks the DB connection, so only this schema
    contract — not a mocked save_chart call — can catch a regression here.
    """

    def test_id_column_is_text_not_uuid(self):
        with open(_SCHEMA_SQL, encoding="utf-8") as fh:
            schema = fh.read()
        create = re.search(r"CREATE TABLE[^(]*\((.*?)\n\);", schema, re.S)
        self.assertIsNotNone(create, "could not locate CREATE TABLE agent_charts")
        id_line = next(
            (ln.strip() for ln in create.group(1).splitlines()
             if re.match(r"\s*id\s", ln) and "--" not in ln.split("id", 1)[0]),
            None,
        )
        self.assertIsNotNone(id_line, "no id column found in agent_charts schema")
        self.assertRegex(id_line.lower(), r"^id\s+text\b")
        self.assertNotIn("uuid", id_line.lower())


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
