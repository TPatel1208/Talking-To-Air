import json
import os
import re
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
        from tta_backend.repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("tta_backend.repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored = await chart_repository.save_chart("thread-1", {"type": "heatmap"}, "user-1")

        self.assertTrue(stored["chart_id"])
        # Same content + thread + user always yields the same id (dedup).
        with patch("tta_backend.repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored_again = await chart_repository.save_chart("thread-1", {"type": "heatmap"}, "user-1")
        self.assertEqual(stored["chart_id"], stored_again["chart_id"])

    async def test_honors_a_pre_set_chart_id_instead_of_recomputing(self):
        from tta_backend.repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("tta_backend.repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored = await chart_repository.save_chart(
                "thread-1",
                {"type": "heatmap", "chart_id": "map_abc123"},
                "user-1",
            )

        self.assertEqual(stored["chart_id"], "map_abc123")


class SaveChartNonFiniteSanitisationTests(unittest.IsolatedAsyncioTestCase):
    """Postgres `json`/`jsonb` rejects the literal `Infinity`/`-Infinity`/`NaN`
    tokens Python's json serialiser happily emits for non-finite floats.

    Regression for the HCHO crash: collections.yaml pins `valid_max: .inf`
    ("no upper bound") for TEMPO_HCHO/OMHCHOd, which flowed via
    plot_tools._variable_definition into the chart payload and blew up the
    INSERT with `invalid input syntax for type json ... Token "Infinity"`.
    save_chart is the persistence boundary, so it must clamp non-finite
    floats to null for ANY payload, not just HCHO's valid_ranges.
    """

    async def test_payload_with_non_finite_floats_persists_as_valid_json_with_nulls(self):
        from tta_backend.repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        payload = {
            "type": "heatmap",
            "provenance": {
                "variable_definition": {
                    "valid_ranges": {"min": 0.0, "max": float("inf")},
                },
            },
            "stats": {"values": [float("-inf"), float("nan"), 1.5]},
            "metadata": {"vmax": float("inf"), "name": "hcho"},
        }

        with patch("tta_backend.repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            stored = await chart_repository.save_chart("thread-1", payload, "user-1")

        # The exact property Postgres enforces: RFC-compliant JSON. A dump
        # with allow_nan=False raises ValueError if any Infinity/NaN token
        # would reach the jsonb column.
        args = conn.execute.await_args.args[1]
        jsonb_payload, jsonb_metadata = args[3], args[4]
        json.dumps(jsonb_payload.obj, allow_nan=False)
        json.dumps(jsonb_metadata.obj, allow_nan=False)

        # inf/-inf/nan are clamped to null; finite values survive untouched.
        ranges = stored["provenance"]["variable_definition"]["valid_ranges"]
        self.assertIsNone(ranges["max"])
        self.assertEqual(ranges["min"], 0.0)
        self.assertEqual(stored["stats"]["values"], [None, None, 1.5])
        self.assertIsNone(stored["metadata"]["vmax"])
        self.assertEqual(stored["metadata"]["name"], "hcho")

    async def test_sanitised_payloads_still_dedupe_to_the_same_hash_id(self):
        from tta_backend.repositories import chart_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("tta_backend.repositories.chart_repository.pg_connection", _fake_pg_connection(conn)):
            with_inf = await chart_repository.save_chart(
                "thread-1", {"type": "heatmap", "vmax": float("inf")}, "user-1"
            )
            with_null = await chart_repository.save_chart(
                "thread-1", {"type": "heatmap", "vmax": None}, "user-1"
            )

        self.assertEqual(with_inf["chart_id"], with_null["chart_id"])


if __name__ == "__main__":
    unittest.main()
