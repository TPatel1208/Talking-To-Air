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
    os.path.join(BACKEND_DIR, "..", "sql", "init_agent_artifacts.sql")
)


class SchemaContractTests(unittest.TestCase):
    """The `agent_artifacts.id` column must be TEXT, not UUID.

    ArtifactStore mints `tbl_<hex12>` ids, and the frontend cites that exact
    value back to fetch a page or export a CSV. A UUID column rejects it at
    INSERT ("invalid input syntax for type uuid"), silently breaking every
    table artifact. Mirrors the same regression test for agent_charts
    (test_chart_repository.py) -- the unit suite mocks the DB connection, so
    only this schema contract catches it.
    """

    def test_id_column_is_text_not_uuid(self):
        with open(_SCHEMA_SQL, encoding="utf-8") as fh:
            schema = fh.read()
        create = re.search(r"CREATE TABLE[^(]*\((.*?)\n\);", schema, re.S)
        self.assertIsNotNone(create, "could not locate CREATE TABLE agent_artifacts")
        id_line = next(
            (ln.strip() for ln in create.group(1).splitlines()
             if re.match(r"\s*id\s", ln) and "--" not in ln.split("id", 1)[0]),
            None,
        )
        self.assertIsNotNone(id_line, "no id column found in agent_artifacts schema")
        self.assertRegex(id_line.lower(), r"^id\s+text\b")
        self.assertNotIn("uuid", id_line.lower())


def _fake_pg_connection(conn):
    @asynccontextmanager
    async def _factory(*args, **kwargs):
        yield conn

    return _factory


class SaveAndGetArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_then_get_round_trips_the_full_payload(self):
        from repositories import artifact_repository

        store = {}

        async def fake_execute(query, params=None):
            if query.strip().startswith("INSERT"):
                artifact_id, user_id, thread_id, title, columns, rows, metadata = params
                store[artifact_id] = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "title": title,
                    "columns": columns.obj,
                    "rows": rows.obj,
                    "metadata": metadata.obj,
                }
                return MagicMock()
            if query.strip().startswith("SELECT"):
                (artifact_id,) = params
                row = store.get(artifact_id)
                cursor = MagicMock()
                if row is None:
                    cursor.fetchone = AsyncMock(return_value=None)
                else:
                    cursor.fetchone = AsyncMock(return_value=(
                        row["user_id"], row["thread_id"], row["title"],
                        row["columns"], row["rows"], row["metadata"],
                    ))
                return cursor
            raise AssertionError(f"unexpected query: {query}")

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        with patch("repositories.artifact_repository.pg_connection", _fake_pg_connection(conn)):
            await artifact_repository.save_artifact(
                "tbl_abc123", "user-1", "thread-1", "Sample Table",
                ["date", "value"], [{"date": "2024-01-01", "value": 10}], {"dataset": "EPA AQS"},
            )
            fetched = await artifact_repository.get_artifact("tbl_abc123")

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["user_id"], "user-1")
        self.assertEqual(fetched["thread_id"], "thread-1")
        self.assertEqual(fetched["title"], "Sample Table")
        self.assertEqual(fetched["columns"], ["date", "value"])
        self.assertEqual(fetched["rows"], [{"date": "2024-01-01", "value": 10}])
        self.assertEqual(fetched["metadata"], {"dataset": "EPA AQS"})

    async def test_get_missing_artifact_returns_none(self):
        from repositories import artifact_repository

        cursor = MagicMock()
        cursor.fetchone = AsyncMock(return_value=None)
        conn = MagicMock()
        conn.execute = AsyncMock(return_value=cursor)

        with patch("repositories.artifact_repository.pg_connection", _fake_pg_connection(conn)):
            fetched = await artifact_repository.get_artifact("tbl_missing")

        self.assertIsNone(fetched)


class SaveArtifactNonFiniteSanitisationTests(unittest.IsolatedAsyncioTestCase):
    """Postgres `jsonb` rejects the literal `Infinity`/`-Infinity`/`NaN`
    tokens Python's json serialiser happily emits for non-finite floats --
    the exact regression chart_repository.save_chart's _sanitize_non_finite
    already guards against (the HCHO `valid_max: .inf` crash). EPA/ground
    stat rows can carry the same kind of value (e.g. a divide-by-zero in a
    percentage column), so save_artifact needs the same guard on rows and
    metadata.
    """

    async def test_row_and_metadata_non_finite_floats_persist_as_null(self):
        from repositories import artifact_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        rows = [{"date": "2024-01-01", "value": float("inf")}, {"date": "2024-01-02", "value": float("nan")}]
        metadata = {"dataset": "EPA AQS", "coverage_fraction": float("-inf")}

        with patch("repositories.artifact_repository.pg_connection", _fake_pg_connection(conn)):
            await artifact_repository.save_artifact(
                "tbl_abc123", "user-1", "thread-1", "Sample Table", ["date", "value"], rows, metadata,
            )

        args = conn.execute.await_args.args[1]
        rows_arg, metadata_arg = args[5], args[6]
        import json as _json
        _json.dumps(rows_arg.obj, allow_nan=False)
        _json.dumps(metadata_arg.obj, allow_nan=False)
        self.assertIsNone(rows_arg.obj[0]["value"])
        self.assertIsNone(rows_arg.obj[1]["value"])
        self.assertIsNone(metadata_arg.obj["coverage_fraction"])
        self.assertEqual(metadata_arg.obj["dataset"], "EPA AQS")


class DeleteArtifactsForSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_scoped_to_thread_and_user(self):
        from repositories import artifact_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("repositories.artifact_repository.pg_connection", _fake_pg_connection(conn)):
            await artifact_repository.delete_artifacts_for_session("thread-1", "user-1")

        conn.execute.assert_awaited_once()
        query, params = conn.execute.await_args.args
        self.assertIn("DELETE FROM agent_artifacts", query)
        self.assertEqual(params, ("thread-1", "user-1"))


class DeleteExpiredUnclaimedTests(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_rows_unclaimed_past_the_ttl(self):
        from repositories import artifact_repository

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()

        with patch("repositories.artifact_repository.pg_connection", _fake_pg_connection(conn)):
            await artifact_repository.delete_expired_unclaimed(1800)

        conn.execute.assert_awaited_once()
        query, params = conn.execute.await_args.args
        self.assertIn("DELETE FROM agent_artifacts", query)
        self.assertIn("claimed_at IS NULL", query)
        self.assertEqual(params, (1800,))


if __name__ == "__main__":
    unittest.main()
