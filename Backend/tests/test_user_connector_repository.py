import os
import sys
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _fake_pg_connection(conn):
    @asynccontextmanager
    async def _factory(*args, **kwargs):
        yield conn

    return _factory


class _FakeCursor:
    def __init__(self, fetchone_result=None, fetchall_result=None):
        self.execute = AsyncMock()
        self.fetchone = AsyncMock(return_value=fetchone_result)
        self.fetchall = AsyncMock(return_value=fetchall_result if fetchall_result is not None else [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_conn(cursor=None, execute_result=None):
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor or _FakeCursor())
    conn.execute = AsyncMock(return_value=execute_result or MagicMock(rowcount=0))
    conn.commit = AsyncMock()
    return conn


class EnsureTableSchemaContractTests(unittest.IsolatedAsyncioTestCase):
    """CREATE TABLE IF NOT EXISTS pattern, same lifecycle as users/revoked_tokens
    -- no migration framework, so the shape asserted here IS the schema."""

    async def test_table_ddl_has_a_unique_constraint_on_user_id_and_connector_type(self):
        from repositories import user_connector_repository

        conn = _fake_conn()
        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            await user_connector_repository.ensure_user_connector_table()

        ddl = conn.execute.await_args.args[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS user_connectors", ddl)
        self.assertIn("UNIQUE (user_id, connector_type)", ddl)
        self.assertIn("encrypted_secret", ddl)
        self.assertIn("last_used_at", ddl)


class UpsertConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_inserts_with_an_on_conflict_upgrade_and_never_returns_the_secret(self):
        from repositories import user_connector_repository

        row = {
            "connector_type": "earthdata", "auth_method": "token", "expires_at": datetime.now(timezone.utc),
            "status": "connected", "connected_at": datetime.now(timezone.utc), "last_used_at": None,
        }
        cursor = _FakeCursor(fetchone_result=row)
        conn = _fake_conn(cursor=cursor)

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            result = await user_connector_repository.upsert_connector(
                "user-1", "earthdata", "token", "encrypted-blob", datetime.now(timezone.utc) + timedelta(days=60),
            )

        sql, params = cursor.execute.await_args.args
        self.assertIn("ON CONFLICT (user_id, connector_type) DO UPDATE", sql)
        self.assertIn("RETURNING", sql)
        self.assertNotIn("encrypted_secret", sql.split("RETURNING")[1])
        self.assertIn("encrypted-blob", params)
        self.assertNotIn("encrypted_secret", result)


class ListConnectorsForUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_scopes_the_select_to_the_caller_and_never_selects_the_secret_column(self):
        from repositories import user_connector_repository

        cursor = _FakeCursor(fetchall_result=[{"connector_type": "earthdata"}])
        conn = _fake_conn(cursor=cursor)

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            rows = await user_connector_repository.list_connectors_for_user("user-1")

        sql, params = cursor.execute.await_args.args
        self.assertIn("WHERE user_id = %s", sql)
        self.assertEqual(params, ("user-1",))
        self.assertNotIn("encrypted_secret", sql)
        self.assertEqual(rows, [{"connector_type": "earthdata"}])


class DeleteConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_scopes_to_both_user_id_and_connector_type(self):
        from repositories import user_connector_repository

        conn = _fake_conn(execute_result=MagicMock(rowcount=1))

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            deleted = await user_connector_repository.delete_connector("user-1", "earthdata")

        sql, params = conn.execute.await_args.args
        self.assertIn("WHERE user_id = %s AND connector_type = %s", sql)
        self.assertEqual(params, ("user-1", "earthdata"))
        self.assertTrue(deleted)

    async def test_delete_reports_false_when_nothing_matched(self):
        from repositories import user_connector_repository

        conn = _fake_conn(execute_result=MagicMock(rowcount=0))

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            deleted = await user_connector_repository.delete_connector("user-1", "earthdata")

        self.assertFalse(deleted)


class GetConnectorSecretRowTests(unittest.IsolatedAsyncioTestCase):
    """T31: the one repository read that includes encrypted_secret -- used
    only by services/connector_credential_service.py for per-call injection
    resolution, never by an API response."""

    async def test_selects_the_secret_scoped_to_user_and_connector_type(self):
        from repositories import user_connector_repository

        row = {"encrypted_secret": "blob", "expires_at": datetime.now(timezone.utc), "status": "connected"}
        cursor = _FakeCursor(fetchone_result=row)
        conn = _fake_conn(cursor=cursor)

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            result = await user_connector_repository.get_connector_secret_row("user-1", "earthdata")

        sql, params = cursor.execute.await_args.args
        self.assertIn("encrypted_secret", sql)
        self.assertIn("WHERE user_id = %s AND connector_type = %s", sql)
        self.assertEqual(params, ("user-1", "earthdata"))
        self.assertEqual(result, row)

    async def test_returns_none_when_no_row_exists(self):
        from repositories import user_connector_repository

        cursor = _FakeCursor(fetchone_result=None)
        conn = _fake_conn(cursor=cursor)

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            result = await user_connector_repository.get_connector_secret_row("user-1", "earthdata")

        self.assertIsNone(result)


class TouchLastUsedAtTests(unittest.IsolatedAsyncioTestCase):
    async def test_updates_last_used_at_scoped_to_user_and_connector_type(self):
        from repositories import user_connector_repository

        conn = _fake_conn()

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            await user_connector_repository.touch_last_used_at("user-1", "earthdata")

        sql, params = conn.execute.await_args.args
        self.assertIn("SET last_used_at = NOW()", sql)
        self.assertIn("WHERE user_id = %s AND connector_type = %s", sql)
        self.assertEqual(params, ("user-1", "earthdata"))


class SetConnectorStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_updates_status_scoped_to_user_and_connector_type(self):
        from repositories import user_connector_repository

        conn = _fake_conn()

        with patch("repositories.user_connector_repository.pg_connection", _fake_pg_connection(conn)):
            await user_connector_repository.set_connector_status("user-1", "earthdata", "error")

        sql, params = conn.execute.await_args.args
        self.assertIn("SET status = %s", sql)
        self.assertIn("WHERE user_id = %s AND connector_type = %s", sql)
        self.assertEqual(params, ("error", "user-1", "earthdata"))


if __name__ == "__main__":
    unittest.main()
