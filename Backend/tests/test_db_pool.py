import os
import sys
import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("psycopg") is None, "psycopg is not installed")
class DbPoolTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        self.psycopg = psycopg

    def tearDown(self):
        from utils import db
        import asyncio
        asyncio.run(db.close_db_pool())

    def _make_fake_connection(self):
        psycopg = self.psycopg

        class FakeConnection:
            def __init__(self):
                self.autocommit = False
                self.closed = False
                self.info = SimpleNamespace(
                    transaction_status=psycopg.pq.TransactionStatus.IDLE
                )

            async def rollback(self):
                pass
            async def commit(self):
                pass
            async def set_autocommit(self, value):
                self.autocommit = value

        return FakeConnection()

    def test_pg_connection_acquires_from_pool_and_restores_autocommit(self):
        from utils import db
        import asyncio

        conn = self._make_fake_connection()

        class FakeConnectionContext:
            def __init__(self, c):
                self.conn = c
            async def __aenter__(self):
                return self.conn
            async def __aexit__(self, *args):
                return False

        class FakePool:
            def __init__(self, *args, **kwargs):
                self.closed = False
                self.connection_calls = 0
                self._conn = conn
            async def open(self):
                pass
            def connection(self):
                self.connection_calls += 1
                return FakeConnectionContext(self._conn)
            async def close(self):
                self.closed = True

        async def run_test():
            await db.close_db_pool()
            with patch("utils.db.AsyncConnectionPool", FakePool):
                pool = await db.init_db_pool()
                async with db.pg_connection(autocommit=True) as c:
                    self.assertTrue(c.autocommit)

                self.assertEqual(pool.connection_calls, 1)
                self.assertFalse(pool._conn.autocommit)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
