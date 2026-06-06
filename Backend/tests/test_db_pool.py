import os
import sys
import importlib.util
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class FakeConnection:
    def __init__(self):
        self.autocommit = False
        self.closed = False


class FakeConnectionContext:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.conn = FakeConnection()
        self.connection_calls = 0

    def connection(self):
        self.connection_calls += 1
        return FakeConnectionContext(self.conn)

    def close(self):
        self.closed = True


@unittest.skipIf(importlib.util.find_spec("psycopg") is None, "psycopg is not installed")
class DbPoolTests(unittest.TestCase):
    def tearDown(self):
        from utils import db

        db.close_db_pool()

    def test_pg_connection_acquires_from_pool_and_restores_autocommit(self):
        from utils import db

        db.close_db_pool()
        with patch("utils.db.ConnectionPool", FakePool):
            pool = db.init_db_pool()
            with db.pg_connection(autocommit=True) as conn:
                self.assertTrue(conn.autocommit)

            self.assertEqual(pool.connection_calls, 1)
            self.assertFalse(pool.conn.autocommit)


if __name__ == "__main__":
    unittest.main()
