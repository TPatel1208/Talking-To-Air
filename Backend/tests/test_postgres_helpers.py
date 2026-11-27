"""The isolated-schema helper really isolates, cleans up and rolls back."""
import unittest

import psycopg

from tests.postgres_helpers import TEST_DATABASE_URL, PostgresTestCase, isolated_schema


async def _schema_exists(name: str) -> bool:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        cur = await conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (name,))
        return await cur.fetchone() is not None


async def _current_schema(connect) -> str:
    async with connect() as conn:
        cur = await conn.execute("SELECT current_schema()")
        row = await cur.fetchone()
        assert row is not None
        return str(row[0])


class IsolatedSchemaTests(PostgresTestCase):
    async def test_tables_are_invisible_across_schemas(self):
        async with isolated_schema() as first, isolated_schema() as second:
            async with first() as conn:
                await conn.execute("CREATE TABLE widgets (id int)")
                await conn.execute("INSERT INTO widgets VALUES (1)")
            async with first() as conn:
                cur = await conn.execute("SELECT count(*) FROM widgets")
                row = await cur.fetchone()
                assert row is not None
                self.assertEqual(row[0], 1)
            async with second() as conn:
                cur = await conn.execute("SELECT to_regclass('widgets')")
                row = await cur.fetchone()
                assert row is not None
                self.assertIsNone(row[0])

    async def test_schema_is_dropped_afterwards(self):
        async with isolated_schema() as connect:
            name = await _current_schema(connect)
            self.assertNotEqual(name, "public")
            self.assertTrue(await _schema_exists(name))
        self.assertFalse(await _schema_exists(name))

    async def test_exception_rolls_back(self):
        async with isolated_schema() as connect:
            async with connect() as conn:
                await conn.execute("CREATE TABLE widgets (id int)")
            with self.assertRaises(RuntimeError):
                async with connect() as conn:
                    await conn.execute("INSERT INTO widgets VALUES (1)")
                    raise RuntimeError("boom")
            async with connect() as conn:
                cur = await conn.execute("SELECT count(*) FROM widgets")
                row = await cur.fetchone()
                assert row is not None
                self.assertEqual(row[0], 0)


if __name__ == "__main__":
    unittest.main()
