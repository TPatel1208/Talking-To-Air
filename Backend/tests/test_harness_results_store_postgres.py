"""The results store on a real Postgres: the same behaviour as in memory, in an isolated schema."""
from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_results_store import ResultStoreContract, done_result
from tta_backend.harness import results_store
from tta_backend.harness.results_store import PostgresResultStore
from tta_backend.harness.trace import new_run_id


class PostgresResultStoreTests(ResultStoreContract, PostgresTestCase):
    async def asyncSetUp(self) -> None:
        schema = isolated_schema()
        self.connect = await schema.__aenter__()
        self.addAsyncCleanup(schema.__aexit__, None, None, None)
        await results_store.ensure_tables(self.connect)
        self.store = PostgresResultStore(self.connect)

    async def test_the_table_is_created_once(self):
        await results_store.ensure_tables(self.connect)
        run_id = new_run_id()
        await self.store.put("user-a", done_result(run_id))
        async with self.connect() as conn:
            cur = await conn.execute("SELECT user_id, run_id, status FROM harness_results")
            self.assertEqual(await cur.fetchall(), [("user-a", run_id, "done")])
