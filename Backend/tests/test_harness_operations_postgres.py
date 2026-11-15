"""Two operations chained against the real Postgres handle registry and trace store."""
import datetime
import tempfile

import numpy as np

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_operations import MASKING, REGION, THREE_DAYS, cube, ramp
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.operations import compare, reduce
from tta_backend.harness.handles import Descriptor, HandleError
from tta_backend.harness.handles.model import Period
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.testing import make_run_context, make_run_env

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)


class OperationsPostgresTests(PostgresTestCase):
    async def test_aggregate_then_difference_records_lineage_and_trace(self):
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                stores = postgres_stores(connect)
                env = await make_run_env(root, context=make_run_context(stores=stores))

                async def grid(offset: float, days: list[str]) -> str:
                    record = await env.resolver.mint(
                        user_id="user-a", conversation_id="conv-1", kind="grid", data=cube(ramp(offset), days),
                        descriptor=Descriptor(
                            summary="NO2 over New Jersey", quantity="no2_tropospheric_column", unit="ppb",
                            product="TEMPO_NO2", region=REGION, period=Period(start=days[0], end=days[-1]),
                            masking=MASKING,
                        ),
                        operation="sources.satellite.get_data", created_at=NOW,
                    )
                    return record.id

                june = await grid(0.0, THREE_DAYS)
                july = await grid(4.0, ["2024-07-01", "2024-07-02", "2024-07-03"])
                mean_july = await invoke(env, reduce.aggregate_time, {"data": july}, step="july")
                diff = await invoke(env, compare.difference, {"a": june, "b": mean_july.handle}, step="diff")

                self.assertEqual(diff.status, "done")
                self.assertAlmostEqual(diff.summary["mean"], 4.0)
                stored = await env.resolver.open("user-a", diff.handle)
                np.testing.assert_allclose(stored["no2"].values, np.full((2, 3), 4.0))

                record = await stores.handles.get("user-a", diff.handle)
                self.assertEqual((record.origin, record.operation, record.parents), ("derived", "operations.difference", (june, mean_july.handle)))
                self.assertEqual(record.run_id, env.run_id)
                self.assertEqual(record.descriptor.masking["inputs"], [MASKING, MASKING])
                parent = await stores.handles.get("user-a", mean_july.handle)
                self.assertEqual(parent.parents, (july,))

                entries = await stores.traces.entries("user-a", env.run_id)
                self.assertEqual(
                    [(e.capability, e.status, e.step) for e in entries],
                    [("operations.aggregate_time", "done", "july"), ("operations.difference", "done", "diff")],
                )
                self.assertEqual(entries[1].result["summary"]["paired_cells"], 6)
                self.assertEqual(entries[1].params, {"a": june, "b": mean_july.handle, "change": "absolute", "over": "two_periods", "threshold": None})

                with self.assertRaises(HandleError):
                    await stores.handles.get("user-b", diff.handle)
