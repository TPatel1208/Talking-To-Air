"""The event-vs-baseline workflow paused on Postgres and answered by a fresh runner, as a later turn would."""
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import xarray as xr

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_grids import granule
from tests.test_harness_sources_satellite import NO2, FakeService
from tests.test_harness_workflow_event_vs_baseline import BASELINE_VALUE, EVENT, EVENT_VALUE, NAME, OFFERED
from tests.test_harness_workflow_packs_run import EDL_SECRET, PACKS
from tests.test_harness_workflow_runner import Fixture
from tests.test_harness_workflow_runner_postgres import saver_in
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.database import Connect
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.storage import Stores, ensure_tables, postgres_stores
from tta_backend.harness.workflows.loader import load_packs


class EventVsBaselinePostgresTests(PostgresTestCase):
    @asynccontextmanager
    async def world(self) -> AsyncIterator[tuple[Connect, Stores, str]]:
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                yield connect, postgres_stores(connect), root

    async def test_the_paused_run_is_answered_with_an_edited_list_by_another_runner(self):
        service = FakeService()
        retrieved: list[str] = []

        def file_for(day: str) -> xr.Dataset:
            retrieved.append(day)
            return granule(day=day, value=EVENT_VALUE if day == EVENT else BASELINE_VALUE)

        service.make_file = file_for
        registry = build_registry()
        catalogue = load_packs(PACKS, registry, load_registry())

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        def fixture(root: str, stores: Stores, saver: Any) -> Fixture:
            return Fixture(root, registry, catalogue, earthdata=service.earthdata(), open_retrieval=open_retrieval,
                           credentials={"earthdata": EDL_SECRET}, stores=stores, saver=saver)

        edited = [day for day in OFFERED if day not in ("2024-06-05", "2024-06-18")]
        async with self.world() as (connect, stores, root):
            async with saver_in(connect) as saver:
                first = fixture(root, stores, saver)
                paused = await first.start({"quantity": NO2, "area": "New Jersey", "event_date": EVENT}, name=NAME)
                self.assertEqual(paused.status, "needs_input", paused.reason)
                self.assertEqual((paused.question or {})["items"], OFFERED)
                self.assertEqual(await first.run_status(paused.trace_id), "needs_input")
                self.assertEqual(retrieved, [EVENT])

            # A later turn: its own connection, checkpointer and runner.
            async with saver_in(connect) as saver:
                later = fixture(root, stores, saver)
                done = await later.runner.resume(await later.env(paused.trace_id), edited, step="days")
                self.assertEqual(done.status, "done", done.reason)
                self.assertEqual(sorted(retrieved), sorted([EVENT, *edited]))
                self.assertEqual(await later.run_status(paused.trace_id), "done")
                findings = {f.name: f.value for f in done.findings}
                self.assertEqual(findings["baseline_days"], 8)
                self.assertAlmostEqual(findings["mean_difference"] / (EVENT_VALUE - BASELINE_VALUE), 1.0)
                asked = [e for e in await stores.traces.entries("user-a", paused.trace_id) if e.kind == "ask"]
                self.assertEqual([(e.status, e.result) for e in asked], [
                    ("waiting", {"offered": 10}), ("done", {"answer": edited}),
                ])
                self.assertEqual(len(done.charts), 1)
