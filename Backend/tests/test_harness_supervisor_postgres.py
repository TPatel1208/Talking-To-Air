"""The supervisor over the real Postgres stores and LangGraph's Postgres checkpointer, in one isolated schema."""
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_sources_satellite import NO2, FakeService
from tests.test_harness_specialists import Model
from tests.test_harness_supervisor import MAP_INPUTS, ledger_text, supervisor_choice
from tests.test_harness_workflow_runner import EDL_SECRET, Fixture, three_days
from tests.test_harness_workflow_runner_postgres import checkpoint_bytes, saver_in
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.context.ledger import HEADING
from tta_backend.harness.database import Connect
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF
from tta_backend.harness.storage import Stores, ensure_tables, postgres_stores
from tta_backend.harness.supervisor import Supervisor
from tta_backend.harness.testing import ScriptedModels, answer, tool_call
from tta_backend.harness.workflows.loader import load_packs

PACKS = Path(__file__).resolve().parent.parent / "packs"


class SupervisorPostgresTests(PostgresTestCase):
    @asynccontextmanager
    async def world(self) -> AsyncIterator[tuple[Connect, Stores, str]]:
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                yield connect, postgres_stores(connect), root

    async def test_a_two_turn_conversation_carries_the_ledger_across_turns(self):
        service = FakeService()
        service.make_file = lambda day: three_days(day, 2e15)
        registry = build_registry()
        catalogue = load_packs(PACKS, registry, load_registry())

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        model = Model(script=[
            tool_call("run_workflow", {"name": "map", "inputs": MAP_INPUTS}),
            tool_call("resolve_area", {"place": "NJ"}),
            answer("The NO2 map over New Jersey is ready."),
            answer("Yes, the map from the last turn."),
        ])
        async with self.world() as (connect, stores, root), saver_in(connect) as saver:
            f = Fixture(root, registry, catalogue, earthdata=service.earthdata(), open_retrieval=open_retrieval,
                        credentials={"earthdata": EDL_SECRET}, stores=stores, saver=saver)
            supervisor = Supervisor(
                capabilities=registry, catalogue=catalogue, runner=f.runner, notes=NoteCatalogue([]),
                checkpointer=saver, models=ScriptedModels({"supervisor": supervisor_choice(model)}), backoff=NO_BACKOFF,
            )

            first = await supervisor.turn(f.context, f.build, service.earthdata(), f"Map {NO2} over New Jersey.")
            self.assertEqual(first.status, "answered")
            [result] = first.results
            self.assertEqual(result.status, "done", result.reason)
            workflow_run, lookup_run = first.run_ids
            self.assertEqual(await stores.results.get("user-a", workflow_run), result)
            self.assertEqual((await stores.traces.get_run("user-a", lookup_run)).status, "done")
            self.assertTrue(first.text.startswith("Data sources: TEMPO_NO2\n\nThe NO2 map over New Jersey is ready."))

            second = await supervisor.turn(f.context, f.build, service.earthdata(), "Is that the map you made?")
            self.assertEqual(second.text, "Yes, the map from the last turn.")
            ledger = ledger_text(model.received[3])
            self.assertTrue(ledger.startswith(HEADING))
            for expected in (workflow_run, lookup_run, result.handles[0], *result.charts):
                self.assertIn(expected, ledger)
            # The earlier turn is in the checkpointed history; the ledger never is.
            self.assertIn("The NO2 map over New Jersey is ready.", [m.text for m in model.received[3]])
            blob = await checkpoint_bytes(connect)
            self.assertIn(b"Is that the map you made?", blob)
            self.assertNotIn(HEADING.encode(), blob)
