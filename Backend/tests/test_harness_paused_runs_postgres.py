"""Listing and cancelling paused runs over the real Postgres stores, in one isolated schema."""
import tempfile
from pathlib import Path
from typing import Any

from harness_chat_helpers import make_chat
from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tests.test_harness_sources_satellite import FakeService
from tests.test_harness_specialists import Model
from tests.test_harness_supervisor import MAP_INPUTS, supervisor_choice
from tests.test_harness_workflow_runner import EDL_SECRET, Fixture, three_days
from tests.test_harness_workflow_runner_postgres import saver_in
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF
from tta_backend.harness.storage import ensure_tables, postgres_stores
from tta_backend.harness.supervisor import Supervisor
from tta_backend.harness.testing import ScriptedModels
from tta_backend.harness.workflows.loader import load_packs
from tta_backend.services import paused_runs
from tta_backend.services.harness_chat import TurnPorts, question_payload
from tta_backend.utils.streaming import user_id_context

PACKS = Path(__file__).resolve().parent.parent / "packs"


class PausedRunsPostgresTests(PostgresTestCase):
    async def test_a_question_run_lists_for_its_owner_and_conversation_only_and_cancels(self):
        service = FakeService()
        service.make_file = lambda day: three_days(day, 2e15)
        service.estimated_bytes = 100_000_000
        registry = build_registry()
        catalogue = load_packs(PACKS, registry, load_registry())

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        model = Model(script=[])
        async with isolated_schema() as connect, saver_in(connect) as saver:
            await ensure_tables(connect)
            stores = postgres_stores(connect)
            with tempfile.TemporaryDirectory() as root:
                f = Fixture(root, registry, catalogue, earthdata=service.earthdata(), open_retrieval=open_retrieval,
                            credentials={"earthdata": EDL_SECRET}, stores=stores, saver=saver)
                supervisor = Supervisor(
                    capabilities=registry, catalogue=catalogue, runner=f.runner, notes=NoteCatalogue([]),
                    checkpointer=saver, models=ScriptedModels({"supervisor": supervisor_choice(model)}),
                    backoff=NO_BACKOFF,
                )
                chat = make_chat(
                    supervisor, root, stores=stores, credentials={"earthdata": EDL_SECRET}, clock=f.context.clock,
                    ports=lambda user_id, report_job: TurnPorts(build_env=f.build, earthdata=service.earthdata()),
                )
                with user_id_context("user-a"):
                    asked = await f.runner.start(f.context, f.build, "map", 1, MAP_INPUTS)
                self.assertEqual(asked.status, "needs_input")

                [row] = await paused_runs.list_paused_runs(chat, "conv-1", "user-a")
                self.assertEqual((row["run_id"], row["status"]), (asked.trace_id, "needs_input"))
                self.assertEqual(row["question"], question_payload(asked.question))
                self.assertEqual(await paused_runs.list_paused_runs(chat, "conv-1", "user-b"), [])
                self.assertEqual(await paused_runs.list_paused_runs(chat, "conv-2", "user-a"), [])

                with self.assertRaises(paused_runs.RunNotPaused):
                    await paused_runs.cancel_paused_run(chat, "user-b", asked.trace_id)
                cancelled = await paused_runs.cancel_paused_run(chat, "user-a", asked.trace_id)
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertEqual((await stores.traces.get_run("user-a", asked.trace_id)).status, "cancelled")
                self.assertEqual(await paused_runs.list_paused_runs(chat, "conv-1", "user-a"), [])
                self.assertEqual(model.calls, 0)
