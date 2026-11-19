"""Two chat turns through the host on the real Postgres stores, transcript and checkpointer."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from harness_chat_helpers import make_chat  # noqa: E402
from tests.postgres_helpers import PostgresTestCase, isolated_schema  # noqa: E402
from tests.test_harness_sources_satellite import NO2, FakeService  # noqa: E402
from tests.test_harness_specialists import Model  # noqa: E402
from tests.test_harness_supervisor import MAP_INPUTS, ledger_text, supervisor_choice  # noqa: E402
from tests.test_harness_workflow_runner import EDL_SECRET, Fixture, three_days  # noqa: E402
from tests.test_harness_workflow_runner_postgres import saver_in  # noqa: E402
from tta_backend.datasets.registry import load_registry  # noqa: E402
from tta_backend.harness.capabilities.catalogue import build_registry  # noqa: E402
from tta_backend.harness.context.ledger import HEADING  # noqa: E402
from tta_backend.harness.database import Connect  # noqa: E402
from tta_backend.harness.handles import HandleRecord  # noqa: E402
from tta_backend.harness.knowledge import NoteCatalogue  # noqa: E402
from tta_backend.harness.models import NO_BACKOFF  # noqa: E402
from tta_backend.harness.storage import Stores, ensure_tables, postgres_stores  # noqa: E402
from tta_backend.harness.supervisor import Supervisor  # noqa: E402
from tta_backend.harness.testing import ScriptedModels, answer, tool_call  # noqa: E402
from tta_backend.harness.workflows.loader import load_packs  # noqa: E402
from tta_backend.models.transcript import TranscriptTurn  # noqa: E402
from tta_backend.repositories import chart_repository, transcript_repository  # noqa: E402
from tta_backend.services.artifact_store import artifact_store  # noqa: E402
from tta_backend.services.chart_service import ChartLink, ChartService  # noqa: E402
from tta_backend.services.result_export import ResultExporter, ResultNotFound, chart_provenance  # noqa: E402
from tta_backend.services.harness_chat import TurnPorts, transcript_history  # noqa: E402

PACKS = Path(__file__).resolve().parent.parent / "packs"


class HarnessChatPostgresTests(PostgresTestCase):
    @asynccontextmanager
    async def world(self) -> AsyncIterator[tuple[Connect, Stores, str]]:
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                with patch.object(transcript_repository, "pg_connection", connect), \
                     patch.object(chart_repository, "pg_connection", connect):
                    await transcript_repository.ensure_transcript_table()
                    await chart_repository.ensure_chart_table()
                    yield connect, postgres_stores(connect), root

    async def test_two_turns_on_one_thread_are_in_history_and_the_ledger_carries_across(self):
        service = FakeService()
        service.make_file = lambda day: three_days(day, 2e15)
        registry = build_registry()
        catalogue = load_packs(PACKS, registry, load_registry())

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        model = Model(script=[
            tool_call("run_workflow", {"name": "map", "inputs": MAP_INPUTS}),
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
            chat = make_chat(
                supervisor, root, stores=stores, credentials={"earthdata": EDL_SECRET}, clock=f.context.clock,
                ports=lambda user_id, report_job: TurnPorts(build_env=f.build, earthdata=service.earthdata()),
            )

            first = [frame async for frame in chat.stream_chat_events(f"Map {NO2} over New Jersey.", "thread-1",
                                                                       "user-a", "req-1")]
            second = [frame async for frame in chat.stream_chat_events("Is that the map you made?", "thread-1",
                                                                        "user-a", "req-2")]
            self.assertTrue(first[-1].startswith("event: done"))
            self.assertTrue(second[-1].startswith("event: done"))

            turns = await transcript_repository.list_turns("thread-1", "user-a")
            self.assertEqual([t.turn_index for t in turns], [0, 1])
            self.assertEqual(turns[0].sources, ["TEMPO_NO2"])
            self.assertEqual(turns[0].label.kind, "draft")
            self.assertEqual(turns[1].answer_text, "Yes, the map from the last turn.")
            done = json.loads(first[-1].partition("data: ")[2])
            [result] = done["results"]
            self.assertEqual(result["status"], "done")
            workflow_run = result["run_id"]
            ledger = ledger_text(model.received[2])
            self.assertTrue(ledger.startswith(HEADING))
            self.assertIn(workflow_run, ledger)
            [reference] = turns[0].charts
            self.assertIn(reference["chart_id"], ledger)

            with patch("tta_backend.services.artifact_store.artifact_repository.get_artifact", AsyncMock()):
                messages = await transcript_history("thread-1", "user-a", ChartService())
            self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user", "assistant"])
            self.assertEqual(messages[2]["content"], "Is that the map you made?")
            self.assertEqual(messages[3]["content"], "Yes, the map from the last turn.")
            # A reload shows the answer, with its sources line, exactly as it streamed.
            self.assertEqual(messages[1]["content"], done["response"])
            self.assertEqual(messages[1]["content"].splitlines()[0], "Data sources: TEMPO_NO2")
            [chart] = messages[1]["charts"]
            self.assertEqual(chart["chart_id"], reference["chart_id"])
            self.assertEqual(chart["user_id"], "user-a")
            self.assertTrue(messages[1]["label"].startswith("Draft workflow: map v"))
            self.assertEqual(messages[1]["results"], done["results"])
            self.assertEqual(messages[1]["suggested_followups"], done.get("suggested_followups", []))
            self.assertEqual(await transcript_history("thread-1", "user-b", ChartService()), [])

            # The stored chart carries its link to the run, and the run's result exports from the same stores.
            link = ChartLink.of(chart)
            assert link is not None
            self.assertEqual((link.run_id, link.mode.view()), (workflow_run, result["mode"]))
            view = await chart_provenance(stores, "user-a", link, None)
            self.assertEqual((view["run_id"], view["handle"]), (workflow_run, link.handle))
            self.assertEqual(view["product"]["key"], "TEMPO_NO2")
            exporter = ResultExporter(stores, chart_service=ChartService(), artifacts=artifact_store, payload_root=root)
            methods = await exporter.methods("user-a", workflow_run)
            self.assertTrue(methods.startswith("# Methods: Draft workflow: map v"))
            with zipfile.ZipFile(io.BytesIO(await exporter.archive("user-a", workflow_run))) as archive:
                self.assertEqual(archive.read("methods.md").decode(), methods)
                self.assertEqual(len([n for n in archive.namelist() if n.startswith("figures/")]), 1)
            with self.assertRaises(ResultNotFound):
                await exporter.methods("user-b", workflow_run)

    async def test_history_shows_only_the_users_own_charts_reads_them_in_one_query_and_writes_nothing(self):
        def chart(title: str) -> dict[str, Any]:
            return {"type": "timeseries", "title": title, "series": [], "metadata": {}}

        def reference(chart_id: str) -> dict[str, Any]:
            return {"chart_id": chart_id, "type": "timeseries", "reference": True}

        async def rows(connect: Connect) -> list[tuple[Any, ...]]:
            async with connect() as conn:
                cursor = await conn.execute("SELECT id, thread_id, user_id, payload FROM agent_charts ORDER BY id")
                return [tuple(row) for row in await cursor.fetchall()]

        async with self.world() as (connect, stores, root):
            mine = await chart_repository.save_chart("thread-1", chart("Mine"), "user-a")
            other = await chart_repository.save_chart("thread-9", chart("Theirs"), "user-b")
            also = await chart_repository.save_chart("thread-1", chart("Also mine"), "user-a")
            gone = "00000000-0000-0000-0000-0000000000aa"
            # The first turn points at two of the user's own charts, one that no longer exists and another user's.
            for charts in ([reference(mine["chart_id"]), reference(gone), reference(other["chart_id"]),
                            reference(also["chart_id"])],
                           [reference(mine["chart_id"])], []):
                await transcript_repository.append_turn(TranscriptTurn(
                    thread_id="thread-1", user_id="user-a", user_text="Plot it", answer_text="Plotted.", charts=charts,
                ))
            await transcript_repository.append_turn(TranscriptTurn(
                thread_id="thread-1", user_id="user-b", user_text="Mine too", answer_text="Plotted.",
                charts=[reference(mine["chart_id"])],
            ))
            before = await rows(connect)

            with patch.object(chart_repository, "get_charts", wraps=chart_repository.get_charts) as reads, \
                 patch.object(chart_repository, "save_chart", wraps=chart_repository.save_chart) as writes:
                messages = await transcript_history("thread-1", "user-a", ChartService())
                theirs = await transcript_history("thread-1", "user-b", ChartService())

            assistants = [m for m in messages if m["role"] == "assistant"]
            self.assertEqual([[c["chart_id"] for c in m["charts"]] for m in assistants],
                             [[mine["chart_id"], also["chart_id"]], [mine["chart_id"]], []])
            self.assertEqual(assistants[0]["charts"][0]["user_id"], "user-a")
            # Another user's chart is not shown to the one whose transcript points at it, and the reverse.
            self.assertEqual([c["chart_id"] for m in theirs for c in m.get("charts", [])], [])
            # Each history is one read for every chart of every turn, and nothing is written, no stub for the missing one.
            self.assertEqual(reads.await_count, 2)
            self.assertEqual(len(reads.await_args_list[0].args[0]), 5)
            self.assertEqual(writes.await_count, 0)
            self.assertEqual(await rows(connect), before)
            self.assertEqual({r[0]: r[2] for r in before}, {
                mine["chart_id"]: "user-a", also["chart_id"]: "user-a", other["chart_id"]: "user-b",
            })
