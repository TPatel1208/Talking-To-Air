"""The workflow runner on the real Postgres stores and LangGraph's Postgres checkpointer, in one isolated schema."""
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row

from tests.postgres_helpers import TEST_DATABASE_URL, PostgresTestCase, isolated_schema
from tests.test_harness_sources_satellite import NO2, FakeService
from tests.test_harness_workflow_agent import NOTES, AgentFixture, agent_spec, choice, said
from tests.test_harness_workflow_runner import (
    AREA_ENTRY,
    ASKING,
    DATASETS,
    EDL_SECRET,
    FETCH,
    FIXTURES,
    SECRET,
    Fixture,
    Script,
    fake_registry,
    make,
    spec,
    three_days,
)
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.database import Connect
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.storage import Stores, ensure_tables, postgres_stores
from tta_backend.harness.testing import ScriptedChatModel, ScriptedModels
from tta_backend.harness.trace import RunNotFound
from tta_backend.harness.workflows import BadAnswer
from tta_backend.harness.workflows.loader import load_specs

CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
CAPABILITY_ASKS = spec(
    [{"id": "choose", "use": "ops.ask", "with": {"label": "choose"}}, make("after", product="$choose.values.level")],
    {"data": ["$after"]}, inputs={}, name="levels",
)


@asynccontextmanager
async def saver_in(connect: Connect) -> AsyncIterator[AsyncPostgresSaver]:
    """A checkpointer on its own connection, in the same schema as ``connect``'s.

    The checkpointer needs autocommit and dict rows, which the harness's own
    connections do not have, so it gets a connection of its own.
    """
    async with connect() as conn:
        row = await (await conn.execute("SELECT current_schema()")).fetchone()
    assert row is not None
    async with await psycopg.AsyncConnection.connect(
        TEST_DATABASE_URL, autocommit=True, prepare_threshold=0, row_factory=dict_row,
        options=f"-c search_path={row[0]}",
    ) as saver_conn:
        saver = AsyncPostgresSaver(saver_conn)
        await saver.setup()
        yield saver


async def checkpoint_bytes(connect: Connect) -> bytes:
    """Everything the checkpointer stored, as one blob to search."""
    async with connect() as conn:
        stored: list[bytes] = []
        for table in CHECKPOINT_TABLES:
            cur = await conn.execute(f"SELECT * FROM {table}")  # noqa: S608
            for row in await cur.fetchall():
                for value in row:
                    if isinstance(value, bytes | memoryview):
                        stored.append(bytes(value))
                    elif value is not None:
                        stored.append(str(value).encode())
    return b"\n".join(stored)


class RunnerPostgresTests(PostgresTestCase):
    @asynccontextmanager
    async def world(self) -> AsyncIterator[tuple[Connect, Stores, str]]:
        with tempfile.TemporaryDirectory() as root:
            async with isolated_schema() as connect:
                await ensure_tables(connect)
                yield connect, postgres_stores(connect), root

    async def test_the_map_workflow_end_to_end(self):
        service = FakeService()
        service.make_file = lambda day: three_days(day, 2e15)
        registry = build_registry()
        catalogue = load_specs({"map": (FIXTURES / "map.yaml").read_text(encoding="utf-8")}, registry, DATASETS)

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return service.files[record.id]

        async with self.world() as (connect, stores, root), saver_in(connect) as saver:
            f = Fixture(root, registry, catalogue, earthdata=service.earthdata(), open_retrieval=open_retrieval,
                        credentials={"earthdata": EDL_SECRET}, stores=stores, saver=saver)
            result = await f.start(
                {"quantity": NO2, "area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-03"}},
                name="map",
            )
            self.assertEqual(result.status, "done", result.reason)
            self.assertEqual(await f.entries(result.trace_id), [
                AREA_ENTRY,
                ("use", "sources.satellite.get_data", "data", "waiting"),
                ("use", "sources.satellite.get_data", "data", "done"),
                ("use", "render.map", "figure", "done"),
            ])
            self.assertEqual(await f.run_status(result.trace_id), "done")
            self.assertEqual(result.handles, ("obs_1",))
            self.assertEqual(len(result.charts), 1)
            steps = await f.runner.steps(await f.env(result.trace_id))
            self.assertEqual([s.step for s in steps], ["data", "figure"])

    async def test_an_agent_step_answers_and_its_answer_reaches_the_result(self):
        script = Script()
        registry = fake_registry(script)
        catalogue = load_specs({"agent": agent_spec()}, registry, DATASETS)
        model = ScriptedChatModel(script=[said()])
        models = ScriptedModels({"agent_step": choice((model, "fake:primary"))})
        async with self.world() as (connect, stores, root), saver_in(connect) as saver:
            f = AgentFixture(root, registry, catalogue, script, models=models, notes=NOTES, stores=stores, saver=saver)
            result = await f.start({"quantity": NO2})
            self.assertEqual(result.status, "done", result.reason)
            self.assertEqual(result.interpretation, "The total is 3.")
            self.assertEqual(await f.entries(result.trace_id), [
                ("use", "ops.make", "base", "done"),
                ("use", "ops.make", "total", "done"),
                ("agent", None, "interpret", "done"),
            ])
            entry = await f.entry(result.trace_id, "interpret")
            self.assertEqual(entry.params["notes"], ["column-not-surface", "tempo-hours"])
            self.assertEqual(entry.result, {
                "answer": {"text": "The total is 3.", "ok": True},
                "model": "fake:primary", "model_calls": 1, "failed_model_calls": 0,
                "input_tokens": 12, "output_tokens": 7,
            })
            steps = await f.runner.steps(await f.env(result.trace_id))
            self.assertEqual([(s.step, s.kind) for s in steps], [("base", "use"), ("total", "use"), ("interpret", "agent")])

            blob = await checkpoint_bytes(connect)
            self.assertIn(b"The total is 3.", blob)
            # The model's own run is not part of the workflow's checkpoint.
            self.assertNotIn(b"You carry out one step", blob)

    async def test_a_waiting_run_resumes_in_another_runner_and_only_its_user_can_touch_it(self):
        script = Script()
        registry = fake_registry(script)
        catalogue = load_specs({"fetch": FETCH}, registry, DATASETS)
        script.statuses["job_1"] = ["running"]
        async with self.world() as (connect, stores, root):
            async with saver_in(connect) as saver:
                first = Fixture(root, registry, catalogue, script, stores=stores, saver=saver)
                waiting = await first.start({})
                self.assertEqual(waiting.status, "waiting")
                self.assertEqual(await first.run_status(waiting.trace_id), "waiting")

            # Another process: its own connection, checkpointer and runner.
            async with saver_in(connect) as saver:
                other = Fixture(root, registry, catalogue, script, stores=stores, saver=saver, user="user-b")
                other_env = await other.env(waiting.trace_id)
                for action in (other.runner.resume, other.runner.steps, other.runner.cancel):
                    with self.subTest(action=action.__name__), self.assertRaises(RunNotFound):
                        await action(other_env)

                owner = Fixture(root, registry, catalogue, script, stores=stores, saver=saver)
                self.assertEqual(await owner.run_status(waiting.trace_id), "waiting")
                done = await owner.runner.resume(await owner.env(waiting.trace_id))
                self.assertEqual(done.status, "done", done.reason)
                self.assertEqual(script.submissions(), 1)
                self.assertEqual(await owner.run_status(waiting.trace_id), "done")
                self.assertEqual((await owner.entries(waiting.trace_id))[-1], ("use", "ops.make", "use_it", "done"))

                blob = await checkpoint_bytes(connect)
                self.assertIn(waiting.trace_id.encode(), blob)
                self.assertIn(b"job_1", blob)
                self.assertNotIn(SECRET.encode(), blob)

    async def test_a_question_is_answered_into_the_run_by_another_runner(self):
        script = Script()
        registry = fake_registry(script)
        catalogue = load_specs({"asks": ASKING["confirm_list"], "level": CAPABILITY_ASKS}, registry, DATASETS)
        async with self.world() as (connect, stores, root):
            async with saver_in(connect) as saver:
                first = Fixture(root, registry, catalogue, script, stores=stores, saver=saver)
                listed = await first.start({})
                level = await first.start({}, name="levels")
                self.assertEqual((listed.status, level.status), ("needs_input", "needs_input"))
                with self.assertRaises(BadAnswer):
                    await first.runner.resume(await first.env(listed.trace_id), ["not a date"])
                self.assertEqual(await first.run_status(listed.trace_id), "needs_input")

            # A later turn: its own connection, checkpointer and runner.
            async with saver_in(connect) as saver:
                later = Fixture(root, registry, catalogue, script, stores=stores, saver=saver)
                done = await later.runner.resume(await later.env(listed.trace_id), ["2024-06-02", "2024-06-09"])
                self.assertEqual(done.status, "done", done.reason)
                self.assertEqual(script.calls[-1][1]["days"], ["2024-06-02", "2024-06-09"])
                answered = [e for e in await stores.traces.entries("user-a", listed.trace_id) if e.kind == "ask"]
                self.assertEqual([(e.status, e.result) for e in answered], [
                    ("waiting", {"offered": 3}), ("done", {"answer": ["2024-06-02", "2024-06-09"]}),
                ])

                chosen = await later.runner.resume(await later.env(level.trace_id), "surface")
                self.assertEqual(chosen.status, "done", chosen.reason)
                self.assertEqual(script.calls[-1][1]["product"], "surface")
                self.assertEqual(await later.run_status(level.trace_id), "done")
                self.assertEqual((await later.entries(level.trace_id))[-3:], [
                    ("ask", "ops.ask", "choose", "done"), ("use", "ops.ask", "choose", "done"),
                    ("use", "ops.make", "after", "done"),
                ])
