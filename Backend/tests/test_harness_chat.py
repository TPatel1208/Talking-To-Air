"""Chat turns through the harness supervisor: the SSE frames, the transcript and the host's ports."""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from harness_chat_helpers import (  # noqa: E402
    MemoryCharts,
    MemoryTranscripts,
    PlaybackSupervisor,
    make_chat,
    make_turn,
    memory_repositories,
)
from redis_helpers import REDIS_URL, requires_redis  # noqa: E402
from tests.test_harness_specialists import Model, Routed, choice_of, final, input_handle, results  # noqa: E402
from tests.test_harness_supervisor import SupervisorCase, run_map  # noqa: E402
from tests.test_harness_sources_satellite import NO2  # noqa: E402
from tests.test_harness_workflow_packs_run import EDL_SECRET, JULY, JUNE, GroundWorkflowRuns, PackRun  # noqa: E402
from tta_backend.harness.testing import FakeToolError, ScriptedModels, StaticCredentials, env_factory, tool_call  # noqa: E402
from tta_backend.config.error_templates import (  # noqa: E402
    render_error_answer,
    render_overloaded_answer,
    render_turn_timeout_answer,
)
from tta_backend.earthdata_mcp.connection import STATE_READY  # noqa: E402
from tta_backend.earthdata_mcp.results import MCPToolError  # noqa: E402
from tta_backend.harness.results import Mode, Result  # noqa: E402
from tta_backend.harness.storage import memory_stores  # noqa: E402
from tta_backend.harness.trace import Run, new_run_id  # noqa: E402
from tta_backend.harness.supervisor.supervisor import FAILED_WORDING  # noqa: E402
from tta_backend.harness.testing import answer, rate_limited  # noqa: E402
from tta_backend.services import admission, cube_cache  # noqa: E402
from tta_backend.services.harness_chat import (  # noqa: E402
    ConnectorCredentials,
    JobReporting,
    ReadyEarthdata,
    StoppedConnectors,
    StoredArtifacts,
    TurnPorts,
    connector_reachable,
    picker_payload,
    result_summaries,
    transcript_history,
)
from tta_backend.services.harness_chat import env_factory as app_env_factory  # noqa: E402
from tta_backend.services.harness_host import open_retrieval_over  # noqa: E402
from tta_backend.harness.supervisor.answer import unavailable_wording  # noqa: E402
from tta_backend.harness.testing import FakeEarthdata, FakePlaces, InlineCompute, TEST_LIMITS  # noqa: E402
from tta_backend.utils.streaming import emit_status  # noqa: E402
from harness_handle_helpers import retrieval_record  # noqa: E402
from tta_backend.services.chart_service import ChartService  # noqa: E402

RUN_A = "run_" + "a" * 32
RUN_B = "run_" + "b" * 32
CHART = {"type": "timeseries", "title": "NO2 over time", "series": [], "metadata": {}}


def parse(frames: list[str]) -> list[tuple[str, Any]]:
    parsed = []
    for frame in frames:
        head, _, data = frame.partition("\ndata: ")
        parsed.append((head.removeprefix("event: "), json.loads(data)))
    return parsed


def events(frames: list[str]) -> list[str]:
    return [name for name, _ in parse(frames)]


def last(frames: list[str], name: str) -> Any:
    return [data for event, data in parse(frames) if event == name][-1]


class Collecting:
    """Runs a turn and collects its frames, watching when the transcript is written."""

    def __init__(self, transcripts: MemoryTranscripts) -> None:
        self.transcripts = transcripts

    async def run(self, chat: Any, message: str = "hi", thread_id: str = "thread-1", user_id: str = "user-a") -> list[str]:
        frames: list[str] = []
        self.transcripts.watch = frames
        async for frame in chat.stream_chat_events(message, thread_id, user_id, "req-1"):
            frames.append(frame)
        return frames


class ScriptedSupervisorTurnTests(SupervisorCase):
    """The real supervisor on scripted models, run through the chat host."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.transcripts = MemoryTranscripts()
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(self.transcripts, self.charts))
        self.collect = Collecting(self.transcripts)

    def chat(self, model: Model) -> Any:
        return make_chat(
            self.supervisor(model), self.root, stores=self.f.stores,
            ports=lambda user_id, report_job: TurnPorts(build_env=self.build, earthdata=self.service.earthdata()),
            credentials={"earthdata": EDL_SECRET}, clock=self.f.context.clock,
        )

    async def test_a_workflow_turn_streams_status_text_chart_artifact_then_done(self):
        model = Model(script=[run_map(), answer("The NO2 map over New Jersey is ready.")])
        frames = await self.collect.run(self.chat(model), "Map NO2 over New Jersey for June 1 to 3, 2024.")

        names = events(frames)
        self.assertEqual(names[-1], "done")
        self.assertIn("status", names)
        self.assertLess(names.index("status"), names.index("chart"))
        self.assertLess(names.index("chart"), names.index("text"))
        self.assertEqual(names.count("chart"), 1)
        stages = [data.get("stage") for event, data in parse(frames) if event == "status"]
        self.assertIn("workflow.map", stages)

        done = last(frames, "done")
        self.assertEqual(set(done) - {"suggested_followups", "variable_choice"},
                         {"thread_id", "response", "artifacts", "tool_calls", "results", "label"})
        self.assertEqual(done["thread_id"], "thread-1")
        self.assertEqual(done["tool_calls"], [])
        self.assertTrue(done["response"].startswith("Data sources: TEMPO_NO2\n\nThe NO2 map over New Jersey is ready."))
        [result] = done["results"]
        latest = self.catalogue.latest("map").version
        self.assertEqual(result["status"], "done")
        self.assertTrue(result["run_id"].startswith("run_"))
        mode = {
            "kind": "draft", "label": f"Draft workflow: map v{latest}", "workflow": {"name": "map", "version": latest},
        }
        self.assertEqual(result["mode"], mode)
        self.assertEqual((result["reason"], result["connector"]), (None, None))
        self.assertEqual(done["label"], f"Draft workflow: map v{latest}")
        streamed = "".join(data["content"] for event, data in parse(frames) if event == "text")
        self.assertEqual(streamed, done["response"])

        # Every status frame of the workflow run names it; the first announces its steps.
        run_id = result["run_id"]
        statuses = [data for event, data in parse(frames) if event == "status"]
        announce = next(s for s in statuses if s.get("stage") == "workflow.map")
        self.assertEqual(announce["run_id"], run_id)
        spec = self.catalogue.latest("map")
        self.assertEqual(announce["detail"]["steps"], list(spec.order()))
        self.assertEqual(announce["detail"]["mode"], mode)
        self.assertEqual((announce["detail"]["workflow"], announce["detail"]["version"]), ("map", latest))
        steps = [s for s in statuses if s is not announce and s.get("run_id") == run_id]
        self.assertTrue(steps)
        # Inputs are resolved before the first step, under ids of their own.
        self.assertTrue(all(s["detail"]["step"] in spec.order() or s["detail"]["step"].startswith("input.") for s in steps))

        chart = last(frames, "chart")
        self.assertIn(chart["chart_id"], self.charts.rows)
        self.assertEqual(self.charts.saves, 1)
        self.assertEqual((chart["run_id"], chart["mode"]), (run_id, mode))
        link = self.charts.rows[chart["chart_id"]]["metadata"]["harness"]
        self.assertEqual(link["run_id"], run_id)
        self.assertEqual(link["handle"], chart["metadata"]["source_handles"][0])
        artifacts = [data for event, data in parse(frames) if event == "artifact"]
        self.assertTrue(artifacts)
        for artifact in artifacts:
            self.assertEqual((artifact["run_id"], artifact["mode"]), (run_id, mode))
        for artifact in done["artifacts"]:
            self.assertIn(artifact, artifacts)

        # Written once, before done.
        [turn] = self.transcripts.turns
        self.assertEqual(self.transcripts.written_after, [len(frames) - 1])
        self.assertEqual((turn.user_id, turn.thread_id, turn.status), ("user-a", "thread-1", "done"))
        self.assertEqual(turn.user_text, "Map NO2 over New Jersey for June 1 to 3, 2024.")
        self.assertEqual(turn.answer_text, done["response"])
        self.assertEqual(turn.label, Mode.draft("map", latest))
        self.assertEqual(turn.sources, ["TEMPO_NO2"])
        [reference] = turn.charts
        self.assertEqual((reference["chart_id"], reference["reference"]), (chart["chart_id"], True))
        self.assertNotIn("series", reference)
        drawn = ("run_id", "mode")
        self.assertEqual([a.model_dump(exclude_none=True) for a in turn.artifacts],
                         [{k: v for k, v in a.items() if k not in drawn} for a in done["artifacts"]])
        self.assertEqual([r.view() for r in turn.results], done["results"])
        self.assertEqual(turn.suggested_followups, done.get("suggested_followups", []))

    async def test_parallel_specialists_each_get_their_own_labelled_run_and_their_jobs_and_charts_arrive_once(self):
        def retrieve_and_map(dates: dict[str, str]) -> list[Any]:
            return [
                lambda m: tool_call("sources__satellite__get_data", {
                    "area": input_handle(m, "polygons"), "quantity": NO2, "dates": dates, "date_role": "range",
                }),
                lambda m: tool_call("render__map", {"grid": next(r["handle"] for r in results(m) if r.get("kind") == "grid")}),
                final("Mapped."),
            ]

        objectives = ("Map NO2 over NJ in early June.", "Map NO2 over NJ in mid June.")
        self.explorer = Routed(turns={objectives[0]: retrieve_and_map(JUNE), objectives[1]: retrieve_and_map(JULY)})
        session = self.service.earthdata()

        def ports(user_id: str, report_job: Any) -> TurnPorts:
            build = env_factory(
                self.root, earthdata=JobReporting(session, report_job), open_retrieval=self.open_retrieval,
                places=self.f.places, models=ScriptedModels({"explore": choice_of(self.explorer)}),
            )
            return TurnPorts(build_env=build, earthdata=session)

        tasks = [{"objective": o, "bundle": "analyse", "places": ["NJ"]} for o in objectives]
        model = Model(script=[tool_call("explore", {"tasks": tasks}), answer("Both maps are ready.")])
        chat = make_chat(
            self.supervisor(model), self.root, stores=self.f.stores, ports=ports,
            credentials={"earthdata": EDL_SECRET}, clock=self.f.context.clock,
        )
        frames = await self.collect.run(chat, "Map NO2 over NJ for early and mid June 2024.")

        done = last(frames, "done")
        runs = [r["run_id"] for r in done["results"]]
        self.assertEqual([r["status"] for r in done["results"]], ["done", "done"], done["results"])
        self.assertEqual(len(set(runs)), 2)
        parsed = parse(frames)
        statuses = [(i, data) for i, (event, data) in enumerate(parsed) if event == "status"]
        for run, objective in zip(runs, objectives, strict=True):
            with self.subTest(run=run):
                mine = [(i, d) for i, d in statuses if d.get("run_id") == run]
                # One row opens the run, labelled with its objective, before anything else of the run's.
                # The queued row comes first, then the one that opens the run.
                openings = [(i, d) for i, d in mine if d.get("stage") == "explore"]
                [(queued, held), (opened, opening)] = openings
                self.assertEqual(held["detail"], {"objective": objective, "bundle": "analyse", "queued": True})
                self.assertEqual(opening["detail"], {"objective": objective, "bundle": "analyse"})
                self.assertEqual(queued, min(i for i, _ in mine))
                self.assertGreater(len(mine), 2)
                charts = [(i, data) for i, (event, data) in enumerate(parsed) if event == "chart" and data.get("run_id") == run]
                [(drawn, chart)] = charts
                self.assertGreater(drawn, opened)
                self.assertEqual(chart["mode"]["label"], "Exploratory")
        # Nothing that opens a run is sent without it.
        self.assertTrue(all("run_id" in d for _, d in statuses if d.get("stage") == "explore"))
        chart_ids = [data["chart_id"] for event, data in parsed if event == "chart"]
        self.assertEqual(len(chart_ids), 2)
        self.assertEqual(len(set(chart_ids)), 2)
        self.assertEqual(self.charts.saves, 2)
        self.assertEqual([a["id"] for a in done["artifacts"]], chart_ids)
        jobs = [data for event, data in parsed if event == "job_progress"]
        submitted = [j["job_handle"] for j in jobs if j["status"] == "submitted"]
        self.assertEqual(sorted(submitted), ["job_1", "job_2"])
        self.assertTrue(all(set(j) == {"job_handle", "status", "progress", "phase", "message", "note"} for j in jobs))
        self.assertTrue(all(EDL_SECRET not in frame for frame in frames))

    async def test_the_apps_connection_state_decides_whether_a_turn_starts_anything(self):
        class Manager:
            state = "connecting"

        manager = Manager()
        model = Model(script=[run_map(), answer("No map."), run_map(), answer("The map is ready.")])
        chat = make_chat(
            self.supervisor(model), self.root, stores=self.f.stores, reachable=connector_reachable(manager),
            ports=lambda user_id, report_job: TurnPorts(build_env=self.build, earthdata=self.service.earthdata()),
            credentials={"earthdata": EDL_SECRET}, clock=self.f.context.clock,
        )

        frames = await self.collect.run(chat)
        wording = unavailable_wording("earthdata")
        done = last(frames, "done")
        self.assertEqual(done["response"].count(wording), 1)
        self.assertEqual(
            done["results"], [{"run_id": None, "status": "failed", "mode": None, "reason": wording, "connector": "earthdata"}],
        )
        self.assertEqual(await self.f.stores.traces.list_runs("user-a", "thread-1"), [])

        manager.state = STATE_READY
        frames = await self.collect.run(chat, "Try again.")
        done = last(frames, "done")
        self.assertNotIn(wording, done["response"])
        self.assertEqual([r["status"] for r in done["results"]], ["done"])

    def test_only_the_earthdata_service_can_be_down(self):
        self.assertFalse(connector_reachable(None)("earthdata"))
        self.assertTrue(connector_reachable(None)("epa_aqs"))

        class Manager:
            state = STATE_READY

        self.assertTrue(connector_reachable(Manager())("earthdata"))

    async def test_a_failed_result_reaches_the_user_in_fixed_wording_and_the_next_turn_works(self):
        earthdata = self.service.earthdata()

        def unsupported(arguments: dict[str, Any]) -> dict[str, Any]:
            raise FakeToolError("unsupported_grid", "detail only the service knows")

        earthdata.tools["search_datasets"] = unsupported
        build = env_factory(
            self.root, earthdata=earthdata, open_retrieval=self.open_retrieval, places=self.f.places,
            models=ScriptedModels({"explore": choice_of(self.explorer)}),
        )
        model = Model(script=[run_map(), answer("The map could not be made."), answer("Yes, still here.")])
        chat = make_chat(
            self.supervisor(model), self.root, stores=self.f.stores, clock=self.f.context.clock,
            ports=lambda user_id, report_job: TurnPorts(build_env=build, earthdata=earthdata),
            credentials={"earthdata": EDL_SECRET},
        )

        first = await self.collect.run(chat, "Map NO2 over New Jersey for June 1 to 3, 2024.")

        # A failed run is a result, not a failed turn: no error frame, and the reason is in the answer.
        self.assertEqual(events(first)[-1], "done")
        self.assertNotIn("error", events(first))
        done = last(first, "done")
        [result] = done["results"]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["reason"])
        self.assertIn(f"Not finished:\n- {result['reason']}", done["response"])
        self.assertNotIn("detail only the service knows", "".join(first))
        self.assertEqual(events(first).count("chart"), 0)
        [turn] = self.transcripts.turns
        self.assertEqual((turn.status, [r.status for r in turn.results], turn.charts), ("done", ["failed"], []))
        self.assertEqual(turn.answer_text, done["response"])

        second = await self.collect.run(chat, "Are you still there?")
        self.assertEqual(last(second, "done")["response"], "Yes, still here.")
        self.assertEqual([t.turn_index for t in self.transcripts.turns], [0, 1])
        self.assertEqual(self.transcripts.turns[1].status, "done")

    async def test_a_failing_model_ends_with_error_then_done_and_an_error_transcript(self):
        model = Model(script=[rate_limited(), rate_limited(), rate_limited()])
        frames = await self.collect.run(self.chat(model), "Map NO2 over New Jersey.")

        self.assertEqual(events(frames)[-2:], ["error", "done"])
        self.assertEqual(last(frames, "error"), {"detail": render_error_answer("contract", "request")})
        self.assertIn(FAILED_WORDING, last(frames, "done")["response"])
        [turn] = self.transcripts.turns
        self.assertEqual(turn.status, "error")
        self.assertIn(FAILED_WORDING, turn.answer_text)
        self.assertEqual(self.transcripts.written_after, [len(frames) - 2])

    async def test_a_needs_input_result_sends_its_question_to_the_picker(self):
        self.service.estimated_bytes = 100_000_000
        model = Model(script=[run_map(), answer("The retrieval is large.")])
        frames = await self.collect.run(self.chat(model), "Map NO2 over New Jersey for June 1 to 3, 2024.")

        done = last(frames, "done")
        [result] = done["results"]
        self.assertEqual(result["status"], "needs_input")
        choice = done["variable_choice"]
        self.assertEqual(choice["run_id"], result["run_id"])
        self.assertTrue(choice["message"])
        self.assertEqual([c["name"] for c in choice["candidates"]], ["confirm", "narrow"])
        self.assertEqual([c["prompt"] for c in choice["candidates"]], ["confirm", "narrow"])

    async def test_two_users_on_the_same_thread_id_have_separate_conversations(self):
        model = Model(script=[answer("Hello A."), answer("Hello B.")])
        chat = self.chat(model)
        await self.collect.run(chat, "hi", user_id="user-a")
        await self.collect.run(chat, "hi", user_id="user-b")

        # User B's first model call saw none of user A's conversation.
        self.assertNotIn("Hello A.", [m.text for m in model.received[1]])
        self.assertEqual([t.user_id for t in self.transcripts.turns], ["user-a", "user-b"])
        self.assertEqual([t.turn_index for t in self.transcripts.turns], [0, 0])


class HostFrameTests(unittest.IsolatedAsyncioTestCase):
    """The host around a supervisor double that plays a turn into the run context."""

    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.transcripts = MemoryTranscripts()
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(self.transcripts, self.charts))
        self.collect = Collecting(self.transcripts)

    async def test_a_chart_emitted_twice_is_stored_once_and_sent_once_with_its_artifact(self):
        supervisor = PlaybackSupervisor(("chart", CHART), ("emit", {**CHART, "chart_id": "chart-1"}), ("text", "Done."))
        frames = await self.collect.run(make_chat(supervisor, self.root))

        self.assertEqual(events(frames), ["chart", "artifact", "text", "done"])
        self.assertEqual(self.charts.saves, 1)
        chart = last(frames, "chart")
        self.assertEqual((chart["chart_id"], chart["user_id"], chart["thread_id"]), ("chart-1", "user-a", "thread-1"))
        artifact = last(frames, "artifact")
        self.assertEqual((artifact["id"], artifact["type"]), ("chart-1", "timeseries"))
        self.assertEqual(last(frames, "done")["artifacts"], [artifact])
        [turn] = self.transcripts.turns
        self.assertEqual([c["chart_id"] for c in turn.charts], ["chart-1"])

    async def test_a_table_is_sent_as_an_artifact_and_kept_in_the_transcript(self):
        supervisor = PlaybackSupervisor(("table", ("Sites", ["site"], [{"site": "A"}])), ("text", "Here."))
        frames = await self.collect.run(make_chat(supervisor, self.root))

        self.assertEqual(events(frames), ["artifact", "text", "done"])
        artifact = last(frames, "artifact")
        self.assertEqual((artifact["type"], artifact["title"], artifact["row_count"]), ("table", "Sites", 1))
        self.assertEqual(last(frames, "done")["artifacts"], [artifact])
        [turn] = self.transcripts.turns
        self.assertEqual([a.id for a in turn.artifacts], [artifact["id"]])

    async def test_a_table_a_run_drew_carries_the_runs_id_and_mode_and_only_for_its_own_user(self):
        stores = memory_stores()
        run_id = new_run_id()
        await stores.traces.start_run(Run(
            id=run_id, user_id="user-a", conversation_id="thread-1", mode=Mode.validated("trend", 2),
            started=datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC),
        ))
        sent: list[Any] = []
        for user in ("user-a", "user-b"):
            artifacts = StoredArtifacts(ChartService(), "thread-1", user, sent.append, stores.traces)
            await artifacts.store_table("Sites", ["site"], [{"site": "A"}], {"run_id": run_id})

        [(_, mine), (_, theirs)] = sent
        self.assertEqual((mine["run_id"], mine["mode"]), (run_id, Mode.validated("trend", 2).view()))
        self.assertNotIn("run_id", theirs)
        self.assertNotIn("mode", theirs)

    async def test_progress_becomes_status_frames_with_stage_and_detail(self):
        supervisor = PlaybackSupervisor(
            ("status", {"message": "Running workflow map v1", "stage": "workflow.map", "detail": {"run_id": RUN_A}}),
            ("status", {"message": "Preparing a table"}),
        )
        frames = await self.collect.run(make_chat(supervisor, self.root))

        statuses = [data for event, data in parse(frames) if event == "status"]
        self.assertEqual(statuses, [
            {"message": "Running workflow map v1", "stage": "workflow.map", "detail": {"run_id": RUN_A},
             "run_id": RUN_A},
            {"message": "Preparing a table"},
        ])

    async def test_the_user_reaches_the_run_context_and_the_earthdata_user_context(self):
        supervisor = PlaybackSupervisor(("text", "hi"))
        chat = make_chat(supervisor, self.root)
        await self.collect.run(chat, thread_id="thread-9", user_id="user-z")

        [context] = supervisor.contexts
        self.assertEqual((context.identity.user_id, context.identity.conversation_id), ("user-z", "thread-9"))
        self.assertEqual(supervisor.users_seen, ["user-z"])
        self.assertIs(context.storage, chat.stores)

    async def test_a_second_users_thread_is_separate(self):
        supervisor = PlaybackSupervisor(("text", "hi"))
        chat = make_chat(supervisor, self.root)
        await self.collect.run(chat, user_id="user-a")
        await self.collect.run(chat, user_id="user-b")

        self.assertEqual([c.identity.user_id for c in supervisor.contexts], ["user-a", "user-b"])
        self.assertEqual(len(await self.transcripts.list_turns("thread-1", "user-a")), 1)
        self.assertEqual(len(await self.transcripts.list_turns("thread-1", "user-b")), 1)

    async def test_retrievals_a_run_submits_and_polls_are_sent_as_job_progress_with_nothing_else(self):
        from tta_backend.harness.testing import FakeEarthdata

        secret = "edl-secret-7f1c"
        earthdata = FakeEarthdata({
            "retrieve_subset": lambda arguments: {
                "job_handle": "job-1", "workspace_id": "user-a", "request": {"edl_token": secret},
            },
            "retrieve_timeseries": lambda arguments: {"job_handle": "job-2", "status": "queued"},
            "get_retrieval_status": lambda arguments: {
                "status": "running", "progress": 40, "phase": "materializing", "message": "working",
                "workspace_id": "user-a",
            },
            "search_datasets": lambda arguments: {"job_handle": "not-a-job", "status": "ok"},
        })
        supervisor = PlaybackSupervisor(
            ("call", ("retrieve_subset", {"short_name": "TEMPO_NO2_L3", "edl_token": secret})),
            ("call", ("retrieve_timeseries", {"short_name": "TEMPO_NO2_L3"})),
            ("call", ("get_retrieval_status", {"job_handle": "job-1"})),
            ("call", ("search_datasets", {"query": "no2"})),
            # The supervisor's own check of waiting runs' jobs is not this turn's job.
            ("check", ("get_retrieval_status", {"job_handle": "job-old"})),
            ("text", "Done."),
        )
        frames = await self.collect.run(make_chat(supervisor, self.root, earthdata=earthdata))

        jobs = [data for event, data in parse(frames) if event == "job_progress"]
        empty = {"progress": None, "phase": None, "message": None, "note": None}
        self.assertEqual(jobs, [
            {"job_handle": "job-1", "status": "submitted", **empty},
            {"job_handle": "job-2", "status": "queued", **empty},
            {"job_handle": "job-1", "status": "running", "progress": 40, "phase": "materializing",
             "message": "working", "note": None},
        ])
        for frame in frames:
            self.assertNotIn(secret, frame)
            self.assertNotIn("workspace_id", frame)
            self.assertNotIn("TEMPO_NO2_L3", frame)
        self.assertEqual(events(frames)[-2:], ["text", "done"])

    async def test_a_failing_job_report_does_not_fail_the_call(self):
        from tta_backend.services.harness_chat import JobReporting

        def broken(payload: dict[str, Any]) -> None:
            raise RuntimeError("queue gone")

        inner = AsyncMock()
        inner.call = AsyncMock(return_value={"job_handle": "job-1"})
        with self.assertLogs("tta_backend.services.harness_chat", level="WARNING"):
            reply = await JobReporting(inner, broken).call("retrieve_subset", {})
        self.assertEqual(reply, {"job_handle": "job-1"})

    async def test_a_status_line_from_open_handle_reaches_the_stream_under_its_run(self):
        run = new_run_id()

        class Opening(PlaybackSupervisor):
            async def turn(self, context: Any, build_env: Any, earthdata: Any, message: str) -> Any:
                emit_status("Resolving the area...", stage="area")
                await context.storage.handles.record_retrieval(retrieval_record(context.identity.user_id))
                env = await build_env(context, run)
                await env.resolver.open(context.identity.user_id, "obs_tempo_no2_1")
                emit_status("Back outside any run.", stage="after")
                return make_turn("Opened.")

        def ports(user_id: str, report_job: Any) -> TurnPorts:
            session = FakeEarthdata()
            return TurnPorts(
                build_env=app_env_factory(
                    self.root, earthdata=session, open_retrieval=open_retrieval_over({}, user_id),
                    compute=InlineCompute(), limits=TEST_LIMITS, places=FakePlaces(), models=ScriptedModels({}),
                ),
                earthdata=session,
            )

        with patch("tta_backend.services.open_handle._serve_from_index", return_value="the data"):
            frames = await self.collect.run(make_chat(Opening(), self.root, ports=ports))

        statuses = [data for event, data in parse(frames) if event == "status"]
        self.assertEqual(statuses, [
            {"message": "Resolving the area...", "stage": "area"},
            {"message": "Opening retrieved data...", "stage": "open", "run_id": run},
            {"message": "Back outside any run.", "stage": "after"},
        ])

    async def test_a_turn_refused_for_a_down_source_names_it_in_done_and_the_transcript(self):
        supervisor = PlaybackSupervisor(turn=make_turn("Nothing ran.", unavailable=("earthdata",)))
        frames = await self.collect.run(make_chat(supervisor, self.root))

        wording = unavailable_wording("earthdata")
        expected = {"run_id": None, "status": "failed", "mode": None, "reason": wording, "connector": "earthdata"}
        done = last(frames, "done")
        self.assertEqual(done["results"], [expected])
        self.assertIsNone(done["label"])
        [turn] = self.transcripts.turns
        self.assertEqual([r.view() for r in turn.results], [expected])
        history = await transcript_history("thread-1", "user-a", ChartService())
        self.assertEqual(history[-1]["results"], [expected])

    async def test_followups_reach_done(self):
        supervisor = PlaybackSupervisor(turn=make_turn("Done.", followups=("And in July?",)))
        frames = await self.collect.run(make_chat(supervisor, self.root))
        self.assertEqual(last(frames, "done")["suggested_followups"], ["And in July?"])

    async def test_a_turn_with_no_results_has_no_label(self):
        frames = await self.collect.run(make_chat(PlaybackSupervisor(("text", "Hello.")), self.root))
        done = last(frames, "done")
        self.assertEqual((done["results"], done["label"]), ([], None))
        self.assertNotIn("suggested_followups", done)
        self.assertNotIn("variable_choice", done)

    async def test_the_label_is_the_first_results_mode(self):
        first = Result(status="done", mode=Mode.validated("trend", 2), trace_id=RUN_A)
        second = Result(status="done", mode=Mode.exploratory(), trace_id=RUN_B)
        frames = await self.collect.run(
            make_chat(PlaybackSupervisor(turn=make_turn("Done.", results=(first, second))), self.root),
        )
        done = last(frames, "done")
        self.assertEqual(done["label"], "trend v2")
        self.assertEqual([r["mode"]["label"] for r in done["results"]], ["trend v2", "Exploratory"])
        self.assertEqual(self.transcripts.turns[0].label, Mode.validated("trend", 2))

    async def test_an_exception_ends_with_the_error_frame_alone(self):
        supervisor = PlaybackSupervisor(("text", "Working"), ("raise", RuntimeError("boom")))
        with self.assertLogs("tta_backend.services.harness_chat", level="ERROR"):
            frames = await self.collect.run(make_chat(supervisor, self.root))

        self.assertEqual(events(frames), ["text", "error"])
        self.assertEqual(last(frames, "error"), {"detail": render_error_answer("contract", "request")})
        [turn] = self.transcripts.turns
        self.assertEqual(turn.status, "error")
        self.assertTrue(turn.answer_text.startswith("Working\n\n"))

    async def test_admission_refusing_the_turn_ends_with_the_overloaded_answer(self):
        supervisor = PlaybackSupervisor(("raise", admission.AdmissionOverloaded("busy")))
        frames = await self.collect.run(make_chat(supervisor, self.root))

        self.assertEqual(events(frames), ["error", "done"])
        self.assertEqual(last(frames, "error"), {"detail": render_overloaded_answer()})
        self.assertEqual(last(frames, "done")["response"], render_overloaded_answer())
        self.assertEqual(self.transcripts.turns[0].status, "error")

    async def test_the_turn_deadline_ends_with_the_timeout_answer(self):
        blocked = asyncio.Event()
        supervisor = PlaybackSupervisor(blocked)
        frames = await self.collect.run(make_chat(supervisor, self.root, timeout=0.05))

        self.assertEqual(events(frames), ["error", "done"])
        self.assertEqual(last(frames, "done")["response"], render_turn_timeout_answer([]))
        [turn] = self.transcripts.turns
        self.assertEqual((turn.status, turn.answer_text), ("error", render_turn_timeout_answer([])))

    async def test_a_stopped_turn_keeps_what_it_said_as_a_stopped_transcript(self):
        blocked = asyncio.Event()
        supervisor = PlaybackSupervisor(("text", "Half an answer"), blocked)
        chat = make_chat(supervisor, self.root)
        frames: list[str] = []

        async def consume() -> None:
            async for frame in chat.stream_chat_events("hi", "thread-1", "user-a", "req-1"):
                frames.append(frame)

        task = asyncio.create_task(consume())
        while not frames:
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        [turn] = self.transcripts.turns
        self.assertEqual((turn.status, turn.answer_text), ("stopped", "Half an answer"))

    async def test_silence_is_filled_with_a_heartbeat(self):
        blocked = asyncio.Event()
        supervisor = PlaybackSupervisor(blocked, ("text", "Done."))
        chat = make_chat(supervisor, self.root)
        asyncio.get_running_loop().call_later(0.3, blocked.set)
        with patch("tta_backend.utils.streaming.HEARTBEAT_INTERVAL_SECONDS", 0.05):
            frames = await self.collect.run(chat)

        beats = [data for event, data in parse(frames) if event == "status"]
        self.assertTrue(beats)
        self.assertEqual(beats[0]["stage"], "working")
        self.assertEqual(events(frames)[-2:], ["text", "done"])


class Hanging:
    """A supervisor that waits forever, noting when it is cancelled."""

    def __init__(self) -> None:
        self.cancelled = False

    async def turn(self, context: Any, build_env: Any, earthdata: Any, message: str) -> Any:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Watching(PlaybackSupervisor):
    """A playback supervisor noting whether the process is marked busy while it runs."""

    def __init__(self, *steps: Any, **kw: Any) -> None:
        super().__init__(*steps, **kw)
        self.busy_during: list[bool] = []

    async def turn(self, context: Any, build_env: Any, earthdata: Any, message: str) -> Any:
        self.busy_during.append(cube_cache.turn_is_active())
        result = await super().turn(context, build_env, earthdata, message)
        await asyncio.sleep(0)
        self.busy_during.append(cube_cache.turn_is_active())
        return result


class TurnLifetimeTests(unittest.IsolatedAsyncioTestCase):
    """What the host does around a turn that runs long, is cut off, or keeps talking."""

    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.transcripts = MemoryTranscripts()
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(self.transcripts, self.charts))
        self.collect = Collecting(self.transcripts)

    async def test_a_turn_past_its_deadline_has_its_running_work_cancelled(self):
        supervisor = Hanging()
        frames = await self.collect.run(make_chat(supervisor, self.root, timeout=0.05))

        self.assertEqual(events(frames), ["error", "done"])
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertTrue(supervisor.cancelled)

    async def test_a_slow_turn_within_its_deadline_completes_normally(self):
        pause = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, pause.set)
        supervisor = PlaybackSupervisor(("text", "Taking a while. "), pause, ("text", "Done."))
        frames = await self.collect.run(make_chat(supervisor, self.root, timeout=5.0))

        self.assertEqual(events(frames), ["text", "text", "done"])
        self.assertEqual(last(frames, "done")["response"], "Taking a while. Done.")
        self.assertEqual(self.transcripts.turns[0].status, "done")

    async def test_a_turn_that_keeps_talking_gets_no_heartbeat(self):
        class Chatty:
            async def turn(self, context: Any, build_env: Any, earthdata: Any, message: str) -> Any:
                for n in range(8):
                    context.events.text(f"{n} ")
                    await asyncio.sleep(0.03)
                return make_turn("0 1 2 3 4 5 6 7 ")

        with patch("tta_backend.utils.streaming.HEARTBEAT_INTERVAL_SECONDS", 0.2):
            frames = await self.collect.run(make_chat(Chatty(), self.root))

        self.assertEqual(events(frames), ["text"] * 8 + ["done"])

    async def test_a_heartbeat_between_two_pieces_of_the_answer_leaves_them_in_order(self):
        blocked = asyncio.Event()
        supervisor = PlaybackSupervisor(("text", "First part. "), blocked, ("text", "Second part."))
        asyncio.get_running_loop().call_later(0.3, blocked.set)
        with patch("tta_backend.utils.streaming.HEARTBEAT_INTERVAL_SECONDS", 0.05):
            frames = await self.collect.run(make_chat(supervisor, self.root))

        names = events(frames)
        self.assertEqual((names[0], names[-2:]), ("text", ["text", "done"]))
        middle = names[1:-2]
        self.assertTrue(middle)
        self.assertEqual(set(middle), {"status"})
        self.assertTrue(all(d["stage"] == "working" for event, d in parse(frames) if event == "status"))
        self.assertEqual(
            [d["content"] for event, d in parse(frames) if event == "text"], ["First part. ", "Second part."],
        )
        self.assertEqual(last(frames, "done")["response"], "First part. Second part.")

    async def test_the_process_is_marked_busy_for_the_whole_turn_and_idle_after_it(self):
        self.assertFalse(cube_cache.turn_is_active())
        supervisor = Watching(("text", "Done."))
        await self.collect.run(make_chat(supervisor, self.root))
        self.assertEqual(supervisor.busy_during, [True, True])
        self.assertFalse(cube_cache.turn_is_active())

        failing = Watching(("raise", RuntimeError("boom")))
        with self.assertLogs("tta_backend.services.harness_chat", level="ERROR"):
            await self.collect.run(make_chat(failing, self.root))
        self.assertEqual(failing.busy_during, [True])
        self.assertFalse(cube_cache.turn_is_active())

    async def test_a_malformed_chart_is_still_sent_with_a_warning_and_a_good_one_has_none(self):
        broken = {"type": "heatmap", "title": "A map with no bounds"}
        supervisor = PlaybackSupervisor(("chart", broken), ("text", "Done."))
        with self.assertLogs("tta_backend.services.harness_chat", level="WARNING") as logged:
            frames = await self.collect.run(make_chat(supervisor, self.root))
        self.assertEqual(events(frames), ["chart", "text", "done"])
        self.assertTrue(any("chart_artifact_reference_invalid" in line for line in logged.output), logged.output)

        supervisor = PlaybackSupervisor(("chart", CHART), ("text", "Done."))
        with self.assertNoLogs("tta_backend.services.harness_chat", level="WARNING"):
            frames = await self.collect.run(make_chat(supervisor, self.root))
        self.assertEqual(events(frames), ["chart", "artifact", "text", "done"])


class StoppedConnectorTests(PackRun):
    """Real workflow runs that stop for a connector, read back as the done frame's ``connector``."""

    def connectors(self) -> StoppedConnectors:
        return StoppedConnectors(self.f.registry, self.catalogue)

    async def summary(self, result: Result) -> dict[str, Any]:
        [summary] = await result_summaries([result], self.f.stores.traces, "user-a", self.connectors())
        return summary.view()

    async def test_a_workflow_that_did_not_start_without_its_connector_names_it(self):
        self.f.context = dataclasses.replace(self.f.context, credentials=StaticCredentials({"earthdata": EDL_SECRET}))
        result = await self.f.start(GroundWorkflowRuns.inputs, name="monitor-readings")

        self.assertEqual(result.status, "needs_input")
        summary = await self.summary(result)
        self.assertEqual((summary["connector"], summary["reason"]), ("epa_aqs", result.reason))

    async def test_a_credential_the_source_rejected_mid_run_names_its_connector(self):
        earthdata = self.service.earthdata()

        def expired(args: dict[str, Any]) -> dict[str, Any]:
            raise FakeToolError("token_expired")

        earthdata.tools["search_datasets"] = expired

        async def open_retrieval(user_id: str, record: Any) -> Any:
            return self.service.files[record.id]

        self.f.build = env_factory(self.root(), earthdata=earthdata, open_retrieval=open_retrieval, places=self.f.places)
        result = await self.f.start({"quantity": NO2, "area": "New Jersey", "dates": JUNE}, name="map")

        self.assertEqual(result.status, "failed")
        summary = await self.summary(result)
        self.assertEqual(summary["connector"], "earthdata")
        self.assertIn("credential has expired", summary["reason"])

    async def test_a_done_result_or_another_failure_names_no_connector(self):
        result, _ = await self.run_workflow("map", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        self.assertEqual(await self.summary(result), {
            "run_id": result.trace_id, "status": "done", "mode": result.mode.view(), "reason": None, "connector": None,
        })
        failed = await self.f.start({"quantity": NO2, "area": "Atlantis", "dates": JUNE}, name="map")
        self.assertNotEqual(failed.status, "done")
        self.assertIsNone((await self.summary(failed))["connector"])

    def root(self) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name


class PickerTests(unittest.TestCase):
    def test_options_become_candidates_the_frontend_picker_reads(self):
        payload = picker_payload({"run_id": RUN_A, "prompt": "Which product?", "options": [
            "TEMPO_NO2", {"label": "OMI NO2", "value": "OMI"}, {"other": 1},
        ]})
        self.assertEqual(payload["message"], "Which product?")
        self.assertEqual(payload["run_id"], RUN_A)
        self.assertEqual([c["name"] for c in payload["candidates"]], ["TEMPO_NO2", "OMI NO2", '{"other": 1}'])
        for candidate in payload["candidates"]:
            # extractVariableChoice keeps a candidate only with a string name and a non-empty prompt.
            self.assertIsInstance(candidate["name"], str)
            self.assertEqual(candidate["prompt"], candidate["name"])
            self.assertEqual(candidate["category"], "distinct")


class CredentialTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_connector_resolves_the_users_own_secret_and_nothing_else(self):
        from tta_backend.services.connector_credential_service import AqsCredentials

        edl = AsyncMock()
        edl.resolve = AsyncMock(return_value="edl-token")
        aqs = AsyncMock()
        aqs.resolve_credentials = AsyncMock(return_value=AqsCredentials(email="a@example.test", key="k"))
        credentials = ConnectorCredentials("user-a", edl, aqs)

        self.assertEqual(await credentials.secret("earthdata"), "edl-token")
        self.assertEqual(json.loads(await credentials.secret("epa_aqs")), {"email": "a@example.test", "key": "k"})
        self.assertIsNone(await credentials.secret("unknown"))
        edl.resolve.assert_awaited_with("user-a")
        aqs.resolve_credentials.assert_awaited_with("user-a")

    async def test_a_connector_not_connected_has_no_secret(self):
        edl = AsyncMock()
        edl.resolve = AsyncMock(return_value=None)
        aqs = AsyncMock()
        aqs.resolve_credentials = AsyncMock(return_value=None)
        credentials = ConnectorCredentials("user-a", edl, aqs)
        self.assertIsNone(await credentials.secret("earthdata"))
        self.assertIsNone(await credentials.secret("epa_aqs"))


class EarthdataGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_calls_fail_as_provider_unavailable_until_the_connection_is_ready(self):
        class Manager:
            state = "connecting"

        manager = Manager()
        tool = AsyncMock()
        session = ReadyEarthdata(manager, {"search_datasets": tool})
        with self.assertRaises(MCPToolError) as raised:
            await session.call("search_datasets", {})
        self.assertEqual(raised.exception.category, "provider_unavailable")

        manager.state = STATE_READY
        with patch("tta_backend.services.harness_host.call_tool", AsyncMock(return_value='{"ok": true}')), \
             patch("tta_backend.services.harness_host.parse_tool_result", lambda raw: {"ok": True}):
            self.assertEqual(await session.call("search_datasets", {}), {"ok": True})


class TranscriptHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_turn_is_a_user_and_an_assistant_message_with_its_charts_resolved(self):
        from tta_backend.models.artifact import ArtifactReference
        from tta_backend.models.transcript import TranscriptResult, TranscriptTurn

        transcripts = MemoryTranscripts()
        charts = MemoryCharts()
        stored = await charts.save_chart("thread-1", CHART, "user-a")
        transcripts.turns = [
            TranscriptTurn(
                thread_id="thread-1", user_id="user-a", turn_index=0, user_text="Map it", answer_text="Mapped.",
                charts=[{"chart_id": stored["chart_id"], "type": "timeseries", "reference": True},
                        {"chart_id": "gone", "type": "timeseries", "reference": True}],
                artifacts=[ArtifactReference(id="tbl_1", type="table", title="T", row_count=1)],
                label=Mode.validated("trend", 2),
                results=[
                    TranscriptResult(run_id=RUN_A, status="done", mode=Mode.validated("trend", 2)),
                    TranscriptResult(run_id=RUN_B, status="needs_input", mode=Mode.exploratory(),
                                     reason="Connect it.", connector="epa_aqs"),
                ],
                suggested_followups=["And last year?"],
            ),
            TranscriptTurn(thread_id="thread-1", user_id="user-a", turn_index=1, user_text="Thanks",
                           answer_text="Welcome.", status="stopped"),
        ]
        with memory_repositories(transcripts, charts):
            messages = await transcript_history("thread-1", "user-a", ChartService())

        self.assertEqual(messages, [
            {"role": "user", "content": "Map it", "toolCalls": []},
            {"role": "assistant", "content": "Mapped.", "toolCalls": [], "charts": [stored],
             "artifacts": [{"id": "tbl_1", "type": "table", "title": "T", "row_count": 1, "metadata": {}}],
             "label": "trend v2",
             "results": [
                 {"run_id": RUN_A, "status": "done", "reason": None, "connector": None,
                  "mode": {"kind": "validated", "label": "trend v2", "workflow": {"name": "trend", "version": 2}}},
                 {"run_id": RUN_B, "status": "needs_input", "reason": "Connect it.", "connector": "epa_aqs",
                  "mode": {"kind": "exploratory", "label": "Exploratory", "workflow": None}},
             ],
             "suggested_followups": ["And last year?"]},
            {"role": "user", "content": "Thanks", "toolCalls": []},
            {"role": "assistant", "content": "Welcome.", "toolCalls": [], "charts": [], "artifacts": [],
             "label": None, "results": [], "suggested_followups": []},
        ])
        with memory_repositories(transcripts, charts):
            self.assertEqual(await transcript_history("thread-1", "user-b", ChartService()), [])


_REQUIRED = ["fastapi", "httpx", "jwt"]


@requires_redis
@unittest.skipIf(
    any(importlib.util.find_spec(module) is None for module in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class ChatEndpointUsesTheHarnessTests(unittest.IsolatedAsyncioTestCase):
    """``POST /chat`` (detached) runs the turn on the app's chat host, and the history endpoint reads the transcript."""

    async def asyncSetUp(self) -> None:
        import httpx

        import tta_backend.api as api
        from tta_backend.infrastructure.settings import get_settings
        from tta_backend.services.turn_event_log import TurnEventLog
        from tta_backend.services.turn_registry import TurnRegistry

        self.httpx = httpx
        self.api = api
        self.enterContext(patch.dict(os.environ, {"CHAT_DETACHED_TURNS_ENABLED": "1"}))
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)
        self.log = TurnEventLog(REDIS_URL)
        self.addAsyncCleanup(self.log.aclose)
        self.registry = TurnRegistry(self.log, url=REDIS_URL)
        self.addAsyncCleanup(self.registry.aclose)
        api.app.state.turn_event_log = self.log
        api.app.state.turn_registry = self.registry
        self.addCleanup(setattr, api.app.state, "turn_registry", None)
        self.addCleanup(setattr, api.app.state, "turn_event_log", None)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.supervisor = PlaybackSupervisor(("status", {"message": "Working", "stage": "workflow.map"}),
                                             ("chart", CHART), ("text", "The answer."))
        api.app.state.agent = make_chat(self.supervisor, tmp.name)
        self.addCleanup(setattr, api.app.state, "agent", None)

        self.transcripts = MemoryTranscripts()
        self.charts = MemoryCharts()
        self.enterContext(memory_repositories(self.transcripts, self.charts))
        self.user = auth_helpers.user("user-1", email="tester@example.com")
        self.headers = {"Authorization": f"Bearer {auth_helpers.make_token(self.user.id, email=self.user.email)}"}
        self.thread_id = str(uuid.uuid4())

        async def nothing(*args: Any) -> None:
            return None

        async def metadata(thread_id: str) -> dict[str, str]:
            return {"user_id": self.user.id}

        async def owns(thread_id: str, user_id: str) -> bool:
            return user_id == self.user.id

        self.enterContext(auth_helpers.patch_verifier())
        for name, value in (("save_session_metadata_once", nothing), ("mark_session_activity", nothing),
                            ("get_session_metadata", metadata), ("session_belongs_to_user", owns)):
            self.enterContext(patch.object(api, name, value))

    async def test_a_detached_turn_runs_on_the_harness_and_history_reads_its_transcript(self):
        async with self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app), base_url="http://testserver",
        ) as client:
            posted = await client.post("/chat", json={"message": "Map NO2", "thread_id": self.thread_id},
                                       headers=self.headers)
            self.assertEqual(posted.status_code, 202, posted.text)
            turn_id = posted.json()["turn_id"]
            await asyncio.wait_for(self.registry.wait(turn_id), timeout=10.0)
            replay = await client.get(f"/chat/{self.thread_id}/stream", params={"turn": turn_id},
                                      headers=self.headers)
            history = await client.get(f"/session/{self.thread_id}/history", headers=self.headers)

        self.assertEqual(self.supervisor.messages, ["Map NO2"])
        [context] = self.supervisor.contexts
        self.assertEqual((context.identity.user_id, context.identity.conversation_id), (self.user.id, self.thread_id))
        for name in ("event: status", "event: chart", "event: artifact", "event: text", "event: done"):
            self.assertIn(name, replay.text)
        self.assertIn('"response": "The answer."', replay.text)

        self.assertEqual(history.status_code, 200)
        user, assistant = history.json()["messages"]
        self.assertEqual(user, {"role": "user", "content": "Map NO2", "toolCalls": []})
        self.assertEqual(assistant["content"], "The answer.")
        self.assertEqual([c["chart_id"] for c in assistant["charts"]], ["chart-1"])
        self.assertEqual(assistant["charts"][0]["series"], [])
        self.assertEqual([a["id"] for a in assistant["artifacts"]], ["chart-1"])
        self.assertIsNone(assistant["label"])


@unittest.skipIf(
    any(importlib.util.find_spec(module) is None for module in _REQUIRED),
    "chat endpoint dependencies are not installed",
)
class ReconnectedEarthdataTests(unittest.IsolatedAsyncioTestCase):
    """A turn's earthdata session reads the app's live tools at each call, so a reconnect mid-turn reaches it."""

    async def test_calls_after_the_connection_is_replaced_use_the_new_sessions_tools(self):
        import tta_backend.api as api

        class Manager:
            state = STATE_READY

        live: dict[str, Any] = {"search_datasets": "dead-session-tool", "check_coverage": "dead-session-coverage"}
        previous = getattr(api.app.state, "earthdata_mcp_tools", None)
        api.app.state.earthdata_mcp_tools = live
        self.addCleanup(setattr, api.app.state, "earthdata_mcp_tools", previous)
        # Built when the turn starts, before the reconnect.
        session = ReadyEarthdata(Manager(), live)
        called: list[Any] = []

        async def call_tool(tool: Any, arguments: dict[str, Any]) -> str:
            called.append(tool)
            return json.dumps({"used": tool})

        with patch("tta_backend.services.harness_host.call_tool", call_tool):
            self.assertEqual(await session.call("search_datasets", {}), {"used": "dead-session-tool"})
            await api._on_earthdata_mcp_ready({"search_datasets": "new-session-tool"})
            self.assertIs(api.app.state.earthdata_mcp_tools, live)
            self.assertEqual(await session.call("search_datasets", {}), {"used": "new-session-tool"})
            # A tool the new session no longer offers is gone, not served from the dead session.
            with self.assertRaises(MCPToolError):
                await session.call("check_coverage", {})
        self.assertEqual(called, ["dead-session-tool", "new-session-tool"])


if __name__ == "__main__":
    unittest.main()
