"""Paused runs in the app: the runs endpoints, cancelling one, and continuing one through the chat host."""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import sys
import unittest
from typing import Any
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from harness_chat_helpers import (  # noqa: E402
    MemoryCharts,
    MemoryTranscripts,
    PlaybackSupervisor,
    make_chat,
    memory_repositories,
)
from tests.test_harness_chat import last  # noqa: E402
from tests.test_harness_ledger import SHORT_WAIT, STILL_RUNNING, TickingClock  # noqa: E402
from tests.test_harness_specialists import Model  # noqa: E402
from tests.test_harness_supervisor import MAP_INPUTS, BrokenEarthdata, SupervisorCase, ledger_text  # noqa: E402
from tta_backend.harness.results import Result  # noqa: E402
from tta_backend.harness.supervisor.supervisor import (  # noqa: E402
    CONTINUE_GONE,
    CONTINUE_NOT_READY,
    CONTINUE_UNCHECKED,
    AppAnswer,
    ResumeRequest,
)
from tta_backend.harness.testing import answer  # noqa: E402
from tta_backend.harness.workflows.runner import EXPIRED_REASON, RunNotResumable, cancel_paused  # noqa: E402
from tta_backend.services import paused_runs  # noqa: E402
from tta_backend.services.harness_chat import TurnPorts, question_payload  # noqa: E402

class Collecting:
    """Runs a turn and collects its frames."""

    def __init__(self, transcripts: MemoryTranscripts) -> None:
        self.transcripts = transcripts

    async def run(
        self, chat: Any, message: str = "hi", thread_id: str = "conv-1", user_id: str = "user-a", **request: Any,
    ) -> list[str]:
        return [frame async for frame in chat.stream_chat_events(message, thread_id, user_id, "req-1", **request)]


LARGE = 100_000_000
OTHER = "run_" + "0" * 32
_REQUIRED = ["fastapi", "httpx", "jwt"]


@unittest.skipIf(any(importlib.util.find_spec(m) is None for m in _REQUIRED), "endpoint dependencies are not installed")
class PausedRunsCase(SupervisorCase):
    """The real supervisor on scripted models behind the app's chat host and its runs endpoints."""

    async def asyncSetUp(self) -> None:
        import httpx

        import tta_backend.api as api

        await super().asyncSetUp()
        self.httpx, self.api = httpx, api
        self.model = Model(script=[])
        self.transcripts, self.charts = MemoryTranscripts(), MemoryCharts()
        self.enterContext(memory_repositories(self.transcripts, self.charts))
        self.collect = Collecting(self.transcripts)
        self.enterContext(auth_helpers.patch_verifier())
        self.headers = {"Authorization": f"Bearer {auth_helpers.make_token('user-a', email='a@example.com')}"}
        self.other_headers = {"Authorization": f"Bearer {auth_helpers.make_token('user-b', email='b@example.com')}"}
        self.chat = self.host()
        self.addCleanup(setattr, api.app.state, "agent", None)

    def host(self, clock: Any = None) -> Any:
        chat = make_chat(
            self.supervisor(self.model), self.root, stores=self.f.stores,
            ports=lambda user_id, report_job: TurnPorts(build_env=self.build, earthdata=self.service.earthdata()),
            credentials={"earthdata": "secret"}, clock=clock or self.f.context.clock,
        )
        self.api.app.state.agent = chat
        return chat

    async def ask(self) -> Result:
        """A map run paused on a question."""
        self.service.estimated_bytes = LARGE
        result = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual(result.status, "needs_input")
        return result

    async def wait(self, *, ready: bool = False) -> Result:
        """A map run waiting on its retrieval job; ``ready`` finishes the job afterwards."""
        self.build = self.env_factory(limits=SHORT_WAIT)
        self.service.statuses = list(STILL_RUNNING)
        result = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual(result.status, "waiting")
        if ready:
            self.service.statuses = []
        return result

    async def get(self, path: str, headers: dict[str, str] | None = None, **params: str) -> Any:
        async with self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app), base_url="http://testserver",
        ) as client:
            return await client.get(path, params=params, headers=self.headers if headers is None else headers)

    async def post(self, path: str, headers: dict[str, str] | None = None, **body: Any) -> Any:
        async with self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app), base_url="http://testserver",
        ) as client:
            return await client.post(path, json=body or None, headers=self.headers if headers is None else headers)

    async def runs(self, thread: str = "conv-1", headers: dict[str, str] | None = None) -> list[dict[str, Any]]:
        response = await self.get("/runs", headers, thread_id=thread)
        self.assertEqual(response.status_code, 200, response.text)
        runs: list[dict[str, Any]] = response.json()["runs"]
        return runs


class ListRunsTests(PausedRunsCase):
    async def test_a_run_waiting_on_a_question_lists_with_the_question_the_done_frame_carries(self):
        asked = await self.ask()

        [row] = await self.runs()

        self.assertEqual(row["run_id"], asked.trace_id)
        self.assertEqual(row["status"], "needs_input")
        self.assertEqual(row["mode"], asked.mode.view())
        self.assertEqual(row["reason"], asked.reason)
        self.assertEqual(row["question"], question_payload(asked.question))
        self.assertEqual(row["question"]["kind"], "pick_one")
        self.assertNotIn("ready", row)
        started = datetime.datetime.fromisoformat(row["started"])
        self.assertLessEqual(started, datetime.datetime.fromisoformat(row["paused"]))
        self.assertEqual(self.model.calls, 0)

    async def test_a_waiting_run_is_marked_ready_once_its_jobs_finish_and_nothing_is_resumed(self):
        waiting = await self.wait()
        [row] = await self.runs()
        self.assertEqual((row["status"], row["ready"], row["reason"]), ("waiting", False, waiting.reason))
        self.assertNotIn("question", row)

        self.service.statuses = []
        [row] = await self.runs()

        self.assertEqual((row["run_id"], row["ready"]), (waiting.trace_id, True))
        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "waiting")
        self.assertEqual(self.model.calls, 0)

    async def test_only_paused_runs_list(self):
        done = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual(done.status, "done")
        self.assertEqual(await self.runs(), [])

    async def test_another_users_and_another_conversations_runs_never_appear(self):
        await self.ask()

        self.assertEqual(await self.runs(headers=self.other_headers), [])
        self.assertEqual(await self.runs(thread="conv-2"), [])
        self.assertEqual(len(await self.runs()), 1)

    async def test_a_run_paused_past_its_expiry_is_cancelled_and_not_listed(self):
        asked = await self.ask()
        self.host(clock=TickingClock(self.f.context.clock.now() + datetime.timedelta(days=31)))

        self.assertEqual(await self.runs(), [])

        saved = await self.f.stores.results.get("user-a", asked.trace_id)
        self.assertEqual((saved.status, saved.reason), ("cancelled", EXPIRED_REASON))

    async def test_the_endpoints_need_authentication_and_a_thread(self):
        unauthenticated = self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app), base_url="http://testserver",
        )
        async with unauthenticated as client:
            self.assertEqual((await client.get("/runs", params={"thread_id": "conv-1"})).status_code, 401)
            self.assertEqual((await client.post(f"/runs/{OTHER}/cancel")).status_code, 401)
        self.assertEqual((await self.get("/runs")).status_code, 422)
        self.assertEqual((await self.get("/runs", thread_id="a b")).status_code, 422)

    async def test_an_unready_chat_host_is_a_503(self):
        self.api.app.state.agent = None
        with patch.object(self.api, "agent", None):
            self.assertEqual((await self.get("/runs", thread_id="conv-1")).status_code, 503)


class CancelRunTests(PausedRunsCase):
    async def test_cancelling_a_question_run_finishes_it_without_a_model_call(self):
        asked = await self.ask()

        response = await self.post(f"/runs/{asked.trace_id}/cancel")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["run_id"], body["status"]), (asked.trace_id, "cancelled"))
        self.assertEqual((await self.f.stores.traces.get_run("user-a", asked.trace_id)).status, "cancelled")
        self.assertEqual((await self.f.stores.results.get("user-a", asked.trace_id)).status, "cancelled")
        self.assertEqual(await self.runs(), [])
        self.assertEqual(self.model.calls, 0)

    async def test_cancelling_a_waiting_run_leaves_its_provider_job_alone(self):
        waiting = await self.wait()
        jobs = dict(self.service.jobs)

        response = await self.post(f"/runs/{waiting.trace_id}/cancel")

        self.assertEqual((response.status_code, response.json()["status"]), (200, "cancelled"))
        self.assertEqual(self.service.jobs, jobs)
        self.assertEqual(await self.runs(), [])

    async def test_a_run_that_is_not_paused_gone_or_another_users_is_a_404_and_stays_as_it_was(self):
        asked = await self.ask()

        self.assertEqual((await self.post(f"/runs/{asked.trace_id}/cancel", self.other_headers)).status_code, 404)
        self.assertEqual((await self.post(f"/runs/{OTHER}/cancel")).status_code, 404)
        self.assertEqual((await self.post("/runs/run_1/cancel")).status_code, 422)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", asked.trace_id)).status, "needs_input")

        self.assertEqual((await self.post(f"/runs/{asked.trace_id}/cancel")).status_code, 200)
        self.assertEqual((await self.post(f"/runs/{asked.trace_id}/cancel")).status_code, 404)

    async def test_a_run_resumed_before_the_cancel_lands_is_not_cancelled(self):
        waiting = await self.wait()
        run = await self.f.stores.traces.get_run("user-a", waiting.trace_id)
        # A stale Cancel read the run as waiting; the resume got there first.
        await self.f.stores.traces.resume_run("user-a", waiting.trace_id)

        with self.assertRaises(RunNotResumable):
            await cancel_paused(self.f.context, self.f.runner, self.build, run)
        self.assertEqual((await self.post(f"/runs/{waiting.trace_id}/cancel")).status_code, 404)

        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "running")
        saved = await self.f.stores.results.get("user-a", waiting.trace_id)
        self.assertEqual(saved.status, "waiting")

    async def test_a_finished_run_is_not_cancelled(self):
        done = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual((await self.post(f"/runs/{done.trace_id}/cancel")).status_code, 404)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", done.trace_id)).status, "done")


class ContinueRunTests(PausedRunsCase):
    async def test_a_ready_run_is_resumed_before_any_model_call_and_the_model_writes_only_the_prose(self):
        waiting = await self.wait(ready=True)
        seen: list[str] = []

        def prose(messages: Any) -> Any:
            from langchain_core.messages import AIMessage, ToolMessage

            seen.append(json.loads([m for m in messages if isinstance(m, ToolMessage)][-1].text)["status"])
            call = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls)
            seen.append(json.dumps(call.tool_calls[0]["args"]))
            return answer("The map is ready.")

        self.model.script.append(prose)

        frames = await self.collect.run(
            self.chat, "Continue the paused run", resume={"run_id": waiting.trace_id},
        )

        self.assertEqual(seen, ["done", json.dumps({"run_id": waiting.trace_id})])
        self.assertEqual(self.model.calls, 1)
        done = last(frames, "done")
        [result] = done["results"]
        self.assertEqual((result["run_id"], result["status"]), (waiting.trace_id, "done"))
        self.assertIn("The map is ready.", done["response"])
        self.assertEqual(self.transcripts.turns[0].user_text, "Continue the paused run")
        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "done")

    async def test_a_run_whose_jobs_are_not_finished_is_not_continued(self):
        waiting = await self.wait()

        frames = await self.collect.run(self.chat, "Continue", resume={"run_id": waiting.trace_id})

        done = last(frames, "done")
        self.assertEqual(done["response"], CONTINUE_NOT_READY)
        self.assertEqual(done["results"], [])
        self.assertEqual(self.model.calls, 0)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "waiting")

    async def test_a_run_that_is_not_waiting_or_not_this_conversations_is_refused_with_no_model_call(self):
        asked = await self.ask()
        self.service.estimated_bytes = 1000
        waiting = await self.wait(ready=True)
        for thread, run, user in (
            ("conv-1", asked.trace_id, "user-a"), ("conv-1", OTHER, "user-a"),
            ("conv-2", waiting.trace_id, "user-a"), ("conv-1", waiting.trace_id, "user-b"),
        ):
            with self.subTest(thread=thread, run=run, user=user):
                frames = await self.collect.run(
                    self.chat, "Continue", thread_id=thread, user_id=user, resume={"run_id": run},
                )
                done = last(frames, "done")
                self.assertTrue(done["response"].startswith(CONTINUE_GONE), done["response"])
        self.assertEqual(self.model.calls, 0)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "waiting")

    async def test_a_refused_continue_of_a_run_on_a_question_asks_it_again(self):
        asked = await self.ask()
        done = last(
            await self.collect.run(self.chat, "Continue", resume={"run_id": asked.trace_id}), "done",
        )
        self.assertEqual(done["question"]["run_id"], asked.trace_id)

    async def test_a_run_that_pauses_again_after_continue_is_not_shown_ready_to_the_model(self):
        waiting = await self.wait(ready=True)

        class ReadyOnce:
            """Says the job is ready the first time it is asked, running after."""

            def __init__(self, inner: Any) -> None:
                self.inner, self.asked = inner, 0

            async def call(self, tool: str, arguments: dict[str, Any], **kw: Any) -> dict[str, Any]:
                if tool != "get_retrieval_status":
                    return await self.inner.call(tool, arguments, **kw)
                self.asked += 1
                if self.asked == 1:
                    return {"job_handle": arguments["job_handle"], "status": "ready"}
                return {"job_handle": arguments["job_handle"], "status": "running", "phase": "processing"}

        # The resume polls through the build's own session, which sees the job still running.
        self.service.statuses = list(STILL_RUNNING)
        self.model.script.append(answer("Still waiting."))
        turn = await self.supervisor(self.model).turn(
            self.f.context, self.build, ReadyOnce(self.service.earthdata()), "Continue the paused run",
            resume=ResumeRequest(run_id=waiting.trace_id),
        )

        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (waiting.trace_id, "waiting"))
        ledger = ledger_text(self.model.received[0])
        self.assertIn("still running", ledger)
        self.assertNotIn("Its retrieval jobs are ready", ledger)

    async def test_a_failed_job_check_is_not_said_to_be_a_run_that_is_not_waiting(self):
        waiting = await self.wait(ready=True)

        turn = await self.supervisor(self.model).turn(
            self.f.context, self.build, BrokenEarthdata(), "Continue the paused run",
            resume=ResumeRequest(run_id=waiting.trace_id),
        )

        self.assertTrue(turn.text.startswith(CONTINUE_UNCHECKED), turn.text)
        self.assertEqual((turn.model_calls, turn.results), (0, ()))
        self.assertEqual(self.model.calls, 0)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", waiting.trace_id)).status, "waiting")

    async def test_the_supervisor_takes_an_answer_or_a_resume_not_both(self):
        asked = await self.ask()
        with self.assertRaises(ValueError):
            await self.supervisor(self.model).turn(
                self.f.context, self.build, self.service.earthdata(), "x",
                answer=AppAnswer(run_id=asked.trace_id, step="data", value="confirm"),
                resume=ResumeRequest(run_id=asked.trace_id),
            )


@unittest.skipIf(any(importlib.util.find_spec(m) is None for m in _REQUIRED), "endpoint dependencies are not installed")
class ChatRequestResumeTests(unittest.IsolatedAsyncioTestCase):
    """``POST /chat`` takes ``resume`` beside ``answer``, never both, and hands it to the turn."""

    async def asyncSetUp(self) -> None:
        import tempfile

        import httpx

        import tta_backend.api as api
        from tta_backend.infrastructure.settings import get_settings

        self.httpx, self.api = httpx, api
        self.enterContext(patch.dict(os.environ, {"CHAT_DETACHED_TURNS_ENABLED": "0"}))
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.supervisor = PlaybackSupervisor(("text", "Continued."))
        api.app.state.agent = make_chat(self.supervisor, tmp.name)
        self.addCleanup(setattr, api.app.state, "agent", None)
        self.enterContext(memory_repositories(MemoryTranscripts(), MemoryCharts()))
        self.enterContext(auth_helpers.patch_verifier())
        self.headers = {"Authorization": f"Bearer {auth_helpers.make_token('user-1', email='a@example.com')}"}

        async def nothing(*args: Any) -> None:
            return None

        async def metadata(thread_id: str) -> dict[str, str]:
            return {"user_id": "user-1"}

        for name, value in (("save_session_metadata_once", nothing), ("mark_session_activity", nothing),
                            ("get_session_metadata", metadata)):
            self.enterContext(patch.object(api, name, value))

    async def post(self, body: dict[str, Any]) -> Any:
        async with self.httpx.AsyncClient(
            transport=self.httpx.ASGITransport(app=self.api.app), base_url="http://testserver",
        ) as client:
            return await client.post("/chat", json=body, headers=self.headers)

    async def test_a_resume_reaches_the_turn_and_is_refused_beside_an_answer_or_when_malformed(self):
        run = "run_" + "a" * 32
        accepted = await self.post(
            {"message": "Continue the paused run", "thread_id": "thread-1", "resume": {"run_id": run}},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertIn("event: done", accepted.text)
        [given] = self.supervisor.resumes
        self.assertEqual(given.run_id, run)
        self.assertEqual(self.supervisor.answers, [None])

        both = {"message": "x", "thread_id": "thread-1", "resume": {"run_id": run},
                "answer": {"run_id": run, "step": "days", "value": "x"}}
        for bad in (both, {"message": "x", "resume": {"run_id": "run_1"}}, {"message": "x", "resume": {}}):
            with self.subTest(bad=bad):
                self.assertEqual((await self.post(bad)).status_code, 422)
        self.assertEqual(len(self.supervisor.resumes), 1)

    async def test_a_plain_message_gives_the_turn_neither(self):
        await self.post({"message": "hello", "thread_id": "thread-1"})
        self.assertEqual((self.supervisor.answers, self.supervisor.resumes), ([None], [None]))


class RunRowTests(unittest.TestCase):
    def test_an_exploration_is_never_ready_to_continue(self):
        from tta_backend.harness.results import Mode
        from tta_backend.harness.trace import Run, new_run_id

        run = Run(id=new_run_id(), user_id="u", conversation_id="c", mode=Mode.exploratory(),
                  started=datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC), status="waiting")
        self.assertEqual(paused_runs.run_row(run, None, "ready")["ready"], False)
        self.assertIsNone(paused_runs.run_row(run, None, "ready")["reason"])


if __name__ == "__main__":
    unittest.main()
