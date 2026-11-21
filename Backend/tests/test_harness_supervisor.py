"""The supervisor run on scripted models: tools, ledger, limits, answer assembly, streaming and checkpoints."""
import asyncio
import dataclasses
import datetime
import json
import tempfile
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from tests.test_harness_ledger import SHORT_WAIT, STILL_RUNNING, TickingClock
from tests.test_harness_sources_satellite import NO2
from tests.test_harness_specialists import Model, call_on_area, choice_of, final, input_handle
from tests.test_harness_workflow_packs_run import JUNE, PackRun
from tta_backend.harness.context.ledger import EMPTY, HEADING
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF, ModelChoice, NamedModel
from tta_backend.harness.results import Finding, Gap, Mode, Result
from tta_backend.harness.run_context import RunContext
from tta_backend.harness.specialists import ExploreLimits
from tta_backend.harness.specialists.explore import UNREACHABLE_DISCLOSURE
from tta_backend.harness.supervisor import (
    MAX_TASKS,
    TOOL_NAMES,
    Supervisor,
    SupervisorLimits,
    Turn,
    assemble,
    result_view,
    sources_line,
    thread_id,
    unmatched_numbers,
)
from tta_backend.harness.supervisor.answer import data_sources, unavailable_wording
from tta_backend.harness.supervisor.supervisor import (
    FAILED_WORDING,
    LIMIT_WORDING,
    MAX_FOLLOWUPS,
    NUMBERS_STEP,
    TOOL_LIMIT,
    UNFINISHED_CALL,
)
from tta_backend.harness.supervisor.supervisor import ANSWER_GONE, AppAnswer
from tta_backend.harness.supervisor.tools import (
    ANSWER_NEEDED,
    ASK_AGAIN,
    BAD_ANSWER,
    NOT_ASKING,
    NOT_CANCELLABLE,
    NOT_READY,
    STOP_RETRYING,
    TOO_MANY_TASKS,
    UNKNOWN_WORKFLOW,
    tool_schemas,
)
from tta_backend.harness.workflows.answers import PICK_ONE
from tta_backend.harness.testing import (
    RecordingEvents,
    ScriptedModels,
    answer,
    env_factory,
    rate_limited,
    tool_call,
)
from tta_backend.harness.workflows.runner import EXPIRED_REASON, JOB_WAITING

MAP_INPUTS = {"quantity": NO2, "area": "New Jersey", "dates": JUNE}
RESOLVE = "sources.areas.resolve"
# What a lookup resolving NJ discloses, appended to the answer.
NJ_NOTE = "\n\nNotes:\n- 'NJ' was resolved to New Jersey."
# The keys a compact result view may hold.
VIEW_KEYS = {"run_id", "status", "mode", "findings", "handles", "charts", "reason", "question", "interpretation",
             "followups"}


class Partial:
    """A streamed turn that sends ``text`` and then fails with ``error``."""

    def __init__(self, text: str, error: BaseException) -> None:
        self.text = text
        self.error = error


class Streamed(Model):
    """A scripted model that streams like a provider: its text a line at a
    time, then its tool calls as chunks."""

    async def _astream(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        self.received.append(list(messages))
        self.tools_offered.append(list(kwargs.get("tool_names", [])))
        self.tool_choices.append(kwargs.get("tool_choice"))
        turn = self.script[self.position]
        self.position += 1
        if callable(turn) and not isinstance(turn, AIMessage | BaseException | Partial):
            turn = turn(messages)
        if isinstance(turn, BaseException):
            raise turn
        if isinstance(turn, Partial):
            yield ChatGenerationChunk(message=AIMessageChunk(content=turn.text))
            raise turn.error
        for line in turn.text.splitlines(keepends=True):
            yield ChatGenerationChunk(message=AIMessageChunk(content=line))
        if turn.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(turn.tool_calls)
            ]))


class BrokenEarthdata:
    """An earthdata session whose every call fails unclassified, so waiting runs' jobs go unchecked."""

    async def call(self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        raise RuntimeError("down")


class BrokenRegistry:
    async def get(self, user_id: str, handle_id: str) -> Any:
        raise RuntimeError("the registry is down")


def supervisor_choice(model: Model) -> ModelChoice:
    return ModelChoice(primary=NamedModel(model, "fake:supervisor"))


def run_map(**inputs: Any) -> AIMessage:
    return tool_call("run_workflow", {"name": "map", "inputs": {**MAP_INPUTS, **inputs}})


def tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def last_tool(model: Model, call: int = -1) -> dict[str, Any]:
    return json.loads(tool_messages(model.received[call])[-1].text)


def ledger_text(messages: list[BaseMessage]) -> str:
    """The ledger part of the per-turn message, after the date preamble it opens with."""
    [ledger] = [m for m in messages if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") == "ledger"]
    preamble, _, body = ledger.text.partition("\n\n")
    assert preamble.startswith("Today's date is "), preamble
    return body


def walk(node: Any) -> list[Any]:
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(walk(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(walk(value))
    return found


class SupervisorCase(PackRun):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.f.context = dataclasses.replace(self.f.context, clock=TickingClock(self.f.context.clock.now()))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.saver = InMemorySaver()
        self.explorer = Model(script=[])
        self.build = self.env_factory()

    def env_factory(self, **kw: Any) -> Any:
        return env_factory(
            self.root, earthdata=self.service.earthdata(), open_retrieval=self.open_retrieval, places=self.f.places,
            models=ScriptedModels({"explore": choice_of(self.explorer)}), **kw,
        )

    async def open_retrieval(self, user_id: str, record: Any) -> Any:
        return self.service.files[record.id]

    def supervisor(
        self, model: Model, *, limits: SupervisorLimits | None = None, summary: Model | None = None,
        saver: InMemorySaver | None = None,
    ) -> Supervisor:
        roles: dict[Any, ModelChoice] = {"supervisor": supervisor_choice(model)}
        if summary is not None:
            roles["agent_step"] = supervisor_choice(summary)
        return Supervisor(
            capabilities=self.f.registry, catalogue=self.catalogue, runner=self.f.runner, notes=NoteCatalogue([]),
            checkpointer=saver or self.saver, models=ScriptedModels(roles), limits=limits,
            explore_limits=ExploreLimits(retrieval_wait_seconds=0.05), backoff=NO_BACKOFF,
        )

    async def turn(self, supervisor: Supervisor, message: str = "Map NO2 over New Jersey for June 1 to 3, 2024.",
                   context: RunContext | None = None, build: Any = None, earthdata: Any = None,
                   answer: AppAnswer | None = None) -> Turn:
        return await supervisor.turn(
            context or self.f.context, build or self.build, earthdata or self.service.earthdata(), message,
            answer=answer,
        )

    async def checkpoint(self, context: RunContext | None = None, saver: InMemorySaver | None = None) -> list[BaseMessage]:
        context = context or self.f.context
        saved = await (saver or self.saver).aget_tuple({"configurable": {"thread_id": thread_id(context.identity)}})
        assert saved is not None
        return list(saved.checkpoint["channel_values"]["messages"])

    def context_for(self, user_id: str) -> RunContext:
        return dataclasses.replace(
            self.f.context, identity=dataclasses.replace(self.f.context.identity, user_id=user_id),
            events=RecordingEvents(),
        )


class WorkflowTurnTests(SupervisorCase):
    async def test_a_turn_runs_a_workflow_end_to_end_and_the_next_turn_sees_it_in_the_ledger(self):
        model = Model(script=[run_map(), answer("The NO2 map over New Jersey is ready."), answer("It is ready.")])
        supervisor = self.supervisor(model)

        turn = await self.turn(supervisor)

        self.assertEqual(turn.status, "answered")
        [result] = turn.results
        latest = self.catalogue.latest("map").version
        self.assertEqual((result.status, result.mode), ("done", Mode.draft("map", latest)))
        self.assertEqual(result.mode.label(), f"Draft workflow: map v{latest}")
        self.assertEqual(turn.run_ids, (result.trace_id,))
        self.assertEqual(await self.f.stores.results.get("user-a", result.trace_id), result)
        self.assertTrue(result.disclosures)
        self.assertEqual(
            turn.text,
            f"{sources_line(['TEMPO_NO2'])}\n\n{assemble('The NO2 map over New Jersey is ready.', [result])}",
        )
        for disclosure in result.disclosures:
            self.assertIn(disclosure, turn.text)
        view = last_tool(model, 1)
        self.assertEqual(view, json.loads(result_view(result)))
        self.assertEqual(view["mode"], f"Draft workflow: map v{latest}")
        self.assertEqual((turn.model_calls, turn.tool_calls), (2, 1))

        await self.turn(supervisor, "Is it ready?")
        ledger = ledger_text(model.received[2])
        self.assertTrue(ledger.startswith(HEADING))
        self.assertIn(result.trace_id, ledger)
        self.assertIn(result.handles[0], ledger)

    async def test_the_checkpoint_holds_compact_views_while_the_chart_reaches_the_ui(self):
        model = Model(script=[run_map(), answer("Done.")])
        turn = await self.turn(self.supervisor(model))

        [result] = turn.results
        [chart] = result.charts
        payload = json.dumps(self.f.artifacts.stored[chart])
        self.assertTrue(self.f.events.charts)
        saved = await self.checkpoint()
        for message in saved:
            self.assertNotIn(payload, message.text)
        [tool] = tool_messages(saved)
        view = json.loads(tool.text)
        self.assertLessEqual(set(view), VIEW_KEYS)
        self.assertEqual(view["charts"], [chart])
        self.assertLess(len(tool.text), 1000)
        # Human and model messages are kept as they were.
        self.assertEqual(saved[0].text, "Map NO2 over New Jersey for June 1 to 3, 2024.")
        self.assertEqual(saved[-1].text, "Done.")

    async def test_the_sources_line_is_code_built_and_a_model_written_one_is_dropped(self):
        prose = "Data sources: a dataset the model made up\nThe map is ready.\n\n**Sources:** also made up\nLook north."
        model = Model(script=[run_map(), answer(prose)])
        turn = await self.turn(self.supervisor(model))

        self.assertTrue(turn.text.startswith(f"{sources_line(['TEMPO_NO2'])}\n\nThe map is ready.\n\nLook north."))
        self.assertNotIn("made up", turn.text)
        self.assertEqual(turn.text.count("Data sources:"), 1)

    async def test_satellite_products_come_before_monitors_in_the_sources_line(self):
        ground = {"pollutant": "o3", "area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-02"}}
        model = Model(script=[
            tool_call("run_workflow", {"name": "monitor-readings", "inputs": ground}), run_map(), answer("Both are ready."),
        ])
        turn = await self.turn(self.supervisor(model))

        self.assertEqual([r.status for r in turn.results], ["done", "done"], [r.reason for r in turn.results])
        self.assertTrue(turn.text.startswith(f"{sources_line(['TEMPO_NO2', 'EPA AQS'])}\n\n"))

    async def test_a_dataset_behind_two_results_is_named_once(self):
        later = {"start": "2024-06-10", "end": "2024-06-12"}
        model = Model(script=[run_map(), run_map(dates=later), answer("Both maps are ready.")])
        turn = await self.turn(self.supervisor(model))

        self.assertEqual([r.status for r in turn.results], ["done", "done"])
        self.assertEqual(turn.text.splitlines()[0], "Data sources: TEMPO_NO2")
        self.assertEqual(turn.text.count("Data sources:"), 1)

    async def test_a_turn_with_no_data_results_has_no_sources_line(self):
        model = Model(script=[tool_call("resolve_area", {"place": "NJ"}), answer("New Jersey is a state.")])
        turn = await self.turn(self.supervisor(model), "What is NJ?")
        # The lookup's own disclosure is still appended.
        self.assertEqual(turn.text, "New Jersey is a state." + NJ_NOTE)


class InputTests(SupervisorCase):
    async def test_bad_inputs_come_back_with_the_reason_then_a_fixed_refusal(self):
        bad = {"dates": "June"}
        model = Model(script=[run_map(**bad), run_map(**bad), run_map(), answer("I could not run it.")])
        turn = await self.turn(self.supervisor(model))

        first, second, third = (last_tool(model, i) for i in (1, 2, 3))
        self.assertEqual(first, {"status": "refused", "reason": (
            'Input dates must be a date range with an ISO start date and end date, '
            'like {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.'
        )})
        self.assertEqual(second, {"status": "refused", "reason": STOP_RETRYING})
        # Not run again in this turn, even with good inputs.
        self.assertEqual(third, {"status": "refused", "reason": STOP_RETRYING})
        self.assertEqual(turn.results, ())
        self.assertEqual(turn.text, "I could not run it.")
        # Both refused runs were made by this turn; the third call ran nothing.
        runs = await self.f.stores.traces.list_runs("user-a", "conv-1")
        self.assertEqual(turn.run_ids, tuple(run.id for run in runs))
        self.assertEqual([run.status for run in runs], ["failed", "failed"])
        # Nothing ran under them: no step was written.
        for run in runs:
            self.assertEqual(await self.f.stores.traces.entries("user-a", run.id), [])

    async def test_one_part_of_a_combined_place_is_refused_everywhere_and_runs_nothing(self):
        from tta_backend.harness.supervisor.tools import SPLIT_PLACE

        message = "Map NO2 over NY + Wakanda for June 1 to 3, 2024."
        model = Model(script=[
            tool_call("resolve_area", {"place": "Wakanda"}),
            run_map(area="NY"),
            tool_call("explore", {"tasks": [{"objective": "Map NO2.", "bundle": "analyse", "places": ["Wakanda"]}]}),
            answer("Which place did you mean?"),
        ])
        turn = await self.turn(self.supervisor(model), message)

        expected = {
            1: SPLIT_PLACE.format(place="Wakanda"),
            2: SPLIT_PLACE.format(place="NY"),
            3: SPLIT_PLACE.format(place="Wakanda") + " Nothing was run.",
        }
        for call, reason in expected.items():
            with self.subTest(call=call):
                self.assertEqual(last_tool(model, call), {"status": "refused", "reason": reason})
        self.assertEqual(turn.results, ())
        self.assertEqual(await self.f.stores.traces.list_runs("user-a", "conv-1"), [])

    def test_only_a_side_of_a_plus_the_user_wrote_is_a_part(self):
        from tta_backend.harness.supervisor.tools import part_of_combined_place

        message = "map NO2 over NY + Wakanda for June 1-7 2024"
        for place, part in (
            ("NY", True), ("ny", True), ("Wakanda", True), ("NY + Wakanda", False), ("New York", False), ("Wak", False),
        ):
            with self.subTest(place=place):
                self.assertIs(part_of_combined_place(place, message), part)
        self.assertFalse(part_of_combined_place("Newark", "What was NO2 in Newark yesterday?"))

    async def test_missing_inputs_come_back_to_the_model(self):
        model = Model(script=[tool_call("run_workflow", {"name": "map", "inputs": {"quantity": NO2}}), answer("Which dates?")])
        turn = await self.turn(self.supervisor(model))
        self.assertEqual(last_tool(model)["status"], "refused")
        self.assertIn("The workflow needs more before it can run", last_tool(model)["reason"])
        self.assertEqual(turn.results, ())

    async def test_an_unknown_workflow_is_refused_without_a_run(self):
        model = Model(script=[tool_call("run_workflow", {"name": "nope", "inputs": {}}), answer("No such workflow.")])
        turn = await self.turn(self.supervisor(model))
        self.assertEqual(last_tool(model), {"status": "refused", "reason": UNKNOWN_WORKFLOW})
        self.assertEqual(await self.f.stores.traces.list_runs("user-a", "conv-1"), [])
        self.assertEqual(turn.run_ids, ())


class ExploreAndLookupTests(SupervisorCase):
    async def test_an_explore_call_runs_a_specialist_with_the_place_as_the_user_wrote_it(self):
        self.explorer.script = [
            call_on_area("sources__satellite__get_data", quantity=NO2, dates=JUNE, date_role="range"),
            final("NO2 was retrieved over New Jersey."),
        ]
        task = {"objective": "Retrieve NO2 over NJ.", "bundle": "analyse", "places": ["NJ"]}
        model = Model(script=[tool_call("explore", {"tasks": [task]}), answer("NO2 was retrieved.")])
        turn = await self.turn(self.supervisor(model), "Get NO2 over NJ for June 1-3 2024.")

        [result] = turn.results
        self.assertEqual((result.status, result.mode), ("done", Mode.exploratory()), result.reason)
        [view] = last_tool(model)["results"]
        self.assertEqual(view, json.loads(result_view(result)))
        self.assertEqual(self.f.places.asked, ["NJ"])
        [entry] = [e for e in await self.f.stores.traces.entries("user-a", result.trace_id) if e.kind == "explore"]
        self.assertEqual(entry.params["inputs"]["places"], ["NJ"])
        self.assertIn("Area: New Jersey", self.explorer.received[0][1].text)
        # The specialist's own text never reaches the answer stream.
        self.assertNotIn("NO2 was retrieved over New Jersey.", turn.text)
        self.assertTrue(turn.text.startswith(sources_line(["TEMPO_NO2"]) or ""))

    async def test_a_specialists_narration_and_answer_call_never_reach_the_answer_stream(self):
        narration = "Retrieving the NO2 grid now."
        summary = "The specialist's own account of what it found."
        self.explorer = Streamed(script=[
            lambda m: tool_call("sources__satellite__get_data", {
                "area": input_handle(m, "polygons"), "quantity": NO2, "dates": JUNE, "date_role": "range",
            }, content=narration),
            final(summary),
        ])
        self.build = self.env_factory()
        task = {"objective": "Retrieve NO2 over NJ.", "bundle": "analyse", "places": ["NJ"]}
        model = Streamed(script=[tool_call("explore", {"tasks": [task]}), answer("NO2 was retrieved.")])
        turn = await self.turn(self.supervisor(model), "Get NO2 over NJ for June 1-3 2024.")

        [result] = turn.results
        self.assertEqual((result.status, result.interpretation), ("done", summary), result.reason)
        # The specialist's model did stream, both turns of it.
        self.assertEqual(self.explorer.calls, 2)
        streamed = "".join(self.f.events.texts)
        self.assertNotIn(narration, streamed)
        self.assertNotIn(summary, streamed)
        self.assertNotIn("SpecialistAnswer", streamed)
        # Only the supervisor's own answer is streamed; stage events from the capability calls still arrive.
        self.assertIn("NO2 was retrieved.", streamed)
        self.assertTrue(any(e["stage"] for e in self.f.events.progress_events))

    async def test_explore_refuses_more_than_four_tasks_and_bad_tasks(self):
        task = {"objective": "Find NO2 data.", "bundle": "find-data"}
        model = Model(script=[
            tool_call("explore", {"tasks": [task] * (MAX_TASKS + 1)}),
            tool_call("explore", {"tasks": [{"objective": "x", "bundle": "nope", "secret": "s3cr3t-value"}]}),
            answer("Too many."),
        ])
        turn = await self.turn(self.supervisor(model))
        self.assertEqual(last_tool(model, 1), {"status": "refused", "reason": TOO_MANY_TASKS})
        bad = last_tool(model, 2)
        self.assertEqual(bad["status"], "refused")
        self.assertIn("Task 1: bundle:", bad["reason"])
        self.assertNotIn("s3cr3t-value", bad["reason"])
        self.assertEqual(self.explorer.calls, 0)
        self.assertEqual(turn.results, ())

    async def test_a_lookup_turn_makes_one_exploratory_run_whose_area_the_next_turn_sees(self):
        model = Model(script=[
            tool_call("resolve_area", {"place": "NJ"}), tool_call("search_datasets", {"query": "NO2"}),
            answer("New Jersey is resolved."), answer("Yes."),
        ])
        supervisor = self.supervisor(model)
        turn = await self.turn(supervisor, "Resolve NJ and look for NO2 data.")

        self.assertEqual(turn.results, ())
        [run_id] = turn.run_ids
        run = await self.f.stores.traces.get_run("user-a", run_id)
        self.assertEqual((run.status, run.mode), ("done", Mode.exploratory()))
        entries = await self.f.stores.traces.entries("user-a", run_id)
        self.assertEqual([e.capability for e in entries], [RESOLVE, "sources.satellite.search"])
        resolved = last_tool(model, 1)
        self.assertEqual((resolved["status"], resolved["kind"]), ("done", "polygons"))
        saved = await self.f.stores.results.get("user-a", run_id)
        self.assertEqual(saved.handles, (resolved["handle"],))

        await self.turn(supervisor, "Is NJ still resolved?")
        ledger = ledger_text(model.received[3])
        self.assertIn(f"New Jersey (asked as \"NJ\"): {resolved['handle']}, run {run_id}", ledger)

    async def test_a_failed_lookup_returns_its_fixed_reason(self):
        model = Model(script=[tool_call("check_availability", {"product": "TEMPO_NO2", "area": "nope", "dates": "x"}),
                              answer("It failed.")])
        await self.turn(self.supervisor(model))
        failed = last_tool(model)
        self.assertEqual((failed["status"], failed["category"], failed["retry"]), ("failed", "bad_input", False))


class LedgerTests(SupervisorCase):
    async def test_the_ledger_sits_before_the_newest_message_on_every_call_and_is_never_saved(self):
        model = Model(script=[tool_call("resolve_area", {"place": "NJ"}), answer("Resolved."), answer("Again.")])
        supervisor = self.supervisor(model)
        await self.turn(supervisor, "Resolve NJ.")
        await self.turn(supervisor, "And again?")

        for call, messages in enumerate(model.received):
            with self.subTest(call=call):
                newest = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and not m.additional_kwargs)
                ledger = messages[newest - 1]
                self.assertEqual(ledger.additional_kwargs.get("lc_source"), "ledger")
                body = ledger_text(messages)
                self.assertTrue(body == EMPTY or body.startswith(HEADING))
                self.assertEqual(sum(m.additional_kwargs.get("lc_source") == "ledger" for m in messages), 1)
        self.assertEqual(ledger_text(model.received[0]), EMPTY)
        self.assertTrue(ledger_text(model.received[2]).startswith(HEADING))
        for message in await self.checkpoint():
            self.assertNotEqual(message.additional_kwargs.get("lc_source"), "ledger")
            self.assertNotIn(HEADING, message.text)
            self.assertNotIn(EMPTY, message.text)


    async def test_the_per_turn_message_opens_with_todays_date(self):
        model = Model(script=[answer("Hello.")])
        await self.turn(self.supervisor(model), "What can you do?")
        [ledger] = [
            m for m in model.received[0]
            if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") == "ledger"
        ]
        today = self.f.context.clock.today().isoformat()
        self.assertTrue(ledger.text.startswith(f"Today's date is {today} (UTC)."))
        self.assertEqual(ledger_text(model.received[0]), EMPTY)

class ResumeTests(SupervisorCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.build = self.env_factory(limits=SHORT_WAIT)

    async def test_resume_is_refused_until_the_jobs_are_ready_then_continues_the_run(self):
        self.service.statuses = list(STILL_RUNNING)
        waiting = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual((waiting.status, waiting.reason), ("waiting", JOB_WAITING.format(step="data")))
        run = waiting.trace_id
        model = Model(script=[
            tool_call("resume_run", {"run_id": run}), answer("Still running."),
            tool_call("resume_run", {"run_id": "run_" + "0" * 32}), tool_call("resume_run", {"run_id": run}),
            answer("The map is ready."),
        ])
        supervisor = self.supervisor(model)

        turn = await self.turn(supervisor, "Is the map ready?")
        self.assertEqual(last_tool(model), {"status": "refused", "reason": NOT_READY})
        self.assertIn(f"resume_run({run}) once they are ready", ledger_text(model.received[0]))
        self.assertEqual(turn.results, ())

        self.service.statuses = []
        turn = await self.turn(supervisor, "And now?")
        self.assertIn(f"continue it with resume_run({run})", ledger_text(model.received[2]))
        self.assertEqual(last_tool(model, 3), {"status": "refused", "reason": NOT_READY})
        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (run, "done"), result.reason)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", run)).status, "done")


    async def test_a_run_whose_jobs_could_not_be_checked_can_be_resumed(self):
        self.service.statuses = list(STILL_RUNNING)
        waiting = await self.f.runner.start(self.f.context, self.build, "map", 1, MAP_INPUTS)
        self.assertEqual(waiting.status, "waiting")
        self.service.statuses = []
        model = Model(script=[tool_call("resume_run", {"run_id": waiting.trace_id}), answer("The map is ready.")])

        turn = await self.turn(self.supervisor(model), "Is the map ready?", earthdata=BrokenEarthdata())

        self.assertIn("have not been checked", ledger_text(model.received[0]))
        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (waiting.trace_id, "done"), result.reason)


class LimitTests(SupervisorCase):
    async def test_at_the_tool_call_limit_the_next_call_must_answer(self):
        model = Model(script=[
            tool_call("resolve_area", {"place": "NJ"}),
            tool_call("resolve_area", {"place": "New Jersey"}, content="New Jersey is resolved."),
        ])
        turn = await self.turn(self.supervisor(model, limits=SupervisorLimits(tool_calls=1)), "Resolve NJ.")

        self.assertEqual(model.tool_choices, [None, "none"])
        self.assertEqual(turn.status, "answered")
        self.assertEqual(turn.text, "New Jersey is resolved." + NJ_NOTE)
        self.assertEqual((turn.model_calls, turn.tool_calls), (2, 1))
        [run_id] = turn.run_ids
        self.assertEqual(len(await self.f.stores.traces.entries("user-a", run_id)), 1)
        self.assertIsInstance((await self.checkpoint())[-1], AIMessage)
        self.assertEqual((await self.checkpoint())[-1].tool_calls, [])

    async def test_calls_past_the_tool_call_limit_in_one_turn_are_not_run(self):
        both = AIMessage(content="", tool_calls=[
            {"name": "resolve_area", "args": {"place": "NJ"}, "id": "call_a", "type": "tool_call"},
            {"name": "resolve_area", "args": {"place": "New Jersey"}, "id": "call_b", "type": "tool_call"},
        ])
        model = Model(script=[both, answer("One was resolved.")])
        turn = await self.turn(self.supervisor(model, limits=SupervisorLimits(tool_calls=1)), "Resolve NJ.")

        contents = [m.text for m in tool_messages(model.received[1])]
        self.assertEqual(contents.count(TOOL_LIMIT.format(limit=1)), 1)
        self.assertEqual(turn.tool_calls, 1)

    async def test_the_last_allowed_model_call_must_answer_and_wanting_tools_there_is_the_limit(self):
        model = Model(script=[
            tool_call("resolve_area", {"place": "NJ"}, content="Looking it up."),
            tool_call("search_datasets", {"query": "NO2"}, content="New Jersey is resolved."),
        ])
        turn = await self.turn(self.supervisor(model, limits=SupervisorLimits(model_calls=2)), "Resolve NJ.")

        self.assertEqual(model.tool_choices, [None, "none"])
        self.assertEqual(turn.status, "limit")
        self.assertEqual(model.calls, 2)
        self.assertEqual(turn.text, f"New Jersey is resolved.\n\n{LIMIT_WORDING.format(limit=2)}{NJ_NOTE}")
        self.assertEqual(turn.tool_calls, 1)

    async def test_an_answer_on_the_last_allowed_call_is_not_the_limit(self):
        model = Model(script=[tool_call("resolve_area", {"place": "NJ"}), answer("New Jersey is resolved.")])
        turn = await self.turn(self.supervisor(model, limits=SupervisorLimits(model_calls=2)), "Resolve NJ.")
        self.assertEqual(model.tool_choices, [None, "none"])
        self.assertEqual((turn.status, turn.text), ("answered", "New Jersey is resolved." + NJ_NOTE))

    async def test_every_model_failing_ends_failed_and_keeps_the_results(self):
        model = Model(script=[run_map(), rate_limited(), rate_limited(), rate_limited()])
        turn = await self.turn(self.supervisor(model))

        self.assertEqual(turn.status, "failed")
        [result] = turn.results
        self.assertEqual(result.status, "done")
        self.assertEqual(turn.text, f"{sources_line(['TEMPO_NO2'])}\n\n{FAILED_WORDING}\n\n{assemble('', [result])}")


class QuestionTests(SupervisorCase):
    async def test_a_question_with_options_is_the_turns_question_and_variable_choice(self):
        self.service.estimated_bytes = 100_000_000
        model = Model(script=[run_map(), answer("The retrieval is large.")])
        turn = await self.turn(self.supervisor(model))

        [result] = turn.results
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertEqual(turn.question, result.question)
        self.assertEqual(turn.variable_choice, {
            "run_id": result.trace_id, "prompt": result.question["prompt"], "options": ["confirm", "narrow"],
            "step": "data", "kind": "pick_one", "question": result.question["prompt"],
        })
        self.assertIn(f"Question: {result.question['prompt']}", turn.text)
        self.assertIn("Options: confirm | narrow", turn.text)

    async def test_a_question_without_options_has_no_variable_choice(self):
        model = Model(script=[run_map(area="Atlantis"), answer("Which place?")])
        turn = await self.turn(self.supervisor(model))
        [result] = turn.results
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(turn.question, result.question)
        self.assertIsNone(turn.variable_choice)


LARGE = 100_000_000
CONFIRM_OPTIONS = "confirm | narrow"


class PausedRunTests(SupervisorCase):
    """Runs paused on a question, answered, refused and cancelled through the supervisor."""

    async def asked(self, model: Model) -> tuple[Supervisor, Result]:
        self.service.estimated_bytes = LARGE
        model.script[:0] = [run_map(), answer("The retrieval is large.")]
        supervisor = self.supervisor(model)
        [result] = (await self.turn(supervisor)).results
        self.assertEqual(result.status, "needs_input")
        return supervisor, result

    async def test_resume_run_answers_a_paused_run_and_the_ledger_says_how(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        model.script.extend([tool_call("resume_run", {"run_id": run, "answer": "Confirm"}), answer("The map is ready.")])

        turn = await self.turn(supervisor, "Yes, go ahead.")

        ledger = ledger_text(model.received[2])
        self.assertIn(f"It waits on step data (pick_one): when the user answers, call resume_run({run}, answer)", ledger)
        self.assertIn(f"Options: {CONFIRM_OPTIONS}", ledger)
        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (run, "done"), result.reason)
        self.assertEqual(last_tool(model)["status"], "done")
        self.assertEqual(len(self.service.jobs), 1)
        self.assertIsNone(turn.question)
        self.assertIn("The map is ready.", turn.text)

    async def test_a_bad_answer_comes_back_once_with_its_reason_then_the_model_is_told_to_ask_again(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        model.script.extend([
            tool_call("resume_run", {"run_id": run, "answer": "maybe"}),
            tool_call("resume_run", {"run_id": run, "answer": "perhaps"}),
            tool_call("resume_run", {"run_id": run, "answer": "confirm"}),
            answer("Please choose confirm or narrow."),
        ])

        turn = await self.turn(supervisor, "Maybe.")

        reason = PICK_ONE.format(options=CONFIRM_OPTIONS)
        self.assertEqual(last_tool(model, 3), {"status": "refused", "reason": BAD_ANSWER.format(reason=reason)})
        self.assertEqual(last_tool(model, 4), {"status": "refused", "reason": ASK_AGAIN})
        self.assertEqual(last_tool(model, 5), {"status": "refused", "reason": ASK_AGAIN})
        self.assertEqual(turn.results, ())
        self.assertEqual((await self.f.stores.traces.get_run("user-a", run)).status, "needs_input")
        self.assertEqual(len(self.service.jobs), 0)

        # A later turn may try again.
        model.script.extend([tool_call("resume_run", {"run_id": run, "answer": "confirm"}), answer("Done.")])
        [result] = (await self.turn(supervisor, "Confirm.")).results
        self.assertEqual(result.status, "done", result.reason)

    async def test_an_answer_is_refused_for_a_run_not_waiting_on_one_and_needed_for_one_that_is(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        self.service.statuses = list(STILL_RUNNING)
        self.service.estimated_bytes = 1000
        build = self.env_factory(limits=SHORT_WAIT)
        waiting = await self.f.runner.start(self.f.context, build, "map", 1, MAP_INPUTS)
        self.assertEqual(waiting.status, "waiting")
        model.script.extend([
            tool_call("resume_run", {"run_id": asked.trace_id}),
            tool_call("resume_run", {"run_id": waiting.trace_id, "answer": "confirm"}),
            answer("Waiting."),
        ])
        await self.turn(supervisor, "Go on.", build=build)
        self.assertEqual(last_tool(model, 3), {"status": "refused", "reason": ANSWER_NEEDED})
        self.assertEqual(last_tool(model, 4), {"status": "refused", "reason": NOT_ASKING})

    async def test_cancel_run_drops_a_paused_run_and_the_ledger_shows_it_cancelled(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        model.script.extend([
            tool_call("cancel_run", {"run_id": run}), tool_call("cancel_run", {"run_id": run}), answer("Dropped."),
            answer("Nothing is waiting."),
        ])

        turn = await self.turn(supervisor, "Never mind, drop it.")

        cancelled = last_tool(model, 3)
        self.assertEqual((cancelled["run_id"], cancelled["status"]), (run, "cancelled"))
        self.assertEqual(last_tool(model, 4), {"status": "refused", "reason": NOT_CANCELLABLE})
        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (run, "cancelled"))
        self.assertEqual((await self.f.stores.traces.get_run("user-a", run)).status, "cancelled")
        await self.turn(supervisor, "Anything waiting?")
        self.assertIn(f"- {run} cancelled, {asked.mode.label()}: The run was cancelled.", ledger_text(model.received[5]))

    async def test_an_answer_from_the_app_resumes_the_run_before_any_model_call(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        seen_status: list[str] = []

        def prose(messages: list[BaseMessage]) -> AIMessage:
            seen_status.append(json.loads(tool_messages(messages)[-1].text)["status"])
            return answer("The map is ready.")

        model.script.append(prose)
        turn = await self.turn(supervisor, "confirm", answer=AppAnswer(run_id=run, step="data", value="confirm"))

        self.assertEqual(model.calls, 3)
        self.assertEqual(seen_status, ["done"])
        sent = model.received[2]
        call = next(m for m in reversed(sent) if isinstance(m, AIMessage) and m.tool_calls)
        self.assertEqual(call.tool_calls[0]["name"], "resume_run")
        self.assertEqual(call.tool_calls[0]["args"], {"run_id": run, "answer": "confirm"})
        self.assertEqual([m.text for m in sent if isinstance(m, HumanMessage)][-1], "confirm")
        [result] = turn.results
        self.assertEqual((result.trace_id, result.status), (run, "done"), result.reason)
        self.assertEqual((turn.model_calls, turn.tool_calls), (1, 0))
        self.assertTrue(turn.text.endswith("The map is ready.") or "The map is ready." in turn.text)
        saved = await self.checkpoint()
        self.assertTrue(any(isinstance(m, ToolMessage) and m.tool_call_id == call.tool_calls[0]["id"] for m in saved))

    async def test_an_answer_from_the_app_for_a_run_of_another_conversation_resumes_nothing(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        elsewhere = dataclasses.replace(
            self.f.context, identity=dataclasses.replace(self.f.context.identity, conversation_id="conv-2"),
            events=RecordingEvents(),
        )

        turn = await self.turn(supervisor, "confirm", context=elsewhere,
                               answer=AppAnswer(run_id=run, step="data", value="confirm"))

        self.assertEqual(turn.text, ANSWER_GONE)
        self.assertEqual((turn.results, turn.question, turn.variable_choice), ((), None, None))
        self.assertEqual(model.calls, 2)
        self.assertEqual((await self.f.stores.traces.get_run("user-a", run)).status, "needs_input")
        self.assertEqual(len(self.service.jobs), 0)
        self.assertEqual(
            [e.kind for e in await self.f.stores.traces.entries("user-a", run)].count("ask"), 0,
        )

    async def test_an_answer_from_the_app_that_does_not_fit_is_refused_without_a_model_call(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        run = asked.trace_id
        turn = await self.turn(supervisor, "maybe", answer=AppAnswer(run_id=run, step="data", value="maybe"))

        self.assertEqual(model.calls, 2)
        self.assertTrue(turn.text.startswith(PICK_ONE.format(options=CONFIRM_OPTIONS)), turn.text)
        assert asked.question is not None
        self.assertIn(f"Question: {asked.question['prompt']}", turn.text)
        self.assertIn(f"Options: {CONFIRM_OPTIONS}", turn.text)
        self.assertEqual(turn.question, asked.question)
        assert turn.variable_choice is not None
        self.assertEqual(turn.variable_choice["run_id"], run)
        self.assertEqual((turn.model_calls, turn.status), (0, "answered"))
        self.assertEqual((await self.f.stores.traces.get_run("user-a", run)).status, "needs_input")

        for stale in (AppAnswer(run_id=run, step="figure", value="confirm"),
                      AppAnswer(run_id="run_" + "0" * 32, step="data", value="confirm")):
            with self.subTest(stale=stale):
                refused = await self.turn(supervisor, "confirm", answer=stale)
                self.assertTrue(refused.text.startswith(ANSWER_GONE), refused.text)
        self.assertEqual(model.calls, 2)

    async def test_a_run_paused_past_its_expiry_is_cancelled_at_the_next_turn(self):
        model = Model(script=[])
        supervisor, asked = await self.asked(model)
        model.script.append(answer("Hello."))
        later = dataclasses.replace(
            self.f.context, clock=TickingClock(self.f.context.clock.now() + datetime.timedelta(days=31)),
        )
        await self.turn(supervisor, "Hello again.", context=later)
        saved = await self.f.stores.results.get("user-a", asked.trace_id)
        self.assertEqual((saved.status, saved.reason), ("cancelled", EXPIRED_REASON))
        self.assertIn(f"- {asked.trace_id} cancelled, {asked.mode.label()}: {EXPIRED_REASON}",
                      ledger_text(model.received[2]))


class DoneExtrasTests(SupervisorCase):
    async def test_followups_from_several_results_merge_in_order_without_repeats(self):
        self.explorer.script = [
            final("One.", followups=["Why?", "Where?"]), final("Two.", followups=["Where?", "When?"]),
            final("Three.", followups=[]),
        ]
        find = {"objective": "Find NO2 data.", "bundle": "find-data"}
        model = Model(script=[
            tool_call("explore", {"tasks": [find]}), tool_call("explore", {"tasks": [find]}),
            tool_call("explore", {"tasks": [find]}), answer("Three searches ran."),
        ])
        turn = await self.turn(self.supervisor(model), "What NO2 data is there?")

        self.assertEqual([r.followups for r in turn.results], [("Why?", "Where?"), ("Where?", "When?"), ()])
        self.assertEqual(turn.followups, ("Why?", "Where?", "When?"))

    async def test_a_turns_followups_are_capped_after_merging_and_dropping_repeats(self):
        self.explorer.script = [
            final("One.", followups=["A?", "B?", "A?"]), final("Two.", followups=["B?", "C?", "D?"]),
        ]
        find = {"objective": "Find NO2 data.", "bundle": "find-data"}
        model = Model(script=[
            tool_call("explore", {"tasks": [find]}), tool_call("explore", {"tasks": [find]}), answer("Two searches ran."),
        ])
        turn = await self.turn(self.supervisor(model), "What NO2 data is there?")

        self.assertEqual(MAX_FOLLOWUPS, 3)
        self.assertEqual(turn.followups, ("A?", "B?", "C?"))
        # Each result keeps all it suggested.
        self.assertEqual(turn.results[1].followups, ("B?", "C?", "D?"))

    async def test_a_turn_whose_results_suggest_nothing_has_no_followups(self):
        turn = await self.turn(self.supervisor(Model(script=[run_map(), answer("Done.")])))
        self.assertEqual(turn.followups, ())

    async def test_with_several_questions_the_first_with_options_is_the_picker(self):
        self.service.estimated_bytes = 100_000_000
        model = Model(script=[run_map(), run_map(dates={"start": "2024-06-10", "end": "2024-06-12"}), answer("Both are large.")])
        turn = await self.turn(self.supervisor(model))

        first, second = turn.results
        self.assertEqual([r.status for r in turn.results], ["needs_input", "needs_input"])
        self.assertNotEqual(first.trace_id, second.trace_id)
        assert first.question is not None
        self.assertEqual(turn.question, first.question)
        self.assertEqual(turn.variable_choice, {
            "run_id": first.trace_id, "prompt": first.question["prompt"], "options": ["confirm", "narrow"],
            "step": "data", "kind": "pick_one", "question": first.question["prompt"],
        })


EARTHDATA_DOWN = unavailable_wording("earthdata")


class SourceDownTests(SupervisorCase):
    def down(self, *connectors: str) -> RunContext:
        return dataclasses.replace(self.f.context, reachable=lambda connector: connector not in connectors)

    def needing(self, connector: str) -> set[str]:
        """The tool names of the capabilities that cannot run without ``connector``."""
        registry = self.f.registry
        return {
            c.name.replace(".", "__") for group in registry.groups() for c in registry.in_group(group.name)
            if c.connector == connector and c.connector_required
        }

    async def no_runs(self) -> None:
        self.assertEqual(await self.f.stores.traces.list_runs("user-a", "conv-1"), [])

    def test_the_wording_names_the_source(self):
        self.assertEqual(
            EARTHDATA_DOWN, "NASA Earthdata is unavailable right now, so nothing was started; try again in a few minutes.",
        )
        self.assertIn("EPA AQS", unavailable_wording("epa_aqs"))

    async def test_a_workflow_that_needs_a_down_source_starts_nothing_and_the_answer_says_so_once(self):
        model = Model(script=[run_map(), run_map(), answer("The map could not be made.")])
        turn = await self.turn(self.supervisor(model), context=self.down("earthdata"))

        self.assertEqual(last_tool(model, 1), {"status": "refused", "reason": EARTHDATA_DOWN})
        self.assertEqual(turn.results, ())
        self.assertEqual((turn.run_ids, turn.unavailable), ((), ("earthdata",)))
        await self.no_runs()
        self.assertEqual(turn.text.count(EARTHDATA_DOWN), 1)
        self.assertEqual(turn.text, f"The map could not be made.\n\nNot finished:\n- {EARTHDATA_DOWN}")

    async def test_explore_tasks_run_without_the_down_sources_capabilities_and_disclose_it(self):
        disclosure = UNREACHABLE_DISCLOSURE.format(source="NASA Earthdata")
        for bundle in ("analyse", "find-data"):
            with self.subTest(bundle=bundle):
                self.explorer.script, self.explorer.position = [final("Only the ground monitors could be searched.")], 0
                self.explorer.tools_offered.clear()
                task = {"objective": "Get NO2 over NJ.", "bundle": bundle, "places": ["NJ"]}
                model = Model(script=[tool_call("explore", {"tasks": [task]}), answer("Here is what could be found.")])
                turn = await self.turn(self.supervisor(model), context=self.down("earthdata"))

                [result] = turn.results
                self.assertEqual(result.status, "done")
                self.assertEqual(result.disclosures.count(disclosure), 1)
                offered = {name for names in self.explorer.tools_offered for name in names}
                self.assertTrue(offered)
                self.assertFalse(offered & self.needing("earthdata"), offered)
                self.assertIn("sources__monitors__find", offered)
                # Nothing was refused, so the answer names no unavailable source as unfinished.
                self.assertEqual(turn.unavailable, ())
                self.assertNotIn(EARTHDATA_DOWN, turn.text)
                self.assertNotIn("Not finished:", turn.text)
                self.assertIn(disclosure, turn.text)

    async def test_with_every_source_up_an_explore_task_is_offered_the_whole_bundle_and_discloses_nothing(self):
        self.explorer.script = [final("Found it.")]
        task = {"objective": "Get NO2 over NJ.", "bundle": "find-data", "places": ["NJ"]}
        model = Model(script=[tool_call("explore", {"tasks": [task]}), answer("Found.")])
        turn = await self.turn(self.supervisor(model))

        [result] = turn.results
        self.assertFalse(any("unavailable" in d for d in result.disclosures))
        offered = {name for names in self.explorer.tools_offered for name in names}
        self.assertIn("sources__satellite__search", offered)
        self.assertTrue(offered & self.needing("earthdata"))

    async def test_a_task_that_works_on_existing_handles_is_not_held_by_a_down_source(self):
        self.explorer.script = [final("Nothing to work on.")]
        task = {"objective": "Review the earlier numbers.", "bundle": "investigate"}
        model = Model(script=[tool_call("explore", {"tasks": [task]}), answer("Reviewed.")])
        turn = await self.turn(self.supervisor(model), context=self.down("earthdata"))

        self.assertEqual(turn.unavailable, ())
        self.assertEqual(len(turn.results), 1)

    async def test_a_satellite_lookup_is_refused_before_its_run_starts_and_a_place_lookup_is_not(self):
        model = Model(script=[
            tool_call("search_datasets", {"query": "NO2"}), tool_call("resolve_area", {"place": "NJ"}),
            answer("Searching was not possible."),
        ])
        turn = await self.turn(self.supervisor(model), context=self.down("earthdata"))

        self.assertEqual(last_tool(model, 1), {"status": "refused", "reason": EARTHDATA_DOWN})
        self.assertEqual(last_tool(model, 2)["status"], "done")
        self.assertEqual(turn.unavailable, ("earthdata",))
        # Only the place lookup made a run.
        [run_id] = turn.run_ids
        entries = await self.f.stores.traces.entries("user-a", run_id)
        self.assertEqual([e.capability for e in entries], [RESOLVE])
        self.assertEqual(turn.text.count(EARTHDATA_DOWN), 1)

    async def test_a_source_that_is_up_is_unaffected(self):
        turn = await self.turn(
            self.supervisor(Model(script=[run_map(), answer("Mapped.")])), context=self.down("epa_aqs"),
        )

        [result] = turn.results
        self.assertEqual(result.status, "done")
        self.assertEqual(turn.unavailable, ())
        self.assertNotIn("unavailable", turn.text)


class NumbersTests(SupervisorCase):
    async def test_an_unmatched_number_is_recorded_on_the_lookup_run_and_a_matched_one_is_not(self):
        def prose(messages: list[BaseMessage]) -> AIMessage:
            view = json.loads(tool_messages(messages)[-1].text)
            days = next(f["value"] for f in view["findings"] if f["name"] == "days_with_data")
            return answer(f"Data covered {days} days in 2024, and 12345 is invented.")

        model = Model(script=[
            tool_call("run_workflow", {"name": "timeseries", "inputs": MAP_INPUTS}), prose,
        ])
        turn = await self.turn(self.supervisor(model), "Plot the NO2 time series over NJ.")

        result, lookup = turn.run_ids
        self.assertEqual(turn.results[0].trace_id, result)
        entries = await self.f.stores.traces.entries("user-a", lookup)
        [entry] = entries
        self.assertEqual((entry.kind, entry.step, entry.status), ("agent", NUMBERS_STEP, "done"))
        self.assertEqual(entry.result, {"unmatched": ["12345"]})
        self.assertEqual((await self.f.stores.traces.get_run("user-a", lookup)).status, "done")

    async def test_no_entry_when_every_number_matches(self):
        model = Model(script=[answer("Nothing to report.")])
        turn = await self.turn(self.supervisor(model), "Hello 42")
        self.assertEqual(turn.run_ids, ())

    def test_matching_rules(self):
        result = Result(status="done", mode=Mode.exploratory(), trace_id="run_" + "a" * 32, findings=(
            Finding(name="mean", value=0.0123456, unit="ppm", source_handle="scalar_" + "b" * 32),
            Finding(name="big", value=2.5e15, source_handle="scalar_" + "b" * 32),
            Finding(name="share", value=0.42, source_handle="scalar_" + "b" * 32),
        ))
        prose = ("The mean was 0.0123 ppm (0.01235 rounded) on 2024-06-01 at 14:00 in 2024; the column was "
                 "2.5 × 10^15, 42% of cells, 7 monitors and 3.5 others.")
        self.assertEqual(unmatched_numbers(prose, [result], "Over 7 monitors?"), ["3.5"])
        self.assertEqual(unmatched_numbers("NO2 and PM2.5 at site 34-003-0010.", [result]), [])
        listed = "1. First item\n2) Second over 24 hours across 3 sites in 2023, on 1 day and 2 monitors."
        self.assertEqual(unmatched_numbers(listed, []), [])
        # Units are claims, so numbers carrying them are checked.
        self.assertEqual(unmatched_numbers("It was 41 ppb, 12 µg/m³ and 30% of 3 days.", []), ["41", "12", "30%"])


class ThreadTests(SupervisorCase):
    async def test_two_users_with_the_same_conversation_id_never_share_a_checkpoint(self):
        model = Model(script=[answer("Hello A."), answer("Hello B.")])
        supervisor = self.supervisor(model)
        other = self.context_for("user-b")
        await self.turn(supervisor, "I am user A.")
        await self.turn(supervisor, "I am user B.", context=other)

        self.assertEqual([m.text for m in model.received[1] if isinstance(m, HumanMessage) and not m.additional_kwargs],
                         ["I am user B."])
        self.assertNotEqual(thread_id(self.f.context.identity), thread_id(other.identity))
        self.assertEqual([m.text for m in await self.checkpoint()], ["I am user A.", "Hello A."])
        self.assertEqual([m.text for m in await self.checkpoint(other)], ["I am user B.", "Hello B."])

    def test_thread_ids_escape_their_parts(self):
        from tta_backend.harness.run_context import Identity
        self.assertNotEqual(thread_id(Identity("a/b", "c")), thread_id(Identity("a", "b/c")))
        self.assertEqual(thread_id(Identity("user-a", "conv-1")), "supervisor/user-a/conv-1")


class StreamingTests(SupervisorCase):
    async def test_the_final_prose_streams_line_by_line_after_the_sources_line(self):
        model = Streamed(script=[run_map(), answer("Line one.\n\n\nLine two.\nLine three.")])
        turn = await self.turn(self.supervisor(model))

        texts = self.f.events.texts
        self.assertEqual(texts[:4], [sources_line(["TEMPO_NO2"]), "\n\nLine one.", "\n\nLine two.", "\nLine three."])
        self.assertTrue(texts[-1].startswith("\n\nNotes:\n- "))
        [result] = turn.results
        self.assertEqual(turn.text, f"{sources_line(['TEMPO_NO2'])}\n\n{assemble('Line one.\n\nLine two.\nLine three.', [result])}")

    async def test_narration_before_a_tool_call_streams_but_the_text_keeps_the_sources_and_final_prose(self):
        model = Streamed(script=[
            tool_call("run_workflow", {"name": "map", "inputs": MAP_INPUTS}, content="Let me check.\n"),
            answer("The map is ready."),
        ])
        turn = await self.turn(self.supervisor(model))

        [result] = turn.results
        self.assertEqual(turn.text, f"{sources_line(['TEMPO_NO2'])}\n\n{assemble('The map is ready.', [result])}")
        texts = self.f.events.texts
        self.assertEqual(texts[0], "Let me check.")
        opening = texts.index("\n\n" + (sources_line(["TEMPO_NO2"]) or ""))
        self.assertEqual(texts[opening + 1], "\n\nThe map is ready.")

    async def test_a_call_that_fails_mid_stream_is_retried_and_only_the_retry_is_the_answer(self):
        model = Streamed(script=[Partial("Partial ans", rate_limited()), answer("Full answer.")])
        turn = await self.turn(self.supervisor(model), "Hello")

        self.assertEqual((turn.status, turn.text), ("answered", "Full answer."))
        self.assertEqual(model.calls, 2)
        self.assertNotIn("Partial ans", "".join(self.f.events.texts))
        self.assertEqual("".join(self.f.events.texts), "Full answer.")

    async def test_a_host_without_text_still_gets_the_turn(self):
        class OldEvents:
            def progress(self, message: str, *, stage: str | None = None, detail: Any = None) -> None: ...
            def chart(self, payload: dict[str, Any]) -> None: ...
            def question(self, payload: dict[str, Any]) -> None: ...

        context = dataclasses.replace(self.f.context, events=OldEvents())
        model = Model(script=[answer("Hi.")])
        turn = await self.turn(self.supervisor(model), "Hello", context=context)
        self.assertEqual(turn.text, "Hi.")


class UnfinishedCallTests(SupervisorCase):
    async def test_a_tool_call_left_without_a_result_is_answered_on_the_next_turn(self):
        self.f.places.refusals["stuck"] = asyncio.CancelledError()
        model = Model(script=[tool_call("resolve_area", {"place": "stuck"}, call_id="call_stuck"), answer("Hello again.")])
        supervisor = self.supervisor(model)
        # The tool is cut off mid-call, so the turn ends before its result is saved.
        interrupted = await self.turn(supervisor, "Resolve stuck.")
        self.assertEqual(interrupted.status, "failed")
        saved = await self.checkpoint()
        self.assertEqual(saved[-1].tool_calls[0]["id"], "call_stuck")

        turn = await self.turn(supervisor, "Are you there?")

        self.assertEqual((turn.status, turn.text), ("answered", "Hello again."))
        sent = model.received[1]
        index = next(i for i, m in enumerate(sent) if isinstance(m, AIMessage) and m.tool_calls)
        answered = sent[index + 1]
        self.assertIsInstance(answered, ToolMessage)
        self.assertEqual((answered.tool_call_id, answered.text), ("call_stuck", UNFINISHED_CALL))
        self.assertEqual(sent[-1].text, "Are you there?")
        # Only the request is repaired; the saved state keeps the call as it was.
        self.assertFalse(any(isinstance(m, ToolMessage) for m in await self.checkpoint()))


class HistoryTests(SupervisorCase):
    async def test_long_history_is_summarised_and_the_ledger_still_comes_last(self):
        long = "A long answer about ozone. " * 20
        model = Model(script=[answer(long), answer("Short.")])
        summary = Model(script=[answer("Earlier the user asked about ozone.")])
        supervisor = self.supervisor(model, summary=summary, limits=SupervisorLimits(summarise_above_tokens=50, keep_messages=2))

        await self.turn(supervisor, "Tell me about ozone.")
        self.assertEqual(summary.calls, 0)
        turn = await self.turn(supervisor, "And now?")

        self.assertEqual((turn.status, turn.text), ("answered", "Short."))
        self.assertEqual(summary.calls, 1)
        sent = model.received[1]
        self.assertIn("Earlier the user asked about ozone.", sent[1].text)
        self.assertNotIn("Tell me about ozone.", [m.text for m in sent])
        self.assertEqual(sent[-2].additional_kwargs.get("lc_source"), "ledger")
        self.assertEqual(sent[-1].text, "And now?")
        saved = [m.text for m in await self.checkpoint()]
        self.assertNotIn("Tell me about ozone.", saved)
        self.assertEqual(saved[-1], "Short.")

    async def test_old_tool_results_are_cleared_from_what_the_model_sees_but_kept_in_the_checkpoint(self):
        model = Model(script=[tool_call("resolve_area", {"place": "NJ"}), answer("Done.")])
        limits = SupervisorLimits(clear_tool_results_above_tokens=1, keep_tool_results=0)
        await self.turn(self.supervisor(model, limits=limits), "Resolve NJ.")

        [cleared] = tool_messages(model.received[1])
        self.assertEqual(cleared.text, "[cleared]")
        [kept] = tool_messages(await self.checkpoint())
        self.assertEqual(json.loads(kept.text)["kind"], "polygons")


class ToolSetTests(SupervisorCase):
    async def test_exactly_the_seven_tools_are_offered(self):
        model = Model(script=[answer("Hi.")])
        await self.turn(self.supervisor(model), "Hello")
        self.assertEqual(model.tools_offered, [list(TOOL_NAMES)])

    def test_every_schema_uses_only_keywords_every_provider_accepts(self):
        schemas = tool_schemas(["map", "timeseries"], self.f.registry)
        self.assertEqual(set(schemas), set(TOOL_NAMES))
        for name, schema in schemas.items():
            with self.subTest(tool=name):
                self.assertEqual(schema["type"], "object")
                for node in walk(schema):
                    if isinstance(node, dict):
                        self.assertFalse(set(node) & {"$defs", "$ref", "oneOf", "allOf", "anyOf", "discriminator",
                                                      "additionalProperties"}, node)
        dates = schemas["check_availability"]["properties"]["dates"]
        self.assertNotIn("type", dates)
        self.assertIn("One of: an ISO date; an object with start, end; a list of ISO dates.", dates["description"])
        self.assertEqual(schemas["run_workflow"]["properties"]["name"]["enum"], ["map", "timeseries"])

    def test_the_schemas_and_prompt_ask_for_places_as_the_user_wrote_them(self):
        schemas = tool_schemas(["map"], self.f.registry)
        task = schemas["explore"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("exactly as the user wrote them", task["places"]["description"])
        self.assertIn("as the user wrote it", schemas["resolve_area"]["properties"]["place"]["description"])
        prompt = self.supervisor(Model(script=[])).prompt
        self.assertIn("places in the user's own words", prompt)

    def test_the_prompt_leaves_sources_disclosures_and_failure_text_to_code(self):
        prompt = self.supervisor(Model(script=[])).prompt
        self.assertIn("Code appends the result's disclosures, gaps, fixed-wording reasons, questions and data sources", prompt)
        self.assertIn("write a sources line", prompt)
        self.assertIn("State only numbers that appear in the results", prompt)


RUN = "run_" + "c" * 32


class AnswerTests(SupervisorCase):
    def test_code_appends_disclosures_gaps_reasons_and_questions_once_whatever_the_prose_says(self):
        handle = "scalar_" + "d" * 32
        done = Result(status="done", mode=Mode.exploratory(), trace_id=RUN, disclosures=("Product: A.", "Masked."),
                      gaps=(Gap(step="figure", reason="The figure could not be drawn."),))
        failed = Result(status="failed", mode=Mode.exploratory(), trace_id="run_" + "e" * 32,
                        reason="The source is down.", disclosures=("Masked.",))
        asking = Result(status="needs_input", mode=Mode.exploratory(), trace_id="run_" + "f" * 32,
                        reason="Pick one.", question={"kind": "pick_one", "prompt": "Pick one.", "options": ["a", {"b": 1}]},
                        findings=(Finding(name="n", value=1, source_handle=handle),))

        text = assemble("Product: A. The answer.", [done, failed, asking])

        self.assertEqual(text, "\n\n".join([
            "Product: A. The answer.",
            "Notes:\n- Product: A.\n- Masked.",
            "Gaps:\n- The figure could not be drawn.",
            "Not finished:\n- The source is down.",
            'Question: Pick one.\nOptions: a | {"b":1}',
        ]))
        self.assertEqual(assemble("Only prose.", []), "Only prose.")
        again = asking.model_copy(update={"trace_id": "run_" + "9" * 32, "reason": "Step x needs an answer."})
        self.assertEqual(assemble("", [asking, again]).count("Question: Pick one."), 1)

    def test_the_compact_view_holds_no_disclosures_or_data(self):
        handle = "grid_" + "1" * 32
        result = Result(status="done", mode=Mode.draft("map", 1), trace_id=RUN, handles=(handle,), charts=("chart-1",),
                        disclosures=("Product: TEMPO_NO2.",), interpretation="Higher in the north.", followups=("Why?",),
                        findings=(Finding(name="mean", value=1.5, unit="ppb", source_handle=handle),))
        view = json.loads(result_view(result))
        self.assertEqual(view, {
            "run_id": RUN, "status": "done", "mode": "Draft workflow: map v1",
            "findings": [{"name": "mean", "value": 1.5, "unit": "ppb", "source_handle": handle}],
            "handles": [handle], "charts": ["chart-1"], "interpretation": "Higher in the north.", "followups": ["Why?"],
        })

    async def test_a_failed_result_names_no_source(self):
        model = Model(script=[run_map(), answer("Done.")])
        turn = await self.turn(self.supervisor(model))
        [result] = turn.results
        failed = Result.model_validate({**result.model_dump(), "status": "failed", "reason": "It failed."})
        registry = self.f.stores.handles
        self.assertEqual(await data_sources(registry, "user-a", [result]), ["TEMPO_NO2"])
        self.assertEqual(await data_sources(registry, "user-a", [failed]), [])
        self.assertEqual(await data_sources(registry, "user-b", [result]), [])
        self.assertIsNone(sources_line([]))
        with self.assertLogs("tta_backend.harness.supervisor.answer", "ERROR") as logged:
            self.assertEqual(await data_sources(BrokenRegistry(), "user-a", [result]), [])  # type: ignore[arg-type]
        self.assertNotIn("the registry is down", "".join(logged.output))
