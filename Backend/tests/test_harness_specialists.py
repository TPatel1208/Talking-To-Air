"""Exploration in memory: tools, the allowlist, places, questions, notes, budgets, waiting, the answer and the trace."""
import asyncio
import datetime
import json
import re
import tempfile
import unittest
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from tests.test_harness_grids import NJ
from tta_backend.harness.capabilities import (
    Call,
    Capability,
    CapabilityRegistry,
    CapabilityResult,
    Group,
    InputEvicted,
    JobStillRunning,
    Params,
    handle_param,
    invoke,
    needs_choice,
)
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, Period, ScalarInput
from tta_backend.harness.knowledge import Note, NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF, ModelChoice, NamedModel
from tta_backend.harness.results import Result
from tta_backend.harness.run_context import NO_MODEL_REASON, PlaceRefused, RunContext, RunEnv
from tta_backend.harness.specialists import (
    ANSWER_TOOL,
    ExploreLimits,
    ExploreTask,
    explore,
    get_bundle,
    tool_name,
    tool_schema,
)
from tta_backend.harness.specialists.explore import (
    INTERNAL_REASON,
    MODEL_BUDGET,
    NO_SUMMARY,
    TOOL_BUDGET,
    UNKNOWN_HANDLE,
    UNRESOLVED_DISCLOSURE,
    with_wait,
)
from tta_backend.harness.specialists import prompt as prompt_module
from tta_backend.harness.specialists import tools as tools_module
from tta_backend.harness.specialists.answer import CORRECTION, NOT_RUN
from tta_backend.harness.specialists.prompt import ROLE
from tta_backend.harness.specialists.state import TaskState
from tta_backend.harness.specialists.tools import Caller, capability_tool, capability_tools
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.testing import (
    FakePlaces,
    FakeToolError,
    FixedClock,
    ScriptedChatModel,
    ScriptedModels,
    answer,
    env_factory,
    make_run_context,
    rate_limited,
    tool_call,
)
from tta_backend.harness.trace import FailureCategory

HANDLE_ID = re.compile(r"\b(?:grid|series|points|polygons|table|scalar)_[0-9a-f]{32}\b")
TOMORROW_NOTE = "TEMPO note body: hourly and North America only."
COLUMN_NOTE = "Column note body: a column is not a surface value."
OZONE_NOTE = "Ozone note body: AQS hourly ozone."
NOTES = NoteCatalogue([
    Note(id="tempo-note", title="TEMPO note", body=TOMORROW_NOTE, datasets=("TEMPO_NO2",)),
    Note(id="column-note", title="Column note", body=COLUMN_NOTE, quantities=("no2_tropospheric_column",)),
    Note(id="ozone-note", title="Ozone note", body=OZONE_NOTE, pollutants=("o3",)),
])


# Fake capabilities


class AreaParams(Params):
    area: Annotated[str, handle_param("polygons")]
    value: float = 2.5


class ScaleParams(Params):
    data: Annotated[str, handle_param("scalar")]
    factor: float = 2.0


class DrawParams(Params):
    data: Annotated[str, handle_param("scalar")]


class LabelParams(Params):
    label: str = "x"


class MonitorParams(Params):
    pollutant: str


class Fakes:
    """What the fake capabilities saw, and what they wait on."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.arrived = 0
        self.both = asyncio.Event()
        self.active = 0
        self.peak = 0


def fake_registry(fakes: Fakes, *, workflow_only: bool = False) -> CapabilityRegistry:
    async def measure(call: Call[AreaParams]) -> CapabilityResult:
        fakes.ran.append("measure")
        record = await call.mint("scalar", ScalarInput(value=call.params.value), Descriptor(
            summary="Mean NO2 column over the area", unit="molecules/cm^2", product="TEMPO_NO2",
            quantity="no2_tropospheric_column", region={"name": "New Jersey"},
            period=Period(start="2024-06-01", end="2024-06-01"),
        ))
        return call.done(
            record, summary={"value": call.params.value}, values={"rule": "approved_preference"},
            disclosures=("Measured over the area.",),
        )

    async def scale(call: Call[ScaleParams]) -> CapabilityResult:
        fakes.ran.append("scale")
        held = (await call.open("data")).value
        source = call.handle("data")
        record = await call.mint("scalar", ScalarInput(value=held * call.params.factor), Descriptor(
            summary="Scaled value", unit=source.descriptor.unit,
        ))
        return call.done(record, summary={"value": held * call.params.factor})

    async def draw(call: Call[DrawParams]) -> CapabilityResult:
        fakes.ran.append("draw")
        return CapabilityResult(status="done", charts=("chart-1",), summary={"n": 1}, disclosures=("Drawn as dots.",))

    async def lookup(call: Call[LabelParams]) -> CapabilityResult:
        fakes.ran.append("lookup")
        return CapabilityResult(status="done", summary={"datasets": 3})

    async def ask(call: Call[LabelParams]) -> CapabilityResult:
        fakes.ran.append("ask")
        return needs_choice("Which level should be used?", ["surface", "column"])

    async def monitors(call: Call[MonitorParams]) -> CapabilityResult:
        fakes.ran.append("monitors")
        return CapabilityResult(status="done", summary={"monitors": 2})

    async def meet(call: Call[LabelParams]) -> CapabilityResult:
        fakes.arrived += 1
        if fakes.arrived == 2:
            fakes.both.set()
        await asyncio.wait_for(fakes.both.wait(), timeout=5)
        return CapabilityResult(status="done", summary={"met": True})

    async def hold(call: Call[LabelParams]) -> CapabilityResult:
        fakes.active += 1
        fakes.peak = max(fakes.peak, fakes.active)
        await asyncio.sleep(0.05)
        fakes.active -= 1
        return CapabilityResult(status="done", summary={"held": True})

    async def fetch_begin(call: Call[LabelParams]) -> dict[str, Any]:
        return {"job": "job_1"}

    async def fetch_complete(call: Call[LabelParams], job: dict[str, Any]) -> CapabilityResult:
        raise JobStillRunning(call.capability.name, job, ["job_1"])

    def capability(name: str, params: Any, output_kind: Any = None, **kw: Any) -> Capability[Any]:
        return Capability(
            name=name, description=f"Fake {name}.", params=params, output_kind=output_kind, label=f"Running {name}",
            method="The fake step ran.", retrieves=kw.pop("retrieves", False), retry_safe=True, **kw,
        )

    registry = CapabilityRegistry()
    for group in (
        areas.GROUP,
        Group("sources.fake", "A fake source.", notes="Fake source notes: measure takes an area handle."),
        Group("operations", "Fake operations.", notes="Fake operations notes."),
        Group("render", "Fake rendering.", notes="Fake render notes."),
    ):
        registry.add_group(group)
    registry.register(areas.resolve)
    registry.register(capability("sources.fake.measure", AreaParams, "scalar", run=measure, retrieves=True))
    registry.register(capability("sources.fake.lookup", LabelParams, run=lookup))
    registry.register(capability("sources.fake.ask", LabelParams, run=ask))
    registry.register(capability("sources.fake.monitors", MonitorParams, run=monitors, retrieves=True))
    registry.register(capability("sources.fake.meet", LabelParams, run=meet))
    registry.register(capability("sources.fake.hold", LabelParams, run=hold))
    registry.register(capability(
        "sources.fake.fetch", LabelParams, "scalar", begin=fetch_begin, complete=fetch_complete, retrieves=True,
    ))
    registry.register(capability("operations.scale", ScaleParams, "scalar", run=scale))
    registry.register(capability("render.draw", DrawParams, run=draw))
    if workflow_only:
        registry.register(capability("operations.regulatory", LabelParams, run=lookup, workflow_only=True))
    return registry


# A scripted model whose turns may depend on what it was sent


Turn = AIMessage | BaseException | Callable[[list[BaseMessage]], AIMessage]


class Model(ScriptedChatModel):
    """A scripted model whose turn may be a function of the messages it is sent, to name handles minted at run time."""

    def _turn(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> Any:
        if self.position < len(self.script):
            turn = self.script[self.position]
            if callable(turn) and not isinstance(turn, AIMessage | BaseException):
                self.script[self.position] = turn(messages)
        return super()._turn(messages, kwargs)


class Routed(Model):
    """A scripted model taking each turn from the queue for the task's objective, for tasks run together."""

    turns: dict[str, list[Any]] = {}

    def _turn(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> Any:
        objective = re.search(r"Objective: (.*)", human(messages))
        assert objective is not None
        self.script = [self.turns[objective.group(1)].pop(0)]
        self.position = 0
        return super()._turn(messages, kwargs)


def turns_together(*calls: Any) -> Turn:
    """One turn making several tool calls; each is (name, args) or a function of the messages giving one."""
    def turn(m: list[BaseMessage]) -> AIMessage:
        made = [c(m) if callable(c) else c for c in calls]
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_together_{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(made)
        ])
    return turn


def human(messages: list[BaseMessage]) -> str:
    return next(m for m in messages if isinstance(m, HumanMessage)).text


def input_handle(messages: list[BaseMessage], kind: str) -> str:
    return next(h for h in HANDLE_ID.findall(human(messages)) if h.startswith(f"{kind}_"))


def last_result(messages: list[BaseMessage]) -> dict[str, Any]:
    return json.loads(next(m for m in reversed(messages) if isinstance(m, ToolMessage)).text)


def results(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [json.loads(m.text) for m in messages if isinstance(m, ToolMessage) and m.name != ANSWER_TOOL]


def call_on_area(tool: str, **args: Any) -> Turn:
    return lambda m: tool_call(tool, {"area": input_handle(m, "polygons"), **args})


def call_on_last(tool: str, param: str = "data", **args: Any) -> Turn:
    return lambda m: tool_call(tool, {param: last_result(m)["handle"], **args})


def final(summary: str = "Done.", findings: Callable[[list[BaseMessage]], list[dict[str, Any]]] | None = None,
          **extra: Any) -> Turn:
    def turn(m: list[BaseMessage]) -> AIMessage:
        args = {"summary": summary, "findings": findings(m) if findings else [], "followups": [], **extra}
        return tool_call(ANSWER_TOOL, args, usage=(30, 10))
    return turn


def on_last_scalar(name: str = "mean", value: Any = None, **extra: Any) -> Callable[[list[BaseMessage]], list[dict[str, Any]]]:
    def findings(m: list[BaseMessage]) -> list[dict[str, Any]]:
        last = [r for r in results(m) if r.get("kind") == "scalar"][-1]
        return [{"name": name, "value": last["summary"]["value"] if value is None else value,
                 "handle": last["handle"], **extra}]
    return findings


def choice_of(model: ScriptedChatModel, *fallbacks: ScriptedChatModel) -> ModelChoice:
    return ModelChoice(
        primary=NamedModel(model, "fake:explore"),
        fallbacks=tuple(NamedModel(f, f"fake:fallback-{i}") for i, f in enumerate(fallbacks, 1)),
    )


def task(objective: str = "Find the mean NO2 over the area.", **kw: Any) -> ExploreTask:
    return ExploreTask(objective=objective, bundle=kw.pop("bundle", "analyse"), **kw)


class ExploreCase(unittest.IsolatedAsyncioTestCase):
    clock = FixedClock(datetime.datetime(2024, 7, 1, 12, tzinfo=datetime.UTC))

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fakes = Fakes()
        self.registry = fake_registry(self.fakes)
        self.stores: Stores = memory_stores()
        self.places = FakePlaces({"nj": NJ, "new jersey": NJ}, refusals={
            "nj + atlantis": PlaceRefused("member_not_found", {"member": "atlantis"}),
        })
        self.context = make_run_context(clock=self.clock, stores=self.stores)
        self.envs: list[RunEnv] = []

    def build_env(self, models: Any) -> Callable[[RunContext, str], Any]:
        base = env_factory(self._tmp.name, places=self.places, models=models)

        async def build(context: RunContext, run_id: str) -> RunEnv:
            env = await base(context, run_id)
            self.envs.append(env)
            return env

        return build

    async def go(self, *tasks: ExploreTask, script: list[Turn] | None = None, model: ScriptedChatModel | None = None,
                  models: Any = "default", limits: ExploreLimits | None = None,
                  registry: CapabilityRegistry | None = None) -> list[Result]:
        self.model = model or Model(script=list(script or []))
        if models == "default":
            models = ScriptedModels({"explore": choice_of(self.model)})
        return await explore(
            self.context, self.build_env(models), list(tasks), registry=registry or self.registry, notes=NOTES,
            limits=limits, backoff=NO_BACKOFF,
        )

    async def entries(self, result: Result) -> list[Any]:
        return await self.stores.traces.entries("user-a", result.trace_id)

    async def uses(self, result: Result) -> list[str]:
        return [e.capability for e in await self.entries(result) if e.kind == "use"]

    async def explore_entry(self, result: Result) -> Any:
        [entry] = [e for e in await self.entries(result) if e.kind == "explore"]
        return entry

    async def run_status(self, result: Result) -> str:
        return (await self.stores.traces.get_run("user-a", result.trace_id)).status

    async def mint_scalar(self, value: float = 7.0, product: str | None = None) -> str:
        """A scalar this user owns, made outside any task."""
        async def make(call: Call[LabelParams]) -> CapabilityResult:
            record = await call.mint("scalar", ScalarInput(value=value), Descriptor(
                summary="An earlier scalar", unit="ppb", product=product,
            ))
            return call.done(record)

        env = await self.build_env(None)(self.context, await self._started_run())
        made = Capability(
            name="operations.make", description="Make a scalar.", params=LabelParams, output_kind="scalar",
            label="Making", method="Made.", retrieves=False, retry_safe=True, run=make,
        )
        result = await invoke(env, made, {})
        assert result.handle is not None
        return result.handle

    async def _started_run(self) -> str:
        from tta_backend.harness.results import Mode
        from tta_backend.harness.trace import Run, new_run_id

        run = Run(id=new_run_id(), user_id="user-a", conversation_id="conv-1", mode=Mode.exploratory(),
                  started=self.clock.now())
        await self.stores.traces.start_run(run)
        return run.id


class ToolTests(unittest.TestCase):
    def test_every_catalogue_capability_makes_a_provider_safe_tool_name(self):
        registry = build_registry()
        for group in registry.groups():
            for capability in registry.in_group(group.name):
                name = tool_name(capability)
                self.assertRegex(name, r"^[a-zA-Z0-9_-]{1,64}$")
                self.assertEqual(name, capability.name.replace(".", "__"))
        self.assertEqual(tool_name(registry.get("sources.satellite.get_data")), "sources__satellite__get_data")

    def test_the_schema_is_the_capabilitys_with_handle_kinds_said_in_words(self):
        capability = build_registry().get("operations.statistic")
        schema = tool_schema(capability)
        original = capability.schema()
        self.assertEqual(set(schema["properties"]), set(original["properties"]))
        self.assertNotIn("x-handle-kinds", json.dumps(schema))
        self.assertIn("A handle id of kind grid or series.", schema["properties"]["data"]["description"])
        self.assertEqual(schema["required"], original["required"])
        self.assertIn("x-handle-kinds", original["properties"]["data"])

    def test_a_tool_carries_the_capability_description_and_schema(self):
        registry = build_registry()
        caller = Caller(None, TaskState(), NOTES, 1)  # type: ignore[arg-type]
        capabilities = get_bundle("analyse").capabilities(registry)
        tools = capability_tools(capabilities, caller)
        self.assertEqual([t.name for t in tools], [tool_name(c) for c in capabilities])
        get_data = next(t for t in tools if t.name == "sources__satellite__get_data")
        self.assertEqual(get_data.description, registry.get("sources.satellite.get_data").description)
        self.assertEqual(get_data.args_schema, tool_schema(registry.get("sources.satellite.get_data")))

    def test_a_workflow_only_capability_is_never_wrapped(self):
        registry = fake_registry(Fakes(), workflow_only=True)
        caller = Caller(None, TaskState(), NOTES, 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "only in validated workflows"):
            capability_tool(registry.get("operations.regulatory"), caller)
        names = [c.name for c in get_bundle("analyse").capabilities(registry)]
        self.assertNotIn("operations.regulatory", names)

    def test_a_parameter_langchain_keeps_for_itself_is_refused(self):
        class Reserved(Params):
            config: str

        async def run(call: Call[Reserved]) -> CapabilityResult:
            return CapabilityResult(status="done")

        capability = Capability(
            name="operations.reserved", description="Reserved.", params=Reserved, output_kind=None,
            label="Reserved", method="Ran.", retrieves=False, retry_safe=True, run=run,
        )
        with self.assertRaisesRegex(ValueError, "cannot receive"):
            capability_tool(capability, Caller(None, TaskState(), NOTES, 1))  # type: ignore[arg-type]


class ToolCallTests(ExploreCase):
    async def test_the_model_is_offered_one_tool_per_bundle_capability_and_the_answer(self):
        [result] = await self.go(task(places=("NJ",)), script=[final()])
        self.assertEqual(result.status, "done")
        expected = sorted(tool_name(c) for c in get_bundle("analyse").capabilities(self.registry))
        self.assertEqual(sorted(self.model.tools_offered[0]), sorted([*expected, ANSWER_TOOL]))

    async def test_the_model_sees_small_json_never_data(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final(),
        ])
        seen = last_result(self.model.received[1])
        self.assertEqual(seen["status"], "done")
        self.assertEqual(seen["kind"], "scalar")
        self.assertTrue(seen["handle"].startswith("scalar_"))
        self.assertEqual(seen["summary"], {"value": 2.5})
        self.assertEqual(seen["disclosures"], ["Measured over the area."])
        self.assertNotIn("method", seen)

    async def test_a_failure_is_a_tool_result_with_its_category_and_fixed_reason(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure", value="not a number"),
            final(),
        ])
        self.assertEqual(result.status, "done")
        seen = last_result(self.model.received[1])
        self.assertEqual(seen, {
            "status": "failed", "category": "bad_input",
            "reason": "The parameters for sources.fake.measure are not valid: value.", "retry": False,
        })

    async def test_a_handle_outside_the_task_is_refused_before_invoke(self):
        foreign = await self.mint_scalar()
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("operations__scale", {"data": foreign}),
            tool_call("operations__scale", {"data": "anything"}),
            final(),
        ])
        self.assertEqual(result.status, "done")
        first = last_result(self.model.received[1])
        self.assertEqual(first["status"], "refused")
        self.assertEqual(first["reason"], tools_module.FOREIGN_HANDLE.format(handle=foreign))
        self.assertEqual(last_result(self.model.received[2])["reason"], tools_module.FOREIGN_VALUE.format(param="data"))
        self.assertNotIn("scale", self.fakes.ran)
        self.assertEqual(await self.uses(result), ["sources.areas.resolve"])
        # Refused calls are tool calls all the same.
        self.assertEqual((await self.explore_entry(result)).result["tool_calls"], 2)
        self.assertNotIn(foreign, result.handles)

    async def test_the_allowlist_grows_with_each_handle_a_call_returns(self):
        given = await self.mint_scalar()
        [result] = await self.go(task(handles=(given,)), script=[
            tool_call("operations__scale", {"data": given}),
            call_on_last("operations__scale"),
            final(findings=on_last_scalar("scaled twice")),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.fakes.ran, ["scale", "scale"])
        self.assertEqual(len(result.handles), 3)
        self.assertEqual(result.handles[0], given)
        self.assertEqual(result.findings[0].value, 28.0)


class PlaceTests(ExploreCase):
    async def test_places_are_resolved_before_the_model_and_join_the_handles(self):
        [result] = await self.go(task(places=("NJ",)), script=[final()])
        area = result.handles[0]
        self.assertTrue(area.startswith("polygons_"))
        self.assertIn(f"- {area} (polygons): Area: New Jersey", human(self.model.received[0]))
        self.assertIn("'NJ' was resolved to New Jersey.", result.disclosures)

    async def test_a_refused_place_ends_the_task_before_any_model_call(self):
        [result] = await self.go(task(places=("NJ + atlantis",)), script=[final()])
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(self.model.calls, 0)
        self.assertIn("'atlantis' in 'NJ + atlantis' could not be found", result.question["prompt"])
        self.assertEqual(result.reason, tools_module.QUESTION_REASON.format(name="sources.areas.resolve"))
        self.assertEqual(await self.run_status(result), "needs_input")
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.result["status"], entry.result["model_calls"]), ("waiting", "needs_input", 0))

    async def test_an_unknown_place_asks_without_a_model_call(self):
        [result] = await self.go(task(places=("Atlantis",)), script=[final()])
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(self.model.calls, 0)
        self.assertIn("'Atlantis' could not be found as a place", result.question["prompt"])

    async def test_a_place_that_fails_ends_the_task_with_its_category_and_reason(self):
        class DownPlaces:
            async def resolve(self, wording: str) -> Any:
                raise FakeToolError("provider_unavailable", "https://secret.example/geocode?key=abc")

        self.places = DownPlaces()  # type: ignore[assignment]
        [result] = await self.go(task(places=("NJ",)), script=[final()])
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(
            result.reason, "sources.areas.resolve could not reach its data source, which is temporarily unavailable.",
        )
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.SOURCE_UNAVAILABLE))
        self.assertNotIn("secret", json.dumps([e.reason for e in await self.entries(result)]))

    async def test_a_given_handle_that_does_not_exist_fails_the_task(self):
        [result] = await self.go(task(handles=("scalar_" + "0" * 32,)), script=[final()])
        self.assertEqual((result.status, result.reason), ("failed", UNKNOWN_HANDLE))
        self.assertEqual(self.model.calls, 0)


class QuestionTests(ExploreCase):
    async def test_a_question_from_a_tool_ends_the_task_with_its_wording(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("sources__fake__ask", {"label": "level"}, content="I will ask which level and then guess."),
            final(),
        ])
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(result.question, {
            "kind": "pick_one", "prompt": "Which level should be used?", "options": ["surface", "column"], "items": [],
        })
        self.assertEqual(result.reason, "sources.fake.ask needs an answer from the user before the exploration can continue.")
        self.assertIsNone(result.interpretation)
        self.assertEqual(await self.run_status(result), "needs_input")

    async def test_a_mid_task_area_refusal_stops_the_task_before_another_area_is_used(self):
        [result] = await self.go(task(), script=[
            tool_call("sources__areas__resolve", {"place": "NJ + atlantis"}),
            tool_call("sources__areas__resolve", {"place": "New Jersey"}),
            final("New Jersey has a mean of 2.5."),
        ])
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(self.model.calls, 1)
        self.assertIn("atlantis", result.question["prompt"])
        self.assertIsNone(result.interpretation)
        self.assertEqual(self.places.asked, ["NJ + atlantis"])


class NoteTests(ExploreCase):
    async def test_the_system_prompt_holds_the_role_group_notes_and_notes_for_the_inputs(self):
        given = await self.mint_scalar(product="TEMPO_NO2")
        [result] = await self.go(task(handles=(given,)), script=[final()])
        system = str(next(m for m in self.model.received[0] if isinstance(m, SystemMessage)).content)
        self.assertTrue(system.startswith(ROLE))
        self.assertIn("sources.fake: Fake source notes: measure takes an area handle.", system)
        self.assertIn("operations: Fake operations notes.", system)
        self.assertIn(f"TEMPO note: {TOMORROW_NOTE}", system)
        self.assertNotIn(COLUMN_NOTE, system)
        self.assertNotIn(OZONE_NOTE, system)

    async def test_a_note_comes_with_the_tool_result_that_brings_it_into_play_once(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            call_on_area("sources__fake__measure", value=3.0),
            tool_call("sources__fake__monitors", {"pollutant": "o3"}),
            final(),
        ])
        system = str(self.model.received[0][0].content)
        self.assertNotIn(TOMORROW_NOTE, system)
        first, second, monitors = results(self.model.received[3])[0:3]
        self.assertEqual(first["notes"], [f"Column note: {COLUMN_NOTE}", f"TEMPO note: {TOMORROW_NOTE}"])
        self.assertNotIn("notes", second)
        # A monitor result names no pollutant; it comes from the call's own parameters.
        self.assertEqual(monitors["notes"], [f"Ozone note: {OZONE_NOTE}"])
        # The system prompt never changes during the task, so its cached prefix holds.
        self.assertEqual({str(call[0].content) for call in self.model.received}, {system})

    async def test_notes_shown_in_the_prompt_are_not_repeated_on_a_result(self):
        given = await self.mint_scalar(product="TEMPO_NO2")
        [result] = await self.go(task(handles=(given,), places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final(),
        ])
        self.assertEqual(last_result(self.model.received[1])["notes"], [f"Column note: {COLUMN_NOTE}"])


class DateTests(ExploreCase):
    clock = FixedClock(datetime.datetime(2026, 3, 4, 21, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))))

    async def test_the_task_message_opens_with_the_utc_date_from_the_clock(self):
        [result] = await self.go(
            task("Map NO2 last week.", places=("NJ",), variable="NO2", dates="last week"), script=[final()],
        )
        message = human(self.model.received[0])
        self.assertTrue(message.startswith(
            "Today's date is 2026-03-05 (UTC). It is authoritative: trust it over any assumption about the current "
            "date, and work out relative dates such as \"yesterday\" or \"last week\" from it. Never refuse a date "
            "on or before today as being in the future. If you doubt that data exists for a date, check with a "
            "tool and report what it returns."
        ))
        self.assertIn("Objective: Map NO2 last week.", message)
        self.assertIn("Variable, as the user wrote it: NO2", message)
        self.assertIn("Dates, as the user wrote them: last week", message)
        self.assertNotIn("2026-03-05", str(self.model.received[0][0].content))


class BudgetTests(ExploreCase):
    async def test_running_out_of_model_calls_ends_incomplete_keeping_what_was_found(self):
        [result] = await self.go(task(places=("NJ",)), limits=ExploreLimits(model_calls=2), script=[
            call_on_area("sources__fake__measure"),
            call_on_area("sources__fake__measure", value=4.0),
            final(),
        ])
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.reason, MODEL_BUDGET.format(limit=2))
        self.assertEqual(self.model.calls, 2)
        self.assertEqual([f.value for f in result.findings], [2.5, 4.0])
        self.assertEqual(result.findings[0].name, "Mean NO2 column over the area")
        self.assertEqual(result.findings[0].unit, "molecules/cm^2")
        self.assertEqual(len(result.handles), 3)
        self.assertIn("Measured over the area.", result.disclosures)
        self.assertEqual(await self.run_status(result), "incomplete")
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.result["status"], entry.reason), ("done", "incomplete", result.reason))

    async def test_running_out_of_tool_calls_ends_incomplete_before_the_extra_call_runs(self):
        [result] = await self.go(task(places=("NJ",)), limits=ExploreLimits(tool_calls=1), script=[
            call_on_area("sources__fake__measure"),
            call_on_last("operations__scale"),
            final(),
        ])
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.reason, TOOL_BUDGET.format(limit=1))
        self.assertEqual(self.fakes.ran, ["measure"])
        self.assertEqual([f.value for f in result.findings], [2.5])
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.fake.measure"])

    async def test_the_answer_is_not_counted_as_a_tool_call(self):
        [result] = await self.go(task(places=("NJ",)), limits=ExploreLimits(tool_calls=1), script=[
            call_on_area("sources__fake__measure"),
            final(findings=on_last_scalar()),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual((await self.explore_entry(result)).result["tool_calls"], 1)

    async def test_a_reprompt_the_budget_stops_keeps_the_checked_findings(self):
        [result] = await self.go(task(places=("NJ",)), limits=ExploreLimits(model_calls=2), script=[
            call_on_area("sources__fake__measure"),
            final("Mean is 2.5.", findings=lambda m: [
                *on_last_scalar()(m), {"name": "bad", "value": 1, "handle": "scalar_" + "1" * 32},
            ]),
            final(),
        ])
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.interpretation, "Mean is 2.5.")
        self.assertEqual([f.name for f in result.findings], ["mean"])


class WaitingTests(ExploreCase):
    async def test_a_job_still_running_ends_the_task_waiting_with_the_job_in_the_trace(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("sources__fake__fetch", {"label": "big"}),
            final(),
        ])
        self.assertEqual(result.status, "waiting")
        self.assertEqual(result.reason, tools_module.JOB_WAITING.format(name="sources.fake.fetch"))
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(await self.run_status(result), "waiting")
        entries = await self.entries(result)
        fetches = [e for e in entries if e.capability == "sources.fake.fetch"]
        self.assertEqual([e.status for e in fetches], ["waiting", "failed"])
        self.assertEqual(fetches[0].result, {"job": {"job": "job_1"}})
        self.assertEqual(fetches[1].failure_category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertEqual(fetches[1].result["still_running"], ["job_1"])
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.result["status"]), ("waiting", "waiting"))

    async def test_the_wait_is_shortened_only_below_the_hosts_timeout(self):
        env = await self.build_env(None)(self.context, await self._started_run())
        limits = with_wait(env, ExploreLimits(retrieval_wait_seconds=0.05)).limits
        self.assertEqual(limits.retrieval_timeout_seconds, 0.05)
        self.assertLessEqual(limits.poll_min_seconds, limits.poll_max_seconds)
        self.assertLessEqual(limits.poll_max_seconds, 0.05)
        self.assertIs(with_wait(env, ExploreLimits(retrieval_wait_seconds=600)), env)
        self.assertIs(with_wait(env, ExploreLimits()), env)


class AnswerTests(ExploreCase):
    async def test_a_checked_finding_carries_the_scalars_value_unit_region_and_period(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("Mean is 2.5.", findings=on_last_scalar(unit="ppb"), followups=["What about July?", " "]),
        ])
        self.assertEqual(result.status, "done")
        [finding] = result.findings
        scalar = result.handles[1]
        self.assertEqual((finding.name, finding.value, finding.unit), ("mean", 2.5, "molecules/cm^2"))
        self.assertEqual((finding.region, finding.period, finding.source_handle), ("New Jersey", "2024-06-01", scalar))
        self.assertEqual(result.interpretation, "Mean is 2.5.")
        self.assertEqual(result.followups, ("What about July?",))
        self.assertEqual(result.mode.kind, "exploratory")
        self.assertEqual(self.model.calls, 2)

    async def test_a_failing_answer_is_reprompted_once_with_its_errors_and_only_the_answer_tool(self):
        foreign = "scalar_" + "1" * 32
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("Mean is 2.6.", findings=lambda m: [
                *on_last_scalar(value=2.6)(m), {"name": "other", "value": 1, "handle": foreign},
            ]),
            final("Mean is 2.5.", findings=on_last_scalar()),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 3)
        correction = str(self.model.received[2][-1].content)
        scalar = result.handles[1]
        self.assertEqual(correction, CORRECTION.format(errors="\n".join([
            f"- finding 'mean': {scalar} is a scalar holding 2.5; give exactly that value",
            f"- finding 'other': handle {foreign} isn't part of this task",
        ])))
        self.assertEqual(self.model.tools_offered[2], [ANSWER_TOOL])
        self.assertEqual([(f.name, f.value) for f in result.findings], [("mean", 2.5)])
        self.assertEqual(result.interpretation, "Mean is 2.5.")

    async def test_a_second_failing_answer_is_salvaged_without_its_bad_findings(self):
        bad = {"name": "bad", "value": 1, "handle": "scalar_" + "1" * 32}
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("First try.", findings=lambda m: [bad]),
            final("Mean is 2.5.", findings=lambda m: [*on_last_scalar()(m), bad]),
            final("Never asked."),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(result.interpretation, "Mean is 2.5.")
        self.assertEqual([f.name for f in result.findings], ["mean"])

    async def test_a_salvaged_answer_without_a_summary_uses_fixed_wording(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("  ", findings=on_last_scalar()),
            final("  ", findings=on_last_scalar()),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(result.interpretation, NO_SUMMARY)
        self.assertEqual([f.value for f in result.findings], [2.5])

    async def test_a_wrong_shaped_answer_keeps_its_summary_and_the_findings_that_check(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("Mean is 2.5.", findings=on_last_scalar(), charts=["chart-9"]),
            final("Mean is 2.5.", findings=on_last_scalar(), handles=["scalar_" + "2" * 32]),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(result.interpretation, "Mean is 2.5.")
        self.assertEqual([f.value for f in result.findings], [2.5])
        self.assertEqual(result.charts, ())
        self.assertNotIn("scalar_" + "2" * 32, result.handles)

    async def test_plain_text_twice_keeps_the_text_as_the_summary(self):
        [result] = await self.go(task(places=("NJ",)), script=[answer("It is about 2.5."), answer("Still 2.5.")])
        self.assertEqual(result.status, "done")
        self.assertEqual(result.interpretation, "Still 2.5.")
        self.assertEqual(self.model.calls, 2)

    async def test_nothing_usable_twice_fails_with_fixed_wording(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call(ANSWER_TOOL, {"findings": "none"}),
            tool_call(ANSWER_TOOL, {"summary": 3}),
        ])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "The language model's answer for this exploration did not fit its declared shape.")
        entry = await self.explore_entry(result)
        self.assertEqual(entry.failure_category, FailureCategory.MODEL_FAILURE)
        self.assertEqual(entry.result["model_calls"], 2)

    async def test_an_accepted_answer_runs_no_other_call_from_its_turn(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            turns_together(
                lambda m: (ANSWER_TOOL, {"summary": "Mean is 2.5.", "findings": on_last_scalar()(m), "followups": []}),
                lambda m: ("operations__scale", {"data": last_result(m)["handle"]}),
            ),
            final("Never asked."),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 2)
        self.assertEqual(self.fakes.ran, ["measure"])
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.fake.measure"])
        self.assertEqual(len(result.handles), 2)
        self.assertEqual(result.interpretation, "Mean is 2.5.")

    async def test_a_rejected_answer_runs_no_other_call_from_its_turn(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            turns_together(
                (ANSWER_TOOL, {"summary": "Bad.", "findings": [{"name": "x", "value": 1, "handle": "scalar_" + "3" * 32}]}),
                lambda m: ("sources__fake__measure", {"area": input_handle(m, "polygons")}),
            ),
            final("Nothing measured."),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.fakes.ran, [])
        skipped = [m for m in self.model.received[1] if isinstance(m, ToolMessage) and m.name == "sources__fake__measure"]
        self.assertEqual([m.content for m in skipped], [NOT_RUN])
        self.assertEqual(result.interpretation, "Nothing measured.")

    async def test_each_attempt_is_judged_on_its_own_turn(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            final("First, with a wrong value.", findings=on_last_scalar(value=9.9)),
            answer("Second, in plain words."),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(result.interpretation, "Second, in plain words.")
        self.assertEqual(result.findings, ())

    async def test_disclosures_charts_and_handles_come_from_the_calls(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            call_on_last("render__draw"),
            final("Drawn.", findings=lambda m: []),
        ])
        self.assertEqual(result.status, "done")
        self.assertEqual(result.charts, ("chart-1",))
        self.assertEqual(result.disclosures, (
            "'NJ' was resolved to New Jersey.",
            "Measured over the area.",
            "Product: TEMPO_NO2, chosen by rule approved_preference.",
            "Drawn as dots.",
        ))
        self.assertEqual([h.split("_")[0] for h in result.handles], ["polygons", "scalar"])
        self.assertEqual(result.gaps, ())


class FailureTests(ExploreCase):
    async def test_every_model_failing_ends_the_task_failed_with_fixed_wording(self):
        primary = Model(script=[rate_limited(), rate_limited(), rate_limited()])
        fallback = Model(script=[RuntimeError("provider said: key sk-secret at https://x.example")])
        [result] = await self.go(task(places=("NJ",)), model=primary,
                                  models=ScriptedModels({"explore": choice_of(primary, fallback)}))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "The language model for this exploration did not answer.")
        self.assertEqual((primary.calls, fallback.calls), (3, 1))
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.MODEL_FAILURE))
        self.assertEqual(entry.result["status"], "failed")
        self.assertEqual(entry.result["model"], "fake:explore")
        self.assertNotIn("secret", json.dumps([e.reason for e in await self.entries(result)]))
        self.assertEqual(await self.run_status(result), "failed")

    async def test_a_missing_model_fails_the_task(self):
        [result] = await self.go(task(places=("NJ",)), models=None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, NO_MODEL_REASON.format(role="explore"))
        entry = await self.explore_entry(result)
        self.assertEqual((entry.failure_category, entry.result["model"]), (FailureCategory.INTERNAL_ERROR, "none"))
        self.assertEqual(await self.uses(result), [])

    async def test_an_internal_error_fails_the_task_with_fixed_wording(self):
        async def broken(context: RunContext, run_id: str) -> RunEnv:
            raise RuntimeError("database at postgres://user:pw@host")

        results_ = await explore(self.context, broken, [task()], registry=self.registry, notes=NOTES)
        [result] = results_
        self.assertEqual((result.status, result.reason), ("failed", INTERNAL_REASON))
        self.assertEqual(await self.run_status(result), "failed")

    async def test_a_tool_that_raises_unexpectedly_fails_only_its_task(self):
        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("unexpected")

        caller_call = Caller.call

        async def failing(self_: Caller, capability: Any, params: Any) -> Any:
            if capability.name == "render.draw":
                await boom()
            return await caller_call(self_, capability, params)

        Caller.call = failing  # type: ignore[method-assign]
        self.addCleanup(setattr, Caller, "call", caller_call)
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            call_on_last("render__draw"),
            final(),
        ])
        self.assertEqual((result.status, result.reason), ("failed", INTERNAL_REASON))
        self.assertEqual(len(result.handles), 2)
        entry = await self.explore_entry(result)
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.INTERNAL_ERROR))


class TraceTests(ExploreCase):
    async def test_one_explore_entry_wraps_every_call_with_the_meters_counts(self):
        [result] = await self.go(task(places=("NJ",), variable="NO2"), script=[
            call_on_area("sources__fake__measure", ),
            call_on_last("operations__scale"),
            final("Done.", findings=on_last_scalar()),
        ])
        entries = await self.entries(result)
        [entry] = [e for e in entries if e.kind == "explore"]
        self.assertEqual(entry.params, {
            "objective": "Find the mean NO2 over the area.", "bundle": "analyse",
            "inputs": {"handles": [], "places": ["NJ"], "variable": "NO2"},
        })
        self.assertEqual(entry.result, {
            "status": "done", "model": "fake:explore", "model_calls": 3, "tool_calls": 2,
            "input_tokens": 30, "output_tokens": 10, "cache_read_tokens": 0, "cache_creation_tokens": 0,
        })
        children = [e for e in entries if e.kind == "use"]
        self.assertEqual([e.capability for e in children],
                         ["sources.areas.resolve", "sources.fake.measure", "operations.scale"])
        self.assertTrue(all(e.parent_seq == entry.seq for e in children))
        self.assertEqual(len(entries), 4)
        run = await self.stores.traces.get_run("user-a", result.trace_id)
        self.assertEqual((run.status, run.mode.kind), ("done", "exploratory"))
        self.assertEqual(result.trace_id, run.id)

    async def test_each_task_is_its_own_run(self):
        first, second = await self.go(
            task("One.", places=("NJ",)), task("Two.", bundle="find-data", places=("NJ",)),
            script=[final(), final()],
        )
        self.assertNotEqual(first.trace_id, second.trace_id)
        self.assertEqual((await self.explore_entry(second)).params["bundle"], "find-data")


class ConcurrencyTests(ExploreCase):
    async def test_tasks_run_concurrently_and_one_failing_never_fails_another(self):
        turns: dict[str, list[Turn]] = {
            "Meet and answer.": [tool_call("sources__fake__meet", {"label": "a"}), final("Met.")],
            "Meet and fail.": [tool_call("sources__fake__meet", {"label": "b"}),
                               rate_limited(), rate_limited(), rate_limited()],
        }

        model = Routed(turns=turns)
        done, failed = await self.go(
            task("Meet and answer.", places=("NJ",)), task("Meet and fail.", places=("NJ",)),
            model=model, models=ScriptedModels({"explore": choice_of(model)}),
        )
        self.assertEqual(self.fakes.arrived, 2)
        self.assertEqual((done.status, done.interpretation), ("done", "Met."))
        self.assertEqual((failed.status, failed.reason), ("failed", "The language model for this exploration did not answer."))
        self.assertEqual(await self.run_status(done), "done")
        self.assertEqual(await self.run_status(failed), "failed")

    async def test_at_most_max_concurrent_tasks_run_at_once_and_results_keep_task_order(self):
        objectives = [f"Hold {n}." for n in range(5)]
        model = Routed(turns={
            objective: [tool_call("sources__fake__hold", {"label": objective}), final(objective)]
            for objective in objectives
        })
        results_ = await explore(
            self.context, self.build_env(ScriptedModels({"explore": choice_of(model)})),
            [task(objective) for objective in objectives], registry=self.registry, notes=NOTES,
            backoff=NO_BACKOFF, max_concurrent=2,
        )
        self.assertEqual(self.fakes.peak, 2)
        self.assertEqual([r.interpretation for r in results_], objectives)
        self.assertTrue(all(r.status == "done" for r in results_))
        with self.assertRaises(ValueError):
            await explore(self.context, self.build_env(None), [], registry=self.registry, notes=NOTES, max_concurrent=0)

    async def test_a_cancellation_while_the_run_is_finished_still_leaves_it_finished(self):
        traces = self.stores.traces
        entered, release = asyncio.Event(), asyncio.Event()

        class SlowFinish:
            def __getattr__(self, name: str) -> Any:
                return getattr(traces, name)

            async def finish_run(self, *args: Any, **kwargs: Any) -> Any:
                entered.set()
                await release.wait()
                return await traces.finish_run(*args, **kwargs)

        stores = Stores(handles=self.stores.handles, traces=SlowFinish())  # type: ignore[arg-type]
        self.context = make_run_context(clock=self.clock, stores=stores)
        running = asyncio.ensure_future(self.go(task(places=("NJ",)), script=[final()]))
        await asyncio.wait_for(entered.wait(), timeout=5)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        release.set()
        for _ in range(20):
            await asyncio.sleep(0)
        [run] = await traces.list_runs("user-a", "conv-1")
        self.assertEqual(run.status, "done")

    async def test_no_tasks_is_no_results(self):
        self.assertEqual(await explore(self.context, self.build_env(None), [], registry=self.registry, notes=NOTES), [])


# Prompt caching, retry hints, repeated calls, workflow-only parameters and unresolved places


async def _nothing(call: Call[Any]) -> CapabilityResult:
    return CapabilityResult(status="done")


def fake_capability(name: str, params: Any, run: Any = None, **kw: Any) -> Capability[Any]:
    return Capability(
        name=name, description=f"Fake {name}.", params=params, output_kind=kw.pop("output_kind", None),
        label=f"Running {name}", method="The fake step ran.", retrieves=kw.pop("retrieves", False),
        retry_safe=kw.pop("retry_safe", True), run=run or _nothing, **kw,
    )


def breakpoints(messages: list[BaseMessage]) -> list[int]:
    return [
        i for i, m in enumerate(messages)
        if isinstance(m.content, list) and any(isinstance(b, dict) and "cache_control" in b for b in m.content)
    ]


def with_cache_usage(turn: Turn, read: int, created: int) -> Turn:
    def cached(m: list[BaseMessage]) -> AIMessage:
        message = turn(m) if callable(turn) and not isinstance(turn, AIMessage) else turn
        assert isinstance(message, AIMessage)
        message.usage_metadata = {
            "input_tokens": 100, "output_tokens": 4, "total_tokens": 104,
            "input_token_details": {"cache_read": read, "cache_creation": created},
        }
        return message
    return cached


class PromptCacheTests(ExploreCase):
    async def cached_go(self, *tasks: ExploreTask, script: list[Turn], on: bool = True) -> list[Result]:
        model = Model(script=list(script))
        choice = ModelChoice(primary=NamedModel(model, "fake:explore"), prompt_cache=on)
        return await self.go(*tasks, model=model, models=ScriptedModels({"explore": choice}))

    async def test_only_the_system_prompt_carries_a_breakpoint(self):
        [result] = await self.cached_go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure"),
            call_on_last("operations__scale"),
            final(findings=on_last_scalar()),
        ])
        self.assertEqual(result.status, "done")
        system = self.model.received[0][0]
        self.assertIsInstance(system, SystemMessage)
        [block] = system.content
        self.assertEqual(set(block), {"type", "text", "cache_control"})
        self.assertEqual(block["cache_control"], {"type": "ephemeral"})
        self.assertTrue(block["text"].startswith(ROLE))
        # The conversation is marked by the host's model, in the payload it sends.
        for sent in self.model.received:
            self.assertEqual(breakpoints(sent), [0])

    async def test_without_caching_the_prompt_is_plain_and_nothing_is_marked(self):
        [result] = await self.cached_go(task(places=("NJ",)), script=[call_on_area("sources__fake__measure"), final()],
                                        on=False)
        self.assertEqual(result.status, "done")
        self.assertIsInstance(self.model.received[0][0].content, str)
        self.assertTrue(all(breakpoints(sent) == [] for sent in self.model.received))

    async def test_the_explore_entry_records_cache_reads_and_writes(self):
        [result] = await self.cached_go(task(places=("NJ",)), script=[
            with_cache_usage(call_on_area("sources__fake__measure"), 0, 90),
            with_cache_usage(final(findings=on_last_scalar()), 90, 6),
        ])
        entry = await self.explore_entry(result)
        self.assertEqual(
            {k: entry.result[k] for k in ("input_tokens", "cache_read_tokens", "cache_creation_tokens")},
            {"input_tokens": 200, "cache_read_tokens": 90, "cache_creation_tokens": 96},
        )


class RetryHintTests(unittest.TestCase):
    def test_every_category_says_whether_a_retry_can_help(self):
        self.assertEqual(set(tools_module.RETRY_HELPS), set(FailureCategory))
        safe = fake_capability("sources.fake.safe", LabelParams, retry_safe=True)
        unsafe = fake_capability("sources.fake.unsafe", LabelParams, retry_safe=False)
        expected = {
            "source_unavailable": (True, False), "input_evicted": (True, True),
            "internal_error": (False, False), "unsupported_shape": (False, False), "bad_input": (False, False),
            "no_data": (False, False), "request_too_large": (False, False), "credential_invalid": (False, False),
        }
        for category, (on_safe, on_unsafe) in expected.items():
            with self.subTest(category=category):
                self.assertEqual(tools_module.retry_helps(safe, FailureCategory(category)), on_safe)
                self.assertEqual(tools_module.retry_helps(unsafe, FailureCategory(category)), on_unsafe)
        self.assertFalse(tools_module.retry_helps(safe, FailureCategory.SOURCE_UNAVAILABLE, repeated=True))

    def test_the_role_prompt_says_a_retry_false_call_will_fail_again(self):
        self.assertIn("A failed call marked retry: false will not succeed if repeated.", ROLE)


class EmptyParams(Params):
    label: str = "x"
    n: float = 1.0
    day: datetime.date | None = None


class ConsumeParams(Params):
    data: Annotated[str, handle_param("scalar")]


class RepeatedCallTests(ExploreCase):
    def add_empty(self) -> None:
        async def empty(call: Call[EmptyParams]) -> CapabilityResult:
            self.fakes.ran.append("empty")
            return CapabilityResult(status="insufficient_evidence", reason="No data for the request.")

        self.registry.register(fake_capability("sources.fake.empty", EmptyParams, empty))

    def add_flaky(self, *, retry_safe: bool) -> None:
        async def flaky(call: Call[LabelParams]) -> CapabilityResult:
            self.fakes.ran.append("flaky")
            raise FakeToolError("provider_unavailable", "https://secret.example/down")

        self.registry.register(fake_capability("sources.fake.flaky", LabelParams, flaky, retry_safe=retry_safe))

    async def test_an_identical_failed_call_is_not_run_again_and_a_changed_one_is(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            call_on_area("sources__fake__measure", value="not a number"),
            call_on_area("sources__fake__measure", value="not a number"),
            call_on_area("sources__fake__measure", value=3.0),
            final(),
        ])
        self.assertEqual(result.status, "done")
        seen = results(self.model.received[3])
        self.assertEqual(seen[0]["retry"], False)
        self.assertEqual(seen[1], {
            "status": "refused", "reason": tools_module.REPEATED_CALL.format(category="bad_input"),
        })
        self.assertEqual(seen[1]["reason"], (
            "This exact call already failed in this task with bad_input; it will fail the same way. "
            "Change the request or answer with what you have."
        ))
        self.assertEqual((seen[2]["status"], seen[2]["summary"]), ("done", {"value": 3.0}))
        # The repeat wrote no trace entry; the changed call ran.
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.fake.measure", "sources.fake.measure"])
        self.assertEqual(self.fakes.ran, ["measure"])
        self.assertEqual((await self.explore_entry(result)).result["tool_calls"], 3)

    async def test_the_same_parameters_in_another_order_are_the_same_call(self):
        [result] = await self.go(task(places=("NJ",)), script=[
            lambda m: tool_call("sources__fake__measure", {"area": input_handle(m, "polygons"), "value": "x"}),
            lambda m: tool_call("sources__fake__measure", {"value": "x", "area": input_handle(m, "polygons")}),
            final(),
        ])
        self.assertEqual(results(self.model.received[2])[1]["status"], "refused")
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.fake.measure"])

    async def test_a_repeat_of_a_foreign_handle_refusal_gets_the_fixed_message(self):
        foreign = await self.mint_scalar()
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("operations__scale", {"data": foreign}),
            tool_call("operations__scale", {"data": foreign}),
            final(),
        ])
        first, second = results(self.model.received[2])
        self.assertEqual(first["reason"], tools_module.FOREIGN_HANDLE.format(handle=foreign))
        self.assertEqual(second["reason"], tools_module.REPEATED_CALL.format(category="foreign_handle"))
        self.assertEqual(await self.uses(result), ["sources.areas.resolve"])

    async def test_an_unavailable_retry_safe_source_is_repeated_once(self):
        self.add_flaky(retry_safe=True)
        call = tool_call("sources__fake__flaky", {"label": "a"})
        [result] = await self.go(task(places=("NJ",)), script=[
            call, call.model_copy(), call.model_copy(), final(),
        ])
        first, second, third = results(self.model.received[3])
        self.assertEqual((first["category"], first["retry"]), ("source_unavailable", True))
        self.assertEqual((second["category"], second["retry"]), ("source_unavailable", False))
        self.assertEqual(third["reason"], tools_module.REPEATED_CALL.format(category="source_unavailable"))
        self.assertEqual(self.fakes.ran, ["flaky", "flaky"])
        self.assertNotIn("secret", json.dumps(results(self.model.received[3])))
        self.assertEqual(result.status, "done")

    async def test_a_repeat_of_a_call_that_found_no_data_is_refused(self):
        self.add_empty()
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("sources__fake__empty", {"n": 2}),
            tool_call("sources__fake__empty", {"n": 2}),
            tool_call("sources__fake__empty", {"n": 3}),
            final(),
        ])
        first, second, third = results(self.model.received[3])
        self.assertEqual((first["status"], first["retry"]), ("insufficient_evidence", False))
        self.assertEqual(second, {"status": "refused", "reason": tools_module.REPEATED_CALL.format(category="no_data")})
        self.assertEqual(third["status"], "insufficient_evidence")
        self.assertEqual(self.fakes.ran, ["empty", "empty"])
        self.assertEqual(result.status, "done")

    async def test_equivalent_parameters_are_the_same_call(self):
        self.add_empty()
        await self.go(task(places=("NJ",)), script=[
            tool_call("sources__fake__empty", {"n": 1, "day": "2024-06-01"}),
            # The default label given, 1.0 for 1 and the date written with a time are the same request.
            tool_call("sources__fake__empty", {"label": "x", "n": 1.0, "day": "2024-06-01T00:00:00"}),
            final(),
        ])
        first, second = results(self.model.received[2])
        self.assertEqual(first["status"], "insufficient_evidence")
        self.assertEqual(second["status"], "refused")
        self.assertEqual(self.fakes.ran, ["empty"])

    def test_the_key_is_the_validated_parameters_or_the_given_ones(self):
        capability = fake_capability("sources.fake.empty", EmptyParams)
        key = tools_module.call_key
        self.assertEqual(key(capability, {}), key(capability, {"label": "x", "n": 1, "day": None}))
        self.assertNotEqual(key(capability, {"n": 1}), key(capability, {"n": 2}))
        # Parameters that do not validate are keyed as given.
        self.assertEqual(key(capability, {"n": "lots"}), 'sources.fake.empty {"n":"lots"}')

    async def test_a_success_clears_the_record_so_a_later_failure_counts_from_scratch(self):
        outcomes = ["fail", "ok", "fail", "fail"]

        async def sometimes(call: Call[LabelParams]) -> CapabilityResult:
            self.fakes.ran.append("sometimes")
            if outcomes.pop(0) == "fail":
                raise FakeToolError("provider_unavailable")
            return CapabilityResult(status="done", summary={"ok": True})

        self.registry.register(fake_capability("sources.fake.sometimes", LabelParams, sometimes))
        call = tool_call("sources__fake__sometimes", {"label": "a"})
        await self.go(task(places=("NJ",)), script=[*(call.model_copy() for _ in range(5)), final()])
        seen = results(self.model.received[5])
        self.assertEqual([(s["status"], s.get("retry")) for s in seen], [
            ("failed", True), ("done", None), ("failed", True), ("failed", False), ("refused", None),
        ])
        self.assertEqual(len(self.fakes.ran), 4)

    async def evicted(self, script: list[Turn], handles: tuple[str, ...] = ()) -> dict[str, Any]:
        async def consume(call: Call[ConsumeParams]) -> CapabilityResult:
            raise InputEvicted(call.capability.name, [call.params.data])

        self.registry.register(fake_capability("operations.consume", ConsumeParams, consume))
        await self.go(task(places=("NJ",), handles=handles), script=[*script, final()])
        return last_result(self.model.received[-1])

    async def test_an_evicted_input_made_in_the_task_can_be_remade(self):
        seen = await self.evicted([call_on_area("sources__fake__measure"), call_on_last("operations__consume")])
        self.assertEqual((seen["category"], seen["retry"]), ("input_evicted", True))

    async def test_an_evicted_input_the_task_was_given_cannot_be_remade(self):
        given = await self.mint_scalar()
        seen = await self.evicted([tool_call("operations__consume", {"data": given})], handles=(given,))
        self.assertEqual((seen["category"], seen["retry"]), ("input_evicted", False))
        self.assertEqual(seen["reason"], "operations.consume needed data that is no longer stored; re-run the step that made it.")

    async def test_an_unavailable_source_that_is_not_retry_safe_is_not_repeated(self):
        self.add_flaky(retry_safe=False)
        call = tool_call("sources__fake__flaky", {"label": "a"})
        await self.go(task(places=("NJ",)), script=[call, call.model_copy(), final()])
        first, second = results(self.model.received[2])
        self.assertEqual((first["category"], first["retry"]), ("source_unavailable", False))
        self.assertEqual(second["status"], "refused")
        self.assertEqual(self.fakes.ran, ["flaky"])


class ApprovedParams(Params):
    label: str = "x"
    approved: list[str] | str = "registry_default"


class WorkflowOnlyParamTests(ExploreCase):
    def test_the_catalogues_approved_parameter_is_hidden_from_specialists(self):
        registry = build_registry()
        for name in ("sources.satellite.get_data", "sources.satellite.get_point_series"):
            with self.subTest(name=name):
                capability = registry.get(name)
                self.assertIn("approved", capability.schema()["properties"])
                schema = tool_schema(capability)
                self.assertNotIn("approved", schema["properties"])
                self.assertNotIn("approved", schema.get("required", []))
                self.assertNotIn("approved products", json.dumps(schema))
        self.assertEqual(tools_module.WORKFLOW_ONLY_PARAMS, frozenset({"approved"}))

    async def test_a_call_setting_a_workflow_only_parameter_is_refused_and_one_without_it_runs(self):
        used: list[Any] = []

        async def choose(call: Call[ApprovedParams]) -> CapabilityResult:
            used.append(call.params.approved)
            return CapabilityResult(status="done", summary={"chosen": 1})

        self.registry.register(fake_capability("sources.fake.choose", ApprovedParams, choose))
        [result] = await self.go(task(places=("NJ",)), script=[
            tool_call("sources__fake__choose", {"label": "a", "approved": ["TROPOMI_NO2"]}),
            tool_call("sources__fake__choose", {"label": "a"}),
            final(),
        ])
        refused, ran = results(self.model.received[2])
        self.assertEqual(refused, {"status": "refused", "reason": tools_module.WORKFLOW_PARAM.format(
            param="approved", tool="sources__fake__choose")})
        self.assertEqual(ran["status"], "done")
        self.assertEqual(used, ["registry_default"])
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.fake.choose"])

    def test_a_required_workflow_only_parameter_cannot_be_hidden(self):
        class Required(Params):
            approved: list[str]

        capability = fake_capability("sources.fake.required", Required)
        with self.assertRaisesRegex(ValueError, "workflow-only"):
            capability_tool(capability, Caller(None, TaskState(), NOTES, 1))  # type: ignore[arg-type]


class UnresolvedPlaceTests(ExploreCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        async def find(call: Call[MonitorParams]) -> CapabilityResult:
            self.fakes.ran.append("find")
            return CapabilityResult(status="done", summary={"monitors": 4})

        self.registry.add_group(Group("sources.monitors", "Fake monitors."))
        self.registry.register(fake_capability("sources.monitors.find", MonitorParams, find, retrieves=True))

    async def test_a_find_data_task_carries_on_with_the_place_as_text_and_searches(self):
        [result] = await self.go(task("What NO2 data covers Atlantis?", bundle="find-data", places=("Atlantis",)),
                                 script=[tool_call("sources__monitors__find", {"pollutant": "no2"}), final("Four monitors.")])
        self.assertEqual(result.status, "done")
        self.assertEqual(self.model.calls, 2)
        self.assertEqual(self.fakes.ran, ["find"])
        message = human(self.model.received[0])
        self.assertIn("No handles were given.\n", message)
        self.assertNotIn("resolve or retrieve", message)
        self.assertIn("Places as the user wrote them:\n- Atlantis\n" + prompt_module.UNRESOLVED_NOTE, message)
        self.assertIn(UNRESOLVED_DISCLOSURE.format(place="Atlantis"), result.disclosures)
        self.assertIsNone(result.question)
        self.assertEqual(await self.uses(result), ["sources.areas.resolve", "sources.monitors.find"])
        self.assertEqual((await self.explore_entry(result)).status, "done")
        # Places are resolved in code only: the model cannot resolve another area in its place.
        self.assertNotIn("sources__areas__resolve", self.model.tools_offered[0])
        self.assertIn("sources__monitors__find", self.model.tools_offered[0])

    async def test_a_find_data_task_keeps_the_places_that_resolved_as_handles(self):
        [result] = await self.go(task(bundle="find-data", places=("NJ", "NJ + atlantis")), script=[final()])
        self.assertEqual(result.status, "done")
        message = human(self.model.received[0])
        self.assertIn(f"- {result.handles[0]} (polygons)", message)
        self.assertIn("- NJ + atlantis\n", message)

    async def test_a_place_that_fails_still_ends_a_find_data_task(self):
        class DownPlaces:
            async def resolve(self, wording: str) -> Any:
                raise FakeToolError("provider_unavailable")

        self.places = DownPlaces()  # type: ignore[assignment]
        [result] = await self.go(task(bundle="find-data", places=("NJ",)), script=[final()])
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.model.calls, 0)

    async def test_analyse_and_investigate_still_stop_on_an_unresolved_place(self):
        for bundle in ("analyse", "investigate"):
            with self.subTest(bundle=bundle):
                [result] = await self.go(task(bundle=bundle, places=("Atlantis",)), script=[final()])
                self.assertEqual(result.status, "needs_input")
                self.assertEqual(self.model.calls, 0)
                self.assertIn("'Atlantis' could not be found as a place", result.question["prompt"])


class TaskShapeTests(unittest.TestCase):
    def test_a_task_refuses_what_is_not_a_handle_id_and_unknown_fields(self):
        with self.assertRaises(ValueError):
            ExploreTask(objective="x", bundle="analyse", handles=("not a handle",))
        with self.assertRaises(ValueError):
            ExploreTask(objective="x", bundle="nope")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ExploreTask(objective="   ", bundle="analyse")
        with self.assertRaises(ValueError):
            ExploreTask.model_validate({"objective": "x", "bundle": "analyse", "inputs": {}})

    def test_limits_are_positive(self):
        with self.assertRaises(ValueError):
            ExploreLimits(model_calls=0)
        with self.assertRaises(ValueError):
            ExploreLimits(retrieval_wait_seconds=0)
        self.assertEqual((ExploreLimits().model_calls, ExploreLimits().tool_calls), (12, 20))

