"""The workflow runner in memory: step order, parallel steps, skips, stops, failures, pauses and the result."""
import asyncio
import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, Any, Literal
from unittest.mock import patch

import xarray as xr
from langgraph.checkpoint.memory import InMemorySaver

from tests.test_harness_grids import NJ, granule
from tests.test_harness_sources_satellite import NO2, TODAY, FakeService
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities import (
    Call,
    Capability,
    CapabilityRegistry,
    CapabilityResult,
    Group,
    JobStillRunning,
    Params,
    handle_param,
    needs_choice,
)
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.capabilities.failures import JobFailed
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, HandleRecord, ScalarInput
from tta_backend.harness.results import Gap, Mode, Result
from tta_backend.harness.run_context import PlaceRefused, RunContext, RunEnv
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.testing import (
    FakeEarthdata,
    FakePlaces,
    FakeToolError,
    FixedClock,
    MemoryArtifacts,
    RecordingEvents,
    StaticCredentials,
    env_factory,
    make_run_context,
)
from tta_backend.harness.trace import FailureCategory, RunNotFound, StepFailure
from tta_backend.harness.workflows import (
    AnswerNotSupported,
    compile_workflow,
    NotRunnable,
    RetrySettings,
    RunNotResumable,
    WorkflowRunner,
)
from tta_backend.harness.workflows.loader import UnknownWorkflow, WorkflowCatalogue, load_specs
from tta_backend.harness.workflows import compile as compile_module
from tta_backend.harness.workflows import runner as runner_module
from tta_backend.harness.workflows.runner import INTERNAL_REASON, JOB_WAITING, SOURCE_WAITING

DATASETS = load_registry()
SECRET = "fakeconn-secret-value-9c41"
CONNECTOR = "fakeconn"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "workflows"


class Script:
    """What the fake capabilities do and saw, for one test."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.statuses: dict[str, list[str]] = {}
        self.submit_failures = 0
        self.jobs = 0
        self.arrived = 0
        self.both = asyncio.Event()
        self.holding = asyncio.Event()
        self.release = asyncio.Event()
        self.before_complete: Any = None
        self.earthdata = FakeEarthdata({"submit": self._submit})

    def _submit(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.submit_failures > 0:
            self.submit_failures -= 1
            raise FakeToolError("provider_unavailable")
        self.jobs += 1
        return {"job": f"job_{self.jobs}"}

    def submissions(self) -> int:
        return self.jobs

    def submit_keys(self) -> list[str | None]:
        return [key for (tool, _), key in zip(self.earthdata.calls, self.earthdata.keys, strict=True) if tool == "submit"]


class MakeParams(Params):
    label: str
    value: float = 1.0
    a: Annotated[str | None, handle_param("scalar")] = None
    b: Annotated[str | None, handle_param("scalar")] = None
    area: Annotated[str | None, handle_param("polygons")] = None
    quantity: str | None = None
    approved: list[str] | Literal["registry_default"] | None = None
    product: str | None = None


class LabelParams(Params):
    label: str


class FailParams(Params):
    category: str


class DrawParams(Params):
    data: Annotated[str, handle_param("scalar")]


def fake_registry(script: Script) -> CapabilityRegistry:
    async def make(call: Call[MakeParams]) -> CapabilityResult:
        p = call.params
        script.calls.append(("ops.make", dict(call.recorded)))
        total = p.value
        for name in ("a", "b"):
            if getattr(p, name) is not None:
                total += (await call.open(name)).value
        product = p.product or (p.approved[0] if isinstance(p.approved, list) else None)
        record = await call.mint(
            "scalar", ScalarInput(value=total), Descriptor(summary=f"Made {p.label}", unit="ppb", product=product)
        )
        return call.done(record, summary={"value": total}, values={"label": p.label}, disclosures=(f"Made {p.label}.",))

    async def meet(call: Call[LabelParams]) -> CapabilityResult:
        # Both siblings must be inside at once, so they run in the same superstep.
        script.arrived += 1
        if script.arrived == 2:
            script.both.set()
        await asyncio.wait_for(script.both.wait(), timeout=5)
        record = await call.mint("scalar", ScalarInput(value=1.0), Descriptor(summary=f"Met {call.params.label}"))
        return call.done(record, summary={"value": 1.0})

    async def ask(call: Call[LabelParams]) -> CapabilityResult:
        return needs_choice("Which level should be used?", ["surface", "column"])

    async def empty(call: Call[LabelParams]) -> CapabilityResult:
        return CapabilityResult(status="insufficient_evidence", reason="The fake step found nothing valid.")

    async def fail(call: Call[FailParams]) -> CapabilityResult:
        raise StepFailure(FailureCategory(call.params.category), f"The fake step failed ({call.params.category}).")

    async def draw(call: Call[DrawParams]) -> CapabilityResult:
        return CapabilityResult(status="done", charts=("chart-1",), summary={"n": 3})

    async def fetch_begin(call: Call[LabelParams]) -> dict[str, Any] | CapabilityResult:
        request = {"label": call.params.label}
        reply = await call.earthdata.call("submit", request, idempotency_key=call.idempotency_key("submit", request))
        return {"job": reply["job"]}

    async def hold(call: Call[LabelParams]) -> CapabilityResult:
        script.holding.set()
        await script.release.wait()
        record = await call.mint("scalar", ScalarInput(value=1.0), Descriptor(summary="Held"))
        return call.done(record, summary={"value": 1.0})

    async def fetch_complete(call: Call[LabelParams], job: dict[str, Any]) -> CapabilityResult:
        job_id = str(job["job"])
        if script.before_complete is not None:
            await script.before_complete(call.env.run_id, job_id)
        queue = script.statuses.get(job_id, [])
        state = queue.pop(0) if queue else "ready"
        if state == "running":
            raise JobStillRunning(call.capability.name, job, [job_id])
        if state in ("failed", "expired"):
            raise JobFailed("The fake retrieval failed at the provider.", job=job, state=state, job_ids=[job_id])
        if state == "cancelled":
            raise StepFailure(FailureCategory.BAD_INPUT, "The fake retrieval was cancelled at the data provider.")
        if state == "hiccup":
            # A poll outage: the job itself is fine.
            raise FakeToolError("provider_unavailable")
        record = await call.mint("scalar", ScalarInput(value=1.0), Descriptor(summary="Fetched", unit="ppb"))
        return call.done(record, summary={"value": 1.0}, values={"job_id": job_id})

    def capability(name: str, params: Any, **kw: Any) -> Capability[Any]:
        return Capability(
            name=name, description=f"Fake {name}.", params=params, label=f"Running {name}",
            method=kw.pop("method", "The fake step ran."), retrieves=False, retry_safe=True, **kw,
        )

    registry = CapabilityRegistry()
    for group in (areas.GROUP, Group("ops", "Fake operations."), Group("render", "Fake rendering."),
                  Group("sources.fake", "A fake two-phase source.")):
        registry.add_group(group)
    registry.register(areas.resolve)
    registry.register(capability("ops.make", MakeParams, output_kind="scalar", run=make,
                                 method="{label} was made as {summary.value}."))
    registry.register(capability("ops.meet", LabelParams, output_kind="scalar", run=meet))
    registry.register(capability("ops.hold", LabelParams, output_kind="scalar", run=hold))
    registry.register(capability("ops.ask", LabelParams, output_kind=None, run=ask))
    registry.register(capability("ops.empty", LabelParams, output_kind=None, run=empty))
    registry.register(capability("ops.fail", FailParams, output_kind=None, run=fail))
    registry.register(dataclasses.replace(capability("ops.fail_unsafe", FailParams, output_kind=None, run=fail), retry_safe=False))
    registry.register(capability("render.draw", DrawParams, output_kind=None, run=draw))
    registry.register(capability(
        "sources.fake.fetch", LabelParams, output_kind="scalar", connector=CONNECTOR,
        begin=fetch_begin, complete=fetch_complete,
    ))
    return registry


def spec(steps: list[dict[str, Any]], outputs: dict[str, Any], **over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "fake",
        "version": 1,
        "status": "draft",
        "description": "A workflow for testing the runner.",
        "examples": ["Test the runner."],
        "inputs": {"area": {"type": "area", "ask": "Which area?"}},
        "steps": steps,
        "outputs": outputs,
    }
    raw.update(copy.deepcopy(over))
    return raw


def make(step_id: str, **with_: Any) -> dict[str, Any]:
    extra = {k: with_.pop(k) for k in ("when", "after", "required", "method", "label") if k in with_}
    return {"id": step_id, "use": "ops.make", "with": {"label": step_id, **with_}, **extra}


LINEAR = spec(
    [
        make("one", area="$area", value=2),
        make("two", a="$one"),
        {"id": "figure", "use": "render.draw", "with": {"data": "$two"}},
    ],
    {
        "findings": [
            {"name": "total", "value": "$two.summary.value", "unit": "$two.unit"},
            {"name": "drawn", "value": "$figure.summary.n", "unit": "cells"},
        ],
        "figures": ["$figure"],
        "data": ["$two"],
    },
)


class Fixture:
    """A runner and one user's context on memory stores and fake sources."""

    def __init__(self, tmp: str, registry: CapabilityRegistry, catalogue: WorkflowCatalogue, script: Script | None = None,
                 *, earthdata: Any = None, open_retrieval: Any = None, credentials: dict[str, str] | None = None,
                 stores: Stores | None = None, saver: InMemorySaver | None = None, user: str = "user-a") -> None:
        self.registry = registry
        self.catalogue = catalogue
        self.saver = saver or InMemorySaver()
        self.sleeps: list[float] = []
        self.events = RecordingEvents()
        self.artifacts = MemoryArtifacts()
        self.places = FakePlaces({"nj": NJ, "new jersey": NJ})
        self.stores = stores or memory_stores()
        self.context: RunContext = make_run_context(
            user_id=user, events=self.events, artifacts=self.artifacts, stores=self.stores, clock=FixedClock(TODAY),
            credentials=StaticCredentials({CONNECTOR: SECRET} if credentials is None else credentials),
        )
        self.build = env_factory(
            tmp, earthdata=earthdata or (script.earthdata if script else FakeEarthdata()),
            open_retrieval=open_retrieval, places=self.places,
        )
        self.runner = self.new_runner()

    def new_runner(self) -> WorkflowRunner:
        return WorkflowRunner(
            self.registry, self.catalogue, self.saver,
            retry=RetrySettings(retries=2, backoff=lambda n: float(n), sleep=self.sleeps_into),
        )

    async def sleeps_into(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    async def start(self, inputs: dict[str, Any], name: str = "fake", version: int = 1) -> Result:
        return await self.runner.start(self.context, self.build, name, version, inputs)

    async def env(self, run_id: str) -> RunEnv:
        return await self.build(self.context, run_id)

    async def entries(self, run_id: str) -> list[tuple[str, str | None, str | None, str]]:
        found = await self.stores.traces.entries(self.context.identity.user_id, run_id)
        return [(e.kind, e.capability, e.step, e.status) for e in found]

    async def run_status(self, run_id: str) -> str:
        return (await self.stores.traces.get_run(self.context.identity.user_id, run_id)).status


AREA_ENTRY = ("use", "sources.areas.resolve", "input.area", "done")


class RunnerCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.script = Script()
        self.registry = fake_registry(self.script)

    def fixture(self, *specs: dict[str, Any], **kw: Any) -> Fixture:
        catalogue = load_specs({f"spec{i}": s for i, s in enumerate(specs)}, self.registry, DATASETS)
        return Fixture(self.tmp, self.registry, catalogue, self.script, **kw)


class OrderTests(RunnerCase):
    async def test_a_linear_workflow_writes_one_entry_per_step_in_order(self):
        f = self.fixture(LINEAR)
        result = await f.start({"area": "NJ"})

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(await f.entries(result.trace_id), [
            AREA_ENTRY,
            ("use", "ops.make", "one", "done"),
            ("use", "ops.make", "two", "done"),
            ("use", "render.draw", "figure", "done"),
        ])
        self.assertEqual(await f.run_status(result.trace_id), "done")
        progress = [e for e in f.events.progress_events if e["stage"] == "workflow.fake"]
        self.assertEqual(len(progress), 1)
        self.assertEqual([s["id"] for s in progress[0]["detail"]["steps"]], ["one", "two", "figure"])
        self.assertEqual(progress[0]["detail"]["steps"][0]["label"], "Running ops.make")

    async def test_independent_steps_run_together_and_a_join_waits_for_both(self):
        f = self.fixture(spec(
            [
                {"id": "left", "use": "ops.meet", "with": {"label": "left"}},
                {"id": "right", "use": "ops.meet", "with": {"label": "right"}},
                make("join", a="$left", b="$right"),
            ],
            {"data": ["$join"]},
            inputs={},
        ))
        result = await f.start({})

        self.assertEqual(result.status, "done", result.reason)
        entries = await f.entries(result.trace_id)
        self.assertEqual(sorted(entries[:2]), [("use", "ops.meet", "left", "done"), ("use", "ops.meet", "right", "done")])
        self.assertEqual(entries[2], ("use", "ops.make", "join", "done"))
        joined = self.script.calls[-1][1]
        self.assertTrue(joined["a"].startswith("scalar_") and joined["b"].startswith("scalar_"))

    async def test_a_false_condition_skips_the_step_and_what_reads_it_but_not_what_follows_it(self):
        f = self.fixture(spec(
            [
                make("base"),
                make("maybe", a="$base", when='$mode == "run"'),
                make("reader", a="$maybe"),
                make("follower", after=["maybe"]),
            ],
            {"data": ["$base", {"from": "$reader", "optional": True}, {"from": "$follower", "optional": True}]},
            inputs={"mode": {"type": "choice", "options": ["skip", "run"], "default": "skip"}},
        ))
        result = await f.start({})

        self.assertEqual(result.status, "done", result.reason)
        entries = await f.entries(result.trace_id)
        self.assertEqual(entries[:2], [("use", "ops.make", "base", "done"), ("use", "ops.make", "maybe", "skipped")])
        self.assertEqual(sorted(entries[2:]), [
            ("use", "ops.make", "follower", "done"), ("use", "ops.make", "reader", "skipped"),
        ])
        self.assertEqual(len(result.handles), 2)
        self.assertEqual([label for _, call in self.script.calls for label in [call["label"]]], ["base", "follower"])

        ran = await f.start({"mode": "run"})
        self.assertEqual([e[3] for e in await f.entries(ran.trace_id)], ["done"] * 4)
        self.assertEqual(len(ran.handles), 3)


STOP = spec(
    [
        make("tally", value="$found"),
        {
            "id": "too_few", "stop": "insufficient_evidence", "when": "$tally.summary.value < limits.min_found",
            "reason": "Only {tally.summary.value} were found; at least {limits.min_found} are needed.",
        },
        make("later", a="$tally", after=["too_few"]),
    ],
    {"data": ["$tally", {"from": "$later", "optional": True}]},
    inputs={"found": {"type": "number", "default": 2, "min": 0, "max": 100}},
    limits={"min_found": 5},
)


class StopTests(RunnerCase):
    async def test_a_stop_ends_the_run_with_its_status_and_reason(self):
        f = self.fixture(STOP)
        result = await f.start({})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "Only 2 were found; at least 5 are needed.")
        self.assertEqual(await f.entries(result.trace_id), [
            ("use", "ops.make", "tally", "done"), ("stop", None, "too_few", "done"),
        ])
        stop = (await f.stores.traces.entries("user-a", result.trace_id))[1]
        self.assertEqual(stop.result, {"status": "insufficient_evidence", "reason": result.reason})
        self.assertEqual(await f.run_status(result.trace_id), "insufficient_evidence")
        # A run that ended early keeps what it finished.
        self.assertEqual(len(result.handles), 1)

    async def test_a_stop_whose_condition_is_false_is_skipped(self):
        f = self.fixture(STOP)
        result = await f.start({"found": 9})
        self.assertEqual(result.status, "done")
        self.assertEqual(await f.entries(result.trace_id), [
            ("use", "ops.make", "tally", "done"), ("stop", None, "too_few", "skipped"), ("use", "ops.make", "later", "done"),
        ])


class FailureTests(RunnerCase):
    def failing(self, use: str, required: bool, **with_: Any) -> dict[str, Any]:
        return spec(
            [make("base"), {"id": "extra", "use": use, "with": with_, "required": required}, make("after", a="$base", after=["extra"])],
            {"data": ["$base", {"from": "$after", "optional": True}]},
            inputs={},
        )

    async def test_an_optional_step_that_fails_is_a_gap(self):
        for use, with_ in (("ops.fail", {"category": "bad_input"}), ("ops.empty", {"label": "x"})):
            with self.subTest(use):
                f = self.fixture(self.failing(use, False, **with_))
                result = await f.start({})
                self.assertEqual(result.status, "done")
                [gap] = result.gaps
                self.assertEqual(gap.step, "extra")
                self.assertTrue(gap.reason.startswith("The fake step"))
                self.assertEqual(len(result.handles), 2)

    async def test_a_required_step_that_fails_fails_the_run(self):
        f = self.fixture(self.failing("ops.fail", True, category="unsupported_shape"))
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("failed", "The fake step failed (unsupported_shape)."))
        self.assertEqual(await f.run_status(result.trace_id), "failed")
        entries = await f.entries(result.trace_id)
        self.assertEqual(entries, [("use", "ops.make", "base", "done"), ("use", "ops.fail", "extra", "failed")])
        self.assertEqual(len(result.handles), 1)

    async def test_insufficient_evidence_on_a_required_step_ends_the_run_so(self):
        f = self.fixture(self.failing("ops.empty", True, label="x"))
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("insufficient_evidence", "The fake step found nothing valid."))
        self.assertEqual(await f.run_status(result.trace_id), "insufficient_evidence")

    async def test_a_transient_failure_on_a_step_not_safe_to_repeat_is_not_retried(self):
        f = self.fixture(self.failing("ops.fail_unsafe", True, category="source_unavailable"))
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("failed", "The fake step failed (source_unavailable)."))
        self.assertEqual(f.sleeps, [])


FETCH = spec(
    [
        {"id": "get", "use": "sources.fake.fetch", "with": {"label": "get"}},
        make("use_it", a="$get"),
    ],
    {"data": ["$use_it"]},
    inputs={},
)


class ConnectorAndInputTests(RunnerCase):
    async def test_a_missing_connector_asks_for_it_before_anything_runs(self):
        f = self.fixture(spec(FETCH["steps"], FETCH["outputs"]), credentials={})
        result = await f.start({"area": "NJ"})
        self.assertEqual(result.status, "needs_input")
        self.assertIn("fakeconn connector", result.reason or "")
        self.assertEqual(await f.entries(result.trace_id), [])
        self.assertEqual(self.script.earthdata.calls, [])
        self.assertEqual(f.places.asked, [])
        self.assertEqual(await f.run_status(result.trace_id), "needs_input")

    async def test_missing_inputs_are_asked_for_together(self):
        f = self.fixture(spec(LINEAR["steps"], LINEAR["outputs"], inputs={
            "area": {"type": "area", "ask": "Which area?"},
            "dates": {"type": "date_range", "ask": "Which dates?"},
        }))
        result = await f.start({})
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(result.reason, "The workflow needs more before it can run: Which area? Which dates?")
        assert result.question is not None
        self.assertEqual([i["name"] for i in result.question["inputs"]], ["area", "dates"])
        self.assertEqual(await f.entries(result.trace_id), [])

    async def test_a_bad_input_fails_the_run_naming_it(self):
        f = self.fixture(STOP)
        result = await f.start({"found": 1000})
        self.assertEqual((result.status, result.reason), ("failed", "Input found must be at most 100."))
        self.assertEqual(await f.entries(result.trace_id), [])
        self.assertEqual(await f.run_status(result.trace_id), "failed")

    async def test_area_wording_that_does_not_resolve_ends_the_run_with_that_reason(self):
        f = self.fixture(LINEAR)
        result = await f.start({"area": "Atlantis"})
        self.assertEqual(result.status, "needs_input")
        self.assertTrue((result.reason or "").startswith("'Atlantis' could not be found as a place."))
        self.assertEqual(await f.entries(result.trace_id), [("use", "sources.areas.resolve", "input.area", "waiting")])
        self.assertEqual(self.script.calls, [])

    async def test_an_area_handle_is_used_as_it_is(self):
        f = self.fixture(LINEAR)
        first = await f.start({"area": "NJ"})
        area = self.script.calls[0][1]["area"]
        again = await f.start({"area": area})
        self.assertEqual(again.status, "done")
        self.assertEqual((await f.entries(again.trace_id))[0], ("use", "ops.make", "one", "done"))
        self.assertNotEqual(first.trace_id, again.trace_id)


AREA_FETCH = spec(
    [
        {"id": "get", "use": "sources.fake.fetch", "with": {"label": "get"}},
        make("use_it", a="$get", area="$area"),
    ],
    {"data": ["$use_it"]},
)


class AreaRefusalTests(RunnerCase):
    async def test_an_area_the_host_refuses_ends_the_run_asking_again(self):
        f = self.fixture(AREA_FETCH)
        cases = [
            ("NY + Wakanda", PlaceRefused("member_not_found", {"member": "Wakanda"}),
             "'Wakanda' in 'NY + Wakanda' could not be found as a U.S. state or a country"),
            ("within 3000 miles of NYC", PlaceRefused("distance_too_large", {
                "radius": "3000 miles", "centre": "New York, United States", "cells": "20,979,073", "max_cells": "4,000,000",
            }), "The area within 3000 miles of New York, United States covers about 20,979,073 data cells, above the 4,000,000-cell limit"),
        ]
        for wording, refused, opening in cases:
            with self.subTest(wording=wording):
                f.places.refusals[wording.lower()] = refused
                result = await f.start({"area": wording})
                self.assertEqual(result.status, "needs_input")
                self.assertTrue((result.reason or "").startswith(opening), result.reason)
                assert result.question is not None
                self.assertEqual((result.question["kind"], result.question["input"]), ("text", "area"))
                self.assertEqual(result.question["prompt"], result.reason)
                self.assertEqual(await f.entries(result.trace_id), [("use", "sources.areas.resolve", "input.area", "waiting")])
                self.assertEqual(await f.run_status(result.trace_id), "needs_input")
        self.assertEqual(self.script.earthdata.calls, [])
        self.assertEqual(self.script.calls, [])

    async def test_an_area_that_names_no_place_asks_the_same_way(self):
        f = self.fixture(AREA_FETCH)
        result = await f.start({"area": "Atlantis"})
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertEqual(result.question["input"], "area")
        self.assertEqual(result.question["prompt"], result.reason)
        self.assertEqual(self.script.earthdata.calls, [])


class JitteryTraces:
    """A trace store whose sequence numbers come back after uneven waits, as a database's would."""

    def __init__(self, inner: Any, delays: list[int]) -> None:
        self._inner = inner
        self._delays = delays
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def reserve_seq(self, user_id: str, run_id: str) -> int:
        delay = self._delays[self._calls % len(self._delays)]
        self._calls += 1
        for _ in range(delay):
            await asyncio.sleep(0)
        seq: int = await self._inner.reserve_seq(user_id, run_id)
        return seq


class TraceOrderTests(RunnerCase):
    async def test_steps_ready_together_are_numbered_in_file_order(self):
        steps = [make(name) for name in ("zeta", "alpha", "mu")]
        workflow = spec([*steps, make("last", a="$zeta", b="$alpha", after=["mu"])], {"data": ["$last"]}, inputs={})
        for delays in ([0], [9, 0, 3], [0, 7, 1, 4], [5, 5, 0, 2]):
            with self.subTest(delays=delays):
                stores = memory_stores()
                stores.traces = JitteryTraces(stores.traces, delays)  # type: ignore[assignment]
                f = self.fixture(workflow, stores=stores)
                result = await f.start({})
                self.assertEqual(result.status, "done", result.reason)
                self.assertEqual([e[2] for e in await f.entries(result.trace_id)], ["zeta", "alpha", "mu", "last"])


class ProductTests(RunnerCase):
    async def test_the_approved_products_reach_the_chooser_and_its_product_pins_the_next_step(self):
        f = self.fixture(spec(
            [
                make("a", quantity="$quantity"),
                make("b", quantity="$quantity", product="$a.product"),
            ],
            {"data": ["$a", "$b"]},
            inputs={"quantity": {"type": "quantity", "ask": "Which quantity?"}},
            products={"quantity": ["TEMPO_NO2", "TROPOMI_NO2"]},
        ))
        result = await f.start({"quantity": NO2})
        self.assertEqual(result.status, "done", result.reason)
        (_, a), (_, b) = self.script.calls
        self.assertEqual((a["approved"], a.get("product")), (["TEMPO_NO2", "TROPOMI_NO2"], None))
        self.assertEqual((b["approved"], b["product"]), (["TEMPO_NO2", "TROPOMI_NO2"], "TEMPO_NO2"))

    async def test_a_null_input_reaches_an_optional_parameter_as_none(self):
        f = self.fixture(spec(
            [make("a", product="$pin", area="$area")], {"data": ["$a"]},
            inputs={"pin": {"type": "text", "default": None}, "area": {"type": "area", "default": None}},
        ))
        absent = await f.start({})
        self.assertEqual(absent.status, "done", absent.reason)
        given = await f.start({"pin": "OMI_NO2"})
        self.assertEqual(given.status, "done", given.reason)
        (_, first), (_, second) = self.script.calls
        self.assertEqual((first["product"], first["area"]), (None, None))
        self.assertEqual((second["product"], second["area"]), ("OMI_NO2", None))
        self.assertEqual(f.places.asked, [])

    async def test_an_unknown_quantity_is_bad_input(self):
        f = self.fixture(spec(
            [make("a", quantity="$quantity")], {"data": ["$a"]},
            inputs={"quantity": {"type": "quantity", "ask": "Which quantity?"}}, products={"quantity": ["TEMPO_NO2"]},
        ))
        result = await f.start({"quantity": "unobtainium"})
        self.assertEqual(result.status, "failed")
        self.assertTrue((result.reason or "").startswith("Input quantity must be a quantity"))


class QuestionTests(RunnerCase):
    async def test_a_capability_question_pauses_the_run_and_resume_is_refused(self):
        f = self.fixture(spec(
            [make("base"), {"id": "choose", "use": "ops.ask", "with": {"label": "choose"}}],
            {"data": ["$base"]},
            inputs={},
        ))
        result = await f.start({})
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertEqual(result.question["options"], ["surface", "column"])
        [event] = f.events.questions
        self.assertEqual((event["run_id"], event["step"], event["question"]), (result.trace_id, "choose", result.question))
        self.assertEqual(await f.run_status(result.trace_id), "needs_input")
        self.assertEqual((await f.entries(result.trace_id))[-1], ("use", "ops.ask", "choose", "waiting"))
        with self.assertRaises(AnswerNotSupported):
            await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(await f.run_status(result.trace_id), "needs_input")


class WaitingTests(RunnerCase):
    async def test_a_job_that_outlasts_the_turn_is_saved_and_resumed_by_another_runner(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["running"]
        result = await f.start({})

        self.assertEqual(result.status, "waiting")
        self.assertEqual(result.reason, JOB_WAITING.format(step="get"))
        self.assertEqual(await f.run_status(result.trace_id), "waiting")
        saved = await f.saver.aget_tuple({"configurable": {"thread_id": result.trace_id}})
        assert saved is not None
        self.assertEqual(saved.checkpoint["channel_values"]["jobs"], {"get": {"job": "job_1"}})
        self.assertEqual(await f.entries(result.trace_id), [
            ("use", "sources.fake.fetch", "get", "waiting"), ("use", "sources.fake.fetch", "get", "failed"),
        ])

        f.runner = f.new_runner()
        resumed = await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(resumed.status, "done", resumed.reason)
        self.assertEqual(resumed.trace_id, result.trace_id)
        self.assertEqual(self.script.submissions(), 1)
        self.assertEqual(await f.run_status(result.trace_id), "done")
        self.assertEqual((await f.entries(result.trace_id))[2:], [
            ("use", "sources.fake.fetch", "get", "done"), ("use", "ops.make", "use_it", "done"),
        ])
        with self.assertRaises(RunNotResumable):
            await f.runner.resume(await f.env(result.trace_id))

    async def test_a_job_still_running_after_a_resume_waits_again(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["running", "running"]
        first = await f.start({})
        second = await f.runner.resume(await f.env(first.trace_id))
        self.assertEqual(second.status, "waiting")
        third = await f.runner.resume(await f.env(first.trace_id))
        self.assertEqual(third.status, "done", third.reason)
        self.assertEqual(self.script.submissions(), 1)

    async def test_two_jobs_waiting_at_once_are_both_resumed(self):
        f = self.fixture(spec(
            [
                {"id": "first", "use": "sources.fake.fetch", "with": {"label": "first"}},
                {"id": "second", "use": "sources.fake.fetch", "with": {"label": "second"}},
                make("join", a="$first", b="$second"),
            ],
            {"data": ["$join"]},
            inputs={},
        ))
        self.script.statuses = {"job_1": ["running"], "job_2": ["running", "running"]}
        result = await f.start({})
        self.assertEqual(result.status, "waiting")
        again = await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(again.status, "waiting")
        done = await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(done.status, "done", done.reason)
        self.assertEqual(self.script.submissions(), 2)
        self.assertEqual((await f.entries(result.trace_id))[-1], ("use", "ops.make", "join", "done"))

    async def test_a_job_that_keeps_failing_is_saved_and_resume_begins_it_again(self):
        f = self.fixture(FETCH)
        self.script.statuses = {f"job_{n}": ["failed"] for n in (1, 2, 3)}
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("waiting", SOURCE_WAITING.format(step="get")))
        self.assertEqual(self.script.submissions(), 3)

        resumed = await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(resumed.status, "done", resumed.reason)
        self.assertEqual(self.script.submissions(), 4)
        self.assertEqual(len(set(self.script.submit_keys())), 4)

    async def test_a_transient_failure_is_retried_with_a_new_attempt(self):
        f = self.fixture(FETCH)
        self.script.submit_failures = 1
        result = await f.start({})
        self.assertEqual(result.status, "done", result.reason)
        keys = self.script.submit_keys()
        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(f.sleeps, [1.0])

    async def test_a_job_that_failed_at_the_provider_begins_again_with_a_new_attempt(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["failed"]
        result = await f.start({})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(self.script.submissions(), 2)
        self.assertEqual(len(set(self.script.submit_keys())), 2)

    async def test_retries_exhausted_save_the_run_and_resume_tries_again(self):
        f = self.fixture(FETCH)
        self.script.submit_failures = 3
        result = await f.start({})
        self.assertEqual(result.status, "waiting")
        self.assertEqual(result.reason, SOURCE_WAITING.format(step="get"))
        self.assertEqual(f.sleeps, [1.0, 2.0])
        self.assertEqual(await f.run_status(result.trace_id), "waiting")

        f.runner = f.new_runner()
        resumed = await f.runner.resume(await f.env(result.trace_id))
        self.assertEqual(resumed.status, "done", resumed.reason)
        keys = self.script.submit_keys()
        # Three failed attempts, then a fourth: no attempt number is used twice.
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 4)


class ResultTests(RunnerCase):
    async def test_the_result_envelope(self):
        f = self.fixture(LINEAR)
        result = await f.start({"area": "NJ"})

        self.assertEqual(result.mode, Mode.draft("fake", 1))
        self.assertEqual(result.trace_id[:4], "run_")
        [handle] = result.handles
        self.assertTrue(handle.startswith("scalar_"))
        self.assertEqual(result.charts, ("chart-1",))
        total, drawn = result.findings
        self.assertEqual((total.name, total.value, total.unit, total.source_handle), ("total", 3.0, "ppb", handle))
        # The figure has no handle of its own; the finding points at the step it drew.
        self.assertEqual((drawn.name, drawn.value, drawn.unit, drawn.source_handle), ("drawn", 3, "cells", handle))
        self.assertEqual(result.disclosures, ("'NJ' was resolved to New Jersey.", "Made one.", "Made two."))
        self.assertEqual(result.gaps, ())

        steps = await f.runner.steps(await f.env(result.trace_id))
        self.assertEqual([(s.step, s.status) for s in steps], [("one", "done"), ("two", "done"), ("figure", "done")])
        self.assertEqual(steps[0].method, "one was made as 2.")
        self.assertEqual(steps[2].method, "The fake step ran.")

    async def test_a_spec_method_is_filled_from_the_step_and_the_run(self):
        f = self.fixture(spec(
            [make("one", value=4, method="Step one made {summary.value} from {values.label}; the limit was {limits.top}.")],
            {"data": ["$one"]}, inputs={}, limits={"top": 7},
        ))
        result = await f.start({})
        [step] = await f.runner.steps(await f.env(result.trace_id))
        self.assertEqual(step.method, "Step one made 4 from one; the limit was 7.")

    async def test_gaps_are_listed(self):
        f = self.fixture(spec(
            [make("base"), {"id": "extra", "use": "ops.empty", "with": {"label": "x"}, "required": False}],
            {"data": ["$base"]}, inputs={},
        ))
        result = await f.start({})
        self.assertEqual(result.gaps, (Gap(step="extra", reason="The fake step found nothing valid."),))


class RefusalTests(RunnerCase):
    async def test_an_unknown_workflow_or_version_is_refused_before_anything_starts(self):
        f = self.fixture(LINEAR)
        for name, version in (("nothing", 1), ("fake", 2)):
            with self.subTest(name=name, version=version), self.assertRaises(UnknownWorkflow):
                await f.start({}, name=name, version=version)
        self.assertEqual(await f.stores.traces.list_runs("user-a", "conv-1"), [])

    async def test_a_spec_the_runner_does_not_run_is_refused(self):
        f = self.fixture(spec(
            [make("base"), {"id": "confirm", "ask": {"kind": "text", "question": "Go on?"}}],
            {"data": ["$base"]}, inputs={},
        ))
        with self.assertRaises(NotRunnable) as caught:
            await f.start({})
        self.assertEqual(caught.exception.reason, "step confirm is an ask step, which the runner does not run")
        self.assertEqual(await f.stores.traces.list_runs("user-a", "conv-1"), [])

    async def test_a_finished_run_cannot_be_resumed_or_cancelled(self):
        f = self.fixture(LINEAR)
        result = await f.start({"area": "NJ"})
        env = await f.env(result.trace_id)
        with self.assertRaises(RunNotResumable) as caught:
            await f.runner.resume(env)
        self.assertEqual(caught.exception.status, "done")
        with self.assertRaises(RunNotResumable):
            await f.runner.cancel(env)

    async def test_a_waiting_run_can_be_cancelled(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["running"]
        result = await f.start({})
        run = await f.runner.cancel(await f.env(result.trace_id))
        self.assertEqual(run.status, "cancelled")
        with self.assertRaises(RunNotResumable):
            await f.runner.resume(await f.env(result.trace_id))


class InternalErrorTests(RunnerCase):
    async def test_a_bug_in_a_node_fails_the_run_with_fixed_wording(self):
        f = self.fixture(LINEAR)
        with patch.object(compile_module._StepNodes, "_params", side_effect=RuntimeError("boom " + SECRET)):
            result = await f.start({"area": "NJ"})
        self.assertEqual((result.status, result.reason), ("failed", INTERNAL_REASON))
        self.assertEqual(await f.run_status(result.trace_id), "failed")
        self.assertNotIn(SECRET, result.model_dump_json())


class CompleteRetryTests(RunnerCase):
    async def test_a_hiccup_in_complete_retries_the_same_job(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["hiccup"]
        result = await f.start({})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(self.script.submissions(), 1)
        self.assertEqual(f.sleeps, [1.0])

    async def test_hiccups_past_the_bound_keep_the_job_and_resume_completes_it(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["hiccup"] * 3
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("waiting", SOURCE_WAITING.format(step="get")))
        resumed = await f.new_runner().resume(await f.env(result.trace_id))
        self.assertEqual(resumed.status, "done", resumed.reason)
        self.assertEqual(self.script.submissions(), 1)

    async def test_a_job_the_service_no_longer_holds_begins_again(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["expired"]
        result = await f.start({})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(self.script.submissions(), 2)

    async def test_a_job_submitted_again_is_in_the_checkpoint_before_it_is_waited_on(self):
        f = self.fixture(FETCH)
        self.script.statuses = {"job_1": ["failed"], "job_2": ["running"]}
        seen: list[Any] = []

        async def look(run_id: str, job_id: str) -> None:
            if job_id == "job_2":
                saved = await f.saver.aget_tuple({"configurable": {"thread_id": run_id}})
                assert saved is not None
                seen.append([w[2] for w in saved.pending_writes or [] if w[1] == "__interrupt__"])

        self.script.before_complete = look
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("waiting", JOB_WAITING.format(step="get")))
        [[[interrupt]]] = seen
        self.assertEqual((interrupt.value["kind"], interrupt.value["job"]), ("save", {"job": "job_2"}))
        done = await f.new_runner().resume(await f.env(result.trace_id))
        self.assertEqual(done.status, "done", done.reason)
        self.assertEqual(self.script.submissions(), 2)


class SubmissionCapTests(RunnerCase):
    async def test_a_job_that_dies_every_time_ends_the_run_after_the_cap(self):
        f = self.fixture(FETCH)
        self.script.statuses = {f"job_{n}": ["failed"] for n in range(1, 20)}
        result = await f.start({})
        statuses = [result.status]
        while result.status == "waiting" and len(statuses) < 10:
            result = await f.runner.resume(await f.env(result.trace_id))
            statuses.append(result.status)
        self.assertEqual(statuses, ["waiting", "waiting", "failed"])
        self.assertEqual(result.reason, "The data provider failed the retrieval for step get 6 times, so it was not tried again.")
        self.assertEqual(self.script.submissions(), 6)
        self.assertEqual(await f.run_status(result.trace_id), "failed")
        self.assertEqual((await f.entries(result.trace_id))[-1], ("use", "sources.fake.fetch", "get", "failed"))

    async def test_the_cap_is_injectable(self):
        f = self.fixture(FETCH)
        f.runner = WorkflowRunner(self.registry, f.catalogue, f.saver,
                                  retry=RetrySettings(retries=5, backoff=lambda n: 0.0, sleep=f.sleeps_into, max_submissions=2))
        self.script.statuses = {f"job_{n}": ["failed"] for n in range(1, 20)}
        result = await f.start({})
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.script.submissions(), 2)

    async def test_a_job_the_provider_cancelled_is_not_tried_again(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["cancelled"]
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("failed", "The fake retrieval was cancelled at the data provider."))
        self.assertEqual(self.script.submissions(), 1)
        self.assertEqual(f.sleeps, [])


class EnvelopeOrderTests(RunnerCase):
    async def test_the_trace_status_matches_the_result_when_the_envelope_fails(self):
        f = self.fixture(LINEAR)
        real = runner_module._envelope
        calls = {"n": 0}

        def flaky(*args: Any) -> Result:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("envelope")
            return real(*args)

        with patch.object(runner_module, "_envelope", side_effect=flaky):
            result = await f.start({"area": "NJ"})
        self.assertEqual(result.status, "failed")
        self.assertEqual(await f.run_status(result.trace_id), "failed")


HOLD = spec(
    [{"id": "slow", "use": "ops.hold", "with": {"label": "slow"}}, make("after_slow", a="$slow")],
    {"data": [{"from": "$after_slow", "optional": True}]},
    inputs={},
)


class CancellationTests(RunnerCase):
    async def running(self, f: Fixture) -> tuple[asyncio.Task[Result], str]:
        task = asyncio.create_task(f.start({}))
        await self.script.holding.wait()
        [run] = await f.stores.traces.list_runs("user-a", "conv-1")
        return task, run.id

    async def test_a_run_cancelled_while_it_runs_ends_as_cancelled(self):
        f = self.fixture(HOLD)
        task, run_id = await self.running(f)
        await f.runner.cancel(await f.env(run_id))
        self.script.release.set()
        with self.assertNoLogs("tta_backend.harness.workflows.runner", level="ERROR"):
            result = await task
        self.assertEqual((result.status, result.reason), ("cancelled", "The run was cancelled."))
        self.assertEqual(await f.run_status(run_id), "cancelled")
        # The record closed at the cancel, so the step in flight wrote nothing and none followed it.
        self.assertEqual(await f.entries(run_id), [])

    async def test_a_cancelled_task_finishes_the_run_as_cancelled(self):
        f = self.fixture(HOLD)
        task, run_id = await self.running(f)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await f.run_status(run_id), "cancelled")

    async def test_an_environment_that_cannot_be_built_fails_the_run(self):
        f = self.fixture(HOLD)

        async def broken(context: RunContext, run_id: str) -> RunEnv:
            raise RuntimeError("no environment " + SECRET)

        result = await f.runner.start(f.context, broken, "fake", 1, {})
        self.assertEqual((result.status, result.reason), ("failed", INTERNAL_REASON))
        self.assertEqual(await f.run_status(result.trace_id), "failed")
        self.assertNotIn(SECRET, result.model_dump_json())


class ParallelEndTests(RunnerCase):
    async def test_parallel_required_failures_report_the_step_earlier_in_the_file(self):
        f = self.fixture(spec(
            [
                {"id": "zeta", "use": "ops.fail", "with": {"category": "bad_input"}},
                {"id": "alpha", "use": "ops.fail", "with": {"category": "unsupported_shape"}},
                make("last", after=["zeta", "alpha"]),
            ],
            {"data": [{"from": "$last", "optional": True}]},
            inputs={},
        ))
        result = await f.start({})
        self.assertEqual((result.status, result.reason), ("failed", "The fake step failed (bad_input)."))
        self.assertEqual([e[2] for e in await f.entries(result.trace_id)], ["zeta", "alpha"])


class ExistsTests(RunnerCase):
    async def test_not_exists_runs_as_the_otherwise_branch(self):
        f = self.fixture(spec(
            [
                make("main", when='$mode == "main"'),
                make("fallback", when="not exists($main)"),
                make("reader", a="$main"),
            ],
            {"data": [{"from": "$main", "optional": True}, {"from": "$fallback", "optional": True}]},
            inputs={"mode": {"type": "choice", "options": ["main", "other"], "default": "other"}},
        ))
        other = await f.start({})
        self.assertEqual(sorted(e[2:] for e in await f.entries(other.trace_id)), [
            ("fallback", "done"), ("main", "skipped"), ("reader", "skipped"),
        ])
        main = await f.start({"mode": "main"})
        self.assertEqual(sorted(e[2:] for e in await f.entries(main.trace_id)), [
            ("fallback", "skipped"), ("main", "done"), ("reader", "done"),
        ])


class IsolationTests(RunnerCase):
    async def test_another_user_cannot_resume_cancel_or_read_a_run(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["running"]
        result = await f.start({})
        other = Fixture(self.tmp, self.registry, f.catalogue, self.script, stores=f.stores, saver=f.saver, user="user-b")
        env = await other.env(result.trace_id)
        for action in (other.runner.resume, other.runner.cancel, other.runner.steps):
            with self.subTest(action=action.__name__), self.assertRaises(RunNotFound):
                await action(env)
        self.assertEqual(await f.run_status(result.trace_id), "waiting")


class PlanTests(RunnerCase):
    def test_each_node_has_its_superstep_and_file_rank(self):
        workflow = load_specs({"s": spec(
            [
                {"id": "get", "use": "sources.fake.fetch", "with": {"label": "get"}},
                make("early"),
                make("middle", a="$early"),
                make("join", a="$get", b="$middle"),
            ],
            {"data": ["$join"]},
            inputs={},
        )}, self.registry, DATASETS).get("fake", 1)
        compiled = compile_workflow(workflow, self.registry, checkpointer=InMemorySaver())
        self.assertEqual(dict(compiled.plan), {
            "get.begin": (1, 0), "get": (2, 0), "early": (1, 1), "middle": (2, 2), "join": (3, 3),
        })


class SecretTests(RunnerCase):
    async def test_no_secret_reaches_the_checkpoint_the_trace_or_the_result(self):
        f = self.fixture(FETCH)
        self.script.statuses["job_1"] = ["running"]
        waiting = await f.start({})
        done = await f.runner.resume(await f.env(waiting.trace_id))
        self.assertEqual(done.status, "done")

        saved = repr(f.saver.storage) + repr(f.saver.writes) + repr(f.saver.blobs)
        self.assertIn("job_1", saved)
        recorded = json.dumps(
            [dataclasses.asdict(e) for e in await f.stores.traces.entries("user-a", waiting.trace_id)], default=str
        )
        for text in (saved, recorded, waiting.model_dump_json(), done.model_dump_json()):
            self.assertNotIn(SECRET, text)


# The real capabilities, on the scripted earthdata service.


def three_days(first: str, value: float) -> xr.Dataset:
    days = [f"{first[:8]}{int(first[8:]) + n:02d}" for n in range(3)]
    return xr.concat([granule(day=day, value=value + n * 1e14) for n, day in enumerate(days)], dim="time")


EDL_SECRET = "edl-runner-secret-0b7e"


class FixtureWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.service = FakeService()
        self.service.make_file = lambda day: three_days(day, 2e15 if day < "2024-06-05" else 3e15)
        registry = build_registry()
        catalogue = load_specs(
            {path.name: path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.yaml"))},
            registry, DATASETS,
        )

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.service.files[record.id]

        self.f = Fixture(tmp.name, registry, catalogue, earthdata=self.service.earthdata(),
                         open_retrieval=open_retrieval, credentials={"earthdata": EDL_SECRET})

    async def test_the_map_workflow(self):
        result = await self.f.start(
            {"quantity": NO2, "area": "New Jersey", "dates": {"start": "2024-06-01", "end": "2024-06-03"}}, name="map",
        )
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(await self.f.entries(result.trace_id), [
            AREA_ENTRY,
            ("use", "sources.satellite.get_data", "data", "waiting"),
            ("use", "sources.satellite.get_data", "data", "done"),
            ("use", "render.map", "figure", "done"),
        ])
        self.assertEqual(result.mode, Mode.draft("map", 1))
        self.assertEqual(result.handles, ("obs_1",))
        [chart] = result.charts
        self.assertIn(chart, self.f.artifacts.stored)
        self.assertTrue(result.disclosures[0].startswith("Product: TEMPO_NO2"))
        steps = await self.f.runner.steps(await self.f.env(result.trace_id))
        self.assertTrue((steps[0].method or "").startswith("TEMPO_NO2 was retrieved over New Jersey"))

    async def test_the_period_comparison_workflow(self):
        result = await self.f.start({
            "quantity": NO2, "area": "New Jersey",
            "first": {"start": "2024-06-01", "end": "2024-06-03"},
            "second": {"start": "2024-06-10", "end": "2024-06-12"},
        }, name="period-comparison")
        self.assertEqual(result.status, "done", result.reason)
        entries = await self.f.entries(result.trace_id)
        self.assertEqual(entries[0], AREA_ENTRY)
        done = [e[2] for e in entries if e[3] == "done"]
        self.assertEqual(sorted(done), sorted(["input.area", "a", "b", "mean_a", "mean_b", "diff", "figure", "stat"]))
        # Each step starts only after what it reads has finished.
        first = {step: i for i, (_, _, step, _) in reversed(list(enumerate(entries)))}
        finished = {step: i for i, (_, _, step, status) in enumerate(entries) if status == "done"}
        for step, reads in (("b", "a"), ("mean_a", "a"), ("mean_b", "b"), ("diff", "mean_a"), ("diff", "mean_b"),
                            ("figure", "diff"), ("stat", "diff")):
            self.assertLess(finished[reads], first[step], (step, reads))
        self.assertEqual(len(self.service.jobs), 2)

        steps = {s.step: s for s in await self.f.runner.steps(await self.f.env(result.trace_id))}
        [finding] = result.findings
        self.assertEqual(finding.name, "mean_change")
        self.assertAlmostEqual(float(finding.value) / 1e15, 1.0, places=6)
        assert steps["stat"].result is not None and steps["diff"].result is not None
        self.assertEqual(finding.source_handle, steps["stat"].result["handle"])
        self.assertEqual(finding.unit, steps["diff"].result.get("unit"))
        self.assertEqual(result.handles, tuple(steps[s].result["handle"] for s in ("diff", "mean_a", "mean_b")))
        self.assertEqual(len(result.charts), 1)
        self.assertEqual(steps["b"].method, "The second period uses TEMPO_NO2, the same product as the first.")
        self.assertEqual(steps["diff"].method, "The difference is the second period's mean minus the first's, cell by cell.")
        self.assertEqual(steps["a"].method, (
            "TEMPO_NO2 was retrieved for New Jersey over the first period and screened with its quality flag."
        ))
        self.assertEqual(await self.f.run_status(result.trace_id), "done")


if __name__ == "__main__":
    unittest.main()
