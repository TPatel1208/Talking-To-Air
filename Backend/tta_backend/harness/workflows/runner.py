"""Running a workflow spec: checks before it starts, the graph, pausing, resuming and the result.

A host builds one ``WorkflowRunner`` with the capability registry, the
workflow catalogue and a LangGraph checkpointer (``AsyncPostgresSaver`` in
production, after its ``setup()``; ``InMemorySaver`` in tests), and optionally
the knowledge notes agent steps are given, then:

- ``start(context, build_env, name, version, inputs)`` refuses an unknown or
  not-runnable spec before anything is written, starts the trace run with the
  spec's mode, builds the run's environment with ``build_env(context, run_id)``
  and runs it;
- ``resume(env)`` continues a run saved as ``waiting``, on the spec version it
  started with, from the step that paused; ``env`` is built for that run id;
- ``cancel(env)`` and ``steps(env)`` act on a run the same way.

The checkpoint thread is the run id. The checkpointer is not scoped by user,
so every method first reads the run from the user's own trace store, which
refuses another user's run as missing.

Every outcome is finished in the trace with the matching status and returned
as a ``Result``. The result is saved to the results store first, so a reader
never sees a finished run without one; a paused run's result is replaced when
it finishes. A run already finished elsewhere (a cancel) keeps the result its
finisher saved. A result that cannot be saved is logged as an error and still
returned, and the run still finished: the caller keeps the run's outcome.
``outputs.interpretation`` fills the result's ``interpretation`` from an agent
step's answer.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.results import Finding, Gap, Mode, Result, ResultStatus
from tta_backend.harness.run_context import RunContext, RunEnv
from tta_backend.harness.trace import Run, RunClosed, RunStatus, new_run_id
from tta_backend.harness.workflows.compile import (
    CANCELLED,
    CANCELLED_REASON,
    DONE,
    PAUSE_JOB,
    PAUSE_QUESTION,
    PAUSE_SAVE,
    CompiledWorkflow,
    NotRunnable,
    RetrySettings,
    StateScope,
    StepContext,
    compile_workflow,
)
from tta_backend.harness.workflows.expressions import EvaluationError, Ref, evaluate_value, render_value
from tta_backend.harness.workflows.inputs import InputProblem, resolve_inputs
from tta_backend.harness.workflows.loader import WorkflowCatalogue
from tta_backend.harness.workflows.spec import UseStep, WorkflowSpec

logger = logging.getLogger(__name__)

EnvFactory = Callable[[RunContext, str], Awaitable[RunEnv]]

INTERNAL_REASON = "The workflow stopped on an internal error."
FAILED: Literal["failed"] = "failed"
# Save points resumed in one call before the run is saved as waiting instead.
MAX_SAVE_POINTS = 100
JOB_WAITING = "Step {step} is waiting for its retrieval to finish; resume the run to collect it."
SOURCE_WAITING = (
    "The data source for step {step} is temporarily unavailable; the run is saved and can be resumed to try again."
)
QUESTION_REASON = "Step {step} needs an answer before the workflow can continue."
# Run statuses that are final; needs_input and waiting are pauses.
_TERMINAL = frozenset({"done", "failed", "cancelled", "insufficient_evidence", "incomplete"})


def _connect_reason(connectors: list[str]) -> str:
    names = " and ".join(connectors)
    noun = "connector" if len(connectors) == 1 else "connectors"
    return f"This workflow needs the {names} {noun}. Connect it in Settings -> Connectors, then run it again."


class RunNotResumable(Exception):
    """A run in a status ``resume`` (or ``cancel``) does not act on."""

    def __init__(self, run_id: str, status: str) -> None:
        super().__init__(f"Run {run_id} is {status} and cannot be resumed.")
        self.run_id = run_id
        self.status = status


class AnswerNotSupported(RunNotResumable):
    """A run paused on a question for the user; answering it into the paused run is not supported."""

    def __init__(self, run_id: str) -> None:
        Exception.__init__(
            self,
            f"Run {run_id} is waiting for an answer. Resuming a run with an answer is not supported; "
            "run the workflow again with the answer as an input.",
        )
        self.run_id = run_id
        self.status = "needs_input"


class NotAWorkflowRun(RunNotResumable):
    def __init__(self, run_id: str) -> None:
        Exception.__init__(self, f"Run {run_id} is not a workflow run.")
        self.run_id = run_id
        self.status = "not_a_workflow"


@dataclass(frozen=True)
class StepRecord:
    """One step's outcome as the checkpoint holds it, for the methods export."""

    step: str
    kind: str
    status: str
    method: str | None
    result: dict[str, Any] | None
    reason: str | None


def _config(run_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": run_id}}


def _known_quantities() -> frozenset[str]:
    return frozenset(cfg.quantity for cfg in load_registry().values() if cfg.quantity)


class WorkflowRunner:
    """Runs specs from ``catalogue`` on ``capabilities``, saving each run under ``checkpointer``.

    ``quantities`` is the set of canonical quantities a ``quantity`` input may
    take; it defaults to those the dataset registry records. ``notes`` are the
    knowledge notes agent steps are given; with none, they are given no notes.
    """

    def __init__(
        self,
        capabilities: CapabilityRegistry,
        catalogue: WorkflowCatalogue,
        checkpointer: BaseCheckpointSaver[Any],
        *,
        retry: RetrySettings | None = None,
        quantities: Collection[str] | None = None,
        notes: NoteCatalogue | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._catalogue = catalogue
        self._checkpointer = checkpointer
        self._retry = retry or RetrySettings()
        self._quantities = frozenset(quantities) if quantities is not None else _known_quantities()
        self._notes = notes
        self._compiled: dict[tuple[str, int], CompiledWorkflow] = {}

    # Specs

    def spec(self, name: str, version: int) -> WorkflowSpec:
        """The runnable spec; raises UnknownWorkflow or NotRunnable."""
        spec = self._catalogue.get(name, version)
        if spec.not_runnable_reason is not None:
            raise NotRunnable(name, version, spec.not_runnable_reason)
        return spec

    def _workflow(self, spec: WorkflowSpec) -> CompiledWorkflow:
        key = (spec.name, spec.version)
        if key not in self._compiled:
            self._compiled[key] = compile_workflow(spec, self._capabilities, checkpointer=self._checkpointer)
        return self._compiled[key]

    async def _own_run(self, env: RunEnv) -> tuple[Run, WorkflowSpec]:
        """The run, read as env's user (another user's raises RunNotFound), and its spec."""
        run = await env.context.storage.traces.get_run(env.context.identity.user_id, env.run_id)
        workflow = run.mode.workflow
        if workflow is None:
            raise NotAWorkflowRun(env.run_id)
        return run, self.spec(workflow.name, workflow.version)

    # Starting

    async def start(
        self, context: RunContext, build_env: EnvFactory, name: str, version: int, inputs: Mapping[str, Any]
    ) -> Result:
        """Start and run a workflow. Raises UnknownWorkflow or NotRunnable before anything is written."""
        spec = self.spec(name, version)
        workflow = self._workflow(spec)
        run_id = new_run_id()
        identity = context.identity
        await context.storage.traces.start_run(Run(
            id=run_id, user_id=identity.user_id, conversation_id=identity.conversation_id,
            mode=spec.mode(), started=context.clock.now(),
        ))

        async def body() -> Result:
            env = await build_env(context, run_id)
            return await self._run(env, workflow, inputs)

        return await self._guarded(context, run_id, spec, body)

    async def _run(self, env: RunEnv, workflow: CompiledWorkflow, raw: Mapping[str, Any]) -> Result:
        spec = workflow.spec
        self._announce(env, spec)
        missing = [name for name in sorted(spec.connectors(self._capabilities))
                   if not await env.context.credentials.secret(name)]
        if missing:
            return await self._finish(env, spec, "needs_input", reason=_connect_reason(missing))
        resolved = await resolve_inputs(env, spec, raw, self._capabilities, quantities=self._quantities)
        if isinstance(resolved, InputProblem):
            return await self._finish(env, spec, resolved.status, reason=resolved.reason, question=resolved.question)
        state: dict[str, Any] = {
            "workflow": {"name": spec.name, "version": spec.version},
            "inputs": resolved.values,
            "outcomes": {},
            "jobs": {},
            "attempts": {},
            "gaps": [],
            "end": None,
            "disclosures": list(resolved.disclosures),
            "input_names": dict(resolved.names),
        }
        return await self._drive(env, workflow, state, {})

    def _announce(self, env: RunEnv, spec: WorkflowSpec) -> None:
        steps = []
        for step_id in spec.order():
            step = spec.step(step_id)
            label = step.label
            if label is None and isinstance(step, UseStep):
                label = self._capabilities.get(step.use).label
            steps.append({"id": step_id, "label": label or step_id})
        env.context.events.progress(
            f"Running workflow {spec.name} v{spec.version}",
            stage=f"workflow.{spec.name}",
            detail={"workflow": spec.name, "version": spec.version, "mode": spec.mode().label(), "steps": steps},
        )

    # Resuming, cancelling, reading

    async def resume(self, env: RunEnv) -> Result:
        """Continue the run ``env`` was built for, saved as ``waiting``.

        Raises RunNotFound for a run that is not this user's, AnswerNotSupported
        for one waiting on an answer, and RunNotResumable for any other status.
        """
        run, spec = await self._own_run(env)
        if run.status == "needs_input":
            raise AnswerNotSupported(run.id)
        if run.status != "waiting":
            raise RunNotResumable(run.id, run.status)
        workflow = self._workflow(spec)
        config = _config(run.id)
        snapshot = await workflow.graph.aget_state(config)
        pending = _pending(snapshot.tasks)
        if not pending:
            raise RunNotResumable(run.id, run.status)
        await env.context.storage.traces.resume_run(env.context.identity.user_id, run.id)
        return await self._guarded(env.context, run.id, spec, lambda: self._drive(env, workflow, *_resumption(pending)))

    async def cancel(self, env: RunEnv) -> Run:
        """Finish a run that has not finished as ``cancelled``. Its provider jobs are not cancelled.

        The saved result becomes a cancelled one, keeping what the run had
        reported. It is saved only once the cancel has finished the run, so a
        run that finished first (RunClosed) keeps its own result.
        """
        run, spec = await self._own_run(env)
        if run.status not in ("running", "needs_input", "waiting"):
            raise RunNotResumable(run.id, run.status)
        cancelled = await _cancelled_result(env.context, spec, run.id)
        finished = await env.context.storage.traces.finish_run(
            env.context.identity.user_id, run.id, "cancelled", env.context.clock.now()
        )
        await _save(env.context, cancelled)
        return finished

    async def steps(self, env: RunEnv) -> tuple[StepRecord, ...]:
        """Each step's recorded outcome and method sentence, in run order; empty before any step ran."""
        _, spec = await self._own_run(env)
        snapshot = await self._workflow(spec).graph.aget_state(_config(env.run_id))
        values: Mapping[str, Any] = snapshot.values or {}
        outcomes: Mapping[str, Mapping[str, Any]] = values.get("outcomes") or {}
        return tuple(
            StepRecord(
                step=step_id,
                kind=str(outcome.get("kind")),
                status=str(outcome.get("status")),
                method=outcome.get("method"),
                result=outcome.get("result"),
                reason=outcome.get("reason"),
            )
            for step_id in spec.order()
            if (outcome := outcomes.get(step_id)) is not None
        )

    # Driving the graph and finishing

    async def _drive(
        self, env: RunEnv, workflow: CompiledWorkflow, given: Any, resumed: Mapping[str, Mapping[str, Any]]
    ) -> Result:
        """Run the graph until it finishes or pauses. A pause that is only save
        points is resumed at once: the jobs they hold are in the checkpoint."""
        for _ in range(MAX_SAVE_POINTS):
            context = StepContext(
                env=env, retry=self._retry, resumed=resumed, turns=workflow.turns(resumed), notes=self._notes
            )
            output = await workflow.graph.ainvoke(
                given, _config(env.run_id), context=context, durability="sync", version="v2"
            )
            values: Mapping[str, Any] = output.value if isinstance(output.value, Mapping) else {}
            interrupts = tuple(output.interrupts)
            kinds = {i.value.get("kind") for i in interrupts if isinstance(i.value, dict)}
            if values.get("end") or not interrupts or kinds != {PAUSE_SAVE}:
                break
            snapshot = await workflow.graph.aget_state(_config(env.run_id))
            given, resumed = _resumption(_pending(snapshot.tasks))
        return await self._from_state(env, workflow.spec, values, interrupts)

    async def _from_state(
        self, env: RunEnv, spec: WorkflowSpec, values: Mapping[str, Any], interrupts: tuple[Interrupt, ...]
    ) -> Result:
        end = values.get("end")
        paused = [i.value for i in interrupts if isinstance(i.value, dict)]
        status: ResultStatus
        reason: str | None = None
        question: dict[str, Any] | None = None
        if end:
            status = cast(ResultStatus, end["status"])
            reason = str(end["reason"])
        elif paused:
            asking = next((p for p in paused if p.get("kind") == PAUSE_QUESTION), None)
            if asking is not None:
                status, question = "needs_input", dict(asking["question"])
                reason = QUESTION_REASON.format(step=asking["step"])
            else:
                status = "waiting"
                first = next((p for p in paused if p.get("kind") != PAUSE_SAVE), paused[0])
                wording = JOB_WAITING if first.get("kind") == PAUSE_JOB else SOURCE_WAITING
                reason = wording.format(step=first["step"])
        else:
            status = "done"
        return await self._finish(env, spec, status, reason=reason, question=question, values=values)

    async def _finish(
        self,
        env: RunEnv,
        spec: WorkflowSpec,
        status: ResultStatus,
        *,
        reason: str | None = None,
        question: dict[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
    ) -> Result:
        values = values or {}
        # Built first, so the trace never says a status the result does not carry.
        result = _envelope(spec, env.run_id, status, reason, question, values)
        traces, user_id = env.context.storage.traces, env.context.identity.user_id
        # A run finished elsewhere keeps the result its finisher saved.
        if (await traces.get_run(user_id, env.run_id)).status not in _TERMINAL:
            await _save(env.context, result)
        try:
            await traces.finish_run(user_id, env.run_id, cast(RunStatus, result.status), env.context.clock.now())
        except RunClosed as closed:
            if closed.status != CANCELLED:
                raise
            # Cancelled after the save above, which may have landed after the
            # cancel's own: saved again so the stored result says cancelled, as
            # the trace does. What the run finished is still reported.
            cancelled = _envelope(spec, env.run_id, CANCELLED, CANCELLED_REASON, None, values)
            await _save(env.context, cancelled)
            return cancelled
        return result

    async def _guarded(
        self, context: RunContext, run_id: str, spec: WorkflowSpec, body: Callable[[], Awaitable[Result]]
    ) -> Result:
        """Run ``body`` so the run is always finished.

        A cancelled task finishes the run as cancelled and the cancellation
        goes on. A run cancelled from elsewhere is a cancelled result. Anything
        else is an internal error: the run is finished as failed, with fixed wording.
        """
        try:
            return await body()
        except asyncio.CancelledError:
            await _close(context, _envelope(spec, run_id, CANCELLED, CANCELLED_REASON, None, {}))
            raise
        except RunClosed as closed:
            if closed.status == CANCELLED:
                cancelled = await _cancelled_result(context, spec, run_id)
                await _save(context, cancelled)
                return cancelled
            failure: Exception = closed
        except Exception as exc:
            failure = exc
        # The type name only: an exception's text can carry values or secrets.
        logger.error("workflow %s v%s hit an internal error in run %s: %s", spec.name, spec.version, run_id,
                     type(failure).__name__)
        failed = _envelope(spec, run_id, FAILED, INTERNAL_REASON, None, {})
        await _close(context, failed)
        return failed


async def _save(context: RunContext, result: Result) -> None:
    """Save the run's result. A failure is logged, not raised, so it never hides the run's own outcome."""
    try:
        await context.storage.results.put(context.identity.user_id, result)
    except Exception as exc:
        # The type name only: an exception's text can carry values or secrets.
        logger.error("the result of workflow run %s could not be saved: %s", result.trace_id, type(exc).__name__)


async def _cancelled_result(context: RunContext, spec: WorkflowSpec, run_id: str) -> Result:
    """The run's saved result, marked cancelled; a bare cancelled result when none was saved."""
    try:
        saved = await context.storage.results.get(context.identity.user_id, run_id)
    except Exception:
        # Missing, or a store that cannot be read: the cancel still goes ahead.
        return _envelope(spec, run_id, CANCELLED, CANCELLED_REASON, None, {})
    return Result.model_validate(
        {**saved.model_dump(), "status": CANCELLED, "reason": CANCELLED_REASON, "question": None}
    )


async def _close(context: RunContext, result: Result) -> None:
    """Save ``result`` and finish its run with its status, unless the run is finished already;
    shielded from the cancellation that may have led here."""
    traces, user_id = context.storage.traces, context.identity.user_id

    async def closing() -> None:
        # A finished run keeps the result it was finished with.
        if (await traces.get_run(user_id, result.trace_id)).status in _TERMINAL:
            return
        await _save(context, result)
        await traces.finish_run(user_id, result.trace_id, cast(RunStatus, result.status), context.clock.now())

    try:
        await asyncio.shield(closing())
    except RunClosed:
        pass


def _resumption(pending: list[Interrupt]) -> tuple[Command[str], dict[str, Mapping[str, Any]]]:
    """The command that resumes every paused node, and what each paused with."""
    resumed: dict[str, Mapping[str, Any]] = {
        str(i.value["node"]): i.value for i in pending if isinstance(i.value, dict) and "node" in i.value
    }
    return Command(resume={i.id: {"resumed": True} for i in pending}), resumed


def _pending(tasks: tuple[Any, ...]) -> list[Interrupt]:
    """The interrupts of tasks still paused; a task that has since finished keeps a stale one."""
    found: list[Interrupt] = []
    for task in tasks:
        if task.interrupts and not task.result:
            found.extend(task.interrupts)
    return found


# The result envelope


def _done_result(outcome: Mapping[str, Any] | None) -> CapabilityResult | None:
    """A finished ``use`` step's result; None for any other step."""
    if not outcome or outcome.get("kind") != "use" or outcome.get("status") != DONE or outcome.get("result") is None:
        return None
    return CapabilityResult.model_validate(outcome["result"])


def _any_result(outcome: Mapping[str, Any] | None) -> CapabilityResult | None:
    if not outcome or outcome.get("kind") != "use" or outcome.get("result") is None:
        return None
    try:
        return CapabilityResult.model_validate(outcome["result"])
    except ValidationError:
        return None


def _source_handle(spec: WorkflowSpec, outcomes: Mapping[str, Mapping[str, Any]], step_id: str) -> str | None:
    """The step's handle, else that of the nearest step it reads that has one."""
    seen: set[str] = set()
    frontier = [step_id]
    order = spec.order()
    while frontier:
        nearest: list[str] = []
        for current in frontier:
            result = _done_result(outcomes.get(current))
            if result is not None and result.handle is not None:
                return result.handle
            for ref in spec.compiled(current).references():
                if isinstance(ref, Ref) and ref.root == "step" and ref.name not in seen:
                    seen.add(ref.name)
                    nearest.append(ref.name)
        frontier = sorted(nearest, key=order.index)
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else render_value(value)


def _findings(spec: WorkflowSpec, values: Mapping[str, Any]) -> tuple[Finding, ...]:
    outcomes: Mapping[str, Mapping[str, Any]] = values.get("outcomes") or {}
    scope = StateScope(spec, values)
    found: list[Finding] = []
    for finding in spec.compiled_outputs.findings:
        ref = finding.value
        if not isinstance(ref, Ref) or ref.root != "step":
            continue
        source = _done_result(outcomes.get(ref.name))
        if source is None:
            continue
        try:
            value = evaluate_value(ref, scope)
            unit = None if finding.unit is None else evaluate_value(finding.unit, scope)
            where = None if finding.where is None else evaluate_value(finding.where, scope)
            when = None if finding.when is None else evaluate_value(finding.when, scope)
        except EvaluationError:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            continue
        handle = _source_handle(spec, outcomes, ref.name)
        if handle is None:
            continue
        region = _text(where) or _text(source.values.get("area_name"))
        period = _text(when) or _text(source.values.get("period_text"))
        try:
            found.append(Finding(
                name=finding.name, value=value, unit=_text(unit), region=region, period=period, source_handle=handle,
            ))
        except ValidationError:
            continue
    return tuple(found)


def _disclosures(spec: WorkflowSpec, values: Mapping[str, Any]) -> tuple[str, ...]:
    outcomes: Mapping[str, Mapping[str, Any]] = values.get("outcomes") or {}
    lines: list[str] = [str(line) for line in values.get("disclosures") or ()]
    for step_id in spec.order():
        result = _any_result(outcomes.get(step_id))
        if result is None:
            continue
        lines.extend(result.disclosures)
        rule = result.values.get("rule")
        if result.product and isinstance(rule, str) and not any(result.product in line for line in result.disclosures):
            lines.append(f"Product: {result.product}, chosen by rule {rule}.")
    return tuple(dict.fromkeys(lines))


def _interpretation(spec: WorkflowSpec, values: Mapping[str, Any]) -> str | None:
    """The text ``outputs.interpretation`` reads, when its agent step answered."""
    entry = spec.compiled_outputs.interpretation
    if entry is None:
        return None
    try:
        value = evaluate_value(entry.ref, StateScope(spec, values))
    except EvaluationError:
        return None
    return value if isinstance(value, str) and value.strip() else None


def _envelope(
    spec: WorkflowSpec,
    run_id: str,
    status: ResultStatus,
    reason: str | None,
    question: dict[str, Any] | None,
    values: Mapping[str, Any],
) -> Result:
    outcomes: Mapping[str, Mapping[str, Any]] = values.get("outcomes") or {}
    outputs = spec.compiled_outputs
    handles: list[str] = []
    for entry in outputs.data:
        result = _done_result(outcomes.get(entry.ref.name)) if isinstance(entry.ref, Ref) else None
        if result is not None and result.handle is not None:
            handles.append(result.handle)
    charts: list[str] = []
    for entry in outputs.figures:
        result = _done_result(outcomes.get(entry.ref.name)) if isinstance(entry.ref, Ref) else None
        if result is not None:
            charts.extend(result.charts)
    gaps = tuple(Gap(step=g["step"], reason=g["reason"]) for g in values.get("gaps") or ())
    mode: Mode = spec.mode()
    return Result(
        status=status,
        mode=mode,
        findings=_findings(spec, values),
        handles=tuple(dict.fromkeys(handles)),
        charts=tuple(dict.fromkeys(charts)),
        disclosures=_disclosures(spec, values),
        gaps=gaps,
        trace_id=run_id,
        reason=None if status == "done" else reason,
        question=question if status == "needs_input" else None,
        interpretation=_interpretation(spec, values),
    )
