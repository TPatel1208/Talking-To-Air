"""Running a workflow spec: checks before it starts, the graph, pausing, resuming and the result.

A host builds one ``WorkflowRunner`` with the capability registry, the
workflow catalogue and a LangGraph checkpointer (``AsyncPostgresSaver`` in
production, after its ``setup()``; ``InMemorySaver`` in tests), and optionally
the knowledge notes agent steps are given, then:

- ``start(context, build_env, name, version, inputs)`` refuses an unknown or
  not-runnable spec before anything is written, starts the trace run with the
  spec's mode, builds the run's environment with ``build_env(context, run_id)``
  and runs it;
- ``resume(env, answer, step=...)`` continues a paused run, on the spec
  version it started with, from the step that paused; ``env`` is built for
  that run id. A ``waiting`` run takes no answer. A ``needs_input`` run takes
  the answer to the question it waits on (the first in file order, or the
  one of ``step``), checked in code before anything is resumed; the answer
  is the resume value of that question's interrupt only;
- ``cancel(env)`` and ``steps(env)`` act on a run the same way.

``expire_paused`` cancels a conversation's runs paused for longer than
``EXPIRY``, and ``cancel_paused`` cancels one paused run of any mode.

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
import datetime
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
from tta_backend.harness.run_context import RunContext, RunEnv, report_finished
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
from tta_backend.harness.workflows.answers import AnswerRejected, check_answer
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
_PAUSED = frozenset({"needs_input", "waiting"})

# A paused run left this long, from when it last paused, is cancelled at the next turn.
EXPIRY = datetime.timedelta(days=30)
EXPIRED_REASON = "The run waited more than 30 days to be continued, so it expired and was cancelled."

ANSWER_REQUIRED = "Run {run_id} is waiting for an answer to step {step}; continue it with the answer."
NOT_ASKING_DATA = "Run {run_id} is waiting for its data, not for an answer; continue it without one."
NOT_ASKING_INPUTS = (
    "Run {run_id} stopped before any step ran, so it takes no answer; run the workflow again with the answer "
    "as an input."
)
NOT_ASKING_STEP = "Run {run_id} is not waiting on step {step}; it is waiting on step {asked}."
NOT_ASKING_DONE = "Run {run_id} is {status}, so it is not waiting for an answer."


def connect_reason(connectors: list[str]) -> str:
    """The fixed wording of a run that did not start because ``connectors`` are not connected."""
    names = " and ".join(connectors)
    noun = "connector" if len(connectors) == 1 else "connectors"
    return f"This workflow needs the {names} {noun}. Connect it in Settings -> Connectors, then run it again."


class RunNotResumable(Exception):
    """A run in a status ``resume`` (or ``cancel``) does not act on."""

    def __init__(self, run_id: str, status: str) -> None:
        super().__init__(f"Run {run_id} is {status} and cannot be resumed.")
        self.run_id = run_id
        self.status = status


class AnswerNotTaken(Exception):
    """An answer, or the lack of one, the run does not take; nothing was resumed.
    ``reason`` is fixed wording for the user or the model."""

    def __init__(self, run_id: str, step: str | None, reason: str) -> None:
        super().__init__(reason)
        self.run_id = run_id
        self.step = step
        self.reason = reason


class BadAnswer(AnswerNotTaken):
    """An answer that does not fit the question the run waits on."""


class NotAsking(AnswerNotTaken):
    """An answer for a run that is not waiting on that question."""


class AnswerRequired(AnswerNotTaken):
    """A run waiting on a question, resumed without an answer."""


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
            return await self._finish(env, spec, "needs_input", reason=connect_reason(missing))
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
        """The run's first progress line: the workflow, its mode and its step ids in order, with each step's label."""
        labels: dict[str, str] = {}
        for step_id in spec.order():
            step = spec.step(step_id)
            label = step.label
            if label is None and isinstance(step, UseStep):
                label = self._capabilities.get(step.use).label
            labels[step_id] = label or step_id
        env.context.events.progress(
            f"Running workflow {spec.name} v{spec.version}",
            stage=f"workflow.{spec.name}",
            detail={
                "workflow": spec.name, "version": spec.version, "mode": spec.mode().view(),
                "steps": list(labels), "labels": labels,
            },
            run_id=env.run_id,
        )

    # Resuming, cancelling, reading

    async def resume(self, env: RunEnv, answer: Any = None, *, step: str | None = None) -> Result:
        """Continue the paused run ``env`` was built for.

        A ``waiting`` run is resumed without an answer. A ``needs_input`` run
        is resumed with ``answer`` to the question it waits on: the one of
        ``step`` when given, else the first in file order. The answer is
        checked before anything is resumed and goes to that question only;
        any other pause of the run is resumed as for a waiting run, so a
        question not answered is asked again.

        Raises RunNotFound for a run that is not this user's, AnswerNotTaken
        (BadAnswer, NotAsking, AnswerRequired) for an answer the run does not
        take, and RunNotResumable for a run that is not paused.
        """
        run, spec = await self._own_run(env)
        if run.status not in _PAUSED:
            if answer is not None:
                raise NotAsking(run.id, step, NOT_ASKING_DONE.format(run_id=run.id, status=run.status))
            raise RunNotResumable(run.id, run.status)
        workflow = self._workflow(spec)
        snapshot = await workflow.graph.aget_state(_config(run.id))
        pending = _pending(snapshot.tasks)
        answered: tuple[Interrupt, Any] | None = None
        if run.status == "needs_input":
            answered = _answering(spec, run.id, pending, answer, step)
        elif answer is not None:
            raise NotAsking(run.id, step, NOT_ASKING_DATA.format(run_id=run.id))
        if not pending:
            raise RunNotResumable(run.id, run.status)
        try:
            await env.context.storage.traces.resume_run(env.context.identity.user_id, run.id)
        except RunClosed as closed:
            raise RunNotResumable(run.id, closed.status) from None
        command, resumed = _resumption(pending, answered)
        return await self._guarded(env.context, run.id, spec, lambda: self._drive(env, workflow, command, resumed))

    async def cancel(self, env: RunEnv, *, reason: str = CANCELLED_REASON) -> Run:
        """Finish a run that has not finished as ``cancelled``. Its provider jobs are not cancelled.

        The saved result becomes a cancelled one with ``reason``, keeping what
        the run had reported. It is saved only once the cancel has finished
        the run, so a run that finished first (RunClosed) keeps its own result.
        """
        run, spec = await self._own_run(env)
        if run.status not in ("running", "needs_input", "waiting"):
            raise RunNotResumable(run.id, run.status)
        cancelled = await _cancelled_result(env.context, spec, run.id, reason)
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
            # A capability question that cannot be answered into the run ends it with that question.
            asked = end.get("question")
            question = dict(asked) if isinstance(asked, dict) and status == "needs_input" else None
        elif paused:
            order = spec.step_ids
            questions = [p for p in paused if p.get("kind") == PAUSE_QUESTION]
            asking = min(questions, key=lambda p: _rank(order, p.get("step")), default=None)
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
        """Run ``body`` so the run is always finished, and say so with the
        run's ``finished`` progress line.

        A cancelled task finishes the run as cancelled and the cancellation
        goes on. A run cancelled from elsewhere is a cancelled result. Anything
        else is an internal error: the run is finished as failed, with fixed wording.
        """
        result = await self._finished(context, run_id, spec, body)
        report_finished(context.events, run_id, result.status)
        return result

    async def _finished(
        self, context: RunContext, run_id: str, spec: WorkflowSpec, body: Callable[[], Awaitable[Result]]
    ) -> Result:
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


async def _cancelled_result(
    context: RunContext, spec: WorkflowSpec, run_id: str, reason: str = CANCELLED_REASON
) -> Result:
    """The run's saved result, marked cancelled; a bare cancelled result when none was saved."""
    try:
        saved = await context.storage.results.get(context.identity.user_id, run_id)
    except Exception:
        # Missing, or a store that cannot be read: the cancel still goes ahead.
        return _envelope(spec, run_id, CANCELLED, reason, None, {})
    return _marked_cancelled(saved, reason)


def _marked_cancelled(saved: Result, reason: str) -> Result:
    return Result.model_validate({**saved.model_dump(), "status": CANCELLED, "reason": reason, "question": None})


async def cancel_paused(
    context: RunContext, runner: WorkflowRunner, build_env: EnvFactory, run: Run, *, reason: str = CANCELLED_REASON,
) -> Result:
    """Finish ``run``, paused in ``needs_input`` or ``waiting``, as ``cancelled`` with ``reason``,
    and return its saved result. Provider jobs are not cancelled.

    A workflow run is cancelled through the runner; an exploration, or a run
    whose spec is no longer loaded, is finished here, its saved result marked
    cancelled. Raises RunNotResumable for a run that is not paused and
    RunClosed when it finished meanwhile.
    """
    if run.status not in _PAUSED:
        raise RunNotResumable(run.id, run.status)
    user_id = context.identity.user_id
    workflow = run.mode.workflow
    if workflow is not None and (workflow.name, workflow.version) in runner._catalogue:
        await runner.cancel(await build_env(context, run.id), reason=reason)
        return await context.storage.results.get(user_id, run.id)
    try:
        saved = await context.storage.results.get(user_id, run.id)
        cancelled = _marked_cancelled(saved, reason)
    except Exception:
        cancelled = Result(status=CANCELLED, mode=run.mode, trace_id=run.id, reason=reason)
    await context.storage.traces.finish_run(user_id, run.id, CANCELLED, context.clock.now())
    await _save(context, cancelled)
    return cancelled


def paused_since(run: Run) -> datetime.datetime:
    """When a paused run last paused: the trace sets ``ended`` at each pause and clears it on resume."""
    return run.ended or run.started


async def expire_paused(
    context: RunContext, runner: WorkflowRunner, build_env: EnvFactory, *, after: datetime.timedelta = EXPIRY,
) -> tuple[str, ...]:
    """Cancel, with the expired reason, the conversation's runs paused for longer than ``after``,
    timed from when each last paused; the ids cancelled. A run that cannot be
    cancelled is logged and left, so the turn goes on."""
    traces, identity = context.storage.traces, context.identity
    cutoff = context.clock.now() - after
    expired: list[str] = []
    for run in await traces.list_runs(identity.user_id, identity.conversation_id):
        if run.status not in _PAUSED or paused_since(run) > cutoff:
            continue
        try:
            await cancel_paused(context, runner, build_env, run, reason=EXPIRED_REASON)
        except (RunClosed, RunNotResumable):
            continue
        except Exception as exc:
            # The type name only: an exception's text can carry values or secrets.
            logger.error("paused run %s could not be expired: %s", run.id, type(exc).__name__)
            continue
        expired.append(run.id)
    return tuple(expired)


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


def _resumption(
    pending: list[Interrupt], answered: tuple[Interrupt, Any] | None = None
) -> tuple[Command[str], dict[str, Mapping[str, Any]]]:
    """The command that resumes every paused node, and what each paused with.

    With ``answered`` (a question's interrupt and its checked answer), that
    interrupt alone is resumed with the answer, which its node also finds in
    what it paused with.
    """
    resumed: dict[str, Mapping[str, Any]] = {
        str(i.value["node"]): i.value for i in pending if isinstance(i.value, dict) and "node" in i.value
    }
    values: dict[str, Any] = {i.id: {"resumed": True} for i in pending}
    if answered is not None:
        target, answer = answered
        values[target.id] = {"answer": answer}
        if isinstance(target.value, dict) and "node" in target.value:
            resumed[str(target.value["node"])] = {**target.value, "answer": answer}
    return Command(resume=values), resumed


def _rank(order: tuple[str, ...], step: Any) -> int:
    return order.index(step) if step in order else len(order)


def _answering(
    spec: WorkflowSpec, run_id: str, pending: list[Interrupt], answer: Any, step: str | None
) -> tuple[Interrupt, Any]:
    """The question interrupt ``answer`` is for, and the answer checked against it."""
    questions = sorted(
        (i for i in pending if isinstance(i.value, dict) and i.value.get("kind") == PAUSE_QUESTION),
        key=lambda i: _rank(spec.step_ids, i.value.get("step")),
    )
    if not questions:
        raise NotAsking(run_id, step, NOT_ASKING_INPUTS.format(run_id=run_id))
    first = str(questions[0].value.get("step"))
    if answer is None:
        raise AnswerRequired(run_id, first, ANSWER_REQUIRED.format(run_id=run_id, step=first))
    target = questions[0] if step is None else next((q for q in questions if q.value.get("step") == step), None)
    if target is None:
        raise NotAsking(run_id, step, NOT_ASKING_STEP.format(run_id=run_id, step=step, asked=first))
    asked = str(target.value.get("step"))
    try:
        checked = check_answer(target.value.get("question") or {}, answer)
    except AnswerRejected as rejected:
        raise BadAnswer(run_id, asked, rejected.reason) from None
    return target, checked


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
