"""Running exploration tasks: each its own trace run and specialist, concurrently, each returning a ``Result``.

For one task:

1. A trace run is started with mode exploratory and its environment built
   with ``build_env(context, run_id)``.
2. One ``explore`` entry wraps the task; every capability call nests under it.
3. Before any model call, the given handles are checked and each place is
   resolved through ``sources.areas.resolve``. A failure there ends the task
   without calling the model, and so does a question, except in a
   ``find-data`` task: it needs no area to search, so the place goes to the
   model as text with a note that it could not be resolved.
4. A specialist is built for the bundle, less the capabilities that need a
   connector whose service is down (the result discloses each such source),
   and run. A question from a tool ends
   the task ``needs_input``; a retrieval still running ends it ``waiting``;
   a spent budget ends it ``incomplete``; models that all fail end it ``failed``.
5. The answer is checked, re-prompted once and salvaged as ``answer`` says;
   the rest of the result comes from what the calls did.

A task that fails never fails another.

Each task's result is saved to the results store before its run is finished,
so a reader never sees a finished run without one. A result that cannot be
saved is logged as an error and still returned, and the run still finished.
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphBubbleUp

from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.registry import CapabilityRegistry, Group
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import DEFAULT_BACKOFF, Backoff, ModelChoice, Usage
from tta_backend.harness.results import Finding, Mode, Result, ResultStatus
from tta_backend.harness.run_context import RunContext, RunEnv, report_finished, source_name
from tta_backend.harness.specialists.answer import Answer, AnswerGuard, accepted, salvage, to_finding
from tta_backend.harness.specialists.builder import build_specialist
from tta_backend.harness.specialists.bundles import get_bundle
from tta_backend.harness.specialists.prompt import notes_in_play, system_prompt, task_message
from tta_backend.harness.specialists.state import TaskState, scalar_value
from tta_backend.harness.specialists.task import ExploreLimits, ExploreTask
from tta_backend.harness.specialists.tools import Caller, TaskStop, capability_tools
from tta_backend.harness.trace import (
    EXPLORE_ENTRY_STATUS,
    FailureCategory,
    InvalidTrace,
    Run,
    RunClosed,
    StepFailure,
    new_run_id,
)

logger = logging.getLogger(__name__)

EnvFactory = Callable[[RunContext, str], Awaitable[RunEnv]]

RESOLVE = "sources.areas.resolve"
# The stage of a task's progress lines that name its objective and bundle: first
# queued, then again when it starts.
EXPLORE_STAGE = "explore"
# Bundles whose tasks carry on with a place that could not be resolved.
PLACES_AS_TEXT = frozenset({"find-data"})
UNRESOLVED_DISCLOSURE = "'{place}' could not be resolved to an area, so it was not used as one."
UNREACHABLE_DISCLOSURE = "{source} is unavailable right now, so this exploration ran without it."
# The name the model failure wordings are filled with.
TASK_NAME = "this exploration"
# The entry's model when the task never had one.
NO_MODEL = "none"

MODEL_BUDGET = (
    "The exploration used all {limit} of its model calls before it finished; what it found so far is kept."
)
TOOL_BUDGET = (
    "The exploration reached its limit of {limit} tool calls before it finished; what it found so far is kept."
)
UNKNOWN_HANDLE = "This exploration was given a handle that does not exist."
INPUTS_UNRECORDABLE = "This exploration's inputs are too large to record."
INTERNAL_REASON = "The exploration stopped on an internal error."
CANCELLED_REASON = "The exploration was cancelled."
NO_SUMMARY = "The specialist reported findings but no summary of them."


@dataclass
class _Outcome:
    """How a task ended, before the result is built from it and the task's state."""

    status: ResultStatus
    reason: str | None = None
    question: dict[str, Any] | None = None
    category: FailureCategory | None = None
    answer: Answer | None = None
    usage: Usage | None = None
    model: str = NO_MODEL
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @classmethod
    def failed(cls, category: FailureCategory, reason: str, **kw: Any) -> _Outcome:
        return cls(status="failed", reason=reason, category=category, **kw)


class _TaskFailed(StepFailure):
    """Carries a failed outcome past the recorder, which writes the failed explore entry."""


def _entry_result(outcome: _Outcome) -> dict[str, Any]:
    usage = outcome.usage
    return {
        "status": outcome.status,
        "model": (usage.model_name if usage and usage.model_name else outcome.model),
        "model_calls": usage.model_calls if usage else 0,
        "tool_calls": usage.tool_calls if usage else 0,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "cache_read_tokens": usage.cache_read_tokens if usage else 0,
        "cache_creation_tokens": usage.cache_creation_tokens if usage else 0,
    }


def with_wait(env: RunEnv, limits: ExploreLimits) -> RunEnv:
    """The environment with its retrieval wait shortened to ``retrieval_wait_seconds``, when that is shorter."""
    wait = limits.retrieval_wait_seconds
    if wait is None or wait >= env.limits.retrieval_timeout_seconds:
        return env
    poll_max = min(env.limits.poll_max_seconds, wait)
    shorter = dataclasses.replace(
        env.limits,
        retrieval_timeout_seconds=wait,
        poll_max_seconds=poll_max,
        poll_min_seconds=min(env.limits.poll_min_seconds, poll_max),
    )
    return dataclasses.replace(env, limits=shorter)


def _scalar_findings(state: TaskState) -> tuple[Finding, ...]:
    """A finding for each scalar the task's calls made, for a task stopped before it answered."""
    found: list[Finding] = []
    for record in state.minted_scalars():
        value = scalar_value(record)
        if value is None:
            continue
        finding = to_finding(record, record.descriptor.summary, value, record.descriptor.unit)
        if finding is not None:
            found.append(finding)
    return tuple(found)


class _Task:
    """One task's run: everything between the started trace run and its outcome."""

    def __init__(
        self,
        env: RunEnv,
        task: ExploreTask,
        registry: CapabilityRegistry,
        notes: NoteCatalogue,
        limits: ExploreLimits,
        backoff: Backoff,
        instruments: tuple[str, ...] = (),
    ) -> None:
        self.env = env
        self.task = task
        # Instruments the user's message names; the task's satellite calls must use one of them.
        self.instruments = instruments
        self.registry = registry
        self.notes = notes
        self.limits = limits
        self.backoff = backoff
        self.state = TaskState()
        # Places that could not be resolved, kept as the user's wording.
        self.unresolved: list[str] = []

    async def run(self) -> _Outcome:
        """The task's outcome, with its explore entry recorded."""
        params = {"objective": self.task.objective, "bundle": self.task.bundle, "inputs": self.task.inputs()}
        self.env.context.events.progress(
            f"Exploring: {self.task.objective}", stage=EXPLORE_STAGE,
            detail={"objective": self.task.objective, "bundle": self.task.bundle}, run_id=self.env.run_id,
        )
        outcome: _Outcome | None = None
        entered = False
        try:
            async with self.env.recorder.step(kind="explore", params=params) as scope:
                entered = True
                outcome = await self._work(scope.seq)
                result = _entry_result(outcome)
                if outcome.status == "failed":
                    assert outcome.category is not None and outcome.reason is not None
                    raise _TaskFailed(outcome.category, outcome.reason, result=result)
                scope.status = EXPLORE_ENTRY_STATUS[outcome.status]
                scope.result = result
                scope.reason = outcome.reason
        except _TaskFailed:
            assert outcome is not None
            return outcome
        except InvalidTrace:
            # Refused before the entry was written: the inputs cannot be recorded.
            if not entered:
                return _Outcome.failed(FailureCategory.BAD_INPUT, INPUTS_UNRECORDABLE)
            raise
        return outcome

    async def _work(self, seq: int) -> _Outcome:
        try:
            choice = self.env.model_for("explore")
        except StepFailure as missing:
            return _Outcome.failed(missing.category, missing.reason)
        caller = Caller(self.env, self.state, self.notes, seq, instruments=self.instruments)
        stopped = await self._prepare(caller, choice)
        if stopped is not None:
            return stopped
        return await self._explore(caller, choice)

    async def _prepare(self, caller: Caller, choice: ModelChoice) -> _Outcome | None:
        """Check the given handles and resolve each place; an outcome when that ends the task."""
        user = self.env.context.identity.user_id
        for handle in self.task.handles:
            try:
                self.state.add_input(await self.env.resolver.get(user, handle))
            except HandleError:
                return _Outcome.failed(FailureCategory.BAD_INPUT, UNKNOWN_HANDLE, model=choice.name)
        resolve = self.registry.get(RESOLVE) if self.task.places else None
        for place in self.task.places:
            assert resolve is not None
            try:
                called = await caller.call(resolve, {"place": place})
            except TaskStop as stop:
                if stop.status == "needs_input" and self.task.bundle in PLACES_AS_TEXT:
                    self.unresolved.append(place)
                    self.state.disclose(UNRESOLVED_DISCLOSURE.format(place=place))
                    continue
                return _Outcome(status=stop.status, reason=stop.reason, question=stop.question, model=choice.name)
            except StepFailure as failure:
                return _Outcome.failed(failure.category, failure.reason, model=choice.name)
            if called.result.handle is not None:
                record = self.state.record(called.result.handle)
                assert record is not None
                self.state.add_input(record)
        return None

    def _offered(self) -> tuple[tuple[Capability[Any], ...], tuple[Group, ...]]:
        """The bundle's capabilities and their groups, without those that need a
        connector whose service is down; each such source is disclosed once."""
        everything = get_bundle(self.task.bundle).capabilities(self.registry)
        reachable = self.env.context.reachable
        down = sorted({
            c.connector for c in everything
            if c.connector is not None and c.connector_required and not reachable(c.connector)
        })
        for connector in down:
            self.state.disclose(UNREACHABLE_DISCLOSURE.format(source=source_name(connector)))
        offered = tuple(c for c in everything if not (c.connector_required and c.connector in down))
        groups = tuple(self.registry.group(name) for name in sorted({c.group for c in offered}))
        return offered, groups

    async def _explore(self, caller: Caller, choice: ModelChoice) -> _Outcome:
        inputs = self.state.inputs()
        start_notes = notes_in_play(self.notes, inputs)
        self.state.shown_notes.update(note.id for note in start_notes)
        guard = AnswerGuard(self.state)
        offered, groups = self._offered()
        specialist = build_specialist(
            choice,
            capability_tools(offered, caller),
            system_prompt(groups, start_notes),
            guard,
            self.limits,
            backoff=self.backoff,
        )
        today = self.env.context.clock.now().astimezone(datetime.UTC).date()
        message = HumanMessage(task_message(self.task, today, inputs, self.unresolved))

        def outcome(status: ResultStatus, **kw: Any) -> _Outcome:
            return _Outcome(status=status, usage=specialist.meter.usage(), model=choice.name, **kw)

        try:
            await specialist.agent.ainvoke({"messages": [message]})
        except TaskStop as stop:
            return outcome(stop.status, reason=stop.reason, question=stop.question)
        except ModelCallLimitExceededError:
            return self._incomplete(outcome, guard, MODEL_BUDGET.format(limit=self.limits.model_calls))
        except ToolCallLimitExceededError:
            return self._incomplete(outcome, guard, TOOL_BUDGET.format(limit=self.limits.tool_calls))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if exc is not guard.last_model_error:
                raise
            # The type name only: provider errors can carry request details.
            logger.warning("every model failed in exploration run %s: %s", self.env.run_id, type(exc).__name__)
            category, wording = failures.MODEL_FAILED
            return outcome("failed", category=category, reason=wording.format(name=TASK_NAME))
        answer = accepted(guard.attempts) or salvage(guard.attempts, self.state)
        if answer is None:
            category, wording = failures.MODEL_ANSWER_INVALID
            return outcome("failed", category=category, reason=wording.format(name=TASK_NAME))
        return outcome("done", answer=answer)

    def _incomplete(self, outcome: Callable[..., _Outcome], guard: AnswerGuard, reason: str) -> _Outcome:
        answer = salvage(guard.attempts, self.state)
        findings = answer.findings if answer is not None and answer.findings else _scalar_findings(self.state)
        return outcome("incomplete", reason=reason, answer=answer, findings=findings)


def _result(run_id: str, outcome: _Outcome, state: TaskState) -> Result:
    answer = outcome.answer
    findings: tuple[Finding, ...] = ()
    interpretation: str | None = None
    followups: tuple[str, ...] = ()
    if outcome.status in ("done", "incomplete"):
        findings = outcome.findings or (answer.findings if answer is not None else ())
        if answer is not None:
            interpretation = answer.summary or (NO_SUMMARY if answer.findings else None)
            followups = answer.followups
    return Result(
        status=outcome.status,
        mode=Mode.exploratory(),
        findings=findings,
        handles=state.handles(),
        charts=state.charts(),
        disclosures=state.disclosures(),
        trace_id=run_id,
        reason=None if outcome.status == "done" else outcome.reason,
        question=outcome.question if outcome.status == "needs_input" else None,
        interpretation=interpretation,
        followups=followups,
    )


async def _save(context: RunContext, result: Result) -> None:
    """Save the task's result. A failure is logged, not raised, so it never hides the task's own outcome."""
    try:
        await context.storage.results.put(context.identity.user_id, result)
    except Exception as exc:
        # The type name only: an exception's text can carry values or secrets.
        logger.error("the result of exploration run %s could not be saved: %s", result.trace_id, type(exc).__name__)


async def _close(context: RunContext, result: Result) -> None:
    """Save ``result``, then finish its run with its status. Shielded, so a
    cancellation arriving now still leaves both done. Raises RunClosed when
    the run was finished already."""

    async def closing() -> None:
        await _save(context, result)
        await context.storage.traces.finish_run(
            context.identity.user_id, result.trace_id, result.status, context.clock.now()
        )

    await asyncio.shield(closing())


async def _one(
    context: RunContext,
    build_env: EnvFactory,
    task: ExploreTask,
    registry: CapabilityRegistry,
    notes: NoteCatalogue,
    limits: ExploreLimits,
    backoff: Backoff,
    run_id: str,
    instruments: tuple[str, ...] = (),
) -> Result:
    identity = context.identity
    await context.storage.traces.start_run(Run(
        id=run_id, user_id=identity.user_id, conversation_id=identity.conversation_id,
        mode=Mode.exploratory(), started=context.clock.now(),
    ))
    state = TaskState()
    try:
        env = with_wait(await build_env(context, run_id), limits)
        work = _Task(env, task, registry, notes, limits, backoff, instruments)
        state = work.state
        outcome = await work.run()
    except asyncio.CancelledError:
        try:
            await _close(context, _result(run_id, _Outcome(status="cancelled", reason=CANCELLED_REASON), state))
        except RunClosed:
            pass
        raise
    except Exception as exc:
        # The type name only: an exception's text can carry values or secrets.
        logger.error("exploration run %s hit an internal error: %s", run_id, type(exc).__name__)
        outcome = _Outcome.failed(FailureCategory.INTERNAL_ERROR, INTERNAL_REASON)
    result = _result(run_id, outcome, state)
    try:
        await _close(context, result)
    except RunClosed as closed:
        if closed.status != "cancelled":
            raise
        cancelled = _result(run_id, _Outcome(status="cancelled", reason=CANCELLED_REASON), state)
        await _save(context, cancelled)
        return cancelled
    return result


async def _guarded(
    context: RunContext,
    build_env: EnvFactory,
    task: ExploreTask,
    registry: CapabilityRegistry,
    notes: NoteCatalogue,
    limits: ExploreLimits,
    backoff: Backoff,
    run_id: str,
    instruments: tuple[str, ...] = (),
) -> Result:
    """One task's result whatever happens, so a task that fails never fails
    another; the task's ``finished`` progress line goes out as it ends."""
    try:
        result = await _one(context, build_env, task, registry, notes, limits, backoff, run_id, instruments)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Reached only when the trace store itself failed; the run may be left unfinished.
        logger.error("exploration run %s could not be recorded: %s", run_id, type(exc).__name__)
        result = _result(run_id, _Outcome.failed(FailureCategory.INTERNAL_ERROR, INTERNAL_REASON), TaskState())
        await _save(context, result)
    report_finished(context.events, run_id, result.status)
    return result


async def explore(
    context: RunContext,
    build_env: EnvFactory,
    tasks: Sequence[ExploreTask],
    *,
    registry: CapabilityRegistry,
    notes: NoteCatalogue,
    limits: ExploreLimits | None = None,
    backoff: Backoff = DEFAULT_BACKOFF,
    max_concurrent: int = 4,
    instruments: tuple[str, ...] = (),
) -> list[Result]:
    """Run ``tasks`` concurrently, at most ``max_concurrent`` at once, each as
    its own exploratory run; one result per task, in task order.

    ``build_env(context, run_id)`` builds each task's environment for a run
    already started. ``backoff`` is the wait between model retries.
    ``instruments`` are those the user's message names: each task's
    satellite data calls must use one of them.
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent is at least 1")
    limits = limits or ExploreLimits()
    slots = asyncio.Semaphore(max_concurrent)
    # Allocated up front so a task still waiting for a slot can be shown under its own run id.
    run_ids = [new_run_id() for _ in tasks]
    for task, run_id in zip(tasks, run_ids, strict=True):
        context.events.progress(
            f"Waiting to explore: {task.objective}", stage=EXPLORE_STAGE,
            detail={"objective": task.objective, "bundle": task.bundle, "queued": True}, run_id=run_id,
        )

    async def bounded(task: ExploreTask, run_id: str) -> Result:
        async with slots:
            return await _guarded(context, build_env, task, registry, notes, limits, backoff, run_id, instruments)

    return list(await asyncio.gather(*(bounded(task, run_id) for task, run_id in zip(tasks, run_ids, strict=True))))
