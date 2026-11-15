"""A workflow spec compiled to a LangGraph graph: one node per step, edges from the references.

A ``use`` step whose capability is two-phase is two nodes, ``<id>.begin`` and
``<id>``: the first submits and writes the job reference into state, the
second completes from it, so the reference is in a checkpoint before any
waiting. Every other step is one node named by its id.

State holds handles and small JSON only (see ``WorkflowState``). The run's
environment, retry settings and what a resumed node needs travel in the
graph's context (``StepContext``), never in state.

Each node, in order: does nothing when the run has already ended; is skipped
(one ``skipped`` entry) when a step it reads was skipped or failed as optional,
or when its ``when`` is false; builds its parameters from ``with`` plus
``approved``; and calls the capability through the invoke path with
``step=<id>``. A failure becomes state, so a parallel sibling's work is kept.
A run found cancelled takes no more work. Steps that run in the same superstep
are numbered in the trace in file order (``TraceTurns``), whatever order their
writes complete in.

Pauses are LangGraph interrupts whose value says why (``PAUSE_*``):

- ``question``: the capability came back with a question for the user;
- ``job``: a two-phase job was still running when ``complete`` stopped waiting;
- ``retry``: a transient failure outlasted the bounded retries;
- ``save``: a job submitted again inside the complete node, held in the
  checkpoint before it is waited on; the runner continues past it at once.

A paused node runs again from its start on resume, so the interrupt value
carries what it needs to continue (the job reference, the next attempt); the
runner passes those values back through ``StepContext.resumed``. A node that
pauses again first replays the resume values its task already holds, so each
pause raises a fresh interrupt.

Bounded retries happen inside the node rather than through LangGraph's
``RetryPolicy``: a retry after a job that failed at the provider must go back
through ``begin`` with a new attempt, which spans the step's two nodes.
"""
from __future__ import annotations

import asyncio
import operator
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, NoReturn, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.failures import InputEvicted, JobFailed, JobStillRunning
from tta_backend.harness.capabilities.invoke import begin, complete, invoke
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.result import CapabilityResult, JobRef
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.trace import (
    MAX_REASON_CHARS,
    EntryKind,
    FailureCategory,
    RunClosed,
    StepFailure,
    StepRecorder,
    StepScope,
)
from tta_backend.harness.workflows.expressions import (
    EvaluationError,
    MissingValue,
    StepUnavailable,
    evaluate_condition,
    evaluate_value,
    render_template,
)
from tta_backend.harness.workflows.loader import approved_parameter
from tta_backend.harness.workflows.spec import StopStep, UseStep, WorkflowSpec

APPROVED = "approved"
BEGIN_SUFFIX = ".begin"

# Why a node paused, as the ``kind`` of its interrupt value.
PAUSE_QUESTION = "question"
PAUSE_JOB = "job"
PAUSE_RETRY = "retry"
# A save point: a job submitted again inside the complete node, held in the
# checkpoint before it is waited on. The runner resumes past it at once.
PAUSE_SAVE = "save"

# Outcome statuses a step can have in state.
DONE = "done"
SKIPPED = "skipped"
FAILED_OPTIONAL = "failed_optional"
FAILED = "failed"
CANCELLED: Literal["cancelled"] = "cancelled"
CANCELLED_REASON = "The run was cancelled."
INSUFFICIENT = "insufficient_evidence"
STOPPED = "stopped"
# A step that reads one of these is skipped.
NOT_PRODUCED = frozenset({SKIPPED, FAILED_OPTIONAL})


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


def _first(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    # Siblings in one superstep may both end the run; the one earlier in the file stands.
    if not left:
        return right
    if not right:
        return left
    return right if right.get("rank", 0) < left.get("rank", 0) else left


class WorkflowState(TypedDict, total=False):
    """A run's graph state; plain JSON only.

    ``outcomes`` maps a step id to its outcome: ``status`` (one of the
    statuses above), ``kind`` (use or stop), the recorded capability
    ``result``, the ``method`` sentence, and ``reason`` and ``category`` for a
    step that did not finish. ``end`` is set by the step that ended the run
    early: ``status``, ``reason``, ``step``, ``category`` and the step's
    ``rank`` in the file; when siblings both end it, the earlier step stands.
    A run found cancelled ends with status ``cancelled``.
    """

    workflow: dict[str, Any]
    inputs: dict[str, Any]
    # What resolving the inputs disclosed, such as how area wording was read.
    disclosures: list[str]
    # An area input's resolved name, shown in text in place of its handle id.
    input_names: dict[str, str]
    outcomes: Annotated[dict[str, dict[str, Any]], _merge]
    jobs: Annotated[dict[str, dict[str, Any]], _merge]
    attempts: Annotated[dict[str, int], _merge]
    gaps: Annotated[list[dict[str, str]], operator.add]
    end: Annotated[dict[str, Any] | None, _first]


def default_backoff(retry: int) -> float:
    """Seconds before the given retry (1, 2, ...)."""
    return float(2 ** (retry - 1))


@dataclass(frozen=True)
class RetrySettings:
    """How a transient failure on a retry-safe step is retried before the run is saved to retry later."""

    retries: int = 2
    backoff: Callable[[int], float] = default_backoff
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    # Submissions of a two-phase step over the whole run, across every resume.
    max_submissions: int = 6

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries is zero or more")
        if self.max_submissions < 1:
            raise ValueError("max_submissions is at least 1")


class TraceTurns:
    """Makes the nodes of one superstep take their trace numbers in file order.

    LangGraph starts a superstep's nodes together, and each takes its entry's
    number at its first write, which with a database store is a round trip, so
    the order would otherwise follow whichever write returned first.

    ``plan`` gives each node its superstep (its depth in the graph: one more
    than the deepest node it waits for) and its rank (its step's position in the
    file). Every node runs once per run, so a superstep's nodes are known from
    the graph alone; on a resume the first superstep holds only the nodes that
    paused, given as ``resumed``. Before its first write a node waits until
    every lower-ranked node of its superstep has written or finished.
    """

    def __init__(self, plan: Mapping[str, tuple[int, int]] | None = None, resumed: Collection[str] = ()) -> None:
        self._plan = dict(plan or {})
        self._pending: dict[int, set[int]] = {}
        for level, rank in self._plan.values():
            self._pending.setdefault(level, set()).add(rank)
        again = [self._plan[node] for node in resumed if node in self._plan]
        for level in {level for level, _ in again}:
            self._pending[level] = {rank for at, rank in again if at == level}
        self._changed = asyncio.Event()

    async def wait(self, node: str) -> None:
        if node not in self._plan:
            return
        level, rank = self._plan[node]
        while any(other < rank for other in self._pending.get(level, ())):
            await self._changed.wait()

    def release(self, node: str) -> None:
        if node not in self._plan:
            return
        level, rank = self._plan[node]
        pending = self._pending.get(level)
        if pending is None or rank not in pending:
            return
        pending.discard(rank)
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()


class _TurnRecorder(StepRecorder):
    """A node's recorder: its first entry waits for the node's turn, and taking
    the entry's number ends the turn."""

    def __init__(self, inner: StepRecorder, turns: TraceTurns, node: str) -> None:
        self._inner = inner
        self._turns = turns
        self._node = node

    @asynccontextmanager
    async def step(self, **entry: Any) -> AsyncIterator[StepScope]:
        await self._turns.wait(self._node)
        try:
            async with self._inner.step(**entry) as scope:
                self._turns.release(self._node)
                yield scope
        finally:
            self._turns.release(self._node)


@dataclass(frozen=True)
class StepContext:
    """What every node is handed besides state. ``resumed`` maps a node name to the
    interrupt value it paused with, for a node running again on resume; ``turns``
    orders the trace entries of one superstep."""

    env: RunEnv
    retry: RetrySettings = field(default_factory=RetrySettings)
    resumed: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    turns: TraceTurns = field(default_factory=TraceTurns)


class ResultView:
    """A recorded capability result as a reference reads it; a bare ``$step`` is its handle."""

    def __init__(self, result: CapabilityResult) -> None:
        self.result = result

    def read(self, path: str) -> Any:
        return self.result.read(path)

    def value(self) -> Any:
        return self.result.handle


class StateScope:
    """The run's inputs, finished steps and limits, for evaluating a spec's expressions.

    With ``text``, for filling a method or reason, an area input reads as its
    resolved name rather than its handle id.
    """

    def __init__(self, spec: WorkflowSpec, state: Mapping[str, Any], *, text: bool = False) -> None:
        self._limits = spec.limits
        self._inputs: Mapping[str, Any] = state.get("inputs") or {}
        self._names: Mapping[str, str] = (state.get("input_names") or {}) if text else {}
        self._outcomes: Mapping[str, Mapping[str, Any]] = state.get("outcomes") or {}
        self._views: dict[str, ResultView | None] = {}

    def input(self, name: str) -> Any:
        if name in self._names:
            return self._names[name]
        return self._inputs[name]

    def step(self, step_id: str) -> ResultView | None:
        if step_id not in self._views:
            outcome = self._outcomes.get(step_id)
            recorded = outcome.get("result") if outcome and outcome.get("status") == DONE else None
            self._views[step_id] = None if recorded is None else ResultView(CapabilityResult.model_validate(recorded))
        return self._views[step_id]

    def limit(self, name: str) -> Any:
        return self._limits[name]


def _pause(value: dict[str, Any]) -> NoReturn:
    """Interrupt the node. Resume values its task already holds are replayed
    first, so a node pausing again after a resume raises a fresh interrupt."""
    while True:
        interrupt(value)


def _evaluation_reason(step_id: str, error: EvaluationError) -> str:
    if isinstance(error, StepUnavailable):
        return f"Step {step_id} reads step {error.step_id}, which has no result."
    if isinstance(error, MissingValue):
        return f"Step {step_id} reads {error.reference}, which has no value."
    return f"Step {step_id} could not use a value it reads, because it has the wrong type."


def entry_node(spec: WorkflowSpec, capabilities: CapabilityRegistry, step_id: str) -> str:
    """The node a step starts in: ``<id>.begin`` for a two-phase capability, else ``<id>``."""
    step = spec.step(step_id)
    if isinstance(step, UseStep) and capabilities.get(step.use).two_phase:
        return step_id + BEGIN_SUFFIX
    return step_id


class _StepNodes:
    """The node functions for one step."""

    def __init__(self, spec: WorkflowSpec, capabilities: CapabilityRegistry, step_id: str) -> None:
        self.spec = spec
        self.id = step_id
        self.step = spec.step(step_id)
        self.compiled = spec.compiled(step_id)
        self.capability: Capability[Any] | None = None
        self.approved: Any = None
        if isinstance(self.step, UseStep):
            self.capability = capabilities.get(self.step.use)
            self.approved = approved_parameter(spec, step_id, capabilities)
        self.entry_kind: EntryKind = "stop" if isinstance(self.step, StopStep) else "use"
        self.rank = spec.step_ids.index(step_id)
        # The steps it reads, which skip it when they were skipped. A pure
        # ``after`` is ordering only, and so is a step named only inside
        # ``exists(...)``, so that ``when: not exists($x)`` can run when x did not.
        self.reads = spec.reads(step_id)

    @property
    def capability_name(self) -> str | None:
        return None if self.capability is None else self.capability.name

    def _cap(self) -> Capability[Any]:
        assert self.capability is not None
        return self.capability

    # What every node checks first

    async def _record(self, env: RunEnv, *, status: str, result: Any = None, failure: StepFailure | None = None) -> None:
        try:
            async with env.recorder.step(kind=self.entry_kind, capability=self.capability_name, step=self.id) as scope:
                if failure is not None:
                    raise failure
                scope.status = "skipped" if status == SKIPPED else "done"
                scope.result = result
        except StepFailure:
            if failure is None:
                raise

    async def _skip(self, env: RunEnv) -> dict[str, Any]:
        await self._record(env, status=SKIPPED)
        return {"outcomes": {self.id: {"status": SKIPPED, "kind": self.entry_kind}}}

    async def _evaluation_failure(self, env: RunEnv, error: EvaluationError) -> dict[str, Any]:
        failure = StepFailure(FailureCategory.INTERNAL_ERROR, _evaluation_reason(self.id, error))
        await self._record(env, status=FAILED, failure=failure)
        return self._failed(failure)

    def _reads_absent(self, state: Mapping[str, Any]) -> bool:
        outcomes: Mapping[str, Mapping[str, Any]] = state.get("outcomes") or {}
        return any(outcomes.get(dep, {}).get("status") in NOT_PRODUCED for dep in self.reads)

    async def _gate(self, state: Mapping[str, Any], env: RunEnv) -> dict[str, Any] | None:
        """None when the step goes ahead; else the update for a step that does not run."""
        if state.get("end"):
            return {}
        if self._reads_absent(state):
            return await self._skip(env)
        if self.compiled.when is not None:
            try:
                holds = evaluate_condition(self.compiled.when, StateScope(self.spec, state))
            except EvaluationError as error:
                return await self._evaluation_failure(env, error)
            if not holds:
                return await self._skip(env)
        return None

    def _params(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Raises EvaluationError when a reference cannot be read."""
        params: dict[str, Any] = evaluate_value(self.compiled.with_, StateScope(self.spec, state))
        if self.approved is not None:
            params[APPROVED] = self.approved
        return params

    # Outcomes

    def _end(self, status: str, reason: str, category: str | None) -> dict[str, Any]:
        return {"status": status, "reason": reason, "step": self.id, "category": category, "rank": self.rank}

    def _failed(self, failure: StepFailure) -> dict[str, Any]:
        outcome = {
            "status": FAILED,
            "kind": self.entry_kind,
            "reason": failure.reason,
            "category": failure.category.value,
        }
        if self.step.required or isinstance(failure, InputEvicted):
            end = self._end(FAILED, failure.reason, failure.category.value)
            return {"outcomes": {self.id: outcome}, "end": end}
        return {
            "outcomes": {self.id: {**outcome, "status": FAILED_OPTIONAL}},
            "gaps": [{"step": self.id, "reason": failure.reason}],
        }

    def _method(self, state: Mapping[str, Any], result: CapabilityResult) -> str | None:
        if self.compiled.method is None:
            return result.method
        try:
            return render_template(self.compiled.method, StateScope(self.spec, state, text=True), own=ResultView(result))
        except (EvaluationError, ValueError):
            return result.method

    def _settle(
        self, env: RunEnv, state: Mapping[str, Any], result: CapabilityResult, attempt: int, node: str
    ) -> dict[str, Any]:
        if result.status == "needs_input":
            assert result.question is not None
            question = result.question.model_dump(mode="json")
            env.context.events.question({"run_id": env.run_id, "step": self.id, "question": question})
            _pause({"kind": PAUSE_QUESTION, "node": node, "step": self.id, "question": question})
        if result.status == "done":
            outcome = {"status": DONE, "kind": "use", "result": result.recorded(), "method": self._method(state, result)}
            return {"outcomes": {self.id: outcome}, "attempts": {self.id: attempt}}
        reason = result.reason or ""
        outcome = {
            "status": INSUFFICIENT,
            "kind": "use",
            "result": result.recorded(),
            "reason": reason,
            "category": FailureCategory.NO_DATA.value,
        }
        if self.step.required:
            end = self._end(INSUFFICIENT, reason, FailureCategory.NO_DATA.value)
            return {"outcomes": {self.id: outcome}, "attempts": {self.id: attempt}, "end": end}
        return {
            "outcomes": {self.id: {**outcome, "status": FAILED_OPTIONAL}},
            "attempts": {self.id: attempt},
            "gaps": [{"step": self.id, "reason": reason}],
        }

    def _transient(self, failure: StepFailure) -> bool:
        return (
            failure.category is FailureCategory.SOURCE_UNAVAILABLE
            and not isinstance(failure, JobStillRunning)
            and self._cap().retry_safe
        )

    def _start_attempt(self, state: Mapping[str, Any], resumed: Mapping[str, Any] | None) -> int:
        if resumed is not None and isinstance(resumed.get("attempt"), int):
            return int(resumed["attempt"])
        attempts: Mapping[str, int] = state.get("attempts") or {}
        return attempts.get(self.id, 0)

    async def _too_many(self, env: RunEnv, submissions: int) -> dict[str, Any]:
        """The step's end once the data provider has failed its retrieval ``submissions`` times."""
        failure = StepFailure(
            FailureCategory.SOURCE_UNAVAILABLE,
            f"The data provider failed the retrieval for step {self.id} {submissions} times, so it was not tried again.",
        )
        await self._record(env, status=FAILED, failure=failure)
        return self._failed(failure)

    def _retry_pause(self, node: str, attempt: int) -> NoReturn:
        _pause({"kind": PAUSE_RETRY, "node": node, "step": self.id, "attempt": attempt})

    # Nodes

    async def _node(
        self,
        node: str,
        state: WorkflowState,
        runtime: Runtime[StepContext],
        body: Callable[[WorkflowState, StepContext], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run a node body with a recorder that keeps file order among its siblings.

        A run found cancelled, before the step or by a refused trace write,
        takes no more work and ends with status ``cancelled``.
        """
        context = runtime.context
        env = context.env
        turns = context.turns
        try:
            if state.get("end"):
                return {}
            identity = env.context.identity
            run = await env.context.storage.traces.get_run(identity.user_id, env.run_id)
            if run.status == CANCELLED:
                return {"end": self._end(CANCELLED, CANCELLED_REASON, None)}
            recorder = _TurnRecorder(env.recorder, turns, node)
            return await body(state, replace(context, env=replace(env, recorder=recorder)))
        except RunClosed as closed:
            if closed.status != CANCELLED:
                raise
            return {"end": self._end(CANCELLED, CANCELLED_REASON, None)}
        finally:
            turns.release(node)

    async def run(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """A one-phase ``use`` step."""
        return await self._node(self.id, state, runtime, self._run)

    async def begin(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """A two-phase step's first node."""
        return await self._node(self.id + BEGIN_SUFFIX, state, runtime, self._begin)

    async def complete(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """A two-phase step's second node."""
        return await self._node(self.id, state, runtime, self._complete)

    async def stop(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """A ``stop`` step."""
        return await self._node(self.id, state, runtime, self._stop)

    async def _run(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """A one-phase ``use`` step."""
        env = context.env
        gated = await self._gate(state, env)
        if gated is not None:
            return gated
        try:
            params = self._params(state)
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        attempt = self._start_attempt(state, context.resumed.get(self.id))
        tries = 0
        while True:
            try:
                result = await invoke(env, self._cap(), params, step=self.id, attempt=attempt)
            except StepFailure as failure:
                if not self._transient(failure):
                    return self._failed(failure)
                if tries >= context.retry.retries:
                    self._retry_pause(self.id, attempt + 1)
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                attempt += 1
                continue
            return self._settle(env, state, result, attempt, self.id)

    async def _begin(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """A two-phase step's first node: submit, and save the job reference."""
        env = context.env
        node = self.id + BEGIN_SUFFIX
        gated = await self._gate(state, env)
        if gated is not None:
            return gated
        try:
            params = self._params(state)
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        attempt = self._start_attempt(state, context.resumed.get(node))
        tries = 0
        while True:
            # An attempt is one submission, numbered from 0.
            if attempt >= context.retry.max_submissions:
                return await self._too_many(env, attempt)
            try:
                job = await begin(env, self._cap(), params, step=self.id, attempt=attempt)
            except StepFailure as failure:
                if not self._transient(failure):
                    return self._failed(failure)
                if tries >= context.retry.retries:
                    self._retry_pause(node, attempt + 1)
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                attempt += 1
                continue
            if isinstance(job, CapabilityResult):
                return self._settle(env, state, job, attempt, node)
            return {"jobs": {self.id: job}, "attempts": {self.id: attempt}}

    async def _complete(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """A two-phase step's second node: finish from the saved job.

        A job still running pauses the run with the job kept. A transient
        failure retries ``complete`` with the same job, since the job itself
        may be fine. Only a job that ended at the provider without a result
        (``JobFailed``) makes the step begin again, with the next attempt; the
        new job is then held at a save point in the checkpoint before it is
        waited on. Retries share one bound; past it the run is saved to retry
        later, keeping the job where there still is one.
        """
        env = context.env
        outcomes = state.get("outcomes") or {}
        if state.get("end") or self.id in outcomes:
            return {}
        try:
            params = self._params(state)
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        resumed = context.resumed.get(self.id)
        jobs: Mapping[str, JobRef] = state.get("jobs") or {}
        job: JobRef | None = jobs.get(self.id)
        attempt = self._start_attempt(state, resumed)
        if resumed is not None and resumed.get("kind") in (PAUSE_JOB, PAUSE_RETRY, PAUSE_SAVE):
            saved = resumed.get("job")
            job = dict(saved) if isinstance(saved, dict) else None
        # A save point continues the same round of retries; a resume starts a new one.
        tries = int(resumed.get("tries", 0)) if resumed is not None and resumed.get("kind") == PAUSE_SAVE else 0
        while True:
            if job is None:
                if attempt >= context.retry.max_submissions:
                    return await self._too_many(env, attempt)
                try:
                    got = await begin(env, self._cap(), params, step=self.id, attempt=attempt)
                except StepFailure as failure:
                    if not self._transient(failure):
                        return self._failed(failure)
                    if tries >= context.retry.retries:
                        self._retry_pause(self.id, attempt + 1)
                    tries += 1
                    await context.retry.sleep(context.retry.backoff(tries))
                    attempt += 1
                    continue
                if isinstance(got, CapabilityResult):
                    return self._settle(env, state, got, attempt, self.id)
                _pause({
                    "kind": PAUSE_SAVE, "node": self.id, "step": self.id, "job": got, "attempt": attempt, "tries": tries,
                })
            try:
                result = await complete(env, self._cap(), params, job, step=self.id)
            except JobStillRunning as running:
                _pause({
                    "kind": PAUSE_JOB,
                    "node": self.id,
                    "step": self.id,
                    "job": job,
                    "attempt": attempt,
                    "still_running": list(running.still_running),
                })
            except StepFailure as failure:
                dead = isinstance(failure, JobFailed)
                if not (dead and self._cap().retry_safe) and not self._transient(failure):
                    return self._failed(failure)
                if tries >= context.retry.retries:
                    if dead:
                        self._retry_pause(self.id, attempt + 1)
                    _pause({"kind": PAUSE_RETRY, "node": self.id, "step": self.id, "job": job, "attempt": attempt})
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                if dead:
                    attempt += 1
                    job = None
                continue
            update = self._settle(env, state, result, attempt, self.id)
            update["jobs"] = {self.id: job}
            return update

    async def _stop(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """A ``stop`` step: ends the run with its status and reason when its condition holds."""
        env = context.env
        if state.get("end"):
            return {}
        if self._reads_absent(state):
            return await self._skip(env)
        assert isinstance(self.step, StopStep) and self.compiled.when is not None and self.compiled.reason is not None
        scope = StateScope(self.spec, state)
        try:
            holds = evaluate_condition(self.compiled.when, scope)
            text = StateScope(self.spec, state, text=True)
            reason = render_template(self.compiled.reason, text)[:MAX_REASON_CHARS] if holds else ""
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        if not holds:
            return await self._skip(env)
        recorded = {"status": self.step.stop, "reason": reason}
        await self._record(env, status=DONE, result=recorded)
        outcome = {"status": STOPPED, "kind": "stop", "result": recorded, "reason": reason}
        end = self._end(self.step.stop, reason, None)
        return {"outcomes": {self.id: outcome}, "end": end}


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    graph: CompiledStateGraph[Any, Any, Any, Any]
    # Each node's superstep and rank, for TraceTurns.
    plan: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def turns(self, resumed: Collection[str] = ()) -> TraceTurns:
        return TraceTurns(self.plan, resumed)

    @property
    def key(self) -> tuple[str, int]:
        return (self.spec.name, self.spec.version)


class NotRunnable(ValueError):
    """A spec the runner refuses, with the spec's own fixed wording."""

    def __init__(self, name: str, version: int, reason: str) -> None:
        super().__init__(f"Workflow {name} v{version} cannot run: {reason}.")
        self.name = name
        self.version = version
        self.reason = reason


def compile_workflow(
    spec: WorkflowSpec, capabilities: CapabilityRegistry, *, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledWorkflow:
    """The spec as a compiled graph saving to ``checkpointer``. Raises NotRunnable."""
    if spec.not_runnable_reason is not None:
        raise NotRunnable(spec.name, spec.version, spec.not_runnable_reason)
    builder: Any = StateGraph(WorkflowState, context_schema=StepContext)
    order = spec.order()
    plan: dict[str, tuple[int, int]] = {}
    for step_id in order:
        nodes = _StepNodes(spec, capabilities, step_id)
        if isinstance(nodes.step, StopStep):
            builder.add_node(step_id, nodes.stop)
        elif nodes._cap().two_phase:
            builder.add_node(step_id + BEGIN_SUFFIX, nodes.begin)
            builder.add_node(step_id, nodes.complete)
            builder.add_edge(step_id + BEGIN_SUFFIX, step_id)
        else:
            builder.add_node(step_id, nodes.run)
    depended: set[str] = set()
    for step_id in order:
        deps = sorted(spec.dependencies(step_id), key=order.index)
        entry = entry_node(spec, capabilities, step_id)
        # A node runs one superstep after the last of the nodes it waits for.
        level = 1 + max((plan[dep][0] for dep in deps), default=0)
        rank = spec.step_ids.index(step_id)
        plan[entry] = (level, rank)
        if entry != step_id:
            plan[step_id] = (level + 1, rank)
        if not deps:
            builder.add_edge(START, entry)
        elif len(deps) == 1:
            builder.add_edge(deps[0], entry)
        else:
            # Waits for every dependency, not just the first to finish.
            builder.add_edge(deps, entry)
        depended.update(deps)
    for step_id in order:
        if step_id not in depended:
            builder.add_edge(step_id, END)
    graph: CompiledStateGraph[Any, Any, Any, Any] = builder.compile(checkpointer=checkpointer)
    return CompiledWorkflow(spec=spec, graph=graph, plan=plan)
