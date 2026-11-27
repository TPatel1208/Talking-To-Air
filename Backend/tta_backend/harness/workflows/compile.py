"""A workflow spec compiled to a LangGraph graph: one node per step, edges from the references.

A ``use`` step whose capability is two-phase is two nodes, ``<id>.begin`` and
``<id>``: the first submits and writes the job reference into state, the
second completes from it, so the reference is in a checkpoint before any
waiting. Every other step is one node named by its id.

An ``agent`` step's node asks the run's ``agent_step`` model for an answer of
the step's declared shape (see ``agent``). Its trace entry records the
references it read and the notes it was given; its result is the answer, the
model that gave it and the usage counts. The answer is the step's result in
state, which later steps read as ``$step.field``.

An ``ask`` step's node resolves its options or items, records a ``waiting``
``ask`` entry and pauses on a question (see ``answers`` for the payload). The
runner resumes it with the checked answer as the interrupt's resume value;
the node then records a ``done`` ``ask`` entry holding the answer, which is
the step's result: ``$step`` reads it whole and ``$step.0`` by index.

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

- ``question``: an ``ask`` step, or a capability that came back with a
  question whose answer code knows how to pass back to it. The interrupt
  value carries the question payload and, for a capability, the rule that
  applies the answer; on resume the runner adds the checked ``answer``. The
  node runs again with the answer as a parameter, recorded first as an
  ``ask`` entry, and keeps it in state under ``answers`` for the step's
  later nodes. A capability question with no such rule ends the run as
  ``needs_input`` with the question, as a place that does not resolve does;
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

from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.failures import InputEvicted, JobFailed, JobStillRunning
from tta_backend.harness.capabilities.invoke import begin, complete, invoke
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.result import CapabilityResult, JobRef, walk_path
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.knowledge import Note, NoteCatalogue
from tta_backend.harness.models import DEFAULT_BACKOFF, Backoff, Usage
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.trace import (
    MAX_REASON_CHARS,
    EntryKind,
    FailureCategory,
    InvalidTrace,
    RunClosed,
    StepFailure,
    StepRecorder,
    StepScope,
    serialize_json,
)
from tta_backend.harness.workflows.agent import MAX_READ_BYTES, MAX_READS_BYTES, ask_model, build_prompt, json_size
from tta_backend.harness.workflows.answers import (
    MAX_ITEMS,
    MAX_OPTIONS,
    AnswerRejected,
    apply_rule,
    capability_rule,
    check_answer,
    question_payload,
)
from tta_backend.harness.workflows.expressions import (
    EvaluationError,
    MissingValue,
    Ref,
    StepUnavailable,
    evaluate_condition,
    evaluate_value,
    render_template,
)
from tta_backend.harness.workflows.loader import approved_parameter
from tta_backend.harness.workflows.spec import AgentStep, AskStep, StopStep, UseStep, WorkflowSpec

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
NEEDS_INPUT = "needs_input"
# What an agent entry records in place of an answer too large for the trace.
ANSWER_NOT_RECORDED = "not recorded: larger than the trace holds"
# An agent step's run failing outside its model calls.
AGENT_INTERNAL = "Step {step} hit an internal error while asking its model."
# An answer that stops its step: ``narrow`` to a retrieval's size question.
ANSWER_STOPPED = (
    "The retrieval for step {step} was not confirmed, so the run stopped; "
    "run the workflow again over a smaller area or fewer dates."
)
NOTHING_OFFERED = "Step {step} has no {what} to offer the user."
NOT_A_LIST = "Step {step} offers {what} that are not a list."
TOO_MANY_OFFERED = "Step {step} offers more than {limit} {what}."
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
    statuses above), ``kind`` (use, ask, agent or stop), the ``result`` (the
    recorded capability result, or an ask or agent step's answer), the ``method``
    sentence, and ``reason`` and ``category`` for a step that did not finish. ``end`` is set by the step that ended the run
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
    # The parameters a user's answer to a capability's question set, by step.
    answers: Annotated[dict[str, dict[str, Any]], _merge]
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
    # Retries of each model an agent step tries, and the waits between them.
    model_retries: int = 2
    model_backoff: Backoff = DEFAULT_BACKOFF

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries is zero or more")
        if self.model_retries < 0:
            raise ValueError("model_retries is zero or more")
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
    orders the trace entries of one superstep; ``notes`` are the knowledge
    notes agent steps are given, none when absent."""

    env: RunEnv
    retry: RetrySettings = field(default_factory=RetrySettings)
    resumed: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    turns: TraceTurns = field(default_factory=TraceTurns)
    notes: NoteCatalogue | None = None


class ResultView:
    """A recorded capability result as a reference reads it; a bare ``$step`` is its handle."""

    def __init__(self, result: CapabilityResult) -> None:
        self.result = result

    def read(self, path: str) -> Any:
        return self.result.read(path)

    def value(self) -> Any:
        return self.result.handle


class AnswerView:
    """An agent step's answer as a reference reads it; a bare ``$step`` is the whole answer."""

    def __init__(self, answer: Mapping[str, Any]) -> None:
        self.answer = dict(answer)

    def read(self, path: str) -> Any:
        return walk_path(self.answer, path)

    def value(self) -> Any:
        return self.answer


class AskView:
    """An ask step's answer as a reference reads it; a bare ``$step`` is the whole answer."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer

    def read(self, path: str) -> Any:
        return walk_path(self.answer, path)

    def value(self) -> Any:
        return self.answer


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
        self._views: dict[str, ResultView | AnswerView | AskView | None] = {}

    def input(self, name: str) -> Any:
        if name in self._names:
            return self._names[name]
        return self._inputs[name]

    def step(self, step_id: str) -> ResultView | AnswerView | AskView | None:
        if step_id not in self._views:
            outcome = self._outcomes.get(step_id) or {}
            recorded = outcome.get("result") if outcome.get("status") == DONE else None
            view: ResultView | AnswerView | AskView | None = None
            if recorded is not None:
                kind = outcome.get("kind")
                if kind == "ask":
                    view = AskView(recorded)
                elif kind == "agent":
                    view = AnswerView(recorded)
                else:
                    view = ResultView(CapabilityResult.model_validate(recorded))
            self._views[step_id] = view
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
        self.entry_kind: EntryKind = (
            "stop" if isinstance(self.step, StopStep)
            else "agent" if isinstance(self.step, AgentStep)
            else "ask" if isinstance(self.step, AskStep)
            else "use"
        )
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
        answered: Mapping[str, Mapping[str, Any]] = state.get("answers") or {}
        params.update(answered.get(self.id) or {})
        return params

    # Answers to a capability's question

    async def _take_answer(
        self, env: RunEnv, resumed: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """The parameters the user's answers set for this node, and the update
        that ends the step when an answer stops it.

        Answers applied before a later pause in the node come back with that
        pause as ``answered``. A question resumed with its answer is recorded
        as an ``ask`` entry before the step runs again.
        """
        if resumed is None:
            return {}, None
        earlier = resumed.get("answered")
        answered: dict[str, Any] = dict(earlier) if isinstance(earlier, Mapping) else {}
        rule = resumed.get("rule")
        if resumed.get("kind") != PAUSE_QUESTION or "answer" not in resumed or not isinstance(rule, Mapping):
            return answered, None
        answer = resumed["answer"]
        applied = apply_rule(rule, answer)
        question: Mapping[str, Any] = resumed.get("question") or {}
        entry = {"kind": question.get("kind"), "question": question.get("question")}
        async with env.recorder.step(kind="ask", capability=self.capability_name, params=entry, step=self.id) as scope:
            scope.result = _recorded_ask({"answer": answer, "applied": applied})
        if applied is None:
            reason = ANSWER_STOPPED.format(step=self.id)
            outcome = {"status": STOPPED, "kind": "use", "reason": reason}
            return answered, {"outcomes": {self.id: outcome}, "end": self._end(CANCELLED, reason, None)}
        answered.update(applied)
        return answered, None

    def _keep(self, update: dict[str, Any], answered: Mapping[str, Any]) -> dict[str, Any]:
        """``update`` keeping the parameters answers set, for the step's later nodes."""
        if answered:
            update["answers"] = {self.id: dict(answered)}
        return update

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
        self,
        env: RunEnv,
        state: Mapping[str, Any],
        result: CapabilityResult,
        attempt: int,
        node: str,
        *,
        answered: Mapping[str, Any] | None = None,
        held: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The step's update for a capability's result. A question pauses the
        node, carrying ``held`` (what the node needs to continue) and the
        answers already applied; one whose answer code cannot pass back ends the run."""
        if result.status == "needs_input":
            assert result.question is not None
            asked = result.question
            rule = capability_rule(self._cap(), asked)
            if rule is None:
                outcome: dict[str, Any] = {
                    "status": NEEDS_INPUT, "kind": "use", "result": result.recorded(), "reason": asked.prompt,
                }
                end = {**self._end(NEEDS_INPUT, asked.prompt, None), "question": asked.model_dump(mode="json")}
                return {"outcomes": {self.id: outcome}, "attempts": {self.id: attempt}, "end": end}
            payload = question_payload(
                env.run_id, self.id, asked.kind, asked.prompt, options=asked.options, items=asked.items,
            )
            env.context.events.question({"run_id": env.run_id, "step": self.id, "question": payload})
            _pause({
                "kind": PAUSE_QUESTION, "node": node, "step": self.id, "question": payload, "rule": rule,
                "attempt": attempt, **(held or {}), **_answered(answered),
            })
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

    def _retry_pause(self, node: str, attempt: int, answered: Mapping[str, Any]) -> NoReturn:
        _pause({"kind": PAUSE_RETRY, "node": node, "step": self.id, "attempt": attempt, **_answered(answered)})

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

    async def agent(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """An ``agent`` step."""
        return await self._node(self.id, state, runtime, self._agent)

    async def ask(self, state: WorkflowState, runtime: Runtime[StepContext]) -> dict[str, Any]:
        """An ``ask`` step."""
        return await self._node(self.id, state, runtime, self._ask)

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
        resumed = context.resumed.get(self.id)
        answered, stopped = await self._take_answer(env, resumed)
        if stopped is not None:
            return stopped
        params.update(answered)
        attempt = self._start_attempt(state, resumed)
        tries = 0
        while True:
            try:
                result = await invoke(env, self._cap(), params, step=self.id, attempt=attempt)
            except StepFailure as failure:
                if not self._transient(failure):
                    return self._keep(self._failed(failure), answered)
                if tries >= context.retry.retries:
                    self._retry_pause(self.id, attempt + 1, answered)
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                attempt += 1
                continue
            return self._keep(self._settle(env, state, result, attempt, self.id, answered=answered), answered)

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
        resumed = context.resumed.get(node)
        answered, stopped = await self._take_answer(env, resumed)
        if stopped is not None:
            return stopped
        params.update(answered)
        attempt = self._start_attempt(state, resumed)
        tries = 0
        while True:
            # An attempt is one submission, numbered from 0.
            if attempt >= context.retry.max_submissions:
                return self._keep(await self._too_many(env, attempt), answered)
            try:
                job = await begin(env, self._cap(), params, step=self.id, attempt=attempt)
            except StepFailure as failure:
                if not self._transient(failure):
                    return self._keep(self._failed(failure), answered)
                if tries >= context.retry.retries:
                    self._retry_pause(node, attempt + 1, answered)
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                attempt += 1
                continue
            if isinstance(job, CapabilityResult):
                return self._keep(self._settle(env, state, job, attempt, node, answered=answered), answered)
            return self._keep({"jobs": {self.id: job}, "attempts": {self.id: attempt}}, answered)

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
        answered, stopped = await self._take_answer(env, resumed)
        if stopped is not None:
            return stopped
        params.update(answered)
        carry = _answered(answered)
        jobs: Mapping[str, JobRef] = state.get("jobs") or {}
        job: JobRef | None = jobs.get(self.id)
        attempt = self._start_attempt(state, resumed)
        if resumed is not None and resumed.get("kind") in (PAUSE_JOB, PAUSE_RETRY, PAUSE_SAVE, PAUSE_QUESTION):
            saved = resumed.get("job")
            job = dict(saved) if isinstance(saved, dict) else None
        # A save point continues the same round of retries; a resume starts a new one.
        tries = int(resumed.get("tries", 0)) if resumed is not None and resumed.get("kind") == PAUSE_SAVE else 0
        while True:
            if job is None:
                if attempt >= context.retry.max_submissions:
                    return self._keep(await self._too_many(env, attempt), answered)
                try:
                    got = await begin(env, self._cap(), params, step=self.id, attempt=attempt)
                except StepFailure as failure:
                    if not self._transient(failure):
                        return self._keep(self._failed(failure), answered)
                    if tries >= context.retry.retries:
                        self._retry_pause(self.id, attempt + 1, answered)
                    tries += 1
                    await context.retry.sleep(context.retry.backoff(tries))
                    attempt += 1
                    continue
                if isinstance(got, CapabilityResult):
                    settled = self._settle(env, state, got, attempt, self.id, answered=answered, held={"job": None})
                    return self._keep(settled, answered)
                _pause({
                    "kind": PAUSE_SAVE, "node": self.id, "step": self.id, "job": got, "attempt": attempt, "tries": tries,
                    **carry,
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
                    **carry,
                })
            except StepFailure as failure:
                dead = isinstance(failure, JobFailed)
                if not (dead and self._cap().retry_safe) and not self._transient(failure):
                    return self._keep(self._failed(failure), answered)
                if tries >= context.retry.retries:
                    if dead:
                        self._retry_pause(self.id, attempt + 1, answered)
                    _pause({
                        "kind": PAUSE_RETRY, "node": self.id, "step": self.id, "job": job, "attempt": attempt, **carry,
                    })
                tries += 1
                await context.retry.sleep(context.retry.backoff(tries))
                if dead:
                    attempt += 1
                    job = None
                continue
            update = self._settle(env, state, result, attempt, self.id, answered=answered, held={"job": job})
            update["jobs"] = {self.id: job}
            return self._keep(update, answered)

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

    async def _agent(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """An ``agent`` step: its reads and notes to the model, and the typed answer back.

        A wrong-shaped answer is retried once inside the model run. An answer
        that still does not fit, every model failing, a failure outside the
        model calls, or a run with no model for agent steps fails the step.
        """
        env = context.env
        gated = await self._gate(state, env)
        if gated is not None:
            return gated
        assert isinstance(self.step, AgentStep)
        returns = self.step.agent.returns
        scope = StateScope(self.spec, state)
        try:
            reads = [(ref.text(), evaluate_value(ref, scope)) for ref in self.compiled.reads]
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        notes = await self._notes(env, state, context.notes)
        params = {"reads": [ref for ref, _ in reads], "notes": [note.id for note in notes]}
        try:
            async with env.recorder.step(kind="agent", params=params, step=self.id) as entry:
                oversized = _oversized_reads(self.id, reads)
                if oversized is not None:
                    raise oversized
                choice = env.model_for("agent_step")
                prompt = build_prompt(self.step.agent.instruction, reads, notes, returns)
                asked = await ask_model(
                    choice, prompt, returns, max_retries=context.retry.model_retries, backoff=context.retry.model_backoff,
                )
                usage = _usage(asked.usage)
                if asked.outcome == "internal_error":
                    raise StepFailure(FailureCategory.INTERNAL_ERROR, AGENT_INTERNAL.format(step=self.id), result=usage)
                if asked.answer is None:
                    rule = failures.MODEL_FAILED if asked.outcome == "model_failed" else failures.MODEL_ANSWER_INVALID
                    category, wording = rule
                    raise StepFailure(category, wording.format(name=f"step {self.id}"), result=usage)
                answer = asked.answer
                entry.result = _recorded_answer(answer, usage)
        except StepFailure as failure:
            return self._failed(failure)
        outcome = {"status": DONE, "kind": "agent", "result": answer, "method": self._agent_method(state)}
        return {"outcomes": {self.id: outcome}}

    async def _ask(self, state: WorkflowState, context: StepContext) -> dict[str, Any]:
        """An ``ask`` step: puts its question to the user and takes the checked answer as its result.

        The first run records a ``waiting`` entry and pauses. Resumed with an
        answer, it records a ``done`` entry holding the answer; resumed
        without one, as when the user answered another question of the run,
        it pauses again.
        """
        env = context.env
        gated = await self._gate(state, env)
        if gated is not None:
            return gated
        assert isinstance(self.step, AskStep)
        ask = self.step.ask
        picks = ask.kind in ("pick_one", "pick_many")
        node = self.compiled.options if picks else self.compiled.items
        try:
            offered = [] if node is None else evaluate_value(node, StateScope(self.spec, state))
        except EvaluationError as error:
            return await self._evaluation_failure(env, error)
        problem = None if ask.kind == "text" else _offer_problem(self.id, "options" if picks else "items", offered)
        if problem is not None:
            await self._record(env, status=FAILED, failure=problem)
            return self._failed(problem)
        listed = list(offered or ())
        payload = question_payload(
            env.run_id, self.id, ask.kind, ask.question,
            options=listed if picks else (), items=() if picks else listed, max_items=ask.max_items,
        )
        entry = {"kind": ask.kind, "question": ask.question}
        if context.resumed.get(self.id) is None:
            async with env.recorder.step(kind="ask", params=entry, step=self.id) as scope:
                scope.status = "waiting"
                scope.result = None if ask.kind == "text" else {"offered": len(listed)}
            env.context.events.question({"run_id": env.run_id, "step": self.id, "question": payload})
        answer = _await_answer({"kind": PAUSE_QUESTION, "node": self.id, "step": self.id, "question": payload}, payload)
        async with env.recorder.step(kind="ask", params=entry, step=self.id) as scope:
            scope.result = _recorded_ask({"answer": answer})
        outcome = {"status": DONE, "kind": "ask", "result": answer, "method": self._agent_method(state)}
        return {"outcomes": {self.id: outcome}}

    async def _notes(self, env: RunEnv, state: Mapping[str, Any], catalogue: NoteCatalogue | None) -> tuple[Note, ...]:
        """The notes for the datasets and quantities behind what the step reads:
        the handles of the steps and inputs it reads, and any quantity input it reads."""
        if catalogue is None or not len(catalogue):
            return ()
        outcomes: Mapping[str, Mapping[str, Any]] = state.get("outcomes") or {}
        inputs: Mapping[str, Any] = state.get("inputs") or {}
        datasets: set[str] = set()
        quantities: set[str] = set()
        handles: list[str] = []
        for ref in self.compiled.reads:
            if not isinstance(ref, Ref):
                continue
            if ref.root == "input":
                declared, value = self.spec.inputs[ref.name], inputs.get(ref.name)
                if declared.type == "quantity" and isinstance(value, str):
                    quantities.add(value)
                elif declared.handle_kind is not None and isinstance(value, str):
                    handles.append(value)
            elif ref.root == "step":
                outcome = outcomes.get(ref.name) or {}
                if outcome.get("kind") == "use" and outcome.get("status") == DONE and outcome.get("result"):
                    result = CapabilityResult.model_validate(outcome["result"])
                    if result.product:
                        datasets.add(result.product)
                    if result.handle:
                        handles.append(result.handle)
        user_id = env.context.identity.user_id
        for handle_id in dict.fromkeys(handles):
            try:
                record = await env.resolver.get(user_id, handle_id)
            except HandleError:
                continue
            if record.descriptor.product:
                datasets.add(record.descriptor.product)
            if record.descriptor.quantity:
                quantities.add(record.descriptor.quantity)
        return catalogue.notes_for(datasets=datasets, quantities=quantities)

    def _agent_method(self, state: Mapping[str, Any]) -> str | None:
        if self.compiled.method is None:
            return None
        try:
            return render_template(self.compiled.method, StateScope(self.spec, state, text=True))
        except (EvaluationError, ValueError):
            return None


def _answered(answered: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a pause carries of the answers already applied in its node."""
    return {"answered": dict(answered)} if answered else {}


def _await_answer(value: dict[str, Any], question: Mapping[str, Any]) -> Any:
    """The checked answer an ask step's pause is resumed with. A resume
    without one, or with one that does not fit, pauses again."""
    while True:
        got = interrupt(value)
        if isinstance(got, Mapping) and "answer" in got:
            try:
                return check_answer(question, got["answer"])
            except AnswerRejected:
                continue


def _offer_problem(step_id: str, what: str, offered: Any) -> StepFailure | None:
    """Why an ask step cannot put what it resolved to the user; None when it can."""
    if not isinstance(offered, list | tuple):
        return StepFailure(FailureCategory.INTERNAL_ERROR, NOT_A_LIST.format(step=step_id, what=what))
    if not offered:
        return StepFailure(FailureCategory.NO_DATA, NOTHING_OFFERED.format(step=step_id, what=what))
    limit = MAX_OPTIONS if what == "options" else MAX_ITEMS
    if len(offered) > limit:
        return StepFailure(
            FailureCategory.INTERNAL_ERROR, TOO_MANY_OFFERED.format(step=step_id, limit=limit, what=what),
        )
    return None


def _recorded_ask(result: dict[str, Any]) -> dict[str, Any]:
    """An ask entry's result: the answer when it fits the trace, else a marker."""
    try:
        serialize_json(result, "result")
    except InvalidTrace:
        return {**result, "answer": ANSWER_NOT_RECORDED}
    return result


def _oversized_reads(step_id: str, reads: list[tuple[str, Any]]) -> StepFailure | None:
    """A failure when what an agent step reads is past its size bounds."""
    total = 0
    for ref, value in reads:
        size = json_size(value)
        if size > MAX_READ_BYTES:
            return StepFailure(
                FailureCategory.INTERNAL_ERROR,
                f"Step {step_id} reads {ref}, which is larger than an agent step may read ({MAX_READ_BYTES} bytes).",
            )
        total += size
    if total > MAX_READS_BYTES:
        return StepFailure(
            FailureCategory.INTERNAL_ERROR,
            f"The values step {step_id} reads are larger together than an agent step may read ({MAX_READS_BYTES} bytes).",
        )
    return None


def _usage(usage: Usage) -> dict[str, Any]:
    return {
        "model": usage.model_name,
        "model_calls": usage.model_calls,
        "failed_model_calls": usage.failed_model_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _recorded_answer(answer: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """An agent entry's result: the answer when it fits the trace, else a marker, then the usage."""
    recorded = {"answer": answer, **usage}
    try:
        serialize_json(recorded, "result")
    except InvalidTrace:
        recorded = {"answer": ANSWER_NOT_RECORDED, **usage}
    return recorded


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
        elif isinstance(nodes.step, AgentStep):
            builder.add_node(step_id, nodes.agent)
        elif isinstance(nodes.step, AskStep):
            builder.add_node(step_id, nodes.ask)
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
