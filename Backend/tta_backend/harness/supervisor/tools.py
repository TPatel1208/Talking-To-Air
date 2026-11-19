"""The supervisor's six tools, built for one turn.

- ``run_workflow(name, inputs)``: a runnable workflow at its latest runnable
  version. Inputs that fail the spec's checks come back with the reason; a
  second such failure for the same workflow in a turn is answered with a
  fixed message saying not to retry it.
- ``resume_run(run_id)``: only a waiting workflow run whose jobs the turn's
  ledger shows as ready, failed or unchecked, once per turn.
- ``explore(tasks)``: at most ``MAX_TASKS`` exploration tasks, run together.
- ``resolve_area``, ``search_datasets``, ``check_availability``: lookups
  through ``invoke``. A turn's lookups share one exploratory run, started on
  the first and closed with ``close`` at the end of the turn.

Work that needs a connector whose service is down (``context.reachable``) is
refused before any run starts, in fixed wording; the connector is kept in
``unavailable`` for the answer to name.

Every result reaches the model as ``result_view``; a lookup as the small JSON
specialists see. Refusals use fixed wording. An unexpected exception is
logged by type and the model gets a fixed internal-error message.

Tool schemas hold only keywords every provider accepts: references are
inlined, a nullable field is the field's own type, and a field that takes
several shapes has no type and says the shapes in its description.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp
from pydantic import ValidationError

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.invoke import invoke
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.context.ledger import Ledger
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import Backoff
from tta_backend.harness.results import Mode, Result
from tta_backend.harness.run_context import RunContext, RunEnv
from tta_backend.harness.specialists.explore import explore
from tta_backend.harness.specialists.task import ExploreLimits, ExploreTask
from tta_backend.harness.specialists.tools import failure_view, retry_helps, tool_schema
from tta_backend.harness.specialists.tools import result_view as capability_view
from tta_backend.harness.specialists.bundles import get_bundle
from tta_backend.harness.supervisor.answer import dumps, result_view, unavailable_wording, view_of
from tta_backend.harness.trace import FailureCategory, Run, RunNotFound, StepFailure, new_run_id
from tta_backend.harness.workflows.compile import NotRunnable
from tta_backend.harness.workflows.inputs import input_step_id
from tta_backend.harness.workflows.loader import UnknownWorkflow, WorkflowCatalogue
from tta_backend.harness.workflows.runner import INTERNAL_REASON, EnvFactory, RunNotResumable, WorkflowRunner

logger = logging.getLogger(__name__)

TOOL_NAMES = ("run_workflow", "resume_run", "explore", "resolve_area", "search_datasets", "check_availability")
LOOKUPS = {
    "resolve_area": "sources.areas.resolve",
    "search_datasets": "sources.satellite.search",
    "check_availability": "sources.satellite.check_availability",
}
MAX_TASKS = 4
# Job states a waiting workflow run may be resumed in: ready collects the
# jobs, failed submits again, unknown collects them or waits again.
RESUMABLE = frozenset({"ready", "failed", "unknown"})

UNKNOWN_WORKFLOW = "There is no runnable workflow by that name; use a name from the index, or explore instead."
NOT_RUNNABLE = "That workflow cannot run: {reason}."
STOP_RETRYING = (
    "This workflow's inputs failed its checks twice in this turn. Do not run it again this turn: "
    "explore instead, or answer and ask the user for what is missing."
)
NOT_READY = (
    "Only a workflow run the session ledger shows as waiting, with its retrieval jobs not still running, "
    "can be resumed; this one cannot be resumed now."
)
TOO_MANY_TASKS = f"An explore call takes 1 to {MAX_TASKS} tasks; split the work or drop a task."
SPLIT_PLACE = (
    "'{place}' is one part of a place the user joined with '+'. Pass the whole place exactly as the user "
    "wrote it; code resolves each part and asks the user about any part it cannot find."
)
_QUOTED_PLACE_CHARS = 80


def part_of_combined_place(place: str, message: str) -> bool:
    """Whether ``place`` is one side of a '+' the user wrote, such as 'NY' or
    'Wakanda' in 'NY + Wakanda': a part passed alone would skip the check
    that refuses the whole combined place when a part cannot be found."""
    words = place.split()
    if not words or "+" in place or "+" not in message:
        return False
    text = r"\s+".join(re.escape(word) for word in words)
    joined = re.compile(rf"(?<!\w){text}\s*\+|\+\s*{text}(?!\w)", re.IGNORECASE)
    return joined.search(message) is not None
BAD_TASK = "Task {number}: {problems}."
INTERNAL = "The tool stopped on an internal error. Do not retry it; answer with what you have."

RUN_WORKFLOW_DESCRIPTION = (
    "Start a workflow from the index by name; code picks its version. Give inputs in the index's input "
    "formats: a quantity from the listed ones, and dates as ISO dates or {start, end} ranges worked out from "
    "today's date in the session ledger message. Write places as the user wrote them: never expand, correct "
    "or normalise a place. An area input takes a handle from the session ledger or the place in the user's words."
)
RESUME_DESCRIPTION = (
    "Continue a workflow run the session ledger shows as waiting, as its advice for that run says; "
    "not one whose jobs are still running."
)
EXPLORE_DESCRIPTION = (
    f"Run 1 to {MAX_TASKS} exploration tasks in parallel, each its own exploratory run. Give each task an "
    "objective and a bundle, and any handles from the session ledger. Pass places, the variable and dates "
    "exactly as the user wrote them; do not expand, correct or normalise them, code resolves places."
)
TASK_FIELDS = {
    "objective": "What the task must establish, in one or two sentences.",
    "bundle": "analyse retrieves and computes, investigate works on existing handles, find-data learns what data exists.",
    "handles": "Handle ids from the session ledger the task may use.",
    "places": "Places exactly as the user wrote them, never expanded or corrected.",
    "variable": "The variable or pollutant as the user wrote it.",
    "dates": "The dates or period as the user wrote them.",
}
LOOKUP_NOTES = {
    "resolve_area": " Pass the place exactly as the user wrote it; the area handle it returns can be passed on.",
    "search_datasets": "",
    "check_availability": " Takes an area handle from resolve_area or the session ledger.",
}

# Keywords some providers refuse, dropped from every schema.
_DROPPED = frozenset({"$defs", "additionalProperties", "title", "discriminator", "$schema"})


# Schemas


def _shape(branch: Mapping[str, Any]) -> str:
    kind = branch.get("type")
    if kind == "string":
        return "an ISO date" if branch.get("format") == "date" else "text"
    if kind == "array":
        items = branch.get("items")
        if isinstance(items, Mapping) and items.get("type") == "string":
            return "a list of ISO dates" if items.get("format") == "date" else "a list of texts"
        return "a list of values"
    if kind == "object":
        names = ", ".join(branch.get("properties") or {})
        return f"an object with {names}" if names else "an object"
    return str(kind or "a value")


def provider_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """``schema`` with references inlined and only keywords providers accept.

    A nullable field becomes its own type, left out of ``required`` as it
    already is. A field taking several shapes loses its type and names the
    shapes in its description; the capability's own validation still checks it.
    """
    defs: Mapping[str, Any] = schema.get("$defs") or {}

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, Mapping):
            return node
        if "$ref" in node:
            target = defs[str(node["$ref"]).rsplit("/", 1)[-1]]
            return walk({**target, **{k: v for k, v in node.items() if k != "$ref"}})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROPPED or (key == "default" and value is None):
                continue
            if key == "properties":
                out[key] = {name: walk(prop) for name, prop in value.items()}
            elif key == "const":
                out["enum"] = [value]
            else:
                out[key] = walk(value)
        branches = out.pop("anyOf", None) or out.pop("oneOf", None)
        if branches is None:
            return out
        real = [b for b in branches if b.get("type") != "null"]
        if len(real) == 1:
            return {**real[0], **out}
        shapes = "; ".join(_shape(b) for b in real)
        said = out.get("description", "").rstrip()
        out["description"] = f"{said} One of: {shapes}.".strip()
        return out

    flat: dict[str, Any] = walk(copy.deepcopy(dict(schema)))
    flat.pop("description", None)
    return flat


def _task_schema() -> dict[str, Any]:
    schema = provider_schema(ExploreTask.model_json_schema())
    for name, said in TASK_FIELDS.items():
        schema["properties"][name]["description"] = said
    return schema


def tool_schemas(workflow_names: Sequence[str], capabilities: CapabilityRegistry) -> dict[str, dict[str, Any]]:
    """Each tool's argument schema; the same arguments always give the same schemas."""
    name: dict[str, Any] = {"type": "string", "description": "A workflow name from the index."}
    if workflow_names:
        name["enum"] = list(workflow_names)
    schemas: dict[str, dict[str, Any]] = {
        "run_workflow": {
            "type": "object",
            "properties": {
                "name": name,
                "inputs": {"type": "object", "description": "The workflow's inputs by name, as the index lists them."},
            },
            "required": ["name", "inputs"],
        },
        "resume_run": {
            "type": "object",
            "properties": {"run_id": {"type": "string", "description": "A run id from the session ledger."}},
            "required": ["run_id"],
        },
        "explore": {
            "type": "object",
            "properties": {"tasks": {"type": "array", "items": _task_schema(), "minItems": 1, "maxItems": MAX_TASKS}},
            "required": ["tasks"],
        },
    }
    for tool, capability in LOOKUPS.items():
        schemas[tool] = provider_schema(tool_schema(capabilities.get(capability)))
    return schemas


def latest_runnable(catalogue: WorkflowCatalogue) -> dict[str, int]:
    """Each workflow with a runnable version, mapped to its latest runnable version."""
    found: dict[str, int] = {}
    for workflow in catalogue.names():
        for version in sorted(catalogue.versions(workflow), reverse=True):
            if catalogue.get(workflow, version).not_runnable_reason is None:
                found[workflow] = version
                break
    return found


# One turn's tools


def _problems(error: ValidationError) -> str:
    # Only declared field names reach the wording; an unknown key is never echoed.
    said = []
    for item in error.errors():
        head = item["loc"][0] if item["loc"] else None
        field = head if isinstance(head, str) and head in ExploreTask.model_fields else "an unexpected field"
        said.append(f"{field}: {item['msg']}")
    return "; ".join(dict.fromkeys(said))


def _refused(reason: str) -> str:
    return dumps({"status": "refused", "reason": reason})


class TurnTools:
    """The tools for one turn and what they produced: results in the order
    they came back, and the turn's lookup run."""

    def __init__(
        self,
        *,
        context: RunContext,
        build_env: EnvFactory,
        ledger: Ledger,
        runner: WorkflowRunner,
        workflows: Mapping[str, int],
        capabilities: CapabilityRegistry,
        notes: NoteCatalogue,
        schemas: Mapping[str, dict[str, Any]],
        explore_limits: ExploreLimits | None,
        backoff: Backoff,
        message: str = "",
    ) -> None:
        self.context = context
        # The user's message this turn, which places passed to the tools are checked against.
        self._message = message
        self._build_env = build_env
        self._ledger = ledger
        self._runner = runner
        self._workflows = workflows
        self._capabilities = capabilities
        self._notes = notes
        self._schemas = schemas
        self._explore_limits = explore_limits
        self._backoff = backoff
        self.results: list[Result] = []
        # Every run the turn's tools made, a refused input failure's included, in order.
        self.run_ids: list[str] = []
        # Connectors whose service was down when work needing them was refused, in order.
        self.unavailable: list[str] = []
        self._input_failures: Counter[str] = Counter()
        self._resumed: set[str] = set()
        self._lookup_env: RunEnv | None = None
        self._lookup_lock = asyncio.Lock()
        self.lookup_handles: list[str] = []
        self.lookup_disclosures: list[str] = []

    @property
    def lookup_run_id(self) -> str | None:
        return None if self._lookup_env is None else self._lookup_env.run_id

    def tools(self) -> list[BaseTool]:
        bodies: dict[str, Callable[..., Awaitable[str]]] = {
            "run_workflow": self.run_workflow,
            "resume_run": self.resume_run,
            "explore": self.explore,
        }
        descriptions = {
            "run_workflow": RUN_WORKFLOW_DESCRIPTION,
            "resume_run": RESUME_DESCRIPTION,
            "explore": EXPLORE_DESCRIPTION,
        }
        for tool, name in LOOKUPS.items():
            capability = self._capabilities.get(name)
            bodies[tool] = self._lookup(capability)
            descriptions[tool] = capability.description + LOOKUP_NOTES[tool]
        return [
            StructuredTool.from_function(
                coroutine=self._guarded(tool, bodies[tool]),
                name=tool,
                description=descriptions[tool],
                args_schema=self._schemas[tool],
            )
            for tool in TOOL_NAMES
        ]

    def _guarded(self, tool: str, body: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        async def run(**args: Any) -> str:
            try:
                return await body(**args)
            except (asyncio.CancelledError, GraphBubbleUp):
                raise
            except Exception as exc:
                # The type name only: an exception's text can carry values or secrets.
                logger.error("supervisor tool %s hit an internal error: %s", tool, type(exc).__name__)
                return _refused(INTERNAL)

        return run

    # run_workflow and resume_run

    async def _input_failure(self, result: Result) -> bool:
        """Whether the run ended on its inputs before any step ran: a bad
        input, or required inputs missing."""
        if result.status == "needs_input":
            return result.question is not None and result.question.get("kind") == "inputs"
        if result.status != "failed" or result.reason == INTERNAL_REASON:
            return False
        entries = await self.context.storage.traces.entries(self.context.identity.user_id, result.trace_id)
        if any(not (entry.step or "").startswith(input_step_id("")) for entry in entries):
            return False
        failed = [entry for entry in entries if entry.status == "failed"]
        return all(entry.failure_category == FailureCategory.BAD_INPUT for entry in failed)

    def _produced(self, result: Result) -> None:
        self.results.append(result)
        self.run_ids.append(result.trace_id)

    def _split(self, places: Sequence[Any]) -> str | None:
        """The refusal for the first place that is one part of a combined place, or None."""
        for place in places:
            if isinstance(place, str) and part_of_combined_place(place, self._message):
                quoted = " ".join(place.split())[:_QUOTED_PLACE_CHARS]
                return SPLIT_PLACE.format(place=quoted)
        return None

    def _down(self, needed: Iterable[str | None]) -> str | None:
        """The refusal when a connector in ``needed`` is unreachable, or None; the connector is noted."""
        down = sorted({c for c in needed if c is not None and not self.context.reachable(c)})
        if not down:
            return None
        self.unavailable.extend(c for c in down if c not in self.unavailable)
        return _refused(" ".join(unavailable_wording(c) for c in down))

    def _input_refusal(self, name: str, reason: str) -> str:
        self._input_failures[name] += 1
        return _refused(STOP_RETRYING if self._input_failures[name] >= 2 else reason)

    async def run_workflow(self, name: Any = None, inputs: Any = None, **extra: Any) -> str:
        if not isinstance(name, str) or name not in self._workflows:
            return _refused(UNKNOWN_WORKFLOW)
        if self._input_failures[name] >= 2:
            return _refused(STOP_RETRYING)
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, Mapping) or extra:
            return self._input_refusal(name, "inputs is an object of the workflow's inputs by name.")
        spec = self._runner.spec(name, self._workflows[name])
        down = self._down(spec.connectors(self._capabilities))
        if down is not None:
            return down
        split = self._split([inputs.get(key) for key, declared in spec.inputs.items() if declared.type == "area"])
        if split is not None:
            return self._input_refusal(name, split)
        try:
            result = await self._runner.start(self.context, self._build_env, name, self._workflows[name], inputs)
        except UnknownWorkflow:
            return self._input_refusal(name, UNKNOWN_WORKFLOW)
        except NotRunnable as refused:
            return self._input_refusal(name, NOT_RUNNABLE.format(reason=refused.reason))
        if await self._input_failure(result):
            self.run_ids.append(result.trace_id)
            return self._input_refusal(name, result.reason or UNKNOWN_WORKFLOW)
        self._produced(result)
        return result_view(result)

    async def resume_run(self, run_id: Any = None, **extra: Any) -> str:
        open_run = next((run for run in self._ledger.open_runs if run.run_id == run_id), None)
        if (
            extra or open_run is None or not open_run.workflow or open_run.status != "waiting"
            or open_run.job_state not in RESUMABLE or open_run.run_id in self._resumed
        ):
            return _refused(NOT_READY)
        self._resumed.add(open_run.run_id)
        env = await self._build_env(self.context, open_run.run_id)
        try:
            result = await self._runner.resume(env)
        except (RunNotResumable, RunNotFound):
            return _refused(NOT_READY)
        self._produced(result)
        return result_view(result)

    # explore

    async def explore(self, tasks: Any = None, **extra: Any) -> str:
        if extra or not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
            return _refused(TOO_MANY_TASKS)
        checked: list[ExploreTask] = []
        problems: list[str] = []
        for number, raw in enumerate(tasks, 1):
            try:
                checked.append(ExploreTask.model_validate(raw))
            except ValidationError as error:
                problems.append(BAD_TASK.format(number=number, problems=_problems(error)))
        if problems:
            return _refused(" ".join(problems) + " Nothing was run.")
        split = self._split([place for task in checked for place in task.places])
        if split is not None:
            return _refused(split + " Nothing was run.")
        down = self._down(
            capability.connector
            for task in checked
            for capability in get_bundle(task.bundle).capabilities(self._capabilities)
            if capability.connector_required
        )
        if down is not None:
            return down
        results = await explore(
            self.context, self._build_env, checked, registry=self._capabilities, notes=self._notes,
            limits=self._explore_limits, backoff=self._backoff,
        )
        for result in results:
            self._produced(result)
        return dumps({"results": [view_of(result) for result in results]})

    # Lookups

    async def lookup_env(self) -> RunEnv:
        """The turn's lookup run, started on first use with mode exploratory."""
        async with self._lookup_lock:
            if self._lookup_env is None:
                identity = self.context.identity
                run_id = new_run_id()
                await self.context.storage.traces.start_run(Run(
                    id=run_id, user_id=identity.user_id, conversation_id=identity.conversation_id,
                    mode=Mode.exploratory(), started=self.context.clock.now(),
                ))
                self._lookup_env = await self._build_env(self.context, run_id)
            return self._lookup_env

    def _lookup(self, capability: Capability[Any]) -> Callable[..., Awaitable[str]]:
        async def run(**params: Any) -> str:
            split = self._split([params.get("place")])
            if split is not None:
                return _refused(split)
            down = self._down([capability.connector] if capability.connector_required else [])
            if down is not None:
                return down
            env = await self.lookup_env()
            try:
                result = await invoke(env, capability, params)
            except StepFailure as failure:
                return dumps(failure_view(failure, retry_helps(capability, failure.category)))
            if result.handle is not None and result.handle not in self.lookup_handles:
                self.lookup_handles.append(result.handle)
            self.lookup_disclosures.extend(d for d in result.disclosures if d not in self.lookup_disclosures)
            view = capability_view(result)
            if result.question is not None:
                # The supervisor asks the user itself, so it is shown the question.
                view["question"] = result.question.model_dump(mode="json")
            return dumps(view)

        return run

    def lookup_result(self) -> Result | None:
        """The lookup run's result, once the run is started."""
        if self._lookup_env is None:
            return None
        return Result(
            status="done", mode=Mode.exploratory(), handles=tuple(self.lookup_handles),
            disclosures=tuple(self.lookup_disclosures), trace_id=self._lookup_env.run_id,
        )

    async def close(self) -> Result | None:
        """Save the lookup run's result, then finish the run as done. Failures
        are logged, never raised, so the turn's answer is kept."""
        result = self.lookup_result()
        if result is None:
            return None
        user_id = self.context.identity.user_id
        try:
            await self.context.storage.results.put(user_id, result)
        except Exception as exc:
            logger.error("the result of lookup run %s could not be saved: %s", result.trace_id, type(exc).__name__)
        try:
            await self.context.storage.traces.finish_run(user_id, result.trace_id, "done", self.context.clock.now())
        except Exception as exc:
            logger.error("lookup run %s could not be finished: %s", result.trace_id, type(exc).__name__)
        return result
