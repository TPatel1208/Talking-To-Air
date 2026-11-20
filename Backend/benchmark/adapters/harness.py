"""Runs benchmark cases through the new supervisor end to end, with real models.

Every case's prompt goes in verbatim as the one user message of a fresh
conversation on the fake data boundary the other harness adapters use. The
supervisor picks the workflow or the exploration and writes the answer; the
``Turn`` and the runs' trace entries become the ``Outcome`` the checks score.
"""
from __future__ import annotations

import dataclasses
import re
import time
import uuid
from typing import Any

from benchmark.adapters.explore import tokens_from_trace
from benchmark.adapters.harness_boundary import PACKS_ROOT, USER_ID, harness_boundary
from benchmark.adapters.workflow import stopped_reason
from benchmark.cases import BenchmarkCase
from benchmark.outcome import Outcome, handles_from_calls

_SUPERVISOR = "supervisor"
_SOURCES_HEADER = re.compile(r"\A\s*Data sources:[ \t]*([^\n]+)")


def sources_in_header(answer_text: str) -> set[str] | None:
    """The sources named in the answer's opening "Data sources:" line, or None without one."""
    match = _SOURCES_HEADER.match(answer_text)
    if match is None:
        return None
    return {source.strip() for source in match.group(1).split("·") if source.strip()}


def supervisor_tokens(turn: Any, model: str) -> dict[tuple[str, str, str], int]:
    """The supervisor's own input, output and cache tokens from the turn; zero counts are left out."""
    counts = (
        ("input", turn.input_tokens),
        ("output", turn.output_tokens),
        ("cache_read", turn.cache_read_tokens),
        ("cache_creation", turn.cache_creation_tokens),
    )
    return {(model, _SUPERVISOR, kind): count for kind, count in counts if count}


def mode_label(results: Any) -> str | None:
    """The first result's mode label, or the distinct labels joined when the results differ."""
    labels = list(dict.fromkeys(result.mode.label() for result in results))
    return "; ".join(labels) if labels else None


class HarnessAdapter:
    """Runs the case's prompt through the supervisor and reports what crossed the boundary and what the user would see.

    ``models`` replaces the app's models, for tests with scripted ones.
    """

    def __init__(self, models: Any = None, limits: Any = None, explore_limits: Any = None) -> None:
        self._models = models
        self._limits = limits
        self._explore_limits = explore_limits
        self._registry: Any = None
        self._catalogue: Any = None
        self._notes: Any = None

    @staticmethod
    def covers(case: BenchmarkCase) -> bool:
        return True

    def _load(self) -> tuple[Any, Any, Any]:
        if self._registry is None:
            from tta_backend.datasets.registry import load_registry
            from tta_backend.harness.capabilities.catalogue import build_registry
            from tta_backend.harness.knowledge import load_notes
            from tta_backend.harness.workflows.loader import load_packs

            datasets = load_registry()
            self._registry = build_registry()
            self._catalogue = load_packs(PACKS_ROOT, self._registry, datasets)
            self._notes = load_notes(PACKS_ROOT, datasets)
        return self._registry, self._catalogue, self._notes

    def _models_for_run(self) -> Any:
        if self._models is not None:
            return self._models
        from tta_backend.infrastructure.settings import get_settings
        from tta_backend.services.harness_models import AppModels

        return AppModels(get_settings())

    async def run(self, case: BenchmarkCase, volume: Any) -> Outcome:
        from langgraph.checkpoint.memory import InMemorySaver

        from tta_backend.harness.models import DEFAULT_BACKOFF
        from tta_backend.harness.run_context import Identity
        from tta_backend.harness.supervisor import Supervisor
        from tta_backend.harness.workflows import WorkflowRunner

        registry, catalogue, notes = self._load()
        models = self._models_for_run()
        supervisor_choice = models.for_role(_SUPERVISOR)
        supervisor_model = supervisor_choice.primary.name if supervisor_choice is not None else "none"
        turn = None
        error: str | None = None
        started = time.monotonic()
        async with harness_boundary(case, models=models) as boundary:
            supervisor = Supervisor(
                capabilities=registry, catalogue=catalogue, runner=WorkflowRunner(registry, catalogue, InMemorySaver()),
                notes=notes, checkpointer=InMemorySaver(), models=models, limits=self._limits,
                explore_limits=self._explore_limits, backoff=DEFAULT_BACKOFF,
            )
            # A fresh conversation per case, so no turn sees another's checkpoint or ledger.
            context = dataclasses.replace(
                boundary.context,
                identity=Identity(user_id=USER_ID, conversation_id=f"benchmark-{case.id}-{uuid.uuid4().hex[:8]}"),
            )
            try:
                started = time.monotonic()
                turn = await supervisor.turn(context, boundary.build_env, boundary.earthdata, case.prompt)
            except Exception as exc:
                # The type name only: an exception's text can carry values or secrets.
                error = f"the supervisor turn raised {type(exc).__name__}"
            elapsed_seconds = time.monotonic() - started
            trace = []
            if turn is not None:
                for run_id in turn.run_ids:
                    trace.extend(await boundary.stores.traces.entries(USER_ID, run_id))

        outcome = Outcome(
            boundary_calls=list(boundary.boundary_calls),
            aqs_calls=list(boundary.aqs_calls),
            handles=handles_from_calls(boundary.boundary_calls),
            charts=[dict(chart) for chart in boundary.events.charts],
            artifact_ids=list(boundary.artifacts.stored),
            stages=[event["stage"] for event in boundary.events.progress_events if event["stage"]],
            clock=boundary.run_day,
            clock_end=boundary.clock.today(),
            elapsed_seconds=elapsed_seconds,
            tool_calls=[entry.capability for entry in trace if entry.capability and entry.status != "skipped"],
            tokens=tokens_from_trace(trace),
        )
        if turn is None:
            outcome.error = error
            return outcome
        for key, count in supervisor_tokens(turn, supervisor_model).items():
            outcome.tokens[key] = outcome.tokens.get(key, 0) + count
        outcome.answer_text = turn.text
        outcome.sources_header = sources_in_header(turn.text)
        outcome.mode = mode_label(turn.results)
        outcome.findings = [finding.model_dump(mode="json") for result in turn.results for finding in result.findings]
        if turn.status == "failed":
            outcome.error = "the supervisor turn failed: the model could not be reached or hit an internal error"
        elif turn.status == "limit":
            outcome.error = "the supervisor turn stopped at its model-call limit"
        stopped = next(
            (reason for result in turn.results if (reason := stopped_reason(result, case, "the run")) is not None),
            None,
        )
        if stopped is not None:
            outcome.unfinished = stopped
        outcome.done = stopped is None and outcome.error is None
        return outcome
