"""Runs benchmark cases through the agent harness's exploration specialists, with a real model.

A case that names a bundle runs as one ``ExploreTask`` on the same fake data
boundary the workflow adapter uses, with the app's models behind its gateway.
The case prompt is the task's objective, verbatim. The ``Result`` envelope
becomes an ``Outcome`` the existing checks score, and the model tokens the run
used come from its explore trace entry.
"""
from __future__ import annotations

import time
from typing import Any

from benchmark.adapters.harness_boundary import PACKS_ROOT, USER_ID, harness_boundary
from benchmark.adapters.workflow import compose_answer, stopped_reason
from benchmark.cases import BenchmarkCase
from benchmark.outcome import Outcome, handles_from_calls

_AGENT = "explore"


class NotCovered(ValueError):
    """The case names no bundle, so this adapter does not run it."""


def compose_exploration_answer(result: Any) -> str:
    """The workflow wording, with the model's interpretation after the mode label."""
    lines = compose_answer(result).split("\n")
    if result.interpretation:
        lines.insert(1, result.interpretation)
    return "\n".join(lines)


def tokens_from_trace(trace: list[Any]) -> dict[tuple[str, str, str], int]:
    """Input, output and cache tokens per model from the explore entries; zero counts are left out.

    Input already includes the cache reads and writes, which are recorded as
    their own kinds so they are priced at their own rates.
    """
    tokens: dict[tuple[str, str, str], int] = {}
    for entry in trace:
        if entry.kind != "explore" or not isinstance(entry.result, dict):
            continue
        model = entry.result.get("model") or "none"
        for kind, key in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("cache_read", "cache_read_tokens"),
            ("cache_creation", "cache_creation_tokens"),
        ):
            count = int(entry.result.get(key) or 0)
            if count:
                tokens[(model, _AGENT, kind)] = tokens.get((model, _AGENT, kind), 0) + count
    return tokens


class ExploreAdapter:
    """Runs the case's prompt as an exploration task and reports what crossed the boundary and what the user would see.

    ``models`` replaces the app's models, for tests with a scripted model.
    """

    def __init__(self, models: Any = None, limits: Any = None) -> None:
        self._models = models
        self._limits = limits
        self._registry: Any = None
        self._notes: Any = None

    @staticmethod
    def covers(case: BenchmarkCase) -> bool:
        return case.bundle is not None

    def _load(self) -> tuple[Any, Any]:
        if self._registry is None:
            from tta_backend.datasets.registry import load_registry
            from tta_backend.harness.capabilities.catalogue import build_registry
            from tta_backend.harness.knowledge import load_notes

            self._registry = build_registry()
            self._notes = load_notes(PACKS_ROOT, load_registry())
        return self._registry, self._notes

    def _models_for_run(self) -> Any:
        if self._models is not None:
            return self._models
        from tta_backend.infrastructure.settings import get_settings
        from tta_backend.services.harness_models import AppModels

        return AppModels(get_settings())

    async def run(self, case: BenchmarkCase, volume: Any) -> Outcome:
        if case.bundle is None:
            raise NotCovered(case.id)

        from tta_backend.harness.models import DEFAULT_BACKOFF
        from tta_backend.harness.specialists import ExploreTask, explore

        registry, notes = self._load()
        result = None
        error: str | None = None
        started = time.monotonic()
        async with harness_boundary(case, models=self._models_for_run()) as boundary:
            try:
                task = ExploreTask(objective=case.prompt, bundle=case.bundle, places=case.explore_places)
                started = time.monotonic()
                [result] = await explore(
                    boundary.context, boundary.build_env, [task],
                    registry=registry, notes=notes, limits=self._limits, backoff=DEFAULT_BACKOFF,
                )
            except Exception as exc:
                # The type name only: an exception's text can carry values or secrets.
                error = f"the exploration raised {type(exc).__name__}"
            elapsed_seconds = time.monotonic() - started
            trace = (
                await boundary.stores.traces.entries(USER_ID, result.trace_id) if result is not None else []
            )

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
        if result is None:
            outcome.error = error
            return outcome
        outcome.mode = result.mode.label()
        outcome.findings = [finding.model_dump(mode="json") for finding in result.findings]
        outcome.answer_text = compose_exploration_answer(result)
        if result.status == "failed":
            outcome.error = result.reason or "the exploration failed"
        stopped = stopped_reason(result, case, "the exploration")
        if stopped is not None:
            outcome.unfinished = stopped
        outcome.done = stopped is None and outcome.error is None
        return outcome
