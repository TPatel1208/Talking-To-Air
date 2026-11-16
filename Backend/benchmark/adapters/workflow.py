"""Runs benchmark cases through the agent harness's workflows, with no model.

A case that names a workflow is run directly through ``WorkflowRunner`` on the
benchmark's fake data boundary: the earthdata MCP fake behind the app's own
session, EPA AQS behind a capturing fetch, memory stores and an in-memory
checkpointer. The ``Result`` envelope is turned into an ``Outcome`` the
existing checks score. Nothing here turns a prompt into inputs; the case does.
"""
from __future__ import annotations

import re
import time
from typing import Any

from benchmark.adapters.harness_boundary import (  # noqa: F401  (re-exported)
    AQS_EMAIL,
    AQS_KEY,
    CREDENTIAL_STRINGS,
    EARTHDATA_SECRET,
    PACKS_ROOT,
    USER_ID,
    harness_boundary,
)
from benchmark.cases import BenchmarkCase
from benchmark.outcome import Outcome, handles_from_calls


class NotCovered(ValueError):
    """The case names no workflow, so this adapter does not run it."""


_MISSING_CONNECTOR = re.compile(r"^This workflow needs the .+ connectors?\.")


def _misconfigured(result: Any) -> bool:
    """Whether a needs_input result means the adapter left out a connector or an input.

    Any other needs_input, such as a refused area, is the workflow's own answer.
    """
    question = result.question
    if isinstance(question, dict) and question.get("kind") == "inputs":
        return True
    return bool(_MISSING_CONNECTOR.match(result.reason or ""))


def _question_lines(question: Any) -> list[str]:
    """The question's prompt and its options, as the user would read them."""
    if not isinstance(question, dict) or not question.get("prompt"):
        return []
    lines = [str(question["prompt"])]
    options = question.get("options") or []
    if options:
        lines.append("Options: " + ", ".join(str(option) for option in options))
    return lines


def compose_answer(result: Any) -> str:
    """The envelope's own wording as text: mode, findings, disclosures, gaps and, unless done, the reason and any question."""
    lines = [result.mode.label()]
    for finding in result.findings:
        unit = f" {finding.unit}" if finding.unit else ""
        lines.append(f"{finding.name}: {finding.value}{unit}")
    lines.extend(result.disclosures)
    lines.extend(f"{gap.step}: {gap.reason}" for gap in result.gaps)
    if result.status != "done" and result.reason:
        lines.append(result.reason)
    if result.status == "needs_input":
        lines.extend(_question_lines(result.question))
    return "\n".join(lines)


def stopped_reason(result: Any, case: BenchmarkCase, subject: str) -> str | None:
    """Why a run stopped without finishing, or None if it reached an answer, a refusal the case expects or a failure.

    A question counts as an answer only for a case that expects one.
    """
    reason = result.reason or "no reason given"
    if result.status == "incomplete":
        return f"{subject} ran out of its budget before it finished: {reason}"
    if result.status == "waiting":
        return f"{subject} was still waiting on a retrieval: {reason}"
    if result.status == "cancelled":
        return f"{subject} was cancelled: {reason}"
    if result.status == "needs_input" and not case.expects_question:
        lines = _question_lines(result.question)
        return f"{subject} stopped to ask: {' '.join(lines) if lines else reason}"
    return None


class WorkflowAdapter:
    """Runs the case's workflow and reports what crossed the boundary and what the user would be shown."""

    def __init__(self) -> None:
        self._catalogue: Any = None
        self._registry: Any = None

    @staticmethod
    def covers(case: BenchmarkCase) -> bool:
        return case.workflow is not None

    def _load(self) -> tuple[Any, Any]:
        if self._catalogue is None:
            from tta_backend.datasets.registry import load_registry
            from tta_backend.harness.capabilities.catalogue import build_registry
            from tta_backend.harness.workflows.loader import load_packs

            self._registry = build_registry()
            self._catalogue = load_packs(PACKS_ROOT, self._registry, load_registry())
        return self._registry, self._catalogue

    async def run(self, case: BenchmarkCase, volume: Any) -> Outcome:
        if case.workflow is None:
            raise NotCovered(case.id)

        from langgraph.checkpoint.memory import InMemorySaver

        from tta_backend.harness.workflows import WorkflowRunner
        from tta_backend.harness.workflows.compile import NotRunnable
        from tta_backend.harness.workflows.loader import UnknownWorkflow

        registry, catalogue = self._load()
        result = None
        error: str | None = None
        started = time.monotonic()
        async with harness_boundary(case) as boundary:
            runner = WorkflowRunner(registry, catalogue, InMemorySaver())
            try:
                version = catalogue.latest(case.workflow).version
                started = time.monotonic()
                result = await runner.start(
                    boundary.context, boundary.build_env, case.workflow, version,
                    case.inputs_for(boundary.run_day),
                )
            except (UnknownWorkflow, NotRunnable) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                # The type name only: an exception's text can carry values or secrets.
                error = f"the workflow run raised {type(exc).__name__}"
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
        )
        if result is None:
            outcome.error = error
            return outcome
        outcome.mode = result.mode.label()
        outcome.findings = [finding.model_dump(mode="json") for finding in result.findings]
        outcome.answer_text = compose_answer(result)
        if result.status == "failed":
            outcome.error = result.reason or "the workflow failed"
        elif result.status == "needs_input" and _misconfigured(result):
            outcome.error = f"the adapter did not supply what the workflow needs: {result.reason}"
        stopped = stopped_reason(result, case, "the workflow")
        if stopped is not None and outcome.error is None:
            outcome.unfinished = stopped
        outcome.done = stopped is None and outcome.error is None
        return outcome
