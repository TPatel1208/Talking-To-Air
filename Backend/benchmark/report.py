"""Saving a run's results to disk.

Each run gets its own timestamped folder holding the results table and, per
case, everything needed to see why it passed or failed without running it
again: the answer, every call that reached the data services, the checks and
the tokens.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.pricing import estimate_cost
from benchmark.scoring import CaseResult, format_results_table, format_workflow_table

if TYPE_CHECKING:
    from benchmark.runner import WorkflowSuiteRun


def case_record(result: CaseResult) -> dict:
    outcome = result.outcome
    cost = estimate_cost(outcome)
    return {
        "id": result.case.id,
        "category": result.case.category,
        "prompt": result.case.prompt,
        "passed": result.passed,
        "within_budget": result.within_budget,
        "elapsed_seconds": round(outcome.elapsed_seconds, 2),
        "checks": [asdict(check) for check in result.check_results],
        "done": outcome.done,
        "error": outcome.error,
        "answer_text": outcome.answer_text,
        "tool_calls": outcome.tool_calls,
        "stages": outcome.stages,
        "boundary_calls": [asdict(call) for call in outcome.boundary_calls],
        "aqs_calls": [{"endpoint": endpoint, "params": params} for endpoint, params in outcome.aqs_calls],
        "handles": outcome.handles,
        "charts": outcome.charts,
        "artifact_ids": outcome.artifact_ids,
        "tokens": [
            {"model": model, "agent": agent, "kind": kind, "tokens": count}
            for (model, agent, kind), count in sorted(outcome.tokens.items())
        ],
        "est_cost_dollars": None if cost is None else round(cost.dollars, 6),
        "unpriced_models": [] if cost is None else list(cost.unpriced),
    }


def write_report(results: list[CaseResult], directory: str | Path, *, now: datetime | None = None) -> Path:
    """Write ``table.txt`` and ``cases.json`` into a new timestamped folder under ``directory``; return the folder."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(directory) / stamp
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "table.txt").write_text(format_results_table(results) + "\n", encoding="utf-8")
    records = [case_record(result) for result in results]
    (folder / "cases.json").write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    return folder


def write_workflow_report(run: WorkflowSuiteRun, directory: str | Path, *, now: datetime | None = None) -> Path:
    """Write ``table.txt`` and ``cases.json`` for a workflow-adapter run into a new timestamped folder under ``directory``.

    Each record adds the workflow, its mode label and its findings; ``uncovered.json`` lists the cases skipped.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(directory) / stamp
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "table.txt").write_text(format_workflow_table(run.results, run.uncovered) + "\n", encoding="utf-8")
    records = [
        {**case_record(result), "workflow": result.case.workflow, "mode": result.outcome.mode,
         "findings": result.outcome.findings}
        for result in run.results
    ]
    (folder / "cases.json").write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    (folder / "uncovered.json").write_text(json.dumps(run.uncovered, indent=2) + "\n", encoding="utf-8")
    return folder
