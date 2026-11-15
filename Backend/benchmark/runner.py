"""Running the suite against a system adapter."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from benchmark.cases import BenchmarkCase, build_cases
from benchmark.outcome import Outcome
from benchmark.scoring import CaseResult, score


class SystemAdapter(Protocol):
    async def run(self, case: BenchmarkCase, volume) -> Outcome: ...


class RateLimitDetected(RuntimeError):
    """Provider retry or 429 evidence was logged during a single-user run."""


# Groq's client logs retries on the "groq" logger and the OpenAI client (which
# Portkey goes through) on "openai", rather than raising; httpx logs the raw
# response line, which carries the status code; Google's genai
# client logs its retry backoff on "google_genai", with the APIError's
# "429 RESOURCE_EXHAUSTED" text. Matching those loggers is cruder than
# instrumenting the clients but adds nothing to production.
_RATE_LIMIT_LOG_PATTERN = re.compile(r"retrying request|\b429\b|too many requests|resource_exhausted", re.I)


class _RateLimitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.matches: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if _RATE_LIMIT_LOG_PATTERN.search(message):
            self.matches.append(message)


@contextmanager
def capture_rate_limit_evidence():
    """Watch the groq, openai, httpx and google_genai loggers for retry or 429 evidence.

    Yields the handler; ``handler.matches`` lists what it saw. One user at a
    time is the bar, so any evidence means rate-limit pressure.
    """
    handler = _RateLimitLogHandler()
    watched = [logging.getLogger(name) for name in ("groq", "openai", "httpx", "google_genai")]
    previous_levels = [logger.level for logger in watched]
    for logger in watched:
        # httpx logs its request line at INFO; without lowering the logger's
        # own level, records below the root's never reach the handler.
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        for logger, previous_level in zip(watched, previous_levels, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(previous_level)


async def run_suite(adapter: SystemAdapter, volume) -> list[CaseResult]:
    """Run every case in order and score it; raise if any rate-limit evidence was logged."""
    results = []
    with capture_rate_limit_evidence() as evidence:
        for case in build_cases(volume):
            outcome = await adapter.run(case, volume)
            results.append(score(case, outcome))
    if evidence.matches:
        raise RateLimitDetected(f"provider rate-limit evidence during a single-user run: {evidence.matches}")
    return results


@dataclass
class WorkflowSuiteRun:
    """The scored results of the cases a workflow adapter covers, and the ids of those it does not."""

    results: list[CaseResult] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)


async def run_workflow_suite(adapter, volume) -> WorkflowSuiteRun:
    """Run every case that names a workflow and score it; a case without one is listed, never run."""
    run = WorkflowSuiteRun()
    for case in build_cases(volume):
        if not adapter.covers(case):
            run.uncovered.append(case.id)
            continue
        run.results.append(score(case, await adapter.run(case, volume)))
    return run


def starter_prompts_missing_cases(volume) -> list[str]:
    """Ids of starter prompts that name no benchmark case.

    The empty-chat starter list promises what the system can do, so each
    entry must be backed by a case.
    """
    from tta_backend.config.starter_prompts import STARTER_PROMPTS

    case_ids = {case.id for case in build_cases(volume)}
    return [entry["id"] for entry in STARTER_PROMPTS if entry["id"] not in case_ids]


DEFAULT_RESULTS_DIR = "benchmark-results"
WORKFLOW_RESULTS_SUBDIR = "workflow"


async def _main(system: str, results_dir: str) -> int:
    from benchmark.fake_earthdata import HandleVolume
    from benchmark.report import write_report, write_workflow_report
    from benchmark.scoring import format_results_table, format_workflow_table

    with tempfile.TemporaryDirectory() as tmpdir:
        volume = HandleVolume(tmpdir)
        if system == "workflow":
            from benchmark.adapters.workflow import WorkflowAdapter

            run = await run_workflow_suite(WorkflowAdapter(), volume)
            print(format_workflow_table(run.results, run.uncovered))
            folder = write_workflow_report(run, Path(results_dir) / WORKFLOW_RESULTS_SUBDIR)
            print(f"results saved to {folder}")
            return 0
        from benchmark.adapters.current import CurrentSystemAdapter
        from benchmark.scoring import PASS_THRESHOLD

        results = await run_suite(CurrentSystemAdapter(), volume)
    print(format_results_table(results))
    print(f"results saved to {write_report(results, results_dir)}")
    return 0 if sum(1 for result in results if result.passed) >= PASS_THRESHOLD else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.runner", description="Run the benchmark suite.")
    parser.add_argument(
        "--system", choices=("current", "workflow"), default="current",
        help="current: the production chat stream (needs models and Postgres); "
        "workflow: the harness workflows, no model, scored apart from the gate",
    )
    parser.add_argument(
        "--results-dir", default=os.environ.get("BENCHMARK_RESULTS_DIR", DEFAULT_RESULTS_DIR),
        help="where results are saved; the workflow system writes under its own subfolder",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_main(args.system, args.results_dir))


if __name__ == "__main__":
    raise SystemExit(main())
