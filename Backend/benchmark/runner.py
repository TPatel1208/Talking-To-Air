"""Running the suite against a system adapter."""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
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


def starter_prompts_missing_cases(volume) -> list[str]:
    """Ids of starter prompts that name no benchmark case.

    The empty-chat starter list promises what the system can do, so each
    entry must be backed by a case.
    """
    from tta_backend.config.starter_prompts import STARTER_PROMPTS

    case_ids = {case.id for case in build_cases(volume)}
    return [entry["id"] for entry in STARTER_PROMPTS if entry["id"] not in case_ids]
