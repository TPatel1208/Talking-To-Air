"""
starter_prompts.py
-------------------
The empty-chat's example questions: one backend-owned constant so the
`/capabilities/starters` endpoint, the benchmark's coverage check and the
frontend's empty state cannot drift apart.

Each entry shows one workflow from the packs: ``category`` is the workflow's
name, and ``id`` names the benchmark case (``benchmark/cases.py``) that runs
that workflow on that prompt. A starter prompt with no case is a claim the
suite cannot back up, so ``benchmark.runner.starter_prompts_missing_cases``
fails the suite if one drifts, and the starter tests check that the case runs
the workflow the entry names and asks the same question.
"""
from __future__ import annotations

from typing import TypedDict


class StarterPrompt(TypedDict):
    id: str
    label: str
    prompt: str
    category: str


STARTER_PROMPTS: list[StarterPrompt] = [
    {
        "id": "plotting_single_map",
        "label": "Map a pollutant",
        "prompt": "Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
        "category": "map",
    },
    {
        "id": "plotting_timeseries",
        "label": "See a trend over time",
        "prompt": "Show me how NO2 changed over Newark NJ during January 2024.",
        "category": "timeseries",
    },
    {
        "id": "comparison_period_tempo_no2",
        "label": "Compare two time periods",
        "prompt": "Compare TEMPO NO2 over New Jersey between June 2025 and June 2026 — did it change?",
        "category": "period-comparison",
    },
    {
        "id": "e2e_ground_relative_date",
        "label": "Look up ground monitor readings",
        "prompt": "What was the NO2 level in Newark, New Jersey yesterday?",
        "category": "monitor-readings",
    },
]
