"""
starter_prompts.py
-------------------
The empty-chat's example questions: one backend-owned constant so the
`/capabilities/starters` endpoint, the benchmark's coverage check and the
frontend's empty state cannot drift apart.

Each entry's ``id`` must name a benchmark case (``benchmark/cases.py``). A
starter prompt with no case is a claim the suite cannot back up, so
``benchmark.runner.starter_prompts_missing_cases`` fails the suite if one
drifts. ``category`` spans the workflow types (map, trend, validation,
comparison, ground lookup, discovery) so the breadth of the app is visible at
a glance.
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
        "id": "discovery_no2_dataset",
        "label": "Find an NO2 dataset",
        "prompt": "What NASA datasets are available for NO2 column density over New Jersey?",
        "category": "discovery",
    },
    {
        "id": "plotting_single_map",
        "label": "Plot a pollutant map",
        "prompt": "Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
        "category": "map",
    },
    {
        "id": "plotting_timeseries",
        "label": "See a trend over time",
        "prompt": "Show me how NO2 changed over Newark NJ during January 2024.",
        "category": "trend",
    },
    {
        "id": "ground_validation_tempo_vs_epa",
        "label": "Validate satellite against ground monitors",
        "prompt": "Compare TEMPO NO2 with EPA ground monitors over Newark NJ for the first week of January 2024.",
        "category": "validation",
    },
    {
        "id": "comparison_period_tempo_no2",
        "label": "Compare two time periods",
        "prompt": "Compare TEMPO NO2 over New Jersey between June 2025 and June 2026 — did it change?",
        "category": "comparison",
    },
    {
        "id": "e2e_ground_relative_date",
        "label": "Look up a ground monitor reading",
        "prompt": "What was the NO2 level in Newark, New Jersey yesterday?",
        "category": "ground_lookup",
    },
]
