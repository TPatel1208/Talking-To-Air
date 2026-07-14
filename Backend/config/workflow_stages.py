"""
config/workflow_stages.py
===========================
PRD T19: the closed vocabulary of workflow stage keys a satellite request
passes through. Composites and handle tools tag their ``emit_status`` calls
with one of these via the ``stage`` kwarg (utils/streaming.py) — the
frontend's workflow strip and the eval's stage-sequence assertions both key
off these constants rather than free text, so a renamed status message can
never silently break narration ordering.

Free text stays free text (the human-readable label); only the stage key is
closed.
"""
from __future__ import annotations

STAGE_SEARCH = "search"
STAGE_AOI = "aoi"
STAGE_COVERAGE = "coverage"
STAGE_ESTIMATE = "estimate"
STAGE_SUBMIT = "submit"
STAGE_PROGRESS = "progress"
STAGE_OPEN = "open"
STAGE_RENDER = "render"
STAGE_WORKING = "working"

ALL_STAGES = (
    STAGE_SEARCH,
    STAGE_AOI,
    STAGE_COVERAGE,
    STAGE_ESTIMATE,
    STAGE_SUBMIT,
    STAGE_PROGRESS,
    STAGE_OPEN,
    STAGE_RENDER,
    STAGE_WORKING,
)
