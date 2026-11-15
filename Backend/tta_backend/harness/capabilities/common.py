"""What the operations and render capabilities share: refusals, short text,
plain JSON, a handle's area name and period, and its product's registry facts.

A refusal is a failure with fixed wording filled from observed facts, and
``{"refusal": reason}`` on its trace entry.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import JsonValue

from tta_backend.datasets.registry import load_registry
from tta_backend.domain.earthdata.opening import registry_col_info
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.handles.model import HandleRecord, Period
from tta_backend.harness.trace import FailureCategory, StepFailure

FACT_CHARS = 80
SUMMARY_CHARS = 500
DEFAULT_AREA_NAME = "the area"


def short(text: str, limit: int = SUMMARY_CHARS) -> str:
    """``text`` on one line, cut to ``limit`` characters with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def fact(value: Any) -> str:
    """An observed fact as short text for a refusal's wording."""
    return short(str(value), FACT_CHARS)


def refusal(
    call: Call[Any], wording: str, reason: str, *,
    category: FailureCategory = FailureCategory.BAD_INPUT, **facts: Any,
) -> StepFailure:
    """The failure for ``reason``: ``wording`` filled with the capability's name and ``facts``."""
    text = wording.format(name=call.capability.name, **{k: fact(v) for k, v in facts.items()})
    return StepFailure(category, text, result={"refusal": reason})


def number(value: Any) -> float | None:
    """A plain float, or None for a missing or non-finite value."""
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def plain(value: Any) -> JsonValue:
    """``value`` as plain JSON: numpy scalars unwrapped, non-finite floats as None."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return number(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def region_name(record: HandleRecord) -> str:
    """The name of the area a handle covers, or ``DEFAULT_AREA_NAME``."""
    region = record.descriptor.region
    name = region.get("name") if isinstance(region, dict) else None
    return name if isinstance(name, str) and name else DEFAULT_AREA_NAME


def period_json(period: Period | None) -> dict[str, JsonValue] | None:
    """A recorded period as plain JSON, or None."""
    return period.model_dump() if period is not None else None


def col_info_for(record: HandleRecord) -> dict[str, Any] | None:
    """The registry facts for the handle's product, as the masking and provenance code reads them."""
    cfg = load_registry().get(record.descriptor.product or "")
    return registry_col_info(cfg) if cfg is not None else None
