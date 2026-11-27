"""CSV cells that a spreadsheet opens as data, never as a formula."""
from __future__ import annotations

import math
from typing import Any

# A spreadsheet reads a cell starting with one of these as a formula.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def safe_cell(value: Any) -> Any:
    """``value`` with a single quote in front when it is text a spreadsheet
    would take for a formula; numbers, and text that reads as one, are left alone."""
    if isinstance(value, str) and value.startswith(FORMULA_STARTS) and not _is_number(value):
        return "'" + value
    return value


__all__ = ["FORMULA_STARTS", "safe_cell"]
