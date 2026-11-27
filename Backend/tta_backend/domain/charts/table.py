"""A DataFrame as the rows of a table artifact: plain JSON values, capped."""
from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd


def json_cell(value: Any) -> Any:
    """A cell as plain JSON: times as ISO text, missing and non-finite numbers as None."""
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp | datetime.datetime | datetime.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, list | tuple | np.ndarray):
        return [json_cell(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_cell(v) for k, v in value.items()}
    return str(value)


def table_rows(frame: pd.DataFrame, max_rows: int) -> tuple[list[str], list[dict[str, Any]]]:
    """The column names and the first ``max_rows`` rows of ``frame``."""
    if max_rows < 1:
        raise ValueError("max_rows is at least 1")
    columns = [str(column) for column in frame.columns]
    head = frame.head(max_rows)
    rows = [
        {name: json_cell(value) for name, value in zip(columns, record, strict=True)}
        for record in head.itertuples(index=False, name=None)
    ]
    return columns, rows
