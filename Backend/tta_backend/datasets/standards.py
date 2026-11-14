"""
datasets/standards.py
=====================
Loads and validates the US EPA air quality standards and AQI breakpoints
from standards.yaml.

Usage
-----
    from tta_backend.datasets.standards import aqi_category, load_standards

    cat = aqi_category("pm25", 40.2)        # 24-hour PM2.5, ug/m3
    if isinstance(cat, AqiBreakpoint):
        print(cat.category, cat.aqi_low, cat.aqi_high)
    print(load_standards()["no2"].naaqs)
"""

from __future__ import annotations

import logging
import math
import pathlib
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

_STANDARDS_PATH = pathlib.Path(__file__).parent / "standards.yaml"


class NaaqsStandard(BaseModel):
    """One primary National Ambient Air Quality Standard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level:            float = Field(gt=0)
    unit:             str
    averaging_period: str
    form:             str


class AqiBreakpoint(BaseModel):
    """One AQI category: its AQI range and the concentration range that maps to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category:           str
    aqi_low:            int = Field(ge=0)
    aqi_high:           int = Field(ge=0)
    concentration_low:  float = Field(ge=0)
    concentration_high: float = Field(ge=0)
    averaging_period:   str

    @model_validator(mode="after")
    def _ranges_ordered(self) -> "AqiBreakpoint":
        if self.aqi_low > self.aqi_high:
            raise ValueError(f"{self.category}: aqi_low must be <= aqi_high")
        if self.concentration_low > self.concentration_high:
            raise ValueError(f"{self.category}: concentration_low must be <= concentration_high")
        return self


class OutOfTable(Enum):
    """A concentration the breakpoint table does not cover."""

    BELOW = "below"    # under the table's lowest breakpoint
    BEYOND = "beyond"  # above the table's highest breakpoint


class PollutantStandards(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name:              str
    unit:              str
    default_averaging: str
    decimals:          int = Field(ge=0, le=6)
    naaqs:             list[NaaqsStandard] = Field(min_length=1)
    breakpoints:       list[AqiBreakpoint] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "PollutantStandards":
        periods = {b.averaging_period for b in self.breakpoints}
        if self.default_averaging not in periods:
            raise ValueError(f"default_averaging {self.default_averaging!r} has no breakpoints")
        for period in periods:
            rows = self.breakpoints_for(period)
            for prev, row in zip(rows, rows[1:], strict=False):
                if row.concentration_low <= prev.concentration_high:
                    raise ValueError(
                        f"{period} breakpoints must ascend without overlap: "
                        f"{row.category} starts at {row.concentration_low}, "
                        f"{prev.category} ends at {prev.concentration_high}"
                    )
                if row.aqi_low <= prev.aqi_high:
                    raise ValueError(
                        f"{period} AQI ranges must ascend without overlap: "
                        f"{row.category} starts at {row.aqi_low}, "
                        f"{prev.category} ends at {prev.aqi_high}"
                    )
        return self

    def breakpoints_for(self, averaging_period: str) -> list[AqiBreakpoint]:
        return [b for b in self.breakpoints if b.averaging_period == averaging_period]


@lru_cache(maxsize=1)
def load_standards(path: str | None = None) -> dict[str, PollutantStandards]:
    """
    Load, validate, and cache the standards file.

    Returns a dict mapping pollutant key ('o3', 'pm25', 'pm10', 'no2', 'so2',
    'co') to its standards.

    Raises
    ------
    FileNotFoundError  if the YAML file is missing.
    ValueError         if any pollutant fails validation.
    """
    yaml_path = pathlib.Path(path) if path else _STANDARDS_PATH

    if not yaml_path.exists():
        raise FileNotFoundError(f"Air quality standards not found: {yaml_path}")

    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    standards: dict[str, PollutantStandards] = {}
    errors: list[str] = []

    for key, values in raw.items():
        try:
            standards[key] = PollutantStandards(**values)
        except Exception as exc:
            errors.append(f"  [{key}] {exc}")

    if errors:
        raise ValueError("Air quality standards validation failed:\n" + "\n".join(errors))

    logger.info("Air quality standards loaded: %d pollutants", len(standards))
    return standards


def aqi_category(
    pollutant: str,
    concentration: float,
    averaging_period: Optional[str] = None,
) -> AqiBreakpoint | OutOfTable:
    """
    The AQI category for a concentration, in the pollutant's breakpoint unit.

    The concentration is truncated to the breakpoint precision first, as EPA
    specifies, and a negative concentration is treated as zero.
    ``averaging_period`` defaults to the pollutant's default.

    Returns the matching AqiBreakpoint, or OutOfTable.BELOW for a value under
    the table's lowest breakpoint (ozone 1-hour under 0.125 ppm), or
    OutOfTable.BEYOND for a value above its highest. BEYOND is not the top
    category: where EPA switches period (SO2 1-hour above its top, ozone
    8-hour above 0.200 ppm) the other period's table applies, and past AQI 500
    the index is undefined.

    Raises ValueError for an unknown pollutant or averaging period, or a
    non-finite concentration.
    """
    try:
        info = load_standards()[pollutant]
    except KeyError:
        raise ValueError(f"unknown pollutant {pollutant!r}") from None
    period = averaging_period or info.default_averaging
    rows = info.breakpoints_for(period)
    if not rows:
        raise ValueError(f"{pollutant} has no {period!r} breakpoints")
    if not math.isfinite(concentration):
        raise ValueError(f"concentration must be finite, got {concentration}")

    scale = 10 ** info.decimals
    value = math.floor(max(concentration, 0.0) * scale + 1e-9) / scale
    if value > rows[-1].concentration_high:
        return OutOfTable.BEYOND
    found: AqiBreakpoint | OutOfTable = OutOfTable.BELOW
    for row in rows:
        if row.concentration_low <= value:
            found = row
    return found
