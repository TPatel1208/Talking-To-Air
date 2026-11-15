"""How each pollutant in datasets/standards.yaml is asked for and read in AQS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tta_backend.datasets.standards import NaaqsStandard, load_standards

Pollutant = Literal["o3", "pm25", "pm10", "no2", "so2", "co"]


@dataclass(frozen=True)
class AqsStandard:
    """One NAAQS as AQS reports it: the ``pollutant_standard`` its daily,
    quarterly and annual rows carry, and the daily field compared with it."""

    averaging_period: str
    name: str
    daily_field: str
    statistic: str


@dataclass(frozen=True)
class AqsPollutant:
    key: Pollutant
    param_code: str
    standards: tuple[AqsStandard, ...]

    def standard(self, averaging_period: str | None = None) -> AqsStandard:
        """The standard for ``averaging_period``; the first listed when None."""
        if averaging_period is None:
            return self.standards[0]
        for standard in self.standards:
            if standard.averaging_period == averaging_period:
                return standard
        raise KeyError(averaging_period)

    def naaqs(self, averaging_period: str) -> NaaqsStandard:
        """The NAAQS level and form from datasets/standards.yaml."""
        for entry in load_standards()[self.key].naaqs:
            if entry.averaging_period == averaging_period:
                return entry
        raise KeyError(averaging_period)

    @property
    def name(self) -> str:
        return load_standards()[self.key].name


_MAX_1H = "daily maximum 1-hour value"
_MAX_8H = "daily maximum 8-hour average"
_MEAN_24H = "24-hour mean"

# The short-term standard is listed first: it is the default.
POLLUTANTS: dict[str, AqsPollutant] = {
    "o3": AqsPollutant("o3", "44201", (
        AqsStandard("8-hour", "Ozone 8-hour 2015", "first_max_value", _MAX_8H),
    )),
    "pm25": AqsPollutant("pm25", "88101", (
        AqsStandard("24-hour", "PM25 24-hour 2024", "arithmetic_mean", _MEAN_24H),
        AqsStandard("annual", "PM25 Annual 2024", "arithmetic_mean", _MEAN_24H),
    )),
    "pm10": AqsPollutant("pm10", "81102", (
        AqsStandard("24-hour", "PM10 24-hour 2006", "arithmetic_mean", _MEAN_24H),
    )),
    "no2": AqsPollutant("no2", "42602", (
        AqsStandard("1-hour", "NO2 1-hour 2010", "first_max_value", _MAX_1H),
        AqsStandard("annual", "NO2 Annual 1971", "arithmetic_mean", _MEAN_24H),
    )),
    "so2": AqsPollutant("so2", "42401", (
        AqsStandard("1-hour", "SO2 1-hour 2010", "first_max_value", _MAX_1H),
    )),
    "co": AqsPollutant("co", "42101", (
        AqsStandard("8-hour", "CO 8-hour 1971", "first_max_value", _MAX_8H),
        AqsStandard("1-hour", "CO 1-hour 1971", "first_max_value", _MAX_1H),
    )),
}


def convert(value: float, from_unit: str | None, to_unit: str) -> float | None:
    """``value`` in ``to_unit``; None when the units are not the same quantity."""
    if from_unit == to_unit:
        return value
    if (from_unit, to_unit) == ("ppm", "ppb"):
        return value * 1000.0
    if (from_unit, to_unit) == ("ppb", "ppm"):
        return value / 1000.0
    return None
