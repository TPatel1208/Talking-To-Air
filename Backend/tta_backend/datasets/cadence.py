"""Publishing cadences of registered products, finest first, and the calendar
period each coarse cadence averages over.

A monthly, seasonal or annual granule is one mean over a calendar period.
Seasons are meteorological: DJF, MAM, JJA and SON. DJF starts in December and
is named by that December's year, as the files are, so DJF 2018 runs from
2018-12-01 to 2019-02-28.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

Cadence = Literal["hourly", "daily", "monthly", "seasonal", "annual"]

CADENCES: tuple[Cadence, ...] = ("hourly", "daily", "monthly", "seasonal", "annual")

#: Finer cadences have smaller numbers.
CADENCE_ORDER: dict[str, int] = {cadence: index for index, cadence in enumerate(CADENCES)}

#: Months in one period of each calendar-period cadence.
PERIOD_MONTHS: dict[str, int] = {"monthly": 1, "seasonal": 3, "annual": 12}

# Months a period starts after January, modulo its length: seasons start in
# December, March, June and September.
_PERIOD_OFFSET: dict[str, int] = {"monthly": 0, "seasonal": 1, "annual": 0}

#: Pandas period alias whose periods are each calendar-period cadence's.
#: Quarters ending in November are the meteorological seasons, and a DJF
#: quarter starts on its December.
PANDAS_PERIOD: dict[str, str] = {"monthly": "M", "seasonal": "Q-NOV", "annual": "Y"}

#: Pandas alias a timestamp is floored to for the fixed-length cadences.
PANDAS_FLOOR: dict[str, str] = {"hourly": "h", "daily": "D"}

#: Pandas alias stepping from one bucket start to the next. Months, seasons
#: and years are not fixed durations, so they walk period starts.
PANDAS_STEP: dict[str, str] = {
    "hourly": "h", "daily": "D", "monthly": "MS", "seasonal": "QS-DEC", "annual": "YS",
}


def shift_period(day: date, cadence: str, periods: int) -> date:
    """The first day of the period ``periods`` away from the one holding ``day``."""
    months = PERIOD_MONTHS[cadence]
    index = day.year * 12 + day.month - 1
    start = index - (index + _PERIOD_OFFSET[cadence]) % months + periods * months
    return date(start // 12, start % 12 + 1, 1)


def period_start(day: date, cadence: str) -> date:
    """The first day of the period holding ``day``."""
    return shift_period(day, cadence, 0)


def period_end(day: date, cadence: str) -> date:
    """The last day of the period holding ``day``."""
    return shift_period(day, cadence, 1) - timedelta(days=1)
