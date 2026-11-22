"""Choosing comparison days for an event day: same day type, nearby, never the event's own neighbours."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

DayType = Literal["weekday", "weekend"]

# Candidates come from this many days either side of the event.
WINDOW_DAYS = 28
# The days either side of the event that are never candidates, so the event's own build-up and aftermath stay out.
EXCLUDED_NEIGHBOURS = 1
# The most days proposed; it bounds the retrieval, not the method.
MAX_DAYS = 10


def day_type(day: date) -> DayType:
    """Monday to Friday is a weekday; Saturday and Sunday are the weekend."""
    return "weekend" if day.weekday() >= 5 else "weekday"


def comparison_days(
    event: date,
    *,
    latest: date,
    earliest: date | None = None,
    window: int = WINDOW_DAYS,
    limit: int = MAX_DAYS,
) -> list[date]:
    """Up to ``limit`` days of ``event``'s day type within ``window`` days either side of it,
    from ``earliest`` to ``latest`` inclusive, nearest the event first, returned in date order.

    The event day and the day either side of it are never included. Two days
    equally near the event are taken earlier first.
    """
    if window < 1 or limit < 1:
        raise ValueError("window and limit must be at least 1")
    wanted = day_type(event)
    found: list[tuple[int, date]] = []
    for offset in range(EXCLUDED_NEIGHBOURS + 1, window + 1):
        for day in (event - timedelta(days=offset), event + timedelta(days=offset)):
            if day > latest or (earliest is not None and day < earliest):
                continue
            if day_type(day) == wanted:
                found.append((offset, day))
    nearest = sorted(found)[:limit]
    return sorted(day for _, day in nearest)


DropReason = Literal["event_day", "next_to_event", "after_latest", "before_earliest"]


def usable_days(
    event: date, days: list[date], *, latest: date, earliest: date | None = None,
) -> tuple[list[date], list[tuple[date, DropReason]]]:
    """The days of ``days`` a baseline for ``event`` may use, in date order, and each other day with why it
    was left out: the event day, the day either side of it, a day after ``latest`` or before ``earliest``.

    Nothing else is judged: a day the user chose of another type or further away is theirs to choose.
    """
    kept: list[date] = []
    dropped: list[tuple[date, DropReason]] = []
    for day in sorted(set(days)):
        offset = abs((day - event).days)
        if offset == 0:
            dropped.append((day, "event_day"))
        elif offset <= EXCLUDED_NEIGHBOURS:
            dropped.append((day, "next_to_event"))
        elif day > latest:
            dropped.append((day, "after_latest"))
        elif earliest is not None and day < earliest:
            dropped.append((day, "before_earliest"))
        else:
            kept.append(day)
    return kept, dropped
