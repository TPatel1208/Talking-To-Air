"""Proposing comparison days for an event day, and checking the days a user confirmed, by date arithmetic alone;
nothing is retrieved."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Any

from pydantic import Field, JsonValue

from tta_backend.datasets.registry import CollectionConfig, load_registry
from tta_backend.domain.analysis.dates import (
    MAX_DAYS,
    WINDOW_DAYS,
    DropReason,
    comparison_days,
    day_type,
    usable_days,
)
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params
from tta_backend.harness.capabilities.failures import UNKNOWN_PRODUCT, failure
from tta_backend.harness.capabilities.operations.common import insufficient
from tta_backend.harness.capabilities.operations.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult

__all__ = ["CAPABILITIES", "GROUP", "check_comparison_days", "select_comparison_days"]

NO_DAYS = (
    "No comparison days: no {day_type} within {window} days either side of {event}, other than the event day "
    "and the day either side of it, falls between {earliest} and {latest}."
)


class SelectComparisonDaysParams(Params):
    event_date: date = Field(description="The event day the comparison days are for.")
    product: str | None = Field(
        default=None, max_length=100,
        description="The registered product the days will be retrieved from; days outside its published record are left out.",
    )


def _bounds(cfg: CollectionConfig | None, today: date) -> tuple[date | None, date]:
    """The first and last day ``cfg`` has published data for by ``today``; with no product, up to today."""
    latest = today
    earliest: date | None = None
    if cfg is None:
        return earliest, latest
    if cfg.latency_days is not None:
        latest = min(latest, today - timedelta(days=cfg.latency_days))
    if cfg.record_end is not None:
        latest = min(latest, cfg.record_end)
    if cfg.record_start is not None:
        earliest = cfg.record_start
    if cfg.retention_days is not None:
        oldest = today - timedelta(days=cfg.retention_days)
        earliest = oldest if earliest is None else max(earliest, oldest)
    return earliest, latest


def _product(call: Call[Any], key: str | None) -> CollectionConfig | None:
    """The registry entry for ``key``; an unknown key is refused."""
    if key is None:
        return None
    registry = load_registry()
    if key not in registry:
        raise failure(UNKNOWN_PRODUCT, name=call.capability.name, keys=", ".join(sorted(registry)))
    return registry[key]


async def _select(call: Call[SelectComparisonDaysParams]) -> CapabilityResult:
    p = call.params
    earliest, latest = _bounds(_product(call, p.product), call.today)
    kind = day_type(p.event_date)
    days = comparison_days(p.event_date, latest=latest, earliest=earliest)
    if not days:
        return insufficient("the comparison days", situation="no_comparison_days", reason=NO_DAYS.format(
            day_type=kind, window=WINDOW_DAYS, event=p.event_date.isoformat(),
            earliest=earliest.isoformat() if earliest is not None else "the start of the record",
            latest=latest.isoformat(),
        ))
    values: dict[str, JsonValue] = {
        "days": [day.isoformat() for day in days],
        "event_date": p.event_date.isoformat(),
        "day_type": kind,
        "window_days": WINDOW_DAYS,
        "limit": MAX_DAYS,
        "latest": latest.isoformat(),
        "selection": f"{kind}s within {WINDOW_DAYS} days either side of {p.event_date.isoformat()}, nearest first",
    }
    # No product is named on the result: the days are chosen for it, not the product.
    return call.done(values=values)


select_comparison_days = Capability(
    name="operations.select_comparison_days",
    description=(
        "Propose comparison days for an event day: days of the same type (weekday or weekend) within 4 weeks "
        "either side, at most 10, nearest first; nothing is retrieved."
    ),
    params=SelectComparisonDaysParams,
    output_kind=None,
    label="Choosing comparison days",
    method=(
        "Up to {limit} {day_type}s within {window_days} days either side of {event_date} were proposed, nearest "
        "the event first, leaving out the event day, the day either side of it and any day after {latest}; "
        "the cap of {limit} days bounds the retrieval and is not part of the method."
    ),
    retrieves=False,
    retry_safe=True,
    # The days are put to the user by a workflow's question, which an exploration cannot ask.
    workflow_only=True,
    run=_select,
)


# check_comparison_days

# Why a confirmed day was left out, filled from observed facts only.
DROPPED: dict[DropReason, str] = {
    "event_day": "{day} was left out of the baseline: it is the event day.",
    "next_to_event": "{day} was left out of the baseline: it is next to the event day, {event}.",
    "after_latest": "{day} was left out of the baseline: it is after {latest}, the last day {source} has published.",
    "before_earliest": "{day} was left out of the baseline: it is before {earliest}, the first day {source} holds.",
}


class CheckComparisonDaysParams(Params):
    event_date: date = Field(description="The event day the comparison days are for.")
    days: Annotated[list[date], Field(min_length=1, max_length=366, description="The days the user confirmed.")]
    product: str | None = Field(
        default=None, max_length=100,
        description="The registered product the days will be retrieved from; days outside its published record are left out.",
    )


async def _check(call: Call[CheckComparisonDaysParams]) -> CapabilityResult:
    p = call.params
    earliest, latest = _bounds(_product(call, p.product), call.today)
    kept, dropped = usable_days(p.event_date, list(p.days), latest=latest, earliest=earliest)
    facts = {
        "event": p.event_date.isoformat(), "latest": latest.isoformat(),
        "earliest": earliest.isoformat() if earliest is not None else "",
        "source": p.product or "the data service",
    }
    disclosures = tuple(DROPPED[reason].format(day=day.isoformat(), **facts) for day, reason in dropped)
    values: dict[str, JsonValue] = {
        "days": [day.isoformat() for day in kept],
        "kept": len(kept),
        "confirmed": len(set(p.days)),
        "dropped": [{"day": day.isoformat(), "reason": reason} for day, reason in dropped],
        "event_date": p.event_date.isoformat(),
        "latest": latest.isoformat(),
    }
    return call.done(values=values, disclosures=disclosures)


check_comparison_days = Capability(
    name="operations.check_comparison_days",
    description=(
        "Check the comparison days a user confirmed for an event day: the event day, the day either side of it "
        "and days the product has not published are left out, each with a stated reason; nothing is retrieved."
    ),
    params=CheckComparisonDaysParams,
    output_kind=None,
    label="Checking the confirmed days",
    method=(
        "{kept} of the {confirmed} confirmed days were kept; the event day, the day either side of it and any day "
        "after {latest} or before the product's record were left out."
    ),
    retrieves=False,
    retry_safe=True,
    # It checks the answer to a workflow's question, which an exploration cannot ask.
    workflow_only=True,
    run=_check,
)


CAPABILITIES: tuple[Capability[Any], ...] = (select_comparison_days, check_comparison_days)
