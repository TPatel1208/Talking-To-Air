"""Choose the data product for a quantity, from the request and registry facts.

A pure function: no I/O and no clock. The caller passes today's date, the
registry mapping and the workflow's approved products, already filtered by
their ``when:`` conditions and in preference order. ``today`` and every date
are UTC calendar days.

Rules, in order:

1. A pinned product (one product per variable per run) is used or the result
   is unavailable. A user-named product or instrument is used or the result is
   unavailable. Neither falls through to another product.
2. The workflow's approved products, in preference order; or, in registry
   default mode, every entry recording this quantity.
3. Registry facts rule candidates out: quantity, coverage, record dates,
   retention window, publication latency, cadence.
4. One candidate left: use it. Several: the first in the workflow's order, or
   the best registry rank (ties broken by key, and said so).
5. Nothing fits: unavailable, naming the no-data situation when one applies.

Vouched and unvouched candidates. A candidate is vouched when it was pinned,
named uniquely (by key or a unique short name) or is on the approved list.
For every candidate a missing fact skips that check and is listed in
``unchecked``. An unvouched candidate (registry default mode, or one of
several matches for a named instrument) must also record its quantity and
coverage, or it is ruled out as ``facts_missing``. Registry default mode only
considers entries whose recorded quantity equals the request's.

Areas. A request box may cross the antimeridian (west > east). A product
serves it only when the whole box lies inside the product's coverage; a box
that only partly overlaps is ruled out as ``area_partly_outside_coverage``.

Dates. A single date, a ``DateRange`` or a list. Every requested day must be
inside the record, still held and published. A rolling window of N
``retention_days`` holds today and the N days before it, so the oldest day
held is today minus N. Published means on or before today minus
``latency_days``, and never after today. A seasonal or annual file holds its
whole period, so for those products the period holding the last requested day
must have ended on or before today minus ``latency_days``.

Cadence. ``cadence_needed`` is the finest cadence the analysis needs; a
coarser product is ruled out as ``cadence_too_coarse``. Cadences from finest:
hourly, daily, monthly, seasonal, annual.

Names and cadence words. A name that is no key or short name is read as an
instrument, optionally with one cadence word (hourly, daily, monthly,
seasonal, annual or yearly) before or after it: "TROPOMI annual" matches the
instrument's annual products. An instrument named without a cadence word
matches only its products at monthly cadence or finer, so a seasonal or
annual mean is used only when asked for by name or cadence word, or when the
registry picks it because nothing finer fits. When several products of one
instrument and cadence fit, the registry rank decides, and a regional
product on a finer grid ranks ahead of its global counterpart: an area
inside CONUS gets the 0.01-degree CONUS product, one outside it the global
one.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from tta_backend.datasets.cadence import CADENCE_ORDER, Cadence, period_end
from tta_backend.datasets.registry import CollectionConfig, CoverageBox

REGISTRY_DEFAULT = "registry_default"

CandidatesFrom = Literal["pinned", "user_named", "approved", "registry_default"]

Rule = Literal[
    # The pinned product passed the checks.
    "pinned",
    # The user named exactly one product and it passed the checks.
    "user_named",
    # One candidate passed the checks.
    "only_candidate",
    # Several passed; the first in the workflow's preference order was taken.
    "approved_preference",
    # Several passed; the best registry rank was taken.
    "highest_rank",
    # Several shared the best registry rank; the first by key was taken.
    "highest_rank_tie_by_key",
]

RuleOutReason = Literal[
    "not_registered",
    "not_approved",
    "wrong_quantity",
    "facts_missing",
    "area_outside_coverage",
    "area_partly_outside_coverage",
    "outside_record",
    # Inside the record, but older than the rolling window the product keeps.
    "outside_retention",
    "not_published_yet",
    "cadence_too_coarse",
]

UnavailableReason = Literal[
    # The pinned product cannot serve the request.
    "pinned_product_unsuitable",
    # The named product or instrument cannot serve the request.
    "named_product_unsuitable",
    # The name matches no registry key, short name or instrument.
    "named_not_registered",
    # The user named a product other than the one pinned for this run.
    "named_conflicts_with_pin",
    # The workflow approves no product for this quantity.
    "no_approved_products",
    # Registry default mode and no entry records this quantity.
    "no_registered_product",
    # Candidates existed and every one was ruled out.
    "nothing_fits",
]

NoDataSituation = Literal["outside_record", "not_published_yet"]

# Cadence words a name may carry beside an instrument.
_CADENCE_WORDS: dict[str, Cadence] = {
    "hourly": "hourly", "daily": "daily", "monthly": "monthly",
    "seasonal": "seasonal", "annual": "annual", "yearly": "annual",
}

# Cadences whose file holds one mean over a whole season or year.
_PERIOD_MEANS = frozenset({"seasonal", "annual"})


@dataclass(frozen=True)
class BBox:
    """A request area in degrees. west > east means it crosses the antimeridian."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("west", self.west, 180),
            ("east", self.east, 180),
            ("south", self.south, 90),
            ("north", self.north, 90),
        ):
            if not -limit <= value <= limit:
                raise ValueError(f"{name} {value} is outside -{limit}..{limit}")
        if self.south > self.north:
            raise ValueError(f"south ({self.south}) must be <= north ({self.north})")

    def lon_spans(self) -> tuple[tuple[float, float], ...]:
        if self.west <= self.east:
            return ((self.west, self.east),)
        return ((self.west, 180.0), (-180.0, self.east))


@dataclass(frozen=True)
class DateRange:
    """Every day from start to end, inclusive."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"range end ({self.end}) is before its start ({self.start})")


Dates = date | DateRange | Sequence[date]


@dataclass(frozen=True)
class RuledOut:
    reason: RuleOutReason
    detail: str


@dataclass(frozen=True)
class Considered:
    """One product the resolver looked at. ``fits`` when nothing ruled it out."""

    key: str
    vouched: bool
    ruled_out: tuple[RuledOut, ...] = ()
    # Facts the entry does not record, whose checks were skipped.
    unchecked: tuple[str, ...] = ()

    @property
    def fits(self) -> bool:
        return not self.ruled_out

    @property
    def reasons(self) -> frozenset[RuleOutReason]:
        return frozenset(r.reason for r in self.ruled_out)


@dataclass(frozen=True)
class Resolution:
    """The chosen product and why, or why there is none.

    ``considered`` lists every product looked at, in candidate order, with
    the reasons each was ruled out."""

    product: str | None
    rule: Rule | None
    candidates_from: CandidatesFrom
    considered: tuple[Considered, ...]
    unavailable: UnavailableReason | None = None
    no_data: NoDataSituation | None = None
    # Keys sharing the best rank when ``rule`` is highest_rank_tie_by_key.
    tied: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.product is not None

    @property
    def skipped(self) -> tuple[Considered, ...]:
        """Candidates ahead of the chosen one that were ruled out."""
        out: list[Considered] = []
        for c in self.considered:
            if c.key == self.product:
                break
            if not c.fits:
                out.append(c)
        return tuple(out)


def resolve_product(
    *,
    quantity: str,
    area: BBox,
    dates: Dates,
    registry: Mapping[str, CollectionConfig],
    today: date,
    approved: Sequence[str] | Literal["registry_default"],
    cadence_needed: Cadence | None = None,
    named: str | None = None,
    pinned: str | None = None,
) -> Resolution:
    """Choose the product for ``quantity`` over ``area`` and ``dates``.

    ``approved`` is the workflow's approved keys in preference order, or
    ``REGISTRY_DEFAULT`` to rank every entry recording the quantity.
    ``named`` is a product or instrument the user named; ``pinned`` the
    product already fixed for this variable in this run."""
    if isinstance(approved, str) and approved != REGISTRY_DEFAULT:
        raise ValueError(f"approved must be a list of keys or {REGISTRY_DEFAULT!r}")
    first, last = _date_bounds(dates)
    approved_keys = None if isinstance(approved, str) else tuple(approved)

    def check(key: str, vouched: bool) -> Considered:
        return _check(
            key, vouched, registry, approved_keys, quantity, area,
            first, last, today, cadence_needed,
        )

    named_keys: tuple[str, ...] | None = None
    named_unique = False
    if named is not None:
        named_keys, named_unique = match_named(named, registry)

    if pinned is not None:
        if named_keys == ():
            return Resolution(None, None, "pinned", (), unavailable="named_not_registered")
        if named_keys is not None and pinned not in named_keys:
            return Resolution(None, None, "pinned", (), unavailable="named_conflicts_with_pin")
        pin = check(pinned, True)
        if pin.fits:
            return Resolution(pinned, "pinned", "pinned", (pin,))
        return _unavailable("pinned", (pin,), "pinned_product_unsuitable")

    if named_keys is not None:
        if not named_keys:
            return Resolution(None, None, "user_named", (), unavailable="named_not_registered")
        ordered = _order(named_keys, approved_keys, registry)
        considered = tuple(
            check(k, named_unique or (approved_keys is not None and k in approved_keys))
            for k in ordered
        )
        fitting = [c.key for c in considered if c.fits]
        if not fitting:
            return _unavailable("user_named", considered, "named_product_unsuitable")
        if named_unique:
            return Resolution(fitting[0], "user_named", "user_named", considered)
        return _pick("user_named", fitting, considered, approved_keys, registry)

    if approved_keys is not None:
        if not approved_keys:
            return Resolution(None, None, "approved", (), unavailable="no_approved_products")
        considered = tuple(check(k, True) for k in dict.fromkeys(approved_keys))
        fitting = [c.key for c in considered if c.fits]
        if not fitting:
            return _unavailable("approved", considered, "nothing_fits")
        return _pick("approved", fitting, considered, approved_keys, registry)

    keys = [k for k, cfg in registry.items() if cfg.quantity == quantity]
    if not keys:
        return Resolution(None, None, "registry_default", (), unavailable="no_registered_product")
    considered = tuple(check(k, False) for k in _order(keys, None, registry))
    fitting = [c.key for c in considered if c.fits]
    if not fitting:
        return _unavailable("registry_default", considered, "nothing_fits")
    return _pick("registry_default", fitting, considered, None, registry)


def _pick(
    source: CandidatesFrom,
    fitting: list[str],
    considered: tuple[Considered, ...],
    approved_keys: tuple[str, ...] | None,
    registry: Mapping[str, CollectionConfig],
) -> Resolution:
    if len(fitting) == 1:
        return Resolution(fitting[0], "only_candidate", source, considered)
    if approved_keys is not None:
        return Resolution(fitting[0], "approved_preference", source, considered)
    best = registry[fitting[0]].rank
    tied = tuple(k for k in fitting if registry[k].rank == best)
    if len(tied) > 1:
        return Resolution(tied[0], "highest_rank_tie_by_key", source, considered, tied=tied)
    return Resolution(fitting[0], "highest_rank", source, considered)


def _unavailable(
    source: CandidatesFrom,
    considered: tuple[Considered, ...],
    reason: UnavailableReason,
) -> Resolution:
    return Resolution(
        None, None, source, considered, unavailable=reason, no_data=_no_data(considered)
    )


def _no_data(considered: tuple[Considered, ...]) -> NoDataSituation | None:
    """The no-data situation, when some product was ruled out by dates alone.

    A date older than a rolling archive's window counts as outside the
    record. Not published yet wins over outside the record: it is the one
    that resolves by waiting or by a near-real-time product."""
    date_only = [
        c.reasons for c in considered
        if c.ruled_out and c.reasons <= {"outside_record", "outside_retention", "not_published_yet"}
    ]
    if any(r == {"not_published_yet"} for r in date_only):
        return "not_published_yet"
    if date_only:
        return "outside_record"
    return None


def _order(
    keys: Sequence[str],
    approved_keys: tuple[str, ...] | None,
    registry: Mapping[str, CollectionConfig],
) -> list[str]:
    """Approved keys first in preference order; the rest by rank, then key."""
    unique = list(dict.fromkeys(keys))
    pref = {k: i for i, k in enumerate(approved_keys or ())}

    def sort_key(k: str) -> tuple[int, int, bool, int, str]:
        cfg = registry.get(k)
        rank = cfg.rank if cfg is not None else None
        return (k not in pref, pref.get(k, 0), rank is None, rank or 0, k)

    return sorted(unique, key=sort_key)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_named(
    named: str, registry: Mapping[str, CollectionConfig]
) -> tuple[tuple[str, ...], bool]:
    """Registry keys a user's name refers to, and whether it is one product.

    Tried in order, the first non-empty match wins: the registry key, the
    short name, then the instrument (the whole string, or any part split on
    '/', '+', ',' or spaces, so "OMI" and "Aura" both match "OMI/Aura").
    Case, spaces and punctuation are ignored. An instrument may carry one
    cadence word, which keeps its products of that cadence; without one, its
    seasonal and annual products are left out."""
    want = _norm(named)
    if not want:
        return (), False
    by_key = tuple(k for k in registry if _norm(k) == want)
    if by_key:
        return by_key, len(by_key) == 1
    by_short = tuple(k for k, c in registry.items() if c.short_name and _norm(c.short_name) == want)
    if by_short:
        return by_short, len(by_short) == 1
    words = [w for w in re.split(r"[^a-z0-9]+", named.lower()) if w]
    cadences = {_CADENCE_WORDS[w] for w in words if w in _CADENCE_WORDS}
    if len(cadences) > 1:
        return (), False
    instrument = "".join(w for w in words if w not in _CADENCE_WORDS)
    by_instrument = tuple(
        k for k, c in registry.items()
        if c.instrument and instrument and (
            _norm(c.instrument) == instrument
            or instrument in {_norm(p) for p in re.split(r"[/+,\s]+", c.instrument)}
        ) and (c.cadence in cadences if cadences else c.cadence not in _PERIOD_MEANS)
    )
    return by_instrument, False


def _date_bounds(dates: Dates) -> tuple[date, date]:
    if isinstance(dates, DateRange):
        return dates.start, dates.end
    if isinstance(dates, date):
        return dates, dates
    days = list(dates)
    if not days:
        raise ValueError("no dates requested")
    return min(days), max(days)


def _contains(cov: CoverageBox, area: BBox) -> bool:
    return cov.south <= area.south and area.north <= cov.north and all(
        cov.west <= w and e <= cov.east for w, e in area.lon_spans()
    )


def _overlaps(cov: CoverageBox, area: BBox) -> bool:
    return area.south <= cov.north and cov.south <= area.north and any(
        w <= cov.east and cov.west <= e for w, e in area.lon_spans()
    )


def _check(
    key: str,
    vouched: bool,
    registry: Mapping[str, CollectionConfig],
    approved_keys: tuple[str, ...] | None,
    quantity: str,
    area: BBox,
    first: date,
    last: date,
    today: date,
    cadence_needed: Cadence | None,
) -> Considered:
    cfg = registry.get(key)
    if cfg is None:
        return Considered(key, vouched, (RuledOut("not_registered", "not in the dataset registry"),))

    out: list[RuledOut] = []
    unchecked: list[str] = []

    if approved_keys is not None and key not in approved_keys:
        out.append(RuledOut("not_approved", "not among this workflow's approved products"))

    if not vouched:
        missing = [n for n in ("quantity", "coverage") if getattr(cfg, n) is None]
        if missing:
            out.append(RuledOut(
                "facts_missing",
                f"no {' or '.join(missing)} recorded, so registry facts cannot choose it",
            ))

    if cfg.quantity is None:
        unchecked.append("quantity")
    elif cfg.quantity != quantity:
        out.append(RuledOut("wrong_quantity", f"measures {cfg.quantity}, not {quantity}"))

    if cfg.coverage is None:
        unchecked.append("coverage")
    elif isinstance(cfg.coverage, CoverageBox):
        box = cfg.coverage
        where = f"W{box.west} S{box.south} E{box.east} N{box.north}"
        if not _overlaps(box, area):
            out.append(RuledOut("area_outside_coverage", f"the area is outside its coverage ({where})"))
        elif not _contains(box, area):
            out.append(RuledOut(
                "area_partly_outside_coverage",
                f"part of the area is outside its coverage ({where})",
            ))

    if cfg.record_start is None:
        unchecked.append("record_start")
    if (cfg.record_start is not None and first < cfg.record_start) or (
        cfg.record_end is not None and last > cfg.record_end
    ):
        start = cfg.record_start or "unknown"
        end = cfg.record_end or "ongoing"
        out.append(RuledOut(
            "outside_record",
            f"dates {first}..{last} fall outside its record ({start}..{end})",
        ))

    # Days before record_start are already outside_record.
    if cfg.retention_days is not None:
        oldest = today - timedelta(days=cfg.retention_days)
        if first < oldest and (cfg.record_start is None or first >= cfg.record_start):
            out.append(RuledOut(
                "outside_retention",
                f"keeps only the last {cfg.retention_days} days (from {oldest}); dates start {first}",
            ))

    if cfg.latency_days is None:
        unchecked.append("latency_days")
        newest = today
    else:
        newest = today - timedelta(days=cfg.latency_days)
    # One file holds the whole season or year, published after the period ends.
    covered = period_end(last, cfg.cadence) if cfg.cadence in _PERIOD_MEANS else last
    if covered > newest and (cfg.record_end is None or last <= cfg.record_end):
        latency = "unknown" if cfg.latency_days is None else f"{cfg.latency_days}-day"
        if covered == last:
            detail = f"published through {newest} ({latency} latency); dates run to {last}"
        elif cfg.latency_days is None:
            detail = f"its {cfg.cadence} mean holding {last} runs to {covered}, after today"
        else:
            detail = (
                f"its {cfg.cadence} mean holding {last} runs to {covered} and is published "
                f"{cfg.latency_days} days after that, from {covered + timedelta(days=cfg.latency_days)}"
            )
        out.append(RuledOut("not_published_yet", detail))

    if cadence_needed is not None and CADENCE_ORDER[cfg.cadence] > CADENCE_ORDER[cadence_needed]:
        article = "an" if cfg.cadence[0] in "aeiou" else "a"
        out.append(RuledOut(
            "cadence_too_coarse",
            f"{article} {cfg.cadence} product, so {cadence_needed} values are not available",
        ))

    return Considered(key, vouched, tuple(out), tuple(unchecked))
