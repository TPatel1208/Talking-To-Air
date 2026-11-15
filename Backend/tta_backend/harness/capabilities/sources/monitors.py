"""Ground monitors from EPA AQS: monitor locations, readings, summaries and exceedance days.

Each capability takes the user's epa_aqs connector secret from the call and
nothing else; there is no shared credential.
"""
from __future__ import annotations

import datetime
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import Field, JsonValue, model_validator
from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.prepared import PreparedGeometry, prep

from tta_backend.datasets.standards import load_standards, truncate_to_reporting_precision
from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.domain.monitors import parsing, queries
from tta_backend.domain.monitors.aqs import AqsNoDataError, Fetch
from tta_backend.domain.monitors.pollutants import POLLUTANTS, AqsPollutant, AqsStandard, Pollutant, convert
from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import points_frame, series_frame
from tta_backend.harness.capabilities.sources.areas import area_facts, area_geometry
from tta_backend.harness.handles.model import Descriptor, Period
from tta_backend.harness.trace import FailureCategory

CONNECTOR = "epa_aqs"
PRODUCT = "EPA AQS"

MAX_MONITORS = 10
MAX_REQUESTS = 12
MAX_HOURLY_DAYS = 31
MAX_DAYS_LISTED = 40
# An area request reads every monitor in the area's bounding box: the box may
# span at most this many square degrees (New Jersey's is about 6), and the
# monitors inside the area times the calendar years may be at most this many.
MAX_AREA_SQUARE_DEGREES = 25.0
MAX_AREA_MONITOR_YEARS = 40
MAX_DIFFERING_LISTED = 10
_SITE_ID = r"^\d{2}-\d{3}-\d{4}$"
_KM_PER_MILE = 1.609344
# Certified as well as not needing certification.
_CERTIFIED = {"certified", "certification not required"}

AREA_TOO_LARGE = (
    FailureCategory.REQUEST_TOO_LARGE,
    "{name} covers too large an area for EPA AQS; narrow the area or name the monitors.",
)

GROUP = Group(
    name="sources.monitors",
    description="EPA AQS ground monitors: find monitors, readings, daily/quarterly/annual summaries, exceedance days.",
    notes=(
        "Needs the user's EPA AQS connector (epa_aqs). Pollutants are o3, pm25, pm10, no2, so2 and co.\n"
        "Start with find, near an area handle or a lat/lon point; it returns monitor locations as points, "
        "nearest first, and their ids in values.monitor_ids. Near an area it may list monitors just outside "
        "it, marked inside_area false.\n"
        "Pass those ids (up to 10), or the points handle, to readings, the summaries or exceedance_days. "
        "Summaries and exceedance_days also take an area handle directly and keep only the monitors inside it; "
        "a large area is refused, so narrow it or name the monitors.\n"
        "readings gives one series line per monitor: hourly for up to 31 days, or daily values of the "
        "statistic the pollutant's standard uses.\n"
        "Summary and exceedance tables carry site_id, name, lat, lon, value and unit, so they can be drawn "
        "on a monitor map.\n"
        "AQS publishes with a lag of about two months, and recent data is not yet certified; results say so.\n"
        "exceedance_days compares daily values, truncated to the standard's reporting precision, with the "
        "NAAQS level for the chosen averaging period (default: the short-term standard) and names the "
        "standard it applied."
    ),
)


class _Refused(Exception):
    """A request the source cannot run, classified for the failure wording."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _bad_input() -> _Refused:
    return _Refused("user_input")


def _too_large() -> _Refused:
    return _Refused("too_large")


MonitorId = Annotated[str, Field(pattern=_SITE_ID)]


class _Dates(Params):
    pollutant: Pollutant
    start: datetime.date
    end: datetime.date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start > self.end:
            raise ValueError("start must be on or before end")
        return self


class FindParams(_Dates):
    area: Annotated[str | None, handle_param("polygons")] = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _one_place(self) -> Self:
        point = (self.lat is not None, self.lon is not None)
        if point == (True, False) or point == (False, True):
            raise ValueError("give lat and lon together")
        if (self.area is None) == (self.lat is None):
            raise ValueError("give an area or a point, not both")
        return self


class _Monitors(_Dates):
    monitors: list[MonitorId] | None = Field(default=None, min_length=1, max_length=MAX_MONITORS)
    points: Annotated[str | None, handle_param("points")] = None


class ReadingsParams(_Monitors):
    frequency: Literal["hourly", "daily"] = "daily"

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.monitors is None) == (self.points is None):
            raise ValueError("give monitors or points, not both")
        return self


class SummaryParams(_Monitors):
    area: Annotated[str | None, handle_param("polygons")] = None
    averaging: str | None = Field(default=None, description="The standard's averaging period, e.g. 8-hour or annual")

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        given = sum(x is not None for x in (self.monitors, self.points, self.area))
        if given != 1:
            raise ValueError("give exactly one of monitors, points or area")
        return self


@dataclass
class _Target:
    """Where a request reads from: monitor ids, or an area's [west, south, east,
    north] box whose rows are kept only inside the area's footprint."""

    monitors: tuple[str, ...]
    bounds: tuple[float, float, float, float] | None
    region: JsonValue
    footprint: PreparedGeometry | None = None
    disclosures: list[str] = field(default_factory=list)

    @property
    def requests(self) -> int:
        return max(1, len(self.monitors))

    def inside(self, lat: Any, lon: Any) -> bool:
        """Whether a location lies in the area's footprint; always true without an area."""
        if self.footprint is None:
            return True
        lat_f, lon_f = parsing.float_or_none(lat), parsing.float_or_none(lon)
        if lat_f is None or lon_f is None or not (math.isfinite(lat_f) and math.isfinite(lon_f)):
            return False
        return bool(self.footprint.covers(Point(lon_f, lat_f)))


async def _area_target(call: Call[Any]) -> _Target:
    facts = area_facts(call.handle("area"), call=call)
    try:
        geometry = shape(await area_geometry(call, "area"))
        if not geometry.is_valid:
            geometry = make_valid(geometry)
    except (ValueError, TypeError, AttributeError, KeyError):
        raise _bad_input() from None
    if geometry.is_empty:
        raise _bad_input()
    return _Target((), facts.bounds, facts.region(), prep(geometry))


async def _target(call: Call[Any], params: _Monitors) -> _Target:
    if getattr(params, "area", None) is not None:
        return await _area_target(call)
    if params.monitors is not None:
        ids = tuple(dict.fromkeys(params.monitors))
        return _Target(ids, None, {"monitors": list(ids)})
    frame = await call.open("points")
    found = [str(i) for i in frame["id"] if re.match(_SITE_ID, str(i))]
    ids = tuple(dict.fromkeys(found))
    if not ids:
        raise _bad_input()
    target = _Target(ids[:MAX_MONITORS], None, {"monitors": list(ids[:MAX_MONITORS])})
    if len(ids) > MAX_MONITORS:
        target.disclosures.append(
            f"The points handle lists {len(ids)} monitors; the first {MAX_MONITORS} were used."
        )
    return target


def _check_size(
    call: Call[Any], target: _Target, start: datetime.date, end: datetime.date,
) -> list[tuple[datetime.date, datetime.date]]:
    spans = parsing.year_spans(start, end)
    if target.bounds is not None:
        west, south, east, north = target.bounds
        if (east - west) * (north - south) > MAX_AREA_SQUARE_DEGREES:
            raise failures.failure(AREA_TOO_LARGE, name=call.capability.name)
    if len(spans) * target.requests > MAX_REQUESTS:
        raise _too_large()
    return spans


async def _check_area_monitors(
    call: Call[Any], fetch: Fetch, pollutant: AqsPollutant, target: _Target,
    spans: list[tuple[datetime.date, datetime.date]],
) -> int:
    """How many monitors AQS lists inside the area for the period. Refuses an
    area whose monitors times calendar years exceed MAX_AREA_MONITOR_YEARS."""
    assert target.bounds is not None
    west, south, east, north = target.bounds
    listed = await queries.fetch_monitors_in_box(
        fetch, [south, north, west, east], pollutant.param_code,
        spans[0][0].strftime("%Y%m%d"), spans[-1][1].strftime("%Y%m%d"),
    )
    inside = {parsing.site_id(m) for m in listed if target.inside(m.get("latitude"), m.get("longitude"))}
    if len(inside) * len(spans) > MAX_AREA_MONITOR_YEARS:
        raise failures.failure(AREA_TOO_LARGE, name=call.capability.name)
    return len(inside)


# (state, county, site, cbsa, minlat, maxlat, minlon, maxlon), as queries take them.
_Filter = tuple[Any, Any, Any, Any, Any, Any, Any, Any]


def _filter_args(target: _Target) -> list[_Filter]:
    """One filter per request."""
    if target.bounds is not None:
        west, south, east, north = target.bounds
        return [(None, None, None, None, south, north, west, east)]
    return [(None, None, monitor, None, None, None, None, None) for monitor in target.monitors]


def _fetch(call: Call[Any]) -> Fetch:
    return aqs_client.fetcher(aqs_client.credential_from_secret(call.secret))


def _pollutant(key: str) -> AqsPollutant:
    return POLLUTANTS[key]


def _standard(pollutant: AqsPollutant, averaging: str | None) -> AqsStandard:
    try:
        return pollutant.standard(averaging)
    except KeyError:
        raise _bad_input() from None


def _unit(pollutant: AqsPollutant) -> str:
    return load_standards()[pollutant.key].unit


def _decimals(pollutant: AqsPollutant) -> int:
    return load_standards()[pollutant.key].decimals


def _at_precision(pollutant: AqsPollutant, value: float) -> float:
    """``value``, in the standards unit, truncated to the standard's reporting precision."""
    return truncate_to_reporting_precision(pollutant.key, value)


def _period(start: datetime.date, end: datetime.date) -> Period:
    return Period(start=start.isoformat(), end=end.isoformat())


def _period_text(start: datetime.date, end: datetime.date) -> str:
    return start.isoformat() if start == end else f"{start.isoformat()} to {end.isoformat()}"


def _no_data(what: str, where: str, start: datetime.date, end: datetime.date, today: datetime.date) -> CapabilityResult:
    reason = f"EPA AQS returned no {what} for {where} for {_period_text(start, end)}."
    if parsing.is_recent(end, today=today):
        reason += " AQS publishes measurements with a lag of about two months, so this period may not be published yet."
    return CapabilityResult(status="insufficient_evidence", reason=reason)


def _preliminary(rows: Iterable[dict[str, Any]], end: datetime.date, today: datetime.date) -> list[str]:
    """Say when data may still change: rows not certified, or a year not yet due for certification."""
    indicators = [r.get("certification_indicator") for r in rows if r.get("certification_indicator")]
    if indicators:
        pending = sum(str(i).strip().lower() not in _CERTIFIED for i in indicators)
        if pending:
            return [f"{pending} of {len(indicators)} AQS rows are not yet certified by the reporting agency and may change."]
        return []
    # Agencies certify a year's data by May 1 of the following year.
    certified_through = today.year - 1 if today >= datetime.date(today.year, 5, 1) else today.year - 2
    if end.year > certified_through:
        return [f"Data from {certified_through + 1} onward is preliminary: not yet certified, and it may change."]
    return []


def _stats(values: Iterable[float]) -> dict[str, JsonValue]:
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return {"min": None, "max": None, "mean": None}
    return {"min": round(float(series.min()), 6), "max": round(float(series.max()), 6), "mean": round(float(series.mean()), 6)}


def _poc(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**6


def _one_line(rows: list[dict[str, Any]], value_of: Callable[[dict[str, Any]], float | None]) -> tuple[list[dict[str, Any]], tuple[str, str] | None]:
    """The rows of one instrument (POC and sample duration) at a site: the one
    with the most values, then the lowest POC. Also the choice, when there was one."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("poc")), str(row.get("sample_duration") or "")), []).append(row)
    if len(groups) <= 1:
        return rows, None
    key = min(groups, key=lambda k: (-sum(value_of(r) is not None for r in groups[k]), _poc(k[0]), k[1]))
    return groups[key], key


def _by_site(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sites: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sites.setdefault(parsing.site_id(row), []).append(row)
    return sites


def _choice_disclosure(site: str, choice: tuple[str, str]) -> str:
    poc, duration = choice
    return f"Monitor {site} reports more than one instrument; its values use POC {poc} ({duration or 'no duration'})."


def _converted(value: float | None, row: dict[str, Any], unit: str) -> float | None:
    if value is None:
        return None
    return convert(value, parsing.unit_code(row.get("units_of_measure")), unit)


def _where(target: _Target) -> str:
    return "the area" if target.bounds is not None else "the requested monitors"


# find


async def _find(call: Call[FindParams]) -> CapabilityResult:
    p = call.params
    pollutant = _pollutant(p.pollutant)
    inside_of: Callable[[Any, Any], bool]
    if p.area is not None:
        area = await _area_target(call)
        assert area.bounds is not None
        west, south, east, north = area.bounds
        bbox = parsing.enforce_min_bbox([south, north, west, east])
        lat_q, lon_q = (south + north) / 2, (west + east) / 2
        region: JsonValue = area.region
        where = "the area"
        inside_of = area.inside
    else:
        assert p.lat is not None and p.lon is not None
        lat_q, lon_q = p.lat, p.lon
        bbox = parsing.bbox_from_point(lat_q, lon_q)
        box = (bbox[2], bbox[0], bbox[3], bbox[1])
        region = {"point": [p.lat, p.lon]}
        where = "the point"

        def inside_of(lat: Any, lon: Any) -> bool:
            return bool(box[1] <= float(lat) <= box[3] and box[0] <= float(lon) <= box[2])

    found = await queries.fetch_active_monitors(
        _fetch(call), bbox, pollutant.param_code, p.start.strftime("%Y%m%d"), p.end.strftime("%Y%m%d"), p.limit,
        count_sites=True,
    )
    if not found:
        return _no_data(f"active {pollutant.name} monitors", f"{where} or within {parsing.BBOX_EXPANSIONS[-1]:g} degrees of it", p.start, p.end, call.today)

    nearest = parsing.nearest_k(found, lat_q, lon_q, len(found))
    sites: dict[str, dict[str, Any]] = {}
    pocs: dict[str, list[str]] = {}
    for monitor in nearest:
        sid = parsing.site_id(monitor)
        sites.setdefault(sid, monitor)
        pocs.setdefault(sid, []).append(str(monitor.get("poc", "")))
    chosen = list(sites.items())[: p.limit]
    frame = points_frame(pd.DataFrame([
        {
            "id": sid,
            "name": m.get("local_site_name") or m.get("address") or sid,
            "lat": float(m["latitude"]),
            "lon": float(m["longitude"]),
            "pollutant": pollutant.key,
            "param_code": pollutant.param_code,
            "parameter": m.get("parameter_name"),
            "pocs": ",".join(x for x in pocs[sid] if x),
            "distance_km": round(m["_dist"] * _KM_PER_MILE, 3),
            "inside_area": inside_of(m["latitude"], m["longitude"]),
            "city": m.get("city_name"),
            "county": m.get("county_name"),
            "state": m.get("state_name"),
            "open_date": m.get("open_date"),
            "close_date": m.get("close_date"),
        }
        for sid, m in chosen
    ]))
    inside = int(frame["inside_area"].sum())
    record = await call.mint(
        "points", frame,
        Descriptor(
            summary=f"{len(frame)} EPA AQS {pollutant.name} monitors near {where}, nearest first",
            quantity=f"{pollutant.name} monitor locations", product=PRODUCT, region=region,
            period=_period(p.start, p.end),
        ),
    )
    disclosures = [
        f"Monitors are those AQS lists as measuring {pollutant.name} (parameter {pollutant.param_code}) "
        f"during {_period_text(p.start, p.end)}.",
    ]
    if inside < len(frame):
        if p.area is not None:
            disclosures.append(
                f"{len(frame) - inside} of the {len(frame)} monitors lie outside the area's boundary; "
                "they are listed as nearby monitors with inside_area false."
            )
        else:
            disclosures.append(f"{len(frame) - inside} of the {len(frame)} monitors lie outside the search box.")
    return call.done(
        record,
        summary={
            "monitors": len(frame),
            "inside": inside,
            "nearest_km": float(frame["distance_km"].min()),
            "farthest_km": float(frame["distance_km"].max()),
            "period": {"start": p.start.isoformat(), "end": p.end.isoformat()},
        },
        values={"monitor_ids": list(frame["id"].astype(str))},
        disclosures=tuple(disclosures),
    )


find = Capability(
    name="sources.monitors.find",
    description="Find EPA AQS monitors measuring a pollutant near an area handle or a lat/lon point, nearest first.",
    params=FindParams,
    output_kind="points",
    label="Finding ground monitors",
    method="EPA AQS monitors measuring {pollutant} from {start} to {end} were listed nearest first; {summary.monitors} were found.",
    retrieves=True,
    retry_safe=True,
    connector=CONNECTOR,
    run=_find,
)


# readings


async def _hourly_rows(fetch: Fetch, pollutant: AqsPollutant, target: _Target, spans: list[tuple[datetime.date, datetime.date]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for span_start, span_end in spans:
        for args in _filter_args(target):
            records, _, _ = await queries.fetch_samples(
                fetch, pollutant.param_code, span_start.strftime("%Y%m%d"), span_end.strftime("%Y%m%d"), *args
            )
            rows.extend(r for r in records if str(r.get("sample_duration", "")).upper() == "1 HOUR")
    return rows


@dataclass
class _Summaries:
    """AQS summary rows, one event type per monitor-period, inside the target."""

    rows: list[dict[str, Any]]
    disclosures: list[str]


_PERIOD_WORDS = {"dailyData": "day", "quarterlyData": "quarter", "annualData": "year"}


def _event_disclosures(prefix: str, differing: list[dict[str, Any]], used: dict[str, int]) -> list[str]:
    word = _PERIOD_WORDS[prefix]
    lines = [
        f"Where AQS gives a monitor-{word} one row per exceptional-event type, the row with every measured "
        "value is used: event type 'No Events', or 'Events Included' when an event was flagged."
    ]
    fallback = {kind: n for kind, n in used.items() if kind and kind not in parsing.INCLUDED_EVENT_TYPES}
    for kind, n in sorted(fallback.items()):
        lines.append(f"{n} monitor-{word}s had no row including event data, so their '{kind}' row is used.")
    if differing:
        listed = ", ".join(f"{d['site_id']} {d['period']}" for d in differing[:MAX_DIFFERING_LISTED])
        more = f" and {len(differing) - MAX_DIFFERING_LISTED} more" if len(differing) > MAX_DIFFERING_LISTED else ""
        lines.append(
            f"{len(differing)} monitor-{word}s have different values with and without exceptional-event data "
            f"(such as wildfire smoke): {listed}{more}."
        )
    return lines


async def _summary_rows(
    fetch: Fetch, prefix: str, pollutant: AqsPollutant, standard: AqsStandard, target: _Target,
    spans: list[tuple[datetime.date, datetime.date]], today: datetime.date,
) -> _Summaries:
    raw: list[dict[str, Any]] = []
    for span_start, span_end in spans:
        for args in _filter_args(target):
            try:
                records, _, _ = await queries.fetch_summary(
                    fetch, prefix, pollutant.param_code, span_start, span_end,
                    span_start.strftime("%Y%m%d"), span_end.strftime("%Y%m%d"), *args,
                    None, None, standard.name, today=today,
                )
            except AqsNoDataError:
                continue
            raw.extend(records)
    disclosures: list[str] = []
    if target.footprint is not None:
        outside = {parsing.site_id(r) for r in raw if not target.inside(r.get("latitude"), r.get("longitude"))}
        raw = [r for r in raw if parsing.site_id(r) not in outside]
        if outside:
            disclosures.append(
                f"{len(outside)} monitors in the area's bounding box but outside its boundary were left out."
            )
    rows, differing, used = parsing.one_event_type(raw)
    if rows:
        disclosures.extend(_event_disclosures(prefix, differing, used))
    return _Summaries(rows, disclosures)


def _field_value(name: str, row: dict[str, Any]) -> float | None:
    return parsing.float_or_none(row.get(name))


def _daily_time(row: dict[str, Any]) -> str | None:
    day = row.get("date_local")
    return str(day) if day else None


def _hourly_time(row: dict[str, Any]) -> str | None:
    day, hour = row.get("date_gmt"), row.get("time_gmt")
    return f"{day}T{hour}" if day and hour else None


async def _readings(call: Call[ReadingsParams]) -> CapabilityResult:
    p = call.params
    pollutant = _pollutant(p.pollutant)
    standard = pollutant.standard()
    unit = _unit(pollutant)
    target = await _target(call, p)
    if p.frequency == "hourly" and (p.end - p.start).days + 1 > MAX_HOURLY_DAYS:
        raise _too_large()
    spans = _check_size(call, target, p.start, p.end)
    fetch = _fetch(call)

    disclosures = list(target.disclosures)
    value_of: Callable[[dict[str, Any]], float | None]
    time_of: Callable[[dict[str, Any]], str | None]
    if p.frequency == "hourly":
        raw = await _hourly_rows(fetch, pollutant, target, spans)
        value_of, time_of = partial(_field_value, "sample_measurement"), _hourly_time
        event_lines: list[str] = []
    else:
        daily = await _summary_rows(fetch, "dailyData", pollutant, standard, target, spans, call.today)
        raw, event_lines = daily.rows, daily.disclosures
        value_of, time_of = partial(_field_value, standard.daily_field), _daily_time

    lines: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    unconverted = 0
    qualified = 0
    sites = _by_site(raw)
    for site in target.monitors:
        rows, choice = _one_line(sites.get(site, []), value_of)
        if choice is not None:
            disclosures.append(_choice_disclosure(site, choice))
        for row in rows:
            when, value = time_of(row), _converted(value_of(row), row, unit)
            if when is None or value_of(row) is None:
                continue
            if value is None:
                unconverted += 1
                continue
            if row.get("qualifier"):
                qualified += 1
            lines.append({"time": when, "value": value, "series": site, "poc": str(row.get("poc", ""))})
            used.append(row)
    if not lines:
        return _no_data(f"{p.frequency} {pollutant.name} readings", "the requested monitors", p.start, p.end, call.today)

    frame = series_frame(pd.DataFrame(lines))
    with_data = list(dict.fromkeys(frame["series"].astype(str)))
    missing = [m for m in target.monitors if m not in with_data]
    if missing:
        disclosures.append(f"No {p.frequency} readings for monitors: {', '.join(missing)}.")
    if p.frequency == "daily":
        disclosures.append(
            f"Daily values are the {standard.statistic} from AQS rows for the {standard.name} standard, "
            "one per local calendar day, stamped at midnight UTC."
        )
    else:
        disclosures.append("Hourly values are AQS 1-hour samples, stamped in UTC.")
    disclosures.extend(event_lines)
    if qualified:
        disclosures.append(f"{qualified} values carry an AQS qualifier flag.")
    if unconverted:
        disclosures.append(f"{unconverted} values in a unit other than {unit} were left out.")
    disclosures.extend(_preliminary(used, p.end, call.today))

    record = await call.mint(
        "series", frame,
        Descriptor(
            summary=f"EPA AQS {p.frequency} {pollutant.name} at {len(with_data)} monitors, {_period_text(p.start, p.end)}",
            quantity=f"{pollutant.name} concentration", unit=unit, product=PRODUCT, region=target.region,
            period=_period(p.start, p.end),
        ),
    )
    times = frame["time"]
    return call.done(
        record,
        summary={
            "monitors": len(with_data),
            "values": len(frame),
            **_stats(frame["value"]),
            "period": {"start": times.min().isoformat(), "end": times.max().isoformat()},
        },
        values={"monitor_ids": with_data},
        disclosures=tuple(disclosures),
    )


readings = Capability(
    name="sources.monitors.readings",
    description="Hourly or daily EPA AQS readings for one or several monitors over a period, one series line per monitor.",
    params=ReadingsParams,
    output_kind="series",
    label="Fetching ground monitor readings",
    method="EPA AQS {frequency} {pollutant} readings from {start} to {end} were fetched for {summary.monitors} monitors.",
    retrieves=True,
    retry_safe=True,
    connector=CONNECTOR,
    run=_readings,
)


async def _area_rows(
    call: Call[SummaryParams], prefix: str, pollutant: AqsPollutant, standard: AqsStandard, target: _Target,
) -> _Summaries | None:
    """The summary rows for the target, refusing an area too large to read;
    None when an area holds no monitors."""
    p = call.params
    spans = _check_size(call, target, p.start, p.end)
    fetch = _fetch(call)
    if target.footprint is not None and not await _check_area_monitors(call, fetch, pollutant, target, spans):
        return None
    return await _summary_rows(fetch, prefix, pollutant, standard, target, spans, call.today)


# summaries

_PERIODS = {"daily": "dailyData", "quarterly": "quarterlyData", "annual": "annualData"}


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in row.items() if k != "peak"}
    for key, value in (row.get("peak") or {}).items():
        flat[f"peak_{key}"] = value
    return flat


def _quarter_overlaps(row: dict[str, Any], start: datetime.date, end: datetime.date) -> bool:
    try:
        year, quarter = int(row["year"]), int(row["quarter"])
    except (KeyError, TypeError, ValueError):
        return True
    first = datetime.date(year, 3 * quarter - 2, 1)
    last = datetime.date(year + (quarter == 4), (3 * quarter) % 12 + 1, 1) - datetime.timedelta(days=1)
    return first <= end and last >= start


def _summary_value(period: str, standard: AqsStandard) -> tuple[str, str]:
    """The summary column the map's ``value`` takes, and what it is."""
    if period == "daily":
        if standard.daily_field == "first_max_value":
            return "max", f"the {standard.statistic} (highest across a site's instruments)"
        return "mean", f"the {standard.statistic} (mean across a site's instruments)"
    return "mean", f"the {period} arithmetic mean AQS reports for the {standard.name} standard"


def _map_columns(frame: pd.DataFrame, column: str, unit: str) -> tuple[pd.DataFrame, int]:
    """``frame`` with the monitor-map columns name, lat, lon, value and unit,
    ``value`` in ``unit``; also how many values could not be put in it."""
    names = [
        next((str(v) for v in (row.get("local_site_name"), row.get("monitor_name")) if isinstance(v, str) and v), row["site_id"])
        for row in frame.to_dict("records")
    ]
    values = [
        None if parsing.float_or_none(v) is None else convert(float(v), parsing.unit_code(u), unit)
        for v, u in zip(frame[column], frame["units"], strict=True)
    ]
    unconverted = sum(v is None and parsing.float_or_none(raw) is not None for v, raw in zip(values, frame[column], strict=True))
    out = frame.assign(
        name=names,
        lat=pd.to_numeric(frame["latitude"], errors="coerce"),
        lon=pd.to_numeric(frame["longitude"], errors="coerce"),
        value=pd.Series(values, index=frame.index, dtype="float64"),
        unit=unit,
    )
    return out, unconverted


def _summary(period: Literal["daily", "quarterly", "annual"]) -> Capability[SummaryParams]:
    prefix = _PERIODS[period]

    async def run(call: Call[SummaryParams]) -> CapabilityResult:
        p = call.params
        pollutant = _pollutant(p.pollutant)
        standard = _standard(pollutant, p.averaging)
        target = await _target(call, p)
        found = await _area_rows(call, prefix, pollutant, standard, target)
        raw = found.rows if found is not None else []
        if period == "quarterly":
            raw = [r for r in raw if _quarter_overlaps(r, p.start, p.end)]
        if not raw or found is None:
            return _no_data(f"{period} summaries of {pollutant.name}", _where(target), p.start, p.end, call.today)
        body, sites, _ = parsing.aggregate_summary_records(raw, period)
        unit = _unit(pollutant)
        column, statistic = _summary_value(period, standard)
        frame, unconverted = _map_columns(pd.DataFrame([_flatten(row) for row in body]), column, unit)
        units = {parsing.unit_code(u) for u in frame["units"].dropna()}
        disclosures = list(target.disclosures) + found.disclosures
        disclosures.append(f"Rows are AQS {period} statistics for the {standard.name} standard.")
        disclosures.append(f"The value column is {statistic}, in {unit}.")
        if sites > parsing.SUMMARY_SITE_CAP:
            disclosures.append(f"AQS matched {sites} monitors; the table keeps the first {parsing.SUMMARY_SITE_CAP} by site id.")
        if units != {unit}:
            disclosures.append("The mean, min and max columns are in the AQS units given in the units column.")
        if unconverted:
            disclosures.append(f"{unconverted} values in a unit other than {unit} have no value.")
        if period == "quarterly":
            disclosures.append("Quarterly rows cover whole quarters that overlap the period.")
        elif period == "annual":
            disclosures.append("Annual rows cover whole calendar years that overlap the period.")
        disclosures.extend(_preliminary(raw, p.end, call.today))
        record = await call.mint(
            "table", frame,
            Descriptor(
                summary=(
                    f"EPA AQS {period} {pollutant.name} summaries for {frame['site_id'].nunique()} monitors, "
                    f"{_period_text(p.start, p.end)}; value is {statistic}"
                ),
                quantity=f"{pollutant.name} {statistic}", unit=unit, product=PRODUCT, region=target.region,
                period=_period(p.start, p.end),
            ),
        )
        return call.done(
            record,
            summary={
                "rows": len(frame),
                "monitors": int(frame["site_id"].nunique()),
                "min": _stats(frame["min"])["min"],
                "max": _stats(frame["max"])["max"],
                "mean": _stats(frame["mean"])["mean"],
                "period": {"start": p.start.isoformat(), "end": p.end.isoformat()},
            },
            values={"standard": standard.name, "value": statistic},
            disclosures=tuple(disclosures),
        )

    return Capability(
        name=f"sources.monitors.{period}_summary",
        description=f"EPA AQS {period} summary statistics per monitor for monitor ids, a points handle or an area handle.",
        params=SummaryParams,
        output_kind="table",
        label=f"Fetching {period} ground monitor summaries",
        method=f"EPA AQS {period} summaries of {{pollutant}} for the {{standard}} standard from {{start}} to {{end}} were fetched for {{summary.monitors}} monitors.",
        retrieves=True,
        retry_safe=True,
        connector=CONNECTOR,
        run=run,
    )


daily_summary = _summary("daily")
quarterly_summary = _summary("quarterly")
annual_summary = _summary("annual")


# exceedance days


async def _exceedance_days(call: Call[SummaryParams]) -> CapabilityResult:
    p = call.params
    pollutant = _pollutant(p.pollutant)
    standard = _standard(pollutant, p.averaging)
    if standard.averaging_period == "annual":
        raise _bad_input()
    naaqs = pollutant.naaqs(standard.averaging_period)
    level, unit = naaqs.level, naaqs.unit
    target = await _target(call, p)
    found = await _area_rows(call, "dailyData", pollutant, standard, target)
    if found is None:
        return _no_data(f"daily {pollutant.name} values", _where(target), p.start, p.end, call.today)

    value_of = partial(_field_value, standard.daily_field)
    disclosures = list(target.disclosures) + found.disclosures
    days: list[dict[str, Any]] = []
    checked = 0
    unconverted = 0
    used: list[dict[str, Any]] = []
    for site, site_rows in sorted(_by_site(found.rows).items()):
        rows, choice = _one_line(site_rows, value_of)
        if choice is not None:
            disclosures.append(_choice_disclosure(site, choice))
        for row in rows:
            day = row.get("date_local")
            if not day or value_of(row) is None:
                continue
            measured = _converted(value_of(row), row, unit)
            if measured is None:
                unconverted += 1
                continue
            checked += 1
            used.append(row)
            value = _at_precision(pollutant, measured)
            if value > level:
                days.append({
                    "date": day, "site_id": site, "name": row.get("local_site_name") or site,
                    "lat": parsing.float_or_none(row.get("latitude")), "lon": parsing.float_or_none(row.get("longitude")),
                    "value": value, "unit": unit, "measured": measured, "level": level,
                    "aqi": parsing.float_or_none(row.get("aqi")),
                })
    if not checked:
        return _no_data(f"daily {pollutant.name} values", _where(target), p.start, p.end, call.today)

    columns = ["date", "site_id", "name", "lat", "lon", "value", "unit", "measured", "level", "aqi"]
    frame = pd.DataFrame(days, columns=columns).sort_values(["date", "site_id"], kind="stable").reset_index(drop=True)
    dates = sorted(set(frame["date"].astype(str)))
    decimals = _decimals(pollutant)
    statistic = f"{standard.statistic}, truncated to {decimals} decimals in {unit}"
    disclosures.append(
        f"Applied the {pollutant.name} {standard.averaging_period} NAAQS of {level:g} {unit} "
        f"to the {standard.statistic} (AQS standard {standard.name}); a day exceeds when its value is above the level."
    )
    disclosures.append(
        f"Daily values were truncated to the standard's reporting precision ({decimals} decimals in {unit}) "
        "before comparison; the value column holds the truncated value and measured the value AQS reports."
    )
    disclosures.append("An exceedance day is not a violation: the NAAQS form is evaluated over several years.")
    if unconverted:
        disclosures.append(f"{unconverted} values in a unit other than {unit} were left out.")
    disclosures.extend(_preliminary(used, p.end, call.today))
    monitors = int(pd.Series([parsing.site_id(r) for r in used]).nunique())
    record = await call.mint(
        "table", frame,
        Descriptor(
            summary=(
                f"{len(dates)} days above the {pollutant.name} {standard.averaging_period} NAAQS at {monitors} monitors, "
                f"{_period_text(p.start, p.end)}; value is the {statistic}"
            ),
            quantity=f"{pollutant.name} {statistic}, on days above the NAAQS", unit=unit, product=PRODUCT,
            region=target.region, period=_period(p.start, p.end),
        ),
    )
    return call.done(
        record,
        summary={
            "exceedance_days": len(dates),
            "monitor_days_exceeding": len(frame),
            "monitor_days_checked": checked,
            "monitors": monitors,
            **({"max": round(float(frame["value"].max()), 6)} if len(frame) else {"max": None}),
            "period": {"start": p.start.isoformat(), "end": p.end.isoformat()},
        },
        values={
            "standard": f"{pollutant.name} {standard.averaging_period} NAAQS ({standard.name})",
            "level": level,
            "value": statistic,
            "days": dates[:MAX_DAYS_LISTED],
            "days_truncated": len(dates) > MAX_DAYS_LISTED,
        },
        disclosures=tuple(disclosures),
    )


exceedance_days = Capability(
    name="sources.monitors.exceedance_days",
    description="Days on which EPA AQS monitors exceeded a pollutant's NAAQS level, for monitor ids, a points handle or an area handle.",
    params=SummaryParams,
    output_kind="table",
    label="Finding exceedance days",
    method=(
        "Daily {pollutant} values from {start} to {end}, truncated to the standard's precision, were compared "
        "with the {standard} level of {level} {unit}; {summary.exceedance_days} days exceeded it."
    ),
    retrieves=True,
    retry_safe=True,
    connector=CONNECTOR,
    run=_exceedance_days,
)


CAPABILITIES: tuple[Capability[Any], ...] = (
    find,
    readings,
    daily_summary,
    quarterly_summary,
    annual_summary,
    exceedance_days,
)
