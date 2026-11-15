"""Pure AQS calculations: query filters, nearest monitors, summaries, exceedances and readings."""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from tta_backend.domain.monitors.aqs import RECENT_DATA_WINDOW_DAYS

# Smallest bbox half-width in degrees: AQS returns nothing for a street-sized box.
MIN_BBOX_HALF = 0.25
# Degrees added on every side, in turn, while too few monitors are found.
BBOX_EXPANSIONS = [0.0, 0.5, 1.5, 3.0, 5.0]
SUMMARY_SITE_CAP = 25
_EARTH_RADIUS_MILES = 3958.8
_PLACEHOLDER_FILTER_VALUES = {
    "",
    "site",
    "site_id",
    "site_number",
    "station_id",
    "monitor",
    "monitor_id",
    "unknown",
    "n/a",
    "na",
    "??",
}

# API endpoint prefix -> the plain-English name a researcher reads.
SUMMARY_KINDS = {
    "dailyData": "daily summary",
    "quarterlyData": "quarterly summary",
    "annualData": "annual summary",
}

# AQS ``units_of_measure`` -> the unit codes used in datasets/standards.yaml.
_UNIT_CODES = {
    "parts per billion": "ppb",
    "parts per million": "ppm",
    "micrograms/cubic meter (lc)": "ug/m3",
    "micrograms/cubic meter (25 c)": "ug/m3",
    "micrograms/cubic meter": "ug/m3",
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) degree points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def enforce_min_bbox(bbox: list[float]) -> list[float]:
    """A [south, north, west, east] bbox at least MIN_BBOX_HALF degrees from its centre each way."""
    south, north, west, east = bbox
    lat_c = (south + north) / 2
    lon_c = (west + east) / 2
    half_lat = max((north - south) / 2, MIN_BBOX_HALF)
    half_lon = max((east - west) / 2, MIN_BBOX_HALF)
    return [lat_c - half_lat, lat_c + half_lat, lon_c - half_lon, lon_c + half_lon]


def expand_bbox(bbox: list[float], degrees: float) -> list[float]:
    """A [south, north, west, east] bbox grown by ``degrees`` on all sides."""
    south, north, west, east = bbox
    return [south - degrees, north + degrees, west - degrees, east + degrees]


def bbox_from_point(lat: float, lon: float) -> list[float]:
    """The minimum-sized [south, north, west, east] bbox centred on a point."""
    return [lat - MIN_BBOX_HALF, lat + MIN_BBOX_HALF, lon - MIN_BBOX_HALF, lon + MIN_BBOX_HALF]


def positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: '{value}' must be an integer or castable to integer.") from exc
    if parsed < 1:
        raise ValueError(f"Invalid {name}: {parsed} must be >= 1.")
    return parsed


def resolve_dates(bdate: str | None, edate: str | None, *, today: date | None = None) -> tuple[date, date, str, str]:
    """Parse YYYY-MM-DD dates; bdate defaults to a year before today, edate to bdate.

    Returns (bdate, edate, bdate as YYYYMMDD, edate as YYYYMMDD).
    """
    default_date = (today or date.today()) - timedelta(days=365)
    bdate_obj = date.fromisoformat(bdate) if bdate else default_date
    edate_obj = date.fromisoformat(edate) if edate else bdate_obj
    if bdate_obj > edate_obj:
        raise ValueError(f"bdate ({bdate_obj}) must be <= edate ({edate_obj})")
    return bdate_obj, edate_obj, bdate_obj.strftime("%Y%m%d"), edate_obj.strftime("%Y%m%d")


def year_spans(start: date, end: date) -> list[tuple[date, date]]:
    """``start``..``end`` cut at calendar years, since AQS wants both dates in one year."""
    spans = []
    cursor = start
    while cursor <= end:
        stop = min(end, date(cursor.year, 12, 31))
        spans.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return spans


def is_recent(edate_obj: date, *, today: date | None = None) -> bool:
    """Whether a window ending on ``edate_obj`` may not be published yet."""
    return edate_obj >= (today or date.today()) - timedelta(days=RECENT_DATA_WINDOW_DAYS)


def no_data_message(kind: str, param_code: str, bdate_obj: date, edate_obj: date, *, today: date | None = None) -> str:
    """Plain-language explanation of an empty result; never endpoint or params."""
    window = (
        bdate_obj.isoformat()
        if bdate_obj == edate_obj
        else f"{bdate_obj.isoformat()} to {edate_obj.isoformat()}"
    )
    if is_recent(edate_obj, today=today):
        return (
            f"EPA has not published {kind} data for {window} yet — AQS "
            "measurements are typically released with a lag of about two "
            "months. Try a date at least three months in the past."
        )
    return (
        f"No EPA {kind} measurements were found for parameter {param_code} "
        f"for {window} in the requested area. Try a wider area, a "
        "different date range, or a different pollutant."
    )


def nearest_k(monitors: list[dict[str, Any]], lat_q: float, lon_q: float, k: int) -> list[dict[str, Any]]:
    """The ``k`` monitors nearest the point, each given ``_dist`` in miles; sorts ``monitors`` in place."""
    for m in monitors:
        m["_dist"] = haversine_miles(lat_q, lon_q, float(m["latitude"]), float(m["longitude"]))
    monitors.sort(key=lambda m: m["_dist"])
    return monitors[: min(k, len(monitors))]


def build_body(nearest: list[dict[str, Any]], param_code: str) -> list[dict[str, Any]]:
    """Monitor dicts in the monitor tools' response Body shape."""
    body = []
    for m in nearest:
        station_id = "-".join(
            str(m.get(key, "??")) for key in ("state_code", "county_code", "site_number")
        )
        station_name = m.get("local_site_name") or m.get("address", "N/A")
        body.append({
            "station_id": station_id,
            "station_name": station_name,
            "monitor_name": station_name,
            "latitude": float(m["latitude"]),
            "longitude": float(m["longitude"]),
            "distance_miles": round(m["_dist"], 3),
            "state_code": m.get("state_code"),
            "county_code": m.get("county_code"),
            "site_number": m.get("site_number"),
            "city_name": m.get("city_name"),
            "county_name": m.get("county_name"),
            "state_name": m.get("state_name"),
            "param_code": param_code,
        })
    return body


def normalise_numeric_filter(name: str, value: Any, *, min_width: int = 0) -> str:
    text = str(value).strip()
    if text.lower() in _PLACEHOLDER_FILTER_VALUES:
        raise ValueError(
            f"Invalid {name}: received placeholder {value!r}. "
            "Use the actual numeric code returned by the monitor lookup tool."
        )
    if not text.isdigit():
        raise ValueError(f"Invalid {name}: {value!r} must contain digits only.")
    return text.zfill(min_width) if min_width else text


def normalise_site_filter(state_code: Any, county_code: Any, site_number: Any) -> tuple[str, str, str]:
    """Zero-padded (state, county, site), from three codes or one 'state-county-site' id."""
    site_text = str(site_number).strip()
    if "-" in site_text:
        parts = site_text.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid site_number/station_id: {site_number!r}. "
                "Expected station_id format state-county-site, e.g. '34-019-0007'."
            )
        state_code, county_code, site_number = parts

    # EPA wants state=2 digits, county=3, site=4.
    return (
        normalise_numeric_filter("state_code", state_code, min_width=2),
        normalise_numeric_filter("county_code", county_code, min_width=3),
        normalise_numeric_filter("site_number", site_number, min_width=4),
    )


def resolve_filter(
    prefix: str,
    state_code: Any, county_code: Any, site_number: Any,
    cbsa_code: Any, minlat: Any, maxlat: Any, minlon: Any, maxlon: Any,
) -> tuple[str, dict[str, Any]]:
    """(endpoint, filter params) for a data prefix and one filter group."""
    if site_number and ((county_code and state_code) or "-" in str(site_number)):
        state_code, county_code, site_number = normalise_site_filter(state_code, county_code, site_number)
        return f"{prefix}/bySite", {"state": state_code, "county": county_code, "site": site_number}
    elif county_code and state_code:
        state_code = normalise_numeric_filter("state_code", state_code, min_width=2)
        county_code = normalise_numeric_filter("county_code", county_code, min_width=3)
        return f"{prefix}/byCounty", {"state": state_code, "county": county_code}
    elif state_code:
        state_code = normalise_numeric_filter("state_code", state_code, min_width=2)
        return f"{prefix}/byState", {"state": state_code}
    elif cbsa_code:
        cbsa_code = normalise_numeric_filter("cbsa_code", cbsa_code)
        return f"{prefix}/byCBSA", {"cbsa": cbsa_code}
    elif minlat is not None and maxlat is not None and minlon is not None and maxlon is not None:
        minlat, maxlat = float(minlat), float(maxlat)
        minlon, maxlon = float(minlon), float(maxlon)
        return f"{prefix}/byBox", {"minlat": minlat, "maxlat": maxlat, "minlon": minlon, "maxlon": maxlon}
    else:
        raise ValueError(
            "Must provide one of: (state_code + county_code + site_number), "
            "(state_code + county_code), state_code, cbsa_code, or "
            "(minlat + maxlat + minlon + maxlon)."
        )


def site_id(r: dict[str, Any]) -> str:
    return "-".join([r.get("state_code", "??"), r.get("county_code", "??"), r.get("site_number", "??")])


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def unit_code(units_of_measure: Any) -> str | None:
    """The standards-file unit for an AQS ``units_of_measure``, or the text itself."""
    if units_of_measure is None or units_of_measure == "":
        return None
    text = str(units_of_measure)
    return _UNIT_CODES.get(text.strip().lower(), text)


def _period_key(record: dict[str, Any], period: str) -> str:
    if period == "daily":
        return str(record.get("date_local") or "")
    if period == "quarterly":
        return f"{record.get('year', '')}-Q{record.get('quarter', '')}"
    return str(record.get("year") or "")


def _period_fields(record: dict[str, Any], period: str) -> dict[str, Any]:
    if period == "daily":
        return {"period": _period_key(record, period), "date": record.get("date_local")}
    if period == "quarterly":
        return {
            "period": _period_key(record, period),
            "year": record.get("year"),
            "quarter": record.get("quarter"),
        }
    return {"period": _period_key(record, period), "year": record.get("year")}


def aggregate_summary_records(
    records: list[dict[str, Any]],
    period: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """One row per (site, period) for the first SUMMARY_SITE_CAP sites.

    Returns (rows, sites matched, site-periods matched).
    """
    site_period_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    site_ids = set()
    for record in records:
        sid = site_id(record)
        site_ids.add(sid)
        site_period_records.setdefault((sid, _period_key(record, period)), []).append(record)

    body = []
    selected_sites = set(sorted(site_ids)[:SUMMARY_SITE_CAP])
    for sid, period_id in sorted(site_period_records):
        if sid not in selected_sites:
            continue

        site_records = site_period_records[(sid, period_id)]
        first = site_records[0]
        means = [v for v in (float_or_none(r.get("arithmetic_mean")) for r in site_records) if v is not None]
        # min and max come only from AQS statistics, never from the mean.
        # dailyData has no minimum_value/maximum_value; its first_max_value is
        # the day's highest sample, so it stands in for max only.
        minima = [
            v
            for v in (float_or_none(r.get("minimum_value")) for r in site_records)
            if v is not None
        ]
        maxima = [
            v
            for v in (
                float_or_none(r.get("maximum_value"))
                if r.get("maximum_value") is not None
                else float_or_none(r.get("first_max_value"))
                for r in site_records
            )
            if v is not None
        ]

        peak_record = None
        peak_value = None
        for record in site_records:
            value = float_or_none(record.get("first_max_value")) if period == "daily" else None
            if value is None:
                value = float_or_none(record.get("maximum_value"))
            if value is None:
                value = float_or_none(record.get("arithmetic_mean"))
            if value is not None and (peak_value is None or value > peak_value):
                peak_record = record
                peak_value = value

        peak: dict[str, Any] = {"value": peak_value}
        if peak_record:
            if period == "daily":
                peak["date"] = peak_record.get("date_local")
                if peak_record.get("first_max_hour") is not None:
                    peak["first_max_hour"] = peak_record.get("first_max_hour")
            elif period == "quarterly":
                peak["year"] = peak_record.get("year")
                peak["quarter"] = peak_record.get("quarter")
            else:
                peak["year"] = peak_record.get("year")

        observation_counts = [
            v for v in (float_or_none(r.get("observation_count")) for r in site_records) if v is not None
        ]
        observation_percents = [
            v for v in (float_or_none(r.get("observation_percent")) for r in site_records) if v is not None
        ]

        item = {
            "site_id": sid,
            **_period_fields(first, period),
            "local_site_name": first.get("local_site_name"),
            "longitude": float_or_none(first.get("longitude")),
            "latitude": float_or_none(first.get("latitude")),
            "monitor_name": first.get("monitor_name") or first.get("local_site_name"),
            "units": first.get("units_of_measure"),
            "pollutant_standard": first.get("pollutant_standard"),
            "sample_duration": first.get("sample_duration"),
            "n_periods": len(site_records),
            "mean": mean(means),
            "min": min(minima) if minima else None,
            "max": max(maxima) if maxima else None,
            "peak": peak,
        }
        if observation_counts:
            item["observation_count"] = int(sum(observation_counts))
        if observation_percents:
            item["observation_percent"] = mean(observation_percents)
        valid_day_counts = [
            v for v in (float_or_none(r.get("valid_day_count")) for r in site_records) if v is not None
        ]
        required_day_counts = [
            v for v in (float_or_none(r.get("required_day_count")) for r in site_records) if v is not None
        ]
        if valid_day_counts:
            item["valid_day_count"] = int(sum(valid_day_counts))
        if required_day_counts:
            item["required_day_count"] = int(sum(required_day_counts))
        body.append(item)

    return body, len(site_ids), len(site_period_records)


# AQS ``event_type`` values in the order rows are preferred. A monitor-period
# with an exceptional event (such as a wildfire) has one row per type; "No
# Events" and "Events Included" hold every measured value, so they come first.
EVENT_TYPES = ("No Events", "Events Included", "Concurred Events Excluded", "Events Excluded")
INCLUDED_EVENT_TYPES = frozenset(EVENT_TYPES[:2])
# Fields compared to tell whether a monitor-period's event types differ.
_EVENT_VALUE_FIELDS = ("arithmetic_mean", "first_max_value", "maximum_value", "minimum_value", "aqi")


def _event_rank(row: dict[str, Any]) -> tuple[int, str]:
    kind = str(row.get("event_type") or "")
    return (EVENT_TYPES.index(kind) if kind in EVENT_TYPES else len(EVENT_TYPES), kind)


def _event_group(row: dict[str, Any]) -> tuple[str, ...]:
    """One monitor (site, POC, sample duration, standard) for one period."""
    return (
        site_id(row), str(row.get("poc")), str(row.get("sample_duration") or ""),
        str(row.get("pollutant_standard") or ""), str(row.get("date_local") or ""),
        str(row.get("year") or ""), str(row.get("quarter") or ""),
    )


def one_event_type(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """One summary row per monitor-period, of the first type in EVENT_TYPES.

    Returns (rows kept in their original order, the monitor-periods whose
    values differ between event types as {"site_id", "period", "event_types"},
    and how many kept rows had each event type).
    """
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(_event_group(record), []).append((index, record))
    kept: list[tuple[int, dict[str, Any]]] = []
    differing: dict[tuple[str, str], dict[str, Any]] = {}
    used: dict[str, int] = {}
    for key, members in groups.items():
        index, chosen = min(members, key=lambda m: (_event_rank(m[1]), m[0]))
        kept.append((index, chosen))
        kind = str(chosen.get("event_type") or "")
        used[kind] = used.get(kind, 0) + 1
        kinds = {str(r.get("event_type") or "") for _, r in members}
        values = {tuple(float_or_none(r.get(f)) for f in _EVENT_VALUE_FIELDS) for _, r in members}
        if len(kinds) > 1 and len(values) > 1:
            period = key[4] or (f"{key[5]}-Q{key[6]}" if key[6] else key[5])
            entry = differing.setdefault((period, key[0]), {"site_id": key[0], "period": period, "event_types": []})
            entry["event_types"] = sorted(set(entry["event_types"]) | kinds)
    kept.sort(key=lambda m: m[0])
    return [r for _, r in kept], [differing[k] for k in sorted(differing)], used


def flag_exceedances(
    records: list[dict[str, Any]],
    measurement_field: str,
    hard_threshold: float | None,
    percentile_threshold: float | None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Days above ``hard_threshold`` or at or above the percentile cutoff, by date.

    Returns (rows, percentile cutoff value).
    """
    def _val(r: dict[str, Any]) -> float | None:
        v = r.get(measurement_field)
        return float(v) if v is not None else None

    values = [_val(r) for r in records]
    valid_values = [v for v in values if v is not None]

    percentile_cutoff = None
    if percentile_threshold is not None:
        if not (0 <= percentile_threshold <= 100):
            raise ValueError("percentile_threshold must be between 0 and 100.")
        sorted_vals = sorted(valid_values)
        idx = min(int(len(sorted_vals) * percentile_threshold / 100), len(sorted_vals) - 1)
        percentile_cutoff = sorted_vals[idx]

    body: list[dict[str, Any]] = []
    for r, v in zip(records, values, strict=True):
        if v is None:
            continue
        triggered = []
        if hard_threshold is not None and v > hard_threshold:
            triggered.append("hard")
        if percentile_cutoff is not None and v >= percentile_cutoff:
            triggered.append("percentile")
        if not triggered:
            continue
        body.append({
            "date": r.get("date_local"),
            "site_id": site_id(r),
            "value": v,
            "aqi": r.get("aqi"),
            "triggered": triggered,
            "local_site_name": r.get("local_site_name"),
        })

    # A record with no date_local sorts first.
    body.sort(key=lambda x: x["date"] or "")
    return body, percentile_cutoff


def sample_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hourly sample records as rows sorted by site, then local time."""
    body = [
        {
            "site_id": site_id(r),
            "datetime_local": r.get("date_local", "") + " " + r.get("time_local", ""),
            "date": r.get("date_local"),
            "hour": int(r["time_local"].split(":")[0]) if r.get("time_local") else None,
            "value": r.get("sample_measurement"),
            "units": r.get("units_of_measure"),
            "sample_duration": r.get("sample_duration"),
            "qualifier": r.get("qualifier"),
            "method": r.get("method"),
            "local_site_name": r.get("local_site_name"),
        }
        for r in records
    ]
    body.sort(key=lambda x: (x["site_id"], x["datetime_local"]))
    return body
