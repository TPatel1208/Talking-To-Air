from langchain.tools import tool
import requests
from typing import Dict, Any, List, Optional, Union
import os
import sys
import time
import math
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import get_settings
from utils.plotting import GeocodingService

geocoding_service = GeocodingService()

settings = get_settings()

AQS_BASE_URL = "https://aqs.epa.gov/data/api"
AQS_EMAIL = settings.aqs_api_email
AQS_KEY = settings.aqs_api_key
DEFAULT_PARAM_CODE = "42602"  # NO2

# Initial bbox half-width for street-level addresses (degrees, ~17 miles).
# Nominatim returns a ~10m box for a street address which AQS returns empty for;
# we clamp to this minimum before the expansion ladder runs.
_MIN_BBOX_HALF = 0.25

# Expansion steps added on top of the initial box when no monitors are found
_BBOX_EXPANSIONS = [0.0, 0.5, 1.5, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _aqs_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET request to the AQS API; raises on HTTP errors or unexpected statuses."""
    full_params = {**params, "email": AQS_EMAIL, "key": AQS_KEY}
    resp = requests.get(f"{AQS_BASE_URL}/{endpoint}", params=full_params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    header = data.get("Header", [{}])
    status = header[0].get("status", "").lower()
    # "No data matched your selection" is a valid empty result, not an error
    if status not in ("success", "no data matched your selection", ""):
        raise RuntimeError(f"AQS API error: {header[0]}")
    return data


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) degree points."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _enforce_min_bbox(bbox: List[float]) -> List[float]:
    """
    Ensure a [south, north, west, east] bbox is at least _MIN_BBOX_HALF degrees
    in each direction from its centre. Street-level Nominatim results are often
    only ~10m wide, which AQS returns nothing for.
    """
    south, north, west, east = bbox
    lat_c = (south + north) / 2
    lon_c = (west + east) / 2
    half_lat = max((north - south) / 2, _MIN_BBOX_HALF)
    half_lon = max((east - west) / 2, _MIN_BBOX_HALF)
    return [lat_c - half_lat, lat_c + half_lat, lon_c - half_lon, lon_c + half_lon]


def _expand_bbox(bbox: List[float], degrees: float) -> List[float]:
    """Expand [south, north, west, east] bbox outward by `degrees` on all sides."""
    south, north, west, east = bbox
    return [south - degrees, north + degrees, west - degrees, east + degrees]


def _bbox_from_point(lat: float, lon: float) -> List[float]:
    """Return a minimum-sized [south, north, west, east] bbox centred on a point."""
    return [lat - _MIN_BBOX_HALF, lat + _MIN_BBOX_HALF,
            lon - _MIN_BBOX_HALF, lon + _MIN_BBOX_HALF]


def _resolve_dates(bdate: Optional[str], edate: Optional[str]):
    """
    Parse and validate bdate/edate strings (YYYY-MM-DD).
    Defaults: bdate = 1 year ago, edate = bdate.
    Returns (bdate_obj, edate_obj, bdate_str_YYYYMMDD, edate_str_YYYYMMDD).
    """
    default_date = date.today() - timedelta(days=365)
    bdate_obj = date.fromisoformat(bdate) if bdate else default_date
    edate_obj = date.fromisoformat(edate) if edate else bdate_obj
    if bdate_obj > edate_obj:
        raise ValueError(f"bdate ({bdate_obj}) must be <= edate ({edate_obj})")
    return bdate_obj, edate_obj, bdate_obj.strftime("%Y%m%d"), edate_obj.strftime("%Y%m%d")


def _fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k=1):
    best = []
    for expansion in _BBOX_EXPANSIONS:
        south, north, west, east = _expand_bbox(bbox, expansion)
        data = _aqs_get(
            "monitors/byBox",
            {
                "param": param_code,
                "bdate": bdate_str,
                "edate": edate_str,
                "minlat": south,
                "maxlat": north,
                "minlon": west,
                "maxlon": east,
            },
        )
        monitors = data.get("Data", data.get("Body", []))
        if len(monitors) > len(best):
            best = monitors
        if len(best) >= k:
            return best

    return best


def _nearest_k(monitors: List[Dict], lat_q: float, lon_q: float, k: int) -> List[Dict]:
    """Linear haversine scan — correct and fast for the small byBox result sets."""
    for m in monitors:
        m["_dist"] = _haversine_miles(lat_q, lon_q, float(m["latitude"]), float(m["longitude"]))
    monitors.sort(key=lambda m: m["_dist"])
    return monitors[: min(k, len(monitors))]


def _build_body(nearest: List[Dict], param_code: str) -> List[Dict]:
    """Format monitor dicts into the standard response Body shape."""
    body = []
    for m in nearest:
        station_id = "-".join(
            str(m.get(key, "??")) for key in ("state_code", "county_code", "site_number")
        )
        body.append({
            "station_id": station_id,
            "station_name": m.get("local_site_name") or m.get("address", "N/A"),
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


# ---------------------------------------------------------------------------
# Proximity Tools
# ---------------------------------------------------------------------------

@tool
def list_states() -> Dict[str, Any]:
    """
    Retrieve a list of all US states with EPA AQS air quality monitoring data.

    Returns each state with its 2-digit FIPS code, required for constructing
    other AQS API requests.

    Returns
    -------
    dict : raw AQS API response with Header and Data fields.
    """
    resp = requests.get(
        f"{AQS_BASE_URL}/list/states",
        params={"email": AQS_EMAIL, "key": AQS_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


@tool
def find_closest_monitor(
    location: str,
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    k: int = 1,
) -> Dict[str, Any]:
    """
    Find the closest active EPA AQS monitor to a location name or address.

    Use find_closest_monitor_by_coords instead if you already have lat/lon.

    Args:
        location   : City name or address (geocoded via Nominatim).
        param_code : AQS parameter code (default '42602' = NO2).
        bdate      : Start date YYYY-MM-DD (defaults to 1 year ago).
        edate      : End date YYYY-MM-DD (defaults to bdate).
        k          : Number of nearest monitors to return (default 1).

    Returns Body fields: station_id, station_name, latitude, longitude,
    distance_miles, state_code, county_code, site_number.
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)

    geo = geocoding_service.geocode(location)
    if geo is None:
        raise ValueError(f"Could not geocode location: '{location}'")
    lat_q, lon_q = geo["latitude"], geo["longitude"]

    # Clamp Nominatim bbox to minimum size before the expansion ladder
    bbox = _enforce_min_bbox(geo["bbox"])

    monitors = _fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
    if not monitors:
        raise RuntimeError(
            f"No active {param_code} monitors found near '{location}' "
            f"between {bdate_obj.isoformat()} and {edate_obj.isoformat()} "
            f"even after expanding the search area to ±{_BBOX_EXPANSIONS[-1]}°."
        )

    nearest = _nearest_k(monitors, lat_q, lon_q, k)
    return {
        "Header": [{
            "status": "success",
            "rows": len(nearest),
            "query_location": location,
            "query_lat": lat_q,
            "query_lon": lon_q,
            "param_code": param_code,
            "bdate": bdate_obj.isoformat(),
            "edate": edate_obj.isoformat(),
        }],
        "Body": _build_body(nearest, param_code),
    }


@tool
def find_closest_monitor_by_coords(
    latitude: Union[float, str],
    longitude: Union[float, str],
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    k: int = 1,
) -> Dict[str, Any]:
    """
    Find the closest active EPA AQS monitor to a lat/lon point.

    Use find_closest_monitor instead if you have a location name.

    Args:
        latitude   : Decimal degrees (e.g. 40.7128).
        longitude  : Decimal degrees (e.g. -74.0060).
        param_code : AQS parameter code (default '42602' = NO2).
        bdate      : Start date YYYY-MM-DD (defaults to 1 year ago).
        edate      : End date YYYY-MM-DD (defaults to bdate).
        k          : Number of nearest monitors to return (default 1).

    Returns Body fields: station_id, station_name, latitude, longitude,
    distance_miles, state_code, county_code, site_number.
    """
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        raise ValueError(f"Invalid latitude or longitude: '{latitude}', '{longitude}' must be float or castable to float.")
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)

    bbox = _bbox_from_point(latitude, longitude)
    monitors = _fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
    if not monitors:
        raise RuntimeError(
            f"No active {param_code} monitors found near ({latitude}, {longitude}) "
            f"between {bdate_obj.isoformat()} and {edate_obj.isoformat()} "
            f"even after expanding the search area to ±{_BBOX_EXPANSIONS[-1]}°."
        )

    nearest = _nearest_k(monitors, latitude, longitude, k)
    return {
        "Header": [{
            "status": "success",
            "rows": len(nearest),
            "query_lat": latitude,
            "query_lon": longitude,
            "param_code": param_code,
            "bdate": bdate_obj.isoformat(),
            "edate": edate_obj.isoformat(),
        }],
        "Body": _build_body(nearest, param_code),
    }


# ---------------------------------------------------------------------------
# Shared summary helper (daily / quarterly / annual share identical structure)
# ---------------------------------------------------------------------------
 
def _resolve_filter(
    prefix: str,
    state_code, county_code, site_number,
    cbsa_code, minlat, maxlat, minlon, maxlon,
) -> tuple:
    """Return (endpoint, filter_params) for a given data prefix and filter inputs."""
    if site_number and county_code and state_code:
        return f"{prefix}/bySite", {"state": state_code, "county": county_code, "site": site_number}
    elif county_code and state_code:
        return f"{prefix}/byCounty", {"state": state_code, "county": county_code}
    elif state_code:
        return f"{prefix}/byState", {"state": state_code}
    elif cbsa_code:
        return f"{prefix}/byCBSA", {"cbsa": cbsa_code}
    elif all(v is not None for v in [minlat, maxlat, minlon, maxlon]):
        minlat, maxlat = float(minlat), float(maxlat)
        minlon, maxlon = float(minlon), float(maxlon)
        return f"{prefix}/byBox", {"minlat": minlat, "maxlat": maxlat, "minlon": minlon, "maxlon": maxlon}
    else:
        raise ValueError(
            "Must provide one of: (state_code + county_code + site_number), "
            "(state_code + county_code), state_code, cbsa_code, or "
            "(minlat + maxlat + minlon + maxlon)."
        )
 
 
def _fetch_summary(
    prefix: str,
    param_code: str,
    bdate_obj, edate_obj, bdate_str, edate_str,
    state_code, county_code, site_number,
    cbsa_code, minlat, maxlat, minlon, maxlon,
    cbdate, cedate, pollutant_standard,
) -> tuple:
    """
    Shared fetch + filter logic for daily, quarterly, and annual summaries.
    Returns (records, endpoint, filter_params).
    """
    endpoint, filter_params = _resolve_filter(
        prefix, state_code, county_code, site_number,
        cbsa_code, minlat, maxlat, minlon, maxlon,
    )
    params = {"param": param_code, "bdate": bdate_str, "edate": edate_str, **filter_params}
    if cbdate:
        params["cbdate"] = date.fromisoformat(cbdate).strftime("%Y%m%d")
    if cedate:
        params["cedate"] = date.fromisoformat(cedate).strftime("%Y%m%d")
 
    data = _aqs_get(endpoint, params)
    records = data.get("Data", data.get("Body", []))
 
    if pollutant_standard:
        records = [r for r in records if r.get("pollutant_standard") == pollutant_standard]
 
    if not records:
        raise RuntimeError(
            f"No {prefix} data found for param {param_code} "
            f"between {bdate_obj.isoformat()} and {edate_obj.isoformat()} "
            f"using {endpoint} with {filter_params}"
            + (f" and pollutant_standard='{pollutant_standard}'." if pollutant_standard else ".")
        )
    return records, endpoint, filter_params
 
 
def _build_summary_header(
    rows, endpoint, param_code, bdate_obj, edate_obj, pollutant_standard
):
    return [{
        "status": "success",
        "rows": rows,
        "endpoint": endpoint,
        "param_code": param_code,
        "bdate": bdate_obj.isoformat(),
        "edate": edate_obj.isoformat(),
        "pollutant_standard": pollutant_standard,
    }]
 
 
def _site_id(r):
    return "-".join([r.get("state_code", "??"), r.get("county_code", "??"), r.get("site_number", "??")])
 
 
# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------
 
@tool
def get_daily_summary(
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    state_code: Optional[str] = None,
    county_code: Optional[str] = None,
    site_number: Optional[str] = None,
    cbsa_code: Optional[str] = None,
    minlat: Optional[Union[float, str]] = None,
    maxlat: Optional[Union[float, str]] = None,
    minlon: Optional[Union[float, str]] = None,
    maxlon: Optional[Union[float, str]] = None,
    cbdate: Optional[str] = None,
    cedate: Optional[str] = None,
    pollutant_standard: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Daily summary stats (midnight-to-midnight local). Use for day-level analysis
    or to feed find_exceedance_days. For trends use quarterly/annual instead.
    Filter (one group): state+county+site | state+county | state | cbsa_code | bbox
    Always pass pollutant_standard (see ground prompt table).
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    records, endpoint, _ = _fetch_summary(
        "dailyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code,
        minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
    )
    body = [
        {
            "date": r.get("date_local"),
            "site_id": _site_id(r),
            "arithmetic_mean": r.get("arithmetic_mean"),
            "maximum_value": r.get("maximum_value"),
            "aqi": r.get("aqi"),
            "units": r.get("units_of_measure"),
            "sample_duration": r.get("sample_duration"),
            "pollutant_standard": r.get("pollutant_standard"),
            "observation_count": r.get("observation_count"),
            "observation_percent": r.get("observation_percent"),
            "first_max_value": r.get("first_max_value"),
            "first_max_hour": r.get("first_max_hour"),
            "local_site_name": r.get("local_site_name"),
        }
        for r in records
    ]
    return {
        "Header": _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard),
        "Body": body,
    }
 
 
# ---------------------------------------------------------------------------
# Quarterly summary
# ---------------------------------------------------------------------------
 
@tool
def get_quarterly_summary(
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    state_code: Optional[str] = None,
    county_code: Optional[str] = None,
    site_number: Optional[str] = None,
    cbsa_code: Optional[str] = None,
    minlat: Optional[Union[float, str]] = None,
    maxlat: Optional[Union[float, str]] = None,
    minlon: Optional[Union[float, str]] = None,
    maxlon: Optional[Union[float, str]] = None,
    cbdate: Optional[str] = None,
    cedate: Optional[str] = None,
    pollutant_standard: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quarterly summary stats (Q1=Jan-Mar … Q4=Oct-Dec). Only year portion of
    bdate/edate used — all 4 quarters per year returned. Use for seasonal trends.
    Filter (one group): state+county+site | state+county | state | cbsa_code | bbox
    Always pass pollutant_standard (see ground prompt table).
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    records, endpoint, _ = _fetch_summary(
        "quarterlyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code,
        minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
    )
    body = [
        {
            "year": r.get("year"),
            "quarter": r.get("quarter"),
            "site_id": _site_id(r),
            "arithmetic_mean": r.get("arithmetic_mean"),
            "minimum_value": r.get("minimum_value"),
            "maximum_value": r.get("maximum_value"),
            "percentile_25": r.get("first_quartile"),
            "percentile_75": r.get("third_quartile"),
            "percentile_98": r.get("ninety_eighth_percentile"),
            "units": r.get("units_of_measure"),
            "sample_duration": r.get("sample_duration"),
            "pollutant_standard": r.get("pollutant_standard"),
            "observation_count": r.get("observation_count"),
            "observation_percent": r.get("observation_percent"),
            "local_site_name": r.get("local_site_name"),
        }
        for r in records
    ]
    return {
        "Header": _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard),
        "Body": body,
    }
 
 
# ---------------------------------------------------------------------------
# Annual summary
# ---------------------------------------------------------------------------
 
@tool
def get_annual_summary(
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    state_code: Optional[str] = None,
    county_code: Optional[str] = None,
    site_number: Optional[str] = None,
    cbsa_code: Optional[str] = None,
    minlat: Optional[Union[float, str]] = None,
    maxlat: Optional[Union[float, str]] = None,
    minlon: Optional[Union[float, str]] = None,
    maxlon: Optional[Union[float, str]] = None,
    cbdate: Optional[str] = None,
    cedate: Optional[str] = None,
    pollutant_standard: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Annual summary stats. Only year portion of bdate/edate used — whole calendar
    years returned. Includes design values for NAAQS compliance. Use for long-term trends.
    Filter (one group): state+county+site | state+county | state | cbsa_code | bbox
    Always pass pollutant_standard (see ground prompt table).
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    records, endpoint, _ = _fetch_summary(
        "annualData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code,
        minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
    )
    body = [
        {
            "year": r.get("year"),
            "site_id": _site_id(r),
            "arithmetic_mean": r.get("arithmetic_mean"),
            "minimum_value": r.get("minimum_value"),
            "maximum_value": r.get("maximum_value"),
            "percentile_25": r.get("first_quartile"),
            "percentile_75": r.get("third_quartile"),
            "percentile_98": r.get("ninety_eighth_percentile"),
            "design_value": r.get("design_value"),
            "units": r.get("units_of_measure"),
            "sample_duration": r.get("sample_duration"),
            "pollutant_standard": r.get("pollutant_standard"),
            "observation_count": r.get("observation_count"),
            "observation_percent": r.get("observation_percent"),
            "local_site_name": r.get("local_site_name"),
            "valid_day_count": r.get("valid_day_count"),
            "required_day_count": r.get("required_day_count"),
        }
        for r in records
    ]
    return {
        "Header": _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard),
        "Body": body,
    }
 
 
# ---------------------------------------------------------------------------
# Exceedance days tool
# ---------------------------------------------------------------------------
 
# Regulatory hard thresholds per param_code:
# (pollutant_standard, field_to_check, threshold_value, units)
_REGULATORY_THRESHOLDS = {
    "42602": ("NO2 1-hour 2010",   "first_max_value",   100.0,  "ppb"),
    "88101": ("PM25 24-hour 2024", "arithmetic_mean",    35.0,  "µg/m³"),
    "44201": ("Ozone 8-hour 2015", "first_max_value",    70.0,  "ppb"),
    "42401": ("SO2 1-hour 2010",   "first_max_value",    75.0,  "ppb"),
    "42101": ("CO 8-hour 1971",    "first_max_value",     9.0,  "ppm"),
}
 
 
@tool
def find_exceedance_days(
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    state_code: Optional[str] = None,
    county_code: Optional[str] = None,
    site_number: Optional[str] = None,
    cbsa_code: Optional[str] = None,
    minlat: Optional[Union[float, str]] = None,
    maxlat: Optional[Union[float, str]] = None,
    minlon: Optional[Union[float, str]] = None,
    maxlon: Optional[Union[float, str]] = None,
    hard_threshold: Optional[Union[float, str]] = None,
    percentile_threshold: Optional[Union[float, str]] = None,
) -> Dict[str, Any]:
    """
    Find days exceeding pollutant thresholds. No prior get_daily_summary needed.
    Results include date, value, aqi, triggered flag — ready to pass to satellite agent.
    Filter (one group): state+county+site | state+county | state | cbsa_code | bbox
    hard_threshold: fixed value (defaults to regulatory limit for known param_codes).
    percentile_threshold: top N% of period, e.g. 90.0 = top 10%. Both can combine.
    """
    # Coerce string inputs — the LLM occasionally passes numbers as strings
    if hard_threshold is not None:
        hard_threshold = float(hard_threshold)
    if percentile_threshold is not None:
        percentile_threshold = float(percentile_threshold)

    # Resolve the regulatory standard and measurement field for this param
    reg = _REGULATORY_THRESHOLDS.get(param_code)
    if reg is None and hard_threshold is None and percentile_threshold is None:
        raise ValueError(
            f"No regulatory threshold known for param_code '{param_code}'. "
            "Provide hard_threshold or percentile_threshold explicitly."
        )
 
    pollutant_standard = reg[0] if reg else None
    measurement_field  = reg[1] if reg else "first_max_value"
    regulatory_limit   = reg[2] if reg else None
 
    effective_hard = hard_threshold if hard_threshold is not None else regulatory_limit
 
    if effective_hard is None and percentile_threshold is None:
        raise ValueError("Provide at least one of: hard_threshold, percentile_threshold.")
 
    # Fetch daily summaries using the shared helper
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    records, endpoint, filter_params = _fetch_summary(
        "dailyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code,
        minlat, maxlat, minlon, maxlon, None, None, pollutant_standard,
    )
 
    # Extract the measurement value for each day
    def _val(r):
        v = r.get(measurement_field)
        return float(v) if v is not None else None
 
    values = [_val(r) for r in records]
    valid_values = [v for v in values if v is not None]
 
    # Compute percentile cutoff if requested
    percentile_cutoff = None
    if percentile_threshold is not None:
        if not (0 <= percentile_threshold <= 100):
            raise ValueError("percentile_threshold must be between 0 and 100.")
        sorted_vals = sorted(valid_values)
        idx = min(int(len(sorted_vals) * percentile_threshold / 100), len(sorted_vals) - 1)
        percentile_cutoff = sorted_vals[idx]
 
    # Flag days
    body = []
    for r, v in zip(records, values):
        if v is None:
            continue
        triggered = []
        if effective_hard is not None and v > effective_hard:
            triggered.append("hard")
        if percentile_cutoff is not None and v >= percentile_cutoff:
            triggered.append("percentile")
        if not triggered:
            continue
        body.append({
            "date": r.get("date_local"),
            "site_id": _site_id(r),
            "value": v,
            "aqi": r.get("aqi"),
            "triggered": triggered,
            "local_site_name": r.get("local_site_name"),
        })
 
    body.sort(key=lambda x: x["date"])
 
    return {
        "Header": [{
            "status": "success",
            "rows": len(body),
            "endpoint": endpoint,
            "param_code": param_code,
            "bdate": bdate_obj.isoformat(),
            "edate": edate_obj.isoformat(),
            "pollutant_standard": pollutant_standard,
            "measurement_field": measurement_field,
            "hard_threshold": effective_hard,
            "percentile_threshold": percentile_threshold,
            "percentile_cutoff_value": percentile_cutoff,
            "total_days_in_period": len(records),
        }],
        "Body": body,
    }
 
# ---------------------------------------------------------------------------
# Sample data (hourly readings)
# ---------------------------------------------------------------------------
 
@tool
def get_sample_data(
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    state_code: Optional[str] = None,
    county_code: Optional[str] = None,
    site_number: Optional[str] = None,
    cbsa_code: Optional[str] = None,
    minlat: Optional[Union[float, str]] = None,
    maxlat: Optional[Union[float, str]] = None,
    minlon: Optional[Union[float, str]] = None,
    maxlon: Optional[Union[float, str]] = None,
) -> Dict[str, Any]:
    """
    Retrieve raw hourly measurements from EPA AQS monitors.

    Use after find_exceedance_days to profile a flagged day (rush-hour spike vs
    overnight vs sustained). Keep date ranges short — 1–3 days max. For
    aggregated stats use get_daily_summary instead.

    Filter (one group, most specific wins):
    state_code + county_code + site_number | state_code + county_code
    | state_code | cbsa_code | minlat + maxlat + minlon + maxlon

    Returns hourly rows: site_id, datetime_local, date, hour, value, units,
    qualifier (null=clean), sample_duration, method, local_site_name.
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
 
    endpoint, filter_params = _resolve_filter(
        "sampleData", state_code, county_code, site_number,
        cbsa_code, minlat, maxlat, minlon, maxlon,
    )
 
    params = {
        "param": param_code,
        "bdate": bdate_str,
        "edate": edate_str,
        **filter_params,
    }
 
    data = _aqs_get(endpoint, params)
    records = data.get("Data", data.get("Body", []))
 
    if not records:
        raise RuntimeError(
            f"No sample data found for param {param_code} "
            f"between {bdate_obj.isoformat()} and {edate_obj.isoformat()} "
            f"using {endpoint} with {filter_params}."
        )
 
    body = [
        {
            "site_id": _site_id(r),
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
 
    # Sort by site then datetime
    body.sort(key=lambda x: (x["site_id"], x["datetime_local"]))
 
    return {
        "Header": [{
            "status": "success",
            "rows": len(body),
            "endpoint": endpoint,
            "param_code": param_code,
            "bdate": bdate_obj.isoformat(),
            "edate": edate_obj.isoformat(),
        }],
        "Body": body,
    } 
 
# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import json
 
    print("=== find closest monitor ===")
    monitor_result = find_closest_monitor.invoke(
        {"bdate": "2026-04-01", "edate": "2026-05-19", "location": "Tampa Florida", "param_code": "42602", "k": 3}
    )
    print(json.dumps(monitor_result, indent=2))
 
    closest = monitor_result["Body"][0]
 
    print("\n=== daily summary for closest monitor ===")
    summary_result = get_daily_summary.invoke({
        "state_code": closest["state_code"],
        "county_code": closest["county_code"],
        "site_number": closest["site_number"],
        "param_code": closest["param_code"],
        "bdate": monitor_result["Header"][0]["bdate"],
        "edate": monitor_result["Header"][0]["edate"],
        "pollutant_standard": "NO2 1-hour 2010",
    })
    print(json.dumps(summary_result, indent=2))
 
    print("\n=== exceedance days (regulatory + top 10%) ===")
    exceedance_result = find_exceedance_days.invoke({
        "state_code": closest["state_code"],
        "county_code": closest["county_code"],
        "site_number": closest["site_number"],
        "param_code": closest["param_code"],
        "bdate": monitor_result["Header"][0]["bdate"],
        "edate": monitor_result["Header"][0]["edate"],
        "percentile_threshold": 90.0,
    })
    print(json.dumps(exceedance_result, indent=2))
 
    print("\n=== hourly sample data for exceedance day ===")
    if exceedance_result["Body"]:
        first_exceedance = exceedance_result["Body"][0]
        sample_result = get_sample_data.invoke({
            "state_code": closest["state_code"],
            "county_code": closest["county_code"],
            "site_number": closest["site_number"],
            "param_code": closest["param_code"],
            "bdate": first_exceedance["date"],
            "edate": first_exceedance["date"],
        })
        print(json.dumps(sample_result, indent=2))
