import logging
from dataclasses import dataclass

from langchain.tools import tool
from typing import Dict, Any, List, Optional, Union
from datetime import date, timedelta

from tta_backend.domain.monitors import aqs as aqs_client
from tta_backend.domain.monitors import parsing, queries
from tta_backend.domain.monitors.aqs import (
    AqsCredential,
    AqsCredentialsRejected as AqsCredentialsRejected,
    AqsNoDataError as AqsNoDataError,
    AqsUnavailable as AqsUnavailable,
)
from tta_backend.infrastructure.settings import get_settings
from tta_backend.services.artifact_store import artifact_store
from tta_backend.services.connector_credential_service import get_aqs_credential_injector
from tta_backend.utils.plotting import get_geocoding_service
from tta_backend.utils.streaming import current_user_id

logger = logging.getLogger(__name__)

# Shared singleton (utils.plotting.get_geocoding_service) — one cache and
# one rate limiter across every consumer (plot tools and this module), not
# a private instance that can't see the others' cached lookups.
geocoding_service = get_geocoding_service()

settings = get_settings()

AQS_EMAIL = settings.aqs_api_email
AQS_KEY = settings.aqs_api_key
DEFAULT_PARAM_CODE = "42602"  # NO2

# The client's request state, cache and rate limiters live in
# domain.monitors.aqs; these names are kept for callers of this module.
_response_cache = aqs_client._response_cache
_rate_limiters = aqs_client._rate_limiters
_limiter_for = aqs_client._limiter_for
_reset_aqs_request_state = aqs_client.reset_request_state
_MAX_CACHE_ENTRIES = aqs_client._MAX_CACHE_ENTRIES
_AQS_MAX_REQUESTS_PER_WINDOW = aqs_client._AQS_MAX_REQUESTS_PER_WINDOW
_AQS_WINDOW_SECONDS = aqs_client._AQS_WINDOW_SECONDS

_BBOX_EXPANSIONS = parsing.BBOX_EXPANSIONS
_SUMMARY_SITE_CAP = parsing.SUMMARY_SITE_CAP
_DAILY_DETAIL_DAY_CAP = 31

_enforce_min_bbox = parsing.enforce_min_bbox
_bbox_from_point = parsing.bbox_from_point
_positive_int = parsing.positive_int
_resolve_dates = parsing.resolve_dates
_no_data_message = parsing.no_data_message
_nearest_k = parsing.nearest_k
_build_body = parsing.build_body
_resolve_filter = parsing.resolve_filter
_normalise_numeric_filter = parsing.normalise_numeric_filter
_normalise_site_filter = parsing.normalise_site_filter
_site_id = parsing.site_id
_aggregate_summary_records = parsing.aggregate_summary_records


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _AqsCredential(AqsCredential):
    # The researcher whose epa_aqs connector supplied this pair; None for
    # the shared app credential.
    user_id: Optional[str] = None


def _shared_credential() -> _AqsCredential:
    return _AqsCredential(AQS_EMAIL, AQS_KEY)


async def _resolve_credential() -> _AqsCredential:
    """The caller's own epa_aqs connector pair when they have a live one,
    otherwise the shared app credential. A failing lookup degrades to the
    shared credential rather than taking ground data down with it."""
    user_id = current_user_id()
    if user_id is None:
        return _shared_credential()
    try:
        personal = await get_aqs_credential_injector().resolve_credentials(user_id)
    except Exception:
        logger.warning("aqs_credential_lookup_failed", extra={"_event": "aqs_credential_lookup_failed"})
        return _shared_credential()
    if personal is None:
        return _shared_credential()
    return _AqsCredential(personal.email, personal.key, user_id)


async def _aqs_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET request to the AQS API; raises on HTTP errors or unexpected statuses.

    Identical (endpoint, params) pairs are served from the response cache,
    which spans the tool calls of a turn and — for measurements EPA has
    already settled — later turns and other researchers too. A cache hit
    neither waits nor consumes a rate-limit slot, and needs no credential.

    A miss goes out on the caller's own epa_aqs connector pair when they
    have one. If EPA rejects it, the connector is flagged so the Connectors
    tab shows the error, and the call is retried once on the shared app
    credential.
    """
    cached = aqs_client.cached_response(endpoint, params)
    if cached is not None:
        return cached
    credential = await _resolve_credential()
    try:
        data = await aqs_client.request(endpoint, params, credential)
    except AqsCredentialsRejected:
        if credential.user_id is None:
            raise
        logger.warning("aqs_personal_key_rejected", extra={"_event": "aqs_personal_key_rejected"})
        await get_aqs_credential_injector().mark_invalid(credential.user_id)
        data = await aqs_client.request(endpoint, params, _shared_credential())
    else:
        if credential.user_id is not None:
            get_aqs_credential_injector().mark_used(credential.user_id)
    aqs_client.remember_response(endpoint, params, data)
    return data


async def _app_fetch(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # Looks _aqs_get up at call time, so patching it reaches every query.
    return await _aqs_get(endpoint, params)


async def validate_aqs_credentials(email: str, key: str) -> None:
    """Prove EPA accepts ``email``/``key`` with one cheap live call, never
    answered from the response cache (which is keyed without credentials).
    Raises AqsCredentialsRejected for a bad pair, AqsUnavailable when EPA
    gives no verdict."""
    await aqs_client.validate_credentials(AqsCredential(email, key))


def _no_data_response(message: str) -> Dict[str, Any]:
    """Structured empty-result tool response, not a raised exception.

    LangGraph's ToolNode only converts a ToolInvocationError (bad tool args)
    into a ToolMessage automatically; any other exception — including
    RuntimeError, and ToolException isn't special-cased in this version
    either — escapes _execute_tool_async and kills the whole agent turn
    (live 2026-08-07: a stale date banner made "this week" resolve into an
    unpublished window, and the resulting RuntimeError crashed the ground
    sensor agent instead of letting it read the message and apply its own
    monitor-fallback rules). Returning this dict keeps the message on the
    normal tool-result path the model actually sees.

    Reserved for an actual empty result set: :class:`AqsNoDataError`, or a
    monitor search that found no active monitor. See
    :func:`_upstream_error_response` for the other half.
    """
    return {"Header": [{"status": "no_data", "note": message}], "Body": []}


def _upstream_error_response(message: str) -> Dict[str, Any]:
    """Structured *failure* tool response — EPA was unreachable, rejected the
    credentials, or returned an error status.

    Kept off the exception path for the same reason as ``_no_data_response``:
    a raised RuntimeError kills the whole ground turn. But it must never wear
    that function's ``no_data`` status. "No monitors reported NO2 there" is a
    scientific claim; an AQS 503 is not evidence for it, and an agent handed
    ``no_data`` has no way to tell the difference. This status says plainly
    that the question is unanswered, so the model relays a service failure
    and can retry rather than reporting an absence that was never measured.
    """
    return {"Header": [{"status": "upstream_error", "note": message}], "Body": []}


async def _fetch_active_monitors(
    bbox: List[float], param_code: str, bdate_str: str, edate_str: str, k: int = 1,
) -> List[Dict[str, Any]]:
    return await queries.fetch_active_monitors(_app_fetch, bbox, param_code, bdate_str, edate_str, k)


# ---------------------------------------------------------------------------
# Proximity Tools
# ---------------------------------------------------------------------------

@tool
async def list_states() -> Dict[str, Any]:
    """
    Retrieve a list of all US states with EPA AQS air quality monitoring data.

    Returns each state with its 2-digit FIPS code, required for constructing
    other AQS API requests.

    Returns
    -------
    dict : raw AQS API response with Header and Data fields.
    """
    return await _aqs_get("list/states", {})


@tool
async def find_closest_monitor(
    location: str,
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    k: Union[int, str] = 1,
) -> Dict[str, Any]:
    """
    Find the closest active EPA AQS monitor to a location name or address.

    Use find_closest_monitor_by_coords instead if you already have lat/lon.

    Args:
        location   : City name or address (geocoded via Nominatim).
        param_code : AQS parameter code (default '42602' = NO2).
        bdate      : Start date YYYY-MM-DD (defaults to 1 year ago).
        edate      : End date YYYY-MM-DD (defaults to bdate).
        k          : Number of nearest monitors to return (default 1); numeric strings are accepted.

    Returns Body fields: station_id, station_name, monitor_name, latitude, longitude,
    distance_miles, state_code, county_code, site_number.
    """
    k = _positive_int(k, "k")
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)

    geo = await geocoding_service.ageocode(location)
    if geo is None:
        raise ValueError(f"Could not geocode location: '{location}'")
    lat_q, lon_q = geo["latitude"], geo["longitude"]

    # Clamp Nominatim bbox to minimum size before the expansion ladder
    bbox = _enforce_min_bbox(geo["bbox"])

    monitors = await _fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
    if not monitors:
        return _no_data_response(
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
async def find_closest_monitor_by_coords(
    latitude: Union[float, str],
    longitude: Union[float, str],
    param_code: str = DEFAULT_PARAM_CODE,
    bdate: Optional[str] = None,
    edate: Optional[str] = None,
    k: Union[int, str] = 1,
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
        k          : Number of nearest monitors to return (default 1); numeric strings are accepted.

    Returns Body fields: station_id, station_name, monitor_name, latitude, longitude,
    distance_miles, state_code, county_code, site_number.
    """
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError as exc:
        raise ValueError(f"Invalid latitude or longitude: '{latitude}', '{longitude}' must be float or castable to float.") from exc
    k = _positive_int(k, "k")
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)

    bbox = _bbox_from_point(latitude, longitude)
    monitors = await _fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
    if not monitors:
        return _no_data_response(
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

async def _fetch_summary(
    prefix: str,
    param_code: str,
    bdate_obj: date, edate_obj: date, bdate_str: str, edate_str: str,
    state_code: Optional[str], county_code: Optional[str], site_number: Optional[str],
    cbsa_code: Optional[str], minlat: Optional[Union[float, str]], maxlat: Optional[Union[float, str]], minlon: Optional[Union[float, str]], maxlon: Optional[Union[float, str]],
    cbdate: Optional[str], cedate: Optional[str], pollutant_standard: Optional[str],
) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Shared fetch + filter logic for daily, quarterly, and annual summaries.
    Returns (records, endpoint, filter_params).
    """
    return await queries.fetch_summary(
        _app_fetch, prefix, param_code, bdate_obj, edate_obj, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code, minlat, maxlat, minlon, maxlon,
        cbdate, cedate, pollutant_standard,
    )


def _build_summary_header(
    rows: int, endpoint: str, param_code: str, bdate_obj: date, edate_obj: date,
    pollutant_standard: Optional[str],
) -> List[Dict[str, Any]]:
    return [{
        "status": "success",
        "rows": rows,
        "endpoint": endpoint,
        "param_code": param_code,
        "bdate": bdate_obj.isoformat(),
        "edate": edate_obj.isoformat(),
        "pollutant_standard": pollutant_standard,
    }]


def _table_columns(rows: List[Dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _has_coordinates(rows: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(r, dict) and r.get("latitude") is not None and r.get("longitude") is not None
        for r in rows
    )


def _artifact_table_response(
    header: list[dict[str, Any]],
    body: list[dict[str, Any]],
    title: str,
    metadata: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    artifact_ref = artifact_store.put_table(
        title=title,
        columns=_table_columns(body),
        rows=body,
        # Every table this module produces comes from the EPA AQS API; stamp
        # the source dataset at this single artifact boundary so the frontend
        # Metadata tab's "Source dataset" field (tableOverviewFields reads
        # metadata.dataset) has a fact to show instead of "Not available".
        # Callers' own metadata still wins if they ever name it themselves.
        # has_coordinates tells the frontend the table can also be mapped; it
        # stays a "table" artifact, since "map" is the heatmap artifact type.
        metadata={"dataset": "EPA AQS", "has_coordinates": _has_coordinates(body), **(metadata or {})},
    )
    header[0]["artifact_count"] = 1
    header[0]["table_artifact_id"] = artifact_ref.id
    return {
        "Header": header,
        "Body": body,
        "_artifact_refs": [artifact_ref.model_dump(exclude_none=True)],
    }


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

@tool
async def get_daily_summary(
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
    Requests spanning more than 31 days are returned as quarterly period rows.
    Filter (one group): state+county+site | state+county | state | cbsa_code | bbox
    Always pass pollutant_standard (see ground prompt table).
    Returns period rows for capped sites: date/period, n_periods, mean, min, max,
    peak date/value, and coverage.
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    requested_period = "daily"
    fetch_prefix = "dailyData"
    detail_note = None
    if (edate_obj - bdate_obj).days + 1 > _DAILY_DETAIL_DAY_CAP:
        requested_period = "quarterly"
        fetch_prefix = "quarterlyData"
        detail_note = (
            f"Requested daily range exceeds {_DAILY_DETAIL_DAY_CAP} days; "
            "returning quarterly period rows instead. Call with a shorter range for daily detail."
        )

    try:
        records, endpoint, _ = await _fetch_summary(
            fetch_prefix, param_code, bdate_obj, edate_obj, bdate_str, edate_str,
            state_code, county_code, site_number, cbsa_code,
            minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
        )
    except AqsNoDataError as exc:
        return _no_data_response(str(exc))
    except RuntimeError as exc:
        return _upstream_error_response(str(exc))
    body, total_sites, total_periods = _aggregate_summary_records(records, requested_period)
    header = _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard)
    header[0]["total_sites_matched"] = total_sites
    header[0]["sites_returned"] = min(total_sites, _SUMMARY_SITE_CAP)
    header[0]["total_periods_matched"] = total_periods
    header[0]["periods_returned"] = len(body)
    header[0]["granularity"] = requested_period
    if detail_note:
        header[0]["note"] = detail_note
    return _artifact_table_response(
        header,
        body,
        title=f"EPA {requested_period.title()} Summary",
        metadata={"endpoint": endpoint, "granularity": requested_period},
    )


# ---------------------------------------------------------------------------
# Quarterly summary
# ---------------------------------------------------------------------------

@tool
async def get_quarterly_summary(
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
    Returns quarter rows for capped sites: year/quarter/period, n_periods, mean,
    min, max, peak quarter/value, and coverage.
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    try:
        records, endpoint, _ = await _fetch_summary(
            "quarterlyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
            state_code, county_code, site_number, cbsa_code,
            minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
        )
    except AqsNoDataError as exc:
        return _no_data_response(str(exc))
    except RuntimeError as exc:
        return _upstream_error_response(str(exc))
    body, total_sites, total_periods = _aggregate_summary_records(records, "quarterly")
    header = _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard)
    header[0]["total_sites_matched"] = total_sites
    header[0]["sites_returned"] = min(total_sites, _SUMMARY_SITE_CAP)
    header[0]["total_periods_matched"] = total_periods
    header[0]["periods_returned"] = len(body)
    header[0]["granularity"] = "quarterly"
    return _artifact_table_response(
        header,
        body,
        title="EPA Quarterly Summary",
        metadata={"endpoint": endpoint, "granularity": "quarterly"},
    )


# ---------------------------------------------------------------------------
# Annual summary
# ---------------------------------------------------------------------------

@tool
async def get_annual_summary(
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
    Returns annual rows for capped sites: year/period, n_periods, mean, min, max,
    peak year/value, and coverage.
    """
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    try:
        records, endpoint, _ = await _fetch_summary(
            "annualData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
            state_code, county_code, site_number, cbsa_code,
            minlat, maxlat, minlon, maxlon, cbdate, cedate, pollutant_standard,
        )
    except AqsNoDataError as exc:
        return _no_data_response(str(exc))
    except RuntimeError as exc:
        return _upstream_error_response(str(exc))
    body, total_sites, total_periods = _aggregate_summary_records(records, "annual")
    header = _build_summary_header(len(body), endpoint, param_code, bdate_obj, edate_obj, pollutant_standard)
    header[0]["total_sites_matched"] = total_sites
    header[0]["sites_returned"] = min(total_sites, _SUMMARY_SITE_CAP)
    header[0]["total_periods_matched"] = total_periods
    header[0]["periods_returned"] = len(body)
    header[0]["granularity"] = "annual"
    return _artifact_table_response(
        header,
        body,
        title="EPA Annual Summary",
        metadata={"endpoint": endpoint, "granularity": "annual"},
    )


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
async def find_exceedance_days(
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

    # Fetch daily summaries using the shared helper.
    # dailyData endpoint rejects ranges longer than ~1 year; cap at 365 days.
    bdate_obj, edate_obj, bdate_str, edate_str = _resolve_dates(bdate, edate)
    if (edate_obj - bdate_obj).days > 365:
        edate_obj = bdate_obj + timedelta(days=365)
        edate_str = edate_obj.strftime("%Y%m%d")
    try:
        records, endpoint, filter_params = await _fetch_summary(
            "dailyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
            state_code, county_code, site_number, cbsa_code,
            minlat, maxlat, minlon, maxlon, None, None, pollutant_standard,
        )
    except AqsNoDataError as exc:
        return _no_data_response(str(exc))
    except RuntimeError as exc:
        return _upstream_error_response(str(exc))

    body, percentile_cutoff = parsing.flag_exceedances(
        records, measurement_field, effective_hard, percentile_threshold
    )

    header = [{
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
    }]
    return _artifact_table_response(
        header,
        body,
        title="EPA Exceedance Days",
        metadata={"endpoint": endpoint, "measurement_field": measurement_field},
    )

# ---------------------------------------------------------------------------
# Sample data (hourly readings)
# ---------------------------------------------------------------------------

@tool
async def get_sample_data(
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

    records, endpoint, filter_params = await queries.fetch_samples(
        _app_fetch, param_code, bdate_str, edate_str,
        state_code, county_code, site_number, cbsa_code, minlat, maxlat, minlon, maxlon,
    )

    if not records:
        logger.info("aqs_no_data endpoint=%s params=%r", endpoint, filter_params)
        return _no_data_response(
            _no_data_message("hourly sample", param_code, bdate_obj, edate_obj)
        )

    body = parsing.sample_rows(records)

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
