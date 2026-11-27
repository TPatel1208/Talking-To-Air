"""AQS queries run through a Fetch, so the caller decides which credential pays for them."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from tta_backend.domain.monitors.aqs import AqsNoDataError, Fetch
from tta_backend.domain.monitors.parsing import (
    BBOX_EXPANSIONS,
    SUMMARY_KINDS,
    expand_bbox,
    no_data_message,
    resolve_filter,
    site_id,
)

logger = logging.getLogger(__name__)


def records_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of an AQS response."""
    rows: list[dict[str, Any]] = data.get("Data", data.get("Body", []))
    return rows


async def fetch_monitors_in_box(
    fetch: Fetch,
    bbox: list[float],
    param_code: str,
    bdate_str: str,
    edate_str: str,
) -> list[dict[str, Any]]:
    """Monitors measuring ``param_code`` in the [south, north, west, east] bbox, as given."""
    south, north, west, east = bbox
    data = await fetch(
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
    return records_of(data)


async def fetch_active_monitors(
    fetch: Fetch,
    bbox: list[float],
    param_code: str,
    bdate_str: str,
    edate_str: str,
    k: int = 1,
    *,
    count_sites: bool = False,
) -> list[dict[str, Any]]:
    """Monitors measuring ``param_code`` in the [south, north, west, east] bbox,
    widening it step by step until at least ``k`` are found. Monitors are
    counted one per row (instrument), or one per site with ``count_sites``."""

    def found(rows: list[dict[str, Any]]) -> int:
        return len({site_id(r) for r in rows}) if count_sites else len(rows)

    best: list[dict[str, Any]] = []
    for expansion in BBOX_EXPANSIONS:
        monitors = await fetch_monitors_in_box(fetch, expand_bbox(bbox, expansion), param_code, bdate_str, edate_str)
        if found(monitors) > found(best):
            best = monitors
        if found(best) >= k:
            return best

    return best


async def fetch_summary(
    fetch: Fetch,
    prefix: str,
    param_code: str,
    bdate_obj: date, edate_obj: date, bdate_str: str, edate_str: str,
    state_code: Any, county_code: Any, site_number: Any,
    cbsa_code: Any, minlat: Any, maxlat: Any, minlon: Any, maxlon: Any,
    cbdate: str | None, cedate: str | None, pollutant_standard: str | None,
    *,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Daily, quarterly or annual records for one filter group, optionally for
    one pollutant standard. Returns (records, endpoint, filter params).

    Raises AqsNoDataError, in plain language, when nothing matches.
    """
    endpoint, filter_params = resolve_filter(
        prefix, state_code, county_code, site_number,
        cbsa_code, minlat, maxlat, minlon, maxlon,
    )
    params = {"param": param_code, "bdate": bdate_str, "edate": edate_str, **filter_params}
    if cbdate:
        params["cbdate"] = date.fromisoformat(cbdate).strftime("%Y%m%d")
    if cedate:
        params["cedate"] = date.fromisoformat(cedate).strftime("%Y%m%d")

    data = await fetch(endpoint, params)
    records = records_of(data)

    if pollutant_standard:
        records = [r for r in records if r.get("pollutant_standard") == pollutant_standard]

    if not records:
        # Endpoint and params go to the log, never into the message.
        logger.info(
            "aqs_no_data endpoint=%s params=%r pollutant_standard=%r",
            endpoint, filter_params, pollutant_standard,
        )
        raise AqsNoDataError(
            no_data_message(SUMMARY_KINDS.get(prefix, prefix), param_code, bdate_obj, edate_obj, today=today)
        )
    return records, endpoint, filter_params


async def fetch_samples(
    fetch: Fetch,
    param_code: str,
    bdate_str: str,
    edate_str: str,
    state_code: Any, county_code: Any, site_number: Any,
    cbsa_code: Any, minlat: Any, maxlat: Any, minlon: Any, maxlon: Any,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Raw hourly sample records for one filter group; may be empty.
    Returns (records, endpoint, filter params)."""
    endpoint, filter_params = resolve_filter(
        "sampleData", state_code, county_code, site_number,
        cbsa_code, minlat, maxlat, minlon, maxlon,
    )
    params = {"param": param_code, "bdate": bdate_str, "edate": edate_str, **filter_params}
    data = await fetch(endpoint, params)
    return records_of(data), endpoint, filter_params
