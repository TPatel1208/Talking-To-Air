"""
intent_router.py
----------------
Deterministic pre-classifier for user messages.

Classifies messages into GROUND, SATELLITE, BOTH, or LLM routing categories
without making any LLM calls. GROUND/SATELLITE messages are dispatched
directly to the corresponding sub-agent by the chat streaming service,
bypassing the supervisor entirely (T14 router fast path); BOTH/LLM messages
go to the supervisor unchanged.

Intent classes
--------------
GROUND    — monitor lookup, site info, daily/quarterly/annual summaries,
            exceedances, hourly profiles.  Satellite never required.
SATELLITE — TROPOMI/OMI/TEMPO/MODIS plots, spatial maps, column data.
            Ground never required.
BOTH      — explicit cross-source comparison or satellite confirmation of a
            ground event.
LLM       — ambiguous; let the supervisor LLM decide.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_GROUND_PATTERNS: list[str] = [
    # proximity lookups — allow up to two intervening words ("find the nearest", "locate a station")
    r"\b(nearest|closest|find|locate|where\s+is)\b.{0,20}\b(monitor|sensor|station|site)\b",
    # site/station info requests — allow plural ("details")
    r"\b(site|station)\s+(info|details?|information)\b",
    # data type keywords
    r"\b(daily|quarterly|annual)\s+(reading|summary|data|report|average)\b",
    r"\bexceedance(s)?\b",
    r"\bhourly\s+(profile|reading|data)\b",
    r"\bair\s+quality\s+(data|reading|levels?|index|report)\b",
    r"\bAQI\b",
    # explicit EPA/AQS references
    r"\b(EPA|AQS)\s+(monitor|data|station|sensor)\b",
    # pollutant + specific ground-sensor nouns (exclude "data" — too ambiguous with satellite)
    r"\b(NO2|PM2\.5|PM25|ozone|SO2|CO)\s+(monitor|sensor|station|site|reading|level)\b",
]

_SATELLITE_PATTERNS: list[str] = [
    # named satellite instruments
    r"\b(TROPOMI|OMI|TEMPO|MODIS)\b",
    # generic satellite requests
    r"\bsatellite\s+(map|plot|data|image|reading|measurement)\b",
    # explicit visualisation verbs (not "show" — too generic) paired with a known variable
    r"\b(plot|map|visualize|visualise|display|render)\b.{0,40}\b(NO2|ozone|aerosol|CO|SO2|HCHO|formaldehyde)\b",
    # gridded / column density terms — standalone, not combined with ground nouns
    r"\b(gridded|column\s+density|spatial\s+pattern|spatial\s+distribution)\b",
    # Harmony / NASA data pipeline
    r"\b(Harmony|NASA\s+data|satellite\s+granule)\b",
]

_CROSS_SOURCE_PATTERNS: list[str] = [
    r"\b(compare|comparison|versus|vs\.?)\b.{0,60}\b(ground|satellite|TROPOMI|EPA)\b",
    r"\b(ground|EPA).{0,60}\b(satellite|TROPOMI|OMI|TEMPO)\b",
    r"\bconfirm\b.{0,40}\b(ground|exceedance)\b.{0,40}\b(satellite|space)\b",
]

_GROUND_RE = re.compile("|".join(_GROUND_PATTERNS), re.I | re.S)
_SATELLITE_RE = re.compile("|".join(_SATELLITE_PATTERNS), re.I | re.S)
_CROSS_RE = re.compile("|".join(_CROSS_SOURCE_PATTERNS), re.I | re.S)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_intent(message: str) -> str:
    """
    Classify a user message into one of four routing categories.

    Returns
    -------
    "GROUND"    — only the ground sensor agent is needed
    "SATELLITE" — only the satellite agent is needed
    "BOTH"      — both agents required (explicit cross-source request)
    "LLM"       — ambiguous; let the supervisor LLM decide
    """
    if _CROSS_RE.search(message):
        return "BOTH"
    is_ground = bool(_GROUND_RE.search(message))
    is_satellite = bool(_SATELLITE_RE.search(message))
    if is_ground and not is_satellite:
        return "GROUND"
    if is_satellite and not is_ground:
        return "SATELLITE"
    if is_ground and is_satellite:
        return "BOTH"
    return "LLM"
