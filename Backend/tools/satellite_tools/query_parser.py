import calendar
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalRange:
    start: str
    end: str


@dataclass(frozen=True)
class ParsedSatelliteQuery:
    variable: str
    location: str
    temporal: TemporalRange


_MONTH_LOOKUP = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
_MONTH_LOOKUP.update(
    {
        name.lower(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    }
)
_MONTH_PATTERN = "|".join(
    sorted((re.escape(name) for name in _MONTH_LOOKUP), key=len, reverse=True)
)

_TEMPORAL_KEYWORDS = {
    "date",
    "day",
    "month",
    "monthly",
    "year",
    "during",
    "between",
    "through",
    "from",
    "until",
}

_VARIABLE_ALIASES = {
    "TROPOMI_NO2": ("tropomi no2", "tropomi_no2", "satellite no2", "no2", "nitrogen dioxide"),
    "OMI_NO2": ("omi no2", "omi_no2"),
    "TEMPO_NO2": ("tempo no2", "tempo_no2"),
    "TEMPO_O3TOT": ("tempo o3tot", "tempo ozone", "tempo_o3tot"),
    "OMI_O3": ("omi o3", "omi ozone", "omi_o3", "ozone", "o3"),
    "TEMPO_HCHO": ("tempo hcho", "tempo_hcho"),
    "TEMPO_HCHO_V03": ("tempo hcho v03", "tempo_hcho_v03"),
    "OMI_HCHO": ("omi hcho", "omi_hcho", "formaldehyde", "hcho"),
    "MODIS_AOD_TERRA": ("modis aod terra", "modis_aod_terra"),
    "MODIS_AOD_AQUA": ("modis aod aqua", "modis_aod_aqua"),
}


def parse_satellite_plot_query(task: str) -> ParsedSatelliteQuery | None:
    text = _normalize_space(task)
    if not text:
        return None

    temporal, text_without_temporal = _extract_temporal(text)
    if temporal is None:
        return None

    variable = _extract_variable(text)
    if variable is None:
        return None

    location = _extract_location(text_without_temporal)
    if not is_valid_location_candidate(location):
        return None

    return ParsedSatelliteQuery(variable=variable, location=location, temporal=temporal)


def is_valid_location_candidate(location: str | None) -> bool:
    if not location:
        return False

    candidate = _normalize_space(location).strip(" .,")
    if not candidate:
        return False

    lower = candidate.lower()
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lower):
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", lower):
        return False
    if re.search(rf"\b(?:{_MONTH_PATTERN})\b", lower, flags=re.IGNORECASE):
        return False
    if any(re.search(rf"\b{re.escape(keyword)}\b", lower) for keyword in _TEMPORAL_KEYWORDS):
        return False

    return True


def _extract_temporal(text: str) -> tuple[TemporalRange | None, str]:
    extractors = (
        _extract_iso_range,
        _extract_month_range,
        _extract_month_year,
        _extract_iso_day,
    )
    for extractor in extractors:
        result = extractor(text)
        if result is not None:
            temporal, span = result
            return temporal, _remove_span(text, span)
    return None, text


def _extract_iso_range(text: str) -> tuple[TemporalRange, tuple[int, int]] | None:
    match = re.search(
        r"\b(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|through|and|-)\s+(\d{4}-\d{2}-\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = match.group(1), match.group(2)
    return TemporalRange(f"{start}T00:00:00Z", f"{end}T23:59:59Z"), match.span()


def _extract_iso_day(text: str) -> tuple[TemporalRange, tuple[int, int]] | None:
    match = re.search(
        r"\b(?:(?:for|on|during)\s+)?(\d{4}-\d{2}-\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    day = match.group(1)
    return TemporalRange(f"{day}T00:00:00Z", f"{day}T23:59:59Z"), match.span()


def _extract_month_year(text: str) -> tuple[TemporalRange, tuple[int, int]] | None:
    match = re.search(
        rf"\b(?:(?:for|on|during|in)\s+)?(?:the\s+month\s+of\s+)?({_MONTH_PATTERN})(?:\s+of)?\s+((?:19|20)\d{{2}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month = _MONTH_LOOKUP[match.group(1).lower()]
    year = int(match.group(2))
    return _month_temporal(year, month), match.span()


def _extract_month_range(text: str) -> tuple[TemporalRange, tuple[int, int]] | None:
    match = re.search(
        rf"\bfrom\s+({_MONTH_PATTERN})(?:\s+((?:19|20)\d{{2}}))?\s+(?:to|through|-)\s+({_MONTH_PATTERN})(?:\s+((?:19|20)\d{{2}}))\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    start_month = _MONTH_LOOKUP[match.group(1).lower()]
    end_month = _MONTH_LOOKUP[match.group(3).lower()]
    end_year = int(match.group(4))
    start_year = int(match.group(2) or end_year)
    start = _month_temporal(start_year, start_month).start
    end = _month_temporal(end_year, end_month).end
    return TemporalRange(start, end), match.span()


def _month_temporal(year: int, month: int) -> TemporalRange:
    last_day = calendar.monthrange(year, month)[1]
    return TemporalRange(
        f"{year:04d}-{month:02d}-01T00:00:00Z",
        f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59Z",
    )


def _extract_variable(text: str) -> str | None:
    lower = text.lower()
    alias_pairs = [
        (variable, alias)
        for variable, aliases in _VARIABLE_ALIASES.items()
        for alias in aliases
    ]
    alias_pairs.sort(key=lambda item: len(item[1]), reverse=True)
    for variable, alias in alias_pairs:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower):
            return variable
    return None


def _extract_location(text: str) -> str | None:
    match = re.search(r"\b(?:over|in|near|around|for)\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None

    candidate = match.group(1).strip(" .,")
    candidate = re.sub(r"\s+\b(?:please|thanks|thank you)\b.*$", "", candidate, flags=re.IGNORECASE)
    return _normalize_space(candidate).strip(" .,")


def _remove_span(text: str, span: tuple[int, int]) -> str:
    before = text[: span[0]].rstrip(" ,.")
    after = text[span[1] :].lstrip(" ,.")
    return _normalize_space(f"{before} {after}").strip()


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())
