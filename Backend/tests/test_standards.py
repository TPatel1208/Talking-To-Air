"""The standards file loads, rejects bad data, and maps concentrations to AQI categories."""
import math
import textwrap

import pytest

from tta_backend.datasets.standards import (
    AqiBreakpoint,
    OutOfTable,
    PollutantStandards,
    aqi_category,
    load_standards,
)

POLLUTANTS = {"o3", "pm25", "pm10", "no2", "so2", "co"}


def _block(**over):
    base = {
        "name": "Test",
        "unit": "ppb",
        "default_averaging": "1-hour",
        "decimals": 0,
        "naaqs": [{"level": 10, "unit": "ppb", "averaging_period": "1-hour", "form": "f"}],
        "breakpoints": [
            {"category": "Good", "aqi_low": 0, "aqi_high": 50, "concentration_low": 0,
             "concentration_high": 10, "averaging_period": "1-hour"},
            {"category": "Moderate", "aqi_low": 51, "aqi_high": 100, "concentration_low": 11,
             "concentration_high": 20, "averaging_period": "1-hour"},
        ],
    }
    return {**base, **over}


def test_all_pollutants_load():
    standards = load_standards()
    assert set(standards) == POLLUTANTS
    for info in standards.values():
        assert info.naaqs and info.breakpoints


@pytest.mark.parametrize(
    ("pollutant", "level", "unit", "period"),
    [
        ("o3", 0.070, "ppm", "8-hour"),
        ("pm25", 9.0, "ug/m3", "annual"),
        ("pm25", 35, "ug/m3", "24-hour"),
        ("pm10", 150, "ug/m3", "24-hour"),
        ("no2", 100, "ppb", "1-hour"),
        ("no2", 53, "ppb", "annual"),
        ("so2", 75, "ppb", "1-hour"),
        ("co", 9, "ppm", "8-hour"),
        ("co", 35, "ppm", "1-hour"),
    ],
)
def test_primary_standards(pollutant, level, unit, period):
    matches = [
        s for s in load_standards()[pollutant].naaqs
        if s.averaging_period == period
    ]
    assert len(matches) == 1
    assert (matches[0].level, matches[0].unit) == (level, unit)
    assert matches[0].form


def test_default_averaging_categories():
    # 8-hour ozone stops at Very Unhealthy and 1-hour SO2 at Unhealthy: higher
    # categories use another averaging period.
    for key, info in load_standards().items():
        rows = info.breakpoints_for(info.default_averaging)
        names = [r.category for r in rows]
        if key == "so2":
            assert names == ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy"]
        elif key == "o3":
            assert names[-1] == "Very Unhealthy" and len(names) == 5
        else:
            assert names == [
                "Good", "Moderate", "Unhealthy for Sensitive Groups",
                "Unhealthy", "Very Unhealthy", "Hazardous",
            ], key
        assert rows[0].aqi_low == 0 and rows[0].concentration_low == 0


def test_breakpoint_units_match_the_pollutant():
    for info in load_standards().values():
        assert all(s.unit == info.unit for s in info.naaqs)


# (pollutant, concentration, averaging period or None, expected category or OutOfTable)
CATEGORY_CASES = [
    ("pm25", 0, None, "Good"),
    ("pm25", 9.0, None, "Good"),
    ("pm25", 9.04, None, "Good"),
    ("pm25", 9.1, None, "Moderate"),
    ("pm25", 35.4, None, "Moderate"),
    ("pm25", 35.5, None, "Unhealthy for Sensitive Groups"),
    ("pm25", 55.4, None, "Unhealthy for Sensitive Groups"),
    ("pm25", 55.5, None, "Unhealthy"),
    ("pm25", 125.5, None, "Very Unhealthy"),
    ("pm25", 225.5, None, "Hazardous"),
    ("pm25", 325.4, None, "Hazardous"),
    ("pm25", 325.49, None, "Hazardous"),
    ("pm25", 325.5, None, OutOfTable.BEYOND),
    ("pm25", 900, None, OutOfTable.BEYOND),
    ("pm10", 54, None, "Good"),
    ("pm10", 54.9, None, "Good"),
    ("pm10", 55, None, "Moderate"),
    ("pm10", 154, None, "Moderate"),
    ("pm10", 155, None, "Unhealthy for Sensitive Groups"),
    ("pm10", 425, None, "Hazardous"),
    ("o3", 0.054, None, "Good"),
    ("o3", 0.0549, None, "Good"),
    ("o3", 0.055, None, "Moderate"),
    ("o3", 0.070, None, "Moderate"),
    ("o3", 0.0709, None, "Moderate"),
    ("o3", 0.071, None, "Unhealthy for Sensitive Groups"),
    ("o3", 0.086, None, "Unhealthy"),
    ("o3", 0.106, None, "Very Unhealthy"),
    ("o3", 0.2, None, "Very Unhealthy"),
    ("o3", 0.2009, None, "Very Unhealthy"),
    ("o3", 0.201, None, OutOfTable.BEYOND),
    ("o3", 0.3, None, OutOfTable.BEYOND),
    ("o3", 0.1, "1-hour", OutOfTable.BELOW),
    ("o3", 0.124, "1-hour", OutOfTable.BELOW),
    ("o3", -0.001, None, "Good"),
    ("o3", 0.605, "1-hour", OutOfTable.BEYOND),
    ("o3", 0.125, "1-hour", "Unhealthy for Sensitive Groups"),
    ("o3", 0.405, "1-hour", "Hazardous"),
    ("no2", 53, None, "Good"),
    ("no2", 53.9, None, "Good"),
    ("no2", 54, None, "Moderate"),
    ("no2", 100, None, "Moderate"),
    ("no2", 101, None, "Unhealthy for Sensitive Groups"),
    ("no2", 361, None, "Unhealthy"),
    ("no2", 1250, None, "Hazardous"),
    ("no2", 2050, None, OutOfTable.BEYOND),
    ("so2", 35, None, "Good"),
    ("so2", 36, None, "Moderate"),
    ("so2", 75, None, "Moderate"),
    ("so2", 76, None, "Unhealthy for Sensitive Groups"),
    ("so2", 186, None, "Unhealthy"),
    ("so2", 304, None, "Unhealthy"),
    ("so2", 304.9, None, "Unhealthy"),
    ("so2", 305, None, OutOfTable.BEYOND),
    ("so2", 400, None, OutOfTable.BEYOND),
    ("so2", -2, None, "Good"),
    ("so2", 305, "24-hour", "Very Unhealthy"),
    ("so2", 605, "24-hour", "Hazardous"),
    ("so2", 1005, "24-hour", OutOfTable.BEYOND),
    ("co", 4.4, None, "Good"),
    ("co", 4.45, None, "Good"),
    ("co", 4.5, None, "Moderate"),
    ("co", 9.4, None, "Moderate"),
    ("co", 9.5, None, "Unhealthy for Sensitive Groups"),
    ("co", 12.5, None, "Unhealthy"),
    ("co", 15.5, None, "Very Unhealthy"),
    ("co", 30.5, None, "Hazardous"),
]


@pytest.mark.parametrize(
    ("pollutant", "concentration", "period", "expected"),
    CATEGORY_CASES,
    ids=[f"{p}-{c}-{a or 'default'}" for p, c, a, _ in CATEGORY_CASES],
)
def test_aqi_category(pollutant, concentration, period, expected):
    result = aqi_category(pollutant, concentration, period)
    if isinstance(expected, OutOfTable):
        assert result is expected
    else:
        assert isinstance(result, AqiBreakpoint) and result.category == expected


def test_negative_concentration_is_zero():
    assert aqi_category("pm25", -5.0) == aqi_category("pm25", 0)
    assert aqi_category("o3", -0.001, "1-hour") is OutOfTable.BELOW


def test_category_carries_its_aqi_and_concentration_range():
    cat = aqi_category("pm25", 40.0)
    assert isinstance(cat, AqiBreakpoint)
    assert (cat.aqi_low, cat.aqi_high) == (101, 150)
    assert (cat.concentration_low, cat.concentration_high) == (35.5, 55.4)
    assert cat.averaging_period == "24-hour"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_aqi_category_rejects_bad_concentrations(bad):
    with pytest.raises(ValueError):
        aqi_category("pm25", bad)


def test_aqi_category_rejects_unknown_pollutant_and_period():
    with pytest.raises(ValueError, match="unknown pollutant"):
        aqi_category("pb", 1.0)
    with pytest.raises(ValueError, match="no '3-hour' breakpoints"):
        aqi_category("pm25", 1.0, "3-hour")


# Validation.

def test_valid_block_passes():
    assert PollutantStandards(**_block()).name == "Test"


@pytest.mark.parametrize(
    "over",
    [
        {"naaqs": []},
        {"breakpoints": []},
        {"decimals": -1},
        {"default_averaging": "24-hour"},
        {"unexpected": 1},
        {"naaqs": [{"level": 0, "unit": "ppb", "averaging_period": "1-hour", "form": "f"}]},
        {"naaqs": [{"level": 1, "unit": "ppb", "averaging_period": "1-hour"}]},
        {"breakpoints": [{"category": "Good", "aqi_low": 50, "aqi_high": 0, "concentration_low": 0,
                          "concentration_high": 1, "averaging_period": "1-hour"}]},
        {"breakpoints": [{"category": "Good", "aqi_low": 0, "aqi_high": 50, "concentration_low": 5,
                          "concentration_high": 1, "averaging_period": "1-hour"}]},
        {"breakpoints": [
            {"category": "A", "aqi_low": 0, "aqi_high": 50, "concentration_low": 0,
             "concentration_high": 10, "averaging_period": "1-hour"},
            {"category": "B", "aqi_low": 51, "aqi_high": 100, "concentration_low": 10,
             "concentration_high": 20, "averaging_period": "1-hour"}]},
        {"breakpoints": [
            {"category": "A", "aqi_low": 0, "aqi_high": 50, "concentration_low": 0,
             "concentration_high": 10, "averaging_period": "1-hour"},
            {"category": "B", "aqi_low": 50, "aqi_high": 100, "concentration_low": 11,
             "concentration_high": 20, "averaging_period": "1-hour"}]},
    ],
    ids=[
        "no_naaqs", "no_breakpoints", "negative_decimals", "default_without_breakpoints",
        "extra_key", "zero_level", "naaqs_missing_form", "aqi_reversed", "concentration_reversed",
        "concentration_overlap", "aqi_overlap",
    ],
)
def test_invalid_blocks_are_rejected(over):
    with pytest.raises(ValueError):
        PollutantStandards(**_block(**over))


def test_load_reports_every_bad_pollutant(tmp_path):
    path = tmp_path / "standards.yaml"
    path.write_text(textwrap.dedent("""\
        a:
          name: A
        b:
          name: B
        """), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        load_standards.__wrapped__(str(path))
    assert "[a]" in str(err.value) and "[b]" in str(err.value)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_standards.__wrapped__(str(tmp_path / "nope.yaml"))


def test_load_is_cached():
    assert load_standards() is load_standards()
