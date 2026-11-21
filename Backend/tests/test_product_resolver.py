"""The product resolver chooses by its rules, discloses why, and never substitutes."""
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from tta_backend.datasets.registry import CollectionConfig, CoverageBox, load_registry
from tta_backend.harness.capabilities.products import (
    REGISTRY_DEFAULT,
    BBox,
    DateRange,
    resolve_product,
)

NO2 = "no2_tropospheric_column"
TODAY = date(2026, 11, 14)
YESTERDAY = TODAY - timedelta(days=1)
PAST = date(2025, 6, 1)

NORTH_AMERICA = {"west": -140, "south": 10, "east": -40, "north": 70}
NYC = BBox(-75, 40, -73, 41)
EUROPE = BBox(0, 45, 10, 55)
STRADDLES_EAST_EDGE = BBox(-50, 30, -30, 40)


def _cfg(**over):
    base = {
        "collection_id": "C1-TEST",
        "primary_var": "v",
        "units": "u",
        "fill_value": -1,
        "valid_min": 0,
        "valid_max": 1,
    }
    return CollectionConfig(**{**base, **over})


REGISTRY = {
    "NO2_STD": _cfg(
        short_name="TEMPO_NO2_L3", instrument="TEMPO", cadence="hourly",
        quantity=NO2, coverage=NORTH_AMERICA, record_start=date(2023, 8, 1),
        stream="standard", latency_days=3, rank=1,
    ),
    "NO2_NRT": _cfg(
        short_name="TEMPO_NO2_L3_NRT", instrument="TEMPO", cadence="hourly",
        quantity=NO2, coverage=NORTH_AMERICA, record_start=date(2024, 9, 1),
        stream="near_real_time", latency_days=1, rank=2,
    ),
    "NO2_GLOBAL_DAILY": _cfg(
        short_name="OMI_NO2d", instrument="OMI/Aura", cadence="daily",
        quantity=NO2, coverage="global", record_start=date(2004, 10, 1),
        stream="standard", latency_days=2, rank=3,
    ),
    "NO2_MONTHLY": _cfg(
        short_name="TROPOMI_NO2_M", instrument="TROPOMI/Sentinel-5P", cadence="monthly",
        quantity=NO2, coverage="global", record_start=date(2018, 5, 1),
        record_end=date(2025, 12, 31), stream="standard", latency_days=30, rank=4,
    ),
    "HCHO_STD": _cfg(
        short_name="TEMPO_HCHO_L3", instrument="TEMPO", cadence="hourly",
        quantity="hcho_total_column", coverage=NORTH_AMERICA,
        record_start=date(2023, 8, 1), latency_days=3, rank=1,
    ),
    # No product-choice facts at all.
    "BARE": _cfg(short_name="BARE_L3", instrument="TEMPO", cadence="hourly"),
    # A quantity but no coverage.
    "PARTIAL": _cfg(short_name="PARTIAL_L3", instrument="GEMS", quantity=NO2),
}

# A near-real-time product keeping a rolling 14-day window, beside one that
# keeps its whole record.
ROLLING = {
    "NO2_ROLL": _cfg(
        short_name="ROLL_NRT", instrument="TEMPO", cadence="hourly",
        quantity=NO2, coverage=NORTH_AMERICA, record_start=date(2025, 9, 17),
        stream="near_real_time", latency_days=0, retention_days=14, rank=1,
    ),
    "NO2_KEEP": _cfg(
        short_name="KEEP_L3", instrument="TEMPO", cadence="hourly",
        quantity=NO2, coverage=NORTH_AMERICA, record_start=date(2023, 8, 1),
        stream="standard", latency_days=3, rank=2,
    ),
}
OLDEST_HELD = TODAY - timedelta(days=14)
AGED_OUT = OLDEST_HELD - timedelta(days=1)


def _resolve(registry=None, **over):
    args = {
        "quantity": NO2,
        "area": NYC,
        "dates": PAST,
        "registry": REGISTRY if registry is None else registry,
        "today": TODAY,
        "approved": REGISTRY_DEFAULT,
    }
    return resolve_product(**{**args, **over})


# id, resolver arguments, expected (product, rule, candidates_from, unavailable,
# no_data), and the exact reasons expected for each product that was ruled out.
CASES = [
    # Registry default: facts rule out, rank decides.
    ("default_past_highest_rank", {}, ("NO2_STD", "highest_rank", "registry_default", None, None),
     {"PARTIAL": {"facts_missing"}}),
    ("default_yesterday_nrt", {"dates": YESTERDAY},
     ("NO2_NRT", "only_candidate", "registry_default", None, None),
     {"NO2_STD": {"not_published_yet"}, "NO2_GLOBAL_DAILY": {"not_published_yet"},
      "NO2_MONTHLY": {"outside_record"}, "PARTIAL": {"facts_missing"}}),
    ("default_area_outside", {"area": EUROPE, "cadence_needed": "daily"},
     ("NO2_GLOBAL_DAILY", "only_candidate", "registry_default", None, None),
     {"NO2_STD": {"area_outside_coverage"}, "NO2_NRT": {"area_outside_coverage"},
      "NO2_MONTHLY": {"cadence_too_coarse"}, "PARTIAL": {"facts_missing"}}),
    ("default_area_partly_outside", {"area": STRADDLES_EAST_EDGE},
     ("NO2_GLOBAL_DAILY", "highest_rank", "registry_default", None, None),
     {"NO2_STD": {"area_partly_outside_coverage"}, "NO2_NRT": {"area_partly_outside_coverage"},
      "PARTIAL": {"facts_missing"}}),
    ("default_no_entry_for_quantity", {"quantity": "so2_column"},
     (None, None, "registry_default", "no_registered_product", None), {}),
    ("default_range_to_today_not_published", {"dates": DateRange(date(2026, 11, 1), TODAY)},
     (None, None, "registry_default", "nothing_fits", "not_published_yet"),
     {"NO2_STD": {"not_published_yet"}, "NO2_NRT": {"not_published_yet"},
      "NO2_GLOBAL_DAILY": {"not_published_yet"}, "NO2_MONTHLY": {"outside_record"},
      "PARTIAL": {"facts_missing"}}),
    ("default_past_record_end", {"dates": date(2026, 3, 1), "area": EUROPE},
     ("NO2_GLOBAL_DAILY", "only_candidate", "registry_default", None, None),
     {"NO2_STD": {"area_outside_coverage"}, "NO2_NRT": {"area_outside_coverage"},
      "NO2_MONTHLY": {"outside_record"}, "PARTIAL": {"facts_missing"}}),

    # Approved list: preference order decides, registry facts still rule out.
    ("approved_preference", {"approved": ["NO2_STD", "NO2_NRT"]},
     ("NO2_STD", "approved_preference", "approved", None, None), {}),
    ("approved_preference_beats_rank", {"approved": ["NO2_NRT", "NO2_STD"]},
     ("NO2_NRT", "approved_preference", "approved", None, None), {}),
    ("approved_yesterday_moves_to_nrt", {"approved": ["NO2_STD", "NO2_NRT"], "dates": YESTERDAY},
     ("NO2_NRT", "only_candidate", "approved", None, None), {"NO2_STD": {"not_published_yet"}}),
    ("approved_standard_only_yesterday", {"approved": ["NO2_STD"], "dates": YESTERDAY},
     (None, None, "approved", "nothing_fits", "not_published_yet"), {"NO2_STD": {"not_published_yet"}}),
    ("approved_before_launch_moves_on", {"approved": ["NO2_STD", "NO2_GLOBAL_DAILY"], "dates": date(2020, 1, 1)},
     ("NO2_GLOBAL_DAILY", "only_candidate", "approved", None, None), {"NO2_STD": {"outside_record"}}),
    ("approved_outside_every_record", {"approved": ["NO2_STD"], "dates": date(2000, 1, 1)},
     (None, None, "approved", "nothing_fits", "outside_record"), {"NO2_STD": {"outside_record"}}),
    ("approved_cadence_too_coarse", {"approved": ["NO2_GLOBAL_DAILY"], "cadence_needed": "hourly"},
     (None, None, "approved", "nothing_fits", None), {"NO2_GLOBAL_DAILY": {"cadence_too_coarse"}}),
    ("approved_wrong_quantity", {"approved": ["HCHO_STD"]},
     (None, None, "approved", "nothing_fits", None), {"HCHO_STD": {"wrong_quantity"}}),
    ("approved_empty", {"approved": []}, (None, None, "approved", "no_approved_products", None), {}),
    ("approved_not_registered", {"approved": ["NOPE", "NO2_STD"]},
     ("NO2_STD", "only_candidate", "approved", None, None), {"NOPE": {"not_registered"}}),
    ("approved_bare_entry_is_vouched", {"approved": ["BARE"]},
     ("BARE", "only_candidate", "approved", None, None), {}),
    ("approved_partial_entry_is_vouched", {"approved": ["PARTIAL"]},
     ("PARTIAL", "only_candidate", "approved", None, None), {}),
    ("approved_bare_future_date", {"approved": ["BARE"], "dates": TODAY + timedelta(days=1)},
     (None, None, "approved", "nothing_fits", "not_published_yet"), {"BARE": {"not_published_yet"}}),

    # Dates as a range or a list: every day must be served.
    ("range_partly_before_nrt_record", {"approved": ["NO2_NRT", "NO2_STD"], "dates": DateRange(date(2024, 8, 1), date(2024, 10, 1))},
     ("NO2_STD", "only_candidate", "approved", None, None), {"NO2_NRT": {"outside_record"}}),
    ("range_recent_nrt", {"approved": ["NO2_STD", "NO2_NRT"], "dates": DateRange(date(2026, 11, 1), YESTERDAY)},
     ("NO2_NRT", "only_candidate", "approved", None, None), {"NO2_STD": {"not_published_yet"}}),
    ("list_spanning_launch", {"approved": ["NO2_STD", "NO2_GLOBAL_DAILY"], "dates": [PAST, date(2023, 1, 1)]},
     ("NO2_GLOBAL_DAILY", "only_candidate", "approved", None, None), {"NO2_STD": {"outside_record"}}),

    # User named a product or instrument: used, or unavailable, never another.
    ("named_key", {"named": "NO2_GLOBAL_DAILY"},
     ("NO2_GLOBAL_DAILY", "user_named", "user_named", None, None), {}),
    ("named_short_name", {"named": "tempo_no2_l3_nrt"},
     ("NO2_NRT", "user_named", "user_named", None, None), {}),
    ("named_key_unsuitable_no_fallthrough", {"named": "NO2_STD", "dates": YESTERDAY},
     (None, None, "user_named", "named_product_unsuitable", "not_published_yet"),
     {"NO2_STD": {"not_published_yet"}}),
    ("named_not_approved", {"named": "NO2_GLOBAL_DAILY", "approved": ["NO2_STD"]},
     (None, None, "user_named", "named_product_unsuitable", None), {"NO2_GLOBAL_DAILY": {"not_approved"}}),
    ("named_instrument_rank", {"named": "TEMPO"},
     ("NO2_STD", "highest_rank", "user_named", None, None),
     {"HCHO_STD": {"wrong_quantity"}, "BARE": {"facts_missing"}}),
    ("named_instrument_yesterday", {"named": "tempo", "dates": YESTERDAY},
     ("NO2_NRT", "only_candidate", "user_named", None, None),
     {"NO2_STD": {"not_published_yet"}, "HCHO_STD": {"wrong_quantity", "not_published_yet"},
      "BARE": {"facts_missing"}}),
    ("named_instrument_approved_order", {"named": "TEMPO", "approved": ["NO2_NRT", "NO2_STD"]},
     ("NO2_NRT", "approved_preference", "user_named", None, None),
     {"HCHO_STD": {"not_approved", "wrong_quantity"}, "BARE": {"not_approved", "facts_missing"}}),
    ("named_instrument_outside_area", {"named": "TEMPO", "area": EUROPE},
     (None, None, "user_named", "named_product_unsuitable", None),
     {"NO2_STD": {"area_outside_coverage"}, "NO2_NRT": {"area_outside_coverage"},
      "HCHO_STD": {"wrong_quantity", "area_outside_coverage"}, "BARE": {"facts_missing"}}),
    ("named_platform_part", {"named": "aura"},
     ("NO2_GLOBAL_DAILY", "only_candidate", "user_named", None, None), {}),
    ("named_unknown", {"named": "MODIS"}, (None, None, "user_named", "named_not_registered", None), {}),
    ("named_bare_entry", {"named": "BARE"}, ("BARE", "user_named", "user_named", None, None), {}),
    ("named_instrument_partial_unvouched", {"named": "GEMS"},
     (None, None, "user_named", "named_product_unsuitable", None), {"PARTIAL": {"facts_missing"}}),

    # Pinned product: used, or unavailable.
    ("pinned", {"pinned": "NO2_STD"}, ("NO2_STD", "pinned", "pinned", None, None), {}),
    ("pinned_unsuitable", {"pinned": "NO2_STD", "dates": YESTERDAY, "approved": ["NO2_STD", "NO2_NRT"]},
     (None, None, "pinned", "pinned_product_unsuitable", "not_published_yet"), {"NO2_STD": {"not_published_yet"}}),
    ("pinned_not_approved", {"pinned": "NO2_GLOBAL_DAILY", "approved": ["NO2_STD"]},
     (None, None, "pinned", "pinned_product_unsuitable", None), {"NO2_GLOBAL_DAILY": {"not_approved"}}),
    ("pinned_named_conflict", {"pinned": "NO2_STD", "named": "NO2_NRT"},
     (None, None, "pinned", "named_conflicts_with_pin", None), {}),
    ("pinned_named_unknown", {"pinned": "NO2_STD", "named": "MODIS"},
     (None, None, "pinned", "named_not_registered", None), {}),
    ("pinned_inside_named_instrument", {"pinned": "NO2_STD", "named": "TEMPO"},
     ("NO2_STD", "pinned", "pinned", None, None), {}),

    # Rolling archive: days older than today minus retention_days are gone.
    ("retention_inside_window", {"registry": ROLLING, "dates": YESTERDAY},
     ("NO2_ROLL", "only_candidate", "registry_default", None, None),
     {"NO2_KEEP": {"not_published_yet"}}),
    ("retention_on_boundary", {"registry": ROLLING, "dates": OLDEST_HELD},
     ("NO2_ROLL", "highest_rank", "registry_default", None, None), {}),
    ("retention_older_moves_on", {"registry": ROLLING, "dates": AGED_OUT},
     ("NO2_KEEP", "only_candidate", "registry_default", None, None),
     {"NO2_ROLL": {"outside_retention"}}),
    ("retention_older_only_product", {"registry": ROLLING, "dates": AGED_OUT, "approved": ["NO2_ROLL"]},
     (None, None, "approved", "nothing_fits", "outside_record"), {"NO2_ROLL": {"outside_retention"}}),
    ("retention_range_straddles_boundary",
     {"registry": ROLLING, "dates": DateRange(AGED_OUT, YESTERDAY), "approved": ["NO2_ROLL"]},
     (None, None, "approved", "nothing_fits", "outside_record"), {"NO2_ROLL": {"outside_retention"}}),
    ("retention_range_starting_on_boundary",
     {"registry": ROLLING, "dates": DateRange(OLDEST_HELD, YESTERDAY), "approved": ["NO2_ROLL"]},
     ("NO2_ROLL", "only_candidate", "approved", None, None), {}),
    ("retention_straddle_and_unpublished_elsewhere",
     {"registry": ROLLING, "dates": DateRange(AGED_OUT, YESTERDAY), "approved": ["NO2_ROLL", "NO2_KEEP"]},
     (None, None, "approved", "nothing_fits", "not_published_yet"),
     {"NO2_ROLL": {"outside_retention"}, "NO2_KEEP": {"not_published_yet"}}),
    ("retention_aged_out_and_unpublished",
     {"registry": ROLLING, "dates": DateRange(AGED_OUT, TODAY + timedelta(days=1)), "approved": ["NO2_ROLL"]},
     (None, None, "approved", "nothing_fits", "outside_record"),
     {"NO2_ROLL": {"outside_retention", "not_published_yet"}}),
    ("retention_before_record_is_outside_record", {"registry": ROLLING, "dates": PAST},
     ("NO2_KEEP", "only_candidate", "registry_default", None, None), {"NO2_ROLL": {"outside_record"}}),
    ("retention_named_aged_out_not_substituted", {"registry": ROLLING, "dates": AGED_OUT, "named": "NO2_ROLL"},
     (None, None, "user_named", "named_product_unsuitable", "outside_record"),
     {"NO2_ROLL": {"outside_retention"}}),
    ("retention_pinned_aged_out_not_substituted",
     {"registry": ROLLING, "dates": AGED_OUT, "pinned": "NO2_ROLL", "approved": ["NO2_ROLL", "NO2_KEEP"]},
     (None, None, "pinned", "pinned_product_unsuitable", "outside_record"),
     {"NO2_ROLL": {"outside_retention"}}),
    ("retention_approved_aged_out_moves_on",
     {"registry": ROLLING, "dates": AGED_OUT, "approved": ["NO2_ROLL", "NO2_KEEP"]},
     ("NO2_KEEP", "only_candidate", "approved", None, None), {"NO2_ROLL": {"outside_retention"}}),
]


@pytest.mark.parametrize(("args", "expected", "ruled_out"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_resolution(args, expected, ruled_out):
    result = _resolve(**args)
    assert (result.product, result.rule, result.candidates_from, result.unavailable, result.no_data) == expected
    assert result.available == (expected[0] is not None)
    got = {c.key: set(c.reasons) for c in result.considered if not c.fits}
    assert got == ruled_out
    for c in result.considered:
        for r in c.ruled_out:
            assert r.detail


def test_named_key_considers_only_that_product():
    result = _resolve(named="NO2_STD", dates=YESTERDAY)
    assert [c.key for c in result.considered] == ["NO2_STD"]


def test_pinned_considers_only_the_pin():
    result = _resolve(pinned="NO2_STD", dates=YESTERDAY)
    assert [c.key for c in result.considered] == ["NO2_STD"]


def test_skipped_lists_earlier_products_ruled_out():
    result = _resolve(approved=["NO2_STD", "NO2_NRT"], dates=YESTERDAY)
    assert [c.key for c in result.skipped] == ["NO2_STD"]
    assert result.skipped[0].ruled_out[0].reason == "not_published_yet"


def test_outside_retention_detail_names_the_window():
    result = _resolve(registry=ROLLING, dates=AGED_OUT, named="NO2_ROLL")
    (roll,) = result.considered
    assert [r.detail for r in roll.ruled_out] == [
        f"keeps only the last 14 days (from {OLDEST_HELD}); dates start {AGED_OUT}"
    ]


def test_considered_follows_rank_order():
    result = _resolve()
    assert [c.key for c in result.considered] == [
        "NO2_STD", "NO2_NRT", "NO2_GLOBAL_DAILY", "NO2_MONTHLY", "PARTIAL"
    ]


def test_vouched_entry_lists_unchecked_facts():
    result = _resolve(approved=["BARE"])
    (bare,) = result.considered
    assert bare.vouched
    assert set(bare.unchecked) == {"quantity", "coverage", "record_start", "latency_days"}


def test_unvouched_entry_lists_unchecked_facts_too():
    result = _resolve(named="GEMS")
    (partial,) = result.considered
    assert not partial.vouched
    assert partial.reasons == {"facts_missing"}
    assert "coverage" in partial.unchecked and "quantity" not in partial.unchecked


def test_entries_without_quantity_are_not_considered_by_default():
    result = _resolve()
    assert "BARE" not in {c.key for c in result.considered}


def test_rank_tie_is_broken_by_key_and_disclosed():
    registry = {
        "TIE_B": _cfg(quantity=NO2, coverage="global", rank=1),
        "TIE_A": _cfg(quantity=NO2, coverage="global", rank=1),
        "LOWER": _cfg(quantity=NO2, coverage="global", rank=2),
    }
    result = _resolve(registry=registry)
    assert (result.product, result.rule, result.tied) == ("TIE_A", "highest_rank_tie_by_key", ("TIE_A", "TIE_B"))


def test_unranked_entries_tie_and_come_after_ranked():
    registry = {
        "U2": _cfg(quantity=NO2, coverage="global"),
        "U1": _cfg(quantity=NO2, coverage="global"),
        "R": _cfg(quantity=NO2, coverage="global", rank=9),
    }
    assert _resolve(registry=registry).product == "R"
    del registry["R"]
    result = _resolve(registry=registry)
    assert (result.product, result.rule, result.tied) == ("U1", "highest_rank_tie_by_key", ("U1", "U2"))


PACIFIC = {
    "PAC": _cfg(quantity=NO2, coverage={"west": 100, "south": -60, "east": 180, "north": 60}, rank=1),
    "GLOBE": _cfg(quantity=NO2, coverage="global", rank=2),
}


@pytest.mark.parametrize(
    ("area", "product", "pac_reasons"),
    [
        (BBox(175, -5, 179, 5), "PAC", set()),
        (BBox(170, -10, -170, 10), "GLOBE", {"area_partly_outside_coverage"}),
        (BBox(-179, -5, -170, 5), "GLOBE", {"area_outside_coverage"}),
        (BBox(150, 50, 160, 70), "GLOBE", {"area_partly_outside_coverage"}),
    ],
    ids=["inside", "crosses_antimeridian", "far_side", "partly_north"],
)
def test_area_containment(area, product, pac_reasons):
    result = _resolve(registry=PACIFIC, area=area)
    assert result.product == product
    (pac,) = [c for c in result.considered if c.key == "PAC"]
    assert set(pac.reasons) == pac_reasons


def test_point_area_is_allowed():
    assert _resolve(area=BBox(-74, 40.7, -74, 40.7)).product == "NO2_STD"


@pytest.mark.parametrize(
    "bad",
    [lambda: BBox(-181, 0, 0, 1), lambda: BBox(0, 10, 1, 5), lambda: BBox(0, -91, 1, 0)],
)
def test_bbox_rejects_bad_boxes(bad):
    with pytest.raises(ValueError):
        bad()


def test_date_range_rejects_end_before_start():
    with pytest.raises(ValueError):
        DateRange(date(2025, 2, 1), date(2025, 1, 1))


def test_empty_date_list_is_rejected():
    with pytest.raises(ValueError):
        _resolve(dates=[])


def test_approved_must_be_a_list_or_registry_default():
    with pytest.raises(ValueError):
        _resolve(approved="NO2_STD")


# Registry fields.

def test_new_fields_default_to_none():
    cfg = _cfg()
    assert (
        cfg.quantity, cfg.coverage, cfg.record_start, cfg.record_end, cfg.stream, cfg.latency_days,
        cfg.retention_days, cfg.rank,
    ) == (None, None, None, None, None, None, None, None)


def test_fields_parse_from_yaml_shapes():
    cfg = _cfg(
        quantity=NO2, coverage=NORTH_AMERICA, record_start="2023-08-01", record_end=date(2025, 1, 1),
        stream="near_real_time", latency_days=0, retention_days=14, rank=1,
    )
    assert cfg.coverage == CoverageBox(**NORTH_AMERICA)
    assert cfg.retention_days == 14
    assert cfg.record_start == date(2023, 8, 1)
    assert _cfg(coverage="global").coverage == "global"


@pytest.mark.parametrize(
    "over",
    [
        {"coverage": {"west": -40, "south": 10, "east": -140, "north": 70}},
        {"coverage": {"west": -140, "south": 70, "east": -40, "north": 10}},
        {"coverage": {"west": -200, "south": 10, "east": -40, "north": 70}},
        {"coverage": {"west": -140, "south": 10, "east": -40}},
        {"coverage": {"west": -140, "south": 10, "east": -40, "north": 70, "up": 1}},
        {"coverage": "regional"},
        {"record_start": date(2025, 1, 2), "record_end": date(2025, 1, 1)},
        {"record_start": "not a date"},
        {"latency_days": -1},
        {"retention_days": 0},
        {"retention_days": -14},
        {"retention_days": 1.5},
        {"rank": 0},
        {"stream": "fast"},
        {"quantity": "NO2 column"},
        {"quantity": "no2__column"},
    ],
    ids=[
        "west_after_east", "south_after_north", "lon_out_of_range", "box_missing_edge",
        "box_extra_key", "unknown_coverage_word", "record_end_before_start", "bad_date",
        "negative_latency", "retention_zero", "retention_negative", "retention_fractional", "rank_zero", "unknown_stream", "quantity_not_snake", "quantity_double_underscore",
    ],
)
def test_registry_fields_are_validated(over):
    with pytest.raises(ValidationError):
        _cfg(**over)


def test_real_registry_still_loads():
    registry = load_registry()
    assert "TEMPO_NO2" in registry
    assert all(isinstance(cfg, CollectionConfig) for cfg in registry.values())


def test_a_cadence_word_keeps_the_named_instruments_products_of_that_cadence():
    hourly = _resolve(named="TEMPO hourly")
    assert hourly.product == "NO2_STD"
    assert {c.key for c in hourly.considered} == {"NO2_STD", "NO2_NRT", "HCHO_STD", "BARE"}
    assert _resolve(named="monthly TROPOMI").product == "NO2_MONTHLY"
    assert _resolve(named="OMI monthly").unavailable == "named_not_registered"
    assert _resolve(named="annual").unavailable == "named_not_registered"


def test_an_unknown_cadence_is_rejected_by_the_registry():
    with pytest.raises(ValidationError):
        _cfg(cadence="weekly")


PERIODS = {
    "YEARS": _cfg(
        short_name="Y_L3", instrument="SAT", cadence="annual", quantity=NO2, coverage="global",
        record_start=date(2019, 1, 1), latency_days=85, rank=1,
    ),
    "SEASONS": _cfg(
        short_name="S_L3", instrument="SAT", cadence="seasonal", quantity=NO2, coverage="global",
        record_start=date(2018, 6, 1), latency_days=58, rank=2,
    ),
}


@pytest.mark.parametrize(
    ("named", "day", "today", "published"),
    [
        ("YEARS", date(2025, 1, 1), date(2026, 3, 26), True),
        ("YEARS", date(2025, 12, 31), date(2026, 3, 25), False),
        ("SEASONS", date(2026, 2, 28), date(2026, 4, 27), True),
        ("SEASONS", date(2025, 12, 1), date(2026, 4, 26), False),
    ],
)
def test_a_period_product_publishes_a_day_once_its_whole_period_has(named, day, today, published):
    result = _resolve(PERIODS, named=named, dates=day, today=today)
    assert result.available is published
    if not published:
        assert result.considered[0].reasons == {"not_published_yet"}
