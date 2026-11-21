"""The real registry records the product-choice facts, and the resolver chooses sensibly from them."""
from collections import defaultdict
from datetime import date, timedelta

import pytest

from tta_backend.datasets.registry import CoverageBox, load_registry
from tta_backend.harness.capabilities.products import (
    REGISTRY_DEFAULT,
    BBox,
    DateRange,
    resolve_product,
)

FACTS = ("quantity", "coverage", "record_start", "stream", "latency_days", "rank")

# Entries allowed to lack a fact, with the fact and why.
MISSING_FACTS = {
    # Retired product: nothing is published any more, so a latency means nothing.
    "TEMPO_HCHO_V03": {"latency_days"},
}

TODAY = date(2026, 11, 14)
YESTERDAY = TODAY - timedelta(days=1)
LAST_YEAR = date(2025, 6, 1)

NEW_JERSEY = BBox(-75.6, 38.9, -73.9, 41.4)
EUROPE = BBox(0, 45, 10, 55)

NO2 = "no2_tropospheric_column"
HCHO = "hcho_total_column"
AOD = "aod_550nm"


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _resolve(registry, **over):
    args = {
        "quantity": NO2,
        "area": NEW_JERSEY,
        "dates": LAST_YEAR,
        "registry": registry,
        "today": TODAY,
        "approved": REGISTRY_DEFAULT,
    }
    return resolve_product(**{**args, **over})


def test_every_entry_records_its_facts(registry):
    problems = []
    for key, cfg in registry.items():
        missing = {f for f in FACTS if getattr(cfg, f) is None}
        if missing != MISSING_FACTS.get(key, set()):
            problems.append(f"{key}: missing {sorted(missing)}, allowed {sorted(MISSING_FACTS.get(key, set()))}")
    assert not problems, "\n".join(problems)


def test_exceptions_name_real_entries_and_facts(registry):
    assert set(MISSING_FACTS) <= set(registry)
    assert all(fields <= set(FACTS) for fields in MISSING_FACTS.values())


def test_ranks_are_unique_within_a_quantity(registry):
    by_quantity = defaultdict(list)
    for key, cfg in registry.items():
        if cfg.quantity is not None:
            by_quantity[cfg.quantity].append((cfg.rank, key))
    for quantity, entries in by_quantity.items():
        ranks = [r for r, _ in entries]
        assert None not in ranks, f"{quantity}: unranked entry in {entries}"
        assert len(set(ranks)) == len(ranks), f"{quantity}: duplicate rank in {sorted(entries)}"


def test_rank_without_quantity_is_a_mistake(registry):
    assert [k for k, c in registry.items() if c.rank is not None and c.quantity is None] == []


def test_quantity_vocabulary_is_the_expected_set(registry):
    assert {c.quantity for c in registry.values()} == {
        NO2, HCHO, AOD, "o3_total_column", "o3_profile", "so2_total_column",
    }


def test_tempo_entries_cover_north_america_only(registry):
    for key, cfg in registry.items():
        if cfg.instrument == "TEMPO":
            assert isinstance(cfg.coverage, CoverageBox), key
            assert cfg.coverage.west < -100 < cfg.coverage.east < 0, key


def test_near_real_time_entries_are_marked(registry):
    nrt = {k for k, c in registry.items() if c.stream == "near_real_time"}
    assert nrt == {"TEMPO_NO2_NRT", "TEMPO_HCHO_NRT", "AERDA_AOD_NRT", "AERDA_AOD_NRT_3H"}


def test_no2_over_new_jersey_last_year_is_standard_tempo(registry):
    result = _resolve(registry)
    assert (result.product, result.rule) == ("TEMPO_NO2", "highest_rank")


def test_no2_over_new_jersey_yesterday_is_published_by_the_standard_stream(registry):
    result = _resolve(registry, dates=YESTERDAY)
    assert result.product == "TEMPO_NO2"


def test_near_real_time_tempo_is_chosen_when_the_workflow_prefers_it(registry):
    result = _resolve(registry, dates=YESTERDAY, approved=["TEMPO_NO2_NRT", "TEMPO_NO2"])
    assert (result.product, result.rule) == ("TEMPO_NO2_NRT", "approved_preference")


def test_near_real_time_tempo_is_skipped_before_its_record_starts(registry):
    result = _resolve(registry, approved=["TEMPO_NO2_NRT", "TEMPO_NO2"])
    assert result.product == "TEMPO_NO2"
    assert [c.key for c in result.skipped] == ["TEMPO_NO2_NRT"]
    assert result.skipped[0].reasons == {"outside_record"}


def test_near_real_time_tempo_is_skipped_once_a_date_ages_out_of_its_window(registry):
    aged_out = TODAY - timedelta(days=registry["TEMPO_NO2_NRT"].retention_days + 1)
    result = _resolve(registry, dates=aged_out, approved=["TEMPO_NO2_NRT", "TEMPO_NO2"])
    assert result.product == "TEMPO_NO2"
    assert [c.key for c in result.skipped] == ["TEMPO_NO2_NRT"]
    assert result.skipped[0].reasons == {"outside_retention"}


def test_no2_outside_north_america_is_a_global_product(registry):
    result = _resolve(registry, area=EUROPE, cadence_needed="daily")
    assert result.product == "OMI_NO2"
    assert registry[result.product].coverage == "global"
    ruled_out = {c.key: c.reasons for c in result.considered if not c.fits}
    assert ruled_out["TEMPO_NO2"] == {"area_outside_coverage"}
    assert ruled_out["TROPOMI_NO2"] == {"cadence_too_coarse"}


def test_no2_straddling_the_tempo_edge_falls_to_a_global_product(registry):
    result = _resolve(registry, area=BBox(-20, 40, 20, 50))
    assert result.product == "OMI_NO2"
    tempo = next(c for c in result.considered if c.key == "TEMPO_NO2")
    assert tempo.reasons == {"area_partly_outside_coverage"}


def test_date_before_every_record_is_outside_record(registry):
    result = _resolve(registry, dates=date(1999, 1, 1))
    assert result.product is None
    assert (result.unavailable, result.no_data) == ("nothing_fits", "outside_record")
    assert all("outside_record" in c.reasons for c in result.considered)


def test_no2_after_the_omi_record_ends_is_outside_record_outside_north_america(registry):
    result = _resolve(registry, area=EUROPE, dates=date(2026, 9, 1), cadence_needed="daily")
    assert result.product is None
    assert result.no_data == "outside_record"


def test_aod_recent_dates_go_to_the_near_real_time_product(registry):
    result = _resolve(registry, quantity=AOD, area=EUROPE, dates=YESTERDAY)
    assert result.product == "AERDA_AOD_NRT"
    assert registry[result.product].stream == "near_real_time"
    skipped = {c.key: c.reasons for c in result.considered if not c.fits}
    assert skipped["MODIS_AOD_TERRA"] == {"not_published_yet"}


def test_aod_last_year_is_a_standard_product(registry):
    result = _resolve(registry, quantity=AOD, area=EUROPE)
    assert (result.product, result.rule) == ("MODIS_AOD_TERRA", "highest_rank")


def test_aod_before_the_near_real_time_record_has_no_near_real_time_candidate(registry):
    result = _resolve(registry, quantity=AOD, area=EUROPE, dates=date(2024, 1, 1))
    ruled_out = {c.key: c.reasons for c in result.considered if not c.fits}
    assert ruled_out["AERDA_AOD_NRT"] == {"outside_record"}
    assert result.product == "MODIS_AOD_TERRA"


def test_hcho_over_new_jersey_is_standard_tempo(registry):
    assert _resolve(registry, quantity=HCHO, dates=date(2024, 6, 1)).product == "TEMPO_HCHO"


def test_hcho_in_europe_is_omi_and_ends_with_its_record(registry):
    assert _resolve(registry, quantity=HCHO, area=EUROPE).product == "OMI_HCHO"
    late = _resolve(registry, quantity=HCHO, area=EUROPE, dates=date(2026, 9, 1))
    assert (late.product, late.no_data) == (None, "outside_record")


def test_retired_tempo_hcho_version_is_not_chosen_by_default(registry):
    result = _resolve(registry, quantity=HCHO, dates=date(2024, 6, 1))
    assert result.product != "TEMPO_HCHO_V03"
    named = _resolve(registry, quantity=HCHO, dates=date(2024, 6, 1), named="TEMPO_HCHO_V03")
    assert named.product == "TEMPO_HCHO_V03"
    after = _resolve(registry, quantity=HCHO, dates=DateRange(date(2025, 9, 1), date(2025, 10, 1)), named="TEMPO_HCHO_V03")
    assert after.unavailable == "named_product_unsuitable"


def test_so2_is_served_by_omi(registry):
    result = _resolve(registry, quantity="so2_total_column")
    assert (result.product, result.rule) == ("OMI_SO2", "only_candidate")


def test_ozone_profile_is_tempo_only(registry):
    assert _resolve(registry, quantity="o3_profile").product == "TEMPO_O3PROF"
    assert _resolve(registry, quantity="o3_profile", area=EUROPE).unavailable == "nothing_fits"


def test_total_ozone_prefers_tempo_over_omi_in_north_america(registry):
    assert _resolve(registry, quantity="o3_total_column").product == "TEMPO_O3TOT"
    assert _resolve(registry, quantity="o3_total_column", area=EUROPE).product == "OMI_O3"


# The TROPOMI NO2 annual and seasonal products. Facts checked 2026-11-20.
PERIOD_TODAY = date(2026, 11, 20)
TROPOMI_PERIODS = {
    "TROPOMI_NO2_ANNUAL": ("annual", "HAQ_TROPOMI_NO2_GLOBAL_A_L3", "C3087325001-GES_DISC"),
    "TROPOMI_NO2_CONUS_ANNUAL": ("annual", "HAQ_TROPOMI_NO2_CONUS_A_L3", "C2839237129-GES_DISC"),
    "TROPOMI_NO2_CONUS_SEASONAL": ("seasonal", "HAQ_TROPOMI_NO2_CONUS_S_L3", "C2839237223-GES_DISC"),
}


def _period(registry, **over):
    return _resolve(registry, **{"today": PERIOD_TODAY, "dates": LAST_YEAR, **over})


def test_tropomi_period_products_are_registered_like_the_monthly_one(registry):
    monthly = registry["TROPOMI_NO2"]
    for key, (cadence, short_name, collection_id) in TROPOMI_PERIODS.items():
        cfg = registry[key]
        assert (cfg.cadence, cfg.short_name, cfg.collection_id, cfg.version) == (
            cadence, short_name, collection_id, "2.4",
        ), key
        assert cfg.quantity == NO2, key
        assert cfg.rank > monthly.rank, key
        for field in ("primary_var", "variables", "units", "fill_value", "valid_min", "valid_max",
                      "quality_flag_var", "instrument", "provider"):
            assert getattr(cfg, field) == getattr(monthly, field), (key, field)


def test_conus_period_products_cover_conus_only(registry):
    for key in ("TROPOMI_NO2_CONUS_ANNUAL", "TROPOMI_NO2_CONUS_SEASONAL"):
        assert registry[key].coverage == CoverageBox(west=-124.75, south=24.5, east=-66.76, north=49.49)
    assert registry["TROPOMI_NO2_CONUS_SEASONAL"].record_start == date(2018, 6, 1)


def test_the_conus_annual_product_ranks_ahead_of_the_global_one(registry):
    assert registry["TROPOMI_NO2_CONUS_ANNUAL"].rank < registry["TROPOMI_NO2_ANNUAL"].rank


def test_with_no_product_named_no2_in_europe_after_omi_ends_stays_monthly(registry):
    result = _period(registry, area=EUROPE, dates=date(2026, 4, 15))
    assert result.product == "TROPOMI_NO2"
    order = [c.key for c in result.considered]
    assert order.index("TROPOMI_NO2") < min(order.index(k) for k in TROPOMI_PERIODS)


def test_with_no_product_named_the_period_products_rank_last(registry):
    result = _period(registry, area=EUROPE)
    assert result.product == "OMI_NO2"
    order = [c.key for c in result.considered]
    assert order[-3:] == ["TROPOMI_NO2_CONUS_ANNUAL", "TROPOMI_NO2_ANNUAL", "TROPOMI_NO2_CONUS_SEASONAL"]


def test_the_instrument_alone_still_means_the_monthly_product(registry):
    result = _period(registry, named="TROPOMI", dates=date(2024, 1, 15))
    assert (result.product, result.rule) == ("TROPOMI_NO2", "only_candidate")
    assert [c.key for c in result.considered] == ["TROPOMI_NO2"]


def test_the_instrument_alone_does_not_fall_back_to_a_seasonal_mean(registry):
    result = _period(registry, named="TROPOMI", dates=date(2018, 7, 1))
    assert (result.product, result.unavailable) == (None, "named_product_unsuitable")


@pytest.mark.parametrize("named", ["TROPOMI annual", "yearly TROPOMI", "tropomi-annual"])
def test_tropomi_annual_over_new_jersey_is_the_conus_product(registry, named):
    result = _period(registry, named=named)
    assert (result.product, result.rule) == ("TROPOMI_NO2_CONUS_ANNUAL", "highest_rank")
    assert {c.key for c in result.considered} == {"TROPOMI_NO2_ANNUAL", "TROPOMI_NO2_CONUS_ANNUAL"}


def test_tropomi_annual_over_europe_is_the_global_product(registry):
    result = _period(registry, named="TROPOMI annual", area=EUROPE)
    assert (result.product, result.rule) == ("TROPOMI_NO2_ANNUAL", "only_candidate")
    assert [(c.key, c.reasons) for c in result.skipped] == [
        ("TROPOMI_NO2_CONUS_ANNUAL", frozenset({"area_outside_coverage"})),
    ]


def test_tropomi_seasonal_over_europe_is_refused_as_conus_only(registry):
    result = _period(registry, named="TROPOMI seasonal", area=EUROPE)
    assert (result.product, result.unavailable) == (None, "named_product_unsuitable")
    assert [(c.key, c.reasons) for c in result.considered] == [
        ("TROPOMI_NO2_CONUS_SEASONAL", frozenset({"area_outside_coverage"})),
    ]


def test_tropomi_seasonal_reaches_back_to_summer_2018(registry):
    result = _period(registry, named="TROPOMI seasonal", dates=date(2018, 7, 1))
    assert result.product == "TROPOMI_NO2_CONUS_SEASONAL"


@pytest.mark.parametrize("needed", ["daily", "monthly"])
@pytest.mark.parametrize("key", ["TROPOMI_NO2_ANNUAL", "TROPOMI_NO2_CONUS_ANNUAL"])
def test_daily_or_monthly_values_from_an_annual_product_are_too_coarse(registry, key, needed):
    result = _period(registry, named=key, cadence_needed=needed)
    assert result.unavailable == "named_product_unsuitable"
    assert result.considered[0].reasons == {"cadence_too_coarse"}
    assert result.considered[0].ruled_out[0].detail == f"an annual product, so {needed} values are not available"


def test_an_annual_product_serves_an_annual_or_seasonal_need(registry):
    assert _period(registry, named="TROPOMI annual", cadence_needed="annual").product == "TROPOMI_NO2_CONUS_ANNUAL"
    seasonal = _period(registry, named="TROPOMI_NO2_CONUS_SEASONAL", cadence_needed="annual")
    assert seasonal.product == "TROPOMI_NO2_CONUS_SEASONAL"
    too_coarse = _period(registry, named="TROPOMI annual", cadence_needed="seasonal")
    assert all(c.reasons == {"cadence_too_coarse"} for c in too_coarse.considered)


def test_a_year_is_published_only_after_it_ends(registry):
    published = _period(registry, named="TROPOMI_NO2_ANNUAL", dates=date(2025, 1, 10))
    assert published.product == "TROPOMI_NO2_ANNUAL"
    this_year = _period(registry, named="TROPOMI_NO2_ANNUAL", dates=date(2026, 2, 1))
    assert (this_year.unavailable, this_year.no_data) == ("named_product_unsuitable", "not_published_yet")
    assert this_year.considered[0].ruled_out[0].detail == (
        "its annual mean holding 2026-02-01 runs to 2026-12-31 and is published 85 days after that, from 2027-03-26"
    )
    # 2025's file appeared 2026-03-25, about 85 days after the year ended.
    before = _period(registry, named="TROPOMI_NO2_ANNUAL", dates=date(2025, 6, 1), today=date(2026, 3, 20))
    assert before.no_data == "not_published_yet"


def test_a_season_is_published_only_after_it_ends(registry):
    jja = _period(registry, named="TROPOMI seasonal", dates=date(2026, 6, 15), today=date(2026, 10, 28))
    assert jja.product == "TROPOMI_NO2_CONUS_SEASONAL"
    early = _period(registry, named="TROPOMI seasonal", dates=date(2026, 6, 15), today=date(2026, 10, 27))
    assert early.no_data == "not_published_yet"
    son = _period(registry, named="TROPOMI seasonal", dates=date(2026, 9, 15))
    assert son.no_data == "not_published_yet"


def test_a_january_day_belongs_to_the_previous_decembers_season(registry):
    # DJF 2025 ends 2026-02-28, so it is published by late April 2026.
    jan = _period(registry, named="TROPOMI seasonal", dates=date(2026, 1, 15), today=date(2026, 4, 27))
    assert jan.product == "TROPOMI_NO2_CONUS_SEASONAL"
    dec = _period(registry, named="TROPOMI seasonal", dates=date(2025, 12, 15), today=date(2026, 4, 26))
    assert dec.no_data == "not_published_yet"


def test_two_cadence_words_name_nothing(registry):
    assert _period(registry, named="TROPOMI annual seasonal").unavailable == "named_not_registered"
