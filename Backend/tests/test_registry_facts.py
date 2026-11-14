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
