"""Seasonal and annual cadences: calendar periods, nearby dates, equal weighting and labels."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tta_backend.datasets.cadence import CADENCE_ORDER, CADENCES, period_end, period_start, shift_period
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.earthdata.retrieval import nearby_dates
from tta_backend.harness.capabilities.operations.reduce import _time_basis

TODAY = date(2026, 11, 20)


def test_cadences_run_finest_first_with_seasons_and_years_coarser_than_months():
    assert CADENCES == ("hourly", "daily", "monthly", "seasonal", "annual")
    assert CADENCE_ORDER["monthly"] < CADENCE_ORDER["seasonal"] < CADENCE_ORDER["annual"]


@pytest.mark.parametrize(
    ("day", "start", "end"),
    [
        (date(2018, 12, 1), date(2018, 12, 1), date(2019, 2, 28)),
        (date(2019, 1, 15), date(2018, 12, 1), date(2019, 2, 28)),
        (date(2020, 2, 29), date(2019, 12, 1), date(2020, 2, 29)),
        (date(2019, 3, 1), date(2019, 3, 1), date(2019, 5, 31)),
        (date(2019, 8, 31), date(2019, 6, 1), date(2019, 8, 31)),
        (date(2019, 11, 30), date(2019, 9, 1), date(2019, 11, 30)),
    ],
)
def test_a_day_snaps_to_its_meteorological_season(day, start, end):
    assert (period_start(day, "seasonal"), period_end(day, "seasonal")) == (start, end)


def test_a_day_snaps_to_its_calendar_year_and_month():
    assert (period_start(date(2025, 7, 4), "annual"), period_end(date(2025, 7, 4), "annual")) == (
        date(2025, 1, 1), date(2025, 12, 31),
    )
    assert period_end(date(2024, 2, 10), "monthly") == date(2024, 2, 29)


def test_seasons_shift_across_the_year_end():
    assert shift_period(date(2019, 1, 15), "seasonal", -1) == date(2018, 9, 1)
    assert shift_period(date(2018, 10, 1), "seasonal", 1) == date(2018, 12, 1)
    assert shift_period(date(2018, 12, 1), "seasonal", 1) == date(2019, 3, 1)


@pytest.mark.parametrize(
    ("cadence", "day", "end", "expected"),
    [
        ("monthly", date(2024, 3, 10), None, [date(2024, 2, 1), date(2024, 4, 1)]),
        ("seasonal", date(2019, 1, 15), None, [date(2018, 9, 1), date(2019, 3, 1)]),
        ("seasonal", date(2019, 6, 1), date(2019, 8, 31), [date(2019, 3, 1), date(2019, 9, 1)]),
        ("annual", date(2022, 5, 1), None, [date(2021, 1, 1), date(2023, 1, 1)]),
        ("annual", date(2021, 3, 1), date(2022, 6, 1), [date(2020, 1, 1), date(2023, 1, 1)]),
    ],
)
def test_nearby_dates_look_one_period_either_side(cadence, day, end, expected):
    assert nearby_dates(day, cadence, today=TODAY, end=end) == expected


def test_nearby_dates_never_look_past_today():
    assert nearby_dates(date(2026, 3, 1), "annual", today=TODAY) == [date(2025, 1, 1)]
    assert nearby_dates(date(2026, 8, 1), "seasonal", today=TODAY) == [date(2026, 3, 1), date(2026, 9, 1)]


def _series(times, values):
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(len(values), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(times), "lat": [40.0], "lon": [-74.0]},
    )


def test_each_season_counts_once_in_the_temporal_mean():
    # December and February are one season (DJF 2018); March starts the next.
    da = _series(["2018-12-01", "2019-02-01", "2019-03-01"], [10.0, 20.0, 30.0])
    mean = AggregationService().temporal_mean(da, "time", "seasonal")
    assert float(mean.values.reshape(-1)[0]) == pytest.approx(22.5)


def test_each_year_counts_once_in_the_temporal_mean():
    da = _series(["2019-01-01", "2019-07-01", "2020-01-01"], [10.0, 30.0, 40.0])
    mean = AggregationService().temporal_mean(da, "time", "annual")
    assert float(mean.values.reshape(-1)[0]) == pytest.approx(30.0)


def test_granule_labels_name_seasonal_and_annual_cadences():
    service = AggregationService()
    da = _series(["2019-01-01", "2020-01-01"], [1.0, 2.0])
    meta = service._build_meta(da, 2, "annual", "mean", [0, 1], "time")
    assert "2 annual granules" in meta["aggregation_label"]
    meta = service._build_meta(da, 1, "seasonal", "mean", [0], "time")
    assert "1 seasonal granule" in meta["aggregation_label"]


def test_the_time_basis_names_seasonal_and_annual_periods():
    assert _time_basis("mean", "seasonal") == "each seasonal period weighted equally"
    assert _time_basis("mean", "annual") == "each annual period weighted equally"
    assert _time_basis("mean", "unknown") == "each time step weighted equally"
