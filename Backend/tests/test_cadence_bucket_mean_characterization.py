"""T59 D4 Phase 1b: the cadence-bucket temporal mean, characterized.

What this locks down
--------------------
``AggregationService._cadence_weighted_mean`` gives each timestep the weight
``1 / (granules in its cadence bucket)`` and then hands the whole thing to
``.weighted(w).mean(skipna=True)``. Finding #11's stated intent is that **each
cadence bucket counts once**, however many granules happened to land in it.

Those two are the same number only while every bucket is either wholly finite
or wholly NaN at a pixel. ``.weighted().mean(skipna=True)`` renormalizes PER
PIXEL over the timesteps that survived masking, so at a pixel where a 4-granule
bucket kept one granule, that bucket contributes ``1/4`` of a vote instead of
one. The mean of the per-bucket means gives it a whole vote -- which is what
Finding #11 says the answer is supposed to be.

The real-data measurement this encodes (two materialized TEMPO NO2 bundles,
``scripts/characterize_d4_bucket_mean.py``):

* pixels with no partial cadence bucket: the two formulations agreed to
  floating-point noise (``max |F-M|`` of 2 and 16 against values near 1e15);
* pixels with at least one partial bucket: 2.736% and 2.339% characteristic
  delta, with 15.4% and 5.4% of pixels past 1%;
* turning QA masking OFF made the gap LARGER. The NaNs come from swath tiling
  -- the granules inside one hour cover different slices of the domain -- not
  from quality masking.

Why this fixture is synthetic
-----------------------------
The two bundles are multi-hundred-megabyte Harmony ``result.nc.zip`` archives
that live in the backend container's ``/data/harmony`` volume. They are not on
the host, are not committable, and the repo's checked-in test assets are all
small JSON. So the fixture below is deterministic and hand-checkable, and it
reproduces the failure mechanism exactly rather than approximately: **several
granules inside one cadence bucket, each covering a different longitude slice,
so a pixel is finite in some of that bucket's granules and NaN in the rest.**
That is what the real bundles do; the swath tiling is the whole story.

Scope: the reduction math only. Nothing here touches frame storage, the
scrubber, or any payload shape.

Status
------
The fix landed: ``_cadence_weighted_mean`` now weights each timestep by
``1 / (granules in its bucket that have a value AT THIS PIXEL)``. No existing
test in ``test_aggregation_service.py`` or ``test_vertical_profile.py``
changed colour, because Finding #11's own guards all describe FULLY covered
buckets, where the old and new denominators are the same number.

The pre-D4 arithmetic is preserved verbatim as
``legacy_granule_count_weighted_mean``, and the tests below compare production
against BOTH it and an independent bucket-mean reference. That is deliberate:
comparing the fix only against the reference would assert that two correct
things agree and would keep passing if the fix were reverted. Comparing
against the legacy form keeps the defect itself measured -- the 3.56% is still
a live number, not a historical note.
"""
from __future__ import annotations

import importlib.util
import unittest

# The measured characteristic delta on the partial-coverage population of THIS
# fixture, pinned so the test fails if the mechanism changes rather than merely
# if some number moves. Same D16 metric the real-data script reports:
# sum(w|F-M|) / sum(w|M|) with w = cos(latitude), never a mean of per-pixel
# ratios (a geophysical field goes near zero) and never signed (a field off by
# +-10% would cancel to ~0).
#
# 3.56% sits in the same band as the 2.736% and 2.339% the two real TEMPO NO2
# bundles measured -- the fixture reproduces the mechanism at its real size,
# not an exaggerated caricature of it.
PARTIAL_POPULATION_DELTA_PCT = 3.5629534883
# Fraction of the partial-coverage population past 1% -- the real bundles
# measured 15.4% and 5.4% of ALL compared pixels.
PARTIAL_PIXELS_OVER_1PCT_FRACTION = 0.75


def _scalar(da):
    """The one value in a size-1 result, as a float. ``float(da)`` refuses a
    ``(lat=1, lon=1)`` array on numpy 2.x, and squeezing at every call site
    would bury the assertions."""
    return float(da.values.reshape(-1)[0])


def _bucket_labels(stamps, cadence):
    """The bucketing ``_cadence_weighted_mean`` itself does."""
    if cadence == "monthly":
        return stamps.to_period("M")
    return stamps.floor({"hourly": "h", "daily": "D"}[cadence])


def _grouper(da, time_dim, cadence):
    import numpy as np
    import pandas as pd
    import xarray as xr

    labels = _bucket_labels(pd.to_datetime(np.asarray(da[time_dim].values)), cadence)
    return xr.DataArray(
        np.asarray([str(b) for b in labels]),
        dims=[time_dim],
        coords={time_dim: da[time_dim]},
    ).rename("bucket")


def mean_of_bucket_means(da, time_dim, cadence):
    """The formulation Finding #11 describes: reduce within each cadence
    bucket, then average the buckets. A bucket contributes weight 1 at every
    pixel where it holds ANY finite value -- that is the whole difference from
    the weighted form. Mirrors ``scripts/characterize_d4_bucket_mean.py`` so
    the committed test and the exploratory probe cannot drift apart.
    """
    grouper = _grouper(da, time_dim, cadence)
    per_bucket = da.groupby(grouper).mean(dim=time_dim, skipna=True)
    return per_bucket.mean(dim="bucket", skipna=True)


def legacy_granule_count_weighted_mean(da, time_dim, cadence):
    """The pre-D4 reduction, preserved verbatim as a reference.

    Weight ``1 / (granules in the bucket)`` -- counting granules the bucket
    HOLDS rather than granules that reach this pixel -- then the same
    ``.weighted().mean(skipna=True)``. Keeping it here is what lets the tests
    below still measure the defect after the fix landed: without it they could
    only assert that two correct things agree, and would go on passing if the
    fix were reverted.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    stamps = pd.to_datetime(np.asarray(da[time_dim].values))
    buckets = _bucket_labels(stamps, cadence)
    counts = pd.Series(1, index=buckets).groupby(level=0).transform("size")
    weights = xr.DataArray(
        1.0 / np.asarray(counts.values, dtype=float),
        dims=[time_dim],
        coords={time_dim: da[time_dim]},
    )
    return da.weighted(weights).mean(dim=time_dim, skipna=True)


def partial_bucket_count(da, time_dim, cadence):
    """Per pixel, how many cadence buckets are *partially* covered.

    A partial bucket holds more than one granule and, at this pixel, has at
    least one finite value and at least one NaN. That is the exact and only
    condition under which the two formulations can disagree, so splitting on it
    turns "the two differ" into "the two differ *for this reason*".
    """
    import numpy as np

    grouper = _grouper(da, time_dim, cadence)
    finite = np.isfinite(da)
    n_finite = finite.groupby(grouper).sum(dim=time_dim)
    n_total = finite.groupby(grouper).count(dim=time_dim)
    partial = (n_total > 1) & (n_finite > 0) & (n_finite < n_total)
    return partial.sum(dim="bucket")


def delta_pct(f, m, where):
    """``sum(w|F-M|) / sum(w|M|)`` as a percentage over the selected cells."""
    import numpy as np

    from tta_backend.preprocessing.aggregation_service import cos_lat_weights

    weights = cos_lat_weights(m)
    mask = np.isfinite(f) & np.isfinite(m) & where
    diff = np.abs(f - m).where(mask)
    mag = np.abs(m).where(mask)
    if weights is None:
        num, den = float(diff.sum(skipna=True)), float(mag.sum(skipna=True))
    else:
        num = float((diff * weights).sum(skipna=True))
        den = float((mag * weights).sum(skipna=True))
    return 100.0 * num / den if den > 0 else 0.0


# TEMPO-shaped: hourly cadence, and the granules inside one hour are scan tiles
# that each cover a longitude slice. (start_lon_index, end_lon_index) inclusive.
_TILED_HOUR_FIXTURE = (
    # timestamp,                 lon tile covered,  per-granule scale
    ("2025-10-01T13:05:00", (0, 15), 1.00),
    ("2025-10-01T13:25:00", (4, 19), 1.12),
    ("2025-10-01T13:45:00", (8, 23), 0.94),
    ("2025-10-01T14:15:00", (0, 23), 1.30),
    ("2025-10-01T15:10:00", (0, 17), 0.88),
    ("2025-10-01T15:40:00", (6, 23), 1.05),
)
_N_LAT, _N_LON = 12, 24


def _tiled_swath_field():
    """A (time, lat, lon) NO2-shaped field whose NaNs come from swath tiling.

    Buckets: 13:00 holds 3 granules, 14:00 holds 1, 15:00 holds 2. The tiles
    overlap in the middle of the domain and thin toward both edges, so the grid
    carries both populations at once -- pixels every granule saw, and pixels
    only some granules in a bucket saw.

    Each granule is ``base(lat, lon) * scale``, so the relative disagreement
    between the two formulations is a property of the COVERAGE PATTERN alone
    and is not smeared by the field's own spatial structure. ``base`` still
    varies (a plume over a city) so the area-weighted headline is not
    degenerate.
    """
    import numpy as np
    import xarray as xr

    lats = np.linspace(30.0, 52.5, _N_LAT)
    lons = np.linspace(-105.0, -70.0, _N_LON)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    base = 1.5 + 8.0 * np.exp(-(((lat2d - 40.0) / 6.0) ** 2 + ((lon2d + 87.0) / 8.0) ** 2))

    frames, times = [], []
    for stamp, (lo, hi), scale in _TILED_HOUR_FIXTURE:
        frame = np.full((_N_LAT, _N_LON), np.nan)
        frame[:, lo : hi + 1] = base[:, lo : hi + 1] * scale
        frames.append(frame)
        times.append(stamp)

    return xr.DataArray(
        np.stack(frames),
        dims=("time", "lat", "lon"),
        coords={"time": np.array(times, dtype="datetime64[ns]"), "lat": lats, "lon": lons},
        name="no2",
    )


THINNED_LON, WHOLE_LON = -75.0, -74.0


def _thinned_bucket_dataset(xr, np):
    """The mechanism at its smallest, carrying BOTH populations on one grid.

    Day one holds 4 granules, day two holds 1. At ``THINNED_LON`` only the
    first of day one's granules covers the pixel; at ``WHOLE_LON`` all four do.

    * thinned pixel -- intended ``(10 + 0) / 2 = 5``; today the surviving
      granule still carries weight ``1/4``, giving
      ``(0.25*10 + 1*0) / (0.25 + 1) = 2``.
    * whole pixel -- ``(20 + 0) / 2 = 10`` under both formulations.

    The second pixel is load-bearing, not decoration. On a 1x1 grid the three
    thinned granules are wholly-NaN TIMESTEPS, so ``_valid_time_indices`` drops
    them before the reduction ever runs and ``aggregate`` returns the intended
    5.0 by accident -- the bug becomes invisible. Real partial coverage is
    per-pixel within a timestep that is perfectly valid elsewhere, and it takes
    a second pixel to reproduce that.
    """
    nan = np.nan
    return xr.Dataset(
        {"no2": (("time", "lat", "lon"), np.array([
            [[10.0, 20.0]],  # 2024-01-01, granule 1 of 4 -- covers both pixels
            [[nan, 20.0]],   # granules 2-4 miss the thinned pixel only
            [[nan, 20.0]],
            [[nan, 20.0]],
            [[0.0, 0.0]],    # 2024-01-02: 1 granule
        ]))},
        coords={
            "time": np.array([
                "2024-01-01T06:00:00", "2024-01-01T10:00:00",
                "2024-01-01T14:00:00", "2024-01-01T18:00:00",
                "2024-01-02T12:00:00",
            ], dtype="datetime64[ns]"),
            "lat": [40.0], "lon": [THINNED_LON, WHOLE_LON],
        },
        attrs={"cadence": "daily"},
    )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CadenceBucketPopulationSplitTests(unittest.TestCase):
    """The real-data measurement, reproduced on a deterministic fixture."""

    def setUp(self):
        import numpy as np

        from tta_backend.preprocessing.aggregation_service import AggregationService

        self.np = np
        self.svc = AggregationService()
        self.field = _tiled_swath_field()
        self.cadence = "hourly"
        # Three fields over the same fixture: what production answers now, the
        # independent bucket-mean reference it should equal, and the pre-D4
        # arithmetic it should differ from exactly where coverage is partial.
        self.production = self.svc._cadence_weighted_mean(self.field, "time", self.cadence)
        self.bucketed = mean_of_bucket_means(self.field, "time", self.cadence)
        self.legacy = legacy_granule_count_weighted_mean(self.field, "time", self.cadence)
        self.partial = partial_bucket_count(self.field, "time", self.cadence)

    def test_the_fixture_carries_both_coverage_populations(self):
        """Guard on the fixture itself. If a future edit made every pixel
        complete (or every pixel partial) the two assertions below would still
        pass while measuring nothing -- which is precisely the trap the QA-on /
        QA-off comparison already sprang once on this investigation."""
        complete = int((self.partial == 0).sum())
        partial = int((self.partial > 0).sum())

        self.assertEqual(complete + partial, _N_LAT * _N_LON)
        self.assertEqual(complete, _N_LAT * 8)   # lon tiles 8..15, seen by every granule
        self.assertEqual(partial, _N_LAT * 16)
        # And the divergence really is confined to multi-granule buckets: the
        # 14:00 bucket holds one granule and can never be "partial".
        self.assertEqual(int(self.partial.max()), 2)

    def test_pixels_with_no_partial_bucket_were_left_exactly_where_they_were(self):
        """The population D4 must NOT move. Where every bucket is wholly
        covered, counting contributors and counting granules give the same
        denominator, so production must still answer bit-for-bit what the
        pre-D4 arithmetic answered -- not "close enough". A loose tolerance
        here would hide a real change to the complete-coverage answer, which is
        most of every real scene."""
        np = self.np

        sel = self.partial == 0
        new = self.production.where(sel)
        old = self.legacy.where(sel)

        self.assertGreater(int((np.isfinite(new) & np.isfinite(old)).sum()), 0)
        # Neither may produce a value where the other has none.
        self.assertEqual(int((np.isfinite(new) ^ np.isfinite(old)).where(sel).sum()), 0)
        # Identical weights fed to identical arithmetic: exactly equal, no ulp
        # budget at all.
        self.assertEqual(float(np.abs(new - old).max(skipna=True)), 0.0)

    def test_pixels_with_no_partial_bucket_also_match_the_bucket_reference(self):
        """The same population against the independent reference. Here the two
        expressions sum the same terms in a DIFFERENT order, so the last bit is
        allowed to differ and nothing else is -- the real bundles measured the
        same thing (max |F-M| of 2 and 16 against values near 1e15)."""
        np = self.np

        sel = self.partial == 0
        residual = float(np.abs(self.production.where(sel) - self.bucketed.where(sel)).max(skipna=True))
        allowed = 2.0 * float(np.spacing(np.abs(self.bucketed.where(sel)).max(skipna=True)))

        self.assertLessEqual(residual, allowed)

    def test_pixels_with_a_partial_bucket_now_answer_the_bucket_mean(self):
        """The defect, fixed. On the partial-coverage population production now
        tracks the bucket-mean reference to floating-point noise, where before
        D4 it diverged by the pinned 3.56%."""
        np = self.np

        sel = self.partial > 0
        residual = float(np.abs(self.production.where(sel) - self.bucketed.where(sel)).max(skipna=True))
        allowed = 8.0 * float(np.spacing(np.abs(self.bucketed.where(sel)).max(skipna=True)))

        self.assertLessEqual(residual, allowed)
        self.assertLess(delta_pct(self.production, self.bucketed, sel), 1e-10)

    def test_the_fix_moved_the_partial_population_by_the_measured_amount(self):
        """The characterization proper, still measuring the defect after the
        fix by comparing production against the pre-D4 arithmetic. 3.56% here
        against the 2.736% and 2.339% on the two real TEMPO NO2 bundles -- same
        order, same mechanism. Pinned tightly, so reverting the fix, or
        changing how much it moves, fails rather than passing quietly."""
        np = self.np

        sel = self.partial > 0
        moved = delta_pct(self.production, self.legacy, sel)
        self.assertAlmostEqual(moved, PARTIAL_POPULATION_DELTA_PCT, places=6)
        self.assertGreater(moved, 1.0)

        rel = (np.abs(self.production - self.legacy) / np.abs(self.legacy)).where(sel)
        over_1pct = int((rel > 0.01).sum())
        self.assertAlmostEqual(
            over_1pct / int(np.isfinite(rel).sum()),
            PARTIAL_PIXELS_OVER_1PCT_FRACTION,
            places=6,
        )

    def test_the_fix_moved_only_where_a_bucket_is_partly_covered(self):
        """The mechanism claim, stated as an assertion rather than inferred:
        D4 moved the partial-coverage population and left the rest untouched,
        so partial coverage is not merely correlated with the change -- it is
        the whole of it."""
        complete = delta_pct(self.production, self.legacy, self.partial == 0)
        partial = delta_pct(self.production, self.legacy, self.partial > 0)

        self.assertEqual(complete, 0.0)
        self.assertGreater(partial, 1.0)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class BucketFormulationInvariantTests(unittest.TestCase):
    """What "each cadence bucket counts equally" has to mean, tested against
    the bucket-derived formulation directly."""

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _series(self, values, times, cadence_attr="daily"):
        import numpy as np

        return self.xr.DataArray(
            np.array(values, dtype=float).reshape(len(values), 1, 1),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(times, dtype="datetime64[ns]"),
                "lat": [40.0], "lon": [-75.0],
            },
            name="no2",
        )

    def test_a_bucket_counts_once_however_many_granules_it_holds(self):
        """Four granules on day one and one on day two: day one is still one
        day. Splitting a day's coverage into more scans must not make that day
        louder."""
        dense = self._series(
            [10.0, 10.0, 10.0, 10.0, 0.0],
            ["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00",
             "2024-01-01T18:00", "2024-01-02T12:00"],
        )
        sparse = self._series(
            [10.0, 0.0], ["2024-01-01T12:00", "2024-01-02T12:00"],
        )

        self.assertAlmostEqual(
            _scalar(mean_of_bucket_means(dense, "time", "daily")), 5.0, places=12,
        )
        self.assertAlmostEqual(
            _scalar(mean_of_bucket_means(sparse, "time", "daily")), 5.0, places=12,
        )

    def test_a_bucket_counts_once_even_when_masking_thinned_it(self):
        """The invariant that the current implementation does not hold. Day
        one's four granules were cut to one by masking or by swath coverage;
        day one still measured 10, and the period still averages (10+0)/2."""
        thinned = self._series(
            [10.0, self.np.nan, self.np.nan, self.np.nan, 0.0],
            ["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00",
             "2024-01-01T18:00", "2024-01-02T12:00"],
        )

        self.assertAlmostEqual(
            _scalar(mean_of_bucket_means(thinned, "time", "daily")), 5.0, places=12,
        )

    def test_tiling_one_granule_into_several_leaves_the_answer_alone(self):
        """The swath-tiling case in miniature. One granule covering the whole
        domain, versus two granules covering half each with the same values,
        is the same measurement described twice -- the bucket formulation must
        not care which way the producer sliced it."""
        np = self.np

        whole = self.xr.DataArray(
            np.array([[[4.0, 6.0]], [[1.0, 3.0]]]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(["2024-01-01T12:00", "2024-01-02T12:00"],
                                 dtype="datetime64[ns]"),
                "lat": [40.0], "lon": [-75.0, -74.0],
            },
        )
        tiled = self.xr.DataArray(
            np.array([[[4.0, np.nan]], [[np.nan, 6.0]], [[1.0, 3.0]]]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(["2024-01-01T10:00", "2024-01-01T14:00",
                                  "2024-01-02T12:00"], dtype="datetime64[ns]"),
                "lat": [40.0], "lon": [-75.0, -74.0],
            },
        )

        from_whole = mean_of_bucket_means(whole, "time", "daily")
        from_tiled = mean_of_bucket_means(tiled, "time", "daily")

        self.assertEqual(list(from_whole.values.ravel()), [2.5, 4.5])
        self.assertEqual(list(from_tiled.values.ravel()), [2.5, 4.5])

    def test_an_all_nan_bucket_abstains_instead_of_voting_zero(self):
        """A bucket with nothing left at a pixel has no value to contribute;
        it must drop out of the average rather than pull it toward zero."""
        series = self._series(
            [10.0, self.np.nan, 4.0],
            ["2024-01-01T12:00", "2024-01-02T12:00", "2024-01-03T12:00"],
        )

        self.assertAlmostEqual(
            _scalar(mean_of_bucket_means(series, "time", "daily")), 7.0, places=12,
        )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class TemporalMeanProductionPathTests(unittest.TestCase):
    """Finding #11's intent, asserted through every production entry point that
    reduces time: the private reduction, ``temporal_mean`` (the vertical
    profile's) and ``aggregate`` (the map's). All three must agree, or a
    profile and a period map drawn from one bundle would disagree."""

    def setUp(self):
        import numpy as np
        import xarray as xr

        from tta_backend.preprocessing.aggregation_service import AggregationService

        self.np, self.xr = np, xr
        self.svc = AggregationService()
        self.col_info = {"primary_var": "no2", "cadence": "daily", "fill_value": -999.0}

    def test_a_thinned_bucket_still_counts_as_one_whole_bucket(self):
        """The defect D4 fixed, kept as a regression test with the pre-D4
        answer named. Day one's four granules were cut to one at this pixel;
        day one still measured 10, so the two-day period averages (10 + 0) / 2.
        The old arithmetic gave that surviving granule a quarter vote and
        answered 2.0 -- asserted here so a revert cannot pass."""
        ds = _thinned_bucket_dataset(self.xr, self.np)

        got = self.svc._cadence_weighted_mean(ds["no2"], "time", "daily")
        before = legacy_granule_count_weighted_mean(ds["no2"], "time", "daily")

        self.assertAlmostEqual(float(got.sel(lat=40.0, lon=THINNED_LON)), 5.0, places=12)
        self.assertAlmostEqual(float(before.sel(lat=40.0, lon=THINNED_LON)), 2.0, places=12)

    def test_the_whole_bucket_next_door_was_not_disturbed(self):
        """The complete-coverage answer the fix had to leave alone. Same grid,
        same buckets -- the neighbouring pixel every granule covered averages
        (20 + 0) / 2 = 10.0 before D4, after D4, and under the reference."""
        ds = _thinned_bucket_dataset(self.xr, self.np)

        got = self.svc._cadence_weighted_mean(ds["no2"], "time", "daily")
        before = legacy_granule_count_weighted_mean(ds["no2"], "time", "daily")
        reference = mean_of_bucket_means(ds["no2"], "time", "daily")

        for field in (got, before, reference):
            self.assertAlmostEqual(float(field.sel(lat=40.0, lon=WHOLE_LON)), 10.0, places=12)

    def test_temporal_mean_counts_a_thinned_bucket_as_a_whole_bucket(self):
        """``temporal_mean`` is the reduction the vertical profile calls; it
        must deliver Finding #11's intent at a pixel whose bucket was
        thinned."""
        ds = _thinned_bucket_dataset(self.xr, self.np)

        got = self.svc.temporal_mean(ds["no2"], "time", "daily")

        self.assertAlmostEqual(float(got.sel(lat=40.0, lon=THINNED_LON)), 5.0, places=12)

    def test_aggregate_counts_a_thinned_bucket_as_a_whole_bucket(self):
        """The same invariant through the map path, so the fix cannot sit in
        ``temporal_mean`` alone and leave the period map disagreeing with the
        profile drawn from the same bundle."""
        ds = _thinned_bucket_dataset(self.xr, self.np)

        result = self.svc.aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)

        self.assertAlmostEqual(
            float(result.ds["no2"].sel(lat=40.0, lon=THINNED_LON)), 5.0, places=12,
        )

    def test_aggregate_keeps_the_thinned_timesteps_that_expose_the_bug(self):
        """Why the fixture needs two pixels. ``_valid_time_indices`` drops a
        timestep with no finite value ANYWHERE, so on a single-pixel grid the
        three thinned granules disappear before the reduction and ``aggregate``
        returns the intended 5.0 for the wrong reason. Here all five timesteps
        survive, which is what real partial coverage looks like -- a granule
        that missed this pixel and covered its neighbour."""
        ds = _thinned_bucket_dataset(self.xr, self.np)

        result = self.svc.aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)

        self.assertEqual(result.meta["n_granules"], 5)
        self.assertAlmostEqual(
            float(result.ds["no2"].sel(lat=40.0, lon=WHOLE_LON)), 10.0, places=12,
        )
