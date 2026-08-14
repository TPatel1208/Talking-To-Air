"""T59 Phase 3: the bucketed reduction that produces a frame stack.

Scope: the reduction and its disclosure only. Nothing here touches the blob
store (Phase 4), ``plot_singular`` (Phase 5) or the frontend (Phase 6).

The spine of this phase is D5a: **every scientific quantity is computed at
NATIVE resolution, before the block mean.** The float32 frame array exists to
be rendered and stored, and nothing is ever derived from it. Phase 2 measured
what happens if you forget -- at k=8 a frame has already lost 16% of its own
p98 and 70% of its max, and a block is finite if ANY cell in it is finite, so a
sparse hour block-means to 99.6% apparent coverage where the truth is 94.7%.
"""
from __future__ import annotations

import importlib.util
import unittest


def _field(xr, np, times, frames, *, lats=None, lons=None, name="no2"):
    """A (time, lat, lon) field from a list of 2-D frames."""
    lats = np.linspace(30.0, 45.0, 4) if lats is None else np.asarray(lats, dtype=float)
    lons = np.linspace(-100.0, -85.0, 4) if lons is None else np.asarray(lons, dtype=float)
    return xr.DataArray(
        np.stack([np.asarray(f, dtype=float) for f in frames]),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": lats,
            "lon": lons,
        },
        name=name,
    )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class BucketAxisSpansTheRequestTests(unittest.TestCase):
    """Finding 3, the trap this phase is most likely to encode wrongly.

    ``_valid_time_indices`` drops invalid timesteps BEFORE the reduction
    (``valid_da = da.isel({time_dim: valid_indices})``), so a ``groupby(bucket)``
    over what survives emits only buckets that still hold a granule. An hour
    everything was masked out of, and an hour nothing was ever retrieved for,
    both VANISH from the axis instead of appearing as empty -- the naive
    implementation compresses time and the scrubber silently lies about when
    the gap was.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr
        nan = np.nan
        # Four requested days. Day 2's single granule was gutted by masking;
        # day 3 has no granule at all. Both must survive onto the axis.
        self.field = _field(
            xr, np,
            ["2024-01-01T12:00", "2024-01-02T12:00", "2024-01-04T12:00"],
            [
                np.full((4, 4), 10.0),
                np.full((4, 4), nan),
                np.full((4, 4), 20.0),
            ],
        )
        self.span = ("2024-01-01T00:00:00", "2024-01-04T23:59:59")

    def _stack(self):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            self.field, time_dim="time", cadence="daily", span=self.span,
        )

    def test_every_requested_bucket_is_on_the_axis(self):
        """Four days were asked for, so the axis has four stops -- not the two
        that still hold data. A scrubber whose axis is built from surviving
        granules puts January 4th where January 2nd was."""
        stack = self._stack()

        self.assertEqual(
            [f.t_start for f in stack.frames],
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00",
             "2024-01-03T00:00:00", "2024-01-04T00:00:00"],
        )

    def test_an_emptied_bucket_is_present_and_null(self):
        """Day 2's granule survived retrieval and did not survive masking; day
        3 was never retrieved. Neither is a measurement of zero, so both render
        as absent rather than as a clean low field."""
        np = self.np
        stack = self._stack()

        self.assertTrue(np.all(np.isnan(stack.values[1])))
        self.assertTrue(np.all(np.isnan(stack.values[2])))
        self.assertEqual([f.n_granules for f in stack.frames], [1, 0, 0, 1])

    def test_the_buckets_that_do_hold_data_still_hold_it(self):
        """The axis extension must not have shifted the values onto the wrong
        stops -- the failure mode a purely structural assertion would miss."""
        np = self.np
        stack = self._stack()

        self.assertTrue(np.allclose(stack.values[0], 10.0))
        self.assertTrue(np.allclose(stack.values[3], 20.0))


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CadenceIsAPreconditionTests(unittest.TestCase):
    """A frame axis is a claim about intervals, and an unknown cadence has no
    intervals to claim.

    This is not hypothetical. A Harmony bundle member carries no ``short_name``,
    so nothing resolves from the file alone and ``cadence`` lands on
    ``"unknown"`` -- which is exactly when ``_cadence_weighted_mean`` degrades
    to a plain unweighted mean. Silently inventing a bucket width here would
    put a scrubber on top of that, labeled with intervals the product never
    published.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.field = _field(
            xr, np, ["2024-01-01T12:00", "2024-01-02T12:00"],
            [np.ones((4, 4)), np.ones((4, 4)) * 2],
        )

    def test_an_unknown_cadence_is_refused_by_name(self):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        with self.assertRaises(ValueError) as raised:
            build_frame_stack(self.field, time_dim="time", cadence="unknown")

        self.assertIn("unknown", str(raised.exception))

    def test_a_monthly_product_steps_by_months_not_by_days(self):
        """A month is not a fixed duration, so the axis walks month starts.
        Flooring to a 30-day offset would drift a long span out of alignment
        and label February with January's stamp."""
        import numpy as np
        import xarray as xr

        field = _field(
            xr, np, ["2024-01-15T12:00", "2024-03-15T12:00"],
            [np.ones((4, 4)), np.ones((4, 4)) * 3],
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="monthly")

        self.assertEqual(
            [(f.t_start, f.t_end) for f in stack.frames],
            [("2024-01-01T00:00:00", "2024-02-01T00:00:00"),
             ("2024-02-01T00:00:00", "2024-03-01T00:00:00"),
             ("2024-03-01T00:00:00", "2024-04-01T00:00:00")],
        )
        self.assertEqual([f.n_granules for f in stack.frames], [1, 0, 1])


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class BlockMeanToTheCellCeilingTests(unittest.TestCase):
    """D5: frames downsample by BLOCK MEAN to a cell ceiling, and the payload
    records the realized count rather than the ceiling.

    ``k = ceil(sqrt(total / target))`` is an integer, so a real grid lands at
    70-96% of the target and never on it -- 14,124 cells measured against a
    20,000 budget on a 535x658 regional grid, 19,107 on the full 2950x5771
    TEMPO domain. A payload that echoed the budget would describe a resolution
    the frames do not have.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr
        # 5x5 with a target of 6 gives k=3 and a 2x2 frame: both axes have a
        # partial trailing block, which is where "pad" and "trim" differ.
        self.field = _field(
            xr, np, ["2024-01-01T12:00"], [np.arange(25.0).reshape(5, 5)],
            lats=np.linspace(30.0, 34.0, 5), lons=np.linspace(-100.0, -96.0, 5),
        )

    def _stack(self, target_cells=6):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            self.field, time_dim="time", cadence="daily", target_cells=target_cells,
        )

    def test_each_frame_cell_is_the_mean_of_its_block(self):
        """A block mean, not a stride. Phase 2 measured that the stride is
        actually the better peak-preserver (50.6% vs 29.8% of a single-cell
        max retained); the block mean is chosen anyway because it degrades
        deterministically and preserves the field's mass, and D5a's rule --
        every scientific quantity comes from the native field -- is what pays
        for that choice."""
        np = self.np
        stack = self._stack()

        # arange(25).reshape(5, 5) blocked at k=3, pad skipped.
        self.assertTrue(np.allclose(
            stack.values[0], np.array([[6.0, 8.5], [18.5, 21.0]]),
        ))

    def test_the_realized_cell_count_is_reported_not_the_ceiling(self):
        """25 cells against a 6-cell budget quantizes to 4, not 6. 20,000 is a
        ceiling, and the number the reader is told has to be the one the frames
        actually carry."""
        stack = self._stack(target_cells=6)

        self.assertEqual(stack.cells_per_frame, 4)
        self.assertEqual(stack.coarsen_k, (3, 3))
        self.assertEqual(stack.values.shape, (1, 2, 2))

    def test_a_grid_already_under_the_ceiling_is_left_alone(self):
        """Nothing is gained by blurring a small region, and a k of 1 keeps the
        frames at native resolution where they can afford to be."""
        np = self.np
        stack = self._stack(target_cells=100)

        self.assertEqual(stack.coarsen_k, (1, 1))
        self.assertEqual(stack.cells_per_frame, 25)
        self.assertTrue(np.allclose(stack.values[0], np.arange(25.0).reshape(5, 5)))

    def test_the_region_edge_is_padded_not_trimmed(self):
        """``boundary="trim"`` silently clips up to k-1 rows and columns off the
        region edge -- 29 of them at the k=30 the full TEMPO domain needs -- and
        the edge of a masked region is where the interesting gradients are.
        With "pad", the trailing block is the mean of the cells that exist: the
        NaN pad is skipped, not averaged in."""
        stack = self._stack()

        self.assertEqual(stack.values.shape[1:], (2, 2))
        # The 2x2 corner block: 18, 19, 23, 24. Diluted by five NaN pad cells
        # it would read 8.4; trimmed away it would not be here at all.
        self.assertAlmostEqual(float(stack.values[0][1, 1]), 21.0, places=5)
        # The frame's own axes are the block centres, so the reader is not told
        # a coarse cell sits where its first native cell was.
        self.assertEqual(len(stack.lats), 2)
        self.assertAlmostEqual(float(stack.lats[0]), 31.0, places=5)

    def test_an_all_nan_block_stays_absent_rather_than_becoming_zero(self):
        """Verified against xarray in Phase 2, and asserted here because it is
        the difference between "nothing was retrieved in this block" and "this
        block measured zero pollution"."""
        np = self.np
        values = np.arange(25.0).reshape(5, 5)
        values[3:, 3:] = np.nan
        field = _field(
            self.xr, np, ["2024-01-01T12:00"], [values],
            lats=np.linspace(30.0, 34.0, 5), lons=np.linspace(-100.0, -96.0, 5),
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily", target_cells=6)

        self.assertTrue(np.isnan(stack.values[0][1, 1]))

    def test_the_frame_array_is_float32_and_the_reduction_is_not(self):
        """The pipeline is float64 end to end, so the float32 the blob stores
        is a deliberate narrowing performed once, explicitly, at the edge --
        not something the reduction drifted into."""
        stack = self._stack()

        self.assertEqual(stack.values.dtype.name, "float32")


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class NativeResolutionStatisticsTests(unittest.TestCase):
    """D5a, the phase's spine: a frame's statistics are reduced from the
    NATIVE bucket field, never from the array the user downloads.

    Phase 2 measured the cost of getting this wrong: at k=8 a block-meaned
    frame retains 29.8% of its own single-cell max and 84.2% of its p98. A
    scrubber exists to find the hour a plume peaked, so a per-frame max read
    off the rendering grid would understate exactly the thing being looked for
    -- the same defect ``_field_statistics`` already carries a docstring about
    for the map (true max 9.24e17 reported as 6.90e16).
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr
        # One hot cell against a flat background. At k=3 its block mean is
        # (100 + 8) / 9 = 12.0, so native and reduced answers are 8x apart and
        # no tolerance can confuse them.
        values = np.ones((5, 5))
        values[1, 1] = 100.0
        self.field = _field(
            xr, np, ["2024-01-01T12:00"], [values],
            lats=np.linspace(30.0, 34.0, 5), lons=np.linspace(-100.0, -96.0, 5),
        )

    def _stack(self):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            self.field, time_dim="time", cadence="daily", target_cells=6,
        )

    def test_the_peak_reported_is_the_peak_that_was_measured(self):
        """100, the value in the granule -- not 12.0, the value in the picture."""
        stack = self._stack()

        self.assertAlmostEqual(stack.frames[0].statistics["max"], 100.0, places=6)
        self.assertNotAlmostEqual(stack.frames[0].statistics["max"], 12.0, places=3)

    def test_the_floor_reported_is_the_floor_that_was_measured(self):
        """The same rule downward. A block mean pulls a minimum UP toward its
        neighbours, so a frame whose real floor was 1.0 would report 4.0."""
        stack = self._stack()

        self.assertAlmostEqual(stack.frames[0].statistics["min"], 1.0, places=6)

    def test_the_frame_mean_is_the_one_regional_mean_definition(self):
        """cos(latitude) area-weighted, the same ``area_weighted_mean`` the map
        and the statistics tool use. A frame that averaged its cells plainly
        would disagree with the map it was derived from, by a few percent on a
        continental region and invisibly on a small one."""
        from tta_backend.preprocessing.aggregation_service import area_weighted_mean

        stack = self._stack()
        native_bucket_field = self.field.isel(time=0)

        self.assertAlmostEqual(
            stack.frames[0].statistics["mean"],
            area_weighted_mean(native_bucket_field),
            places=9,
        )

    def test_an_empty_frame_reports_no_statistics_rather_than_zeroes(self):
        """A bucket nothing survives in has no maximum. Reporting 0.0 would put
        a measurement where there is an absence -- the same failure the blank
        map itself is guarded against."""
        np = self.np
        field = _field(
            self.xr, np,
            ["2024-01-01T12:00", "2024-01-03T12:00"],
            [np.ones((5, 5)), np.ones((5, 5))],
            lats=np.linspace(30.0, 34.0, 5), lons=np.linspace(-100.0, -96.0, 5),
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertEqual(stack.frames[1].statistics, {"count": 0})


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class ValidFractionTests(unittest.TestCase):
    """D10: ``valid_fraction`` is a correctness requirement, not decoration.

    A 90%-masked bucket renders as a clean low uniform field, and to someone
    scrubbing for an event that is indistinguishable from a calm hour. The
    number is what makes them different, so it has to be right in the two ways
    Phase 2 and T56 each found it can be wrong: derived from the frame grid, or
    denominated on the bounding box.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _sparse_but_evenly_spread(self):
        """Every row a third covered, so the cos(lat)-weighted answer is 1/3
        exactly whatever the weights are -- and every 3x3 block holds three
        finite cells, so the BLOCK-MEANED frame is 100% covered."""
        np = self.np
        row, col = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
        values = np.where((row + col) % 3 == 0, 5.0, np.nan)
        return _field(
            self.xr, np, ["2024-01-01T12:00"], [values],
            lats=np.linspace(30.0, 55.0, 6), lons=np.linspace(-100.0, -95.0, 6),
        )

    def test_coverage_is_measured_natively_not_off_the_frame_grid(self):
        """The hazard the PRD did not anticipate and Phase 2 measured: a block
        is finite if ANY cell in it is finite, so block-meaning a sparse hour
        reports 99.6% coverage where the truth is 94.7%, and it gets worse as
        the budget shrinks. Here the two answers are 33% and 100% -- the frame
        grid would tell a scrubbing user that the emptiest hour of the week was
        the best-observed one."""
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(
            self._sparse_but_evenly_spread(), time_dim="time", cadence="daily",
            target_cells=4,
        )

        self.assertEqual(stack.coarsen_k, (3, 3))
        self.assertAlmostEqual(stack.frames[0].valid_fraction, 1.0 / 3.0, places=9)
        # And the frame array really is the fully-covered thing that would have
        # produced 1.0 -- so this is a measured contrast, not a lucky fixture.
        self.assertTrue(bool(self.np.isfinite(stack.values[0]).all()))

    def test_coverage_is_area_weighted_like_every_other_regional_figure(self):
        """cos(latitude), the same weights ``area_weighted_mean`` and the QA
        pass rate use. A plain cell ratio sitting in the same statistics block
        as an area-weighted mean over-counts shrunken poleward cells, and the
        two numbers then describe different fields."""
        np = self.np
        field = _field(
            self.xr, np, ["2024-01-01T12:00"],
            [np.array([[5.0, 5.0], [np.nan, np.nan]])],
            lats=[0.0, 60.0], lons=[-100.0, -99.0],
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily")

        # cos(0) = 1, cos(60) = 0.5: the observed row is 2/3 of the region's
        # area, not the 1/2 of its cells an unweighted count would report.
        self.assertAlmostEqual(stack.frames[0].valid_fraction, 2.0 / 3.0, places=9)

    def test_the_denominator_is_the_region_not_the_bounding_box(self):
        """T56 got this denominator wrong once, and it is worth restating why
        it matters: a region shaped like anything but a rectangle has a bounding
        box full of cells the geometry masked out. Counting those as missing
        observations reports a complete retrieval over the continental US as
        60% covered, and the reader sees a data problem that does not exist.

        Here the right half of the grid is out of region -- NaN in every
        timestep -- and the left half was fully observed. That is 100% of the
        region and 50% of the box."""
        np = self.np
        values = np.full((4, 4), 7.0)
        values[:, 2:] = np.nan
        field = _field(
            self.xr, np, ["2024-01-01T12:00"], [values],
            lats=[0.0, 0.0, 0.0, 0.0], lons=[-100.0, -99.0, -98.0, -97.0],
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        # cos(0) = 1 everywhere, so the in-region area is simply its 8 cells.
        in_region = build_frame_stack(
            field, time_dim="time", cadence="daily", region_area=8.0,
        )
        bbox = build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertAlmostEqual(in_region.frames[0].valid_fraction, 1.0, places=9)
        self.assertAlmostEqual(bbox.frames[0].valid_fraction, 0.5, places=9)

    def test_an_empty_bucket_is_zero_covered_not_absent(self):
        """The one figure an empty frame still owes the reader. "0% of the
        region had a value" is a measurement; leaving it null would let the
        frontend fall back to the previous frame's number."""
        np = self.np
        field = _field(
            self.xr, np,
            ["2024-01-01T12:00", "2024-01-03T12:00"],
            [np.ones((4, 4)), np.ones((4, 4))],
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertEqual(stack.frames[1].valid_fraction, 0.0)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class QaRollUpByAreaSumsTests(unittest.TestCase):
    """Finding 12: rolling per-timestep QA rates up to a bucket MUST sum the
    cos(latitude)-weighted ``passing_area``/``checked_area`` and divide once.

    ``_by_time_rates`` has already divided, and averaging its ratios back up
    weights a nearly-empty scan exactly as heavily as a full one. On a
    swath-tiled product that is not a rounding difference: one hour holds a
    sliver that clipped the region's corner and a scan that covered all of it.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr
        self.field = _field(
            xr, np,
            ["2024-01-01T06:00", "2024-01-01T18:00"],
            [np.full((4, 4), 5.0), np.full((4, 4), 6.0)],
        )

    def _stack(self, qa_counts):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            self.field, time_dim="time", cadence="daily", qa_counts=qa_counts,
        )

    def test_a_sliver_scan_does_not_vote_as_loudly_as_a_full_one(self):
        """One day, two scans. The sliver checked 1 unit of area and passed all
        of it; the full scan checked 99 and passed none. The day's pass rate is
        1%. Averaging the two rates would report 50% -- a fifty-fold overstatement
        driven entirely by a scan that barely clipped the region."""
        stack = self._stack({
            "times": ["2024-01-01T06:00:00", "2024-01-01T18:00:00"],
            "checked_area_by_time": [1.0, 99.0],
            "passing_area_by_time": [1.0, 0.0],
        })

        self.assertAlmostEqual(stack.frames[0].qa_pass_rate, 0.01, places=9)

    def test_a_bucket_with_nothing_checkable_reports_no_rate_at_all(self):
        """``None``, not 0.0 -- the same absent-not-zero honesty
        ``_by_time_rates`` gives a single timestep. A bucket whose scans were
        entirely fill or entirely outside the region had no QA question to
        answer, and answering "0% passed" would read as a quality collapse."""
        stack = self._stack({
            "times": ["2024-01-01T06:00:00", "2024-01-01T18:00:00"],
            "checked_area_by_time": [0.0, 0.0],
            "passing_area_by_time": [0.0, 0.0],
        })

        self.assertIsNone(stack.frames[0].qa_pass_rate)

    def test_a_bucket_qa_emptied_still_reports_the_rate_that_emptied_it(self):
        """The disclosure that makes an empty frame legible. This hour WAS
        observed and QA rejected all of it, so the frame is blank and its pass
        rate is 0% -- which is a different fact from "no granule was retrieved",
        and the only thing on the frame that can tell them apart."""
        np = self.np
        field = _field(
            self.xr, np,
            ["2024-01-01T12:00", "2024-01-02T12:00"],
            [np.full((4, 4), np.nan), np.full((4, 4), 6.0)],
        )

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily",
            qa_counts={
                "times": ["2024-01-01T12:00:00", "2024-01-02T12:00:00"],
                "checked_area_by_time": [40.0, 40.0],
                "passing_area_by_time": [0.0, 40.0],
            },
        )

        self.assertEqual(stack.frames[0].n_granules, 0)
        self.assertEqual(stack.frames[0].qa_pass_rate, 0.0)
        self.assertEqual(stack.frames[1].qa_pass_rate, 1.0)

    def test_qa_timesteps_are_matched_by_timestamp_not_by_position(self):
        """The counters are reduced over the inner-ALIGNED arrays and the field
        handed here may already have had its invalid timesteps dropped, so the
        two are not index-parallel. Bucketing each counter by its own timestamp
        is what makes a mismatch harmless instead of silently attributing one
        hour's quality to another."""
        stack = self._stack({
            # Reversed order, and carrying a timestep the field does not have.
            "times": ["2024-01-02T09:00:00", "2024-01-01T18:00:00",
                      "2024-01-01T06:00:00"],
            "checked_area_by_time": [50.0, 99.0, 1.0],
            "passing_area_by_time": [50.0, 0.0, 1.0],
        })

        self.assertAlmostEqual(stack.frames[0].qa_pass_rate, 0.01, places=9)

    def test_no_qa_counters_means_no_rate_rather_than_a_fabricated_one(self):
        """QA masking that never ran leaves the key absent, matching the
        existing downgrade-to-not-applied guard: "QA didn't run" has to stay
        distinguishable from a real 0% pass rate."""
        stack = self._stack(None)

        self.assertIsNone(stack.frames[0].qa_pass_rate)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CadenceTierIsExactTests(unittest.TestCase):
    """D14, tier one: where the native cadence fits the frame budget, the frames
    ARE the cadence buckets and the period map is derived from them.

    D4's guarantee, made structural rather than hoped for -- the map is the
    average of what the user scrubs.

    **This grid does not coarsen**, and that is a scope limit rather than an
    incidental fixture choice. Everything below holds where the block mean is
    the identity function; above the cell ceiling it does not, because the
    block mean and the across-frame mean do not commute under partial coverage
    (Phase 8 §1, measured at 1.876% on a real regional TEMPO chart). The
    property IS real in this regime and these tests pin it -- but the regime
    real charts are in is ``ShippedArrayAgreementTests``', and that is where
    the claim a reader is shown gets checked.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr
        nan = np.nan
        # The partial-coverage mechanism the D4 characterization pinned down:
        # day one holds four granules and only the first reaches the left-hand
        # pixel. A period map that is NOT derived from the buckets gives that
        # surviving granule a quarter vote and answers 2.0 where the bucket
        # form answers 5.0, so this fixture can tell the two apart.
        self.field = xr.DataArray(
            np.array([
                [[10.0, 20.0]],
                [[nan, 20.0]],
                [[nan, 20.0]],
                [[nan, 20.0]],
                [[0.0, 0.0]],
            ]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array([
                    "2024-01-01T06:00", "2024-01-01T10:00", "2024-01-01T14:00",
                    "2024-01-01T18:00", "2024-01-02T12:00",
                ], dtype="datetime64[ns]"),
                "lat": [40.0], "lon": [-75.0, -74.0],
            },
            name="no2",
        )

    def _stack(self):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(self.field, time_dim="time", cadence="daily")

    def test_the_frames_are_the_cadence_buckets(self):
        stack = self._stack()

        self.assertEqual(stack.tier, "cadence")
        self.assertEqual(stack.buckets_per_frame, 1)
        self.assertEqual(len(stack.frames), 2)

    def test_the_period_map_is_the_average_of_the_frames(self):
        """Average the stack, get the map -- exactly, on a grid that does not
        coarsen. Frame 0 sits on the same grid by the same method so the two
        arrays are directly comparable at all.

        Kept as it was, and deliberately not generalized: at k=(1,1) this is a
        true property with a real mechanism behind it. What it is not is a
        statement about a downloaded blob, whose grid has been block-meaned --
        ``frame_grid_delta`` is the quantity that speaks to that, and the
        sentences a reader sees are written from it, not from here."""
        np = self.np
        stack = self._stack()

        self.assertTrue(np.allclose(
            np.nanmean(stack.values, axis=0), stack.period_values, rtol=1e-6,
        ))

    def test_the_derived_map_is_the_map_the_aggregate_produces(self):
        """And that average is the period map itself -- ``_cadence_weighted_mean``,
        the reduction every multi-granule map already goes through. If these
        ever diverged, the scrubber and the Map tab would be showing two
        different measurements of one week."""
        np = self.np

        from tta_backend.preprocessing.aggregation_service import AggregationService

        stack = self._stack()
        period = AggregationService()._cadence_weighted_mean(
            self.field, "time", "daily",
        )

        # 5.0 at the thinned pixel, not the 2.0 granule-weighting gives it.
        self.assertAlmostEqual(float(period.sel(lat=40.0, lon=-75.0)), 5.0, places=12)
        self.assertTrue(np.allclose(
            stack.period_values, np.asarray(period.values, dtype="float32"), rtol=1e-6,
        ))

    def test_an_exact_tier_quantifies_nothing_because_there_is_nothing_to_quantify(self):
        """No delta in tier one. Reporting 0.0% would invite the reader to
        wonder what was measured; the relationship is identity, and saying so
        is different from measuring it and finding no difference."""
        stack = self._stack()

        self.assertIsNone(stack.delta)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CoarsenedTierAndItsDeltaTests(unittest.TestCase):
    """D14, tier two: above the frame budget the scrubber is a DIFFERENT
    temporal aggregation, and D16's delta is what makes that honest.

    D3 and D4 are in tension here. The map's weighting is at the product
    cadence, so grouping cadence buckets to fit 60 stops means the mean of the
    coarse means is not the period mean. Refusing frames above 60 would cap a
    TEMPO scrubber at ~5 days; coarsening silently would be the exact failure
    this design exists to avoid. Quantifying it costs a disclosure line.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _daily(self, per_pixel, lats=(0.0,), n_days=5):
        """One granule a day for ``n_days``; ``per_pixel[lat][lon]`` is that
        pixel's series over the days."""
        np = self.np
        series = np.asarray(per_pixel, dtype=float)  # (lat, lon, day)
        values = np.moveaxis(series, -1, 0)
        return self.xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-0{d + 1}T12:00" for d in range(n_days)],
                    dtype="datetime64[ns]",
                ),
                "lat": list(lats),
                "lon": [-100.0 + i for i in range(series.shape[1])],
            },
            name="no2",
        )

    def _stack(self, field, max_frames=2):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            field, time_dim="time", cadence="daily", max_frames=max_frames,
        )

    def test_cadence_buckets_are_grouped_until_the_budget_fits(self):
        """Five days into two stops is three days then two -- and the frame's
        own interval says so, because a stop labeled with one day's timestamp
        that shows three days averaged is a lie the picture cannot correct."""
        field = self._daily([[[3.0, 6.0, 9.0, 100.0, 200.0]]])
        stack = self._stack(field)

        self.assertEqual(stack.tier, "coarsened")
        self.assertEqual(stack.buckets_per_frame, 3)
        self.assertEqual(
            [(f.t_start, f.t_end) for f in stack.frames],
            [("2024-01-01T00:00:00", "2024-01-04T00:00:00"),
             ("2024-01-04T00:00:00", "2024-01-06T00:00:00")],
        )

    def test_a_frame_is_the_mean_of_its_cadence_buckets_not_of_its_granules(self):
        """The coarse frame keeps D4's rule one level up: each cadence bucket
        inside it counts once. Otherwise the tier that exists to fit long spans
        would reintroduce granule weighting exactly where sampling is most
        uneven."""
        field = self._daily([[[3.0, 6.0, 9.0, 100.0, 200.0]]])
        stack = self._stack(field)

        self.assertAlmostEqual(float(stack.values[0][0, 0]), 6.0, places=5)
        self.assertAlmostEqual(float(stack.values[1][0, 0]), 150.0, places=5)
        self.assertEqual([f.n_granules for f in stack.frames], [3, 2])

    def test_the_delta_is_a_ratio_of_weighted_sums_not_a_mean_of_ratios(self):
        """D16, and its own diagnostic vindicated it: a real bundle reported a
        max per-pixel relative difference of 1017% at a pixel where the period
        map sits near zero. A headline built from per-pixel ratios would have
        been that pixel and nothing else.

        Three pixels here. Two disagree by +1.2 and -1.2 against maps of 4.8
        and 7.2; the third's map is 0.0004, so its own relative error is 125,000%
        while it contributes 0.5 of absolute difference. The honest headline is
        24.2%."""
        field = self._daily([[
            [0.0, 0.0, 0.0, 12.0, 12.0],        # F-M = +1.2 against M = 4.8
            [12.0, 12.0, 12.0, 0.0, 0.0],       # F-M = -1.2 against M = 7.2
            [2.0, 2.0, 2.0, -2.999, -2.999],    # F-M = -0.4999 against M = 0.0004
        ]])
        stack = self._stack(field)

        # sum|F-M| / sum|M| = 2.8999 / 12.0004
        self.assertAlmostEqual(stack.delta["headline"], 2.8999 / 12.0004, places=9)
        # The two answers this metric was chosen over, named so a future
        # "simplification" into either one fails here instead of shipping.
        self.assertNotAlmostEqual(stack.delta["headline"], 416.72, places=2)
        self.assertNotAlmostEqual(stack.delta["headline"], -2.8999 / 12.0004, places=6)

    def test_opposite_errors_do_not_cancel_into_a_clean_bill_of_health(self):
        """Never a signed mean. A field off by +10% here and -10% there sums to
        about zero, and reporting that as the delta would certify a scrubber
        that misstates every pixel it draws."""
        field = self._daily([[
            [0.0, 0.0, 0.0, 12.0, 12.0],
            [12.0, 12.0, 12.0, 0.0, 0.0],
        ]])
        stack = self._stack(field)

        # Signed, this is exactly 0: +1.2 and -1.2 against 4.8 and 7.2.
        self.assertAlmostEqual(stack.delta["headline"], 2.4 / 12.0, places=9)

    def test_the_companion_is_the_worst_pixel_in_physical_units(self):
        """A percentage cannot tell a reader whether the disagreement matters
        for their analysis; the largest absolute difference, in the field's own
        units, can."""
        field = self._daily([[
            [0.0, 0.0, 0.0, 12.0, 12.0],
            [12.0, 12.0, 12.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, -2.999, -2.999],
        ]])
        stack = self._stack(field)

        self.assertAlmostEqual(stack.delta["max_abs"], 1.2, places=9)

    def test_the_delta_is_area_weighted_like_every_other_regional_figure(self):
        """cos(latitude) again. Both pixels here have a period map of 10 and
        only the poleward one disagrees, by 1.0 -- which is 3.3% of the region's
        weighted magnitude and 5.0% of its unweighted one."""
        field = self._daily(
            [[[10.0, 10.0, 10.0, 10.0, 10.0]], [[6.0, 6.0, 6.0, 16.0, 16.0]]],
            lats=(0.0, 60.0),
        )
        stack = self._stack(field)

        self.assertAlmostEqual(stack.delta["headline"], 0.5 / 15.0, places=9)
        self.assertNotAlmostEqual(stack.delta["headline"], 1.0 / 20.0, places=6)

    def test_qa_sums_across_every_cadence_bucket_the_frame_holds(self):
        """The roll-up crosses two levels in this tier -- timesteps into cadence
        buckets, cadence buckets into a frame -- and it has to stay one division
        at the end. Three days into one stop: 1 unit of area passing out of 1,
        then 0 out of 99, then 0 out of 0. The frame passed 1% of what it
        checked; a mean of the two defined daily rates would say 50%."""
        field = self._daily([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily", max_frames=2,
            qa_counts={
                "times": [f"2024-01-0{d + 1}T12:00:00" for d in range(5)],
                "checked_area_by_time": [1.0, 99.0, 0.0, 10.0, 10.0],
                "passing_area_by_time": [1.0, 0.0, 0.0, 5.0, 5.0],
            },
        )

        self.assertEqual(stack.buckets_per_frame, 3)
        self.assertAlmostEqual(stack.frames[0].qa_pass_rate, 0.01, places=9)
        self.assertAlmostEqual(stack.frames[1].qa_pass_rate, 0.5, places=9)

    def test_the_delta_is_measured_natively_not_off_the_frame_grid(self):
        """D5a binds here too. Measuring F against M on the 20,000-cell frame
        grid would compare two block means and report a smaller disagreement
        than the one the frames were built from."""
        field = self._daily([[
            [0.0, 0.0, 0.0, 12.0, 12.0],
            [12.0, 12.0, 12.0, 0.0, 0.0],
        ]])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        native = build_frame_stack(
            field, time_dim="time", cadence="daily", max_frames=2,
        )
        # Two lon cells into one: the block mean averages +1.2 against -1.2 and
        # the disagreement vanishes entirely from the reduced grid.
        reduced = build_frame_stack(
            field, time_dim="time", cadence="daily", max_frames=2, target_cells=1,
        )

        self.assertEqual(reduced.cells_per_frame, 1)
        self.assertAlmostEqual(
            reduced.delta["headline"], native.delta["headline"], places=12,
        )
        self.assertGreater(reduced.delta["headline"], 0.0)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class ShippedArrayAgreementTests(unittest.TestCase):
    """``frame_grid_delta``: what a reader gets by averaging the blob they
    downloaded, against the plane that ships beside it.

    A DIFFERENT question from ``delta``, which is why it is a different field.
    ``delta`` asks whether the coarser temporal aggregation is a different
    measurement, and answers it at native resolution because a +1.2 and a -1.2
    inside one block must not be allowed to cancel. This asks whether the two
    arrays in the payload satisfy the identity the payload claims for them --
    and the only honest place to ask that is on the arrays themselves.

    Phase 8 measured the gap this exists to close: on a real regional TEMPO
    retrieval (352,181 native cells, k=(5,5)) averaging the shipped planes
    misses the shipped period plane by **1.876%**, worst pixel 2.72e15 against
    a map ramp of 5.7e14-3.3e15, where the same bundle at native resolution
    reproduces Phase 3's 0.000002%. The block mean and the across-frame mean do
    not commute under partial coverage: ``period_values`` is
    ``block_mean(mean over buckets)`` and the frames are
    ``mean over buckets of block_mean(...)``, so inside a block whose native
    cells were seen in different intervals the two weight different things.
    Phase 1b's mechanism one level down, reintroduced by the rendering
    downsample.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _days(self, *days, lats=(0.0, 1.0)):
        """One daily granule per argument, over a 2x2 grid that coarsens to a
        single cell -- the smallest grid on which a block can be seen unevenly.
        """
        np = self.np
        return self.xr.DataArray(
            np.stack([np.asarray(day, dtype=float) for day in days]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{d + 1:02d}T12:00" for d in range(len(days))],
                    dtype="datetime64[ns]",
                ),
                "lat": np.asarray(lats, dtype=float),
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )

    def _two_days(self, day_one, day_two):
        return self._days(day_one, day_two)

    def _stack(self, field, *, target_cells=1, **kwargs):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            field, time_dim="time", cadence="daily",
            target_cells=target_cells, **kwargs,
        )

    def test_the_shipped_arrays_disagree_where_a_block_was_seen_unevenly(self):
        """The tracer bullet, and the smallest reproduction of Phase 8 §1.

        One 2x2 block. Both cells of the top row are seen on day one; only the
        left one is seen again on day two. Averaging the two shipped frames
        gives (10 + 20) / 2 = **15**. The period plane block-means the native
        per-cell means -- (10+20)/2 = 15 on the left, 10 on the right -- and
        gives **12.5**. A 20% disagreement between two arrays the payload says
        are each other's average.
        """
        nan = self.np.nan
        field = self._two_days(
            [[10.0, 10.0], [nan, nan]],
            [[20.0, nan], [nan, nan]],
        )

        stack = self._stack(field)

        self.assertEqual(stack.tier, "cadence")
        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertAlmostEqual(stack.frame_grid_delta["headline"], 0.2, places=6)
        self.assertAlmostEqual(stack.frame_grid_delta["max_abs"], 2.5, places=6)

    def test_the_identity_is_exact_when_the_block_mean_is_a_no_op(self):
        """The property the k=1 tests pin, now MEASURED rather than asserted.

        Same field, same uneven coverage, no coarsening: the block mean is the
        identity function, the two reductions commute trivially, and the
        disagreement is a real 0.0 rather than an unexamined claim. This is
        what makes the number safe to publish in tier one -- it does not go
        non-zero because a scrubber exists, only because a block was averaged.
        """
        nan = self.np.nan
        field = self._two_days(
            [[10.0, 10.0], [nan, nan]],
            [[20.0, nan], [nan, nan]],
        )

        stack = self._stack(field, target_cells=10_000)

        self.assertEqual(stack.coarsen_k, (1, 1))
        self.assertEqual(stack.frame_grid_delta["headline"], 0.0)
        self.assertEqual(stack.frame_grid_delta["max_abs"], 0.0)

    def test_tier_two_publishes_two_numbers_and_neither_is_the_other(self):
        """Phase 8 §2's gap: the coarsened tier discloses a delta measured at
        native resolution while the arrays on screen disagree by a different
        amount, and the smaller one was the one printed.

        Four days, two buckets per frame, one 2x2 block. Worked by hand: cell A
        is seen on days 1-3 and cell B on days 1, 3 and 4, so the frames come
        out (12.5, 32.5) on the shipped grid against a period plane of 23.33 --
        **1/28**. At native resolution the same field gives **1/7**. Four times
        apart, in the direction that flatters the arrays on screen. Neither
        number bounds the other, which is the whole reason both are published.
        """
        nan = self.np.nan
        field = self._days(
            [[10.0, 10.0], [nan, nan]],
            [[20.0, nan], [nan, nan]],
            [[30.0, 30.0], [nan, nan]],
            [[nan, 40.0], [nan, nan]],
        )

        stack = self._stack(field, max_frames=2)

        self.assertEqual(stack.tier, "coarsened")
        self.assertEqual(stack.buckets_per_frame, 2)
        self.assertAlmostEqual(stack.delta["headline"], 1 / 7, places=6)
        self.assertAlmostEqual(stack.frame_grid_delta["headline"], 1 / 28, places=6)

    def test_the_two_deltas_never_share_a_basis_string(self):
        """``DELTA_BASIS`` exists because a quantity with two accounts of itself
        is how the screen and the document start disagreeing. Two quantities
        sharing ONE account is the same failure wearing the other shoe: a
        reader given "at native resolution" beside a number measured on the
        frame grid has been told something false about a real measurement."""
        from tta_backend.preprocessing.frame_stack import (
            DELTA_BASIS, FRAME_GRID_DELTA_BASIS,
        )

        nan = self.np.nan
        stack = self._stack(
            self._days(
                [[10.0, 10.0], [nan, nan]],
                [[20.0, nan], [nan, nan]],
                [[30.0, 30.0], [nan, nan]],
                [[nan, 40.0], [nan, nan]],
            ),
            max_frames=2,
        )

        self.assertNotEqual(DELTA_BASIS, FRAME_GRID_DELTA_BASIS)
        self.assertEqual(stack.delta["basis"], DELTA_BASIS)
        self.assertEqual(stack.frame_grid_delta["basis"], FRAME_GRID_DELTA_BASIS)
        self.assertIn("native resolution", DELTA_BASIS)
        self.assertIn("stored", FRAME_GRID_DELTA_BASIS)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class PlaneAgreementIdentityTests(unittest.TestCase):
    """D6a decision 8: the selection planes' two associativity identities hold
    EXACTLY, and the test says so with ``assertEqual`` rather than a tolerance.

    Phase 11's G5 measured both on real float64 planes from two live bundles
    and got ``0.0`` max abs difference on every one, including at partially
    covered and all-NaN cells. There was no floating-point reason to expect
    otherwise: ``max`` and ``min`` SELECT a value rather than accumulating one,
    so there is no summation order for rounding to depend on, and the float32
    narrowing rounds the same number the mean plane's cells were rounded from.
    A tolerance here would be hiding a bug rather than accommodating noise that
    does not exist.

    These reuse ``ShippedArrayAgreementTests``' fixtures deliberately. The
    sharpest possible demonstration that these are different statistics with
    different guarantees is the SAME field, where the mean's own disagreement
    is a real 20% and the selections' is a real zero.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _days(self, *days):
        np = self.np
        return self.xr.DataArray(
            np.stack([np.asarray(day, dtype=float) for day in days]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{d + 1:02d}T12:00" for d in range(len(days))],
                    dtype="datetime64[ns]",
                ),
                "lat": np.array([0.0, 1.0]),
                "lon": np.array([-75.0, -74.0]),
            },
            name="no2",
        )

    def _stack(self, field, **kwargs):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            field, time_dim="time", cadence="daily", target_cells=1,
            statistics=("mean", "max", "min"), **kwargs,
        )

    def _unevenly_seen_block(self):
        """Phase 9's own fixture: both top-row cells seen on day one, only the
        left one seen again on day two. One 2x2 block, seen unevenly."""
        nan = self.np.nan
        return self._days(
            [[10.0, 10.0], [nan, nan]],
            [[20.0, nan], [nan, nan]],
        )

    def test_the_selection_planes_agree_exactly_where_the_mean_plane_does_not(self):
        """Identity A, on the field that makes the mean's version non-trivial.

        Averaging the mean plane's two frames gives 15 against a period plane
        of 12.5 -- 20%, because the block mean and the across-frame mean do not
        commute under partial coverage. Taking the MAX of the max plane's two
        frames gives 20, and its period plane is 20, because max commutes with
        everything. Same field, same block, same coverage: the disagreement is
        a property of the statistic, not of the data.
        """
        stack = self._stack(self._unevenly_seen_block())

        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertAlmostEqual(stack.frame_grid_delta["headline"], 0.2, places=6)
        for name in ("max", "min"):
            plane = stack.planes[name]
            self.assertEqual(
                plane.frame_grid_delta["headline"], 0.0,
                f"{name} plane disagreed with its own period plane",
            )
            self.assertEqual(plane.frame_grid_delta["max_abs"], 0.0)

    def test_the_period_plane_is_the_period_extreme_not_the_period_mean(self):
        """What the identity above is an identity BETWEEN. Stop 0 in max mode
        is the period max (D6a decision 3) -- 20 here, the largest value any
        native cell in the block ever held -- and emphatically not the Map
        tab's 12.5, which is what a stack that reused the mean's period plane
        would render at the same stop."""
        stack = self._stack(self._unevenly_seen_block())

        self.assertEqual(float(stack.planes["max"].period_values[0, 0]), 20.0)
        self.assertEqual(float(stack.planes["min"].period_values[0, 0]), 10.0)
        self.assertAlmostEqual(float(stack.period_values[0, 0]), 12.5, places=4)

    def test_grouping_before_or_after_the_block_max_gives_the_same_answer(self):
        """Identity B, and it needs ``group > 1`` to exist at all.

        Four days into two frames. One stack groups the cadence buckets
        temporally and then block-maxes; the other is built at the cadence tier
        and its frames are grouped afterwards, here, by hand. Max is
        associative, so the orders cannot disagree -- and the check is
        ``array_equal`` on the shipped float32 bytes, not ``allclose``.
        """
        np = self.np
        nan = np.nan
        field = self._days(
            [[10.0, 10.0], [nan, nan]],
            [[20.0, nan], [nan, nan]],
            [[30.0, 30.0], [nan, nan]],
            [[nan, 40.0], [nan, nan]],
        )

        grouped = self._stack(field, max_frames=2)
        per_bucket = self._stack(field, max_frames=4)

        self.assertEqual(grouped.tier, "coarsened")
        self.assertEqual(grouped.buckets_per_frame, 2)
        self.assertEqual(per_bucket.tier, "cadence")

        for name, reducer in (("max", np.max), ("min", np.min)):
            after = grouped.planes[name].values
            before = np.stack([
                reducer(per_bucket.planes[name].values[pair:pair + 2], axis=0)
                for pair in (0, 2)
            ])
            self.assertTrue(
                np.array_equal(after, before),
                f"{name}: grouping after the block reduction gave {after.ravel()}, "
                f"grouping before gave {before.ravel()}",
            )

    def test_the_coarsened_tier_selection_planes_still_agree_exactly(self):
        """The tier whose whole reason for existing is that its temporal
        aggregation is a DIFFERENT measurement -- the mean plane discloses
        1/28 on these arrays -- and the selection planes are still exactly
        zero, because grouping buckets is one more max over the same values."""
        nan = self.np.nan
        stack = self._stack(
            self._days(
                [[10.0, 10.0], [nan, nan]],
                [[20.0, nan], [nan, nan]],
                [[30.0, 30.0], [nan, nan]],
                [[nan, 40.0], [nan, nan]],
            ),
            max_frames=2,
        )

        self.assertAlmostEqual(stack.frame_grid_delta["headline"], 1 / 28, places=6)
        for name in ("max", "min"):
            self.assertEqual(stack.planes[name].frame_grid_delta["headline"], 0.0)

    def test_a_selection_plane_never_carries_the_means_account_of_itself(self):
        """``FRAME_GRID_DELTA_BASIS`` says "mean of the stored frame planes",
        and for a max plane that sentence is false about a real measurement --
        the number beside it was reduced with ``max``. Two quantities under one
        basis string is the failure ``DELTA_BASIS`` exists to prevent wearing
        yet another shoe, and it would be an easy one to ship, because the
        wrong basis and the right one describe numbers that are both zero."""
        from tta_backend.preprocessing.frame_stack import (
            FRAME_GRID_DELTA_BASIS, PLANE_AGREEMENT_BASIS,
        )

        stack = self._stack(self._unevenly_seen_block())

        self.assertEqual(stack.frame_grid_delta["basis"], FRAME_GRID_DELTA_BASIS)
        self.assertEqual(PLANE_AGREEMENT_BASIS["mean"], FRAME_GRID_DELTA_BASIS)
        bases = {name: stack.planes[name].frame_grid_delta["basis"] for name in ("max", "min")}
        self.assertEqual(len(set(bases.values()) | {FRAME_GRID_DELTA_BASIS}), 3)
        self.assertIn("max of the stored frame planes", bases["max"])
        self.assertIn("min of the stored frame planes", bases["min"])


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class BlockMaxAndMinPlaneTests(unittest.TestCase):
    """D6a decision 4: a max plane's SPATIAL reducer is the block max, not the
    block mean.

    The distinction is invisible on a smooth field and is the whole product on
    a sharp one, so every fixture here is a single hot cell against a cold or
    empty block -- Phase 11's own trap, restated. It is also all at k=(2,2) or
    larger, never k=(1,1), where every block reduction is the identity function
    and a passing test proves only that ``coarsen`` was not called (Phase 9's
    lesson: both of its original identity tests pinned ``coarsen_k == [1, 1]``,
    which is the regime no real regional retrieval is in).
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _field(self, *days, lats=(0.0, 1.0, 2.0, 3.0)):
        """One daily granule per argument over a 4x4 grid, which coarsens to
        2x2 -- big enough that the blocks differ from each other rather than
        the whole region being one block."""
        np = self.np
        return self.xr.DataArray(
            np.stack([np.asarray(day, dtype=float) for day in days]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{d + 1:02d}T12:00" for d in range(len(days))],
                    dtype="datetime64[ns]",
                ),
                "lat": np.asarray(lats, dtype=float),
                "lon": np.array([-100.0, -99.0, -98.0, -97.0]),
            },
            name="no2",
        )

    def _sharp_day(self):
        """Block (0,0) is one hot cell against three cold ones; block (0,1) was
        never observed at all; block (1,0) is flat; block (1,1) has its own
        smaller peak."""
        nan = self.np.nan
        return [
            [100.0, 1.0, nan, nan],
            [1.0, 1.0, nan, nan],
            [2.0, 2.0, 3.0, 9.0],
            [2.0, 2.0, 3.0, 3.0],
        ]

    def _stack(self, field, *, statistics=("mean", "max", "min"), target_cells=4, **kwargs):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            field, time_dim="time", cadence="daily", target_cells=target_cells,
            statistics=statistics, **kwargs,
        )

    def test_the_max_plane_renders_the_peak_the_mean_plane_averages_away(self):
        """The tracer bullet for the reduction itself.

        One native cell held 100 in a block whose other three held 1. The mean
        plane renders 25.75 -- a value no cell ever had, and a quarter of the
        peak. The max plane renders 100.0 exactly, which is the entire reason
        D6a asks for a second plane rather than a second colour ramp.
        """
        stack = self._stack(self._field(self._sharp_day(), [[5.0] * 4] * 4))

        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertEqual(float(stack.planes["max"].values[0][0, 0]), 100.0)
        self.assertAlmostEqual(float(stack.values[0][0, 0]), 25.75, places=4)

    def test_the_min_plane_renders_the_trough_the_mean_plane_averages_away(self):
        """Min is not an afterthought (D6a decision 2): the same block whose
        peak is 100 has a floor of 1, and the mean plane's 25.75 is 25 times
        it. A layout that generalizes to three statistics has to be exercised
        by three statistics, or decision 2 is a claim nothing checks."""
        stack = self._stack(self._field(self._sharp_day(), [[5.0] * 4] * 4))

        self.assertEqual(float(stack.planes["min"].values[0][0, 0]), 1.0)
        self.assertEqual(float(stack.planes["min"].values[0][1, 1]), 3.0)

    def test_every_block_of_the_max_plane_is_a_value_some_native_cell_held(self):
        """The property that separates a selection from a computation, checked
        on the whole frame rather than at the one cell the fixture was built
        around. A block mean invents values none of its cells had; a block max
        cannot, and that is what makes a peak readable off the picture."""
        np = self.np
        day = self._sharp_day()
        stack = self._stack(self._field(day, [[5.0] * 4] * 4))

        frame = stack.planes["max"].values[0]
        self.assertTrue(np.array_equal(
            np.nan_to_num(frame, nan=-1.0),
            np.array([[100.0, -1.0], [2.0, 9.0]], dtype="float32"),
        ))
        # And the mean's own frame is a different array, so the assertion above
        # is not passing because the two planes are the same bytes.
        self.assertNotAlmostEqual(
            float(stack.values[0][1, 1]), float(frame[1, 1]), places=4,
        )

    def test_a_block_nothing_was_observed_in_stays_absent_on_both_planes(self):
        """G3, made permanent. ``coarsen(...).max(skipna=True)`` over an
        all-NaN block returns the reducer's identity element in some array
        libraries -- ``-inf`` for max, ``+inf`` for min -- which would render
        as the hottest and coldest pixel on the chart at exactly the places
        nothing was measured. It does not here, on either plane, and this is
        the test that keeps it that way."""
        np = self.np
        stack = self._stack(self._field(self._sharp_day()))

        for name in ("max", "min"):
            frame = stack.planes[name].values[0]
            self.assertTrue(
                np.isnan(frame[0, 1]), f"{name} plane rendered {frame[0, 1]}",
            )
            self.assertFalse(np.isinf(frame).any(), f"{name} plane leaked an inf")

    def test_the_pad_is_skipped_rather_than_reduced_with_the_block(self):
        """``boundary="pad"`` on a grid k does not divide. The synthetic control
        Phase 11 ran (``[1,2,3,4,5]`` at k=3 -> ``[3.0, 5.0]`` for max and
        ``[1.0, 4.0]`` for min) reproduced through the public interface: a 5-row
        grid at k=(3,3) keeps its short trailing block instead of trimming the
        region edge away, and the NaN pad neither wins the max nor loses the
        min."""
        np = self.np
        rows = [[float(r + 1)] * 5 for r in range(5)]
        field = self.xr.DataArray(
            np.asarray(rows, dtype=float)[None, ...],
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(["2024-01-01T12:00"], dtype="datetime64[ns]"),
                "lat": np.arange(5.0), "lon": np.arange(5.0),
            },
            name="no2",
        )

        stack = self._stack(field, target_cells=3)

        self.assertEqual(stack.coarsen_k, (3, 3))
        self.assertEqual(
            [float(v) for v in stack.planes["max"].values[0][:, 0]], [3.0, 5.0],
        )
        self.assertEqual(
            [float(v) for v in stack.planes["min"].values[0][:, 0]], [1.0, 4.0],
        )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class PlanesAreOptionalTests(unittest.TestCase):
    """D6a decision 5, read at the Python level: "the mean entry keeps its
    exact current shape, URL and cost."

    Everything in this repository until Phase 13 asks for the mean and nothing
    else, so the shape of the win is that those callers cannot tell this phase
    happened. Not "stayed compatible" -- unchanged, field for field and byte
    for byte, including the numbers that would be the easiest to perturb by
    accident (a delta reduced over a dataset that now carries more variables,
    a pooled scale pooled over more histograms).
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _field(self):
        """Two granules per day for two days, so a bucket's mean, max and min
        are three different fields rather than three names for the granule --
        which they are wherever a bucket holds one granule, and which would
        make every assertion below pass for the wrong reason."""
        np = self.np
        hot = np.full((4, 4), 1.0)
        hot[0, 0], hot[3, 3] = 100.0, 5.0
        cold = np.full((4, 4), 1.0)
        cold[3, 3] = -5.0
        flat = np.full((4, 4), 1.0)
        return self.xr.DataArray(
            np.stack([hot, cold, flat, flat]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    ["2024-01-01T00:00", "2024-01-01T12:00",
                     "2024-01-02T00:00", "2024-01-02T12:00"],
                    dtype="datetime64[ns]",
                ),
                "lat": np.array([0.0, 1.0, 2.0, 3.0]),
                "lon": np.array([-100.0, -99.0, -98.0, -97.0]),
            },
            name="no2",
        )

    def _stack(self, **kwargs):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            self._field(), time_dim="time", cadence="daily", target_cells=4,
            **kwargs,
        )

    def test_a_caller_that_asks_for_nothing_new_gets_nothing_new(self):
        """The default is today's behaviour exactly, and ``planes`` is empty
        rather than holding a mean nobody asked for."""
        stack = self._stack()

        self.assertEqual(stack.planes, {})

    def test_the_mean_tiers_own_numbers_are_untouched_by_the_extra_planes(self):
        """The regression this phase is most likely to cause and least likely
        to notice: two more statistics join the fused dataset the mean's delta,
        pooled scale and per-frame statistics are all read out of."""
        np = self.np
        plain = self._stack()
        widened = self._stack(statistics=("mean", "max", "min"))

        self.assertTrue(np.array_equal(
            np.nan_to_num(plain.values, nan=-1.0),
            np.nan_to_num(widened.values, nan=-1.0),
        ))
        self.assertTrue(np.array_equal(plain.period_values, widened.period_values))
        self.assertEqual(plain.frame_grid_delta, widened.frame_grid_delta)
        self.assertEqual(plain.delta, widened.delta)
        self.assertEqual(plain.value_range, widened.value_range)
        self.assertEqual(
            [f.statistics for f in plain.frames],
            [f.statistics for f in widened.frames],
        )

    def test_each_plane_gets_its_own_pooled_scale(self):
        """D9, per plane. One colour has to mean one value at every stop of a
        scrub, and a max plane scrubbed against the MEAN's clip would saturate
        at every stop that has a peak in it -- which is every stop anyone
        switched to the max plane to look at. Pooled at native resolution
        before the block reduction, the same as the mean's (D5a)."""
        stack = self._stack(statistics=("mean", "max", "min"))

        self.assertIsNotNone(stack.value_range)
        self.assertGreater(
            stack.planes["max"].value_range[1], stack.value_range[1],
            "the max plane's clip did not reach above the mean plane's",
        )
        self.assertLess(
            stack.planes["min"].value_range[0], stack.value_range[0],
            "the min plane's clip did not reach below the mean plane's",
        )

    def test_a_statistic_with_no_plane_behind_it_is_refused(self):
        """A typo, or a caller reaching for a statistic this reduction has
        never measured. The alternative is a stack that silently ships one
        fewer plane than it was asked for -- and the caller finding out in the
        frontend, where the toggle is simply missing."""
        with self.assertRaises(ValueError) as raised:
            self._stack(statistics=("mean", "median"))

        self.assertIn("median", str(raised.exception))

    def test_the_plane_statistics_are_not_the_per_frame_scalars(self):
        """The name collision this phase had to walk around. ``_FRAME_STATS``
        is the per-frame REGIONAL SCALAR -- one number for the whole region,
        which every frame has carried since Phase 3 -- and ``PLANE_STATISTICS``
        is a field, one value per pixel. They are still, deliberately, two
        objects: a chart can carry a max plane while every frame's scalar max
        is exactly what it always was, and this asserts both live at once
        rather than one having quietly become the other.

        The two are not even the same NUMBER on this fixture, which is the
        cleanest possible refutation of "max is max". Frame 0's scalar max is
        **50.5** -- the largest value on the frame's MEAN field, where the hot
        cell's 100 and its own 1 twelve hours later have already been averaged.
        The max plane's peak is **100.0**, because it never took that mean. A
        readout that had been quietly re-pointed at the plane would print 100
        where it has always printed 50.5.
        """
        from tta_backend.preprocessing import frame_stack as module

        stack = self._stack(statistics=("mean", "max"))

        self.assertIsNot(module.PLANE_STATISTICS, module._FRAME_STATS)
        # The regional scalar max of frame 0: one number, off the native MEAN
        # field, exactly as every frame has carried since Phase 3.
        self.assertEqual(stack.frames[0].statistics["max"], 50.5)
        # The max PLANE's frame 0: a grid, and its own top-left block.
        self.assertEqual(stack.planes["max"].values[0].shape, (2, 2))
        self.assertEqual(float(stack.planes["max"].values[0][0, 0]), 100.0)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class ExtentOverstatementTests(unittest.TestCase):
    """D6a decision 9, and the max tier's one real cost.

    Block max paints a block's peak across every cell the block covers, so a
    peak one native cell held is rendered at k^2 cells' worth of area. Phase 11
    measured it at **24.7x** pooled on both live bundles at k=(5,5) -- 99% of
    the k^2=25 ceiling, with the per-frame spread under half a percentage
    point. Not a tail risk on a few hot frames: what block max costs on
    essentially every finite block.

    D16's rules apply to it whole: a ratio of weighted SUMS, never a mean of
    per-block ratios, never signed. It gates nothing -- disclosure is the
    posture the mean tier already took with its own deltas, and D5a's rule
    ("nothing is derived from the reduced array without the reader being told")
    is what makes it mandatory rather than optional.

    Max only. The mean plane's equivalent question is a different one entirely
    -- a block mean does not paint any cell's value anywhere -- and a minimum
    has no analogous overstatement story, since nobody reads a trough as an
    extent. Measuring it for them would be inventing a number to be symmetric
    with, which is exactly what D6a decision 8 refuses for the agreement
    figure in the other direction.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _days(self, *days, lats=(0.0, 60.0)):
        """A 2x2 grid whose two rows are deliberately at cos(lat) 1.0 and 0.5,
        so a metric that counted CELLS instead of AREA gets a different answer
        from a metric that weights them -- and the fixture can tell."""
        np = self.np
        return self.xr.DataArray(
            np.stack([np.asarray(day, dtype=float) for day in days]),
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{d + 1:02d}T12:00" for d in range(len(days))],
                    dtype="datetime64[ns]",
                ),
                "lat": np.asarray(lats, dtype=float),
                "lon": np.array([-75.0, -74.0]),
            },
            name="no2",
        )

    def _stack(self, field, *, target_cells=1, **kwargs):
        from tta_backend.preprocessing.frame_stack import build_frame_stack

        return build_frame_stack(
            field, time_dim="time", cadence="daily", target_cells=target_cells,
            statistics=("mean", "max", "min"), **kwargs,
        )

    def test_one_hot_cell_is_painted_across_its_whole_blocks_area(self):
        """The exact fixture number, not "roughly k^2".

        One 2x2 block, one cell at 9 against three at 1, on both days. The
        block covers cos(0)+cos(0)+cos(60)+cos(60) = 1+1+0.5+0.5 = **3.0**
        cells' worth of weighted area, and the area actually holding the
        block's own peak is the single equatorial cell's **1.0**. So the
        rendered peak overstates its extent by exactly **3.0x** -- not the
        4.0x an unweighted k^2 count would report, which is what makes this
        fixture able to fail on a metric that forgot cos(latitude).
        """
        stack = self._stack(self._days(
            [[9.0, 1.0], [1.0, 1.0]],
            [[9.0, 1.0], [1.0, 1.0]],
        ))

        overstatement = stack.planes["max"].extent_overstatement
        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertAlmostEqual(overstatement["headline"], 3.0, places=9)
        self.assertAlmostEqual(overstatement["worst_frame"], 3.0, places=9)
        self.assertEqual(overstatement["ceiling"], 4)

    def test_the_same_peak_at_a_poleward_cell_overstates_twice_as_much(self):
        """Move the hot cell to the cos(60) row and nothing else. The block's
        rendered area is the same 3.0 and the area at peak is now 0.5, so the
        figure is **6.0x**. A cell-counting metric would report 4.0x for both
        this fixture and the one above."""
        stack = self._stack(self._days(
            [[1.0, 1.0], [9.0, 1.0]],
            [[1.0, 1.0], [9.0, 1.0]],
        ))

        self.assertAlmostEqual(
            stack.planes["max"].extent_overstatement["headline"], 6.0, places=9,
        )

    def test_a_tie_at_the_block_max_is_area_that_really_held_the_peak(self):
        """Two cells at 9, both equatorial: 2.0 of the block's 3.0 really is at
        the rendered value, so the overstatement is **1.5x**. The metric counts
        the area at the block's own max, not "one cell" -- which matters
        because at native TEMPO resolution with continuous float64 values a tie
        is rare, and that rarity is precisely WHY Phase 11 measured ~99% of the
        k^2 ceiling rather than something comfortably below it."""
        stack = self._stack(self._days(
            [[9.0, 9.0], [1.0, 1.0]],
            [[9.0, 9.0], [1.0, 1.0]],
        ))

        self.assertAlmostEqual(
            stack.planes["max"].extent_overstatement["headline"], 1.5, places=9,
        )

    def test_a_block_paints_the_ground_it_covers_not_the_ground_it_observed(self):
        """The definitional choice, pinned. Three of the block's four cells were
        never observed, and the rendered pixel still covers all four cells'
        worth of ground at the one value that was -- so the figure is the same
        **3.0x**. Denominating on observed area instead would make a sparse
        frame look BETTER than a dense one, when the sparse frame is exactly
        where a reader is most likely to mistake one cell's plume for a
        region-wide one."""
        nan = self.np.nan
        stack = self._stack(self._days(
            [[9.0, nan], [nan, nan]],
            [[9.0, nan], [nan, nan]],
        ))

        self.assertAlmostEqual(
            stack.planes["max"].extent_overstatement["headline"], 3.0, places=9,
        )

    def test_the_pooled_figure_is_a_ratio_of_sums_not_a_mean_of_ratios(self):
        """D16's rule, and the fixture is built so the two answers differ.

        Day one's peak is the equatorial cell (3.0x); day two's is the poleward
        one (6.0x). Pooled over both frames the answer is
        (3.0 + 3.0) / (1.0 + 0.5) = **4.0x**, where averaging the two per-frame
        ratios would give 4.5x. The worst frame is disclosed BESIDE the pooled
        figure rather than folded into it, the same way ``delta`` carries a
        headline and a worst pixel.
        """
        stack = self._stack(self._days(
            [[9.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [9.0, 1.0]],
        ))

        overstatement = stack.planes["max"].extent_overstatement
        self.assertAlmostEqual(overstatement["headline"], 4.0, places=9)
        self.assertNotAlmostEqual(overstatement["headline"], 4.5, places=6)
        self.assertAlmostEqual(overstatement["worst_frame"], 6.0, places=9)

    def test_nothing_is_overstated_where_no_block_reduction_ran(self):
        """At k=(1,1) every rendered pixel IS a native cell, so there is no
        extent to overstate and the honest answer is absence rather than a
        1.0x that reads like a measurement. The same reason ``delta`` is
        ``None`` in the cadence tier."""
        stack = self._stack(
            self._days([[9.0, 1.0], [1.0, 1.0]], [[9.0, 1.0], [1.0, 1.0]]),
            target_cells=10_000,
        )

        self.assertEqual(stack.coarsen_k, (1, 1))
        self.assertIsNone(stack.planes["max"].extent_overstatement)

    def test_only_the_max_plane_carries_it(self):
        """Decision 9 is about the max tier. A min plane reporting the same
        ratio would be answering a question about troughs that nobody has
        asked and nobody has measured."""
        stack = self._stack(self._days(
            [[9.0, 1.0], [1.0, 1.0]],
            [[9.0, 1.0], [1.0, 1.0]],
        ))

        self.assertIsNone(stack.planes["min"].extent_overstatement)
        self.assertIsNotNone(stack.planes["max"].extent_overstatement)

    def test_it_carries_its_own_account_of_itself(self):
        """A third quantity beside ``delta`` and ``frame_grid_delta``, and it
        measures something neither of them does -- area painted versus area
        observed, not value agreement. Three quantities, three basis strings,
        for the reason ``DELTA_BASIS`` was written down in the first place."""
        from tta_backend.preprocessing.frame_stack import (
            DELTA_BASIS, EXTENT_OVERSTATEMENT_BASIS, FRAME_GRID_DELTA_BASIS,
        )

        stack = self._stack(self._days(
            [[9.0, 1.0], [1.0, 1.0]],
            [[9.0, 1.0], [1.0, 1.0]],
        ))

        self.assertEqual(
            stack.planes["max"].extent_overstatement["basis"],
            EXTENT_OVERSTATEMENT_BASIS,
        )
        self.assertEqual(
            len({DELTA_BASIS, FRAME_GRID_DELTA_BASIS, EXTENT_OVERSTATEMENT_BASIS}), 3,
        )
        self.assertIn("area", EXTENT_OVERSTATEMENT_BASIS)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class PooledColourScaleTests(unittest.TestCase):
    """D9: one 2-98 clip pooled across every frame including frame 0, for the
    scrubber only.

    Per-frame clipping would make a colour mean a different value at every
    stop, which is the one thing an animation must not do. Reusing the
    aggregate's range saturates the peak frames, which are the point. And the
    frontend's existing ``computeSharedColorScale`` is explicitly NOT reusable
    here: it takes ``min(vmins)/max(vmaxs)``, a union of per-panel clips, and
    over 60 frames that union is whatever the noisiest bucket did.
    """

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np, self.xr = np, xr

    def _days(self, frames):
        np = self.np
        n = len(frames)
        return _field(
            self.xr, np,
            [f"2024-01-{d + 1:02d}T12:00" for d in range(n)],
            frames,
            lats=np.linspace(30.0, 34.0, frames[0].shape[0]),
            lons=np.linspace(-100.0, -96.0, frames[0].shape[1]),
        )

    def test_the_scale_is_pooled_not_a_union_of_per_frame_clips(self):
        """Nine calm days and one with a 3% scatter of outliers. That tenth
        frame's OWN 98th percentile is 5000, so a union of per-panel clips
        hands the scrubber a 5000-wide colour ramp on which every calm day is
        a single flat colour -- the noisiest bucket deciding what the other
        nine look like. Pooled, those outliers are half a percent of the
        distribution and the ramp stays where the data is."""
        np = self.np
        calm = np.linspace(10.0, 11.0, 400).reshape(20, 20)
        loud = calm.copy()
        loud.ravel()[:12] = 5000.0
        stack_of = self._days([calm] * 9 + [loud])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(stack_of, time_dim="time", cadence="daily")
        vmin, vmax = stack.value_range

        self.assertLess(vmax, 12.0)
        self.assertGreater(vmin, 9.9)
        # The loud frame's own p98, i.e. what a union of clips would have used.
        self.assertAlmostEqual(float(np.percentile(loud, 98)), 5000.0, places=3)

    def test_the_scale_is_pooled_at_native_resolution(self):
        """D5a again. 5% of cells carry the peak, so natively the 98th
        percentile IS the peak; block-meaned each of those cells is averaged
        against its eight neighbours and the same percentile reads 12. A
        scrubber scaled that way saturates every plume it was built to show."""
        np = self.np
        values = np.ones((30, 30))
        # One hot cell per 3x3 block, in 45 of the 100 blocks -- 5% of cells.
        for index in range(45):
            block_row, block_col = divmod(index, 10)
            values[block_row * 3 + 1, block_col * 3 + 1] = 100.0
        field = self._days([values, values.copy()])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily", target_cells=100,
        )

        self.assertEqual(stack.coarsen_k, (3, 3))
        self.assertGreater(stack.value_range[1], 90.0)
        # What the frame grid would have said: (100 + 8) / 9.
        self.assertAlmostEqual(float(np.nanpercentile(stack.values, 98)), 12.0, places=3)

    def test_frame_zero_is_in_the_pool(self):
        """Five cells out of a hundred read zero on day one and are covered on
        day two, so the period map reads 50 there. Over the two frames alone
        those zeros are 2.5% of the distribution and the 2nd percentile is 0.
        With frame 0 pooled in they are 1.67%, and the clip moves up to the 50
        the map actually shows -- so the map's own low values are not rendered
        at the bottom of a ramp built without them."""
        np = self.np
        first = np.full((10, 10), 100.0)
        first.ravel()[:5] = 0.0
        field = self._days([first, np.full((10, 10), 100.0)])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertAlmostEqual(stack.value_range[0], 50.0, places=1)
        self.assertNotAlmostEqual(stack.value_range[0], 0.0, places=1)

    def test_a_stack_with_nothing_in_it_has_no_scale_to_offer(self):
        """No values, no percentiles. A fabricated (0, 1) range would render an
        empty week as a uniform field of the colormap's lowest colour."""
        np = self.np
        field = self._days([np.full((4, 4), np.nan), np.full((4, 4), np.nan)])

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        stack = build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertIsNone(stack.value_range)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class TheMaskingSeamSuppliesTheAreaCountersTests(unittest.TestCase):
    """Where the roll-up's numerators and denominators come from.

    ``_count_qa_pixels`` already reduces ``passing_area`` and ``checked_area``
    per timestep -- it has to, to publish ``pass_rate_by_time`` -- but it only
    hands back the quotients. A bucket roll-up cannot be reconstructed from
    quotients, and re-deriving the areas would mean a second full walk of a
    lazily-opened bundle for numbers already paid for. So the counters travel
    alongside the rates.
    """

    def _counts(self):
        import numpy as np
        import xarray as xr

        from tta_backend.preprocessing.aggregation_service import AggregationService

        # cos(0) = 1 against cos(60) = 0.5, so a weighted counter and an
        # unweighted one cannot be confused. Each timestep passes QA at one
        # latitude and fails at the other.
        ds = xr.Dataset(
            {
                "no2": (("time", "lat", "lon"), np.array([[[1.0], [2.0]], [[3.0], [4.0]]])),
                "main_data_quality_flag": (
                    ("time", "lat", "lon"),
                    np.array([[[0], [1]], [[1], [0]]], dtype="int64"),
                ),
            },
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                "lat": [0.0, 60.0],
                "lon": [30.0],
            },
        )
        masked = AggregationService()._resolve_and_mask(
            ds["no2"], variable="no2",
            col_info={"quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]},
            source_ds=ds,
        )
        return masked.counts

    def test_the_per_timestep_areas_travel_with_the_rates(self):
        """cos(latitude)-weighted, the same weights the cumulative rate divides.
        Two pixels checked at each timestep, worth 1.0 and 0.5 of area."""
        counts = self._counts()

        for got, want in zip(counts["checked_area_by_time"], [1.5, 1.5]):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(counts["passing_area_by_time"], [1.0, 0.5]):
            self.assertAlmostEqual(got, want, places=12)

    def test_the_value_extremes_ride_the_same_walk(self):
        """The pooled scale's bin edges, bought on a graph walk that was
        happening anyway. A bucket mean is an average of these values, so this
        range brackets every frame and the period map -- which is all the
        histogram needs, and it is the difference between one I/O pass over a
        multi-granule bundle and two.

        Measured on the fill/valid-masked field, BEFORE the QA mask, so it
        stays a superset of what survives rather than a range that could clip
        it."""
        counts = self._counts()

        self.assertAlmostEqual(counts["value_min"], 1.0, places=12)
        self.assertAlmostEqual(counts["value_max"], 4.0, places=12)

    def test_the_published_rates_are_those_areas_divided(self):
        """The counters and the rates describe the same measurement, so a
        roll-up built from the areas cannot disagree with a per-timestep rate
        the UI is already showing."""
        counts = self._counts()

        for rate, passing, checked in zip(
            counts["pass_rate_by_time"],
            counts["passing_area_by_time"],
            counts["checked_area_by_time"],
        ):
            self.assertAlmostEqual(rate, passing / checked, places=6)


@unittest.skipIf(importlib.util.find_spec("dask") is None, "dask is not installed")
class OneGraphWalkTests(unittest.TestCase):
    """The property that keeps the memory and I/O budgets honest.

    ``groupby(bucket).mean().coarsen(k, boundary="pad").mean()`` composes into
    ONE lazy graph -- verified on a 36-granule full-domain TEMPO open, where it
    stays lazy at 32,042 tasks and materializes (32, 99, 193) = 4.9 MB against
    the 2,179 MB the grouped-then-coarsened intermediate would need. Every
    per-frame number rides that same walk.

    Separate ``compute`` calls share no graph, so each is a fresh I/O pass over
    a lazily-opened bundle. A pass appearing here means someone added a compute
    that could have ridden the existing walk -- the regression T55 spent its
    counter fusion removing.
    """

    def _lazy_days(self, granules, loads):
        """A lazily-opened multi-granule bundle: one dask chunk per granule,
        each backed by a loader that records its read."""
        import dask
        import dask.array as dask_array
        import numpy as np
        import xarray as xr

        def _chunk(index, values):
            block = np.asarray(values, dtype="float64")[None, ...]

            def _load():
                loads.append(index)
                return block

            return dask_array.from_delayed(
                dask.delayed(_load)(), shape=block.shape, dtype="float64",
            )

        science = dask_array.concatenate(
            [_chunk(i, g) for i, g in enumerate(granules)], axis=0,
        )
        return xr.DataArray(
            science,
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(
                    [f"2024-01-{d + 1:02d}T12:00" for d in range(len(granules))],
                    dtype="datetime64[ns]",
                ),
                "lat": [10.0, 20.0], "lon": [30.0, 40.0],
            },
            name="no2",
        )

    def test_the_whole_stack_costs_one_pass_over_the_bundle(self):
        """Frames, per-frame statistics, coverage and the pooled scale all come
        off a single walk. Given the range the masking pass already measured,
        there is nothing left that needs its own."""
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        loads = []
        field = self._lazy_days([np.full((2, 2), float(d)) for d in range(3)], loads)

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily", value_bracket=(0.0, 2.0),
        )

        self.assertEqual(len(stack.frames), 3)
        self.assertEqual(
            sorted(loads), [0, 1, 2],
            f"bundle read {len(loads)} times across 3 granules, want 3 (1 pass): {loads}",
        )

    def test_the_shipped_array_delta_costs_no_extra_pass(self):
        """Phase 9's quantity is free, and this is what "free" has to mean.

        Both arrays it compares are materialized side by side by the walk that
        was happening anyway, so it is arithmetic on bytes already in hand --
        one pass over three granules, exactly as without it. Written as a read
        counter rather than as a comment because the tempting refactor
        (expressing it as lazy terms beside ``_delta_terms``, for symmetry)
        would put a second reduction on the graph, and nothing else here would
        notice.
        """
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        loads = []
        field = self._lazy_days([np.full((2, 2), float(d)) for d in range(3)], loads)

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily", value_bracket=(0.0, 2.0),
            target_cells=1,
        )

        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertIsNotNone(stack.frame_grid_delta["headline"])
        self.assertEqual(
            sorted(loads), [0, 1, 2],
            f"bundle read {len(loads)} times across 3 granules, want 3 (1 pass): {loads}",
        )

    def test_a_second_and_third_plane_cost_no_extra_pass(self):
        """Phase 11's G1, ported from the probe into a permanent test.

        The gate already answered this on real data (an exact 6-read parity on
        synthetic loads, 1.5-1.9x wall-clock on two live bundles, never 3x);
        what a test adds is that it STAYS answered. The whole plane design
        rests on it: D6a decision 6 keeps the build eager precisely because
        three statistics ride one graph walk, and the tempting refactor --
        computing each plane in its own ``.compute()``, which reads far more
        naturally -- would be a fresh I/O pass per statistic over a lazily
        opened bundle, and nothing else here would notice.
        """
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        def _reads(statistics):
            loads = []
            field = self._lazy_days(
                [np.full((2, 2), float(d)) for d in range(3)], loads,
            )
            stack = build_frame_stack(
                field, time_dim="time", cadence="daily", value_bracket=(0.0, 2.0),
                target_cells=1, statistics=statistics,
            )
            return stack, sorted(loads)

        _, mean_only = _reads(("mean",))
        stack, three_planes = _reads(("mean", "max", "min"))

        self.assertEqual(stack.coarsen_k, (2, 2))
        self.assertEqual(set(stack.planes), {"max", "min"})
        self.assertEqual(mean_only, [0, 1, 2])
        self.assertEqual(
            three_planes, mean_only,
            f"three planes read the bundle {three_planes}, one plane {mean_only}: "
            "a statistic bought itself its own pass",
        )

    def test_without_a_measured_range_the_pooled_scale_costs_a_second_pass(self):
        """A quantile is not a streaming reduction, so its bins cannot be
        chosen until the extremes are known -- a real data dependency, not an
        oversight. The masking pass already measures those extremes, so the
        real plot path pays this once and not twice; this is what it costs when
        nothing hands them over."""
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        loads = []
        field = self._lazy_days([np.full((2, 2), float(d)) for d in range(3)], loads)

        build_frame_stack(field, time_dim="time", cadence="daily")

        self.assertEqual(sorted(loads), [0, 0, 1, 1, 2, 2])

    def test_the_masking_counters_are_enough_on_their_own(self):
        """The real plot path hands over ``MaskedField.counts`` and nothing
        else. If the range had to be passed separately, forgetting it would
        double the I/O of every scrubbed plot and nothing would say so --
        exactly the kind of silent regression T55's counter fusion existed to
        remove."""
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        loads = []
        field = self._lazy_days([np.full((2, 2), float(d)) for d in range(3)], loads)

        build_frame_stack(
            field, time_dim="time", cadence="daily",
            qa_counts={"value_min": 0.0, "value_max": 2.0},
        )

        self.assertEqual(sorted(loads), [0, 1, 2])

    def test_a_bracket_that_is_too_narrow_saturates_rather_than_losing_cells(self):
        """A caller handing over a range from a different reduction could get
        it wrong. Saturating into the outer bins is the only failure this may
        have: values dropping out of the pool entirely would make the clip
        describe a subset of the data while claiming to describe all of it."""
        import numpy as np

        from tta_backend.preprocessing.frame_stack import build_frame_stack

        loads = []
        field = self._lazy_days(
            [np.full((2, 2), 0.0), np.full((2, 2), 500.0)], loads,
        )

        stack = build_frame_stack(
            field, time_dim="time", cadence="daily", value_bracket=(0.0, 10.0),
        )

        # Every cell still in the pool: 8 frame cells + 4 period cells, and the
        # 500s pinned at the top edge rather than vanishing.
        self.assertLessEqual(stack.value_range[1], 10.0)
        self.assertGreater(stack.value_range[1], 9.0)


@unittest.skipIf(
    importlib.util.find_spec("rasterio") is None, "rasterio is not installed",
)
class RegionAreaIsRecordedAtTheMaskTests(unittest.TestCase):
    """Where ``build_frame_stack``'s denominator comes from.

    ``mask_data_by_geometry`` already records ``region_cells``, because the
    rasterized mask exists nowhere else and a caller re-rasterizing on cropped
    axes gets a float32 step that differs in the 8th digit and flips boundary
    cells. But a cell COUNT cannot denominate an area-weighted numerator: the
    two would describe different fields, which is the mismatch Finding #13
    already caught once. So the seam records the region's cos(latitude)-weighted
    area alongside its cell count.
    """

    def _masked(self, geometry):
        import numpy as np
        import xarray as xr

        from tta_backend.utils.plotting import mask_data_by_geometry

        band = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [-100.0, -99.0]},
        )
        return mask_data_by_geometry(band, geometry, crop=False)

    def test_a_poleward_cell_contributes_less_region_than_an_equatorial_one(self):
        """Two regions of one cell each. They are the same COUNT and not the
        same AREA -- cos(60) is half of cos(0) -- and a coverage figure
        denominated on the count would call the poleward region twice as
        observed as it is."""
        from shapely.geometry import box

        equatorial = self._masked(box(-100.4, -0.4, -99.6, 0.4))
        poleward = self._masked(box(-100.4, 59.6, -99.6, 60.4))

        self.assertEqual(equatorial.attrs["region_cells"], 1)
        self.assertEqual(poleward.attrs["region_cells"], 1)
        self.assertAlmostEqual(equatorial.attrs["region_area"], 1.0, places=9)
        self.assertAlmostEqual(poleward.attrs["region_area"], 0.5, places=9)
