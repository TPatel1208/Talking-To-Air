"""The frame gate: whether a request may be framed at all.

The reduction (``test_bucketed_frame_stack.py``) and the blob store
(``test_frame_store.py``) are tested elsewhere.

The frame-count limit (2..60) bounds no spatial extent, and a full-domain
TEMPO build is OOM-killed in the 3.9 GB container, so the gate also limits
extent.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
import unittest.mock

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _field(xr, np, times, *, ny=4, nx=4, name="no2"):
    """A (time, lat, lon) field the gate can be asked about."""
    return xr.DataArray(
        np.ones((len(times), ny, nx)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": np.linspace(30.0, 45.0, ny),
            "lon": np.linspace(-100.0, -85.0, nx),
        },
        name=name,
    )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class FrameGateTests(unittest.TestCase):
    def test_an_unbucketable_cadence_is_refused_by_name(self):
        """A Harmony bundle member carries no ``short_name``, so an
        unregistered product resolves to "unknown" -- the same condition under
        which ``_cadence_weighted_mean`` degrades to a plain unweighted mean.
        ``build_frame_stack`` RAISES on it, and a raise reaching the tool is a
        crashed plot. The gate has to turn it into a refusal, and inventing a
        bucket width instead would label a scrubber with intervals the product
        never published."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2024-01-01T01:00"])

        refusal = frame_gate(field, time_dim="time", cadence="unknown")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "cadence_unknown")
        self.assertIn("unknown", refusal.detail)

    def test_granules_inside_one_interval_are_not_a_scrub(self):
        """The lower bound. Three granules in the same hour are one stop on
        an hourly axis, and a one-stop scrubber is a map with a slider glued to
        it. The count that matters is INTERVALS, not granules -- gating on
        ``n_granules >= 2`` would build a single-frame stack here and pay a
        full extra pass over the bundle for it."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = _field(
            xr, np,
            ["2024-01-01T09:05", "2024-01-01T09:20", "2024-01-01T09:50"],
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "single_interval")

    def test_a_field_with_no_time_axis_is_not_a_scrub(self):
        """A single-granule map has no axis to browse at all -- a different
        fact from three granules inside one hour, and reached before any
        bucketing happens."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((4, 4)),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(30.0, 45.0, 4), "lon": np.linspace(-100.0, -85.0, 4)},
            name="no2",
        )

        refusal = frame_gate(field, time_dim=None, cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "no_time_axis")

    def test_a_time_dimension_with_no_coordinate_is_refused_not_crashed(self):
        """A frame axis is built from TIMESTAMPS, and a dimension without a
        coordinate has none. Real granules do this -- TEMPO O3PROF's ``layer``
        carries no coordinate at all -- and ``_apply_quality_mask`` already
        guards its own per-timestep counters the same way (``time_dim in
        checked.coords``). Reaching the span lookup instead is a ``KeyError``
        out of a gate whose entire job is to not let that happen."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=("time", "lat", "lon"),
            coords={"lat": np.linspace(30.0, 45.0, 4), "lon": np.linspace(-100.0, -85.0, 4)},
            name="no2",
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "no_time_axis")

    def test_two_intervals_pass(self):
        """The smallest thing worth scrubbing, so the gate's job is to let it
        through. Written so a later limit cannot tighten into the ordinary
        case unnoticed."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"])

        self.assertIsNone(frame_gate(field, time_dim="time", cadence="hourly"))

    def test_a_field_with_a_vertical_axis_left_on_it_is_refused(self):
        """The blob is ``(1 + N, ny, nx)`` and the payload's ``shape`` says so.
        A field still carrying a vertical axis would reduce to a 4-D stack that
        reshapes to something plausible and renders as the wrong layer."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((2, 3, 4, 4)),
            dims=("time", "layer", "lat", "lon"),
            coords={
                "time": np.array(["2024-01-01T09:30", "2024-01-01T10:30"], dtype="datetime64[ns]"),
                "layer": [1000.0, 850.0, 500.0],
                "lat": np.linspace(30.0, 45.0, 4),
                "lon": np.linspace(-100.0, -85.0, 4),
            },
            name="o3",
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "extra_dimensions")

    def test_an_extent_above_the_measured_ceiling_is_refused(self):
        """The reason this gate exists at all.

        A full-domain TEMPO build (17M native cells, 54 hourly buckets) is
        OOM-killed in the 3.9 GB container, and the frame-count limit bounds no
        spatial extent. Refusing costs the full-domain scrub, which is the
        deliberate, disclosed trade."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import MAX_FRAME_NATIVE_CELLS, frame_gate

        # The full TEMPO domain's shape, without its bytes: the gate reads the
        # sizes, never the values.
        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        oversized = field.isel(lat=[0, 1] * 1475, lon=[0, 1] * 2886)
        self.assertGreater(oversized.sizes["lat"] * oversized.sizes["lon"],
                           MAX_FRAME_NATIVE_CELLS)

        refusal = frame_gate(oversized, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "extent_too_large")
        self.assertIn(f"{MAX_FRAME_NATIVE_CELLS:,}", refusal.detail)

    def test_the_extent_ceiling_admits_the_largest_extent_measured_to_survive(self):
        """The limit is anchored on measurement, not on caution. A CONUS crop
        (1250x3000 = 3.75M native cells, the same 54 hourly buckets) COMPLETED
        at 1,342 MB in the same container the full domain died in, so a gate
        that refused it would be throwing away a scrub that demonstrably
        works."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        conus = field.isel(lat=[0, 1] * 625, lon=[0, 1] * 1500)
        self.assertEqual(conus.sizes["lat"] * conus.sizes["lon"], 1250 * 3000)

        self.assertIsNone(frame_gate(conus, time_dim="time", cadence="hourly"))

    def test_an_extent_that_frames_may_still_not_carry_extra_planes(self):
        """The extra-planes limit is smaller than the frame limit.

        ``MAX_FRAME_NATIVE_CELLS`` was measured on a one-statistic build. A
        three-statistic build OOMs at that extent (and at 1.4M and 1.8M cells);
        the largest measured survivor is 1,050,000 cells.

        So the two limits are genuinely different numbers and this is the
        smaller one. A chart between them is not refused: it keeps the mean
        scrubber it has today, because refusing it would charge an existing
        capability for a new one."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import (
            MAX_FRAME_NATIVE_CELLS,
            MAX_PLANE_NATIVE_CELLS,
            frame_gate,
            plane_gate,
        )

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        # Between the two ceilings: framable, and measured not to survive three
        # statistics at anything like this size.
        between = field.isel(lat=[0, 1] * 625, lon=[0, 1] * 1500)
        cells = between.sizes["lat"] * between.sizes["lon"]
        self.assertGreater(cells, MAX_PLANE_NATIVE_CELLS)
        self.assertLessEqual(cells, MAX_FRAME_NATIVE_CELLS)

        # The scrubber this chart has today is untouched...
        self.assertIsNone(frame_gate(between, time_dim="time", cadence="hourly"))
        # ...and the extra planes are what it may not have.
        refusal = plane_gate(between, time_dim="time")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "plane_extent_too_large")
        self.assertIn(f"{MAX_PLANE_NATIVE_CELLS:,}", refusal.detail)

    def test_an_ordinary_regional_extent_may_carry_every_plane(self):
        """The regime real regional charts are in.

        Real regional TEMPO retrievals are ~352k native cells, well under the
        extra-planes limit. A gate that refused these would withhold the
        planes from every chart anyone actually plots."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import plane_gate

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        regional = field.isel(lat=[0, 1] * 267, lon=[0, 1] * 329)
        self.assertEqual(regional.sizes["lat"] * regional.sizes["lon"], 534 * 658)

        self.assertIsNone(plane_gate(regional, time_dim="time"))

    def test_a_span_beyond_the_backstop_is_refused_rather_than_coarsened(self):
        """The span backstop above coarsening.

        Coarsening handles a span past 60 frames and is the common case (2.2
        days of TEMPO is already 54 hourly frames). But coarsening has no ceiling of its own: it will happily fold
        a year of hourly buckets into 60 frames of six days each and label them
        as stops on a scrub. The backstop is where a frame stops being an
        interval anyone asked for."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import MAX_SPAN_BUCKETS, frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2025-01-01T00:00"])

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "span_too_long")
        # The reader is told the shape of the limit, not just that they hit it.
        self.assertIn(f"{MAX_SPAN_BUCKETS:,}", refusal.detail)

    def test_the_span_backstop_sits_above_the_coarsened_tier(self):
        """A month of hourly granules (720 buckets) must still COARSEN to 60
        frames rather than be refused, or the scrubber would cap at ~2.5 days."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2024-01-31T00:00"])

        self.assertIsNone(frame_gate(field, time_dim="time", cadence="hourly"))

