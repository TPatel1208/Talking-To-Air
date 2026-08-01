"""T50: crop to the AOI before masking, without moving a single number.

``mask_data_by_geometry`` used to apply a full-grid ``.where``, so "mean NO2
over New Jersey" against a continental grid reduced over an array that is
>99% NaN by construction. It now narrows the data -- and the already-
rasterized mask -- to the mask's own footprint before the ``.where``.

Rasterization deliberately stays on the full grid: it costs ~10% of the call,
reads no data, and keeping it there makes the kept cells identical to the
uncropped path's *by construction*. Re-rasterizing from cropped axes does not:
it derives a grid step differing in the 8th digit on the float32 axes real
granules ship, which flips cell centers lying on the geometry's boundary.

The governing assertion here is numerical equivalence, not speed: every value
the cropped path keeps must be exactly the value the uncropped path kept.
"""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from shapely.geometry import box  # noqa: E402

from tta_backend.utils.plotting import mask_data_by_geometry  # noqa: E402


def _grid(lats, lons, name="no2"):
    """A deterministic field on the given axes."""
    values = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={
            "lat": ("lat", np.asarray(lats, dtype=float), {"units": "degrees_north"}),
            "lon": ("lon", np.asarray(lons, dtype=float), {"units": "degrees_east"}),
        },
        name=name,
    )


def _continental_grid():
    """A 0.5-degree grid over most of North America -- the shape a small-AOI
    question actually pays for today."""
    return _grid(np.arange(20.0, 55.01, 0.5), np.arange(-125.0, -60.0 + 0.01, 0.5))


def _finite(da):
    return np.sort(da.values[np.isfinite(da.values)])


# New-Jersey-ish: under two degrees square, far smaller than the grid it lives
# on. Deliberately off the half-degree cell centers -- an AOI edge landing
# exactly on a center is a rasterization tie whose resolution depends on axis
# orientation, which would make the N->S/S->N comparison below test rasterio's
# tie-breaking rather than the crop.
SMALL_AOI = box(-74.9, 39.1, -73.1, 40.9)


class AoiCropEquivalenceTests(unittest.TestCase):
    def test_cropped_mask_keeps_exactly_the_values_the_full_grid_mask_kept(self):
        da = _continental_grid()

        cropped = mask_data_by_geometry(da, SMALL_AOI)
        uncropped = mask_data_by_geometry(da, SMALL_AOI, crop=False)

        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        # ...and it actually stopped reducing over the discarded 99%.
        self.assertLess(cropped.size, uncropped.size / 10)

    def test_subcell_region_still_recovers_the_same_boundary_cells(self):
        """T42: a region covering no cell *center* is rescued with
        ``all_touched``, which keeps cells the geometry merely grazes -- and
        such a cell's center can sit outside the geometry's own bbox. Cropping
        to the *mask's* footprint rather than the geometry's bbox is what
        keeps those cells, so "recovered" never turns back into "no data"."""
        from shapely.geometry import LineString

        da = _grid(np.arange(0.0, 10.01, 1.0), np.arange(0.0, 10.01, 1.0))
        # A thin diagonal sliver offset off the y=x diagonal: it contains no
        # cell center, and it grazes cells at both ends whose centers lie
        # outside its bounding box.
        sliver = LineString([(0.6, 1.0), (4.7, 5.1)]).buffer(0.05)

        cropped = mask_data_by_geometry(da, sliver)
        uncropped = mask_data_by_geometry(da, sliver, crop=False)

        self.assertEqual(cropped.attrs.get("region_type"), "boundary_cells")
        self.assertEqual(uncropped.attrs.get("region_type"), "boundary_cells")
        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        self.assertGreater(len(_finite(cropped)), 0)

    def test_descending_latitude_grid_crops_to_the_same_cells(self):
        """Latitude stored N->S is the order half the L3 products publish;
        the crop must narrow such a grid to the same cells (and actually
        narrow it, not quietly fall back to the full grid)."""
        ascending = _continental_grid()
        descending = ascending.isel(lat=slice(None, None, -1))

        cropped = mask_data_by_geometry(descending, SMALL_AOI)
        uncropped = mask_data_by_geometry(descending, SMALL_AOI, crop=False)

        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        np.testing.assert_array_equal(
            _finite(cropped), _finite(mask_data_by_geometry(ascending, SMALL_AOI))
        )
        self.assertLess(cropped.size, uncropped.size / 10)

    def test_normalized_0_360_grid_crops_to_the_same_western_cells(self):
        """The stat-tools longitude regression (2026-07-16), re-asserted
        through the crop: a western-hemisphere AOI against a grid published on
        0..360 must still answer, not crop itself into an empty selection."""
        from tta_backend.tools.satellite_tools.plot_tools import _normalize_longitudes

        shifted = _grid(np.arange(20.0, 55.01, 0.5), np.arange(235.0, 300.01, 0.5))
        normalized = _normalize_longitudes(shifted, "lon")

        cropped = mask_data_by_geometry(normalized, SMALL_AOI)
        uncropped = mask_data_by_geometry(normalized, SMALL_AOI, crop=False)

        self.assertGreater(len(_finite(cropped)), 0)
        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        self.assertLess(cropped.size, uncropped.size / 10)

    def test_unnormalized_0_360_grid_is_left_exactly_as_today(self):
        """Convention resolved before the crop, never by it: handed a grid
        still on 0..360, the mask is empty and the crop stands down on it --
        today's honest no-data answer, not an invented overlap."""
        shifted = _grid(np.arange(20.0, 55.01, 0.5), np.arange(235.0, 300.01, 0.5))

        cropped = mask_data_by_geometry(shifted, SMALL_AOI)
        uncropped = mask_data_by_geometry(shifted, SMALL_AOI, crop=False)

        xr.testing.assert_identical(cropped, uncropped)

    def test_antimeridian_geometry_keeps_both_lobes(self):
        """A geometry crossing the dateline has a bbox spanning the whole
        longitude axis, so the crop must not narrow it and drop a lobe."""
        from shapely.geometry import MultiPolygon

        da = _grid(np.arange(-10.0, 10.01, 1.0), np.arange(-180.0, 179.01, 1.0))
        fiji_ish = MultiPolygon([box(176.0, -3.0, 179.0, 3.0), box(-179.0, -3.0, -176.0, 3.0)])

        cropped = mask_data_by_geometry(da, fiji_ish)
        uncropped = mask_data_by_geometry(da, fiji_ish, crop=False)

        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        # Both lobes survived: cells on either side of the dateline.
        kept_lons = cropped.lon.values[np.isfinite(cropped.values).any(axis=0)]
        self.assertTrue((kept_lons > 175).any() and (kept_lons < -175).any())

    def test_a_geometry_edge_landing_on_a_cell_center_masks_identically(self):
        """The knife-edge that decides whether this optimization is safe.

        ``geometry_mask`` derives its affine from the axis endpoints, so on
        float32 coordinates (what real granules ship) a crop derives a step
        that differs from the full grid's in the 8th digit. Any cell center
        sitting within that wobble of the geometry's boundary then flips in or
        out -- and against the real TEMPO NJ granule that moved a Newark mean
        by 2%. The cropped mask must be the full-grid mask restricted to the
        crop, ties included.
        """
        lons = np.arange(-75.57, -73.88, 0.02).astype(np.float32)
        lats = np.arange(38.79, 41.36, 0.02).astype(np.float32)
        da = xr.DataArray(
            np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons)),
            dims=("lat", "lon"),
            coords={  # float32, exactly as the granule stores them
                "lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"}),
            },
        )
        # -74.25 is exactly a cell center on this grid: a rasterization tie.
        self.assertIn(np.float32(-74.25), lons)
        newark = box(-74.25, 40.68, -74.10, 40.80)

        cropped = mask_data_by_geometry(da, newark)
        uncropped = mask_data_by_geometry(da, newark, crop=False)

        def kept(masked):
            rows, cols = np.nonzero(np.isfinite(masked.values))
            return sorted(zip(masked.lat.values[rows].tolist(), masked.lon.values[cols].tolist()))

        self.assertEqual(kept(cropped), kept(uncropped))

    def test_lat_lon_coords_that_are_not_dimensions_change_no_outcome(self):
        """The crop may never invent a new failure mode. Handed a grid whose
        lat/lon are 1-D coordinates on differently-named dims, it stands down
        and the seam answers exactly what it answered before."""
        lats, lons = np.arange(20.0, 55.01, 0.5), np.arange(-125.0, -60.0 + 0.01, 0.5)
        da = xr.DataArray(
            np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons)),
            dims=("y", "x"),
            coords={"lat": ("y", lats), "lon": ("x", lons)},
        )

        def outcome(crop):
            try:
                return ("returned", mask_data_by_geometry(da, SMALL_AOI, crop=crop).shape)
            except Exception as exc:  # noqa: BLE001 - the outcome IS the assertion
                return ("raised", type(exc).__name__)

        self.assertEqual(outcome(True), outcome(False))

    def test_irregular_grid_still_refuses_instead_of_cropping_into_a_lie(self):
        """T44: the affine mask math assumes one cell size. A grid whose
        spacing changes only *outside* the AOI would look uniform once
        cropped -- the crop must not launder an unsupported grid into a
        silently mis-placed mask."""
        from tta_backend.earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError

        lons = np.concatenate([np.arange(-125.0, -80.0, 0.5), np.arange(-80.0, -60.0 + 0.01, 1.0)])
        da = _grid(np.arange(20.0, 55.01, 0.5), lons)

        with self.assertRaises(MCPToolError) as ctx:
            mask_data_by_geometry(da, SMALL_AOI)

        self.assertEqual(ctx.exception.category, CATEGORY_UNSUPPORTED_GRID)

    def test_applied_crop_logs_its_before_and_after_cell_counts(self):
        """The win has to be measurable from production logs, not inferred."""
        da = _continental_grid()

        with self.assertLogs("tta_backend.utils.plotting", level="INFO") as captured:
            cropped = mask_data_by_geometry(da, SMALL_AOI)

        events = [r for r in captured.records if r.getMessage() == "aoi_crop_applied"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]._cells_before, da.size)
        self.assertEqual(events[0]._cells_after, cropped.size)
        self.assertEqual(len(events[0]._crop_bounds), 4)

    def test_skipped_crop_logs_nothing(self):
        """Absent when the crop stands down, so a grep for the event counts
        real crops rather than attempts."""
        shifted = _grid(np.arange(20.0, 55.01, 0.5), np.arange(235.0, 300.01, 0.5))

        with self.assertNoLogs("tta_backend.utils.plotting", level="INFO"):
            mask_data_by_geometry(shifted, SMALL_AOI)

    def test_a_crop_that_drops_nothing_is_not_reported_as_a_win(self):
        """A region covering the whole granule has nothing to crop; it must
        pass the array through untouched rather than log a no-op crop and
        leave an operator counting attempts as savings."""
        da = _continental_grid()
        whole_grid = box(-130.0, 15.0, -55.0, 60.0)

        with self.assertNoLogs("tta_backend.utils.plotting", level="INFO"):
            masked = mask_data_by_geometry(da, whole_grid)

        xr.testing.assert_identical(masked, mask_data_by_geometry(da, whole_grid, crop=False))

    def test_every_statistic_is_identical_across_the_crop(self):
        """Story #2: a performance change must never move a published number.
        Runs the real reduction helpers -- including the cos(latitude)
        area-weighted mean, whose weights the crop could plausibly disturb --
        over the shapes real regions actually take."""
        from tta_backend.preprocessing.aggregation_service import AggregationService, area_weighted_mean
        from tta_backend.utils.plotting import load_preset_polygons

        service = AggregationService()
        regions = {
            "small box": SMALL_AOI,
            # Sub-cell: the T42 point-buffer shape.
            "point buffer": box(-74.06, 40.71, -73.96, 40.81),
            # A real, ragged, continental-scale polygon where cos(lat)
            # weighting genuinely matters.
            "conus polygon": load_preset_polygons()["conus"],
        }

        for name, geometry in regions.items():
            with self.subTest(region=name):
                da = _continental_grid()
                cropped = mask_data_by_geometry(da, geometry)
                uncropped = mask_data_by_geometry(da, geometry, crop=False)

                crop_valid = cropped.values[np.isfinite(cropped.values)]
                full_valid = uncropped.values[np.isfinite(uncropped.values)]

                self.assertEqual(len(crop_valid), len(full_valid))
                self.assertGreater(len(crop_valid), 0)
                for stat in ("min", "max", "median", "std", "mean"):
                    self.assertEqual(
                        service.compute_values_stat(crop_valid, stat),
                        service.compute_values_stat(full_valid, stat),
                        f"{stat} moved",
                    )
                np.testing.assert_allclose(
                    area_weighted_mean(cropped), area_weighted_mean(uncropped), rtol=1e-12
                )

    def test_crop_survives_a_time_dimension(self):
        """Real granules arrive as (time, lat, lon); the crop touches only the
        horizontal axes and leaves the temporal reduction alone."""
        da = _continental_grid().expand_dims(time=3).copy()
        da.values = np.arange(da.size, dtype=float).reshape(da.shape)

        cropped = mask_data_by_geometry(da, SMALL_AOI)
        uncropped = mask_data_by_geometry(da, SMALL_AOI, crop=False)

        self.assertEqual(cropped.sizes["time"], 3)
        np.testing.assert_array_equal(_finite(cropped), _finite(uncropped))
        np.testing.assert_array_equal(
            _finite(cropped.mean(dim="time")), _finite(uncropped.mean(dim="time"))
        )

    def test_map_extent_and_pixel_edges_are_unchanged_by_the_crop(self):
        """Story #4: the plot path crops the masked array to the region before
        deriving overlay bounds, so a tighter masked array must land on exactly
        the same extent -- and the same pixel edges the PNG is rendered at."""
        from tta_backend.tools.satellite_tools.plot_tools import _half_cell, _sel_bounds

        da = _continental_grid()
        cropped = _sel_bounds(mask_data_by_geometry(da, SMALL_AOI), "lat", "lon", SMALL_AOI.bounds)
        full = _sel_bounds(
            mask_data_by_geometry(da, SMALL_AOI, crop=False), "lat", "lon", SMALL_AOI.bounds
        )

        xr.testing.assert_identical(cropped, full)
        self.assertEqual(_half_cell(cropped.lon.values), _half_cell(full.lon.values))
        self.assertEqual(_half_cell(cropped.lat.values), _half_cell(full.lat.values))

    def test_the_crop_helpers_are_one_implementation_not_two_copies(self):
        """Decision #5: the bounds crop and the pixel-edge half-cell live in
        utils.plotting as one implementation that plot_tools imports, rather
        than a tool-module copy that export_service reaches through."""
        from tta_backend.tools.satellite_tools import plot_tools
        from tta_backend.utils import plotting

        self.assertIs(plot_tools._sel_bounds, plotting.sel_bounds)
        self.assertIs(plot_tools._half_cell, plotting.half_cell)


if __name__ == "__main__":
    unittest.main()
