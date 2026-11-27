"""
Regional means are cos(latitude) area-weighted
(domain.aggregation.aggregation_service.area_weighted_mean). On a lat/lon
grid, cell area shrinks by cos(latitude) toward the poles; an unweighted mean
over cells over-weights high latitudes (a few percent for a CONUS-scale
region, badly wrong for continental/global ones).
"""
import importlib.util
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


_HERMETIC_MODULES = ["numpy", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in _HERMETIC_MODULES),
    "area_weighted_mean test dependencies are not installed",
)
class AreaWeightedMeanTests(unittest.TestCase):
    """Hermetic — the ONE regional-mean definition, no fake MCP needed."""

    def test_weights_cells_by_cosine_of_latitude(self):
        import xarray as xr
        from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

        # Two lat rows: value 0.0 at the equator (weight 1.0), 1.0 at 60°N
        # (weight 0.5) -> weighted mean 1/3, where the unweighted mean is 0.5.
        da = xr.DataArray(
            [[0.0, 0.0], [1.0, 1.0]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0, 20.0]},
        )

        self.assertAlmostEqual(area_weighted_mean(da), 1.0 / 3.0, places=6)

    def test_nan_cells_are_excluded_not_zero_filled(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

        da = xr.DataArray(
            [[np.nan, np.nan], [1.0, 3.0]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0, 20.0]},
        )

        # Only the 60°N row survives; equal weights within a row -> plain mean.
        self.assertAlmostEqual(area_weighted_mean(da), 2.0, places=6)

    def test_all_nan_padding_around_the_data_changes_nothing(self):
        """The regional mean must depend on the cells that carry data, not on
        how much empty grid surrounds them: masking a small AOI out of a
        continental granule and out of a tight crop of that same granule is
        the same question. Real granules publish float32 latitudes, and
        weights built in that dtype make the answer drift with the number of
        zero-weight cells summed -- 5e-7 on a real TEMPO New Jersey mean."""
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

        lats = np.arange(38.79, 41.35, 0.02).astype(np.float32)
        lons = np.arange(-75.57, -73.88, 0.02).astype(np.float32)
        rng = np.random.default_rng(7)
        values = rng.random((len(lats), len(lons)))
        padded = xr.DataArray(
            values, dims=("lat", "lon"),
            coords={"lat": ("lat", lats), "lon": ("lon", lons)},
        )
        # Blank everything outside a small interior window, then take the same
        # window as a tight array: identical data, very different padding.
        window = {"lat": slice(40, 70), "lon": slice(10, 40)}
        blanked = padded.where(False)
        blanked[window["lat"], window["lon"]] = padded[window["lat"], window["lon"]]
        tight = padded.isel(window)

        np.testing.assert_allclose(
            area_weighted_mean(blanked), area_weighted_mean(tight), rtol=1e-12
        )

    def test_falls_back_to_unweighted_mean_without_a_latitude_axis(self):
        import xarray as xr
        from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

        da = xr.DataArray([1.0, 2.0, 3.0], dims=("x",))

        self.assertAlmostEqual(area_weighted_mean(da), 2.0, places=6)

    def test_raises_when_no_finite_values_remain(self):
        import numpy as np
        import xarray as xr
        from tta_backend.domain.aggregation.aggregation_service import area_weighted_mean

        da = xr.DataArray(
            [[np.nan], [np.nan]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 60.0], "lon": [10.0]},
        )

        with self.assertRaises(ValueError):
            area_weighted_mean(da)


if __name__ == "__main__":
    unittest.main()
