import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["affine", "cartopy", "rasterio", "shapely", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "plotting dependencies are not installed",
)
class NormalizeTo2dTests(unittest.TestCase):
    """T25: the silent `.mean(dim=extra_dims)` fallback is deleted -- a
    surviving non-spatial, non-time dimension with no selection must refuse
    with a structured, candidate-listing error naming the dimension and its
    coordinate values (e.g. a MERRA-2 72-level vertical dim), never a
    confident-looking whole-atmosphere average."""

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr

    def test_squeezes_size_one_dims_with_no_error(self):
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.ones((1, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-01-01"], "lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
            name="no2",
        )

        result = _normalize_to_2d(da)

        self.assertEqual(result.dims, ("lat", "lon"))

    def test_raises_a_candidate_listing_error_for_an_unselected_level_dim(self):
        from earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.ones((3, 2, 2)),
            dims=("lev", "lat", "lon"),
            coords={
                "lev": ("lev", [1000.0, 500.0, 250.0], {"units": "hPa"}),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )

        with self.assertRaises(MCPToolError) as ctx:
            _normalize_to_2d(da)

        self.assertEqual(ctx.exception.category, CATEGORY_DIMENSION_CHOICE_REQUIRED)
        self.assertIn("lev", ctx.exception.message)
        self.assertIn("1000.0", ctx.exception.message)
        self.assertIn("500.0", ctx.exception.message)
        self.assertIn("250.0", ctx.exception.message)
        self.assertIn("hPa", ctx.exception.message)

    def test_dim_selector_resolves_the_level_dim_by_coordinate_value(self):
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            dims=("lev", "lat", "lon"),
            coords={"lev": [1000.0, 500.0], "lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
            name="no2",
        )

        result = _normalize_to_2d(da, dim_selector={"lev": 500.0})

        self.assertEqual(result.dims, ("lat", "lon"))
        self.assertEqual(result.values.tolist(), [[5.0, 6.0], [7.0, 8.0]])

    def test_dim_selector_out_of_range_value_is_rejected_not_snapped(self):
        """Follow-up review #4: a selection outside the coordinate's own
        min--max range (e.g. a bare hPa value against a Pa-unit ``lev`` coord)
        is refused with a structured error, not silently nearest-snapped to a
        plausible-but-wrong edge level."""
        from earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2),
            dims=("lev", "lat", "lon"),
            coords={
                "lev": ("lev", [100000.0, 85000.0, 50000.0], {"units": "Pa"}),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )

        # 500 hPa really means 50000 Pa; passed as a bare 500 it is far below
        # the Pa-unit coordinate's minimum and must be refused, not snapped.
        with self.assertRaises(MCPToolError) as ctx:
            _normalize_to_2d(da, dim_selector={"lev": 500.0})

        self.assertEqual(ctx.exception.category, CATEGORY_DIMENSION_CHOICE_REQUIRED)
        self.assertIn("lev", ctx.exception.message)
        self.assertIn("Pa", ctx.exception.message)

    def test_dim_selector_in_range_value_still_nearest_selects(self):
        """An in-range selection nearest-matches normally -- the range guard
        only rejects values outside the coordinate, not legitimate ones."""
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            dims=("lev", "lat", "lon"),
            coords={
                "lev": ("lev", [100000.0, 50000.0], {"units": "Pa"}),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )

        # 52000 Pa is within [50000, 100000] and nearest-matches 50000.
        result = _normalize_to_2d(da, dim_selector={"lev": 52000.0})

        self.assertEqual(result.dims, ("lat", "lon"))
        self.assertEqual(result.values.tolist(), [[5.0, 6.0], [7.0, 8.0]])

    def test_time_like_dim_still_auto_reduces_without_a_selector(self):
        """Time is the one transparent auto-reduction (PRD T25) -- a
        surviving time-identified dim must not raise, even without a
        selector, unlike a genuine extra dim such as a vertical level."""
        from utils.plotting import _normalize_to_2d

        da = self.xr.DataArray(
            self.np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            dims=("valid_time", "lat", "lon"),
            coords={
                "valid_time": ("valid_time", ["2024-01-01", "2024-01-02"], {"standard_name": "time"}),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )

        result = _normalize_to_2d(da)

        self.assertEqual(result.dims, ("lat", "lon"))


if __name__ == "__main__":
    unittest.main()
