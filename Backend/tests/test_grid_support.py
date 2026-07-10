"""T24: the failure contract for grids the affine mask math can't handle.

`mask_data_by_geometry` assumes a 1-D rectilinear lat/lon grid. Handed a 2-D
curvilinear swath or a projected x/y grid it used to silently mis-mask (or
crash with an opaque empty-coords error). It must instead raise a specific,
T18-typed `unsupported_grid` error that tells the researcher the truth about
the limitation.
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

from earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError  # noqa: E402
from utils.plotting import mask_data_by_geometry  # noqa: E402


class MaskGridSupportTests(unittest.TestCase):
    def test_rectilinear_grid_still_masks(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("latitude", "longitude"),
            coords={
                "latitude": ("latitude", [40.0, 41.0], {"units": "degrees_north"}),
                "longitude": ("longitude", [-75.0, -74.0], {"units": "degrees_east"}),
            },
        )

        masked = mask_data_by_geometry(da, box(-76, 39, -73, 42))

        self.assertIsInstance(masked, xr.DataArray)

    def test_curvilinear_2d_grid_raises_typed_unsupported_error(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("mirror_step", "xtrack"),
            coords={
                "latitude": (("mirror_step", "xtrack"), [[40.0, 40.0], [41.0, 41.0]], {"units": "degrees_north"}),
                "longitude": (("mirror_step", "xtrack"), [[-75.0, -74.0], [-75.0, -74.0]], {"units": "degrees_east"}),
            },
        )

        with self.assertRaises(MCPToolError) as ctx:
            mask_data_by_geometry(da, box(-76, 39, -73, 42))

        self.assertEqual(ctx.exception.category, CATEGORY_UNSUPPORTED_GRID)
        self.assertIn("curvilinear", ctx.exception.message.lower())

    def test_projected_grid_raises_typed_error_naming_the_crs(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
            coords={
                "y": ("y", [0.0, 1000.0], {"units": "m"}),
                "x": ("x", [0.0, 1000.0], {"units": "m"}),
                "crs": ((), 0, {"grid_mapping_name": "lambert_conformal_conic"}),
            },
        )

        with self.assertRaises(MCPToolError) as ctx:
            mask_data_by_geometry(da, box(-76, 39, -73, 42))

        self.assertEqual(ctx.exception.category, CATEGORY_UNSUPPORTED_GRID)
        self.assertIn("projected", ctx.exception.message.lower())
        self.assertIn("lambert_conformal_conic", ctx.exception.message)

    def test_no_recognizable_coords_error_names_the_dims_present(self):
        da = xr.DataArray(np.ones((2, 2)), dims=("scanline", "ground_pixel"))

        with self.assertRaises(ValueError) as ctx:
            mask_data_by_geometry(da, box(-76, 39, -73, 42))

        self.assertIn("scanline", str(ctx.exception))
        self.assertIn("ground_pixel", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
