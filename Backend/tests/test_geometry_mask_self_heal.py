"""
T42 empty-mask self-heal: a region smaller than a grid cell (a neighborhood
on a coarse satellite grid, or the 0.1° point-buffer minted when the geocoder
returns no polygon) can cover zero cell *centers*, so center-containment
rasterization returns an all-False mask and the tool answers "no valid data"
for data that is right there. ``geometry_mask`` now retries an all-False mask
with ``all_touched=True`` and, when that finds the boundary cells, returns
them -- disclosed as ``region_type: boundary_cells`` -- instead of nothing.
A genuinely off-grid polygon stays all-False (today's honest no-data).
"""
import importlib.util
import unittest


REQUIRED_MODULES = ["numpy", "xarray", "shapely", "rasterio", "affine", "cartopy"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "geometry-mask test dependencies are not installed",
)
class GeometryMaskSelfHealTests(unittest.TestCase):
    def _coarse_grid(self):
        """A 3x3 grid on a 10° step: centers at 0/10/20, cells spanning ±5°."""
        import numpy as np
        import xarray as xr

        return xr.DataArray(
            np.arange(9, dtype=float).reshape(3, 3),
            coords={"lat": [0.0, 10.0, 20.0], "lon": [0.0, 10.0, 20.0]},
            dims=["lat", "lon"],
        )

    def test_subcell_polygon_self_heals_to_boundary_cells(self):
        from shapely.geometry import box
        from tta_backend.utils.plotting import geometry_mask

        da = self._coarse_grid()
        # A 2° box near (3,3): covers no cell center, but overlaps the cell
        # centered at (0,0) (which spans [-5,5]×[-5,5]).
        mask = geometry_mask(da, box(2.0, 2.0, 4.0, 4.0))

        self.assertTrue(bool(mask.values.any()), "self-heal should recover the boundary cell")
        self.assertEqual(mask.attrs.get("region_type"), "boundary_cells")
        # It recovered exactly the touched cell, not the whole grid.
        self.assertEqual(int(mask.values.sum()), 1)

    def test_off_grid_polygon_stays_empty_and_unmarked(self):
        from shapely.geometry import box
        from tta_backend.utils.plotting import geometry_mask

        da = self._coarse_grid()
        # Far outside the grid extent: neither pass finds a cell, so the mask
        # is honestly empty and carries no boundary_cells claim.
        mask = geometry_mask(da, box(200.0, 200.0, 201.0, 201.0))

        self.assertFalse(bool(mask.values.any()))
        self.assertNotIn("region_type", mask.attrs)

    def test_normal_polygon_covering_centers_is_not_marked_boundary_cells(self):
        from shapely.geometry import box
        from tta_backend.utils.plotting import geometry_mask

        da = self._coarse_grid()
        # A polygon big enough to contain real cell centers takes the ordinary
        # center-containment path -- no self-heal, no boundary_cells label.
        mask = geometry_mask(da, box(-1.0, -1.0, 11.0, 11.0))

        self.assertTrue(bool(mask.values.any()))
        self.assertNotIn("region_type", mask.attrs)


if __name__ == "__main__":
    unittest.main()
