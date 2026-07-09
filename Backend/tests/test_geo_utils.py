import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

import xarray as xr  # noqa: E402

from utils.geo_utils import find_lat_coord, find_lon_coord  # noqa: E402


class FindLatLonCoordTests(unittest.TestCase):
    def test_finds_bare_coord_names_at_root(self):
        da = xr.DataArray([[1.0]], dims=("lat", "lon"), coords={"lat": [1.0], "lon": [2.0]})

        self.assertEqual(find_lat_coord(da), "lat")
        self.assertEqual(find_lon_coord(da), "lon")

    def test_finds_promoted_coords_from_a_grouped_netcdf_file(self):
        """services/open_handle.py promotes lat/lon-like data variables
        (e.g. TEMPO L3's /geolocation/latitude) to coordinates before a
        science variable is selected out of the merged Dataset -- so by the
        time a DataArray reaches here, "latitude"/"longitude" are ordinary
        bare coordinate names, same as any flat file."""
        da = xr.DataArray(
            [[1.0]],
            dims=("mirror_step", "xtrack"),
            coords={
                "latitude": (("mirror_step", "xtrack"), [[30.0]]),
                "longitude": (("mirror_step", "xtrack"), [[-100.0]]),
            },
        )

        self.assertEqual(find_lat_coord(da), "latitude")
        self.assertEqual(find_lon_coord(da), "longitude")

    def test_returns_none_when_no_coords_match(self):
        da = xr.DataArray([[1.0]], dims=("y", "x"))

        self.assertIsNone(find_lat_coord(da))
        self.assertIsNone(find_lon_coord(da))


if __name__ == "__main__":
    unittest.main()
