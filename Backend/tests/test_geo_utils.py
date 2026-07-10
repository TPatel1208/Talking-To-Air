import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

import xarray as xr  # noqa: E402

from utils.geo_utils import find_lat_coord, find_lon_coord  # noqa: E402
from utils.geo_utils import identify_lat, identify_lon  # noqa: E402


class IdentifyLatLonTests(unittest.TestCase):
    """T24: coordinate identification keys on CF metadata (the signal every
    NASA Earthdata product is published against), not on a hardcoded list of
    variable-name spellings, so datasets we have never opened still work."""

    def test_cf_standard_name_identifies_lat_lon_regardless_of_variable_name(self):
        # Names the allowlist would never guess; only the CF standard_name
        # says which axis is which.
        ds = xr.Dataset(
            {"no2": (("row", "col"), [[1.0]])},
            coords={
                "row": ("row", [40.0], {"standard_name": "latitude"}),
                "col": ("col", [-75.0], {"standard_name": "longitude"}),
            },
        )

        self.assertEqual(identify_lat(ds), "row")
        self.assertEqual(identify_lon(ds), "col")

    def test_cf_units_identify_lat_lon_when_standard_name_is_absent(self):
        # Many products carry only the CF units, not standard_name.
        ds = xr.Dataset(
            {"no2": (("y", "x"), [[1.0]])},
            coords={
                "y": ("y", [40.0], {"units": "degrees_north"}),
                "x": ("x", [-75.0], {"units": "degrees_east"}),
            },
        )

        self.assertEqual(identify_lat(ds), "y")
        self.assertEqual(identify_lon(ds), "x")

    def test_cf_unit_spelling_variants_are_recognized(self):
        ds = xr.Dataset(
            {"no2": (("y", "x"), [[1.0]])},
            coords={
                "y": ("y", [40.0], {"units": "degree_N"}),
                "x": ("x", [-75.0], {"units": "degreesE"}),
            },
        )

        self.assertEqual(identify_lat(ds), "y")
        self.assertEqual(identify_lon(ds), "x")

    def test_name_allowlist_is_the_fallback_for_non_cf_files(self):
        # No CF metadata at all -- fall back to recognizing the name.
        ds = xr.Dataset(
            {"no2": (("latitude", "longitude"), [[1.0]])},
            coords={"latitude": [40.0], "longitude": [-75.0]},
        )

        self.assertEqual(identify_lat(ds), "latitude")
        self.assertEqual(identify_lon(ds), "longitude")

    def test_axis_beats_its_bounds_variable_when_both_match_metadata(self):
        # latitude_bounds carries the same CF units as the axis, so metadata
        # matching alone would tie -- the identifier must return the axis.
        ds = xr.Dataset(
            {"no2": (("latitude",), [1.0])},
            coords={
                "latitude_bounds": (("latitude", "nv"), [[39.5, 40.5]], {"units": "degrees_north"}),
                "latitude": ("latitude", [40.0], {"units": "degrees_north"}),
            },
        )

        self.assertEqual(identify_lat(ds), "latitude")

    def test_unseen_bounds_name_still_loses_to_the_axis_structurally(self):
        # 'lat_edges' is not caught by any *_bounds/_bnds suffix rule; the
        # structural rule (the axis has fewer dims) still wins.
        ds = xr.Dataset(
            {"no2": (("latitude",), [1.0])},
            coords={
                "lat_edges": (("latitude", "nv"), [[39.5, 40.5]], {"units": "degrees_north"}),
                "latitude": ("latitude", [40.0], {"units": "degrees_north"}),
            },
        )

        self.assertEqual(identify_lat(ds), "latitude")

    def test_science_var_coordinates_attribute_is_authoritative(self):
        # 'scanline_lat' (a 1-D nadir latitude) also carries degrees_north
        # and, being a dimension coordinate, would win the structural
        # tiebreak -- but the science var's own `coordinates` pointer names
        # the real 2-D grid, and it wins. The reference uses group paths our
        # merge strips, so it must be matched by leaf name.
        da = xr.DataArray(
            [[1.0]],
            dims=("mirror_step", "xtrack"),
            coords={
                "scanline_lat": ("mirror_step", [40.0], {"units": "degrees_north"}),
                "latitude": (("mirror_step", "xtrack"), [[40.0]], {"units": "degrees_north"}),
                "longitude": (("mirror_step", "xtrack"), [[-75.0]], {"units": "degrees_east"}),
            },
            attrs={"coordinates": "geolocation/longitude geolocation/latitude"},
        )

        self.assertEqual(identify_lat(da), "latitude")
        self.assertEqual(identify_lon(da), "longitude")


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

    def test_finds_cf_metadata_coords_with_unusual_names(self):
        """T24: find_lat_coord/find_lon_coord delegate to the canonical
        identifier, so a coordinate the name allowlist would miss is found
        via its CF standard_name."""
        da = xr.DataArray(
            [[1.0]],
            dims=("row", "col"),
            coords={
                "row": ("row", [40.0], {"standard_name": "latitude"}),
                "col": ("col", [-75.0], {"standard_name": "longitude"}),
            },
        )

        self.assertEqual(find_lat_coord(da), "row")
        self.assertEqual(find_lon_coord(da), "col")


if __name__ == "__main__":
    unittest.main()
