"""
T42 region fidelity: the region you named is the region we computed.

RegionResolver now discloses *what kind* of region it resolved
(``region_type``: polygon | bounding_box | point_buffer | boundary_cells)
and the geocoder's ``display_name``, so "mean over the US" is checkable
against what was actually masked. Preset multi-country concepts (US,
continents) carry real Natural Earth polygons; pure-box concepts (global,
sub-continental US quadrants) stay boxes but say so. Sync and async
resolvers normalize input identically.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-resolver test dependencies are not installed",
)
class RegionResolverFidelityTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from utils.plotting import RegionResolver
        return RegionResolver()

    def test_box_preset_discloses_bounding_box_type_and_display_name(self):
        region = self._resolver().resolve_location("northeast us")
        self.assertEqual(region["region_type"], "bounding_box")
        # display_name travels alongside the internal name so the answer can
        # name the region it actually masked.
        self.assertEqual(region["display_name"], "Northeast US")

    def test_geocoded_polygon_discloses_polygon_type_and_display_name(self):
        geo = {
            "latitude": 40.0,
            "longitude": -75.0,
            "display_name": "Paris, Île-de-France, France",
            # A real (if trivial) polygon boundary, as Nominatim returns for
            # an administrative area.
            "polygon": {
                "type": "Polygon",
                "coordinates": [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]],
            },
            "bbox": [-1.0, 1.0, -1.0, 1.0],
        }
        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=geo):
            region = resolver.resolve_location("Paris")

        self.assertEqual(region["region_type"], "polygon")
        self.assertEqual(region["display_name"], "Paris, Île-de-France, France")

    def test_geocoded_point_without_polygon_discloses_point_buffer(self):
        geo = {
            "latitude": 40.0,
            "longitude": -75.0,
            "display_name": "Some Small Hamlet",
            "polygon": None,  # Nominatim returned no boundary
            "bbox": [],
        }
        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=geo):
            region = resolver.resolve_location("Some Small Hamlet")

        self.assertEqual(region["region_type"], "point_buffer")
        # The minted footprint is the 0.1° box around the centroid.
        minx, miny, maxx, maxy = region["bounds"]
        self.assertAlmostEqual(maxx - minx, 0.2)
        self.assertAlmostEqual(maxy - miny, 0.2)

    async def test_sync_and_async_resolvers_strip_the_prefix_identically(self):
        # The async path stripped a leading "the "; the sync path didn't, so
        # the same string resolved two ways by code path. Both must now
        # normalize identically -- "the north america" hits the preset, never
        # a geocode. geocode is stubbed to None so a normalization miss fails
        # loudly (None has no keys) instead of hitting the network.
        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=None):
            sync_region = resolver.resolve_location("the north america")
        with patch.object(resolver.geocoding_service, "ageocode", return_value=None):
            async_region = await resolver.aresolve_location("the north america")

        self.assertEqual(sync_region["name"], "North America")
        self.assertEqual(sync_region["name"], async_region["name"])
        self.assertEqual(sync_region["region_type"], async_region["region_type"])

    def test_usa_preset_uses_a_real_polygon_not_the_crude_box(self):
        from shapely.geometry import Point

        region = self._resolver().resolve_location("usa")
        self.assertEqual(region["region_type"], "polygon")
        # The old box(-125, 24, -66, 50) averaged in the Gulf of Mexico; the
        # real CONUS boundary excludes it.
        self.assertFalse(region["geometry"].contains(Point(-90, 26)))
        # ...while a US-interior point stays inside.
        self.assertTrue(region["geometry"].contains(Point(-98, 39)))

    async def test_async_geocoded_polygon_matches_sync_disclosure(self):
        """Same geocoder hit -> identical region_type/display_name whether the
        caller took the sync or async path (shared ``_geocoded_region``)."""
        geo = {
            "latitude": 48.8,
            "longitude": 2.3,
            "display_name": "Paris, Île-de-France, France",
            "polygon": {
                "type": "Polygon",
                "coordinates": [[[2.2, 48.8], [2.4, 48.8], [2.4, 48.9], [2.2, 48.9], [2.2, 48.8]]],
            },
            "bbox": [48.8, 48.9, 2.2, 2.4],
        }
        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=geo):
            sync_region = resolver.resolve_location("Paris")
        with patch.object(resolver.geocoding_service, "ageocode", return_value=geo):
            async_region = await resolver.aresolve_location("Paris")

        self.assertEqual(sync_region["region_type"], "polygon")
        self.assertEqual(async_region["region_type"], sync_region["region_type"])
        self.assertEqual(async_region["display_name"], sync_region["display_name"])


if __name__ == "__main__":
    unittest.main()
