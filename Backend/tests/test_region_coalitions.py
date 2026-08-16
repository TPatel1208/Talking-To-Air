"""
T60 Phase 1: a named coalition is a real boundary, and it is reachable.

The Ozone Transport Region is not an OSM place and has no ``global_regions``
entry (D3 forbids one -- its bounding box would silently swallow West
Virginia, most of Ohio and part of Ontario). Before this phase the resolver
had exactly one door: ``if location_lower in self.global_regions``, else
geocode -- so a checked-in polygon with no preset key was unreachable and
``"otc"`` went to Nominatim, which answers with an airport in Chad (V2).

These tests pin the dispatcher seam that makes a polygon-only preset
reachable, the three-state outcome that keeps a broken asset away from the
geocoder (D3b), and the V1 Virginia verdict -- positively, since an
extent-only assertion would pass on a polygon over-including 96% of a state.
"""
import importlib.util
import os
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-coalition test dependencies are not installed",
)
class CoalitionPresetTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_otc_resolves_to_a_real_polygon_without_a_global_regions_entry(self):
        """The whole point of the seam: no preset key, still resolves."""
        resolver = self._resolver()
        # D3: the coalition deliberately has no bounding-box preset to fall
        # back on. If this ever gains one, the fail-closed rule is moot.
        self.assertNotIn("otc", resolver.global_regions)

        region = resolver.resolve_location("otc")

        self.assertIsNotNone(region)
        self.assertEqual(region["region_type"], "polygon")
        self.assertIn("Ozone Transport Region", region["display_name"])

    def test_otc_polygon_covers_every_member_and_no_non_member(self):
        """D3's motivation, pinned: the coalition's *envelope* would swallow
        West Virginia, most of Ohio and part of Ontario. The dissolved
        polygon must not."""
        from shapely.geometry import Point

        geometry = self._resolver().resolve_location("otc")["geometry"]

        # The eleven whole States CAA 184(a) names, plus DC.
        members = {
            "CT": (-72.68, 41.76), "DE": (-75.52, 39.15), "ME": (-69.50, 45.30),
            "MD": (-76.61, 39.29), "MA": (-71.80, 42.26), "NH": (-71.54, 43.21),
            "NJ": (-74.50, 40.22), "NY": (-76.15, 43.05), "PA": (-77.86, 40.79),
            "RI": (-71.41, 41.82), "VT": (-72.58, 44.26), "DC": (-77.02, 38.90),
        }
        for code, (lon, lat) in members.items():
            with self.subTest(member=code):
                self.assertTrue(geometry.contains(Point(lon, lat)))

        non_members = {
            "West Virginia": (-80.50, 38.90),
            "Ohio": (-82.90, 40.00),
            "Ontario": (-79.38, 43.65),
        }
        for place, (lon, lat) in non_members.items():
            with self.subTest(non_member=place):
                self.assertFalse(geometry.contains(Point(lon, lat)))

    def test_virginia_is_excluded_and_the_omission_is_disclosed(self):
        """The V1/D3a verdict, pinned on both halves.

        An extent-only assertion is not the test: the PRD's original Phase 1
        test would have passed on a polygon over-including 96% of Virginia.
        So this asserts the shape *and* that the resolved region tells the
        researcher what was left out -- in ``display_name``, which is what an
        answer cites, not in a code comment nobody reads.
        """
        from shapely.geometry import Point

        region = self._resolver().resolve_location("otc")
        geometry = region["geometry"]

        # CAA 184(a) does not name Virginia; 96.0% of the State is outside
        # the region entirely.
        self.assertFalse(geometry.contains(Point(-79.94, 37.27)))  # Roanoke
        self.assertFalse(geometry.contains(Point(-77.46, 37.54)))  # Richmond
        # ...and the ~4% that IS in the OTR -- the DC-area CMSA -- is out too,
        # because 50m admin-1 cannot express a partial State. That is the
        # known, decided omission; pinned so it cannot change silently.
        self.assertFalse(geometry.contains(Point(-77.30, 38.85)))  # Fairfax County

        # Which the answer must say out loud.
        disclosure = region["display_name"]
        self.assertIn("excludes", disclosure.lower())
        self.assertIn("Virginia", disclosure)

    def test_the_checked_in_asset_documents_the_virginia_omission_itself(self):
        """The disclosure has to survive a rebuild. ``display_name`` lives in
        Python; the *asset* carries the statutory citation, the member list
        and the named omission in its own properties, so the file is
        self-describing to anyone who opens it without this codebase."""
        import json

        from tta_backend.utils.plotting import _PRESET_REGIONS_PATH

        with open(_PRESET_REGIONS_PATH, encoding="utf-8") as fh:
            features = {f["id"]: f["properties"] for f in json.load(fh)["features"]}

        otc = features["otc"]
        self.assertEqual(
            otc["members"],
            ["CT", "DE", "ME", "MD", "MA", "NH", "NJ", "NY", "PA", "RI", "VT", "DC"],
        )
        self.assertNotIn("VA", otc["members"])
        self.assertIn("184(a)", otc["authority"])
        # The omission is named by jurisdiction, not hand-waved.
        for jurisdiction in ("Arlington", "Fairfax", "Loudoun", "Prince William",
                             "Stafford", "Alexandria", "Falls Church", "Manassas"):
            with self.subTest(jurisdiction=jurisdiction):
                self.assertIn(jurisdiction, otc["approximation"])

    def test_new_england_is_the_six_states_not_a_rectangle(self):
        """Design tension 3, pinned. D12 listed ``"new england"`` as an alias
        of ``northeast us``, which is ``box(-80, 37, -66, 48)`` -- 8.3x New
        England's true area, containing New York, Philadelphia, Baltimore and
        DC. A researcher who asks for New England and gets a mean over
        Philadelphia has been told nothing."""
        from shapely.geometry import Point

        region = self._resolver().resolve_location("new england")

        self.assertEqual(region["region_type"], "polygon")
        for name, (lon, lat) in {
            "CT": (-72.68, 41.76), "ME": (-69.50, 45.30), "MA": (-71.80, 42.26),
            "NH": (-71.54, 43.21), "RI": (-71.41, 41.82), "VT": (-72.58, 44.26),
        }.items():
            with self.subTest(member=name):
                self.assertTrue(region["geometry"].contains(Point(lon, lat)))

        # The rectangle's occupants, which New England is not.
        for name, (lon, lat) in {
            "New York City": (-73.94, 40.71), "Philadelphia": (-75.16, 39.95),
            "Baltimore": (-76.61, 39.29), "Washington DC": (-77.02, 38.90),
        }.items():
            with self.subTest(outside=name):
                self.assertFalse(region["geometry"].contains(Point(lon, lat)))

    def test_every_otc_alias_produces_an_identical_region(self):
        """Tension 4: aliases resolve *through* ``_finalize_preset``, not
        around it. An alias table that builds its own dict is a second
        definition of what a region is, and the two will drift. Comparing the
        whole object -- not just ``region_type`` -- is the load-bearing half;
        D12a's collision check is the cheap half."""
        resolver = self._resolver()
        canonical = resolver.resolve_location("otc")

        for alias in ("OTC", "the OTC", "  the   otc  ", "otr",
                      "Ozone Transport Commission", "ozone transport region"):
            with self.subTest(alias=alias):
                self.assertEqual(resolver.resolve_location(alias), canonical)

    def test_northeastern_us_is_identical_to_northeast_us(self):
        """D12's phrasing-variance patch. Today ``"northeastern us"`` misses
        the preset dict and geocodes to a railway platform at Northeastern
        University in Boston (Phase 0 gate, V2)."""
        resolver = self._resolver()

        self.assertEqual(
            resolver.resolve_location("northeastern us"),
            resolver.resolve_location("northeast us"),
        )
        # ...and it is still the honest rectangle it always was, not upgraded
        # by passing through the new door.
        self.assertEqual(
            resolver.resolve_location("northeastern us")["region_type"], "bounding_box"
        )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-coalition test dependencies are not installed",
)
class CoalitionFailsClosedTests(unittest.IsolatedAsyncioTestCase):
    """D3b: a coalition that matched a known name and then failed to produce a
    polygon must NOT fall through to the geocoder.

    The assertion that matters is that the geocoder **was not called**, not
    merely that the answer was empty -- an empty answer is what you would also
    get from a geocode miss, and ``"OTC"`` does not miss. It resolves live to
    an aerodrome in Chad, handed back as ``region_type: polygon`` (V2).

    Verified by mutation, not assumed: a fail-OPEN dispatcher (the D3b
    short-circuit removed) returns ``region is None`` here **too**, because
    the stubbed geocoder returns None -- so an emptiness assertion alone
    cannot tell the two apart. Only ``assert_not_called`` catches it, and
    under the mutant the recorded call is literally ``geocode('otc')``.
    """

    def setUp(self):
        from tta_backend.utils.plotting import load_preset_polygons

        # lru_cache(maxsize=1), process-global. Clear going in *and* coming
        # out, or this test either reads a polygon it meant to hide or leaves
        # an empty dict behind to starve every later test in the process.
        load_preset_polygons.cache_clear()
        self.addCleanup(load_preset_polygons.cache_clear)

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    async def test_missing_asset_returns_not_found_without_geocoding(self):
        resolver = self._resolver()
        with patch("tta_backend.utils.plotting._PRESET_REGIONS_PATH", "no/such/asset.geojson"), \
                patch.object(resolver.geocoding_service, "geocode") as sync_geocode, \
                patch.object(resolver.geocoding_service, "ageocode") as async_geocode:
            sync_region = resolver.resolve_location("otc")
            async_region = await resolver.aresolve_location("otc")

        self.assertIsNone(sync_region)
        self.assertIsNone(async_region)
        sync_geocode.assert_not_called()
        async_geocode.assert_not_called()

    async def test_corrupt_asset_returns_not_found_without_geocoding(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False,
                                         encoding="utf-8") as fh:
            fh.write("{not valid json")
            corrupt_path = fh.name
        self.addCleanup(os.unlink, corrupt_path)

        resolver = self._resolver()
        with patch("tta_backend.utils.plotting._PRESET_REGIONS_PATH", corrupt_path), \
                patch.object(resolver.geocoding_service, "geocode") as sync_geocode, \
                patch.object(resolver.geocoding_service, "ageocode") as async_geocode:
            sync_region = resolver.resolve_location("the OTC")
            async_region = await resolver.aresolve_location("ozone transport region")

        self.assertIsNone(sync_region)
        self.assertIsNone(async_region)
        sync_geocode.assert_not_called()
        async_geocode.assert_not_called()

    async def test_an_unclaimed_string_still_reaches_the_geocoder(self):
        """The other side of the seam: the dispatcher must not swallow
        everything. A string outside the new vocabulary falls through to
        exactly today's behavior."""
        resolver = self._resolver()
        geo = {"latitude": 48.8, "longitude": 2.3, "display_name": "Paris, France",
               "polygon": None, "bbox": []}
        with patch.object(resolver.geocoding_service, "geocode",
                          return_value=geo) as sync_geocode:
            region = resolver.resolve_location("paris")

        sync_geocode.assert_called_once()
        self.assertEqual(region["region_type"], "point_buffer")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-coalition test dependencies are not installed",
)
class DispatcherSeamTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    async def test_sync_and_async_agree_on_every_new_string(self):
        """T42's own shipped bug was these two paths normalizing differently,
        which is why ``_normalize_location_name`` is shared (D11a). A
        dispatcher wired into only one resolver is a region that resolves two
        ways -- and ``export_service`` calls the sync one while every analysis
        tool calls the async one."""
        from tta_backend.utils import region_dispatch

        resolver = self._resolver()
        new_strings = (
            list(region_dispatch.COALITIONS)
            + list(region_dispatch.ALIASES)
            + ["the OTC", "  Ozone   Transport  Region  "]
        )
        # Stubbed to explode: if normalization diverges and a string escapes
        # the dispatcher, it must fail here rather than quietly geocoding.
        with patch.object(resolver.geocoding_service, "geocode", return_value=None), \
                patch.object(resolver.geocoding_service, "ageocode", return_value=None):
            for name in new_strings:
                with self.subTest(location=name):
                    sync_region = resolver.resolve_location(name)
                    async_region = await resolver.aresolve_location(name)
                    self.assertIsNotNone(sync_region)
                    self.assertEqual(sync_region, async_region)

    async def test_every_pre_existing_preset_resolves_unchanged(self):
        """The regression this phase is most likely to cause: a new first step
        ahead of the exact-match gate that swallows or alters a preset.
        Enumerated over the whole dict, not spot-checked."""
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.plotting import RegionResolver

        resolver = self._resolver()
        polygon_keys = RegionResolver._POLYGON_PRESET_IDS

        with patch.object(resolver.geocoding_service, "geocode") as sync_geocode, \
                patch.object(resolver.geocoding_service, "ageocode") as async_geocode:
            for key, preset in resolver.global_regions.items():
                with self.subTest(preset=key):
                    # The dispatcher must not claim anything that already had
                    # a door -- it is an *additional* door, not a detour.
                    self.assertFalse(region_dispatch.dispatch(key, resolver).claimed)

                    sync_region = resolver.resolve_location(key)
                    async_region = await resolver.aresolve_location(key)
                    self.assertEqual(sync_region, async_region)
                    self.assertEqual(sync_region["name"], preset["name"])
                    self.assertEqual(
                        sync_region["region_type"],
                        "polygon" if key in polygon_keys else "bounding_box",
                    )

        sync_geocode.assert_not_called()
        async_geocode.assert_not_called()

    def test_collision_assertion_fires_on_a_shadowing_alias(self):
        """D12a. A hand-maintained table that silently shadows ``"us"`` or
        ``"georgia"`` is the cheapest possible way to reintroduce a confident
        wrong region, and review does not reliably catch it."""
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.region_dispatch import AliasCollisionError

        global_regions = self._resolver().global_regions

        # The real tables must be clean.
        region_dispatch.assert_no_alias_collisions(global_regions)

        shadowing_cases = {
            "alias shadows a preset": {"usa": "otc"},
            "alias shadows a coalition": {"otc": "new england"},
            "alias points at nothing": {"otcx": "no such region"},
        }
        for label, bad_aliases in shadowing_cases.items():
            with self.subTest(case=label):
                with patch.dict(region_dispatch.ALIASES, bad_aliases):
                    with self.assertRaises(AliasCollisionError):
                        region_dispatch.assert_no_alias_collisions(global_regions)

        # ...and a coalition id that grew a bounding-box preset, which would
        # quietly re-open the D3 fallback the whole design forbids.
        with patch.dict(region_dispatch.COALITIONS, {"usa": None}):
            with self.assertRaises(AliasCollisionError):
                region_dispatch.assert_no_alias_collisions(global_regions)


if __name__ == "__main__":
    unittest.main()
