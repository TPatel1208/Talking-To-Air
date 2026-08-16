"""T60 Phase 3a: the fifty-one U.S. states and DC are individually resolvable.

Before this phase every state name fell through ``RegionResolver``'s exact-match
gate to Nominatim. The Phase 3a gate (V12) measured what that actually costs on
eight live queries: six of eight already came back as the correct
``boundary/administrative`` relation, so on the mask plane this phase is mostly
buying *determinism* -- no network, no rate limiter, no OSM relation edits.

Two of the eight were not neutral, and they are the reason the phase ships:

- ``"new york"`` resolved to New York **City** -- 0.47% of the state.
- ``"washington"`` resolved to Washington **DC**, tagged ``place/city`` --
  **0.00%** overlap with Washington State, 47 degrees of longitude away.

The retrieval plane is worse, and structurally so (V13). Even for the tokens
Nominatim geocodes correctly the MCP's extent does **not** contain the shipped
mask: Georgia covers 99.99% and still fails containment because the mask comes
from Natural Earth and the extent comes from OSM, and two providers have
different edges. That gap cannot be closed by a better geocoder -- only by both
planes deriving from the same polygon, which is what these tests pin.
"""
import importlib.util
import json
import os
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region us-state test dependencies are not installed",
)
class StatePresetTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_a_state_resolves_to_a_real_polygon_without_geocoding(self):
        """The tracer bullet. ``"ozone over Pennsylvania"`` is a real request
        that today goes to a geocoder; after this phase it is a checked-in
        boundary, and the geocoder is never consulted."""
        from shapely.geometry import Point

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = resolver.resolve_location("pennsylvania")

        geocode.assert_not_called()
        self.assertEqual(region["region_type"], "polygon")
        self.assertTrue(region["geometry"].contains(Point(-77.86, 40.79)))  # Harrisburg
        self.assertFalse(region["geometry"].contains(Point(-73.94, 40.71)))  # NYC

    async def test_all_fifty_one_resolve_to_a_real_boundary_not_a_rectangle(self):
        """Criterion #2, over the whole set rather than a spot check of three.

        The bite is ``region_type``: a state the asset stops carrying still has
        a ``global_regions`` entry, so it keeps resolving -- as its *envelope*,
        labelled ``bounding_box``. That is a silent 2-3x over-inclusion, which
        is exactly the failure a three-state spot check would sail past. The
        area assertion is the second lock: a real boundary is strictly smaller
        than its own envelope, and the smallest margin across the 51 is
        Colorado's, which is very nearly rectangular."""
        from tta_backend.datasets.us_states import US_STATES
        from shapely.geometry import box

        resolver = self._resolver()
        self.assertEqual(len(US_STATES), 51)

        with patch.object(resolver.geocoding_service, "geocode") as sync_geocode, \
                patch.object(resolver.geocoding_service, "ageocode") as async_geocode:
            for key in US_STATES:
                with self.subTest(state=key):
                    region = resolver.resolve_location(key)
                    self.assertIsNotNone(region)
                    self.assertEqual(region["region_type"], "polygon")

                    geometry = region["geometry"]
                    self.assertIn(geometry.geom_type, ("Polygon", "MultiPolygon"))
                    self.assertGreater(geometry.area, 0.0)
                    # Not the rectangle it degrades to.
                    self.assertLess(geometry.area, box(*geometry.bounds).area)

                    # T42's shipped bug was the two resolvers normalizing
                    # differently; export_service calls the sync one and every
                    # analysis tool calls the async one.
                    self.assertEqual(await resolver.aresolve_location(key), region)

        sync_geocode.assert_not_called()
        async_geocode.assert_not_called()

    def test_the_v12_ambiguous_tokens_land_on_the_state_and_say_so(self):
        """Criterion #3, and the reason the phase ships at all.

        Gate V12, live: ``"washington"`` returned Washington **DC** tagged
        ``place/city`` -- 0.00% overlap with Washington State -- and
        ``"new york"`` returned New York **City**, 0.47% of the state. D15
        hands both tokens to the state, so the answer must *say* which one it
        meant; ``display_name`` is where T42 puts that, and it is what a plot
        title and a provenance block cite."""
        from shapely.geometry import Point

        resolver = self._resolver()

        washington = resolver.resolve_location("washington")
        self.assertEqual(washington["display_name"], "Washington (U.S. state)")
        self.assertTrue(washington["geometry"].contains(Point(-122.33, 47.61)))   # Seattle
        self.assertFalse(washington["geometry"].contains(Point(-77.02, 38.90)))   # DC

        new_york = resolver.resolve_location("new york")
        self.assertEqual(new_york["display_name"], "New York (U.S. state)")
        self.assertTrue(new_york["geometry"].contains(Point(-78.88, 42.89)))      # Buffalo
        self.assertTrue(new_york["geometry"].contains(Point(-73.94, 40.71)))      # NYC too

        # ...and the other Washington is still reachable by its own name, which
        # is what makes D15's order a disambiguation rather than a deletion.
        dc = resolver.resolve_location("district of columbia")
        self.assertEqual(dc["display_name"],
                         "District of Columbia (U.S. federal district)")
        self.assertTrue(dc["geometry"].contains(Point(-77.02, 38.90)))

    def test_every_state_discloses_its_kind_uniformly(self):
        """The disclosure is a rule, not a per-token exception list -- a list
        with 51 entries and 2 exceptions drifts the first time a state is
        added. Derived from Natural Earth's own ``type``, so DC is a federal
        district because the source says so."""
        from tta_backend.datasets.us_states import US_STATES

        resolver = self._resolver()
        suffixes = {"state": "(U.S. state)",
                    "federal_district": "(U.S. federal district)"}

        for key, spec in US_STATES.items():
            with self.subTest(state=key):
                region = resolver.resolve_location(key)
                self.assertEqual(
                    region["display_name"],
                    f"{spec['name']} {suffixes[spec['kind']]}",
                )
                # ``name`` stays the bare label a plot title wants; the
                # disambiguation rides on display_name, not on both.
                self.assertEqual(region["name"], spec["name"])

        self.assertEqual(
            [k for k, s in US_STATES.items() if s["kind"] == "federal_district"],
            ["district of columbia"],
        )


# Areas of the **unsimplified** 50m admin-1 features, deg2, measured against
# ne_50m_admin_1_states_provinces.geojson (2,325,694 bytes) in the Phase 3a
# gate. Checked in as constants because criterion #8 forbids a test touching
# the network, and the whole point of the assertion is to compare the shipped
# geometry against something the shipped geometry cannot influence.
#
# DC and RI are the jurisdictions D2a exists to protect. NH and CT are the
# minimum and maximum retention across all 51, so the band is anchored at both
# real extremes rather than at a guess.
UNSIMPLIFIED_AREA_DEG2 = {
    "district of columbia": 0.01485614,
    "rhode island": 0.28184579,
    "new hampshire": 2.70147822,   # lowest retention of the 51
    "connecticut": 1.36713013,     # highest retention of the 51
}


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region us-state test dependencies are not installed",
)
class SmallJurisdictionFidelityTests(unittest.TestCase):
    """D2a, pinned. Phase 0's V4 measured DC at **58.71%** area retention under
    the builder's pre-existing 0.1 deg constant -- a 41% distortion of a
    jurisdiction 16 km across. 0.01 deg is not a preference, it is the number
    that keeps DC. If ``ADMIN1_TOLERANCE`` is ever raised and the asset
    rebuilt, this is what fails.
    """

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_small_jurisdictions_keep_their_area_within_a_two_sided_band(self):
        """Two-sided on purpose. V4's other finding was that simplification
        can *enlarge*: dropping a concave vertex adds area, and DC reads
        **100.17%**, not 99-point-something. A one-sided ``>= 99%`` assertion
        would pass on a polygon inflated to 150% of the jurisdiction."""
        resolver = self._resolver()

        for key, true_area in UNSIMPLIFIED_AREA_DEG2.items():
            with self.subTest(jurisdiction=key):
                shipped = resolver.resolve_location(key)["geometry"].area
                retention = 100.0 * shipped / true_area
                self.assertGreater(
                    retention, 99.0,
                    f"{key} lost shape: {retention:.3f}% of the unsimplified area",
                )
                self.assertLess(
                    retention, 101.0,
                    f"{key} gained shape: {retention:.3f}% of the unsimplified area",
                )

    def test_dc_and_rhode_island_keep_their_boundary_not_just_their_size(self):
        """Area retention alone is satisfied by a displaced blob of the right
        size. These are the edges that matter: DC's western boundary is the
        Potomac line against non-member Virginia -- the same line the OTR's
        outer boundary runs along, which is why V4 called DC's 58.71% the
        deciding number rather than RI's."""
        from shapely.geometry import Point

        resolver = self._resolver()

        dc = resolver.resolve_location("district of columbia")["geometry"]
        self.assertTrue(dc.contains(Point(-77.023, 38.890)))    # National Mall
        self.assertFalse(dc.contains(Point(-77.087, 38.879)))   # Arlington, VA
        self.assertFalse(dc.contains(Point(-77.100, 38.984)))   # Bethesda, MD

        ri = resolver.resolve_location("rhode island")["geometry"]
        self.assertTrue(ri.contains(Point(-71.412, 41.824)))    # Providence
        self.assertTrue(ri.contains(Point(-71.313, 41.490)))    # Newport
        self.assertFalse(ri.contains(Point(-71.155, 41.701)))   # Fall River, MA
        self.assertFalse(ri.contains(Point(-72.099, 41.355)))   # New London, CT


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region us-state test dependencies are not installed",
)
class CheckedInAssetTests(unittest.TestCase):
    """Criterion #7. The gate (V11) chose one merged asset over two, and
    measured that the two-file argument -- "the 11 features have a byte pin the
    merge would move" -- was false: there was no pin anywhere in the suite. So
    this phase adds the first one rather than moving one.
    """

    def _asset(self):
        from tta_backend.utils.plotting import _PRESET_REGIONS_PATH
        with open(_PRESET_REGIONS_PATH, encoding="utf-8") as fh:
            return _PRESET_REGIONS_PATH, json.load(fh)

    def test_the_asset_is_pinned_at_its_measured_size(self):
        """A boundary revision, a tolerance change or a Natural Earth mirror
        update all land here first, deliberately loudly. Re-measure and move
        both numbers *after* reading why they were what they were -- do not
        patch them to make a red test green."""
        path, fc = self._asset()

        self.assertEqual(len(fc["features"]), 62,
                         "11 pre-existing features + 51 U.S. admin-1 units")
        self.assertEqual(
            os.path.getsize(path), 293041,
            "preset_regions.geojson changed size; rebuild with "
            "scripts/build_preset_regions.py and re-measure deliberately",
        )

    def test_the_generated_table_and_the_asset_cannot_drift(self):
        """The one real cost of tension 3's plain-preset path: the bounding box
        lives in Python (so a missing asset degrades a state to a rectangle
        instead of dropping the key) while the boundary lives in GeoJSON. They
        are emitted by the same run from the same fetch, and this is what says
        they still agree -- exactly, not to a tolerance. A table whose box were
        a hair inside the polygon would put the retrieval extent inside the
        mask, which is the Risk 5 this phase closes."""
        from shapely.geometry import shape

        from tta_backend.datasets.us_states import US_STATES

        _, fc = self._asset()
        features = {f["id"]: f for f in fc["features"]}

        # Identified by the asset's own ``kind``, not by subtracting a list of
        # the other eleven ids -- that list would need editing every time a
        # coalition is added, and editing it to make a test pass is how the
        # drift this asserts against gets waved through.
        self.assertEqual(
            set(US_STATES),
            {f["id"] for f in fc["features"] if "kind" in f["properties"]},
        )
        for key, spec in US_STATES.items():
            with self.subTest(state=key):
                feature = features[key]
                self.assertEqual(tuple(shape(feature["geometry"]).bounds),
                                 tuple(spec["bounds"]))
                # The asset is self-describing without this codebase, the way
                # the OTR's properties are.
                self.assertEqual(feature["properties"]["postal"], spec["postal"])
                self.assertEqual(feature["properties"]["display_name"],
                                 spec["display_name"])

        postals = [s["postal"] for s in US_STATES.values()]
        self.assertEqual(len(set(postals)), 51)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region us-state test dependencies are not installed",
)
class RetrievalPlaneAgreementTests(unittest.TestCase):
    """Criterion #5, and the whole reason gate V13 was a gate.

    Phase 1.5 scoped ``dispatch_extent`` to ``COALITIONS`` u ``ALIASES`` and
    left plain preset keys to reach the MCP untranslated -- defensible for 39
    mostly-continental boxes. V13 measured what that costs once the vocabulary
    is 51 states, and the answer was not "51 times a small thing":

    - ``"new york"`` retrieved New York City (0.56% of the state) and
      ``"washington"`` retrieved Washington DC (**0.00%**).
    - Georgia and Pennsylvania retrieved the *correct* state and **still failed
      containment** -- 0.018 deg and 0.023 deg of mask outside the extent --
      because the mask is Natural Earth 50m and the extent was OSM.

    The second bullet is structural: while the two planes source geometry from
    different providers, containment cannot hold, and no geocoder improvement
    changes that. So the property asserted here is not "the numbers happen to
    line up" but "both planes derive from one polygon", and the seam through
    the MCP is covered separately in test_region_retrieval_extent.py.
    """

    def test_the_extent_contains_the_mask_for_every_one_of_the_fifty_one(self):
        """Containment, never equality -- the envelope of a boundary is a
        strict superset of it and that over-inclusion is correct, because the
        mask clips it (Phase 1.5, tension 3). Over all 51 so a state added or
        dropped by a future asset rebuild cannot skip it."""
        from shapely.geometry import box

        from tta_backend.datasets.us_states import US_STATES
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()

        for key in US_STATES:
            with self.subTest(state=key):
                dispatched = region_dispatch.dispatch_extent(
                    key, resolver.global_regions
                )
                self.assertTrue(dispatched.claimed,
                                f"{key!r} would reach the MCP as a place name")
                self.assertIsNotNone(dispatched.location)

                retrieved = box(*[float(v) for v in dispatched.location.split(",")])
                # The mask plane's own answer for the same string, so the two
                # planes are compared rather than the test reciting a constant.
                masked = resolver.resolve_location(key)["geometry"]

                self.assertTrue(
                    retrieved.contains(masked) or retrieved.equals(masked),
                    f"retrieval extent {retrieved.bounds} does not contain the "
                    f"mask {masked.bounds} for {key!r}",
                )

    def test_a_string_outside_the_vocabulary_is_still_not_claimed(self):
        """Criterion #6, which now gets 51 new chances to break. The seam is an
        *additional* door: "paris" and the 39 pre-existing presets must reach
        ``define_area_of_interest`` byte-identical to pre-T60 behavior."""
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        states = set(__import__(
            "tta_backend.datasets.us_states", fromlist=["US_STATES"]).US_STATES)

        unclaimed = ["paris", "new york city", "washington dc", "pennsylvania ave",
                     "georgia country", "-75,40,-74,41"]
        # ...and every pre-existing preset, which is where a wildcard would show.
        unclaimed += [k for k in resolver.global_regions if k not in states]

        for location in unclaimed:
            with self.subTest(location=location):
                self.assertFalse(
                    region_dispatch.dispatch_extent(
                        location, resolver.global_regions).claimed,
                    f"{location!r} was claimed; it must reach the MCP untouched",
                )

    def test_a_near_miss_string_still_reaches_the_geocoder_on_the_mask_plane(self):
        """Criterion #6's other half. The 51 keys are matched exactly, never by
        prefix or token overlap (D12) -- a confident wrong preset is worse than
        a geocode miss. ``"new york city"`` and ``"washington dc"`` are the
        strings a researcher types when they mean precisely the place D15 just
        took the bare token away from, and they must still get it."""
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        geo = {"latitude": 40.71, "longitude": -73.94, "display_name": "New York City",
               "polygon": None, "bbox": []}

        for location in ("new york city", "washington dc", "paris",
                         "pennsylvania avenue", "west virginia panhandle"):
            with self.subTest(location=location):
                with patch.object(resolver.geocoding_service, "geocode",
                                  return_value=geo) as geocode:
                    region = resolver.resolve_location(location)
                geocode.assert_called_once_with(location)
                self.assertEqual(region["region_type"], "point_buffer")

    def test_the_alias_collision_assertion_still_passes_with_the_states_added(self):
        """D12a, re-run against the 51 new keys. The gate measured zero
        collisions between state names and anything that already resolved --
        including ``"georgia"``, which the PRD worried about and which
        ``global_regions`` never had an entry for. This is what keeps that
        true."""
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        region_dispatch.assert_no_alias_collisions(resolver.global_regions)

        # 39 pre-existing presets + 51 states, none shadowing another.
        self.assertEqual(len(resolver.global_regions), 90)
        self.assertIn("georgia", resolver.global_regions)
        for coalition_id in region_dispatch.COALITIONS:
            self.assertNotIn(coalition_id, resolver.global_regions)

    def test_a_state_key_that_shadows_a_preset_raises_instead_of_overwriting(self):
        """D12a extended to the merge itself, and this is not hypothetical.

        The states are written into ``global_regions`` by assignment, so a key
        that already exists would be *silently replaced* -- and **Phase 4 is
        the phase that adds countries**, where ``"georgia"`` the country meets
        ``"georgia"`` the state head on. A count assertion catches that only
        until someone updates the count; a named error at construction says
        which key and why.

        Verified by mutation rather than assumed: without the guard the merge
        succeeds and ``"asia"`` silently becomes a U.S. state."""
        from tta_backend.datasets import us_states
        from tta_backend.utils.plotting import RegionResolver
        from tta_backend.utils.region_dispatch import AliasCollisionError

        shadowing = dict(us_states.US_STATES)
        shadowing["asia"] = dict(shadowing["alabama"], name="Asia")

        with patch.object(us_states, "US_STATES", shadowing), \
                patch("tta_backend.utils.plotting.US_STATES", shadowing):
            with self.assertRaises(AliasCollisionError) as caught:
                RegionResolver()

        self.assertIn("asia", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
