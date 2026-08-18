"""T60 Phase 5: the ``"within N miles|km of X"`` grammar (D9).

**Why the area assertion the PRD prescribes is not the headline test here.**
The Phase 5 gate (V22) measured an AEQD buffer at 20 N, 179.5 E: its geodesic
area is 31,412.1 km^2, *99.99% of pi r^2*, and the polygon is the **complement**
of the buffer -- a point 10 km east reads ``False``, a point 10 km *west* reads
``False``, and (0 E, 20 N) off West Africa reads ``True``. ``pyproj.Geod`` walks
the ring and the ring is correct; only shapely's planar reading of it wraps the
long way around the planet. So an assertion of the form ``area ~= pi r^2`` pins
a property the bug *preserves*, and the D16 extent gate does not catch it
either (650.4 deg^2, under the 2,148.7 ceiling).

The test that earns its place is therefore a **containment** assertion against
points at a known bearing and geodesic distance from the centre. The area
invariant ships too -- it is what catches a degree-buffer regression (D9) -- but
it is not what proves the shape.

Network-free: the geocoder is faked at the ``GeocodingService`` seam, the same
way ``test_geocoding_service.py`` does it.
"""
import importlib.util
import math
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine", "pyproj"]

# Nominatim's own shape for a hit, so the fake exercises the same keys the real
# geocoder returns. A ``polygon`` of ``None`` is deliberate: what the buffer
# grammar needs from ``X`` is a *point* (D9), so a geocoder hit with no boundary
# is the ordinary case, not a degraded one.
NYC = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "display_name": "New York, United States",
    "polygon": None,
    "bbox": [40.4774, 40.9176, -74.2591, -73.7002],
}

MILES_IN_METRES = 1609.344


def _dest(lat, lon, bearing_deg, metres):
    """A point at a known geodesic bearing and distance -- the ground truth the
    containment assertions are made against, computed by ``Geod`` rather than
    by any code under test."""
    from pyproj import Geod

    lon2, lat2, _ = Geod(ellps="WGS84").fwd(lon, lat, bearing_deg, metres)
    return lon2, lat2


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class BufferShapeTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_within_50_miles_of_nyc_covers_points_at_a_known_bearing(self):
        """The tracer bullet, and it is a containment test on purpose.

        Four bearings at 0.99r must be inside and the same four at 1.01r must
        be outside. That is what fails on the dateline polygon (which passes
        the area test), and it is what fails on a degree-buffer at any latitude
        away from the equator -- 40.7 N shrinks a naive longitude degree by
        ``cos(lat)``, so the east/west points land outside a shape whose area
        might still look plausible."""
        from shapely.geometry import Point

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=NYC):
            region = resolver.resolve_location("within 50 miles of NYC")

        self.assertEqual(region["region_type"], "buffer")
        self.assertEqual(region["region_origin"], "buffer")

        radius = 50 * MILES_IN_METRES
        geometry = region["geometry"]
        for bearing in (0, 90, 180, 270):
            inside = Point(*_dest(NYC["latitude"], NYC["longitude"], bearing, radius * 0.99))
            outside = Point(*_dest(NYC["latitude"], NYC["longitude"], bearing, radius * 1.01))
            self.assertTrue(
                geometry.covers(inside),
                f"a point 49.5 miles out on bearing {bearing} is not covered",
            )
            self.assertFalse(
                geometry.covers(outside),
                f"a point 50.5 miles out on bearing {bearing} is covered",
            )

    def test_the_same_physical_radius_in_km_and_miles_is_one_polygon(self):
        """D16. Asserted as *geometry* equality, not as two areas that match.

        Two areas agreeing is a much weaker claim: the dateline polygon in V22
        has the right area and the wrong shape, so an area-only comparison
        between two broken shapes would agree with itself. Symmetric difference
        and Hausdorff distance both compare the boundaries point for point."""
        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=NYC):
            miles = resolver.resolve_location("within 50 miles of NYC")["geometry"]
            km = resolver.resolve_location("within 80.4672 km of NYC")["geometry"]

        residue = miles.symmetric_difference(km).area
        self.assertLess(
            residue / miles.area, 1e-9,
            f"50 miles and 80.4672 km disagree by {residue / miles.area:.3e} of the area",
        )
        self.assertLess(miles.hausdorff_distance(km), 1e-9)

    def test_a_buffer_at_10n_and_at_60n_have_the_same_area(self):
        """The PRD's cos-latitude invariant, and the one number that catches a
        degree-buffer regression.

        Measured in the gate: a ``Point.buffer(100/111.32)`` retains 97.8% of
        the intended area at 10 N and **50.2%** at 60 N, because a longitude
        degree shrinks by ``cos(latitude)``. The AEQD approach holds 99.99% at
        both. The tolerance below is tight enough that the naive
        implementation cannot pass it -- 50.2% versus 97.8% is a factor of
        1.95, against a 0.1% band."""
        from pyproj import Geod

        resolver = self._resolver()
        areas = {}
        for latitude in (10.0, 60.0):
            hit = dict(NYC, latitude=latitude, longitude=0.0)
            with patch.object(resolver.geocoding_service, "geocode", return_value=hit):
                geometry = resolver.resolve_location("within 100 km of somewhere")["geometry"]
            areas[latitude] = abs(
                Geod(ellps="WGS84").geometry_area_perimeter(geometry)[0]
            ) / 1e6

        self.assertAlmostEqual(areas[10.0], areas[60.0], delta=areas[10.0] * 1e-3)
        # ...and both are the *right* area, not merely equal to each other:
        # two identically-wrong shapes would satisfy the comparison above.
        expected = math.pi * 100**2
        for latitude, area in areas.items():
            self.assertAlmostEqual(
                area, expected, delta=expected * 1e-3,
                msg=f"a 100 km buffer at {latitude} N has area {area:,.1f} km^2",
            )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class AntimeridianTests(unittest.IsolatedAsyncioTestCase):
    """V22, and the reason this phase needed a gate.

    A 100 km buffer at 20 N, 179.5 E has a geodesic area of 31,412.1 km^2 --
    99.99% of pi r^2 -- and shapely reads its ring as the whole planet *except*
    the buffer. The envelope is 650.4 deg^2, under the 2,148.7 ceiling, so D16
    passes it. Neither the area assertion nor the extent gate can see this.

    The verdict is **refuse**, option (a). Option (b) -- split at +/-180 into a
    MultiPolygon -- was measured and produces a correct mask behind a bounding
    box of ``-180,19.10,180,20.90``: **1,210,757 cells retrieved for a 6,052
    cell region**, 200x its own footprint, which the gate still cannot catch.
    That is Risk 6 verbatim, and Phase 4 already refuses the USA, Russia, Fiji,
    New Zealand, Kiribati and Antarctica as composite members for exactly this
    property. Option (b) also does not fix the pole.
    """

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def _resolve(self, latitude, longitude, phrase):
        resolver = self._resolver()
        hit = dict(NYC, latitude=latitude, longitude=longitude)
        with patch.object(resolver.geocoding_service, "geocode", return_value=hit):
            return resolver.resolve_location(phrase)

    def test_a_buffer_crossing_the_antimeridian_is_refused_naming_it(self):
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._resolve(20.0, 179.5, "within 100 km of somewhere")

        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
        text = f"{caught.exception.message} {caught.exception.suggestion}".lower()
        self.assertIn("antimeridian", text)
        # The refusal has to say what it refused, or it reads as a tool
        # malfunction rather than an answer about this request.
        self.assertIn("100 km", text)

    def test_no_buffer_the_grammar_returns_ever_covers_a_point_a_world_away(self):
        """The shape invariant, stated over the grammar rather than over one
        polygon -- which is what makes it survive the *refuse* decision.

        Either the request is refused, or the geometry it returns is a real
        disc: everything at 0.99r inside, nothing at 3r outside. The naive
        implementation (no antimeridian guard) fails on the third row -- it
        returns a polygon that covers a point 300 km away, covers (0 E, 20 N)
        off West Africa, and does *not* cover a point 10 km east of its own
        centre.
        """
        from shapely.geometry import Point
        from tta_backend.earthdata_mcp.results import MCPToolError

        cases = [
            ("NYC, 50 miles", 40.7128, -74.0060, "within 50 miles of x", 50 * MILES_IN_METRES),
            ("Suva, Fiji, 100 km", -18.14, 178.44, "within 100 km of x", 100_000),
            ("20 N 179.5 E, 100 km", 20.0, 179.5, "within 100 km of x", 100_000),
            ("Anadyr, 250 km", 64.73, 177.51, "within 250 km of x", 250_000),
            ("Nome, Alaska, 250 km", 64.50, -165.41, "within 250 km of x", 250_000),
        ]
        refused = []
        for label, latitude, longitude, phrase, radius in cases:
            try:
                geometry = self._resolve(latitude, longitude, phrase)["geometry"]
            except MCPToolError:
                refused.append(label)
                continue
            for bearing in (0, 90, 180, 270):
                near = Point(*_dest(latitude, longitude, bearing, radius * 0.99))
                far = Point(*_dest(latitude, longitude, bearing, radius * 3))
                self.assertTrue(geometry.covers(near), f"{label}: {bearing} deg at 0.99r missing")
                self.assertFalse(geometry.covers(far), f"{label}: {bearing} deg at 3r covered")
            self.assertFalse(
                geometry.covers(Point(0.0, 20.0)),
                f"{label}: covers (0 E, 20 N), half a world from its centre",
            )

        # Exactly the two the gate measured as crossing -- not "at least one",
        # which a blanket refusal of every buffer would also satisfy.
        self.assertEqual(refused, ["20 N 179.5 E, 100 km", "Anadyr, 250 km"])


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class PolarTests(unittest.IsolatedAsyncioTestCase):
    """V22, decided separately from the antimeridian because it is a different
    failure and option (b) does not fix it.

    A 200 km buffer at 89.5 N physically contains the North Pole -- the pole is
    55.8 km away. Measured, the AEQD polygon's **maximum latitude is 88.7094**:
    going 200 km north passes over the pole and comes back down the far
    meridian, so the ring never reaches the pole in a lat/lon frame. The
    polygon does not cover the pole and **does not cover its own centre**. The
    region actually wanted is a polar cap with a hole at the top, which is not
    expressible as a lat/lon polygon at all.
    """

    def _resolve(self, latitude, phrase):
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        hit = dict(NYC, latitude=latitude, longitude=0.0)
        with patch.object(resolver.geocoding_service, "geocode", return_value=hit):
            return resolver.resolve_location(phrase)

    def test_a_buffer_containing_a_pole_is_refused_naming_the_pole(self):
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._resolve(89.5, "within 200 km of somewhere")

        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
        text = f"{caught.exception.message} {caught.exception.suggestion}".lower()
        self.assertIn("pole", text)
        # The pole check must run *before* the antimeridian check: this ring
        # wraps 360 degrees of longitude too, so a message naming the
        # antimeridian would be true and would send the reader to the wrong
        # fix -- no radius shrink helps if you are standing on the pole, and
        # moving off the antimeridian does nothing at all.
        self.assertNotIn("antimeridian", text)

    def test_a_high_latitude_buffer_that_misses_the_pole_still_resolves(self):
        """The precision that makes the refusal honest: this is not "high
        latitudes are refused". At 85 N the pole is 558.5 km away, so a 200 km
        buffer is an ordinary request and stays one -- even though its
        longitude span is 20.6 degrees, ten times NYC's."""
        from pyproj import Geod

        region = self._resolve(85.0, "within 200 km of somewhere")
        self.assertEqual(region["region_type"], "buffer")
        area = abs(Geod(ellps="WGS84").geometry_area_perimeter(region["geometry"])[0]) / 1e6
        self.assertAlmostEqual(area, math.pi * 200**2, delta=math.pi * 200**2 * 1e-3)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class BufferRefusalTests(unittest.IsolatedAsyncioTestCase):
    def _resolve(self, phrase, hit=NYC):
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=hit) as geocode:
            self.geocode = geocode
            return resolver.resolve_location(phrase)

    def test_within_3000_miles_of_nyc_refuses_with_the_named_limit(self):
        """D16. The PRD's own example, and the gate measured it at 11,270.2
        deg^2 = 20,980,751 cells against a 4,000,000-cell limit."""
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("within 3000 miles of NYC")

        self.assertEqual(caught.exception.category, CATEGORY_TOO_LARGE)
        # A refusal quoting a number nothing compared against is how a message
        # ends up misleading its reader -- the same rule ``_too_large`` follows.
        self.assertIn(f"{MAX_FRAME_NATIVE_CELLS:,}", caught.exception.message)

    def test_a_modest_buffer_does_not_trip_the_extent_gate(self):
        """The other half, without which the test above passes on a gate that
        refuses everything. 500 miles is 276.5 deg^2, 13% of the ceiling."""
        self.assertEqual(
            self._resolve("within 500 miles of NYC")["region_type"], "buffer"
        )

    def test_a_bare_number_refuses_naming_both_units(self):
        """Tension 3, decided against a silent default: 50 miles is 80.5 km,
        so guessing is a 61% error for half of users, and D12b is the standing
        precedent against replacing an unknown with a confident guess."""
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("within 50 of NYC")

        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
        text = f"{caught.exception.message} {caught.exception.suggestion}".lower()
        self.assertIn("miles", text)
        self.assertIn("km", text)
        # Named, not defaulted: the refusal must not have quietly resolved.
        self.assertEqual(self.geocode.call_count, 0)

    def test_a_buffer_around_a_named_region_refuses_rather_than_geocoding_it(self):
        """V25, and D9's "X resolves only as a single geocoded point, never
        recursively through PRESET/COMPOSITE".

        Unclaimed, ``"within 50 km of otc"`` geocodes ``"otc"``, which the
        Phase 0 gate measured resolving live to **an aerodrome in Chad**. It
        would ship as ``region_type: buffer``, faithfully disclosed, and be
        wrong by a continent -- D12b's failure mode wearing a new label."""
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        for phrase, token in [
            ("within 50 km of otc", "otc"),
            ("within 50 km of new york", "new york"),
            ("within 50 km of france", "france"),
            ("within 50 km of north america", "north america"),
        ]:
            with self.subTest(phrase=phrase):
                with self.assertRaises(MCPToolError) as caught:
                    self._resolve(phrase)
                self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
                self.assertIn(token, caught.exception.message)
                self.assertEqual(self.geocode.call_count, 0)

    def test_a_city_is_still_an_ordinary_buffer_target(self):
        """The other half of the check above: all 90 ``global_regions`` keys
        are regions, not one city among them, so refusing the vocabulary costs
        nothing a researcher would actually type."""
        for phrase in ("within 50 km of paris", "within 50 km of nyc",
                       "within 50 km of chicago"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self._resolve(phrase)["region_type"], "buffer")

    def test_an_unresolvable_target_refuses_and_never_falls_back_to_the_phrase(self):
        """D8's rule, applied to this grammar: once a string is syntactically a
        buffer the geocoder is off the table for the *phrase*. Gate V23
        measured that ``"within 50 miles of NYC"`` returns **zero** Nominatim
        hits, and T46 Phase 2's V6 measured what the agent does next -- it
        silently substitutes a different region and retrieves that."""
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("within 50 km of wakanda", hit=None)

        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
        self.assertIn("wakanda", caught.exception.message)
        # The target, once -- never the whole phrase as a place name.
        self.geocode.assert_called_once_with("wakanda")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class SyncAsyncForkTests(unittest.IsolatedAsyncioTestCase):
    """D11b, and gate V24 verified all three surfaces against the live code:
    ``export_service`` calls the sync ``resolve_location``; ``stat_tools``,
    ``plot_tools`` and ``validation_tools`` call the async one exclusively; and
    the retrieval-plane wrapper's ``_call`` is itself ``async def``.

    A single blocking twin would put ``requests.get(timeout=15)`` on the event
    loop, which is the hazard the split exists to prevent."""

    async def test_the_async_resolver_builds_the_same_buffer_via_ageocode(self):
        from unittest.mock import AsyncMock

        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        with patch.object(resolver.geocoding_service, "ageocode",
                          AsyncMock(return_value=NYC)) as ageocode, \
             patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = await resolver.aresolve_location("within 50 miles of NYC")

        ageocode.assert_awaited_once_with("nyc")
        # The blocking one must not be reachable from the event loop at all.
        geocode.assert_not_called()
        self.assertEqual(region["region_type"], "buffer")

        # ...and byte-for-byte the same shape the sync twin builds, which is
        # what "only the geocode forks" has to mean to be worth claiming.
        sync_resolver = RegionResolver()
        with patch.object(sync_resolver.geocoding_service, "geocode", return_value=NYC):
            twin = sync_resolver.resolve_location("within 50 miles of NYC")
        self.assertTrue(region["geometry"].equals_exact(twin["geometry"], 0.0))
        self.assertEqual(region["display_name"], twin["display_name"])

    async def test_the_async_resolver_refuses_by_the_same_rules(self):
        """The fork is one line wide, so every check must be reachable from
        both twins. A refusal that only the sync path enforces is a hole the
        analysis tools -- which use the async path exclusively -- fall into."""
        from unittest.mock import AsyncMock

        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.utils.plotting import RegionResolver

        cases = [
            ("within 50 of NYC", NYC),
            ("within 3000 miles of NYC", NYC),
            ("within 50 km of otc", NYC),
            ("within 100 km of x", dict(NYC, latitude=20.0, longitude=179.5)),
            ("within 200 km of x", dict(NYC, latitude=89.5, longitude=0.0)),
        ]
        for phrase, hit in cases:
            with self.subTest(phrase=phrase, latitude=hit["latitude"]):
                resolver = RegionResolver()
                with patch.object(resolver.geocoding_service, "ageocode",
                                  AsyncMock(return_value=hit)):
                    with self.assertRaises(MCPToolError):
                        await resolver.aresolve_location(phrase)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class BufferOriginTests(unittest.TestCase):
    """D10a. ``region_type`` is rasterization fidelity, ``region_origin`` is
    shape provenance, and a small buffer on a coarse grid is the *likely* path
    to the self-heal, not a corner -- which is precisely the case D10a was
    written for."""

    def _coarse_grid(self):
        """Cell centres 30 deg apart, so a 50-mile buffer lands entirely
        between them and the mask must self-heal onto the cells it touches."""
        import numpy as np
        import xarray as xr

        return xr.DataArray(
            np.arange(9, dtype=float).reshape(3, 3),
            coords={"lat": [10.0, 40.0, 70.0], "lon": [-110.0, -80.0, -50.0]},
            dims=["lat", "lon"],
        )

    def test_a_buffer_survives_a_boundary_cells_self_heal_with_its_origin(self):
        from tta_backend.utils.plotting import (
            RegionResolver, apply_mask_region_type, mask_data_by_geometry,
        )

        resolver = RegionResolver()
        with patch.object(resolver.geocoding_service, "geocode", return_value=NYC):
            region = resolver.resolve_location("within 50 miles of NYC")
        self.assertEqual(region["region_type"], "buffer")
        # **The constructor states its own origin, before any masking runs.**
        #
        # Found by mutation, and it is the assertion that distinguishes this
        # test from one pinned by accident. Deleting ``region_origin`` from the
        # constructor leaves the two assertions below still passing, because
        # ``apply_mask_region_type`` does ``setdefault("region_origin",
        # region["region_type"])`` -- it *derives* the origin from whatever the
        # slot held. So a self-heal assertion alone cannot tell a constructor
        # that declares its provenance from one that forgot to; it proves
        # ``apply_mask_region_type``, which 3b already proved. This line is
        # what makes the test prove D10a for the buffer path.
        self.assertEqual(region["region_origin"], "buffer")

        masked = mask_data_by_geometry(self._coarse_grid(), region["geometry"])
        apply_mask_region_type(masked, region)

        # The rasterization fact wins the slot it owns...
        self.assertEqual(region["region_type"], "boundary_cells")
        # ...and the shape's provenance is still there to disclose.
        self.assertEqual(region["region_origin"], "buffer")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region buffer test dependencies are not installed",
)
class RetrievalPlaneAgreementTests(unittest.IsolatedAsyncioTestCase):
    """V23. The retrieval plane must claim a buffer, and the argument is
    stronger here than for anything T60 has shipped before.

    A coalition has a name the MCP's geocoder might resolve, badly. A composite
    has no key but does contain real place names. A buffer phrase has
    **nothing** -- measured live in the gate, ``"within 50 miles of NYC"``
    returns *zero* Nominatim hits. Unclaimed, the AOI step simply fails, and
    T46 Phase 2's V6 measured what happens next: the agent silently substitutes
    a different region and retrieves that, after which the mask clips a cube
    that never covered the buffer (Risk 5).
    """

    async def _extent(self, phrase, hit=NYC):
        from unittest.mock import AsyncMock

        from tta_backend.utils import region_buffer
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        with patch.object(resolver.geocoding_service, "ageocode",
                          AsyncMock(return_value=hit)):
            return await region_buffer.adispatch_buffer_extent(phrase, resolver), resolver

    async def test_the_retrieval_extent_contains_the_buffer_mask(self):
        """Containment, never equality -- retrieval must *contain* the mask
        (Phase 1.5's invariant, re-proved over 51 states in 3a and over the
        composites in 3b). Because V22 refuses the wrapping cases, every buffer
        that reaches here has an honest bounding box and the containment is
        exact by construction rather than by tolerance."""
        from unittest.mock import AsyncMock

        from shapely.geometry import box

        from tta_backend.utils.plotting import RegionResolver

        cases = [
            ("within 50 miles of NYC", NYC),
            ("within 50 km of NYC", NYC),
            ("within 500 miles of NYC", NYC),
            ("within 100 km of x", dict(NYC, latitude=-18.14, longitude=178.44)),
            ("within 200 km of x", dict(NYC, latitude=85.0, longitude=0.0)),
        ]
        for phrase, hit in cases:
            with self.subTest(phrase=phrase, latitude=hit["latitude"]):
                dispatched, _ = await self._extent(phrase, hit)
                self.assertTrue(dispatched.claimed,
                                f"{phrase!r} would reach the MCP as a place name")
                self.assertIsNotNone(dispatched.location)
                self.assertNotIn("within", dispatched.location)

                retrieved = box(*[float(v) for v in dispatched.location.split(",")])
                # The mask plane's own answer for the same string, so the two
                # planes are compared rather than the test reciting a constant.
                resolver = RegionResolver()
                with patch.object(resolver.geocoding_service, "ageocode",
                                  AsyncMock(return_value=hit)):
                    masked = (await resolver.aresolve_location(phrase))["geometry"]
                self.assertTrue(
                    retrieved.contains(masked) or retrieved.equals(masked),
                    f"retrieval extent {retrieved.bounds} does not contain the "
                    f"mask {masked.bounds} for {phrase!r}",
                )

    async def test_a_refused_buffer_is_refused_on_the_retrieval_plane_too(self):
        """The failures have to fire *here*, not just on the mask plane. A
        buffer refused for the mask and retrieved anyway would pull a cube for
        a region the analysis then declines to describe."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        for phrase, hit in [
            ("within 3000 miles of NYC", NYC),
            ("within 50 of NYC", NYC),
            ("within 50 km of otc", NYC),
            ("within 100 km of x", dict(NYC, latitude=20.0, longitude=179.5)),
            ("within 200 km of x", dict(NYC, latitude=89.5, longitude=0.0)),
        ]:
            with self.subTest(phrase=phrase, latitude=hit["latitude"]):
                with self.assertRaises(MCPToolError):
                    await self._extent(phrase, hit)

    async def test_a_string_outside_the_grammar_is_not_claimed(self):
        """The regression this seam is most likely to cause. ``"paris"`` and
        ``"otc"`` must reach the rest of the dispatch chain byte-identical."""
        for phrase in ("paris", "otc", "NY + NJ", "conus", "within reach"):
            with self.subTest(phrase=phrase):
                dispatched, resolver = await self._extent(phrase)
                self.assertFalse(dispatched.claimed)
