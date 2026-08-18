"""T60 Phase 3b: the ``+`` grammar, its vocabulary, and its error channel.

Phase 3a made the 51 states individually resolvable. This phase adds the
grammar that resolves *against* them -- ``"NY + NJ"`` -- and it needed two
things Phase 3a deliberately did not ship.

**The postal codes.** 3a held them because ``"DE"`` is also Germany. The 3b
gate (V17) measured that on live Nominatim and it is worse than assumed: five
of seven ambiguous codes resolve to a *foreign country* -- ``DE`` Germany,
``CA`` Canada, ``IN`` India, ``LA`` Laos, and ``ME`` **Montenegro**, which was
not even on the PRD's collision list. Every one of those is a correct answer to
a different question, so claiming bare postal codes would replace a right region
with a confident wrong one -- the D12b regression, re-run. Hence tension 2
option **(b)**: a postal code is a *composite member*, never a bare place name.

**The error channel.** D8 -- hard-fail naming the unresolved token -- is
unimplementable over ``dict | None``: ``"NY + NJ + Wakanda"`` collapses to
``"Could not resolve location: 'NY + NJ + Wakanda'"``, which never names
Wakanda. D14 makes it a raised ``MCPToolError`` instead.
"""
import importlib.util
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class CompositeUnionTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_ny_plus_nj_is_the_union_of_both_real_state_polygons(self):
        """The tracer bullet, and it asserts against *both* members rather
        than that a polygon came back.

        A parser that silently resolved only the first token would still
        return a real Natural Earth boundary with a plausible area; only
        comparing against the true union catches it. The area equality is the
        lock -- ``covers`` alone would pass on a shape twice the size."""
        from shapely.ops import unary_union

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = resolver.resolve_location("NY + NJ")

        geocode.assert_not_called()
        self.assertEqual(region["region_type"], "composite_union")

        ny = resolver.resolve_location("new york")["geometry"]
        nj = resolver.resolve_location("new jersey")["geometry"]
        expected = unary_union([ny, nj])

        self.assertTrue(region["geometry"].covers(ny), "New York is not covered")
        self.assertTrue(region["geometry"].covers(nj), "New Jersey is not covered")
        self.assertAlmostEqual(region["geometry"].area, expected.area, places=9)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class UnresolvedTokenTests(unittest.IsolatedAsyncioTestCase):
    """D8 + D14. The failure has to name the *token*, and it has to be a
    raised ``MCPToolError`` to do so -- over ``dict | None`` the whole
    composition collapses into one string that names nothing."""

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_a_bad_token_raises_naming_that_token_and_never_geocodes(self):
        """Both halves matter and only the second is easy to get wrong.

        ``assert_not_called`` is not decoration: a fail-open dispatcher that
        simply *missed* the composite would also return no geocoder hit in a
        test that only checked the error, because the raw string would be
        handed to Nominatim and Nominatim would answer. Only the call
        assertion distinguishes "refused" from "silently reinterpreted", and
        this bit Phase 1.

        The token assertion is on the **quoted** token, not on the substring.
        A message that merely echoes the raw request contains "wakanda" for
        free -- ``assertIn("wakanda", ...)`` passes on
        ``"'NY + NJ + Wakanda' contains a member that is not a U.S. state"``,
        which is precisely the pre-D14 answer this phase exists to replace.
        Found by mutating the message; the loose assertion did not notice."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            with self.assertRaises(MCPToolError) as caught:
                resolver.resolve_location("NY + NJ + Wakanda")

        geocode.assert_not_called()
        self.assertIn("'wakanda'", caught.exception.message.lower())

    async def test_the_async_resolver_refuses_the_same_way(self):
        """T42's standing bug was the two resolvers disagreeing. The analysis
        tools all use the async one, so a refusal that only the sync path
        raises is a refusal the researcher never sees."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "ageocode") as ageocode:
            with self.assertRaises(MCPToolError) as caught:
                await resolver.aresolve_location("NY + NJ + Wakanda")

        ageocode.assert_not_called()
        self.assertIn("'wakanda'", caught.exception.message.lower())


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class PostalVocabularyTests(unittest.IsolatedAsyncioTestCase):
    """Tension 2 option (b), measured into place by gate V17."""

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_every_postal_code_builds_the_same_region_as_its_full_name(self):
        """All 51, not a spot check, and compared as **whole objects**.

        The bite is that a postal hit must resolve *through* the canonical
        state key rather than assembling its own dict -- a second definition of
        what a region is will drift from the first. Comparing only
        ``region_type`` or only the geometry would miss a wrong
        ``display_name``, which is the field carrying the ``"(U.S. state)"``
        disambiguation V12 put there.

        **50 of 51 as of Phase 4.** ``"georgia"`` in full is now ambiguous
        between the state and the country and fails closed by design (D7, as
        Phase 4's V20 narrowed it), while ``"GA"`` still resolves -- so the
        equivalence this asserts is *deliberately* broken for exactly that one
        name, and D7's whole point is the asymmetry. The exclusion is derived
        from the same function the guard uses rather than hard-coded, so a
        second collision joins it automatically; both readings of Georgia are
        pinned in ``test_region_countries.py::GeorgiaAmbiguityTests``."""
        from tta_backend.datasets.us_states import US_STATES
        from tta_backend.utils.region_composition import ambiguous_member_tokens

        resolver = self._resolver()
        self.assertEqual(len(US_STATES), 51)
        ambiguous = ambiguous_member_tokens()
        self.assertEqual(ambiguous, {"georgia"},
                         "a new state/country collision appeared; D7's guard "
                         "now covers it and this exclusion grew with it")

        for key, state in US_STATES.items():
            if key in ambiguous:
                continue
            with self.subTest(state=key):
                postal = state["postal"]
                # The duplicate-member form, which D15 explicitly permits
                # ("duplicate and overlapping members simply union"). It runs
                # the full composite path while keeping the envelope equal to
                # the single state's own -- pairing every state against a
                # fixed partner instead made "AK + wyoming" span 2,253 deg^2
                # and trip D16's gate, testing the gate rather than the
                # vocabulary.
                by_code = resolver.resolve_location(f"{postal} + {postal}")
                by_name = resolver.resolve_location(f"{key} + {key}")
                self.assertEqual(by_code, by_name)
                # ...and the union of a state with itself is that state.
                self.assertTrue(
                    by_code["geometry"].equals(resolver.resolve_location(key)["geometry"])
                )

    def test_a_bare_postal_code_still_reaches_the_geocoder_untouched(self):
        """Tension 2 (b)'s whole content, and the reason it is (b).

        V17 measured ``"DE"`` resolving live to **Germany** -- a correct answer
        to a different question. Claiming it for Delaware would swap a right
        region for a confident wrong one under a name the user typed, which is
        the D12b regression re-run. Bare codes stay exactly as they are; the
        ``+`` is what supplies the missing context."""
        resolver = self._resolver()

        for token in ("DE", "CA", "IN", "LA", "ME", "NY"):
            with self.subTest(token=token):
                with patch.object(resolver.geocoding_service, "geocode") as geocode:
                    geocode.return_value = None
                    self.assertIsNone(resolver.resolve_location(token))
                geocode.assert_called_once_with(token)

    def test_the_split_happens_before_the_normalization_not_after(self):
        """D11a, requirement #5, and it reads as a stylistic choice but is not.

        ``_normalize_location_name`` strips only the **leading** ``"the "``. So
        under normalize-then-split, ``"the ny + nj"`` resolves and
        ``"ny + the nj"`` hard-fails on the token ``"the nj"`` -- one request,
        two spellings, two outcomes. The last two spellings below are the pair
        that catches it; the first three would pass either way.

        Asserted through **both** resolvers, because T42's standing bug was the
        two paths normalizing differently."""
        resolver = self._resolver()
        canonical = resolver.resolve_location("new york + new jersey")

        for spelling in ("ny+nj", "NY + NJ", "  the ny  +  nj ",
                         "ny + the nj", "the ny + the nj"):
            with self.subTest(spelling=spelling):
                self.assertEqual(resolver.resolve_location(spelling), canonical)

    async def test_both_resolvers_normalize_the_composite_identically(self):
        """The async half of requirement #5. Every analysis tool calls this
        one, so a divergence here is the one a researcher actually meets."""
        resolver = self._resolver()
        canonical = await resolver.aresolve_location("new york + new jersey")

        for spelling in ("ny+nj", "NY + NJ", "  the ny  +  nj ",
                         "ny + the nj", "the ny + the nj"):
            with self.subTest(spelling=spelling):
                self.assertEqual(await resolver.aresolve_location(spelling), canonical)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class ClosedVocabularyTests(unittest.TestCase):
    """D15's closed member vocabulary, and the malformed grammar."""

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_presets_and_coalitions_are_not_composite_members_by_design(self):
        """``"otc + ohio"`` and ``"conus + mexico"`` hard-fail **on purpose**,
        and this test exists so the next reader does not "fix" it.

        D15 keeps the member vocabulary closed and finite. Admitting presets
        would make every future alias a potential composite member and reopen
        the collision surface D6 closed by banning ISO country codes -- and
        ``"conus"`` and ``"otc"`` are both *already resolvable on their own*,
        so this is a restriction on combining them, not on reaching them.
        Revisit as a follow-on PRD, not by deleting this test."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        for raw, token in (("otc + ohio", "otc"), ("conus + mexico", "conus")):
            with self.subTest(raw=raw):
                with patch.object(resolver.geocoding_service, "geocode") as geocode:
                    with self.assertRaises(MCPToolError) as caught:
                        resolver.resolve_location(raw)
                geocode.assert_not_called()
                self.assertIn(f"'{token}'", caught.exception.message)

        # ...and both are still reachable on their own, unchanged.
        self.assertEqual(resolver.resolve_location("otc")["region_type"], "polygon")
        self.assertEqual(resolver.resolve_location("conus")["region_type"], "polygon")

    def test_a_malformed_composition_fails_as_malformed_not_as_a_bad_token(self):
        """Requirement #7. ``"NY +"`` has an *empty* member, which is a
        different mistake from ``"NY + Wakanda"`` and deserves a different
        answer -- reporting ``"'' is not a U.S. state"`` would send someone
        looking for a state they never typed."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        for raw in ("NY +", "+ NJ", "NY + + NJ", "+", "  +  "):
            with self.subTest(raw=raw):
                with patch.object(resolver.geocoding_service, "geocode") as geocode:
                    with self.assertRaises(MCPToolError) as caught:
                        resolver.resolve_location(raw)
                geocode.assert_not_called()
                message = caught.exception.message.lower()
                self.assertIn("empty", message)
                self.assertNotIn("is not a u.s. state", message)

    def test_a_string_with_no_plus_is_untouched(self):
        """Requirement #11. Phase 1.5 and 3a each guarded this; the parser
        gives it a brand-new way to break, because every location string in
        the system now passes through ``is_composite`` first."""
        resolver = self._resolver()

        # A preset, a coalition and a state: all still resolve without a geocode.
        self.assertEqual(resolver.resolve_location("pennsylvania")["region_type"], "polygon")
        self.assertEqual(resolver.resolve_location("otc")["region_type"], "polygon")
        self.assertEqual(
            resolver.resolve_location("northeast us")["region_type"], "bounding_box"
        )
        # ...and an ordinary place still reaches the geocoder, byte-identical.
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            geocode.return_value = None
            resolver.resolve_location("paris")
        geocode.assert_called_once_with("paris")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class ExtentGateTests(unittest.TestCase):
    """D16, on gate V15's measurements.

    Risk 6 is the thing being bounded: a sparse composite's envelope is not
    its footprint. ``"alaska + florida"`` unions two polygons into a box
    spanning 8.56 M native cells at TEMPO L3, and ``_crop_to_mask_footprint``
    crops to that envelope rather than per-part -- the render-path OOM this
    codebase already knows about (T50/T59), reachable in one typed string.

    The ceiling is **not a new tunable number.** V15 measured every candidate
    against the two ceilings this codebase already decided, and
    ``MAX_FRAME_NATIVE_CELLS`` was the only one that fits:

    | case                    | cells      | vs 4,000,000 | vs 1,000,000 |
    |-------------------------|-----------:|--------------|--------------|
    | ``NY + NJ``             |     88,731 | under        | under        |
    | bare ``hawaii``         |    389,240 | under        | under        |
    | bare ``alaska``         |  1,776,288 | under        | **OVER**     |
    | ``CA + NY``             |  1,218,220 | under        | **OVER**     |
    | ``alaska + florida``    |  8,563,388 | **OVER**     | **OVER**     |
    | all 51 unioned          | 10,857,189 | **OVER**     | **OVER**     |

    ``MAX_PLANE_NATIVE_CELLS`` is excluded **by measurement**, not by taste: it
    would refuse bare ``"alaska"``, a token Phase 3a just shipped as working,
    which is a gate with a hole in the opposite direction.
    """

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_a_continent_spanning_composite_refuses_naming_the_limit(self):
        """Requirement #9, both halves. The refusal must name the limit that
        *actually fired* -- a message quoting a number no code compared
        against is how a refusal misleads."""
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            with self.assertRaises(MCPToolError) as caught:
                resolver.resolve_location("alaska + florida")

        geocode.assert_not_called()
        self.assertEqual(caught.exception.category, CATEGORY_TOO_LARGE)
        self.assertIn(f"{MAX_FRAME_NATIVE_CELLS:,}", caught.exception.message)

    def test_the_gate_does_not_fire_on_what_v15_measured_as_acceptable(self):
        """The other side of the band, and the reason it is two-sided.

        A ceiling set too low is not a safe failure -- it withdraws working
        behaviour. ``"CA + NY"`` is the PRD's own sparse-union example and it
        must still resolve; bare ``"alaska"`` is the single 3a token nearest
        the ceiling and it must be untouched, or D16 has become a gate on
        Phase 3a rather than on compositions."""
        resolver = self._resolver()

        self.assertEqual(
            resolver.resolve_location("CA + NY")["region_type"], "composite_union"
        )
        self.assertEqual(
            resolver.resolve_location("CA + ME")["region_type"], "composite_union"
        )
        self.assertEqual(resolver.resolve_location("alaska")["region_type"], "polygon")
        self.assertEqual(resolver.resolve_location("hawaii")["region_type"], "polygon")

    def test_the_ceiling_is_derived_from_the_existing_constant_not_restated(self):
        """Tension 4. Two constants both meaning "too big" is how a refusal
        ends up naming a limit that is not the one that fired, so this phase
        adds no third number -- it derives the envelope ceiling from
        ``MAX_FRAME_NATIVE_CELLS`` through a documented cells/deg^2 anchor.

        This test fails if someone hard-codes the derived value, which is the
        drift it exists to prevent."""
        from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS
        from tta_backend.utils import region_composition

        self.assertAlmostEqual(
            region_composition.MAX_COMPOSITE_ENVELOPE_DEG2,
            MAX_FRAME_NATIVE_CELLS / region_composition.CELLS_PER_SQUARE_DEGREE,
            places=6,
        )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class ShapeProvenanceTests(unittest.TestCase):
    """D10a. ``region_type`` was carrying two orthogonal facts in one slot.

    ``composite_union`` is *shape provenance* -- this shape is a construction,
    not a named place. ``boundary_cells`` is *rasterization fidelity* -- the
    mask self-healed onto the cells the region touches. Both can be true at
    once, and ``apply_mask_region_type`` unconditionally overwrote the first
    with the second.

    Gate V16 confirmed this is the *likely* path, not a corner: every one of
    the six masking sites calls ``apply_mask_region_type`` **before**
    ``_build_provenance``, so the origin was lost before the researcher could
    ever see it. Without this test D10a is a refactor with no behaviour
    behind it.
    """

    def _coarse_grid(self):
        """Cell centers 30 deg apart, so nothing lands inside NY+NJ and the
        mask has to self-heal onto the cells the region touches."""
        import numpy as np
        import xarray as xr

        return xr.DataArray(
            np.arange(9, dtype=float).reshape(3, 3),
            coords={"lat": [10.0, 40.0, 70.0], "lon": [-110.0, -80.0, -50.0]},
            dims=["lat", "lon"],
        )

    def test_a_composite_survives_a_boundary_cells_self_heal_with_its_origin(self):
        from tta_backend.utils.plotting import (
            RegionResolver, apply_mask_region_type, mask_data_by_geometry,
        )

        resolver = RegionResolver()
        region = resolver.resolve_location("NY + NJ")
        self.assertEqual(region["region_type"], "composite_union")
        # **The constructor states its own origin, before any masking runs.**
        #
        # Added in Phase 5, from a mutation this class did not catch. 3b's own
        # mutation #3 deleted ``region_origin`` from the constructor *and* from
        # ``apply_mask_region_type`` together, and two tests died -- which read
        # as proof. The single-point mutant (constructor only) **survives every
        # assertion below**, because ``apply_mask_region_type`` does
        # ``setdefault("region_origin", region["region_type"])``: it *derives*
        # the origin from whatever the slot held, so it reconstructs
        # ``composite_union`` for free. The assertions below therefore prove
        # ``apply_mask_region_type``, which the sibling test already covers,
        # and not the composite constructor this class is named for.
        #
        # The behaviour was never actually unpinned -- the constructor-only
        # mutant is caught by
        # ``test_a_composite_chart_puts_both_disclosure_fields_on_the_wire``,
        # one file over. But a test whose docstring says "without this test
        # D10a is a refactor with no behaviour behind it" should be the test
        # that dies, and it was not.
        self.assertEqual(region["region_origin"], "composite_union")

        masked = mask_data_by_geometry(self._coarse_grid(), region["geometry"])
        apply_mask_region_type(masked, region)

        # The rasterization fact wins the slot it owns...
        self.assertEqual(region["region_type"], "boundary_cells")
        # ...and the shape's provenance is still there to disclose.
        self.assertEqual(region["region_origin"], "composite_union")

    def test_a_plain_polygon_keeps_its_own_origin_through_the_self_heal(self):
        """The generalisation, and the reason ``apply_mask_region_type`` is
        where the preservation lives rather than the composite constructor:
        Phase 5's ``buffer`` will need exactly this and should not have to
        remember to opt in."""
        from tta_backend.utils.plotting import (
            RegionResolver, apply_mask_region_type, mask_data_by_geometry,
        )

        resolver = RegionResolver()
        region = resolver.resolve_location("rhode island")
        self.assertEqual(region["region_type"], "polygon")

        masked = mask_data_by_geometry(self._coarse_grid(), region["geometry"])
        apply_mask_region_type(masked, region)

        self.assertEqual(region["region_type"], "boundary_cells")
        self.assertEqual(region["region_origin"], "polygon")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition test dependencies are not installed",
)
class RetrievalPlaneAgreementTests(unittest.TestCase):
    """Requirement #10, and tension 3 is not optional here.

    A coalition at least has a *key* the MCP might have heard of. A composite
    has nothing at all -- it is constructed per request, and D8 forbids the
    geocoder from ever seeing the string. So if ``dispatch_extent`` does not
    claim it, the retrieval plane has no fallback whatsoever: the AOI step
    would either fail outright or, worse, resolve ``"NY + NJ"`` to whatever
    Nominatim makes of it, after which the mask clips a cube that never
    covered the region and nothing says so (Risk 5).

    Asserted where the extent is decided, following Phase 3a's precedent --
    51 fake-MCP round trips at ~1.5 s each is a price 3a already declined, and
    the seam itself is proven separately in test_region_retrieval_extent.py.
    """

    def _cases(self):
        return ["NY + NJ", "ny+nj", "CA + NY", "CA + ME",
                "connecticut + rhode island", "TX + NM + OK", "AK + AK"]

    def test_the_retrieval_extent_contains_the_composite_mask(self):
        """Containment, never equality. The envelope of a union is a strict
        superset of it, and that over-inclusion is correct because the mask
        clips it -- retrieval must contain the mask, not equal it (Phase 1.5,
        tension 3; re-proved over 51 states in 3a)."""
        from shapely.geometry import box

        from tta_backend.utils import region_composition
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        for raw in self._cases():
            with self.subTest(composite=raw):
                dispatched = region_composition.dispatch_composite_extent(raw, resolver)
                self.assertTrue(dispatched.claimed,
                                f"{raw!r} would reach the MCP as a place name")
                self.assertIsNotNone(dispatched.location)

                retrieved = box(*[float(v) for v in dispatched.location.split(",")])
                # The mask plane's own answer for the same string, so the two
                # planes are compared rather than the test reciting a constant.
                masked = resolver.resolve_location(raw)["geometry"]
                self.assertTrue(
                    retrieved.contains(masked) or retrieved.equals(masked),
                    f"retrieval extent {retrieved.bounds} does not contain the "
                    f"mask {masked.bounds} for {raw!r}",
                )

    def test_the_postal_table_stays_out_of_global_regions(self):
        """Requirement #13, and it is the structural half of tension 2 (b).

        Phase 3a added ``AliasCollisionError`` precisely because a new name
        table merging into ``global_regions`` by assignment would silently
        overwrite. A postal table is the next such table, and option (b) keeps
        it out of ``global_regions`` entirely -- so the guard stays green *by
        construction* rather than by luck.

        That matters beyond the guard: ``in``, ``or``, ``me``, ``de`` and
        ``la`` as bare region names would be 51 two-letter landmines waiting
        for a future preset or alias to collide with. This test fails the
        moment someone "simplifies" (b) into (a) without revisiting V17."""
        from tta_backend.utils import region_composition, region_dispatch
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        self.assertEqual(len(region_composition.POSTAL_TO_STATE), 51)
        for code in region_composition.POSTAL_TO_STATE:
            with self.subTest(postal=code):
                self.assertNotIn(code, resolver.global_regions)
                self.assertNotIn(code, region_dispatch.ALIASES)
                self.assertNotIn(code, region_dispatch.COALITIONS)

        # The guard itself still passes with everything this phase added.
        region_dispatch.assert_no_alias_collisions(resolver.global_regions)
        self.assertEqual(len(resolver.global_regions), 90)

    def test_a_non_composite_is_not_claimed_by_the_composite_extent_gate(self):
        """The mirror of requirement #11 on the retrieval plane. The composite
        gate is an *additional* door; "paris", a preset and a state must all
        reach ``dispatch_extent`` exactly as they did before."""
        from tta_backend.utils import region_composition
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        for raw in ("paris", "otc", "pennsylvania", "northeast us"):
            with self.subTest(location=raw):
                dispatched = region_composition.dispatch_composite_extent(raw, resolver)
                self.assertFalse(dispatched.claimed)
                self.assertIsNone(dispatched.location)
