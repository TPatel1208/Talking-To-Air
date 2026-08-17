"""T60 Phase 4: countries as composite members.

Phase 3b built the ``+`` grammar over one member vocabulary (the 51 U.S.
states). This phase adds the second, and the Phase 4 gate found that three
properties the design had been relying on stop holding at country scale.

**The tolerance.** ``ADMIN1_TOLERANCE`` holds all 51 states inside a
99.64-100.24% band and retains **33.2% of Vatican City** (V18). A uniform
tolerance is refuted; the shipped scheme is proportional to each feature's own
linear scale.

**The collisions.** ``georgia`` and ``antarctica`` are both a shipped preset id
and an ``ADMIN`` value, so a *merged* asset would silently reduce 304 features
to 302 polygons and mask the Caucasus under the name ``"georgia"`` (V19). Hence
a separate asset, and countries as **member-only** vocabulary: a bare country
name is not claimed by this phase at all.

**The extent.** Ten of 242 countries exceed the composite ceiling on their own,
including France -- whose envelope is 119x its land area because of the
overseas departments (V21). ``"France + Germany"`` therefore refuses, and the
refusal has to say *why*, because 3b's "ask for nearer members" advice is
actively wrong for two countries that share a border.
"""
import importlib.util
import unittest
from unittest.mock import patch


REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio", "affine"]

# Independent of the shipped asset on purpose: real cities, so a test that
# "both members are present" cannot be satisfied by the asset agreeing with
# itself.
#
# All three are **inland**. Lisbon was the first choice for Portugal and is
# wrong: it sits on the Tagus estuary, and the 50m boundary puts it 0.0197 deg
# offshore, so a correct union fails a Lisbon check. A coastal city is a test of
# the coastline's simplification, not of whether the member resolved.
MADRID = (-3.703, 40.417)
COIMBRA = (-8.420, 40.211)
PARIS = (2.352, 48.857)
ATLANTA = (-84.388, 33.749)      # Georgia, the U.S. state
LIBREVILLE = (9.454, 0.392)      # Gabon, whose ISO alpha-2 code is GA
TBILISI = (44.783, 41.716)       # Georgia, the country


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class CountryMemberTests(unittest.TestCase):
    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_spain_plus_portugal_is_the_union_of_both_real_country_boundaries(self):
        """The tracer bullet: ``ADMIN`` matching, unioned, asserted against
        *both* members.

        The three points are the whole test. A parser that resolved only the
        first token returns Spain -- a real Natural Earth boundary of plausible
        area that covers Madrid and misses Lisbon. One that fell back to a
        preset or a geocode would likely cover Paris. Area alone catches
        neither."""
        from shapely.geometry import Point

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = resolver.resolve_location("Spain + Portugal")

        geocode.assert_not_called()
        self.assertEqual(region["region_type"], "composite_union")

        geometry = region["geometry"]
        self.assertTrue(geometry.covers(Point(*MADRID)), "Spain is not covered")
        self.assertTrue(geometry.covers(Point(*COIMBRA)), "Portugal is not covered")
        self.assertFalse(geometry.covers(Point(*PARIS)), "France leaked in")

    def test_ga_resolves_as_the_us_state_and_never_as_gabon(self):
        """D6: **country tokens never match on an ISO code**, only on a name.

        ``GA`` is simultaneously Georgia's USPS code and Gabon's ISO alpha-2
        code -- one of the 26 such collisions Verified Finding 5 measured. The
        design does not resolve that by preferring the state; it resolves it by
        never admitting codes into the country tier at all, so there is no
        second candidate to prefer over. The state tier running first is a
        tiebreak of last resort, not the mechanism.

        Atlanta being covered is what proves ``GA`` took the state reading:
        had it taken Gabon, the union would be Gabon + Portugal and Atlanta
        would fall outside it."""
        from shapely.geometry import Point

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = resolver.resolve_location("GA + Portugal")

        geocode.assert_not_called()
        geometry = region["geometry"]
        self.assertTrue(geometry.covers(Point(*ATLANTA)),
                        "GA did not resolve to the U.S. state")
        self.assertTrue(geometry.covers(Point(*COIMBRA)), "Portugal is not covered")
        self.assertFalse(geometry.covers(Point(*LIBREVILLE)),
                         "GA matched Gabon's ISO code")

    def test_a_two_letter_code_that_is_not_a_state_resolves_to_nothing(self):
        """D6's ban, asserted where it is actually load-bearing.

        **Found by mutation, not by design.** The ``GA``/Gabon test above looks
        like it pins "countries never match on a code", and it does not: every
        token that is *both* a postal code and an ISO code hits the state tier
        first and never reaches the country lookup at all. A mutant that added
        an ISO map to the country tier passed the entire suite.

        ``GB`` is the case with no state to shadow it -- not a USPS code, but
        the ISO alpha-2 for the United Kingdom, which is in the vocabulary
        under ``united kingdom`` and reachable by four spellings. If codes ever
        leak into the country tier, this is the test that notices. ``ZA``
        (South Africa) and ``FR`` (France) are here for the same reason."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        for code in ("GB", "ZA", "FR", "JP"):
            with self.subTest(code=code):
                with self.assertRaises(MCPToolError) as caught:
                    resolver.resolve_location(f"{code} + spain")
                self.assertIn(f"'{code.lower()}'",
                              caught.exception.message.lower())

    def test_a_composite_of_countries_does_not_call_itself_a_composite_of_states(self):
        """``display_name`` is what a T42 answer cites, so it cannot be left
        saying "composite of 2 U.S. states" about Spain and Portugal.

        3b hard-coded that phrase because the member vocabulary was the 51
        states and nothing else. Adding a tier without touching the label is
        how a disclosure quietly starts lying -- the region would be right and
        the sentence describing it wrong, which T42 exists to prevent."""
        resolver = self._resolver()

        countries = resolver.resolve_location("Spain + Portugal")["display_name"]
        self.assertNotIn("U.S. state", countries, countries)
        self.assertIn("countries", countries, countries)

        states = resolver.resolve_location("NY + NJ")["display_name"]
        self.assertIn("U.S. states", states, states)

        mixed = resolver.resolve_location("NY + Portugal")["display_name"]
        self.assertIn("2 members", mixed, mixed)

    def test_an_unknown_token_is_told_it_was_tried_against_countries_too(self):
        """D8's refusal names what the token was tried *against*, which D15
        makes possible by closing the vocabulary. The vocabulary just grew, so
        a message still saying "is not a U.S. state" would send someone
        hunting for a spelling mistake in a country name they typed correctly
        and that simply is not matched by ``ADMIN``."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        with self.assertRaises(MCPToolError) as caught:
            resolver.resolve_location("NY + Wakanda")

        text = f"{caught.exception.message} {caught.exception.suggestion or ''}"
        self.assertIn("'wakanda'", text.lower())
        self.assertIn("country", text.lower(),
                      "the refusal still describes a states-only vocabulary")

    def test_france_refuses_by_naming_france_not_by_blaming_the_phrasing(self):
        """V21, and the phase's most surprising measurement.

        ``"France + Germany"`` was criterion #2 in the phase prompt. It
        **refuses**, and it should: France's envelope is 8,524 deg^2 -- 119x its
        own land area -- because ``ADMIN`` France runs from French Guiana
        (-61.8) to Reunion (+55.8). The union with Germany is 8,988 deg^2 =
        16.7 M native cells, 4.2x the frame ceiling, and letting it through
        would put exactly that allocation into ``_crop_to_mask_footprint``.

        What must not ship is 3b's *message*. It says the members are far apart
        and to ask for nearer ones -- and France and Germany share a border, so
        every word of that is wrong and there is nothing the user can fix. A
        refusal that misdescribes the problem is the trap D7 names in a
        different guise: it leaves the user stuck.

        So this asserts the refusal **names France**, and does not offer the
        nearer-members advice that belongs to the other mistake."""
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError

        resolver = self._resolver()
        with self.assertRaises(MCPToolError) as caught:
            with patch.object(resolver.geocoding_service, "geocode") as geocode:
                resolver.resolve_location("France + Germany")
        geocode.assert_not_called()

        error = caught.exception
        self.assertEqual(error.category, CATEGORY_TOO_LARGE)
        text = f"{error.message} {error.suggestion or ''}".lower()
        self.assertIn("'france'", text,
                      "the refusal must name the member responsible")
        self.assertNotIn("nearer", text,
                         "that advice belongs to the far-apart mistake, not this one")

    def test_a_composite_of_two_distant_but_ordinary_members_still_says_far_apart(self):
        """The other half of the same decision: two mistakes, two answers.

        3b's message is not wrong, it was just being asked to cover a case it
        does not describe. Brazil and Japan each fit under the ceiling on their
        own; it is genuinely their separation that blows the envelope, and
        "ask for fewer, or nearer, members" is exactly the right advice. Pinned
        so the new branch cannot swallow the old one."""
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError

        resolver = self._resolver()
        with self.assertRaises(MCPToolError) as caught:
            resolver.resolve_location("Brazil + Japan")

        error = caught.exception
        self.assertEqual(error.category, CATEGORY_TOO_LARGE)
        self.assertIn("nearer", (error.suggestion or "").lower())


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class PriorPhasesStillHoldTests(unittest.TestCase):
    """Criterion #10, gathered in one place.

    The 3a and 3b suites cover these already. They are re-asserted here because
    the country tier gives every one of them a new way to break -- a token that
    used to fall off the end of ``_member_geometry`` now has a second table to
    fall into, and a bare name that used to reach the geocoder now passes an
    ambiguity guard on the way. A regression net that lives next to the change
    is the one that gets read when this file is edited again.
    """

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_ny_plus_nj_is_unchanged(self):
        from shapely.ops import unary_union

        resolver = self._resolver()
        with patch.object(resolver.geocoding_service, "geocode") as geocode:
            region = resolver.resolve_location("NY + NJ")
        geocode.assert_not_called()

        expected = unary_union([resolver.resolve_location("new york")["geometry"],
                                resolver.resolve_location("new jersey")["geometry"]])
        self.assertEqual(region["region_type"], "composite_union")
        self.assertAlmostEqual(region["geometry"].area, expected.area, places=9)

    def test_a_bad_token_still_names_that_token(self):
        """D14's headline behaviour. The country tier is the obvious place for
        an unknown token to silently acquire a *second* chance at resolving --
        or for the message to start naming the wrong thing."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        with self.assertRaises(MCPToolError) as caught:
            with patch.object(resolver.geocoding_service, "geocode") as geocode:
                resolver.resolve_location("NY + NJ + Wakanda")
        geocode.assert_not_called()
        self.assertIn("'wakanda'", caught.exception.message.lower())

    def test_presets_and_coalitions_are_still_not_composite_members(self):
        """D15, and the country tier is exactly what could have reopened it --
        ``"mexico"`` is now a member while ``"conus"`` still is not."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        resolver = self._resolver()
        for raw, token in (("otc + ohio", "otc"), ("conus + mexico", "conus")):
            with self.subTest(raw=raw):
                with self.assertRaises(MCPToolError) as caught:
                    resolver.resolve_location(raw)
                self.assertIn(f"'{token}'", caught.exception.message.lower())

    def test_the_untouched_strings_are_still_untouched(self):
        """``"pennsylvania"`` and ``"otc"`` resolve to real polygons; ``"paris"``
        still reaches the geocoder and pays nothing for the country asset."""
        resolver = self._resolver()

        for key in ("pennsylvania", "otc", "new england", "conus", "europe"):
            with self.subTest(preset=key):
                self.assertEqual(resolver.resolve_location(key)["region_type"],
                                 "polygon")

        with patch.object(resolver.geocoding_service, "geocode",
                          return_value=None) as geocode:
            resolver.resolve_location("paris")
        geocode.assert_called_once_with("paris")


def _unsimplified_areas() -> dict:
    """Areas of the **unsimplified** 50m admin-0 features, deg^2, measured in
    the Phase 4 gate against ``ne_50m_admin_0_countries.geojson`` (3,083,490
    bytes) and checked in as a fixture.

    Checked in rather than fetched because no test may touch the network, and
    the whole point of the assertion is to compare the shipped geometry against
    something the shipped geometry cannot influence. All 242, because that is
    the set V18's scheme claims to hold -- a spot-check of the seven the gate
    happened to print would not test the claim, it would test the examples."""
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "admin0_unsimplified_areas.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class Admin0FidelityTests(unittest.TestCase):
    """D2a at country scale -- where the constant D2a picked stops working.

    D2 switched the standalone tiers to 50m *specifically* to protect Vatican,
    Monaco, Singapore and Liechtenstein. D2a then picked 0.01 deg on a
    measurement over the 51 U.S. states, which all sit within one order of
    magnitude of each other in size. The countries span six, and at 0.01 the
    protection D2 bought is undone: **33.2% of Vatican City survives**.

    The shipped scheme is proportional to each feature's own linear scale,
    which is why it holds for Vatican and Russia at once.
    """

    def test_every_country_keeps_its_area_within_a_two_sided_band(self):
        """All 242, and two-sided, which is not a formality here.

        Four countries read *above* 101% at the old uniform tolerance --
        American Samoa 103.5%, Indian Ocean Territories 102.7%, Jersey 102.6%
        -- because dropping a concave vertex adds area (3a's DC lesson). A
        one-sided ``>= 99%`` assertion passes every one of them."""
        from tta_backend.utils.plotting import load_admin0_polygons

        shipped = load_admin0_polygons()
        truth = _unsimplified_areas()
        self.assertEqual(set(shipped), set(truth),
                         "the shipped vocabulary and the measured source disagree")

        for key, true_area in truth.items():
            with self.subTest(country=key):
                retention = 100.0 * shipped[key]["geometry"].area / true_area
                self.assertGreater(
                    retention, 99.0,
                    f"{key} lost shape: {retention:.3f}% of the unsimplified area")
                self.assertLess(
                    retention, 101.0,
                    f"{key} gained shape: {retention:.3f}% of the unsimplified area")

    def test_vatican_the_measured_worst_case_survives_essentially_intact(self):
        """Named separately because it is *the* number this scheme exists for.

        At ``ADMIN1_TOLERANCE`` Vatican City retains 33.2% of its area -- a
        1.1 km tolerance applied to a jurisdiction ~0.7 km across. The band
        test above would catch that, but it would report it as one of 242
        subtests; this says out loud which measurement drove the design, so a
        future reader tempted to "simplify the simplification" back to a single
        constant sees the cost first."""
        from tta_backend.utils.plotting import load_admin0_polygons

        retention = (100.0 * load_admin0_polygons()["vatican"]["geometry"].area
                     / _unsimplified_areas()["vatican"])
        self.assertGreater(retention, 99.0, f"Vatican retains {retention:.2f}%")
        self.assertLess(retention, 101.0, f"Vatican inflated to {retention:.2f}%")

    def test_the_states_asset_was_not_re_cut_to_help_the_countries(self):
        """``ADMIN1_TOLERANCE`` is shared by the 51 states and both coalitions.

        Giving the countries their own constants rather than moving that one is
        a decision, not an accident: raising it to help Vatican would silently
        re-cut every state and both coalition boundaries. This pins that the
        countries' scheme is genuinely separate."""
        from scripts.build_preset_regions import (
            ADMIN0_TOLERANCE_CAP, ADMIN0_TOLERANCE_DIVISOR, ADMIN1_TOLERANCE,
        )

        self.assertEqual(ADMIN1_TOLERANCE, 0.01)
        self.assertEqual(ADMIN0_TOLERANCE_CAP, 0.02)
        self.assertEqual(ADMIN0_TOLERANCE_DIVISOR, 100)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class DuplicateNameTests(unittest.TestCase):
    """V19: two names exist in both vocabularies, and one asset would lose one."""

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_antarctica_resolves_to_the_continent_preset_and_not_the_admin0_feature(self):
        """Criterion #5, pinned by identity rather than by plausibility.

        ``Antarctica`` is a duplicate, not an ambiguity: the continent preset
        and the admin-0 feature are *the same place*, so there is nothing to
        refuse -- only a question of which polygon wins. The continent preset
        wins because it is what resolves today and nothing about it is wrong.

        Their areas differ by well under 1%, so an area assertion would pass on
        either. Geometry identity is what stops the answer flipping silently on
        the next asset rebuild."""
        from tta_backend.utils.plotting import (
            load_admin0_polygons, load_preset_polygons,
        )

        resolver = self._resolver()
        region = resolver.resolve_location("antarctica")
        geometry = region["geometry"]

        self.assertTrue(geometry.equals(load_preset_polygons()["antarctica"]),
                        "bare 'antarctica' is no longer the continent preset")
        self.assertFalse(
            geometry.equals(load_admin0_polygons()["antarctica"]["geometry"]),
            "bare 'antarctica' flipped to the admin-0 country feature",
        )

    def test_neither_asset_silently_loses_a_feature_to_a_duplicate_id(self):
        """The V19 finding, as a standing guard rather than a decision record.

        Merging the countries into ``preset_regions.geojson`` produces 304
        features that ``load_preset_polygons`` reduces to **302** polygons --
        both loaders are dict comprehensions keyed on ``feature["id"]``, and
        ``georgia`` and ``antarctica`` are in both vocabularies. Last wins, so
        ``"georgia"`` would mask the Caucasus while ``global_regions['georgia']``
        still reported the U.S. Southeast: zero overlap, no error.

        ``AliasCollisionError`` cannot catch that. It compares keys against
        ``global_regions``, one layer above the asset, and has no view of
        feature ids at all. This is the assertion that does."""
        import json

        from tta_backend.utils import plotting

        for path, loader in (
            (plotting._PRESET_REGIONS_PATH, plotting.load_preset_polygons),
            (plotting._ADMIN0_COUNTRIES_PATH, plotting.load_admin0_polygons),
        ):
            with open(path, encoding="utf-8") as fh:
                features = json.load(fh)["features"]
            self.assertEqual(
                len(features), len(loader()),
                f"{path} has duplicate feature ids; one polygon is being "
                "silently overwritten by another under the same name",
            )

    def test_the_country_alias_table_shadows_nothing_that_already_resolves(self):
        """D12a's mechanism, extended to the table this phase adds.

        A hand-curated table that silently shadows ``"us"`` or a postal code is
        the cheapest possible way to reintroduce a confident wrong region, and
        review is not a reliable check for it. 21 aliases against 90 preset
        keys, 51 states, 51 postal codes, 4 aliases and 2 coalitions."""
        from tta_backend.utils.region_composition import assert_no_country_collisions

        assert_no_country_collisions(self._resolver().global_regions)

    def test_a_country_alias_that_shadows_a_preset_raises_instead_of_winning(self):
        """3a's guard exercised at the new table, by mutation rather than by
        trusting the census above to stay true."""
        from unittest.mock import patch as _patch

        from tta_backend.utils.region_composition import (
            COUNTRY_ALIASES, assert_no_country_collisions,
        )
        from tta_backend.utils.region_dispatch import AliasCollisionError

        resolver = self._resolver()
        shadowing = dict(COUNTRY_ALIASES, **{"conus": "france"})
        with _patch("tta_backend.utils.region_composition.COUNTRY_ALIASES", shadowing):
            with self.assertRaises(AliasCollisionError) as caught:
                assert_no_country_collisions(resolver.global_regions)
        self.assertIn("conus", str(caught.exception))

    def test_a_country_alias_that_shadows_a_postal_code_raises(self):
        """The collision that matters most, because it is invisible: an alias
        ``"ca"`` would make ``"CA + NY"`` mean Canada plus New York, which is
        a real request someone could mean and not the one they typed."""
        from unittest.mock import patch as _patch

        from tta_backend.utils.region_composition import (
            COUNTRY_ALIASES, assert_no_country_collisions,
        )
        from tta_backend.utils.region_dispatch import AliasCollisionError

        resolver = self._resolver()
        shadowing = dict(COUNTRY_ALIASES, **{"ca": "canada"})
        with _patch("tta_backend.utils.region_composition.COUNTRY_ALIASES", shadowing):
            with self.assertRaises(AliasCollisionError) as caught:
                assert_no_country_collisions(resolver.global_regions)
        self.assertIn("ca", str(caught.exception))

    def test_every_country_alias_points_at_a_country_that_exists(self):
        """An alias pointing at nothing is a token that fails with 'not a
        country' while sitting in the table that claims to define countries."""
        from tta_backend.utils.plotting import load_admin0_polygons
        from tta_backend.utils.region_composition import COUNTRY_ALIASES

        vocabulary = load_admin0_polygons()
        for alias, target in COUNTRY_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIn(target, vocabulary,
                              f"alias {alias!r} points at {target!r}, which is "
                              "not an ADMIN value")
                self.assertNotIn(alias, vocabulary,
                                 f"alias {alias!r} shadows the ADMIN value of "
                                 "the same name")

    def test_no_country_name_is_two_letters_so_none_can_shadow_a_postal_code(self):
        """D6 from the other direction. Finding 5 measured 26 of 51 postal
        codes colliding with ISO alpha-2 codes; the grammar bans codes, and
        this pins that no *name* sneaks a code in through the back door."""
        from tta_backend.utils.plotting import load_admin0_polygons
        from tta_backend.utils.region_composition import POSTAL_TO_STATE

        countries = set(load_admin0_polygons())
        self.assertEqual(countries & set(POSTAL_TO_STATE), set())

    def test_the_country_asset_is_not_opened_unless_a_country_is_asked_for(self):
        """The invariant the whole two-asset decision rests on (V19).

        Splitting the assets buys nothing if something on the common path opens
        the country file anyway -- the 165 ms cold parse would simply move, and
        ``"paris"`` would pay it exactly as it would have under a merged asset.
        Two things could reintroduce that and both are easy to write by
        accident: an asset read inside ``RegionResolver.__init__``, and the
        alias collision guard reaching for the vocabulary to validate against.
        ``assert_no_country_collisions`` deliberately does neither.

        Asserted by clearing the cache and checking it is *still* empty, which
        is the only formulation that cannot be satisfied by a warm cache from
        an earlier test in the same process."""
        from tta_backend.utils import plotting

        plotting.load_admin0_polygons.cache_clear()
        self.addCleanup(plotting.load_admin0_polygons.cache_clear)

        resolver = plotting.RegionResolver()          # runs the collision guards
        resolver.resolve_location("conus")            # a plain preset
        with patch.object(resolver.geocoding_service, "geocode", return_value=None):
            resolver.resolve_location("paris")        # a geocoded name

        self.assertEqual(
            plotting.load_admin0_polygons.cache_info().currsize, 0,
            "the country asset was parsed for a request that names no country",
        )

        resolver.resolve_location("NY + NJ")
        self.assertEqual(
            plotting.load_admin0_polygons.cache_info().currsize, 1,
            "a composite must consult the country vocabulary -- it is how the "
            "ambiguity guard knows 'georgia' is one",
        )

    def test_the_two_shared_names_really_are_in_both_assets(self):
        """Without this the guard above passes vacuously if the duplicate is
        ever removed from one side, and the reason for two files quietly
        stops being demonstrated."""
        from tta_backend.utils.plotting import (
            load_admin0_polygons, load_preset_polygons,
        )

        shared = set(load_preset_polygons()) & set(load_admin0_polygons())
        self.assertEqual(shared, {"georgia", "antarctica"})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class AntimeridianAndCeilingTests(unittest.TestCase):
    """V21, both halves: what reaches the MCP, and where D16's hole is."""

    # Phase 3a decision 4 justified giving the states ``global_regions`` entries
    # by writing down that no state crosses the antimeridian -- Alaska's 50m
    # extent is -178.195..-130.014, wholly western. That justification does not
    # transfer: these five each span the full 360 degrees, so their honest
    # bounding box is the entire globe.
    ANTIMERIDIAN = [
        "united states of america", "new zealand", "russia", "antarctica", "fiji",
    ]

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_the_five_really_do_span_the_globe(self):
        """Without this the refusal test below is vacuous -- it would pass just
        as well if these countries were ordinary and small."""
        from tta_backend.utils.plotting import load_admin0_polygons

        polygons = load_admin0_polygons()
        for key in self.ANTIMERIDIAN:
            with self.subTest(country=key):
                west, _, east, _ = polygons[key]["geometry"].bounds
                self.assertGreater(
                    east - west, 340.0,
                    f"{key} no longer has a globe-wide envelope; V21's "
                    "reasoning needs re-running")

    def test_no_antimeridian_country_can_reach_the_retrieval_plane(self):
        """Criterion #8, and the answer is *structural* rather than handled.

        The prompt asked what ``define_area_of_interest`` does with a
        whole-globe ``"-180,41.2,180,81.9"`` bbox. It never finds out, and that
        is the finding: the member gate fires while resolving the token, before
        ``dispatch_composite_extent`` formats any bbox at all. All five exceed
        the 2,148.7 deg^2 ceiling on their own -- Fiji, the smallest, by 1.5x
        and Russia by 6.8x -- so the whole-globe bbox cannot be constructed.

        Asserted at the retrieval plane specifically, because that is the plane
        where a globe-wide extent would do the damage (Risk 5: the mask clips a
        cube that never covered the region)."""
        from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
        from tta_backend.utils.region_composition import dispatch_composite_extent

        resolver = self._resolver()
        for key in self.ANTIMERIDIAN:
            with self.subTest(country=key):
                with self.assertRaises(MCPToolError) as caught:
                    dispatch_composite_extent(f"{key} + japan", resolver)
                self.assertEqual(caught.exception.category, CATEGORY_TOO_LARGE)
                self.assertIn(f"'{key}'", caught.exception.message,
                              "the refusal must name the member responsible")

    def test_a_bare_country_is_not_claimed_by_either_plane(self):
        """Criterion #9: D16 stays composite-only, and the hole is *absent*.

        V15 established the gate has no hole because no bare state token trips
        it. Ten bare country tokens would -- France and Canada among them -- so
        the argument had to be remade, not restated. It is remade by scope:
        countries are **member-only** vocabulary (V21), reachable only between
        ``+`` signs. A bare country never enters the T60 vocabulary, so the
        ceiling has nothing to refuse and nothing that resolves today stops.

        This is what makes ``"france"`` still work while ``"France + Germany"``
        refuses, and it is the only reason those two are consistent."""
        from tta_backend.utils import region_dispatch
        from tta_backend.utils.region_composition import (
            dispatch_composite, dispatch_composite_extent,
        )

        resolver = self._resolver()
        for key in ["france", "canada", "russia", "china", "netherlands"]:
            with self.subTest(country=key):
                self.assertFalse(dispatch_composite(key, resolver).claimed)
                self.assertFalse(dispatch_composite_extent(key, resolver).claimed)
                self.assertFalse(
                    region_dispatch.dispatch_extent(key, resolver.global_regions).claimed,
                    f"bare {key!r} was claimed by the retrieval plane; it would "
                    "now be subject to a ceiling it exceeds")

    def test_a_bare_country_still_reaches_the_geocoder_exactly_as_before(self):
        """The other half of member-only: not claimed is only safe if the old
        path still runs. ``"france"`` geocodes today and must keep doing so."""
        from tta_backend.utils.plotting import RegionResolver

        resolver = RegionResolver()
        with patch.object(resolver.geocoding_service, "geocode",
                          return_value=None) as geocode:
            resolver.resolve_location("france")
        geocode.assert_called_once_with("france")

    def test_no_us_state_is_affected_by_the_new_member_gate(self):
        """The member gate is new behaviour on a shared path, so it has to be
        shown not to bite anything 3a and 3b shipped. V15 measured Alaska, the
        worst of the 51, at 954 deg^2 against a 2,148.7 ceiling."""
        from tta_backend.datasets.us_states import US_STATES
        from tta_backend.utils.region_composition import MAX_COMPOSITE_ENVELOPE_DEG2

        for key, state in US_STATES.items():
            with self.subTest(state=key):
                west, south, east, north = state["bounds"]
                self.assertLess((east - west) * (north - south),
                                MAX_COMPOSITE_ENVELOPE_DEG2)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region country test dependencies are not installed",
)
class GeorgiaAmbiguityTests(unittest.TestCase):
    """D7, as V20 narrowed it: fail closed **where no reading is established**."""

    def _resolver(self):
        from tta_backend.utils.plotting import RegionResolver
        return RegionResolver()

    def test_georgia_as_a_composite_member_refuses_and_names_both_escape_hatches(self):
        """The position where D7 applies in full.

        Composite membership for countries is new surface: ``"georgia +
        florida"`` resolves to nothing today, so there is no established
        reading to destroy, and D15's order would silently hand it to the
        state. That is a genuine coin flip with no incumbent -- D7's case
        exactly.

        Both hatches are asserted, and that is the point rather than a
        flourish. D7 says a bare "ambiguous" message leaves the user stuck;
        naming only ``GA`` would leave *half* of them stuck, with a stated way
        to ask for the state and no way at all to ask for the country."""
        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        resolver = self._resolver()
        with self.assertRaises(MCPToolError) as caught:
            with patch.object(resolver.geocoding_service, "geocode") as geocode:
                resolver.resolve_location("georgia + florida")
        geocode.assert_not_called()

        error = caught.exception
        self.assertEqual(error.category, CATEGORY_USER_INPUT)
        text = f"{error.message} {error.suggestion or ''}".lower()
        self.assertIn("'ga'", text, "no escape hatch named for the U.S. state")
        self.assertIn("'georgia (country)'", text,
                      "no escape hatch named for the country")

    def test_a_bare_georgia_still_resolves_to_the_us_state_untouched(self):
        """V20: D7 applied literally here would *remove* working behaviour.

        ``"georgia"`` resolves to the U.S. state today through
        ``global_regions`` (3a), and 3a's V12 measured live Nominatim already
        preferring the state -- 99.41% coverage, ``boundary/administrative``.
        Both planes agree, and they agree correctly.

        D7's stated justification is that "silently picking either meaning
        would be wrong roughly half the time". That premise does not hold at
        this position: the meaning was not picked by this design, it was picked
        by 3a and by OSM before that, and refusing now would convert a
        currently-correct answer into an error."""
        from shapely.geometry import Point

        resolver = self._resolver()
        region = resolver.resolve_location("georgia")

        self.assertEqual(region["region_type"], "polygon")
        self.assertTrue(region["geometry"].covers(Point(*ATLANTA)),
                        "bare 'georgia' stopped resolving to the U.S. state")
        self.assertFalse(region["geometry"].covers(Point(*TBILISI)),
                         "bare 'georgia' became the country")

    def test_the_postal_hatch_resolves_the_state_inside_a_composite(self):
        """``GA`` must survive the guard -- the check is on the *token*, not on
        the key it resolves through, or D7's own escape hatch would trip it."""
        from shapely.geometry import Point

        resolver = self._resolver()
        region = resolver.resolve_location("GA + florida")
        self.assertTrue(region["geometry"].covers(Point(*ATLANTA)))

    def test_the_country_hatch_resolves_the_country_inside_a_composite(self):
        """And the other side, which is the half D7's text does not supply."""
        from shapely.geometry import Point

        resolver = self._resolver()
        region = resolver.resolve_location("georgia (country) + armenia")

        self.assertTrue(region["geometry"].covers(Point(*TBILISI)),
                        "the country hatch did not reach the country")
        self.assertFalse(region["geometry"].covers(Point(*ATLANTA)),
                         "the country hatch reached the U.S. state")
