"""T60 Phase 1.5 (D13): the retrieval plane speaks the coalition vocabulary.

Phase 1 shipped a correct *mask* for the Ozone Transport Region that no user
could reach from a prompt. The spatial subset that actually pulls data comes
from the MCP's ``define_area_of_interest``, which lives in another repo and has
never heard of the word -- so ``"ozone over the OTC"`` failed at AOI resolution
before a line of T60 code ran.

Measured live in the Phase 1.5 gate (V6): the agent sends ``"Ozone Transport
Commission"``, the MCP's geocoder finds nothing, and the agent then *silently
substitutes* ``"Northeastern United States"`` and retrieves that instead --
which is not a superset of the OTR, so the mask would have clipped against a
cube that never covered a third of the region (Risk 5).

These tests drive the real ``bind_workspace`` seam against the in-process fake
MCP, so what is asserted is what the MCP actually receives.
"""
import importlib.util
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import ProcessCacheIsolation  # noqa: E402 -- needs the TESTS_DIR insert above

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "shapely", "cartopy", "rasterio"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-retrieval-extent test dependencies are not installed",
)
class RetrievalExtentTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        self.seen: list[str] = []

        async def define_area_of_interest(location, workspace_id):
            self.seen.append(location)
            return {"handle": "aoi_fake", "bbox": _parse_bbox(location), "source": "bbox"}

        self.server = FakeEarthdataMCPServer(
            build_fake_mcp({"define_area_of_interest": define_area_of_interest})
        )
        self.server.start()
        self.addCleanup(self.server.stop)

        from tta_backend.config.settings import Settings
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools

        raw = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        )
        from tta_backend.earthdata_mcp.workspace import bind_workspace

        self.tools = bind_workspace(raw, lambda: "17")

    async def test_otc_reaches_the_mcp_as_an_extent_never_as_the_word(self):
        """The tracer bullet. ``"the OTC"`` must arrive at the MCP as a real
        footprint, and the string itself must never be handed to the MCP's own
        geocoder -- which resolves it to an aerodrome in Chad (Phase 0, V2)."""
        from tta_backend.utils.plotting import load_preset_polygons

        await self.tools["define_area_of_interest"].ainvoke({"location": "the OTC"})

        self.assertEqual(len(self.seen), 1)
        sent = self.seen[0]
        self.assertNotIn("otc", sent.lower())

        # It is the OTR's own envelope, not some other rectangle.
        expected = load_preset_polygons()["otc"].bounds
        self.assertEqual(_parse_bbox(sent), [round(v, 6) for v in expected])

    async def test_the_retrieval_extent_contains_the_mask_for_every_coalition(self):
        """Risk 5, pinned. The retrieval extent must *contain* the mask -- not
        equal it. A superset is correct, because the mask clips it; anything
        less and the researcher gets a mean over whatever happened to arrive,
        with no signal that part of the region was never retrieved.

        Asserted over COALITIONS so a coalition added later cannot skip it.
        Containment, never equality: the envelope is 2.453x the OTR's area
        (gate V7) and that is the design, not a bug."""
        from shapely.geometry import box

        from tta_backend.utils.plotting import RegionResolver, load_preset_polygons
        from tta_backend.utils.region_dispatch import COALITIONS

        resolver = RegionResolver()
        polygons = load_preset_polygons()

        for coalition_id in COALITIONS:
            with self.subTest(coalition=coalition_id):
                self.seen.clear()
                await self.tools["define_area_of_interest"].ainvoke({"location": coalition_id})

                retrieved = box(*_parse_bbox(self.seen[0]))
                # The mask plane's own answer for the same string, so the two
                # planes are compared rather than the test reciting a constant.
                masked = resolver.resolve_location(coalition_id)["geometry"]

                self.assertTrue(
                    retrieved.contains(masked) or retrieved.equals(masked),
                    f"retrieval extent {retrieved.bounds} does not contain "
                    f"the mask {masked.bounds} for {coalition_id!r}",
                )
                # ...and it is that region's own envelope, not a bigger box
                # that would trivially satisfy containment.
                self.assertEqual(
                    _parse_bbox(self.seen[0]),
                    [round(v, 6) for v in polygons[coalition_id].bounds],
                )

    async def test_a_state_name_reaches_the_mcp_as_an_extent_never_as_the_word(self):
        """T60 Phase 3a: the states go through this seam too.

        Three tokens, chosen because the Phase 3a gate (V13) measured each of
        them failing live and they fail in two different ways:

        - ``"new york"`` came back as New York **City** -- 0.56% of the state.
        - ``"washington"`` came back as Washington **DC** -- **0.00%** overlap,
          47 degrees of longitude away.
        - ``"pennsylvania"`` came back as the *right* state and still left
          0.023 deg of its northern edge outside what was retrieved.

        The third is the one that generalizes, and the 51-wide containment
        property it implies is asserted where the extent is actually decided --
        ``dispatch_extent``, in test_region_us_states.py -- rather than paid for
        51 times over an HTTP round trip. What this test covers is the wiring:
        that a state reaches the seam at all."""
        for token in ("new york", "washington", "pennsylvania"):
            with self.subTest(location=token):
                self.seen.clear()
                await self.tools["define_area_of_interest"].ainvoke({"location": token})

                sent = self.seen[0]
                self.assertNotIn(token, sent.lower())
                self.assertIsNotNone(
                    _parse_bbox(sent),
                    f"{token!r} reached the MCP as a place name, not an extent",
                )

    async def test_a_composite_reaches_the_mcp_as_an_extent_never_as_the_string(self):
        """T60 Phase 3b: the ``+`` grammar goes through this seam too, and it
        has *less* to fall back on than anything before it.

        A coalition has a name the MCP's geocoder might resolve, badly. A
        composite has no key at all -- it is built per request -- so an
        unclaimed ``"NY + NJ"`` would reach Nominatim as a literal string with
        a ``+`` in it. The 51-wide containment property is asserted where the
        extent is decided (test_region_composition.py); this covers the
        wiring."""
        from tta_backend.utils.plotting import RegionResolver

        await self.tools["define_area_of_interest"].ainvoke({"location": "NY + NJ"})

        self.assertEqual(len(self.seen), 1)
        sent = self.seen[0]
        self.assertNotIn("+", sent)
        # It is the union's own envelope, agreeing with the mask plane exactly.
        masked = RegionResolver().resolve_location("NY + NJ")
        self.assertEqual(_parse_bbox(sent), [round(v, 6) for v in masked["bounds"]])

    async def test_a_composite_with_a_bad_token_refuses_instead_of_geocoding(self):
        """D8 on the retrieval plane. The MCP must never be asked, because its
        geocoder is the same Nominatim the mask plane is keeping the string
        away from -- and a fail-open seam is indistinguishable from a working
        one unless the not-called assertion is made."""
        from tta_backend.earthdata_mcp.results import (
            CATEGORY_USER_INPUT, MCPToolError, parse_tool_result,
        )

        raw = await self.tools["define_area_of_interest"].ainvoke(
            {"location": "NY + NJ + Wakanda"}
        )

        self.assertEqual(self.seen, [])
        with self.assertRaises(MCPToolError) as caught:
            parse_tool_result(raw)
        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)
        self.assertIn("wakanda", caught.exception.message.lower())

    async def test_an_oversized_composite_refuses_before_any_retrieval(self):
        """D16 where it costs least. The extent gate firing here means a
        continent-spanning union never becomes a retrieval at all, rather than
        being fetched and refused downstream by a byte limit that would name
        the wrong problem."""
        from tta_backend.earthdata_mcp.results import (
            CATEGORY_TOO_LARGE, MCPToolError, parse_tool_result,
        )

        raw = await self.tools["define_area_of_interest"].ainvoke(
            {"location": "alaska + florida"}
        )

        self.assertEqual(self.seen, [])
        with self.assertRaises(MCPToolError) as caught:
            parse_tool_result(raw)
        self.assertEqual(caught.exception.category, CATEGORY_TOO_LARGE)

    async def test_a_string_outside_the_vocabulary_is_passed_through_untouched(self):
        """The regression this seam is most likely to cause, and the mirror of
        the mask plane's NOT_CLAIMED fall-through. "paris" must reach the MCP
        byte-identical to pre-T60 behavior -- as must a plain preset key, which
        this phase deliberately does not claim."""
        for location in ("paris", "Paris, France", "northeast us", "north america", "-75,40,-74,41"):
            with self.subTest(location=location):
                self.seen.clear()
                await self.tools["define_area_of_interest"].ainvoke({"location": location})
                self.assertEqual(self.seen, [location])

    async def test_an_alias_onto_a_plain_preset_agrees_with_the_mask_plane(self):
        """Phase 1 *created* this divergence and this phase closes it. Before
        T60, "northeastern us" geocoded to an ~11 m railway platform in Boston
        (Phase 0, V2) on both planes -- consistently wrong. Phase 1 taught the
        mask plane it means the ``northeast us`` box and left the retrieval
        plane on the platform, which is Risk 5 exactly: a mask clipping against
        a cube covering 11 metres of it."""
        from tta_backend.utils.plotting import RegionResolver

        await self.tools["define_area_of_interest"].ainvoke({"location": "northeastern us"})

        masked = RegionResolver().resolve_location("northeastern us")
        self.assertEqual(_parse_bbox(self.seen[0]), [round(v, 6) for v in masked["bounds"]])

    async def test_a_coalition_with_no_polygon_refuses_instead_of_geocoding(self):
        """Phase 1 fails closed on the mask plane; the retrieval plane must not
        fail open behind it. If the boundary asset is missing, the string must
        NOT reach the MCP -- its geocoder is Nominatim, which answers "OTC"
        with an aerodrome in Chad (Phase 0, V2). A not-found beats a retrieval
        over the wrong continent."""
        from unittest.mock import patch

        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError, parse_tool_result

        with patch("tta_backend.utils.plotting.load_preset_polygons", return_value={}):
            raw = await self.tools["define_area_of_interest"].ainvoke({"location": "the OTC"})

        # The MCP was never asked. This is the assertion that bites.
        self.assertEqual(self.seen, [])
        with self.assertRaises(MCPToolError) as caught:
            parse_tool_result(raw)
        self.assertEqual(caught.exception.category, CATEGORY_USER_INPUT)

    async def test_every_aoi_caller_goes_through_the_seam(self):
        """The two backend composites *and* the agent's own direct call. The
        third is the majority path -- workflow step 2 makes the AOI step
        mandatory and first, and no composite edit can intercept it -- so a
        translation wired into only the composites is the same half-delivery
        this phase exists to end."""
        from tta_backend.earthdata_mcp.toolset import curated_model_tools
        from tta_backend.services.discovery_service import _resolve_aoi

        expected = _parse_bbox(
            ",".join(f"{v:.6f}" for v in __import__(
                "tta_backend.utils.plotting", fromlist=["x"]
            ).load_preset_polygons()["otc"].bounds)
        )

        from tta_backend.earthdata_mcp import tool_cache

        curated = {tool.name: tool for tool in curated_model_tools(self.tools)}

        callers = (
            # discovery_service._resolve_aoi, which passes location straight through.
            ("discovery_service", lambda: _resolve_aoi("the OTC", self.tools)),
            # A composite reaching the tool by name off the same dict --
            # exactly how retrieval_composites.py:508 calls it.
            ("composite", lambda: self.tools["define_area_of_interest"].ainvoke({"location": "the OTC"})),
            # The agent's own curated, model-facing tool. The majority path.
            ("agent", lambda: curated["define_area_of_interest"].ainvoke({"location": "the OTC"})),
        )
        for label, call in callers:
            with self.subTest(caller=label):
                # Cleared between callers precisely because they'd otherwise
                # share one T53 entry -- see the cache-collapse test below.
                tool_cache.clear_tool_cache()
                self.seen.clear()
                await call()
                self.assertEqual(
                    _parse_bbox(self.seen[0]), expected, f"{label} missed the seam"
                )

    async def test_the_vocabulary_collapses_to_one_t53_cache_entry(self):
        """Tension 4, pinned. The rewrite sits *before* the T53 lookup, which
        keys on ``kwargs``, so every spelling of the region resolves to one
        entry -- and no key can ever hold a raw coalition string. Were the
        rewrite placed after the lookup instead, the cache would answer "otc"
        with whatever had been stored for the raw string."""
        for spelling in ("otc", "the OTC", "  OTC  ", "otr", "Ozone Transport Region"):
            await self.tools["define_area_of_interest"].ainvoke({"location": spelling})

        # Five spellings, one round trip to the MCP.
        self.assertEqual(len(self.seen), 1)


def _parse_bbox(location: str) -> list[float] | None:
    """The "W,S,E,N" channel the Phase 1.5 gate chose (V5c/V7), parsed back."""
    try:
        parts = [float(p) for p in location.split(",")]
    except (ValueError, AttributeError):
        return None
    return parts if len(parts) == 4 else None


if __name__ == "__main__":
    unittest.main()
