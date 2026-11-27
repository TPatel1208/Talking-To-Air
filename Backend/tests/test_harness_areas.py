"""sources.areas.resolve: a place in the user's words becomes an area handle, or a question."""
import datetime
import tempfile
import unittest
from typing import Annotated, Any

from tta_backend.harness.capabilities import Call, Capability, CapabilityRegistry, CapabilityResult, InputEvicted, Params, handle_param, invoke
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor
from tta_backend.harness.run_context import PLACE_REFUSAL_CODES, MAX_FACT_CHARS, PlaceRefused
from tta_backend.harness.testing import FakePlaces, FakeToolError, box_region, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

NJ = box_region("New Jersey", -75.6, 38.9, -73.9, 41.4, region_type="polygon")
PARIS_TX = box_region("Paris, Lamar County, Texas, United States", -95.6, 33.6, -95.5, 33.7, region_type="polygon")


class AreaParams(Params):
    area: Annotated[str, handle_param("polygons")]
    recorded: str | None = None


async def _read_area(call: Call[AreaParams]) -> CapabilityResult:
    facts = areas.area_facts(call.handle("area"), call=call)
    geometry = await areas.area_geometry(call, "area")
    values: dict[str, Any] = {"name": facts.name, "bounds": list(facts.bounds), "type": geometry["type"]}
    if call.params.recorded is not None:
        by_id, footprint = await areas.area_geometry_of(call, call.params.recorded)
        values["recorded"] = [by_id.id, footprint["type"]]
    return call.done(values=values)


READER = Capability(
    name="operations.read_area", description="Read an area.", params=AreaParams, output_kind=None,
    label="Reading", method="Read.", retrieves=False, retry_safe=True, run=_read_area,
)


class RaisingPlaces:
    async def resolve(self, wording: str) -> None:
        raise FakeToolError("user_input", "bad composite token ' + '")


# Each refusal code: the wording, the facts, and phrases its question must contain.
REFUSALS: list[tuple[str, str, dict[str, str], tuple[str, ...]]] = [
    ("member_not_found", "NY + Wakanda", {"member": "Wakanda"},
     ("'Wakanda' in 'NY + Wakanda' could not be found as a U.S. state or a country", "nothing was substituted")),
    ("member_ambiguous", "Georgia + France", {"member": "Georgia", "state_form": "GA", "country_form": "Georgia (country)"},
     ("names both a U.S. state and a country", "'GA' for the state", "'Georgia (country)' for the country")),
    ("member_too_large", "France + Germany", {"member": "France", "name": "France", "cells": "15,866,824", "max_cells": "4,000,000"},
     ("'France' (France) covers about 15,866,824 data cells", "above the 4,000,000-cell limit")),
    ("combined_too_large", "Brazil + Japan", {"cells": "32,433,586", "max_cells": "4,000,000"},
     ("The parts of 'Brazil + Japan' are far apart", "32,433,586", "4,000,000-cell limit")),
    ("empty_member", "NY +", {}, ("'NY +' has an empty part",)),
    ("distance_unreadable", "within lots of NYC", {}, ("the distance could not be read",)),
    ("distance_unit_missing", "within 50 of NYC", {"amount": "50"},
     ("does not say whether 50 means miles or kilometres", "'within 50 miles of ...'")),
    ("distance_unit_unknown", "within 50 furlongs of NYC", {"unit": "furlongs"}, ("'furlongs' in 'within 50 furlongs of NYC'",)),
    ("centre_is_region", "within 50 km of New Jersey", {"centre": "New Jersey", "radius": "50 km"},
     ("'New Jersey' names a region, not a point", "within 50 km of it")),
    ("centre_not_found", "within 50 km of Atlantis", {"centre": "Atlantis", "radius": "50 km"},
     ("'Atlantis' could not be found as a place", "measure 50 km from", "nothing was substituted")),
    ("reaches_pole", "within 900 km of Alert", {"radius": "900 km", "centre": "Alert, Nunavut", "pole": "North Pole", "pole_km": "838"},
     ("within 900 km of Alert, Nunavut would reach the North Pole, 838 km away", "radius under 838 km")),
    ("crosses_antimeridian", "within 500 km of Suva", {"radius": "500 km", "centre": "Suva, Fiji"},
     ("within 500 km of Suva, Fiji crosses the 180th meridian",)),
    ("distance_too_large", "within 3000 miles of NYC",
     {"radius": "3000 miles", "centre": "New York, United States", "cells": "20,979,073", "max_cells": "4,000,000"},
     ("The area within 3000 miles of New York, United States covers about 20,979,073 data cells",
      "above the 4,000,000-cell limit", "smaller radius")),
]


class RefusalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.places = FakePlaces(refusals={w: PlaceRefused(code, facts) for code, w, facts, _ in REFUSALS})  # type: ignore[arg-type]
        self.env = await make_run_env(self._tmp.name, context=make_run_context(), places=self.places)

    def test_every_code_has_a_case_and_a_prompt(self):
        self.assertEqual({code for code, *_ in REFUSALS}, PLACE_REFUSAL_CODES)
        self.assertEqual(set(areas.REFUSAL_PROMPTS), PLACE_REFUSAL_CODES)

    async def test_a_refused_place_is_a_question_naming_what_was_wrong(self):
        for code, wording, _, phrases in REFUSALS:
            with self.subTest(code=code):
                result = await invoke(self.env, areas.resolve, {"place": wording}, step="area")
                self.assertEqual(result.status, "needs_input")
                assert result.question is not None
                self.assertEqual(result.question.kind, "text")
                self.assertEqual(result.values, {"refusal": code})
                for phrase in phrases:
                    self.assertIn(phrase, result.question.prompt)
                self.assertNotIn("{", result.question.prompt)
                self.assertIsNone(result.handle)

    async def test_a_refusal_whose_facts_do_not_fill_its_prompt_is_the_generic_bad_input(self):
        self.places.refusals["ny + nowhere"] = PlaceRefused("member_not_found", {})
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, areas.resolve, {"place": "NY + nowhere"})
        self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)
        self.assertEqual(caught.exception.reason, "sources.areas.resolve could not use the request as given.")

    def test_a_refusal_has_a_known_code_and_short_facts(self):
        with self.assertRaises(ValueError):
            PlaceRefused("no_such_code")  # type: ignore[arg-type]
        refused = PlaceRefused("member_not_found", {"member": "x" * 500 + " \n  y"})
        self.assertEqual(len(refused.facts["member"]), MAX_FACT_CHARS)
        self.assertNotIn("x" * 200, str(refused))

    async def test_a_long_member_is_quoted_short(self):
        self.places.refusals["ny + " + "z" * 300] = PlaceRefused("member_not_found", {"member": "z" * 300})
        result = await invoke(self.env, areas.resolve, {"place": "NY + " + "z" * 300})
        assert result.question is not None
        self.assertLessEqual(len(result.question.prompt), 500)
        self.assertNotIn("z" * 121, result.question.prompt)


class AreasTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.places = FakePlaces({"new jersey": NJ, "paris": PARIS_TX})
        self.context = make_run_context()
        self.env = await make_run_env(self._tmp.name, context=self.context, places=self.places)

    async def test_a_place_becomes_an_area_handle_with_name_bounds_and_wording(self):
        result = await invoke(self.env, areas.resolve, {"place": "New  Jersey"}, step="area")

        self.assertEqual((result.status, result.kind), ("done", "polygons"))
        self.assertEqual(result.summary, {"name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4], "region_type": "polygon"})
        self.assertEqual(result.disclosures, ())
        record = await self.env.resolver.get("user-a", result.handle, "polygons")
        self.assertEqual(record.descriptor.region, {
            "name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4], "wording": "New  Jersey", "region_type": "polygon",
        })
        self.assertEqual(record.operation, "sources.areas.resolve")
        collection = await self.env.resolver.open("user-a", result.handle, "polygons")
        geometry = areas.collection_geometry(collection)
        assert geometry is not None
        self.assertEqual(geometry["type"], "Polygon")
        self.assertEqual(self.places.asked, ["New  Jersey"])
        self.assertEqual(result.method, "The area 'New  Jersey' was resolved to New Jersey.")

    async def test_a_resolution_to_a_differently_named_place_is_disclosed(self):
        result = await invoke(self.env, areas.resolve, {"place": "Paris"})
        self.assertEqual(result.disclosures, ("'Paris' was resolved to Paris, Lamar County, Texas, United States.",))

    async def test_a_latitude_longitude_pair_is_a_point_area_without_asking_the_places(self):
        result = await invoke(self.env, areas.resolve, {"place": "40.735,-74.172 (Newark, NJ)"})

        self.assertEqual(result.status, "done")
        assert result.summary is not None
        self.assertEqual(result.summary["name"], "Newark, NJ (40.735, -74.172)")
        self.assertEqual(result.summary["region_type"], "point")
        west, south, east, north = result.summary["bounds"]
        self.assertAlmostEqual((west + east) / 2, -74.172)
        self.assertAlmostEqual((south + north) / 2, 40.735)
        self.assertEqual(self.places.asked, [])
        self.assertEqual(
            result.disclosures, ("'40.735,-74.172 (Newark, NJ)' was resolved to Newark, NJ (40.735, -74.172).",),
        )

    def test_only_one_decimal_latitude_longitude_pair_reads_as_a_point(self):
        for wording, name in (
            ("the single point 40.735,-74.172 (Newark, NJ)", "Newark, NJ (40.735, -74.172)"),
            ("40.7, -74.1", "40.7, -74.1"),
        ):
            with self.subTest(wording=wording):
                region = areas.point_region(wording)
                assert region is not None
                self.assertEqual(region.name, name)
        for wording in (
            "-75.0,39.0,-73.0,41.0", "within 50 km of Boston", "New Jersey", "95.0,10.0", "Route 1, 2 lanes",
            "40.7,-74.1 within 5 km", "40.7,-74.1 and 41.0,-73.9",
        ):
            with self.subTest(wording=wording):
                self.assertIsNone(areas.point_region(wording))

    async def test_an_unknown_place_is_a_question_and_nothing_is_minted(self):
        result = await invoke(self.env, areas.resolve, {"place": "Atlantis"})
        self.assertEqual(result.status, "needs_input")
        assert result.question is not None
        self.assertEqual(result.question.kind, "text")
        self.assertIn("'Atlantis' could not be found", result.question.prompt)
        self.assertIsNone(result.handle)
        [entry] = await self.context.storage.traces.entries("user-a", self.env.run_id)
        self.assertEqual(entry.status, "waiting")

    async def test_a_malformed_place_is_bad_input(self):
        env = await make_run_env(self._tmp.name, context=self.context, places=RaisingPlaces())
        with self.assertRaises(StepFailure) as caught:
            await invoke(env, areas.resolve, {"place": "NY + "})
        self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)

    async def test_the_area_helpers_read_the_facts_and_the_footprint(self):
        area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle
        record = await self.env.resolver.get("user-a", area)
        facts = areas.area_facts(record)
        self.assertEqual(facts, areas.AreaFacts(
            id=area, name="New Jersey", bounds=(-75.6, 38.9, -73.9, 41.4), wording="New Jersey", region_type="polygon",
        ))
        self.assertEqual(facts.region(), {
            "name": "New Jersey", "bounds": [-75.6, 38.9, -73.9, 41.4], "area": area, "wording": "New Jersey",
        })
        result = await invoke(self.env, READER, {"area": area, "recorded": area})
        self.assertEqual(result.values["type"], "Polygon")
        self.assertEqual(result.values["recorded"], [area, "Polygon"])

    async def test_a_handle_that_is_not_an_area_is_refused_as_bad_input(self):
        cases = {
            "no region": Descriptor(summary="Smoke", region="full product"),
            "bad bounds": Descriptor(summary="Odd", region={"name": "X", "bounds": [1, "2", 3, 4]}),
            "inverted bounds": Descriptor(summary="Odd", region={"name": "X", "bounds": [3.0, 2.0, 1.0, 4.0]}),
            "no name": Descriptor(summary="Odd", region={"bounds": [1.0, 2.0, 3.0, 4.0]}),
        }
        for label, descriptor in cases.items():
            with self.subTest(label):
                record = await self.env.resolver.mint(
                    user_id="user-a", conversation_id="conv-1", kind="polygons",
                    data={"type": "FeatureCollection", "features": []}, descriptor=descriptor,
                    operation="test.seed", created_at=NOW,
                )
                with self.assertRaises(StepFailure) as caught:
                    await invoke(self.env, READER, {"area": record.id})
                self.assertEqual(caught.exception.category, FailureCategory.BAD_INPUT)
                self.assertEqual(caught.exception.reason, (
                    "operations.read_area: area is not an area handle; resolve the place with sources.areas.resolve."
                ))

    async def test_an_area_without_a_footprint_or_an_unknown_recorded_area_is_refused(self):
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="polygons",
            data={"type": "FeatureCollection", "features": []},
            descriptor=Descriptor(summary="Empty", region={"name": "X", "bounds": [1.0, 2.0, 3.0, 4.0]}),
            operation="test.seed", created_at=NOW,
        )
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, READER, {"area": record.id})
        self.assertEqual(caught.exception.reason, "operations.read_area: the area handle area holds no footprint.")

        area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle
        with self.assertRaises(StepFailure) as unknown:
            await invoke(self.env, READER, {"area": area, "recorded": "polygons_" + "0" * 32})
        self.assertEqual(unknown.exception.category, FailureCategory.BAD_INPUT)

    async def test_an_evicted_area_is_input_evicted(self):
        area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle
        await self.env.resolver.evict("user-a", area, NOW)
        for params in ({"area": area}, ):
            with self.assertRaises(InputEvicted) as caught:
                await invoke(self.env, READER, params)
            self.assertEqual(caught.exception.handle_ids, (area,))

    def test_the_group_and_capabilities_register(self):
        registry = CapabilityRegistry()
        registry.add_group(areas.GROUP)
        for capability in areas.CAPABILITIES:
            registry.register(capability)
        self.assertEqual([c.name for c in registry.in_group("sources.areas")], ["sources.areas.resolve"])
        self.assertTrue(areas.GROUP.notes)


if __name__ == "__main__":
    unittest.main()
