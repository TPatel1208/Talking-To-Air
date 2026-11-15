"""sources.areas.resolve: a place in the user's words becomes an area handle, or a question."""
import datetime
import tempfile
import unittest
from typing import Annotated, Any

from tta_backend.harness.capabilities import Call, Capability, CapabilityRegistry, CapabilityResult, InputEvicted, Params, handle_param, invoke
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor
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
