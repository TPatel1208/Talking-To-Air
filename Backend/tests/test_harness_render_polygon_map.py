"""render.polygon_map on the framework fakes: payload, storage, empty maps and refusals."""
import unittest
from typing import Any

from tests.test_harness_render import RenderCase, points
from tta_backend.harness.capabilities.render import polygon_map
from tta_backend.harness.trace import FailureCategory


def plume(*corners: tuple[float, float], density: str = "light") -> dict[str, Any]:
    ring = [list(c) for c in corners] + [list(corners[0])]
    return {"type": "Feature", "properties": {"density": density}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


class PolygonMapTests(RenderCase):
    async def test_a_chart_is_stored_and_emitted_with_its_sources(self):
        smoke = await self.seed(
            "polygons", collection(plume((-75, 39), (-74, 39), (-74, 40)), plume((-73, 41), (-72, 41), (-72, 42))),
            product="NOAA HMS",
        )
        result = await invoke_map(self, {"polygons": [smoke.id], "title": "Smoke"})

        self.assertEqual(result.status, "done")
        payload = self.assert_stored_and_emitted(result)
        self.assertEqual(payload["type"], "polygon_map")
        self.assertEqual(payload["title"], "Smoke")
        self.assertEqual(payload["bounds"], [-75.0, 39.0, -72.0, 42.0])
        self.assertEqual(payload["polygons"][0]["handle"], smoke.id)
        self.assertEqual(len(payload["polygons"][0]["geojson"]["features"]), 2)
        self.assertEqual(payload["metadata"]["source_handles"], [smoke.id])
        self.assertEqual(payload["provenance"]["source_handles"], [smoke.id])
        self.assertEqual(payload["export"]["source_handles"], [smoke.id])
        self.assertEqual(result.values["features"], 2)
        self.assertEqual(result.summary["render_type"], "polygon_map")
        self.assertIn("2 feature", result.method or "")

    async def test_several_polygons_handles_share_one_map(self):
        a = await self.seed("polygons", collection(plume((-75, 39), (-74, 39), (-74, 40))))
        b = await self.seed("polygons", collection(plume((-80, 35), (-79, 35), (-79, 36), density="heavy")))
        result = await invoke_map(self, {"polygons": [a.id, b.id]})
        payload = self.assert_stored_and_emitted(result)
        self.assertEqual([p["handle"] for p in payload["polygons"]], [a.id, b.id])
        self.assertEqual(payload["bounds"], [-80.0, 35.0, -74.0, 40.0])
        self.assertEqual(payload["metadata"]["source_handles"], [a.id, b.id])
        self.assertEqual(result.values["features"], 2)
        self.assertEqual(result.values["handles"], 2)

    async def test_no_polygons_is_an_empty_map_with_a_disclosure(self):
        empty = await self.seed("polygons", collection())
        result = await invoke_map(self, {"polygons": [empty.id]})
        self.assertEqual(result.status, "done")
        payload = self.assert_stored_and_emitted(result)
        self.assertIsNone(payload["bounds"])
        self.assertEqual(result.values["features"], 0)
        self.assertIn(polygon_map.NO_POLYGONS, result.disclosures)

    async def test_wrong_handle_kinds_are_refused(self):
        monitors = await self.seed_points(points([0.05, 0.06, 0.07]))
        failure = await self.failure(polygon_map.render_polygon_map, {"polygons": [monitors.id]})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        smoke = await self.seed("polygons", collection())
        failure = await self.failure(polygon_map.render_polygon_map, {"polygons": [smoke.id], "points": smoke.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(self.artifacts.stored, {})

    async def test_points_are_carried_as_an_overlay(self):
        smoke = await self.seed("polygons", collection(plume((-75, 39), (-74, 39), (-74, 40))))
        monitors = await self.seed_points(points([0.05, None, 0.07]))
        result = await invoke_map(self, {"polygons": [smoke.id], "points": monitors.id})
        payload = self.assert_stored_and_emitted(result)
        overlay = payload["overlays"]["points"]
        self.assertEqual(overlay["handle"], monitors.id)
        self.assertEqual([p["id"] for p in overlay["points"]], ["34-001-0000", "34-001-0001", "34-001-0002"])
        self.assertEqual(overlay["points"][0]["value"], 0.05)
        self.assertEqual(payload["metadata"]["source_handles"], [smoke.id, monitors.id])
        self.assertEqual(result.values["overlay_points"], 3)

    async def test_the_step_is_traced(self):
        smoke = await self.seed("polygons", collection())
        result = await invoke_map(self, {"polygons": [smoke.id]})
        [entry] = await self.entries()
        self.assertEqual((entry.capability, entry.status), ("render.polygon_map", "done"))
        self.assertEqual(entry.result["charts"], list(result.charts))


async def invoke_map(case: RenderCase, params: dict[str, Any]) -> Any:
    from tta_backend.harness.capabilities import invoke

    return await invoke(case.env, polygon_map.render_polygon_map, params)


if __name__ == "__main__":
    unittest.main()
