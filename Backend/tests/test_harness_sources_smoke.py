"""sources.smoke.get_extent on the framework fakes and a fake HTTP transport; no network."""
import datetime
import io
import json
import tempfile
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

import httpx
import pandas as pd
import shapefile
from pyproj import CRS

from tta_backend.harness.capabilities import CapabilityRegistry, invoke
from tta_backend.harness.capabilities.sources import smoke
from tta_backend.harness.handles import Descriptor, HandleRecord
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
DATE = "2026-01-01"
ARCHIVE_PATH = "/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/2026/01/hms_smoke20260101.zip"

# (west, south, east, north), density, start, end
PLUMES = [
    ((-120, 35, -110, 40), "Light", "2026001 1200", "2026001 1400"),
    ((-100, 30, -95, 35), "Heavy", "2026001 1500", "2026001 1700"),
    ((-90, 40, -80, 45), "Medium", "2026001 1300", "2026001 1900"),
]


def smoke_archive(plumes: list[Any] = PLUMES) -> bytes:
    parts = {name: io.BytesIO() for name in ("shp", "shx", "dbf")}
    with shapefile.Writer(shp=parts["shp"], shx=parts["shx"], dbf=parts["dbf"], shapeType=shapefile.POLYGON) as writer:
        for name in ("Satellite", "Start", "End", "Density"):
            writer.field(name, "C", size=20)
        for (west, south, east, north), density, start, end in plumes:
            writer.poly([[[west, south], [west, north], [east, north], [east, south], [west, south]]])
            writer.record("GOES", start, end, density)
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        for ext, buffer in parts.items():
            archive.writestr(f"hms_smoke20260101.{ext}", buffer.getvalue())
        archive.writestr("hms_smoke20260101.prj", CRS.from_epsg(4326).to_wkt())
    return out.getvalue()


def serving(archive: bytes | None = None, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == ARCHIVE_PATH
        return httpx.Response(status, content=archive or b"")

    return httpx.MockTransport(handler)


def box_geometry(west: float, south: float, east: float, north: float) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[[west, south], [west, north], [east, north], [east, south], [west, south]]]}


def unreachable() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom: https://secret.example/token=abc")

    return httpx.MockTransport(handler)


class _SmokeCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        context = make_run_context(stores=self.stores)
        self.env: RunEnv = await make_run_env(self._tmp.name, context=context)

    capability = smoke.get_extent

    def serve(self, transport: httpx.MockTransport) -> None:
        self.capability = smoke.make_get_extent(transport)

    async def seed_area(self, region: Any, geometry: dict[str, Any] | None = None, *, footprint: bool = True) -> HandleRecord:
        """An area handle; its footprint defaults to its bounding box."""
        if geometry is None and footprint and isinstance(region, dict) and len(region.get("bounds") or []) == 4:
            geometry = box_geometry(*region["bounds"])
        features = [] if geometry is None else [{"type": "Feature", "geometry": geometry, "properties": {}}]
        return await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="polygons",
            data={"type": "FeatureCollection", "features": features},
            descriptor=Descriptor(summary="An area", region=region), operation="test.seed", created_at=NOW,
        )

    async def extent(self, params: dict[str, Any], **kwargs: Any) -> Any:
        return await invoke(self.env, self.capability, params, **kwargs)

    async def features(self, handle: str) -> list[dict[str, Any]]:
        data = await self.env.resolver.open("user-a", handle, "polygons")
        return data["features"]  # type: ignore[no-any-return]

    async def entries(self) -> list[Any]:
        return await self.stores.traces.entries("user-a", self.env.run_id)

    async def failure(self, params: dict[str, Any]) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, self.capability, params)
        return caught.exception


class GetExtentTests(_SmokeCase):
    async def test_a_date_gives_a_polygons_handle_with_density_and_window_per_plume(self):
        self.serve(serving(smoke_archive()))
        result = await self.extent({"date": DATE})

        self.assertEqual((result.status, result.kind, result.product), ("done", "polygons", "NOAA HMS smoke"))
        self.assertEqual(result.summary["plumes"], 3)
        self.assertEqual(result.summary["by_density"], {"light": 1, "medium": 1, "heavy": 1, "unknown": 0})
        self.assertEqual(result.summary["first_start"], "2026-01-01T12:00:00Z")
        self.assertEqual(result.summary["last_end"], "2026-01-01T19:00:00Z")
        # 10x5 + 5x5 + 10x5 degrees: a positive area well under the whole earth.
        self.assertGreater(result.summary["area_km2"], 1_000_000)
        self.assertEqual(result.method, "NOAA HMS smoke for 2026-01-01 was retrieved, giving 3 plume(s).")
        self.assertEqual(result.values["source_url"].rsplit("/", 1)[1], "hms_smoke20260101.zip")
        self.assertEqual(len(result.values["sha256"]), 64)
        self.assertTrue(any("not surface PM2.5" in text for text in result.disclosures))
        self.assertIn("Polygon counts are not counts of fires.", result.disclosures)

        collection = await self.env.resolver.open("user-a", result.handle, "polygons")
        self.assertEqual(collection["type"], "FeatureCollection")
        first = collection["features"][0]
        self.assertEqual(first["type"], "Feature")
        self.assertEqual(first["geometry"]["type"], "Polygon")
        self.assertEqual(
            first["properties"],
            {"satellite": "GOES", "start": "2026-01-01T12:00:00Z", "end": "2026-01-01T14:00:00Z",
             "density": "light", "density_raw": "Light"},
        )

        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.descriptor.region, "full product")
        self.assertEqual((record.descriptor.period.start, record.descriptor.period.end), (DATE, DATE))
        self.assertEqual(record.descriptor.product, "NOAA HMS smoke")
        self.assertEqual(record.parents, ())

    async def test_an_area_handle_keeps_only_the_plumes_over_it(self):
        self.serve(serving(smoke_archive()))
        area = await self.seed_area({"name": "Utah-ish", "bounds": [-122, 34, -108, 41], "wording": "utah"})
        result = await self.extent({"date": DATE, "area": area.id})

        self.assertEqual(result.summary["plumes"], 1)
        self.assertEqual(result.summary["by_density"]["light"], 1)
        [feature] = await self.features(result.handle)
        self.assertEqual(feature["properties"]["density"], "light")
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (area.id,))
        self.assertEqual(record.descriptor.region, {"name": "Utah-ish", "bounds": [-122.0, 34.0, -108.0, 41.0],
                                                    "area": area.id, "wording": "utah"})
        self.assertIn("Utah-ish", record.descriptor.summary)

    async def test_the_area_footprint_in_the_payload_clips_the_plume_not_its_bounding_box(self):
        self.serve(serving(smoke_archive()))
        # An L-shaped area: its bounding box holds the whole first plume, its
        # footprint only the plume's western half.
        west_half = box_geometry(-120, 35, -115, 40)
        strip = box_geometry(-120, 40, -110, 41)
        footprint = {"type": "MultiPolygon", "coordinates": [west_half["coordinates"], strip["coordinates"]]}
        area = await self.seed_area({"name": "L", "bounds": [-120, 35, -110, 41]}, footprint)
        result = await self.extent({"date": DATE, "area": area.id})

        boxed = await self.seed_area({"name": "Box", "bounds": [-120, 35, -110, 41]})
        whole = (await self.extent({"date": DATE, "area": boxed.id})).summary["area_km2"]
        self.assertEqual(result.summary["plumes"], 1)
        self.assertAlmostEqual(result.summary["area_km2"] / whole, 0.5, delta=0.02)
        [feature] = await self.features(result.handle)
        self.assertLessEqual(max(x for x, _ in feature["geometry"]["coordinates"][0]), -115)
        self.assertIn("Plumes were clipped to the boundary of L; area_km2 counts only the smoke inside it.", result.disclosures)

    async def test_an_area_without_usable_bounds_is_bad_input(self):
        self.serve(serving(smoke_archive()))
        for region in (None, "Utah", {"name": "No bounds"}, {"name": "A", "bounds": [-108, 34, -122, 41]},
                       {"name": "A", "bounds": [1, 2, 3]}, {"bounds": [-122, 34, -108, 41]}):
            with self.subTest(region=region):
                area = await self.seed_area(region)
                failure = await self.failure({"date": DATE, "area": area.id})
                self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
                self.assertEqual(failure.reason, "sources.smoke.get_extent: area is not an area handle; "
                                                 "resolve the place with sources.areas.resolve.")

    async def test_an_area_without_a_footprint_is_bad_input(self):
        self.serve(serving(smoke_archive()))
        area = await self.seed_area({"name": "Empty", "bounds": [-122, 34, -108, 41]}, footprint=False)

        failure = await self.failure({"date": DATE, "area": area.id})

        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        self.assertEqual(failure.reason, "sources.smoke.get_extent: the area handle area holds no footprint.")

    async def test_no_plumes_is_done_with_zero_and_a_disclosure(self):
        self.serve(serving(smoke_archive()))
        area = await self.seed_area({"name": "Ocean", "bounds": [0, 0, 10, 10]})
        result = await self.extent({"date": DATE, "area": area.id})

        self.assertEqual(result.status, "done")
        self.assertEqual(result.summary["plumes"], 0)
        self.assertEqual(result.summary["area_km2"], 0.0)
        self.assertNotIn("first_start", result.summary)
        self.assertIn("No smoke was analysed", result.disclosures[0])
        self.assertIn("does not establish smoke-free", result.disclosures[0])
        self.assertEqual(await self.features(result.handle), [])

    async def test_an_empty_product_is_also_zero_plumes(self):
        self.serve(serving(smoke_archive([])))
        result = await self.extent({"date": DATE})
        self.assertEqual((result.status, result.summary["plumes"]), ("done", 0))

    async def test_a_missing_product_is_insufficient_evidence_with_fixed_wording(self):
        self.serve(serving(status=404))
        result = await self.extent({"date": DATE})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.handle)
        self.assertEqual(result.reason, smoke.MISSING_PRODUCT)
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.NO_DATA))

    async def test_a_network_failure_is_source_unavailable_without_exception_text(self):
        self.serve(unreachable())
        with patch("tta_backend.domain.smoke.hms.asyncio.sleep", new=AsyncMock()):
            failure = await self.failure({"date": DATE})

        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertEqual(
            failure.reason,
            "sources.smoke.get_extent could not reach its data source, which is temporarily unavailable.",
        )
        self.assertNotIn("secret.example", json.dumps([e.reason for e in await self.entries()]))

    async def test_a_server_error_is_source_unavailable(self):
        self.serve(serving(status=400))
        failure = await self.failure({"date": DATE})
        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)

    async def test_a_corrupt_archive_is_an_internal_failure(self):
        self.serve(serving(b"not a zip"))
        failure = await self.failure({"date": DATE})
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)

    async def test_a_bad_or_future_date_is_bad_input(self):
        self.serve(serving(smoke_archive()))
        for value in ("2026-02-30", "yesterday", "2999-01-01"):
            with self.subTest(value=value):
                failure = await self.failure({"date": value})
                self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
                self.assertEqual(
                    failure.reason,
                    "sources.smoke.get_extent needs a date that is not in the future, as YYYY-MM-DD.",
                )

    async def test_the_area_must_be_a_polygons_handle(self):
        table = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="table", data=pd.DataFrame({"a": [1]}),
            descriptor=Descriptor(summary="Rows"), operation="test.seed", created_at=NOW,
        )
        failure = await self.failure({"date": DATE, "area": table.id})
        self.assertEqual(failure.reason, "sources.smoke.get_extent: area must be a handle of kind polygons.")

    async def test_one_use_entry_records_the_call(self):
        self.serve(serving(smoke_archive()))
        area = await self.seed_area({"name": "Utah-ish", "bounds": [-122, 34, -108, 41]})
        result = await self.extent({"date": DATE, "area": area.id}, step="extent")

        [entry] = await self.entries()
        self.assertEqual(
            (entry.kind, entry.capability, entry.step, entry.status),
            ("use", "sources.smoke.get_extent", "extent", "done"),
        )
        self.assertEqual(entry.params, {"date": DATE, "area": area.id})
        self.assertEqual(entry.result["handle"], result.handle)
        self.assertEqual(entry.result["summary"]["plumes"], 1)
        self.assertEqual(entry.result["method"], result.method)


class RegistrationTests(unittest.TestCase):
    def test_the_module_registers_cleanly(self):
        registry = CapabilityRegistry()
        registry.add_group(smoke.GROUP)
        for capability in smoke.CAPABILITIES:
            registry.register(capability)
        self.assertEqual(registry.in_group("sources.smoke"), (smoke.get_extent,))
        schema = registry.schema("sources.smoke.get_extent")
        self.assertEqual(schema["properties"]["area"]["x-handle-kinds"], ["polygons"])
        self.assertEqual(schema["required"], ["date"])


if __name__ == "__main__":
    unittest.main()
