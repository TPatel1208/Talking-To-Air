"""Typed handle records, their payload store and the resolver, on the in-memory registry."""
import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import xarray as xr
from pydantic import SecretStr, ValidationError

from tests.harness_handle_helpers import (
    LATER,
    NOW,
    TRAVERSAL_IDS,
    describe,
    grid_dataset,
    monitor_readings,
    retrieval_record,
    scalar_record,
    table_record,
)
from tta_backend.harness.handles import (
    KINDS,
    HandleError,
    HandleRecord,
    HandleResolver,
    InMemoryHandleRegistry,
    InlineScalar,
    PayloadStore,
    Period,
    Scalar,
    ScalarInput,
    new_handle_id,
    require_kind,
)
from tta_backend.harness.handles import payloads as payloads_module
from tta_backend.harness.handles.model import RETRIEVAL_PREFIXES, is_derived_id, is_retrieval_id
from tta_backend.harness.handles.payloads import relative_path
from tta_backend.harness.release import HARNESS_RELEASE

RUN_ID = "run_" + "0123456789abcdef" * 2


class RecordModelTests(unittest.TestCase):
    def test_derived_ids_never_collide_with_service_prefixes(self):
        for kind in KINDS:
            handle_id = new_handle_id(kind)
            self.assertTrue(handle_id.startswith(f"{kind}_"))
            self.assertTrue(is_derived_id(handle_id))
            self.assertFalse(is_retrieval_id(handle_id))
            self.assertNotIn(kind, RETRIEVAL_PREFIXES)
        self.assertNotEqual(new_handle_id("grid"), new_handle_id("grid"))

    def test_traversal_shaped_ids_are_not_handle_ids(self):
        for handle_id in TRAVERSAL_IDS:
            with self.subTest(handle_id=handle_id):
                self.assertFalse(is_derived_id(handle_id))
                self.assertFalse(is_retrieval_id(handle_id))

    def test_derived_id_prefix_must_be_its_kind(self):
        with self.assertRaises(ValidationError):
            HandleRecord(**{**scalar_record("u").model_dump(), "id": new_handle_id("grid")})

    def test_retrieval_record_keeps_service_id_and_has_no_local_lineage(self):
        record = retrieval_record("u")
        self.assertEqual(record.id, "obs_tempo_no2_1")
        self.assertEqual(record.release, HARNESS_RELEASE)
        base = record.model_dump()
        for bad in ({"id": "obs_../x"}, {"id": new_handle_id("grid")}, {"parents": ("obs_other",)}, {"operation": "mean"}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                HandleRecord(**{**base, **bad})

    def test_payload_must_match_kind(self):
        record = table_record("u")
        with self.assertRaises(ValidationError):
            HandleRecord(**{**record.model_dump(), "payload": {"state": "stored", "format": "zarr", "path": "x"}})
        with self.assertRaises(ValidationError):
            HandleRecord(**{**record.model_dump(), "payload": {"state": "service"}})

    def test_parameters_are_json_only(self):
        for bad in (SecretStr("hunter2"), b"bytes", object(), (1, 2)):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValidationError):
                scalar_record("u", parameters={"value": bad})

    def test_run_id_is_optional_and_well_formed(self):
        self.assertIsNone(scalar_record("u").run_id)
        self.assertIsNone(retrieval_record("u").run_id)
        self.assertEqual(scalar_record("u", run_id=RUN_ID).run_id, RUN_ID)
        for bad in ("", "run_", "run_" + "A" * 32, "run_" + "a" * 31, "trace_" + "a" * 32, RUN_ID + "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                scalar_record("u", run_id=bad)

    def test_non_finite_floats_become_null(self):
        record = scalar_record("u", value=float("inf"), parameters={"threshold": float("nan"), "ok": [1.0, float("-inf")]})
        self.assertIsNone(record.payload.value)
        self.assertEqual(record.parameters, {"threshold": None, "ok": [1.0, None]})

    def test_summary_is_one_line_and_created_at_is_aware(self):
        with self.assertRaises(ValidationError):
            describe("two\nlines")
        with self.assertRaises(ValidationError):
            HandleRecord(**{**scalar_record("u").model_dump(), "created_at": datetime.datetime(2026, 1, 1)})
        with self.assertRaises(ValidationError):
            Period(start="last tuesday", end="2026-01-02")

    def test_record_is_immutable_and_round_trips_json(self):
        record = HandleRecord(
            **{
                **table_record("u", parents=("obs_a",)).model_dump(),
                "descriptor": describe("x", region={"bbox": [-119, 33, -117, 35]}, period=Period(start="2026-01-01", end="2026-01-31T23:00:00+00:00"), masking={"qa": "good"}),
            }
        )
        with self.assertRaises(ValidationError):
            record.kind = "grid"
        self.assertEqual(HandleRecord.model_validate_json(record.model_dump_json()), record)

    def test_require_kind(self):
        record = table_record("u")
        self.assertIs(require_kind(record, "table", "series"), record)
        with self.assertRaises(HandleError) as caught:
            require_kind(record, "grid")
        self.assertEqual(caught.exception.reason, "wrong_kind")


class _StoreCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.payloads = PayloadStore(self.root)
        self.registry = InMemoryHandleRegistry()
        self.opened_retrievals = []

        async def open_retrieval(user_id, record):
            self.opened_retrievals.append((user_id, record.id))
            return {"opened": record.id}

        self.resolver = HandleResolver(self.registry, self.payloads, open_retrieval)

    def tearDown(self):
        self._tmp.cleanup()

    def files(self):
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def top_level_entries(self):
        return sorted(p.name for d in self.root.iterdir() for p in d.iterdir())

    async def mint(self, kind, data, user_id="user-a", parents=(), **kwargs):
        return await self.resolver.mint(
            user_id=user_id,
            conversation_id="conv-1",
            kind=kind,
            data=data,
            descriptor=kwargs.pop("descriptor", describe(f"A {kind}")),
            operation=kwargs.pop("operation", "test_op"),
            created_at=NOW,
            parents=parents,
            **kwargs,
        )


class PayloadStoreTests(_StoreCase):
    async def test_each_kind_round_trips(self):
        grid = await self.mint("grid", grid_dataset())
        series = await self.mint("series", monitor_readings())
        points = await self.mint("points", pd.DataFrame({"site": ["a", "b"], "lat": [34.0, 34.1], "lon": [-118.0, -118.1]}))
        table = await self.mint("table", pd.DataFrame({"site": ["a"], "exceedance_days": [3]}))
        geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}
        polygons = await self.mint("polygons", geojson)
        scalar = await self.mint("scalar", ScalarInput(value=42.5, uncertainty={"sd": 1.2}), descriptor=describe("Mean", unit="ppb"))

        opened = await self.resolver.open("user-a", grid.id, "grid")
        xr.testing.assert_equal(opened["no2"].load(), grid_dataset())
        pd.testing.assert_frame_equal(await self.resolver.open("user-a", series.id), monitor_readings())
        self.assertEqual(list((await self.resolver.open("user-a", points.id))["site"]), ["a", "b"])
        self.assertEqual(int((await self.resolver.open("user-a", table.id))["exceedance_days"][0]), 3)
        self.assertEqual(await self.resolver.open("user-a", polygons.id), geojson)
        self.assertEqual(await self.resolver.open("user-a", scalar.id), Scalar(value=42.5, unit="ppb", uncertainty={"sd": 1.2}))
        self.assertIsInstance(scalar.payload, InlineScalar)

    async def test_writes_leave_no_staging_and_failures_leave_nothing(self):
        await self.mint("table", pd.DataFrame({"a": [1]}))
        await self.mint("grid", grid_dataset())
        self.assertFalse(any(".staging-" in name for name in self.top_level_entries()))
        before = self.top_level_entries()
        for kind, bad in (("table", {"not": "a frame"}), ("grid", pd.DataFrame()), ("polygons", {"no": "type"}), ("polygons", {"type": "Point", "coordinates": [float("nan"), 0]})):
            with self.subTest(kind=kind), self.assertRaises((TypeError, ValueError)):
                await self.mint(kind, bad)
        self.assertEqual(self.top_level_entries(), before)

    async def test_user_ids_never_shape_paths(self):
        record = await self.mint("table", pd.DataFrame({"a": [1]}), user_id="../../escape")
        self.assertTrue(all(".." not in part for part in Path(record.payload.path).parts))
        self.assertEqual(len(self.files()), 1)
        self.assertTrue((self.root / self.files()[0]).resolve().is_relative_to(self.root.resolve()))
        a = relative_path("user-a", record.id, "parquet")
        b = relative_path("user-b", record.id, "parquet")
        self.assertNotEqual(a.split("/")[0], b.split("/")[0])

    async def test_forged_path_into_another_users_data_is_refused(self):
        theirs = await self.mint("table", pd.DataFrame({"secret": [1]}), user_id="user-a")
        forged = HandleRecord(**{**theirs.model_dump(), "user_id": "user-b"})
        with self.assertRaises(HandleError) as caught:
            await self.payloads.open(forged)
        self.assertEqual(caught.exception.reason, "not_found")
        traversal = HandleRecord(**{**theirs.model_dump(), "payload": {"state": "stored", "format": "parquet", "path": "../outside.parquet"}})
        with self.assertRaises(HandleError):
            await self.payloads.open(traversal)

    async def test_single_file_payloads_are_synced_before_the_rename(self):
        events = []
        real_fsync, real_rename = os.fsync, os.rename
        with (
            mock.patch.object(payloads_module.os, "fsync", side_effect=lambda fd: (events.append("fsync"), real_fsync(fd))),
            mock.patch.object(payloads_module.os, "rename", side_effect=lambda a, b: (events.append("rename"), real_rename(a, b))),
        ):
            await self.mint("table", pd.DataFrame({"a": [1]}))
            await self.mint("polygons", {"type": "Point", "coordinates": [0, 0]})
        self.assertEqual(events, ["fsync", "rename", "fsync", "rename"])

    async def test_missing_bytes_read_as_evicted(self):
        record = await self.mint("table", pd.DataFrame({"a": [1]}))
        os.remove(self.root / record.payload.path)
        with self.assertRaises(HandleError) as caught:
            await self.resolver.open("user-a", record.id)
        self.assertEqual(caught.exception.reason, "evicted")


class ResolverTests(_StoreCase):
    async def test_open_checks_kind_before_reading(self):
        record = await self.mint("table", pd.DataFrame({"a": [1]}))
        with self.assertRaises(HandleError) as caught:
            await self.resolver.open("user-a", record.id, "grid", "series")
        self.assertEqual(caught.exception.reason, "wrong_kind")

    async def test_retrieval_handles_open_through_the_injected_callable(self):
        await self.registry.record_retrieval(retrieval_record("user-a"))
        self.assertEqual(await self.resolver.open("user-a", "obs_tempo_no2_1", "grid"), {"opened": "obs_tempo_no2_1"})
        self.assertEqual(self.opened_retrievals, [("user-a", "obs_tempo_no2_1")])

    async def test_another_users_retrieval_handle_is_never_opened(self):
        await self.registry.record_retrieval(retrieval_record("user-a"))
        with self.assertRaises(HandleError) as caught:
            await self.resolver.open("user-b", "obs_tempo_no2_1")
        self.assertEqual(caught.exception.reason, "not_found")
        self.assertEqual(self.opened_retrievals, [])

    async def test_mint_from_retrieval_parent(self):
        await self.registry.record_retrieval(retrieval_record("user-a"))
        series = await self.mint("series", monitor_readings(), parents=("obs_tempo_no2_1",), operation="area_mean", parameters={"dims": ["lat", "lon"]})
        self.assertEqual(series.parents, ("obs_tempo_no2_1",))
        self.assertEqual((await self.registry.get("user-a", series.id)).parameters, {"dims": ["lat", "lon"]})

    async def test_parent_of_another_user_is_refused_before_any_write(self):
        theirs = await self.mint("table", pd.DataFrame({"a": [1]}), user_id="user-a")
        before = self.files()
        for parent in (theirs.id, "obs_tempo_no2_1", "../../x"):
            with self.subTest(parent=parent), self.assertRaises((HandleError, ValidationError)) as caught:
                await self.mint("table", pd.DataFrame({"b": [2]}), user_id="user-b", parents=(parent,))
            if isinstance(caught.exception, HandleError):
                self.assertEqual(caught.exception.reason, "parent_not_found")
        self.assertEqual(self.files(), before)

    async def test_refused_record_leaves_no_payload(self):
        with self.assertRaises(ValidationError):
            await self.mint("table", pd.DataFrame({"a": [1]}), parameters={"raw": b"bytes"})
        self.assertEqual(self.files(), [])

    async def test_mint_records_the_run(self):
        inside = await self.mint("table", pd.DataFrame({"a": [1]}), run_id=RUN_ID)
        outside = await self.mint("scalar", ScalarInput(value=1))
        self.assertEqual((await self.resolver.get("user-a", inside.id)).run_id, RUN_ID)
        self.assertIsNone((await self.resolver.get("user-a", outside.id)).run_id)

    async def test_a_failed_delete_is_finished_by_retrying_the_eviction(self):
        record = await self.mint("grid", grid_dataset())
        real_delete = self.payloads.delete
        with mock.patch.object(self.payloads, "delete", side_effect=OSError("disk busy")):
            with self.assertRaises(OSError):
                await self.resolver.evict("user-a", record.id, LATER)
        self.assertEqual((await self.resolver.get("user-a", record.id)).payload.state, "evicted")
        self.assertNotEqual(self.files(), [])
        with mock.patch.object(self.payloads, "delete", side_effect=real_delete) as delete:
            evicted = await self.resolver.evict("user-a", record.id, LATER)
        delete.assert_awaited_once()
        self.assertEqual(evicted.payload.evicted_at, LATER)
        self.assertEqual(self.files(), [])

    async def test_evict_keeps_record_and_open_names_the_reason(self):
        record = await self.mint("grid", grid_dataset())
        evicted = await self.resolver.evict("user-a", record.id, LATER)
        self.assertEqual(evicted.payload.state, "evicted")
        self.assertEqual(evicted.payload.evicted_at, LATER)
        self.assertEqual(self.files(), [])
        self.assertEqual((await self.resolver.get("user-a", record.id)).descriptor, record.descriptor)
        with self.assertRaises(HandleError) as caught:
            await self.resolver.open("user-a", record.id)
        self.assertEqual(caught.exception.reason, "evicted")
        self.assertEqual((await self.resolver.evict("user-a", record.id, LATER)).payload.state, "evicted")

    async def test_scalars_and_retrievals_are_not_evictable(self):
        scalar = await self.mint("scalar", ScalarInput(value=1))
        await self.registry.record_retrieval(retrieval_record("user-a"))
        for handle_id in (scalar.id, "obs_tempo_no2_1"):
            with self.subTest(handle_id=handle_id), self.assertRaises(HandleError) as caught:
                await self.resolver.evict("user-a", handle_id, LATER)
            self.assertEqual(caught.exception.reason, "not_evictable")


class IsolationTests(_StoreCase):
    async def assert_not_found(self, coro):
        with self.assertRaises(HandleError) as caught:
            await coro
        self.assertEqual(caught.exception.reason, "not_found")

    async def test_other_users_handle_is_indistinguishable_from_none(self):
        record = await self.mint("table", pd.DataFrame({"secret": [1]}), user_id="user-a")
        missing = new_handle_id("table")
        await self.assert_not_found(self.resolver.get("user-b", record.id))
        await self.assert_not_found(self.resolver.get("user-b", missing))
        await self.assert_not_found(self.resolver.open("user-b", record.id))
        await self.assert_not_found(self.resolver.open("user-b", missing))
        await self.assert_not_found(self.resolver.open("user-b", record.id, "grid"))
        await self.assert_not_found(self.resolver.evict("user-b", record.id, LATER))
        self.assertEqual(self.files(), [record.payload.path])
        self.assertIsNotNone(await self.resolver.open("user-a", record.id))

    async def test_traversal_shaped_ids_are_not_found(self):
        await self.mint("table", pd.DataFrame({"a": [1]}))
        for handle_id in TRAVERSAL_IDS:
            with self.subTest(handle_id=handle_id):
                await self.assert_not_found(self.resolver.open("user-a", handle_id))
                await self.assert_not_found(self.resolver.evict("user-a", handle_id, LATER))

    async def test_same_retrieval_id_under_two_users(self):
        a = await self.registry.record_retrieval(retrieval_record("user-a", summary="A's view"))
        b = await self.registry.record_retrieval(retrieval_record("user-b", summary="B's view"))
        self.assertEqual((await self.resolver.get("user-a", a.id)).descriptor.summary, "A's view")
        self.assertEqual((await self.resolver.get("user-b", b.id)).descriptor.summary, "B's view")
        await self.resolver.open("user-b", b.id)
        self.assertEqual(self.opened_retrievals, [("user-b", "obs_tempo_no2_1")])

    async def test_stored_record_cannot_be_mutated_through_a_returned_copy(self):
        record = await self.mint("table", pd.DataFrame({"a": [1]}), parameters={"window": [1, 2]})
        fetched = await self.resolver.get("user-a", record.id)
        fetched.parameters["window"].append(3)
        self.assertEqual((await self.resolver.get("user-a", record.id)).parameters, {"window": [1, 2]})


class GroundMonitorTests(_StoreCase):
    """Monitor data uses the ordinary kinds; nothing here is specific to the source."""

    async def test_readings_locations_and_exceedances(self):
        readings = await self.mint(
            "series",
            monitor_readings(),
            operation="fetch_monitor_readings",
            parameters={"parameter": "88101", "site": "06-037-1103", "start": "2026-01-01", "end": "2026-01-01"},
            descriptor=describe("PM2.5 hourly at one monitor", quantity="pm25", unit="ug/m3", product="AQS"),
        )
        locations = await self.mint(
            "points",
            pd.DataFrame({"site": ["06-037-1103", "06-037-4004"], "lat": [34.06, 33.92], "lon": [-118.23, -118.20]}),
            operation="fetch_monitor_sites",
            descriptor=describe("PM2.5 monitors in LA county", product="AQS"),
        )
        exceedances = await self.mint(
            "table",
            pd.DataFrame({"site": ["06-037-1103"], "days_over_standard": [2]}),
            parents=(readings.id,),
            operation="exceedance_days",
            parameters={"standard": 35.0},
            descriptor=describe("Days over the 24h PM2.5 standard", product="AQS"),
        )
        satellite = await self.mint("series", pd.DataFrame({"time": [NOW], "value": [12.0]}), operation="area_mean")

        self.assertEqual((readings.kind, locations.kind, exceedances.kind), ("series", "points", "table"))
        # A step that takes series accepts monitor and satellite series alike.
        for record in (readings, satellite):
            await self.resolver.open("user-a", record.id, "series")
        with self.assertRaises(HandleError):
            await self.resolver.open("user-a", locations.id, "series")
        pd.testing.assert_frame_equal(await self.resolver.open("user-a", readings.id), monitor_readings())


if __name__ == "__main__":
    unittest.main()
