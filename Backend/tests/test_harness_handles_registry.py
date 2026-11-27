"""One contract for every handle registry, run against memory and a real Postgres."""
import tempfile
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from unittest import mock

import pandas as pd
from pydantic import ValidationError

from tests.harness_handle_helpers import (
    LATER,
    NOW,
    TRAVERSAL_IDS,
    FakeProvenance,
    describe,
    grid_dataset,
    monitor_readings,
    retrieval_record,
    scalar_record,
    table_record,
)
from tests.postgres_helpers import PostgresTestCase, isolated_schema
from tta_backend.harness.handles import (
    HandleError,
    HandleRecord,
    HandleResolver,
    InMemoryHandleRegistry,
    PayloadStore,
    PostgresHandleRegistry,
    ScalarInput,
    dataset_handles,
    ensure_tables,
    lineage,
    new_handle_id,
)
from tta_backend.harness.release import HARNESS_RELEASE


class RegistryContract:
    """Mixed into a TestCase whose asyncSetUp sets ``self.registry``."""

    async def assert_reason(self, reason, coro):
        with self.assertRaises(HandleError) as caught:
            await coro
        self.assertEqual(caught.exception.reason, reason)

    async def test_mint_and_get_round_trip(self):
        parent = await self.registry.record_retrieval(retrieval_record("user-a"))
        record = scalar_record("user-a", parents=(parent.id,), parameters={"dims": ["lat"], "threshold": float("nan"), "nested": {"k": 1.5}})
        await self.registry.mint(record)
        fetched = await self.registry.get("user-a", record.id)
        self.assertEqual(fetched, record)
        self.assertEqual(fetched.release, HARNESS_RELEASE)
        self.assertEqual(fetched.created_at, NOW)
        self.assertEqual(fetched.parameters["threshold"], None)
        table = table_record("user-a", parents=(record.id,))
        await self.registry.mint(table)
        self.assertEqual(await self.registry.get("user-a", table.id), table)

    async def test_other_users_and_missing_handles_look_the_same(self):
        record = await self.registry.mint(scalar_record("user-a"))
        await self.assert_reason("not_found", self.registry.get("user-b", record.id))
        await self.assert_reason("not_found", self.registry.get("user-a", new_handle_id("scalar")))
        await self.assert_reason("not_found", self.registry.mark_evicted("user-b", record.id, LATER))
        self.assertEqual(await self.registry.get("user-a", record.id), record)

    async def test_run_id_round_trips(self):
        run_id = "run_" + "0123456789abcdef" * 2
        inside = await self.registry.mint(scalar_record("user-a", run_id=run_id))
        outside = await self.registry.mint(scalar_record("user-a"))
        obs = await self.registry.record_retrieval(retrieval_record("user-a"))
        self.assertEqual((await self.registry.get("user-a", inside.id)).run_id, run_id)
        self.assertIsNone((await self.registry.get("user-a", outside.id)).run_id)
        self.assertIsNone((await self.registry.get("user-a", obs.id)).run_id)

    async def test_nul_is_refused_in_every_string_and_other_characters_round_trip(self):
        base = scalar_record("user-a").model_dump()
        descriptor = base["descriptor"]
        places = {
            "summary": lambda s: {"descriptor": {**descriptor, "summary": f"Mean{s}"}},
            "quantity": lambda s: {"descriptor": {**descriptor, "quantity": f"no2{s}"}},
            "unit": lambda s: {"descriptor": {**descriptor, "unit": f"ppb{s}"}},
            "product": lambda s: {"descriptor": {**descriptor, "product": f"TEMPO{s}"}},
            "region": lambda s: {"descriptor": {**descriptor, "region": {"name": f"LA{s}"}}},
            "masking key": lambda s: {"descriptor": {**descriptor, "masking": {f"qa{s}": "good"}}},
            "parameter key": lambda s: {"parameters": {f"dim{s}": "time"}},
            "parameter value": lambda s: {"parameters": {"dims": ["lat", f"lon{s}"]}},
            "operation": lambda s: {"operation": f"mean{s}"},
            "conversation id": lambda s: {"conversation_id": f"conv{s}"},
            "user id": lambda s: {"user_id": f"user-a{s}"},
            "uncertainty": lambda s: {"payload": {**base["payload"], "uncertainty": {"note": f"sd{s}"}}},
        }
        for place, change in places.items():
            with self.subTest(place=place):
                with self.assertRaises(ValidationError):
                    HandleRecord(**{**base, "id": new_handle_id("scalar"), **change("\x00")})
                record = HandleRecord(**{**base, "id": new_handle_id("scalar"), **change("\x01é")})
                await self.registry.mint(record)
                self.assertEqual(await self.registry.get(record.user_id, record.id), record)

    async def test_numbers_keep_their_value_not_their_type(self):
        # JSONB hands 1e16 back as an int and -0.0 as 0.0; equality is what holds.
        parameters = {"big": 1e16, "negative_zero": -0.0, "small": 1e-300, "int": 7, "nested": [2.5, {"x": -1e16}]}
        for value in (1e16, -0.0, 0.1, 10**15):
            with self.subTest(value=value):
                record = await self.registry.mint(scalar_record("user-a", value=value, parameters=parameters))
                fetched = await self.registry.get("user-a", record.id)
                self.assertEqual(fetched.parameters, parameters)
                self.assertEqual(fetched.payload.value, value)

    async def test_a_stored_record_loads_after_a_check_is_tightened(self):
        derived = await self.registry.mint(table_record("user-a"))
        obs = await self.registry.record_retrieval(retrieval_record("user-a"))
        again = retrieval_record("user-a", summary="again")
        # Every record now fails validation, as it would under a stricter check.
        with mock.patch("tta_backend.harness.handles.model._has_nul", return_value=True):
            with self.assertRaises(ValidationError):
                HandleRecord.model_validate(derived.model_dump())
            self.assertEqual(await self.registry.get("user-a", derived.id), derived)
            self.assertEqual(await self.registry.record_retrieval(again), obs)
            evicted = await self.registry.mark_evicted("user-a", derived.id, LATER)
            self.assertEqual(evicted.payload.state, "evicted")
            self.assertEqual(await self.registry.get("user-a", derived.id), evicted)

    async def test_traversal_shaped_ids_are_not_found(self):
        for handle_id in TRAVERSAL_IDS:
            with self.subTest(handle_id=handle_id):
                await self.assert_reason("not_found", self.registry.get("user-a", handle_id))

    async def test_parents_must_exist_for_the_same_user(self):
        theirs = await self.registry.mint(scalar_record("user-a"))
        await self.registry.record_retrieval(retrieval_record("user-a"))
        for parent in (theirs.id, "obs_tempo_no2_1", new_handle_id("grid")):
            with self.subTest(parent=parent):
                child = scalar_record("user-b", parents=(parent,))
                await self.assert_reason("parent_not_found", self.registry.mint(child))
                await self.assert_reason("not_found", self.registry.get("user-b", child.id))

    async def test_minting_the_same_id_twice_is_refused(self):
        record = await self.registry.mint(scalar_record("user-a"))
        await self.assert_reason("already_exists", self.registry.mint(record))

    async def test_origin_is_checked_on_the_way_in(self):
        with self.assertRaises(ValueError):
            await self.registry.mint(retrieval_record("user-a"))
        with self.assertRaises(ValueError):
            await self.registry.record_retrieval(scalar_record("user-a"))

    async def test_recording_a_retrieval_twice_is_idempotent(self):
        first = await self.registry.record_retrieval(retrieval_record("user-a", summary="first"))
        again = await self.registry.record_retrieval(retrieval_record("user-a", summary="second"))
        self.assertEqual(again, first)
        self.assertEqual((await self.registry.get("user-a", first.id)).descriptor.summary, "first")

    async def test_same_retrieval_id_under_two_users(self):
        await self.registry.record_retrieval(retrieval_record("user-a", summary="A"))
        await self.registry.record_retrieval(retrieval_record("user-b", summary="B"))
        self.assertEqual((await self.registry.get("user-a", "obs_tempo_no2_1")).descriptor.summary, "A")
        self.assertEqual((await self.registry.get("user-b", "obs_tempo_no2_1")).descriptor.summary, "B")

    async def test_eviction_keeps_the_record(self):
        record = table_record("user-a")
        await self.registry.mint(record)
        evicted = await self.registry.mark_evicted("user-a", record.id, LATER)
        self.assertEqual(evicted.payload.state, "evicted")
        self.assertEqual(evicted.payload.format, "parquet")
        fetched = await self.registry.get("user-a", record.id)
        self.assertEqual(fetched, evicted)
        self.assertEqual(fetched.descriptor, record.descriptor)
        self.assertEqual(await self.registry.mark_evicted("user-a", record.id, NOW), evicted)

    async def test_inline_and_service_payloads_are_not_evictable(self):
        scalar = await self.registry.mint(scalar_record("user-a"))
        obs = await self.registry.record_retrieval(retrieval_record("user-a"))
        for handle_id in (scalar.id, obs.id):
            with self.subTest(handle_id=handle_id):
                await self.assert_reason("not_evictable", self.registry.mark_evicted("user-a", handle_id, LATER))

    async def test_an_evicted_payload_cannot_be_minted(self):
        record = table_record("user-a")
        evicted = HandleRecord(**{**record.model_dump(), "payload": {"state": "evicted", "format": "parquet", "evicted_at": LATER}})
        with self.assertRaises(ValueError):
            await self.registry.mint(evicted)


class InMemoryRegistryContractTests(RegistryContract, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = InMemoryHandleRegistry()


class PostgresRegistryContractTests(RegistryContract, PostgresTestCase):
    async def asyncSetUp(self):
        self._stack = AsyncExitStack()
        self.connect = await self._stack.enter_async_context(isolated_schema())
        await ensure_tables(self.connect)
        await ensure_tables(self.connect)
        self.registry = PostgresHandleRegistry(self.connect)

    async def asyncTearDown(self):
        await self._stack.aclose()

    async def test_tables_are_the_harness_own(self):
        async with self.connect() as conn:
            cur = await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")
            names = [row[0] for row in await cur.fetchall()]
        self.assertEqual(names, ["harness_handles"])


class PostgresEndToEndTests(PostgresTestCase):
    async def asyncSetUp(self):
        self._stack = AsyncExitStack()
        connect = await self._stack.enter_async_context(isolated_schema())
        await ensure_tables(connect)
        tmp = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(tmp)
        self.registry = PostgresHandleRegistry(connect)
        self.opened = []

        async def open_retrieval(user_id, record):
            self.opened.append((user_id, record.id))
            return grid_dataset()

        self.resolver = HandleResolver(self.registry, PayloadStore(self.root), open_retrieval)
        self.provenance = FakeProvenance()

    async def asyncTearDown(self):
        await self._stack.aclose()

    async def test_retrieve_derive_open_trace_evict_and_isolate(self):
        obs = await self.registry.record_retrieval(retrieval_record("user-a"))
        retrieved = await self.resolver.open("user-a", obs.id, "grid")

        common = {"user_id": "user-a", "conversation_id": "conv-1", "created_at": NOW}
        grid = await self.resolver.mint(
            kind="grid", data=retrieved.isel(time=[0]), descriptor=describe("First day of NO2", unit="molec/cm2"),
            operation="select_time", parents=(obs.id,), parameters={"time": "2026-01-01"}, **common,
        )
        series = await self.resolver.mint(
            kind="series", data=monitor_readings(), descriptor=describe("Regional mean NO2"),
            operation="area_mean", parents=(grid.id,), parameters={"dims": ["lat", "lon"]}, **common,
        )
        scalar = await self.resolver.mint(
            kind="scalar", data=ScalarInput(value=33.2, uncertainty={"ci95": [30.1, 36.3]}), descriptor=describe("Mean", unit="ppb"),
            operation="mean", parents=(series.id,), **common,
        )

        self.assertEqual(int((await self.resolver.open("user-a", grid.id, "grid"))["no2"].sizes["time"]), 1)
        pd.testing.assert_frame_equal(await self.resolver.open("user-a", series.id, "series"), monitor_readings())
        self.assertEqual((await self.resolver.open("user-a", scalar.id, "scalar")).value, 33.2)

        trace = await lineage(self.registry, "user-a", [scalar.id], self.provenance)
        self.assertEqual([n["handle"] for n in trace["nodes"]], ["dataset_tempo_no2", f"aoi_{obs.id}", obs.id, grid.id, series.id, scalar.id])
        self.assertEqual(dataset_handles(trace), ["dataset_tempo_no2"])
        self.assertEqual(self.provenance.calls, [("user-a", [obs.id])])

        await self.resolver.evict("user-a", grid.id, LATER)
        with self.assertRaises(HandleError) as caught:
            await self.resolver.open("user-a", grid.id)
        self.assertEqual(caught.exception.reason, "evicted")
        self.assertEqual(await lineage(self.registry, "user-a", [scalar.id], self.provenance), trace)
        self.assertEqual(len([p for p in self.root.rglob("*.parquet")]), 1)
        self.assertEqual(list(self.root.rglob("*.zarr")), [])

        calls_before = len(self.provenance.calls)
        for handle_id in (obs.id, grid.id, series.id, scalar.id):
            with self.subTest(handle_id=handle_id):
                for attempt in (
                    self.resolver.get("user-b", handle_id),
                    self.resolver.open("user-b", handle_id),
                    self.resolver.evict("user-b", handle_id, LATER),
                    lineage(self.registry, "user-b", [handle_id], self.provenance),
                ):
                    with self.assertRaises(HandleError) as caught:
                        await attempt
                    self.assertEqual(caught.exception.reason, "not_found")
                    self.assertEqual(str(caught.exception), "No such handle.")
        self.assertEqual(self.opened, [("user-a", obs.id)])
        self.assertEqual(len(self.provenance.calls), calls_before)
        with self.assertRaises(HandleError) as caught:
            await self.resolver.mint(kind="scalar", data=ScalarInput(value=1), descriptor=describe("x"), operation="mean", parents=(series.id,), user_id="user-b", conversation_id="c", created_at=NOW)
        self.assertEqual(caught.exception.reason, "parent_not_found")

        # User B's own copy of the same retrieval id is theirs alone.
        await self.registry.record_retrieval(retrieval_record("user-b", summary="B's"))
        b_trace = await lineage(self.registry, "user-b", [obs.id], self.provenance)
        self.assertNotIn(grid.id, [n["handle"] for n in b_trace["nodes"]])
        self.assertEqual(self.provenance.calls[-1], ("user-b", [obs.id]))
        self.assertEqual((await self.resolver.get("user-a", obs.id)).descriptor.summary, "TEMPO NO2 over LA")


if __name__ == "__main__":
    unittest.main()
