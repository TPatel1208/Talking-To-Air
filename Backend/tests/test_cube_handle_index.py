"""T54 — the handle→cube index: stop gating cache lookups behind an MCP round-trip.

T52 shipped a cube cache that lives *behind* the expensive upstream operation it
exists to shield you from: ``open_handle`` calls ``export_result``
unconditionally first, and the cache key is derived from the exported file. So
in the two situations a cache should be strongest — the source was evicted
upstream, or the MCP is mid-crash-restart — a perfectly good cube sits on local
disk unreachable, and the evicted case spends *minutes* rematerializing the
input to an answer it already holds.

The behaviours pinned here are the two halves of making the reorder safe. The
re-key (``source_identity`` derived from what the MCP says it *delivered*) is
what makes a drifted rematerialization produce a different key rather than a
silent false hit; the index is what lets the lookup happen before the round-trip
at all. Skipping upstream verification is only defensible with the first in
place, so the tests that matter most here are the ones where the cube must
*refuse* to serve: a drifted digest, a partial digest, a legacy handle, a cube
with a chunk file missing underneath it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
import unittest.mock


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["xarray", "zarr", "numpy", "dask"]

requires_stack = unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "cube cache test dependencies (xarray/zarr/numpy/dask) are not installed",
)


@requires_stack
class HandleIndexTests(unittest.TestCase):
    """The index in isolation: what it maps, and when it refuses to."""

    def setUp(self):
        from test_cube_cache import StoreTestCase

        StoreTestCase.setUp(self)
        self.make_dataset = StoreTestCase.make_dataset.__get__(self)

    def _write(self, handle: str, key: str, *, digest: str = "d1", partial: bool = False, ds=None):
        # A real file on disk, because a partial digest falls back to *filesystem*
        # identity and so genuinely stats the export.
        export = os.path.join(self._store.name, f"{handle}.zip")
        with open(export, "wb") as f:
            f.write(handle.encode())
        source = self.cube_cache.source_identity(
            export, handle=handle, content_digest=digest, partial=partial
        )
        self.cube_cache.write_cube(ds if ds is not None else self.make_dataset(), key, source=source, handle=handle)
        return source

    def test_a_cube_built_from_a_delivered_digest_is_servable_by_its_handle(self):
        """The reorder's read path, with no MCP in sight: the cube is reachable
        from the handle alone, which is the whole point — the handle is what the
        caller has *before* the round-trip."""
        import xarray as xr

        ds = self.make_dataset()
        self._write("obs_1", "key1", ds=ds)

        cached = self.cube_cache.lookup_by_handle("obs_1")

        self.assertIsNotNone(cached)
        xr.testing.assert_identical(ds, cached.load())

    def test_an_unknown_handle_misses(self):
        self.assertIsNone(self.cube_cache.lookup_by_handle("obs_never_seen"))

    def test_a_lost_index_costs_speed_and_nothing_else(self):
        """The index is derived state, never the system of record. Deleting it
        must degrade to T52's ordering — lookup after ``export_result``, correct
        and merely slower — and never to an error."""
        self._write("obs_1", "key1")
        os.remove(os.path.join(self.store_root, "handle_index.json"))

        self.assertIsNone(self.cube_cache.lookup_by_handle("obs_1"))
        self.assertIsNotNone(self.cube_cache.lookup("key1"))  # the cube is fine

    def test_a_startup_sweep_reconstructs_a_lost_index_from_the_manifests(self):
        """Every mapping in the index is also in the cube's own manifest, which
        was the record all along — so a boot puts it back with no migration and
        nothing to remember to run."""
        self._write("obs_1", "key1")
        self._write("obs_2", "key2")
        os.remove(os.path.join(self.store_root, "handle_index.json"))

        self.cube_cache.sweep_store()

        self.assertEqual(self.cube_cache.key_for_handle("obs_1"), "key1")
        self.assertEqual(self.cube_cache.key_for_handle("obs_2"), "key2")

    def test_a_legacy_handle_is_never_served_without_a_round_trip(self):
        """A handle materialized before upstream PRD 023 carries no digest, so
        its cube is keyed on filesystem metadata — which records only that a
        file with these bytes sat at this path, knowledge obtainable only by
        asking upstream. Serving it unverified would assume what the key cannot
        state. The cube stays perfectly usable, the T52 way."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bundle.zip")
            with open(path, "wb") as f:
                f.write(b"legacy")
            source = self.cube_cache.source_identity(path)  # no digest at all
            self.cube_cache.write_cube(self.make_dataset(), "legacy", source=source, handle="obs_old")

        self.assertEqual(self.cube_cache.key_for_handle("obs_old"), "legacy")
        self.assertIsNone(self.cube_cache.lookup_by_handle("obs_old"))
        self.assertIsNotNone(self.cube_cache.lookup("legacy"))

    def test_a_partial_digest_is_never_served_without_a_round_trip(self):
        """Upstream marks a digest partial when the provider named no granules
        (an AppEEARS point sample). A digest that cannot see granules cannot
        detect granule drift, so it is not trusted for the skip — and, for the
        same reason, not used for the key either."""
        self._write("obs_partial", "pkey", digest="d-partial", partial=True)

        self.assertIsNone(self.cube_cache.lookup_by_handle("obs_partial"))
        self.assertIsNotNone(self.cube_cache.lookup("pkey"))


@requires_stack
@unittest.skipIf(
    any(
        importlib.util.find_spec(name) is None
        for name in ("langchain_mcp_adapters", "fastmcp", "uvicorn")
    ),
    "open_handle test dependencies are not installed",
)
class HandleIndexReorderTests(unittest.IsolatedAsyncioTestCase):
    """End to end through ``open_handle``, against a real (fake) MCP server.

    Every assertion here is about the *round-trip*, not the data: whether
    ``export_result`` was called at all. That is the one thing T52 could not
    influence and the only thing T54 changes."""

    async def asyncSetUp(self):
        from tta_backend.config.settings import Settings
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from fake_earthdata_mcp import FakeEarthdataMCPServer, HandleVolume, build_fake_mcp
        from test_cube_cache import StoreTestCase

        StoreTestCase.setUp(self)
        self.set_limits = StoreTestCase.set_limits.__get__(self)
        # Every test here ends on an `open_handle`, and an `open_handle` that
        # misses schedules a fire-and-forget cube write. Nothing awaits it: the
        # task writes on a worker thread and `reset_for_test` only clears the
        # in-flight *set*. So without this the writer is still creating chunk
        # files while StoreTestCase's tempdir is being removed. Registered here,
        # after StoreTestCase.setUp, so the LIFO cleanup stack drains before that
        # tempdir goes away rather than after.
        self.addAsyncCleanup(self._drain_writes)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        self.tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        )
        self.volume.add_netcdf_bundle("obs_cubed", {
            "granule_20260709.nc4": {None: self._granule(9)},
            "granule_20260710.nc4": {None: self._granule(10)},
        })
        self.volume.set_delivered_content("obs_cubed", "digest-v1")

    @staticmethod
    def _granule(day: int):
        import numpy as np
        import xarray as xr

        def factory():
            return xr.Dataset(
                {"no2": (("time", "latitude", "longitude"), [[[1.0 * day, 2.0], [3.0, 4.0]]])},
                coords={
                    "time": [np.datetime64(f"2026-07-{day:02d}T12:00:00")],
                    "latitude": [40.0, 41.0],
                    "longitude": [-75.0, -74.0],
                },
            )

        return factory

    async def _drain_writes(self):
        import asyncio

        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _cube_it(self):
        """Two opens (cubing is earned on the second) plus the background write."""
        from tta_backend.services.open_handle import open_handle

        first = await open_handle("obs_cubed", self.tools)
        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()
        return first

    def _export_calls(self) -> int:
        return self.volume.export_calls.get("obs_cubed", 0)

    async def test_an_indexed_cube_answers_with_no_export_round_trip_at_all(self):
        """User story #2, and the mechanism behind #1: once the cube exists, the
        MCP is not on the path to the answer."""
        import numpy as np

        from tta_backend.services.open_handle import open_handle

        first = await self._cube_it()
        before = self._export_calls()

        served = await open_handle("obs_cubed", self.tools)

        self.assertEqual(self._export_calls(), before)
        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)

    async def test_an_export_delivering_different_content_drops_the_stale_cube(self):
        """User story #4, and the reason the re-key had to come first.

        A late-arriving NRT granule, an unpinned Harmony submit that picked a
        different service chain, an OPeNDAP fallback on replay: the handle and
        its request spec are identical, the bytes are not. Because the key is
        derived from delivered content rather than from a file's mtime, a
        different key here means *the data actually changed* — so the honest
        move is to drop the cube, not to keep serving it.

        Asserted under verify-first ordering, because that is the only ordering
        in which drift is *reachable*: see
        :func:`services.cube_cache.reconcile_handle` — a handle's delivered
        content changes only when a rematerialization completes, and this
        backend only ever rematerializes on the verify path."""
        from tta_backend.services.open_handle import open_handle

        await self._cube_it()
        stale_key = self.cube_cache.key_for_handle("obs_cubed")
        self.assertIsNotNone(stale_key)

        self.volume.set_delivered_content("obs_cubed", "digest-v2")
        self.set_limits(CUBE_SKIP_EXPORT_VERIFY=0)
        await open_handle("obs_cubed", self.tools)

        self.assertNotIn(stale_key, self._store_entries())
        self.assertIsNone(self.cube_cache.key_for_handle("obs_cubed"))

    def _store_entries(self):
        return sorted(os.listdir(self.store_root)) if os.path.isdir(self.store_root) else []

    async def test_an_evicted_source_answers_from_the_cube_without_rematerializing(self):
        """User story #1 — the headline, and the largest single latency win in
        the cache design.

        T52's worst case: the MCP has evicted the materialized file, so
        ``export_result`` answers non-ready, ``open_handle`` enters
        ``rematerialize`` -> ``await_retrieval`` — *minutes* of waiting — and
        only afterwards discovers a perfectly good cube was on local disk the
        entire time. The input to a cached output gets rebuilt to produce an
        output already held. Here it becomes a local Zarr read.

        This also pins the invariant that licenses the skip: an index hit
        performs no rematerialize, so it cannot be serving across a drift."""
        import numpy as np

        from tta_backend.services.open_handle import open_handle

        first = await self._cube_it()
        self.volume.evict("obs_cubed")

        served = await open_handle("obs_cubed", self.tools)

        self.assertEqual(self.volume.rematerialize_calls.get("obs_cubed", 0), 0)
        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)

    async def test_a_cubed_question_answers_while_the_mcp_is_unreachable(self):
        """User story #2. The ``earthdata-mcp`` container has been observed
        crash-restarting silently (RestartCount 11, exit 0); during a restart
        window a cube sitting on local disk was unreachable purely because the
        lookup was gated behind a call to the thing that was down."""
        import numpy as np

        from tta_backend.services.open_handle import open_handle

        first = await self._cube_it()

        class _Restarting:
            async def ainvoke(self, *_args, **_kwargs):
                raise ConnectionError("earthdata-mcp is restarting")

        served = await open_handle(
            "obs_cubed", {**self.tools, "export_result": _Restarting()}
        )

        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)

    async def test_the_override_flag_restores_unconditional_verify_first_ordering(self):
        """The switch that lets an operator disable the skip in production
        without a redeploy, if it is ever implicated in a bad answer. The cube
        is still served — it is the *ordering* that reverts to T52's."""
        import numpy as np

        first = await self._cube_it()
        self.set_limits(CUBE_SKIP_EXPORT_VERIFY=0)
        before = self._export_calls()

        from tta_backend.services.open_handle import open_handle

        served = await open_handle("obs_cubed", self.tools)

        self.assertEqual(self._export_calls(), before + 1)
        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)

    async def test_the_reorders_own_contribution_is_measurable_on_its_own(self):
        """User story #7. An index hit is *also* a cube hit — the open pipeline
        was skipped either way, and letting ``cache_hits_total`` quietly stop
        counting those would make T52's hit rate collapse the day T54 shipped.
        What the index counter adds is the second, separate win: the round-trip
        was skipped too."""
        from prometheus_client import REGISTRY

        from tta_backend.services.open_handle import open_handle

        def value(name, labels=None):
            return REGISTRY.get_sample_value(name, labels) or 0.0

        await self._cube_it()
        index_hits = value("cube_index_hits_total")
        cube_hits = value("cache_hits_total", {"cache_level": "zarr"})

        await open_handle("obs_cubed", self.tools)

        self.assertEqual(value("cube_index_hits_total"), index_hits + 1)
        self.assertEqual(value("cache_hits_total", {"cache_level": "zarr"}), cube_hits + 1)

    async def test_an_invalidation_is_counted_apart_from_an_ordinary_miss(self):
        """Dropping a cube because upstream delivered different content is a
        different event from never having had one, and only the first says the
        science moved."""
        from prometheus_client import REGISTRY

        from tta_backend.services.open_handle import open_handle

        await self._cube_it()
        before = REGISTRY.get_sample_value("cube_index_invalidations_total") or 0.0

        self.volume.set_delivered_content("obs_cubed", "digest-v2")
        self.set_limits(CUBE_SKIP_EXPORT_VERIFY=0)
        await open_handle("obs_cubed", self.tools)

        self.assertEqual(
            REGISTRY.get_sample_value("cube_index_invalidations_total"), before + 1
        )

    async def test_a_cube_with_a_missing_chunk_is_not_served_on_an_index_hit(self):
        """Skipping *upstream* verification is not licence to skip *local*
        verification. The same per-hit integrity sweep T52 runs — manifest
        chunk-count and total bytes against an ``os.scandir`` walk — still
        decides, and it catches exactly what a metadata-only ``open_zarr``
        cannot: a chunk file deleted underneath the store."""
        import numpy as np

        from tta_backend.services.open_handle import open_handle

        first = await self._cube_it()
        key = self.cube_cache.key_for_handle("obs_cubed")
        cube_dir = os.path.join(self.store_root, key, "cube.zarr")
        chunks = [
            os.path.join(dirpath, name)
            for dirpath, _dirs, files in os.walk(cube_dir)
            for name in files
            if not name.startswith(".") and "zarr.json" not in name
        ]
        os.remove(chunks[0])

        served = await open_handle("obs_cubed", self.tools)

        # Answered anyway, from the pipeline — a cache that can refuse is
        # strictly better than one that can be subtly wrong.
        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)
        self.assertGreater(self._export_calls(), 0)
