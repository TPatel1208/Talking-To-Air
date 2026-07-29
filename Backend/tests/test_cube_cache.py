"""T52 — the L4 Zarr cube cache: a versioned, self-invalidating cache of the
open pipeline's *output*.

The behaviours pinned here are the ones whose failure mode is a silently wrong
answer rather than a crash: a cube served after the pipeline that produced it
was fixed, a round-trip that drops ``.encoding`` (which wipes a whole packed-CF
variable through AggregationService's valid_range scaling), and slash-qualified
collided-leaf names that Zarr cannot store flat.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
import unittest.mock

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["xarray", "zarr", "numpy", "dask"]

requires_stack = unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "cube cache test dependencies (xarray/zarr/numpy/dask) are not installed",
)


class StoreTestCase(unittest.TestCase):
    """Points the cube store at a throwaway directory and resets the module's
    process-lifetime state (open counts, in-flight writes) between tests."""

    def setUp(self):
        import services.cube_cache as cube_cache
        from config.settings import get_settings

        self.cube_cache = cube_cache
        self._store = tempfile.TemporaryDirectory()
        self.addCleanup(self._store.cleanup)
        self.store_root = os.path.join(self._store.name, "cube_store")

        patcher = unittest.mock.patch.dict(
            os.environ,
            {"CUBE_STORE_DIR": self.store_root},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        cube_cache.reset_for_test()
        self.addCleanup(cube_cache.reset_for_test)

    def set_limits(self, **env):
        from config.settings import get_settings

        patcher = unittest.mock.patch.dict(os.environ, {k: str(v) for k, v in env.items()})
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()

    def make_dataset(self):
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {"no2": (("time", "lat", "lon"), np.arange(8, dtype="float32").reshape(2, 2, 2))},
            coords={
                "time": np.array(["2026-07-01", "2026-07-02"], dtype="datetime64[ns]"),
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            attrs={"title": "fixture"},
        )


@requires_stack
class CacheKeyTests(unittest.TestCase):
    """The key is the whole invalidation story: it must move when the bytes
    move, when the pipeline's *interpretation* of those bytes changes, and
    when the environment's reader changes."""

    def _identity(self, contents: bytes) -> str:
        from services.cube_cache import source_identity

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bundle.zip")
            with open(path, "wb") as f:
                f.write(contents)
            return source_identity(path)

    def test_cache_key_changes_when_the_pipeline_version_changes(self):
        from services.cube_cache import cache_key

        source = "some-source-identity"
        before = cache_key(source, "1", "h5netcdf")
        after = cache_key(source, "2", "h5netcdf")

        self.assertNotEqual(before, after)

    def test_cache_key_changes_when_the_netcdf_engine_changes(self):
        from services.cube_cache import cache_key

        source = "some-source-identity"
        self.assertNotEqual(
            cache_key(source, "1", "h5netcdf"),
            cache_key(source, "1", "netcdf4"),
        )

    def test_cache_key_is_stable_for_the_same_inputs(self):
        from services.cube_cache import cache_key

        self.assertEqual(cache_key("s", "1", "h5netcdf"), cache_key("s", "1", "h5netcdf"))

    def test_source_identity_changes_when_the_file_contents_change(self):
        self.assertNotEqual(self._identity(b"a" * 16), self._identity(b"b" * 32))


@requires_stack
class RoundTripTests(StoreTestCase):
    """``read_cube(write_cube(D)) == D`` for everything downstream reads.

    The cube is a faithful *mirror*, not a canonical form: downstream code is
    not modified, so anything the round-trip normalizes away is a behaviour
    change on a path none of this repo's correctness fixes were written
    against."""

    def test_a_written_cube_reads_back_with_identical_values_dims_and_coords(self):
        import xarray as xr

        ds = self.make_dataset()
        self.cube_cache.write_cube(ds, "key1")

        cached = self.cube_cache.lookup("key1")

        self.assertIsNotNone(cached)
        xr.testing.assert_identical(ds, cached.load())

    def test_lookup_misses_when_nothing_was_written(self):
        self.assertIsNone(self.cube_cache.lookup("never-written"))

    def test_a_packed_cf_variable_still_masks_correctly_after_a_cube_round_trip(self):
        """Landmine 1, asserted *through the cube*.

        AggregationService reads scale_factor/add_offset from ``.encoding`` and
        uses encoding *presence* as the signal that xarray decoded the data, in
        order to scale a packed CF ``valid_range`` into physical units. Stripping
        ``.encoding`` before ``to_zarr`` — the standard fix for chunk/compressor
        conflicts — leaves a cube of physical floats carrying a still-packed
        ``valid_range: [0, 30000]``; masking a ~6e17 field against ``<= 30000``
        wipes the entire variable while reporting success. This is exactly the
        scenario ``test_aggregation_service`` pins, resurrected on a path that
        test does not cover."""
        import numpy as np
        import xarray as xr

        from preprocessing.aggregation_service import AggregationService

        # scale_factor 2e13, valid_range packed [0, 30000] -> physical [0, 6e17].
        ds = xr.Dataset(
            {"scaled_no2": (
                ("lat", "lon"),
                np.array([[2.0e13, 4.0e13], [6.0e13, 1.0e18]]),
                {"valid_range": [0, 30000]},
            )},
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
        )
        ds["scaled_no2"].encoding["scale_factor"] = 2.0e13
        ds["scaled_no2"].encoding["add_offset"] = 0.0

        self.cube_cache.write_cube(ds, "packed")
        cached = self.cube_cache.lookup("packed")

        self.assertEqual(cached["scaled_no2"].encoding.get("scale_factor"), 2.0e13)
        values = AggregationService().aggregate(cached, variable="scaled_no2").ds["scaled_no2"].values
        # The three in-range physical values survive; only the genuinely
        # out-of-range cell drops. A dropped encoding would NaN all four.
        self.assertEqual(values[0, 0], 2.0e13)
        self.assertEqual(values[1, 0], 6.0e13)
        self.assertTrue(np.isnan(values[1, 1]))

    def test_the_cube_stores_decoded_physical_values_not_repacked_ones(self):
        """The encoding is restored as *metadata*, never re-applied: a cube
        written from a decoded Dataset holds the physical floats. Persisting
        scale_factor into the write path instead would make ``to_zarr`` re-pack
        values that were already decoded, and every subsequent read would
        decode them a second time."""
        import numpy as np
        import xarray as xr

        ds = xr.Dataset({"v": ("x", np.array([2.0e13, 4.0e13]))}, coords={"x": [0, 1]})
        ds["v"].encoding["scale_factor"] = 2.0e13

        self.cube_cache.write_cube(ds, "physical")
        cached = self.cube_cache.lookup("physical")

        np.testing.assert_array_equal(cached["v"].values, [2.0e13, 4.0e13])

    def test_slash_qualified_collided_leaf_names_survive_the_round_trip(self):
        """Landmine 2. Collided leaves are renamed to "group/leaf" (T25's
        refuse-with-candidates protection depends on those qualified names
        downstream), and ``/`` is Zarr's group separator — so they cannot
        round-trip as flat variables. Escaping on write plus the manifest's
        name map is what keeps them intact; relying on Zarr silently turns one
        variable into a nested group."""
        import numpy as np
        import xarray as xr

        ds = xr.Dataset(
            {
                "product/foo": ("x", np.array([1.0, 2.0])),
                "support_data/foo": ("x", np.array([3.0, 4.0])),
            },
            coords={"x": [0, 1]},
        )

        self.cube_cache.write_cube(ds, "collided")
        cached = self.cube_cache.lookup("collided")

        self.assertIn("product/foo", cached.data_vars)
        self.assertIn("support_data/foo", cached.data_vars)
        np.testing.assert_array_equal(cached["product/foo"].values, [1.0, 2.0])
        np.testing.assert_array_equal(cached["support_data/foo"].values, [3.0, 4.0])

    def test_an_ambiguous_bare_name_refuses_identically_cubed_and_uncubed(self):
        """T25's doctrine must not change across the cube: a bare name that
        two groups both claim still refuses with candidates rather than
        silently picking one."""
        import numpy as np
        import xarray as xr

        from preprocessing.aggregation_service import AggregationService

        ds = xr.Dataset(
            {
                "product/foo": ("x", np.array([1.0, 2.0])),
                "support_data/foo": ("x", np.array([3.0, 4.0])),
            },
            coords={"x": [0, 1]},
        )
        self.cube_cache.write_cube(ds, "collided2")
        cached = self.cube_cache.lookup("collided2")

        service = AggregationService()
        with self.assertRaises(Exception) as uncubed:
            service.to_dataarray(ds, "foo")
        with self.assertRaises(Exception) as cubed:
            service.to_dataarray(cached, "foo")

        self.assertEqual(type(uncubed.exception), type(cubed.exception))
        self.assertEqual(str(uncubed.exception), str(cubed.exception))


@requires_stack
class IntegrityTests(StoreTestCase):
    """A cube that is corrupt, half-written or unreadable must be invisible:
    the answer still arrives, just uncached. This is what guarantees the cache
    can never cost a turn."""

    def _entry_dirs(self):
        if not os.path.isdir(self.store_root):
            return []
        return sorted(os.listdir(self.store_root))

    def test_a_write_interrupted_before_the_manifest_leaves_no_visible_cube(self):
        """The manifest is written last and is the completion marker, so a
        crash mid-write can only ever leave a staging directory — never a
        half-populated entry that lookup would serve as complete."""
        ds = self.make_dataset()

        with unittest.mock.patch("services.cube_cache.json.dump", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.cube_cache.write_cube(ds, "interrupted")

        self.assertIsNone(self.cube_cache.lookup("interrupted"))
        self.assertNotIn("interrupted", self._entry_dirs())

    def test_a_second_writer_adopts_the_winners_cube_instead_of_clobbering_it(self):
        ds = self.make_dataset()
        self.cube_cache.write_cube(ds, "raced")

        self.assertTrue(self.cube_cache.write_cube(ds, "raced"))
        self.assertIsNotNone(self.cube_cache.lookup("raced"))

    def test_a_missing_chunk_file_fails_per_hit_validation_and_the_cube_is_dropped(self):
        """A metadata-only ``open_zarr`` succeeds against a store whose chunks
        have been deleted underneath it and only fails much later, inside a
        downstream compute. The per-hit scandir sweep — file count and total
        bytes against the manifest, no data read — is what turns that into a
        clean miss."""
        self.cube_cache.write_cube(self.make_dataset(), "corrupt")
        chunk = self._first_chunk_file(self.cube_cache._cube_path("corrupt"))
        os.remove(chunk)

        self.assertIsNone(self.cube_cache.lookup("corrupt"))
        self.assertNotIn("corrupt", self._entry_dirs())

    def test_a_truncated_chunk_file_fails_per_hit_validation(self):
        self.cube_cache.write_cube(self.make_dataset(), "truncated")
        chunk = self._first_chunk_file(self.cube_cache._cube_path("truncated"))
        with open(chunk, "wb") as f:
            f.write(b"")

        self.assertIsNone(self.cube_cache.lookup("truncated"))

    def test_an_unreadable_store_deletes_the_cube_logs_and_misses(self):
        """Tier 3: any failure on the cube path deletes the entry, logs a
        named event, and falls through to the lazy path rather than raising."""
        import xarray as xr

        self.cube_cache.write_cube(self.make_dataset(), "unreadable")

        with unittest.mock.patch.object(xr, "open_zarr", side_effect=RuntimeError("bad store")):
            with self.assertLogs("services.cube_cache", level="WARNING") as logs:
                self.assertIsNone(self.cube_cache.lookup("unreadable"))

        self.assertTrue(any("cube_read_failed" in line for line in logs.output))
        self.assertNotIn("unreadable", self._entry_dirs())

    def test_startup_sweep_removes_orphaned_staging_dirs_and_manifestless_entries(self):
        os.makedirs(os.path.join(self.store_root, "staging-abc-123"), exist_ok=True)
        os.makedirs(os.path.join(self.store_root, "deadbeef"), exist_ok=True)
        self.cube_cache.write_cube(self.make_dataset(), "good")

        self.cube_cache.sweep_store()

        self.assertEqual(self._entry_dirs(), ["good"])

    def test_a_missing_chunk_reads_as_fill_values_which_is_why_the_sweep_must_precede_the_serve(self):
        """Measured, and worse than assumed: Zarr does **not** raise on a
        missing chunk — it substitutes the array's fill value and reports
        success. So a cube with a deleted chunk is a silent-wrong-numbers
        failure, not a crash, and nothing downstream would ever notice.

        That is what makes the per-hit ``os.scandir`` sweep load-bearing rather
        than a nicety: it is the *only* thing standing between a partially-
        deleted store and a scientific answer computed from fill values. This
        test pins both halves — the silent fill, and the sweep that must
        therefore run before every serve."""
        import numpy as np

        self.cube_cache.write_cube(self.make_dataset(), "goes_bad")
        served = self.cube_cache.lookup("goes_bad")
        os.remove(self._first_chunk_file(os.path.join(self.cube_cache._cube_path("goes_bad"), "no2")))

        # No exception: Zarr fills the hole and the values are simply wrong.
        loaded = served["no2"].load().values
        self.assertFalse(np.array_equal(loaded, self.make_dataset()["no2"].values))

        # The next lookup catches it, drops the entry, and misses — so the
        # caller falls through to the lazy path and gets real numbers again.
        self.assertIsNone(self.cube_cache.lookup("goes_bad"))

    @staticmethod
    def _first_chunk_file(cube_dir: str) -> str:
        for dirpath, _dirnames, filenames in os.walk(cube_dir):
            for name in filenames:
                if not name.startswith("zarr.json") and not name.startswith("."):
                    return os.path.join(dirpath, name)
        raise AssertionError(f"no chunk file under {cube_dir}")


@requires_stack
class RefusalTests(StoreTestCase):
    """Where the round-trip contract cannot be satisfied for a given source,
    the cube is not written — a negative-cache entry is recorded instead, so a
    known-unsafe product is not retried on every open. Because the key carries
    the pipeline version, a later fix un-blacklists it automatically."""

    def _uncubeable(self):
        import numpy as np
        import xarray as xr

        # An attr Zarr cannot serialize: the write fails on the *representation*
        # of this dataset, not on the environment.
        return xr.Dataset({"v": ("x", np.array([1.0, 2.0]))}, attrs={"junk": object()})

    def test_a_source_that_cannot_satisfy_the_contract_is_refused_not_written(self):
        self.assertFalse(self.cube_cache.write_cube(self._uncubeable(), "bad"))

        self.assertTrue(self.cube_cache.is_refused("bad"))
        self.assertIsNone(self.cube_cache.lookup("bad"))

    def test_a_refused_source_is_not_retried_on_the_next_open(self):
        self.cube_cache.write_cube(self._uncubeable(), "bad")

        with unittest.mock.patch.object(
            self.cube_cache, "_prepare_for_write", side_effect=AssertionError("retried")
        ):
            self.assertFalse(self.cube_cache.write_cube(self._uncubeable(), "bad"))

    def test_a_pipeline_version_bump_un_blacklists_a_refused_source(self):
        """The refusal is recorded under the *cache key*, which contains the
        pipeline version — so the fix that makes a product cubeable also
        retires its own negative-cache entry, with nothing to remember to
        clear."""
        from services.cube_cache import cache_key

        before = cache_key("src", "1", "h5netcdf")
        after = cache_key("src", "2", "h5netcdf")
        self.cube_cache.write_cube(self._uncubeable(), before)

        self.assertTrue(self.cube_cache.is_refused(before))
        self.assertFalse(self.cube_cache.is_refused(after))

    def test_a_name_collision_created_by_escaping_is_refused_rather_than_merged(self):
        """Escaping ``/`` is what makes collided leaves storable, but it can
        itself collide. Silently merging two variables into one is precisely
        the T25 failure the qualified names exist to prevent, so refuse."""
        import numpy as np
        import xarray as xr

        from services.cube_cache import _SLASH_ESCAPE

        ds = xr.Dataset({
            "product/foo": ("x", np.array([1.0, 2.0])),
            f"product{_SLASH_ESCAPE}foo": ("x", np.array([3.0, 4.0])),
        })

        self.assertFalse(self.cube_cache.write_cube(ds, "escape_collision"))
        self.assertTrue(self.cube_cache.is_refused("escape_collision"))

    def test_an_environment_failure_is_not_recorded_as_a_refusal(self):
        """A disk error is not the data's fault. Blacklisting a perfectly
        cubeable source because the volume was briefly full would keep it
        uncached until the next pipeline-version bump."""
        with unittest.mock.patch("services.cube_cache.json.dump", side_effect=OSError("no space")):
            with self.assertRaises(OSError):
                self.cube_cache.write_cube(self.make_dataset(), "diskfull")

        self.assertFalse(self.cube_cache.is_refused("diskfull"))


@requires_stack
class EvictionTests(StoreTestCase):
    """The store is bounded by an explicit byte cap — deliberately not a
    percentage of free space, which silently expands to fill any disk (how
    docker_data.vhdx reached 296 GB against ~12 GB live).

    Eviction is LRU by *last access*, not the extract cache's age-only TTL:
    reads there don't touch mtime (so a hot entry would be evicted at one
    hour) and its sweep only fires on new extractions (so a cache that stops
    growing never prunes)."""

    def _big_dataset(self, seed: int):
        import numpy as np
        import xarray as xr

        # Incompressible, so on-disk size tracks nbytes rather than collapsing.
        rng = np.random.default_rng(seed)
        return xr.Dataset(
            {"v": (("y", "x"), rng.random((64, 64)))},
            coords={"y": np.arange(64.0), "x": np.arange(64.0)},
        )

    def test_a_write_that_would_exceed_the_cap_evicts_the_coldest_cube_first(self):
        self.cube_cache.write_cube(self._big_dataset(1), "cold")
        self.cube_cache.write_cube(self._big_dataset(2), "warm")
        # Touch "warm" so "cold" is genuinely the least-recently-*accessed*.
        self.cube_cache.lookup("warm")

        # Cap the store just under what three cubes need.
        self.set_limits(CUBE_STORE_MAX_BYTES=self.cube_cache.store_size_bytes() + 1024)
        self.cube_cache.write_cube(self._big_dataset(3), "new")

        self.assertIsNone(self.cube_cache.lookup("cold"))
        self.assertIsNotNone(self.cube_cache.lookup("warm"))
        self.assertIsNotNone(self.cube_cache.lookup("new"))

    def test_a_cube_read_every_turn_is_never_evicted(self):
        """The whole point of LRU-by-access over TTL: the cube being used is
        the one that must survive."""
        self.cube_cache.write_cube(self._big_dataset(1), "hot")
        # Room for one resident cube plus the incoming one — so every write
        # must evict exactly one entry, and the question is *which*.
        headroom = int(self._big_dataset(0).nbytes) + 2048
        self.set_limits(CUBE_STORE_MAX_BYTES=self.cube_cache.store_size_bytes() + headroom)

        for i in range(4):
            self.assertIsNotNone(self.cube_cache.lookup("hot"))
            self.cube_cache.write_cube(self._big_dataset(10 + i), f"churn{i}")

        self.assertIsNotNone(self.cube_cache.lookup("hot"))

    def test_a_dataset_over_the_per_cube_write_cap_is_skipped(self):
        """Writing a cube reads, compresses and writes the whole dataset —
        heavier than the lazy open ``bundle_open_max_uncompressed_bytes``
        gates, so it gets its own, lower cap."""
        self.set_limits(CUBE_WRITE_MAX_BYTES=1)

        self.assertFalse(self.cube_cache.write_cube(self.make_dataset(), "toobig"))
        self.assertIsNone(self.cube_cache.lookup("toobig"))
        # Not a contract failure — the source is fine, it is merely large, so
        # it must not be blacklisted until the next pipeline-version bump.
        self.assertFalse(self.cube_cache.is_refused("toobig"))

    def test_the_write_cap_stays_below_the_bundle_open_cap(self):
        from config.settings import Settings

        settings = Settings()
        self.assertLess(settings.cube_write_max_bytes, settings.bundle_open_max_uncompressed_bytes)


@requires_stack
class EarnInTests(unittest.IsolatedAsyncioTestCase):
    """Cubing is triggered on the *second* open of a source, never the first,
    so a one-shot question pays nothing and the cold path costs exactly what it
    costs today."""

    def setUp(self):
        StoreTestCase.setUp(self)

    set_limits = StoreTestCase.set_limits
    make_dataset = StoreTestCase.make_dataset

    async def test_the_first_open_of_a_source_writes_no_cube(self):
        task = self.cube_cache.consider_write(self.make_dataset(), "k", source="/data/a.zip")

        self.assertIsNone(task)
        self.assertIsNone(self.cube_cache.lookup("k"))

    async def test_the_second_open_triggers_exactly_one_background_write(self):
        ds = self.make_dataset()
        self.cube_cache.consider_write(ds, "k", source="/data/a.zip")

        task = self.cube_cache.consider_write(ds, "k", source="/data/a.zip")

        self.assertIsNotNone(task)
        await task
        self.assertIsNotNone(self.cube_cache.lookup("k"))

    async def test_a_third_open_does_not_start_a_second_write_for_the_same_key(self):
        ds = self.make_dataset()
        for _ in range(2):
            task = self.cube_cache.consider_write(ds, "k", source="/data/a.zip")
        await task

        self.assertIsNone(self.cube_cache.consider_write(ds, "k", source="/data/a.zip"))

    async def test_concurrent_second_and_third_opens_trigger_only_one_write(self):
        """Two tool calls in the same turn can both miss and both reach the
        earn-in check before either write lands."""
        ds = self.make_dataset()
        self.cube_cache.consider_write(ds, "k", source="/data/a.zip")

        second = self.cube_cache.consider_write(ds, "k", source="/data/a.zip")
        third = self.cube_cache.consider_write(ds, "k", source="/data/a.zip")

        self.assertIsNotNone(second)
        self.assertIsNone(third)
        await second

    async def test_the_count_is_kept_per_source_not_globally(self):
        ds = self.make_dataset()
        self.cube_cache.consider_write(ds, "k1", source="/data/a.zip")

        self.assertIsNone(self.cube_cache.consider_write(ds, "k2", source="/data/b.zip"))


@requires_stack
class ContentionTests(unittest.IsolatedAsyncioTestCase):
    """The process runs --workers 1, so a background writer competes directly
    with the foreground turn for dask's num_workers=2, the _hdf5_open_lock and
    RSS. An OOM here takes down every session, not one."""

    def setUp(self):
        StoreTestCase.setUp(self)

    set_limits = StoreTestCase.set_limits
    make_dataset = StoreTestCase.make_dataset

    async def _earn_in(self, key: str, source: str):
        ds = self.make_dataset()
        self.cube_cache.consider_write(ds, key, source=source)
        return self.cube_cache.consider_write(ds, key, source=source)

    async def test_two_eligible_sources_never_build_a_cube_concurrently(self):
        """Concurrency buys nothing — same disk — and multiplies memory risk."""
        import asyncio

        in_flight = 0
        peak = 0

        def slow_write(*args, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                import time

                time.sleep(0.05)
                return True
            finally:
                in_flight -= 1

        with unittest.mock.patch.object(self.cube_cache, "write_cube", side_effect=slow_write):
            tasks = [await self._earn_in("k1", "/a.zip"), await self._earn_in("k2", "/b.zip")]
            await asyncio.gather(*tasks)

        self.assertEqual(peak, 1)

    async def test_a_cube_is_not_built_while_a_turn_is_in_flight(self):
        """A full cube read/compress/write on the cold path would tax the turn
        the researcher is actually watching, and re-enter the memory regime the
        lazy open exists to escape."""
        import asyncio

        with self.cube_cache.active_turn():
            task = await self._earn_in("k", "/a.zip")
            await asyncio.sleep(0.05)
            self.assertIsNone(self.cube_cache.lookup("k"))  # held off

        await task
        self.assertIsNotNone(self.cube_cache.lookup("k"))

    async def test_a_write_that_waits_out_a_turn_gives_up_rather_than_leaking(self):
        import asyncio

        with unittest.mock.patch.object(self.cube_cache, "_TURN_IDLE_WAIT_SECONDS", 0.01):
            with self.cube_cache.active_turn():
                task = await self._earn_in("k", "/a.zip")
                await asyncio.sleep(0.05)
                await task

        self.assertIsNone(self.cube_cache.lookup("k"))


@requires_stack
class ExtractDirPinningTests(StoreTestCase):
    """A cube is written from a *lazy* Dataset whose members still live in the
    bundle-extract cache. That cache sweeps by directory mtime on a 1-hour TTL
    and reads do not touch mtime — so a long write can have its own source
    files deleted underneath it."""

    def test_a_pinned_extract_dir_survives_a_prune_that_would_otherwise_sweep_it(self):
        import time

        from services.open_handle import _prune_extract_cache

        with tempfile.TemporaryDirectory() as root:
            pinned = os.path.join(root, "pinned")
            unpinned = os.path.join(root, "unpinned")
            for d in (pinned, unpinned):
                os.makedirs(d)
                os.utime(d, (time.time() - 7200, time.time() - 7200))  # long past the TTL

            with self.cube_cache.pin_path(pinned):
                _prune_extract_cache(root)

                self.assertTrue(os.path.isdir(pinned))
                self.assertFalse(os.path.isdir(unpinned))

    def test_the_pin_is_released_when_the_write_finishes(self):
        with tempfile.TemporaryDirectory() as root:
            with self.cube_cache.pin_path(root):
                self.assertTrue(self.cube_cache.is_pinned(root))
            self.assertFalse(self.cube_cache.is_pinned(root))


@requires_stack
class PipelineVersionEnforcementTests(unittest.TestCase):
    """A cube written by superseded pipeline logic reproduces a fixed bug
    indefinitely with no observable signal — the version constant is the only
    thing standing between a correctness fix and a stale interpretation served
    forever. Enforcement, not convention: editing any interpreting function
    without bumping the constant fails here."""

    def test_the_pinned_fingerprint_matches_the_pipeline_source(self):
        from services.open_handle import OPEN_PIPELINE_SOURCE_FINGERPRINT, pipeline_source_fingerprint

        self.assertEqual(
            pipeline_source_fingerprint(),
            OPEN_PIPELINE_SOURCE_FINGERPRINT,
            "The open pipeline's interpreting functions changed. A cube written by the old "
            "logic and served after this change would reproduce the old interpretation "
            "forever, silently. Bump OPEN_PIPELINE_VERSION and update "
            "OPEN_PIPELINE_SOURCE_FINGERPRINT in services/open_handle.py.",
        )

    def test_the_fingerprint_moves_when_a_pipeline_function_is_edited(self):
        import inspect

        from services.open_handle import pipeline_source_fingerprint

        before = pipeline_source_fingerprint()
        real = inspect.getsource

        def altered(obj):
            source = real(obj)
            return source + "\n# an interpretation changed here\n"

        with unittest.mock.patch("inspect.getsource", side_effect=altered):
            after = pipeline_source_fingerprint()

        self.assertNotEqual(before, after)

    def test_every_interpreting_function_is_covered_by_the_fingerprint(self):
        """The guard is only as good as its list. These are the functions that
        *interpret* a file rather than merely read it — each has been wrong and
        then fixed at least once."""
        from services.open_handle import _PIPELINE_SOURCE_FUNCTIONS

        self.assertEqual(
            sorted(fn.__name__ for fn in _PIPELINE_SOURCE_FUNCTIONS),
            [
                "_apply_declared_dimension_names",
                "_open_netcdf",
                "_open_netcdf_bundle",
                "_order_bundle_time",
                "_promote_lat_lon_coords",
                "_strip_concat_unsafe_coord_attrs",
                "_synthesize_member_time_coord",
            ],
        )


@unittest.skipIf(
    any(
        importlib.util.find_spec(name) is None
        for name in REQUIRED_MODULES + ["fastmcp", "langchain_mcp_adapters"]
    ),
    "open_handle integration dependencies are not installed",
)
class OpenHandleCubeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end through ``open_handle``: what a researcher actually gets on
    their second and third question about the same retrieval."""

    set_limits = StoreTestCase.set_limits

    async def asyncSetUp(self):
        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools
        from fake_earthdata_mcp import FakeEarthdataMCPServer, HandleVolume, build_fake_mcp

        StoreTestCase.setUp(self)
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

    def _store_entries(self):
        return sorted(os.listdir(self.store_root)) if os.path.isdir(self.store_root) else []

    async def test_the_first_question_writes_no_cube_and_costs_what_it_costs_today(self):
        from services.open_handle import open_handle

        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()

        self.assertEqual(self._store_entries(), [])

    async def test_the_second_question_cubes_the_retrieval_and_the_third_is_served_from_it(self):
        import numpy as np

        import services.open_handle as open_handle_module
        from services.open_handle import open_handle

        first = await open_handle("obs_cubed", self.tools)
        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()

        # The pipeline must not run again: if the third open still paid for the
        # unzip + N per-member opens + concat, the cache bought nothing.
        with unittest.mock.patch.object(
            open_handle_module, "_open_netcdf_bundle", side_effect=AssertionError("pipeline re-ran")
        ):
            third = await open_handle("obs_cubed", self.tools)

        np.testing.assert_array_equal(third["no2"].values, first["no2"].values)
        self.assertEqual(dict(third.sizes), dict(first.sizes))

    async def test_a_cached_answer_is_numerically_identical_to_an_uncached_one(self):
        """User story #2: a cache can never change a scientific result."""
        import xarray as xr

        from services.open_handle import open_handle

        uncached = (await open_handle("obs_cubed", self.tools)).load()
        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()

        cached = (await open_handle("obs_cubed", self.tools)).load()

        xr.testing.assert_identical(uncached, cached)

    async def test_a_corrupt_cube_costs_the_turn_nothing(self):
        """User story #3: the answer still arrives, just uncached."""
        import numpy as np

        from services.open_handle import open_handle

        first = await open_handle("obs_cubed", self.tools)
        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()
        key = self._store_entries()[0]
        with open(os.path.join(self.store_root, key, "manifest.json"), "w") as f:
            f.write("{ not json")

        served = await open_handle("obs_cubed", self.tools)

        np.testing.assert_array_equal(served["no2"].values, first["no2"].values)

    async def test_a_hit_and_a_miss_are_both_visible_on_metrics(self):
        """User story #6: a cache that is never populated and a cache that
        does not help are different problems with different fixes."""
        from prometheus_client import REGISTRY

        from services.open_handle import open_handle

        def hits():
            return REGISTRY.get_sample_value("cache_hits_total", {"cache_level": "zarr"}) or 0.0

        def misses():
            return REGISTRY.get_sample_value("cache_misses_total") or 0.0

        hits_before, misses_before = hits(), misses()
        await open_handle("obs_cubed", self.tools)
        await open_handle("obs_cubed", self.tools)
        await self._drain_writes()
        self.assertEqual(hits(), hits_before)
        self.assertEqual(misses(), misses_before + 2)

        await open_handle("obs_cubed", self.tools)

        self.assertEqual(hits(), hits_before + 1)

    async def test_a_zarr_export_is_never_cubed(self):
        """A Zarr export is already a cube and a Parquet export is a table —
        neither goes through the interpretation pipeline this caches."""
        import xarray as xr

        from services.open_handle import open_handle

        self.volume.add_zarr("obs_zarr", lambda: xr.Dataset({"no2": (("y", "x"), [[1.0, 2.0]])}))
        for _ in range(3):
            await open_handle("obs_zarr", self.tools)
        await self._drain_writes()

        self.assertEqual(self._store_entries(), [])


@requires_stack
class ChunkingTests(StoreTestCase):
    """T51 measured that the T50 crop does **not** push down to a hyperslab
    read: a 700x reduction in cells reduced bought 3.47x wall-clock and *zero*
    reduction in bytes read, because ``chunks={}`` gives one chunk per variable
    per file and dask materializes the whole granule before the crop trims it.

    So the cube's headroom is the 86.8 MiB, not the 0.17 s — and a spatially
    monolithic cube would reproduce exactly that, making every regional
    question pay the full continental read. Spatial chunking is not a judgment
    call any more."""

    def _wide_dataset(self):
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {"no2": (("time", "lat", "lon"), np.zeros((3, 700, 900), dtype="float32"))},
            coords={
                "time": np.array(
                    ["2026-07-01", "2026-07-02", "2026-07-03"], dtype="datetime64[ns]"
                ),
                "lat": np.linspace(20.0, 55.0, 700),
                "lon": np.linspace(-130.0, -65.0, 900),
            },
        )

    def test_the_cube_is_chunked_spatially_so_a_regional_read_is_partial(self):
        self.cube_cache.write_cube(self._wide_dataset(), "wide")

        cached = self.cube_cache.lookup("wide")
        chunks = cached["no2"].chunksizes

        self.assertLess(max(chunks["lat"]), 700)
        self.assertLess(max(chunks["lon"]), 900)

    def test_the_time_axis_stays_in_one_chunk_so_reducing_over_time_is_one_read(self):
        """Regional subset, reduce over time is the shape every question about
        this data takes."""
        self.cube_cache.write_cube(self._wide_dataset(), "wide")

        cached = self.cube_cache.lookup("wide")

        self.assertEqual(max(cached["no2"].chunksizes["time"]), 3)

    def test_a_grid_smaller_than_the_target_chunk_is_left_in_one_piece(self):
        """Chunking finer than the data is pure overhead — more files, more
        metadata, no smaller read."""
        self.cube_cache.write_cube(self.make_dataset(), "small")

        cached = self.cube_cache.lookup("small")

        self.assertEqual(max(cached["no2"].chunksizes["lat"]), 2)


@requires_stack
class SourceDatasetIsNotMutatedTests(StoreTestCase):
    """The writer cubes the very Dataset the second open produced — the same
    object the turn in flight is still using to answer the current question.
    Preparing it for Zarr (stripping ``.encoding``, escaping names, rechunking)
    must therefore never touch the original, or landmine 1 fires live rather
    than on a later read."""

    def test_writing_a_cube_leaves_the_callers_dataset_untouched(self):
        import numpy as np
        import xarray as xr

        ds = xr.Dataset(
            {"scaled_no2": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"valid_range": [0, 30000]})},
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
        )
        ds["scaled_no2"].encoding["scale_factor"] = 2.0e13

        self.cube_cache.write_cube(ds, "nomutate")

        self.assertEqual(ds["scaled_no2"].encoding.get("scale_factor"), 2.0e13)
        self.assertEqual(ds["scaled_no2"].attrs.get("valid_range"), [0, 30000])

    def test_writing_a_cube_leaves_slash_qualified_names_on_the_callers_dataset(self):
        import numpy as np
        import xarray as xr

        ds = xr.Dataset({"product/foo": ("x", np.array([1.0, 2.0]))})

        self.cube_cache.write_cube(ds, "nomutate2")

        self.assertIn("product/foo", ds.data_vars)


@unittest.skipIf(
    any(
        importlib.util.find_spec(name) is None
        for name in REQUIRED_MODULES + ["fastmcp", "langchain_mcp_adapters"]
    ),
    "chat stream test dependencies are not installed",
)
class TurnActivityWiringTests(unittest.IsolatedAsyncioTestCase):
    """The cube writer holds off while a turn is in flight — which only works
    if something actually tells it a turn is in flight."""

    def setUp(self):
        StoreTestCase.setUp(self)

    async def test_a_chat_turn_marks_the_process_busy_for_the_whole_turn(self):
        import json
        from types import SimpleNamespace

        from services.chart_service import ChartService
        from services.chat_stream_service import ChatStreamService

        cube_cache = self.cube_cache
        seen = []

        class RecordingAgent:
            async def astream(self, input_, config, stream_mode):
                seen.append(cube_cache.turn_is_active())
                envelope = json.dumps({"summary": "done", "artifact_ids": [], "handles": []})
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        [
            event
            async for event in service.stream_chat_events(
                RecordingAgent(), RecordingAgent(), RecordingAgent(),
                "Plot TROPOMI NO2 over New Jersey for 2024-01-15", "t", "u", "r",
            )
        ]

        self.assertEqual(seen, [True])
        self.assertFalse(cube_cache.turn_is_active())
