"""The bundle extract cache must be bounded by size, and swept when idle.

Phase 4 of the admission work. Admission control bounds what the heavy paths
hold in *memory*; this bounds what the open path leaves on *disk*, which is the
other exhaustible resource a burst of large retrievals can run the container
out of -- and the one this project has already been bitten by (docker_data.vhdx
grew to 296 GB and needed a prune plus a diskpart compact to reclaim ~280 GB).

The extract cache is the odd one out among the three on-disk stores:

    cube_store            CUBE_STORE_MAX_BYTES   = 4 GiB   swept at startup
    frame_store           FRAME_STORE_MAX_BYTES  = 1 GiB   swept at startup
    bundle extract cache  (none)                           swept only on write

Nothing caps it, and ``bundle_open_max_uncompressed_bytes`` allows a *single*
bundle to extract up to 8 GiB, so N retrievals inside one TTL window retain
N x 8 GiB with nothing to stop them.

Neither defect is a new discovery. ``cube_cache._evict_for`` already documents
both, in the course of explaining why the cube store deliberately does not copy
this design:

    Deliberately not the extract cache's age-only TTL: reads there don't touch
    mtime, so a hot entry would be evicted at one hour, and its sweep only
    fires on new extractions, so a cache that stops growing never prunes at
    all.

That last clause is the sharper of the two. A backend that stops receiving
bundle retrievals -- overnight, or over a weekend -- never runs the pruner
again, so whatever the busy period left behind stays until something else
happens to extract. The disk is held by a cache nobody is using.
"""
from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch


def _entry(root: str, name: str, size_bytes: int, age_seconds: float = 0.0) -> str:
    """A cache entry of a known size and age."""
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "member.nc"), "wb") as handle:
        handle.write(b"\0" * size_bytes)
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def _names(root: str) -> set[str]:
    return {name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))}


class TheExtractCacheIsBoundedBySizeTests(unittest.TestCase):
    """A cap, evicting least-recently-used first -- like the two sibling stores."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def test_entries_are_evicted_oldest_first_until_under_the_cap(self):
        from tta_backend.services.open_handle import _prune_extract_cache

        _entry(self.root, "oldest", 400, age_seconds=30)
        _entry(self.root, "middle", 400, age_seconds=20)
        _entry(self.root, "newest", 400, age_seconds=10)

        # Well inside the TTL, so age alone removes nothing: only the size cap
        # can act here, which is what makes this a test of the cap rather than
        # of the sweep that already existed.
        _prune_extract_cache(self.root, ttl_seconds=3600.0, max_bytes=900)

        self.assertEqual(
            _names(self.root), {"middle", "newest"},
            "the cache was not trimmed to its byte cap by evicting the least "
            "recently used entry. Without this a single busy hour can leave "
            "tens of GiB of extracted bundles on disk, because one bundle may "
            "extract up to 8 GiB and nothing counts the total.",
        )

    def test_a_cache_already_under_the_cap_is_left_alone(self):
        from tta_backend.services.open_handle import _prune_extract_cache

        _entry(self.root, "a", 100, age_seconds=30)
        _entry(self.root, "b", 100, age_seconds=10)

        _prune_extract_cache(self.root, ttl_seconds=3600.0, max_bytes=10_000)

        self.assertEqual(
            _names(self.root), {"a", "b"},
            "entries were evicted from a cache that was under its cap. "
            "Re-extracting a bundle costs minutes, so evicting one that fits "
            "trades a large amount of wall clock for no disk at all.",
        )

    def test_a_pinned_entry_is_never_evicted_to_make_room(self):
        """T52's pin has to hold against the size cap too, not just the TTL.

        A pinned entry is one a cube write is *currently reading from*, through
        a lazy Dataset whose reads do not touch mtime. Deleting it to reclaim
        disk pulls the files out from under an in-flight write -- the exact
        failure the pin was introduced to prevent, arriving through a new door.
        """
        from tta_backend.services.open_handle import _prune_extract_cache

        _entry(self.root, "pinned_oldest", 400, age_seconds=99)
        _entry(self.root, "evictable", 400, age_seconds=10)

        pinned_path = os.path.join(self.root, "pinned_oldest")
        with patch("tta_backend.services.cube_cache.is_pinned", side_effect=lambda p: p == pinned_path):
            _prune_extract_cache(self.root, ttl_seconds=3600.0, max_bytes=100)

        self.assertIn(
            "pinned_oldest", _names(self.root),
            "the size cap evicted an entry a cube write was reading from. The "
            "pin must bind every eviction path, not only the age sweep -- a "
            "torn write is worse than a full disk.",
        )

    def test_the_cap_still_yields_to_the_ttl(self):
        """Age and size are both reasons to evict, and neither replaces the
        other: the TTL reclaims entries nothing wants, the cap reclaims entries
        that are merely too numerous."""
        from tta_backend.services.open_handle import _prune_extract_cache

        _entry(self.root, "stale", 10, age_seconds=100_000)
        _entry(self.root, "fresh", 10, age_seconds=1)

        _prune_extract_cache(self.root, ttl_seconds=3600.0, max_bytes=10_000)

        self.assertEqual(
            _names(self.root), {"fresh"},
            "the TTL sweep stopped working once a size cap existed.",
        )


class TheExtractCacheIsSweptWhenNothingIsExtractingTests(unittest.TestCase):
    """The pruner must run somewhere other than inside a new extraction.

    Its only trigger today is the start of an extraction, so a backend that
    stops receiving bundle retrievals never prunes again and holds whatever the
    last busy period left. Both sibling stores already avoid this by sweeping
    during lifespan startup (api.py calls cube_cache.sweep_store and
    frame_store.sweep_store); this is the missing third call.
    """

    def test_a_startup_sweep_exists_and_prunes_a_stale_cache(self):
        import tempfile

        from tta_backend.services.open_handle import sweep_extract_cache

        with tempfile.TemporaryDirectory() as home:
            root = os.path.join(home, "tta_bundle_extract")
            os.makedirs(root)
            _entry(root, "left_over_from_last_week", 100, age_seconds=1_000_000)

            with patch("tempfile.gettempdir", return_value=home):
                sweep_extract_cache()

            self.assertEqual(
                _names(root), set(),
                "a stale entry survived the startup sweep, so disk left behind "
                "by the previous run is only reclaimed if and when someone "
                "happens to retrieve another bundle.",
            )

    def test_the_sweep_tolerates_a_cache_that_was_never_created(self):
        """A fresh container has extracted nothing. Startup must not fail on
        the absence of a directory that only exists after the first bundle."""
        import tempfile

        from tta_backend.services.open_handle import sweep_extract_cache

        with tempfile.TemporaryDirectory() as home:
            with patch("tempfile.gettempdir", return_value=home):
                sweep_extract_cache()  # must not raise


class TheDeploymentSweepsEveryOnDiskStoreTests(unittest.TestCase):
    """All three stores are swept at startup, not just the two that were.

    A source-level check rather than a boot test: the failure it guards is
    somebody adding a fourth store and wiring only its cap, which is exactly
    what happened to this one. Reading api.py is what makes the omission
    visible at merge time instead of as unexplained disk growth weeks later.
    """

    def test_lifespan_sweeps_the_extract_cache_alongside_the_other_stores(self):
        here = os.path.dirname(os.path.abspath(__file__))
        api_py = os.path.normpath(os.path.join(here, "..", "tta_backend", "api.py"))
        with open(api_py, "r", encoding="utf-8") as handle:
            source = handle.read()

        for call in ("cube_cache.sweep_store(", "frame_store.sweep_store("):
            self.assertIn(call, source, f"{call} vanished -- this test's premise is stale")

        self.assertIn(
            "sweep_extract_cache(", source,
            "lifespan sweeps the cube store and the frame store but not the "
            "bundle extract cache, which is the only one of the three whose "
            "disk is reclaimed solely as a side effect of new work arriving.",
        )


if __name__ == "__main__":
    unittest.main()
