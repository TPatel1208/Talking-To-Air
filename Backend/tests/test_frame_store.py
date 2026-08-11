"""T59 Phase 4 — the frame blob store.

The design under test is D13's split: the frame *axis* and every per-frame
disclosure live in the chart's jsonb row, and only the float32 values live in
a bounded LRU blob store. That split is not an implementation detail. It is
what makes the slider labeled and functional before the blob lands, what makes
eviction degrade rather than break, and what lets NaN travel with no sentinel.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
import warnings

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

requires_numpy = unittest.skipIf(
    importlib.util.find_spec("numpy") is None, "numpy is not installed"
)


class FrameStoreTestCase(unittest.TestCase):
    """Points the frame store at a throwaway directory for each test."""

    def setUp(self):
        from tta_backend.config.settings import get_settings

        self._store = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._store.cleanup)
        self.store_root = os.path.join(self._store.name, "frame_store")

        self.set_env(FRAME_STORE_DIR=self.store_root)
        self.addCleanup(get_settings.cache_clear)

    def set_env(self, **env):
        from tta_backend.config.settings import get_settings

        patcher = unittest.mock.patch.dict(
            os.environ, {k: str(v) for k, v in env.items()}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()

    def make_stack(self, n_frames: int = 24, ny: int = 60, nx: int = 80):
        return make_frame_stack(n_frames=n_frames, ny=ny, nx=nx)


def make_frame_stack(n_frames: int = 24, ny: int = 60, nx: int = 80):
    """A FrameStack shaped like the real thing: a labeled axis, an empty
    interval among the populated ones, and a NaN-heavy float32 array."""
    import numpy as np

    from tta_backend.preprocessing.frame_stack import CADENCE_TIER, Frame, FrameStack

    rng = np.random.default_rng(59)
    values = rng.normal(4.0e15, 1.0e15, size=(n_frames, ny, nx)).astype("float32")
    values[values < 3.0e15] = np.nan  # a real field is mostly absent
    values[1] = np.nan  # an interval nothing was retrieved for

    frames = []
    for index in range(n_frames):
        empty = index == 1
        frames.append(
            Frame(
                t_start=f"2025-06-14T{index:02d}:00:00",
                t_end=f"2025-06-14T{index + 1:02d}:00:00",
                n_granules=0 if empty else 3,
                valid_fraction=0.0 if empty else 0.94,
                qa_pass_rate=None if empty else 0.87,
                statistics={"count": 0} if empty else {
                    "count": 4200, "mean": 4.1e15, "min": 3.0e15, "max": 9.8e15,
                },
            )
        )

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An all-absent interval has no mean, which is the point of the
        # fixture, not a problem with it.
        warnings.simplefilter("ignore", RuntimeWarning)
        period_values = np.nanmean(values, axis=0).astype("float32")

    return FrameStack(
        frames=frames,
        values=values,
        period_values=period_values,
        lats=np.linspace(25.0, 50.0, ny),
        lons=np.linspace(-125.0, -65.0, nx),
        coarsen_k=(5, 5),
        cells_per_frame=ny * nx,
        cadence="hourly",
        buckets_per_frame=1,
        tier=CADENCE_TIER,
        delta=None,
        value_range=(-3.14e14, 6.21e15),
    )


@requires_numpy
class ChartRowCarriesTheAxisTests(FrameStoreTestCase):
    """D13, first half: what goes in Postgres."""

    def test_the_chart_row_carries_the_whole_axis_and_none_of_the_values(self):
        """Every label and disclosure the scrubber shows is in the jsonb row.

        The slider has to be fully labeled and functional *before* the blob
        arrives — the frontend draws the axis from this block alone, and a stop
        it cannot label is a stop it cannot honestly render (D10: an empty
        bucket is a first-class state, and a 90%-masked one must not read as a
        calm one). So the whole per-frame disclosure set travels here, in
        Postgres, where nothing evicts it.

        And *only* that: the float32 field is the one thing that must not be in
        a jsonb row, both because it is three orders of magnitude larger than
        everything else here and because NaN is not JSON.
        """
        from tta_backend.services.frame_store import store_frame_stack

        stack = self.make_stack()

        block = store_frame_stack(stack, pipeline_version="4")

        self.assertEqual(len(block["frames"]), len(stack.frames))
        first, empty = block["frames"][0], block["frames"][1]
        self.assertEqual(first["t_start"], "2025-06-14T00:00:00")
        self.assertEqual(first["t_end"], "2025-06-14T01:00:00")
        self.assertEqual(first["n_granules"], 3)
        self.assertAlmostEqual(first["valid_fraction"], 0.94)
        self.assertAlmostEqual(first["qa_pass_rate"], 0.87)
        self.assertEqual(first["statistics"]["count"], 4200)
        # "nothing was retrieved" stays distinguishable from "0% passed QA":
        # an absent rate is absent, never a zero.
        self.assertEqual(empty["n_granules"], 0)
        self.assertIsNone(empty["qa_pass_rate"])
        self.assertEqual(empty["statistics"], {"count": 0})

        # The grid the values are laid out on, so a blob is interpretable, and
        # the pooled scale, so one colour means one value at every stop (D9).
        self.assertEqual(block["lats"], [round(float(v), 6) for v in stack.lats])
        self.assertEqual(block["lons"], [round(float(v), 6) for v in stack.lons])
        self.assertEqual(block["cells_per_frame"], stack.cells_per_frame)
        self.assertEqual(block["cadence"], "hourly")
        self.assertEqual(block["tier"], "cadence")
        self.assertEqual(block["buckets_per_frame"], 1)
        self.assertEqual(block["coarsen_k"], [5, 5])
        self.assertEqual(block["value_range"], [-3.14e14, 6.21e15])

        # The scale explains itself wherever the payload goes, the same reason
        # QA_PASS_RATE_BASIS exists and the same way Phase 3's delta already
        # carries its own basis line. A 2-98 pooled across every frame *and*
        # the period mean, at native resolution before the block mean, is not
        # something a reader can infer from two numbers — and D9's whole point
        # is that this scale is not the Map tab's.
        from tta_backend.preprocessing.frame_stack import POOLED_SCALE_BASIS

        self.assertEqual(block["scale_basis"], POOLED_SCALE_BASIS)

        # jsonb, so it has to survive a real serialization round-trip.
        self.assertEqual(json.loads(json.dumps(block)), block)

        # The values are not here, in any shape. A per-frame entry carries
        # scalars and its statistics dict; nothing carries an array.
        self.assertNotIn("values", block)
        self.assertNotIn("period_values", block)
        for frame in block["frames"]:
            for key, value in frame.items():
                self.assertNotIsInstance(value, list, f"frames[].{key} holds an array")
        self.assertLess(
            len(json.dumps(block).encode()), stack.values.nbytes // 10,
            "the jsonb row is carrying something that scales with the field",
        )


@requires_numpy
class BlobRoundTripTests(FrameStoreTestCase):
    """D13, second half: what goes in the store."""

    def test_the_values_round_trip_exactly_and_nan_needs_no_sentinel(self):
        """The blob is the float32 array, bit for bit, NaN included.

        This is the third thing the split buys. A jsonb row cannot hold NaN —
        JSON has no spelling for it, so ``_build_heatmap_payload`` has to write
        ``None`` and the frontend has to know that ``None`` means "no
        observation". A float32 blob has NaN natively, so an absent cell stays
        absent through storage and transport with nothing to agree about and
        nothing to collide with a real measurement.

        The period mean rides the same blob as plane 0 — D5's "frame 0" — so a
        reader parked at the start of the scrub is looking at the Map tab's own
        field, on the frame grid, and in the cadence tier can check D4's
        identity by averaging the planes after it.
        """
        import gzip

        import numpy as np

        from tta_backend.services.frame_store import read_frames, store_frame_stack

        stack = self.make_stack()
        block = store_frame_stack(stack, pipeline_version="4")

        n_frames, ny, nx = stack.values.shape
        self.assertEqual(block["shape"], [n_frames + 1, ny, nx])
        self.assertEqual(block["dtype"], "float32")
        self.assertEqual(block["period_index"], 0)

        blob = read_frames(block["_key"], pipeline_version="4")
        self.assertIsNotNone(blob, "the values did not land in the store")

        planes = np.frombuffer(
            gzip.decompress(blob.gzipped), dtype="float32",
        ).reshape(n_frames + 1, ny, nx)

        self.assertTrue(
            np.array_equal(planes[0], stack.period_values, equal_nan=True),
            "plane 0 is not the period mean",
        )
        self.assertTrue(
            np.array_equal(planes[1:], stack.values, equal_nan=True),
            "the frames did not survive the round trip",
        )
        # Not vacuous: the fixture is mostly absent, and frame 1 — plane 2 —
        # is the interval nothing was retrieved for.
        self.assertTrue(np.isnan(planes[1]).any())
        self.assertTrue(np.isnan(planes[2]).all())

    def test_the_axis_survives_when_the_values_cannot_be_stored(self):
        """A failed write costs the slider, never the chart.

        The store is a rendering convenience D8 already refuses to regenerate,
        so nothing about it may cost a caller their chart — the same posture
        ``_render_and_store_overlay`` takes toward a failed PNG render. The
        block comes back whole and simply carries no key, which is exactly the
        state an evicted chart reaches later: an axis with no values behind it,
        drawn and labeled and not scrubbable.
        """
        from tta_backend.services.frame_store import store_frame_stack

        stack = self.make_stack()
        # A store root that cannot be created: a file where the directory goes.
        blocked = os.path.join(self._store.name, "not-a-directory")
        with open(blocked, "w", encoding="utf-8") as f:
            f.write("")
        self.set_env(FRAME_STORE_DIR=os.path.join(blocked, "frames"))

        block = store_frame_stack(stack, pipeline_version="4")

        self.assertNotIn("_key", block)
        self.assertEqual(len(block["frames"]), len(stack.frames))
        self.assertEqual(block["cells_per_frame"], stack.cells_per_frame)
        self.assertEqual(block["value_range"], [-3.14e14, 6.21e15])


@requires_numpy
class EvictionTests(FrameStoreTestCase):
    """The store is bounded by an explicit byte cap, and an LRU sweeper is the
    whole of its policy.

    No per-entry cap, deliberately — a difference from ``cube_store`` worth
    stating. ``cube_write_max_store_fraction`` exists to stop one entry big
    enough to evict the store to fit itself and then be evicted by the next
    write. D7 caps frames at 60 and D5 caps cells at 20,000, so an entry is at
    most 4.58 MiB against 1 GiB and that thrash cannot occur. Adding a cap
    anyway would be cargo-culting.
    """

    def small_stack(self, seed: int):
        stack = self.make_stack(n_frames=4, ny=20, nx=20)
        # Distinct bytes per entry, so nothing is deduplicated by accident.
        stack.values[0, 0, 0] = float(seed)
        return stack

    def sparse_stack(self):
        """The full-domain regime: 66% absent, which is what makes a real
        frame stack gzip at x3.25 (Phase 2 §2)."""
        import numpy as np

        stack = self.make_stack(n_frames=8, ny=40, nx=40)
        stack.values[:, 12:, :] = np.nan
        stack.period_values[12:, :] = np.nan
        return stack

    def test_a_write_past_the_cap_evicts_the_coldest_entry_first(self):
        from tta_backend.services.frame_store import (
            read_frames, store_size_bytes, write_frames,
        )

        cold = write_frames(self.small_stack(1), pipeline_version="4")
        warm = write_frames(self.small_stack(2), pipeline_version="4")
        read_frames(warm, pipeline_version="4")  # so "cold" is genuinely coldest

        # Room for the two resident entries and not the third.
        self.set_env(FRAME_STORE_MAX_BYTES=store_size_bytes() + 1024)
        new = write_frames(self.small_stack(3), pipeline_version="4")

        self.assertIsNone(read_frames(cold, pipeline_version="4"))
        self.assertIsNotNone(read_frames(warm, pipeline_version="4"))
        self.assertIsNotNone(read_frames(new, pipeline_version="4"))

    def test_the_cap_is_fixed_bytes_and_what_eviction_actually_enforces(self):
        """1 GiB, fixed, with an env override — never a percentage of free
        space.

        A share-of-free-space cap silently expands to fill whatever disk it
        lands on, which is how ``docker_data.vhdx`` reached 296 GB against
        ~12 GB live. ``settings.py`` already carries that lesson for
        ``cube_store``; this is a quarter of that cap, which is the right
        proportion for something D8 refuses to regenerate.

        Asserted through eviction rather than by reading the setting back: a
        cap nothing enforces is a comment.
        """
        from tta_backend.config.settings import get_settings
        from tta_backend.services.frame_store import store_size_bytes, write_frames

        self.assertEqual(get_settings().frame_store_max_bytes, 1024 ** 3)

        write_frames(self.sparse_stack(), pipeline_version="4")
        budget = store_size_bytes() * 2
        self.set_env(FRAME_STORE_MAX_BYTES=budget)
        for _ in range(6):
            write_frames(self.sparse_stack(), pipeline_version="4")

        self.assertLessEqual(store_size_bytes(), budget)

    def test_an_entry_is_charged_the_bytes_it_occupies_not_its_array_size(self):
        """Account by real bytes on disk, storing gzipped.

        A deliberate divergence from ``cube_store``, which measured
        ``ds.nbytes`` — the uncompressed in-memory footprint — while accounting
        the store in disk bytes, and so charged cubes up to 5x what they cost
        and rejected them hardest on the multi-granule retrievals the cache
        existed for. Frames compress by x1.36 on a dense regional field and
        x3.25 full-domain, and the sparse ones that compress best are precisely
        the ones an nbytes-accounting store would overcharge most.
        """
        from tta_backend.services.frame_store import (
            read_frames, store_size_bytes, write_frames,
        )

        stack = self.sparse_stack()
        raw_bytes = stack.values.nbytes + stack.period_values.nbytes

        first = write_frames(stack, pipeline_version="4")
        on_disk = store_size_bytes()
        self.assertLess(on_disk * 4, raw_bytes, "the fixture does not compress")

        # Room for three entries by what they cost, and not even one by what
        # they would weigh uncompressed in memory.
        self.set_env(FRAME_STORE_MAX_BYTES=on_disk * 3)
        second = write_frames(self.sparse_stack(), pipeline_version="4")
        third = write_frames(self.sparse_stack(), pipeline_version="4")

        for key in (first, second, third):
            self.assertIsNotNone(
                read_frames(key, pipeline_version="4"),
                "an entry was evicted for bytes it does not occupy",
            )

    def test_the_stack_a_reader_keeps_scrubbing_is_never_evicted(self):
        """The point of LRU-by-access over an age TTL: the frames someone is
        actually scrubbing are the ones that must survive. A slider click that
        finds its own blob evicted is the one failure this ordering exists to
        prevent."""
        from tta_backend.services.frame_store import (
            read_frames, store_size_bytes, write_frames,
        )

        hot = write_frames(self.small_stack(1), pipeline_version="4")
        # Room for one resident entry plus the incoming one, so every write
        # must evict exactly one and the question is only *which*.
        self.set_env(FRAME_STORE_MAX_BYTES=store_size_bytes() * 2 + 1024)

        for seed in range(4):
            self.assertIsNotNone(read_frames(hot, pipeline_version="4"))
            write_frames(self.small_stack(10 + seed), pipeline_version="4")

        self.assertIsNotNone(read_frames(hot, pipeline_version="4"))


@requires_numpy
class SelfHealTests(FrameStoreTestCase):
    """``cube_cache``'s posture, borrowed: any integrity failure drops the
    entry and returns a MISS rather than serving something subtly wrong.

    A store that can refuse is strictly better than one that can be quietly
    incorrect — and here a miss costs a slider, which D8 already says is
    disposable, while a wrong answer is a field a researcher would read."""

    def test_a_truncated_blob_is_dropped_and_read_as_a_miss(self):
        """Truncation is the failure a metadata-only check cannot see.

        Half a stack still gunzips to *something*: whole frames' worth of
        float32 that reshape cleanly and render as a plausible field with the
        tail of the scrub silently missing."""
        from tta_backend.services.frame_store import (
            read_frames, store_size_bytes, write_frames,
        )

        key = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")
        blob = os.path.join(self.store_root, key, "frames.f32.gz")
        with open(blob, "rb") as f:
            payload = f.read()
        with open(blob, "wb") as f:
            f.write(payload[: len(payload) // 2])

        self.assertIsNone(read_frames(key, pipeline_version="4"))
        # Dropped, not merely refused: a failing entry that stays resident goes
        # on failing on every read while charging the store for the privilege.
        self.assertFalse(os.path.exists(os.path.join(self.store_root, key)))
        self.assertEqual(store_size_bytes(), 0)

    def test_a_corrupted_blob_of_the_right_length_is_dropped_and_read_as_a_miss(self):
        """A bit flip that leaves the length intact. Caught by the recorded
        digest, which is the only check that sees it — and the bytes have to be
        read to be served anyway, so it costs a hash of what is already in
        hand."""
        from tta_backend.services.frame_store import read_frames, write_frames

        key = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")
        blob = os.path.join(self.store_root, key, "frames.f32.gz")
        with open(blob, "r+b") as f:
            f.seek(os.path.getsize(blob) // 2)
            f.write(b"\x00\xff")

        self.assertIsNone(read_frames(key, pipeline_version="4"))
        self.assertFalse(os.path.exists(os.path.join(self.store_root, key)))


@requires_numpy
class PipelineVersionGuardTests(FrameStoreTestCase):
    """D8: frames are never served or rebuilt across a version change.

    This is the only thing standing between D4's guarantee and a quiet lapse.
    ``_open_netcdf`` does not merely read files, it *interprets* them, and each
    interpretation has been wrong and then fixed. A stack computed under the
    old one keeps rendering forever with nothing in the picture to reveal it —
    and D8 forbids regenerating in place, so the honest outcome is a miss and
    a disabled slider, not a silently stale field."""

    def test_frames_written_under_another_pipeline_version_are_never_served(self):
        from tta_backend.services.frame_store import read_frames, write_frames

        key = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")

        self.assertIsNone(read_frames(key, pipeline_version="5"))
        # And dropped: no future version will ever match it either, so keeping
        # it would only crowd out entries that can be served.
        self.assertFalse(os.path.exists(os.path.join(self.store_root, key)))

    def test_a_startup_sweep_reclaims_what_the_store_can_never_serve(self):
        """The same reclaim ``cube_cache.sweep_store`` does, for the same
        reason it needed one.

        Entries under a superseded pipeline version keep perfectly valid
        manifests, so nothing would touch them and ``_entries`` would go on
        counting their bytes against the cap — a store full of unservable
        frames crowding out servable ones, given up only as LRU grinds them out
        one write at a time. That is what turns "a version bump costs the store
        once" into "a version bump degrades it until it happens to churn".

        A crash mid-write leaves a staging directory, and an entry whose
        manifest never landed is incomplete by definition. Neither is ever
        served, so both are space rather than correctness.
        """
        from tta_backend.services.frame_store import (
            store_size_bytes, sweep_store, write_frames,
        )

        stale = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="3")
        current = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")
        staging = os.path.join(self.store_root, "staging-abandoned-xyz")
        os.makedirs(staging)
        with open(os.path.join(staging, "frames.f32.gz"), "wb") as f:
            f.write(b"half a write")
        headless = os.path.join(self.store_root, "no-manifest")
        os.makedirs(headless)
        with open(os.path.join(headless, "frames.f32.gz"), "wb") as f:
            f.write(b"never completed")
        self.assertGreater(store_size_bytes(), 0)

        sweep_store("4")

        self.assertFalse(os.path.exists(os.path.join(self.store_root, stale)))
        self.assertFalse(os.path.exists(staging))
        self.assertFalse(os.path.exists(headless))
        self.assertTrue(os.path.exists(os.path.join(self.store_root, current)))

    def test_the_endpoint_reads_at_the_version_the_open_pipeline_is_actually_on(self):
        """Not a literal in two places. The store is handed
        ``OPEN_PIPELINE_VERSION`` itself, so bumping that constant — which is
        what a pipeline fix does — invalidates the frames it invalidates."""
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION
        from tta_backend.services.frame_store import read_frames, store_frame_stack

        block = store_frame_stack(
            self.make_stack(n_frames=4, ny=20, nx=20),
            pipeline_version=OPEN_PIPELINE_VERSION,
        )

        self.assertEqual(block["pipeline_version"], OPEN_PIPELINE_VERSION)
        self.assertIsNotNone(
            read_frames(block["_key"], pipeline_version=OPEN_PIPELINE_VERSION)
        )


if __name__ == "__main__":
    unittest.main()
