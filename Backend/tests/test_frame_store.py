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

    from tta_backend.preprocessing.frame_stack import (
        CADENCE_TIER, FRAME_GRID_DELTA_BASIS, Frame, FrameStack,
    )

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
        # Phase 8's own measurement on `map_2ea3dd7b34cf`, a real regional
        # TEMPO chart at k=(5,5): the cadence tier discloses no `delta` because
        # the temporal relationship there IS identity, and still disagrees by
        # 1.876% on the arrays it ships. That is the pair this block has to
        # carry without collapsing into one.
        frame_grid_delta={
            "headline": 0.018760,
            "max_abs": 2.72e15,
            "basis": FRAME_GRID_DELTA_BASIS,
        },
        value_range=(-3.14e14, 6.21e15),
    )


def with_planes(stack, statistics=("max", "min")):
    """The same stack, carrying its own max/min planes (T59 Phase 12).

    Each plane's arrays are the mean's offset by a constant large enough that
    no finite value appears in two planes, so a test that silently read the
    *mean's* entry where it asked for a plane's fails rather than passes. The
    offset preserves the fixture's NaN pattern, because ``NaN + x`` is NaN —
    the planes stay as absent as the field they came from, which is the state
    the round trip has to carry.
    """
    from dataclasses import replace

    from tta_backend.preprocessing.frame_stack import (
        EXTENT_OVERSTATEMENT_BASIS, PLANE_AGREEMENT_BASIS, StatisticPlane,
    )

    offsets = {"max": 5.0e15, "min": -5.0e15}
    planes = {}
    for name in statistics:
        offset = offsets[name]
        planes[name] = StatisticPlane(
            values=(stack.values + offset).astype("float32"),
            period_values=(stack.period_values + offset).astype("float32"),
            # D6a decision 8's exact zero: a selection plane's frames and its
            # period plane agree bit for bit, and Phase 12 measured exactly
            # that on two live bundles.
            frame_grid_delta={
                "headline": 0.0, "max_abs": 0.0,
                "basis": PLANE_AGREEMENT_BASIS[name],
            },
            # This plane's own pooled 2-98 (D9), never the mean's.
            value_range=(-3.14e14 + offset, 6.21e15 + offset),
            # D6a decision 9, on the max plane alone. Phase 12's live figure.
            extent_overstatement=None if name != "max" else {
                "headline": 24.6985, "worst_frame": 24.8599, "ceiling": 25,
                "basis": EXTENT_OVERSTATEMENT_BASIS,
            },
        )
    return replace(stack, planes=planes)


@requires_numpy
class TheMeanEntryIsUntouchedTests(FrameStoreTestCase):
    """D6a decision 5, read as a test rather than as an intention.

    *"The mean entry keeps its exact shape, URL and cost"* is what lets D15's
    additive-and-optional rule hold **literally** — an unupgraded consumer is
    not merely compatible with a three-plane chart, it is reading the identical
    bytes at the identical URL for the identical price. That is a claim about
    the mean entry, so it is checkable on the mean entry, and it is the
    regression this phase is most likely to cause and least likely to notice.
    """

    def test_attaching_planes_moves_nothing_about_the_mean_entry(self):
        from tta_backend.services.frame_store import read_frames, store_frame_stack

        plain = store_frame_stack(self.make_stack(), pipeline_version="4")
        planed = store_frame_stack(
            with_planes(self.make_stack()), pipeline_version="4",
        )

        # The blob, byte for byte — and therefore the ETag, which is its
        # digest. Two entries, two minted keys, one identical body.
        before = read_frames(plain["_key"], pipeline_version="4")
        after = read_frames(planed["_key"], pipeline_version="4")
        self.assertEqual(before.gzipped, after.gzipped, "the mean blob's bytes moved")
        self.assertEqual(before.etag, after.etag, "the mean blob's ETag moved")

        # And the block: every key the mean chart already had, equal, with the
        # new one the only difference. `_key` is exempt because it is minted
        # per write and was never derived from the content (Phase 4 note 2).
        self.assertEqual(set(planed) - set(plain), {"planes"})
        for key in plain:
            if key == "_key":
                continue
            self.assertEqual(plain[key], planed[key], f"the mean block's {key!r} moved")

        # A chart with no planes carries no `planes` key at all, rather than an
        # empty one. Nothing in this phase produces a plane — `_attach_frames`
        # still asks for the mean alone — so every payload on the wire has to
        # be byte-identical to the one that shipped before it, and an
        # always-present empty key would be a wire change on every chart.
        self.assertNotIn("planes", plain)


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

        # Both deltas, side by side and never merged. `delta` is absent here
        # because a cadence-tier frame IS a cadence bucket; the shipped-array
        # disagreement is a separate fact that survives that identity, and a
        # block carrying only the first would let the tier that has no delta
        # go on claiming an identity its own arrays do not satisfy.
        self.assertIsNone(block["delta"])
        self.assertAlmostEqual(block["frame_grid_delta"]["headline"], 0.018760)
        self.assertAlmostEqual(block["frame_grid_delta"]["max_abs"], 2.72e15)
        self.assertEqual(
            block["frame_grid_delta"]["basis"], stack.frame_grid_delta["basis"],
        )

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
class PlaneBlobRoundTripTests(FrameStoreTestCase):
    """D6a decision 5, second half: *one store entry per statistic*.

    Per statistic, not per chart — that is the whole of the decision. Each
    plane is independently addressable so it can be fetched lazily (nobody pays
    for a max they never toggle to), and independently evictable so losing one
    degrades that statistic through the existing ``expired``/self-heal path
    rather than taking the chart's scrubber with it.
    """

    def test_each_plane_round_trips_through_an_entry_of_its_own(self):
        import gzip

        import numpy as np

        from tta_backend.services.frame_store import read_frames, store_frame_stack

        stack = with_planes(self.make_stack(n_frames=4, ny=20, nx=20))
        block = store_frame_stack(stack, pipeline_version="4")

        n_frames, ny, nx = stack.values.shape
        keys = {block["_key"]}
        for name, plane in stack.planes.items():
            key = block["planes"][name]["_key"]
            self.assertNotIn(key, keys, f"the {name} plane shares an entry")
            keys.add(key)

            blob = read_frames(key, pipeline_version="4", statistic=name)
            self.assertIsNotNone(blob, f"the {name} plane's values did not land")
            stored = np.frombuffer(
                gzip.decompress(blob.gzipped), dtype="float32",
            ).reshape(n_frames + 1, ny, nx)

            # The same layout the mean's entry uses — the period plane at
            # `period_index`, the frames in axis order after it — because a
            # plane is a fourth entry speaking the existing protocol, not a
            # reason to invent a second one.
            self.assertTrue(
                np.array_equal(stored[0], plane.period_values, equal_nan=True),
                f"plane 0 is not the {name} plane's period values",
            )
            self.assertTrue(
                np.array_equal(stored[1:], plane.values, equal_nan=True),
                f"the {name} plane's frames did not survive the round trip",
            )
            # Not vacuous. Every finite value here is offset clear of the mean
            # field's, so an entry that quietly served the mean instead would
            # reshape cleanly, render plausibly, and fail exactly here — which
            # is the whole reason the fixture is built that way.
            self.assertFalse(
                np.array_equal(stored[1:], stack.values, equal_nan=True),
                f"the {name} plane's key served the mean's values",
            )
            # The absent cells are still absent: NaN needs no sentinel in a
            # plane either.
            self.assertTrue(np.isnan(stored[2]).all())

    def test_a_planes_block_carries_its_own_numbers_and_none_of_the_axis(self):
        """The four fields that genuinely differ per plane, and nothing else.

        The axis is plane-independent and it must stay one axis. A block that
        repeated `t_start` or `valid_fraction` or `statistics` per plane would
        be keeping a second account of a quantity that has exactly one, and the
        two accounts would be free to disagree — which is the failure mode
        `DELTA_BASIS`, `FRAME_GRID_DELTA_BASIS` and `EXTENT_OVERSTATEMENT_BASIS`
        already exist in triplicate to prevent one level up.
        """
        from tta_backend.services.frame_store import store_frame_stack

        stack = with_planes(self.make_stack(n_frames=4, ny=20, nx=20))

        block = store_frame_stack(stack, pipeline_version="4")

        self.assertEqual(sorted(block["planes"]), ["max", "min"])
        # The mean is not a plane. It stays in the top-level fields where it
        # has always been, which is what makes decision 5's "keeps its exact
        # shape, URL and cost" true at the Python level and not only on a wire.
        self.assertNotIn("mean", block["planes"])

        maximum = block["planes"]["max"]
        self.assertEqual(
            sorted(maximum),
            ["_key", "extent_overstatement", "frame_grid_delta", "value_range"],
        )
        # Its own clip, not the mean's — the two are disjoint in this fixture.
        self.assertEqual(maximum["value_range"], [-3.14e14 + 5.0e15, 6.21e15 + 5.0e15])
        self.assertNotEqual(maximum["value_range"], block["value_range"])
        # Its own agreement figure, under its own basis. D6a decision 8's
        # exact zero, and a basis string that says which reduction produced it.
        self.assertEqual(maximum["frame_grid_delta"]["headline"], 0.0)
        self.assertNotEqual(
            maximum["frame_grid_delta"]["basis"], block["frame_grid_delta"]["basis"],
        )
        # D6a decision 9, on the max plane alone. A minimum has no analogous
        # overstatement story, and measuring one to be symmetric would be
        # inventing a number.
        self.assertAlmostEqual(maximum["extent_overstatement"]["headline"], 24.6985)
        self.assertIsNone(block["planes"]["min"]["extent_overstatement"])

        # None of the axis, in any plane, in any spelling.
        for name, plane_block in block["planes"].items():
            for field in (
                "frames", "shape", "dtype", "period_index", "lats", "lons",
                "cells_per_frame", "coarsen_k", "cadence", "tier",
                "buckets_per_frame", "delta", "scale_basis", "pipeline_version",
            ):
                self.assertNotIn(field, plane_block, f"{name} restates {field!r}")

        # jsonb, so the whole thing still has to survive serialization.
        self.assertEqual(json.loads(json.dumps(block)), block)


@requires_numpy
class StatisticGuardTests(FrameStoreTestCase):
    """An entry is only served to a request for the statistic it holds.

    This is T54's "content_digest ALONE is an unsafe cube key" lesson arriving
    at a third axis, and it is the *only* guard that catches this failure.
    Every plane of one chart has the same shape, the same dtype, the same grid
    and the same pipeline version, so serving the max blob where the mean was
    asked for passes every other check the store makes and renders a believable
    field with the wrong numbers in it. ``_intact`` asks whether these are the
    bytes that were *written*; nothing else asks whether they are the bytes
    that were *asked for*.
    """

    def test_a_plane_entry_is_not_served_to_a_request_for_another_statistic(self):
        from tta_backend.services.frame_store import read_frames, store_frame_stack

        stack = with_planes(self.make_stack(n_frames=4, ny=20, nx=20))
        block = store_frame_stack(stack, pipeline_version="4")
        max_key = block["planes"]["max"]["_key"]

        self.assertIsNone(read_frames(max_key, pipeline_version="4"))
        self.assertIsNone(
            read_frames(max_key, pipeline_version="4", statistic="min"),
        )
        # The other direction, and the one a defaulted argument would miss.
        self.assertIsNone(
            read_frames(block["_key"], pipeline_version="4", statistic="max"),
        )

    def test_a_refused_statistic_is_logged_where_a_wrong_field_would_not_be(self):
        """The failure this guard prevents is silent by construction — a
        plausible field, correctly shaped, with the wrong numbers in it. So the
        refusal has to say so somewhere a person will look, under the same
        event name the store's other read failures already use."""
        from tta_backend.services.frame_store import read_frames, store_frame_stack

        block = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        max_key = block["planes"]["max"]["_key"]

        with self.assertLogs("tta_backend.services.frame_store", level="WARNING") as logs:
            read_frames(max_key, pipeline_version="4")

        self.assertTrue(
            any("frame_read_failed" in record.getMessage() for record in logs.records),
        )
        self.assertTrue(
            any(getattr(record, "_reason", "").startswith("statistic_mismatch")
                for record in logs.records),
            f"the mismatch is not named in the log: {[r.__dict__ for r in logs.records]}",
        )

    def test_a_refused_statistic_leaves_the_entry_where_it_is(self):
        """Refused, never dropped — and this is the one place this module
        deliberately parts company with its own ``pipeline_version`` guard.

        That guard *drops* because the version only moves forward, so no future
        request can ever match the entry and keeping it would charge the cap for
        something unreachable. The premise is inverted here: an entry refused
        for the wrong statistic is perfectly servable to the right one. A
        mismatch means the *request* is wrong, and dropping would let a buggy
        read of the mean permanently destroy a healthy max — turning one
        degraded statistic into two, on a path whose whole job is to degrade
        narrowly.
        """
        import os

        from tta_backend.services.frame_store import read_frames, store_frame_stack

        block = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        max_key = block["planes"]["max"]["_key"]

        self.assertIsNone(read_frames(max_key, pipeline_version="4"))

        self.assertTrue(os.path.exists(os.path.join(self.store_root, max_key)))
        self.assertIsNotNone(
            read_frames(max_key, pipeline_version="4", statistic="max"),
            "a wrong-statistic read destroyed an entry that was never wrong",
        )

    def test_an_entry_written_before_statistics_existed_reads_as_the_mean(self):
        """Every entry written before this phase is a mean entry, by
        construction — no other kind could be produced. So a manifest with no
        ``statistic`` is not ambiguous and must not be treated as one: reading
        it as anything else would strip the scrubber off every chart already in
        a deployed store, for a version bump that never happened.
        """
        import json
        import os

        from tta_backend.services.frame_store import read_frames, write_frames

        key = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")
        path = os.path.join(self.store_root, key, "manifest.json")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        del manifest["statistic"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        self.assertIsNotNone(read_frames(key, pipeline_version="4"))
        # And it is still not a max entry.
        self.assertIsNone(read_frames(key, pipeline_version="4", statistic="max"))

    def test_the_stores_statistic_names_are_the_reductions_own(self):
        """``STORABLE_STATISTICS`` is a mirror, not a second opinion.

        It exists so ``api.py`` can constrain a URL segment without importing
        ``frame_stack``, which would drag xarray into API startup. A mirror that
        can drift is worse than the import it avoids, so it is pinned here —
        where importing the reduction costs nothing.
        """
        from tta_backend.preprocessing.frame_stack import PLANE_STATISTICS
        from tta_backend.services.frame_store import MEAN_STATISTIC, STORABLE_STATISTICS

        self.assertEqual(tuple(STORABLE_STATISTICS), tuple(PLANE_STATISTICS))
        self.assertIn(MEAN_STATISTIC, STORABLE_STATISTICS)


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
class PlaneEvictionTests(FrameStoreTestCase):
    """One entry per statistic means one *eviction* per statistic.

    That is the half of D6a decision 5 that costs something: three entries per
    chart reach the cap three times as fast, and the LRU will pull them apart.
    A partially-resident chart is therefore a normal state rather than a
    corrupt one, and it has to degrade the way an evicted mean already does —
    per statistic, through the existing miss-and-disable path.
    """

    def test_an_evicted_plane_costs_that_statistic_and_nothing_else(self):
        """The chart keeps scrubbing in every statistic that is still there.

        This is the state Phase 15's toggle will actually meet, and the reason
        the block keeps naming a key the store no longer holds: the axis lives
        in Postgres where nothing evicts it, so the frontend can still label
        the max as a statistic this chart *has* and report it unavailable —
        which is a different and more honest thing to draw than a chart that
        never had one.
        """
        import shutil

        from tta_backend.services.frame_store import read_frames, store_frame_stack

        block = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        max_key = block["planes"]["max"]["_key"]

        shutil.rmtree(os.path.join(self.store_root, max_key))

        self.assertIsNone(read_frames(max_key, pipeline_version="4", statistic="max"))
        # The default statistic is untouched, which is the one that matters:
        # it is what every existing consumer reads.
        self.assertIsNotNone(read_frames(block["_key"], pipeline_version="4"))
        self.assertIsNotNone(
            read_frames(block["planes"]["min"]["_key"], pipeline_version="4",
                        statistic="min"),
        )
        # And the axis is whole — every stop still labeled, in every mode.
        self.assertEqual(len(block["frames"]), 4)
        self.assertEqual(block["planes"]["max"]["extent_overstatement"]["ceiling"], 25)

    def test_a_charts_own_planes_never_evict_its_own_mean(self):
        """Three writes, one cap, and the LRU would do this silently.

        ``write_frames`` evicts to fit before every write, so a chart storing
        three planes near the cap can have its **max** entry evict its own
        **mean** — correctly, by LRU's own rules, and disastrously, because
        mean is the default and the statistic every existing consumer reads.
        The chart would scrub in max mode and be dead in the mode nobody
        toggled to. So a chart's entries are protected from *its own* writes,
        and only from those: the cold entry below is still evicted, because
        that is what the cap is for.

        The store may sit briefly over the cap as a result, which is the same
        shape of bound ``evict_to_fit``'s docstring already describes for an
        entry larger than the store: the next unprotected write reclaims it.
        """
        from tta_backend.services.frame_store import (
            read_frames, store_frame_stack, store_size_bytes, write_frames,
        )

        cold = write_frames(self.make_stack(n_frames=4, ny=20, nx=20), pipeline_version="4")
        one_entry = store_size_bytes()
        # Room for two entries, against a chart that is about to write three.
        self.set_env(FRAME_STORE_MAX_BYTES=one_entry * 2)

        block = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )

        # The cap is still enforced against everything else in the store.
        self.assertIsNone(
            read_frames(cold, pipeline_version="4"),
            "the cold entry survived, so this test never reached the cap",
        )
        # And the chart that did the writing is whole, in every statistic.
        self.assertIsNotNone(
            read_frames(block["_key"], pipeline_version="4"),
            "a chart's own plane evicted its own mean",
        )
        for name in ("max", "min"):
            self.assertIsNotNone(
                read_frames(block["planes"][name]["_key"], pipeline_version="4",
                            statistic=name),
                f"a chart's own write evicted its own {name} plane",
            )

    def test_protection_does_not_outlive_the_write_that_asked_for_it(self):
        """The next chart evicts the last one, exactly as before.

        Protection is scoped to a single ``store_frame_stack`` call — it stops
        a chart cannibalizing itself mid-write and nothing more. If it leaked
        beyond that, entries would become permanently unevictable and the cap
        would stop being a cap.
        """
        from tta_backend.services.frame_store import (
            read_frames, store_frame_stack, store_size_bytes,
        )

        first = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        self.set_env(FRAME_STORE_MAX_BYTES=store_size_bytes())

        second = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )

        self.assertIsNone(read_frames(first["_key"], pipeline_version="4"))
        self.assertIsNotNone(read_frames(second["_key"], pipeline_version="4"))


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

    def test_a_plane_entry_is_never_served_across_a_version_change_either(self):
        """D8 is about an *interpretation of the source files*, so it binds
        every plane reduced from them, not just the one that shipped first."""
        from tta_backend.services.frame_store import read_frames, store_frame_stack

        block = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        max_key = block["planes"]["max"]["_key"]

        self.assertIsNone(read_frames(max_key, pipeline_version="5", statistic="max"))
        self.assertFalse(os.path.exists(os.path.join(self.store_root, max_key)))

    def test_a_startup_sweep_reclaims_plane_entries_too(self):
        """Otherwise a version bump leaves *three* unservable entries per chart
        counting against the cap instead of one, and the store degrades three
        times as fast as the failure this sweep was written for.

        The mixed chart is the case worth constructing: one chart whose planes
        straddle the version boundary cannot arise from a single
        ``store_frame_stack``, which stamps them all alike — but it is exactly
        what a half-finished deploy leaves on disk, and the sweep has to judge
        each entry on its own manifest rather than by whatever chart it
        belongs to.
        """
        from tta_backend.services.frame_store import (
            store_frame_stack, sweep_store, write_frames,
        )

        stale = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="3",
        )
        current = store_frame_stack(
            with_planes(self.make_stack(n_frames=4, ny=20, nx=20)),
            pipeline_version="4",
        )
        mixed = with_planes(self.make_stack(n_frames=4, ny=20, nx=20))
        mixed_mean = write_frames(mixed, pipeline_version="4")
        mixed_max = write_frames(
            mixed.planes["max"], pipeline_version="3", statistic="max",
        )

        sweep_store("4")

        def resident(key):
            return os.path.exists(os.path.join(self.store_root, key))

        for name in ("max", "min"):
            self.assertFalse(resident(stale["planes"][name]["_key"]),
                             f"a superseded {name} plane survived the sweep")
            self.assertTrue(resident(current["planes"][name]["_key"]),
                            f"the sweep reclaimed a current {name} plane")
        self.assertFalse(resident(stale["_key"]))
        self.assertTrue(resident(current["_key"]))
        # Judged per entry: the half-deployed chart keeps the plane that is
        # still servable and loses only the one that is not.
        self.assertTrue(resident(mixed_mean))
        self.assertFalse(resident(mixed_max))

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
