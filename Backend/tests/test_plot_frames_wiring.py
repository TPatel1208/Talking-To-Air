"""T59 Phase 5: the gate, and wiring the frame stack into ``plot_singular``.

Scope: whether a request may be framed at all, and what the tool does with the
stack when it may. Nothing here re-tests the reduction (Phase 3,
``test_bucketed_frame_stack.py``) or the blob store (Phase 4,
``test_frame_store.py``).

The gate is first because everything else is written against it. Phase 3
measured a real limit that Phase 4 deliberately did not fold in: a
production-shaped ``build_frame_stack`` on the FULL TEMPO domain (2950x5771,
54 hourly buckets) is OOM-killed in the 3.9 GB container, while Phase 2's bare
composition survives the same bundle. D7's ``2 <= N <= 60`` is a FRAME-COUNT
gate and bounds no spatial extent at all, so without an extent limit the
auto-upgrade reaches a reduction the kernel will kill.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

TOOL_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]

# The endpoint half of the cross-seam test below needs the API app as well.
API_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "langgraph"]

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")


def _field(xr, np, times, *, ny=4, nx=4, name="no2"):
    """A (time, lat, lon) field the gate can be asked about."""
    return xr.DataArray(
        np.ones((len(times), ny, nx)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": np.linspace(30.0, 45.0, ny),
            "lon": np.linspace(-100.0, -85.0, nx),
        },
        name=name,
    )


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class FrameGateTests(unittest.TestCase):
    def test_an_unbucketable_cadence_is_refused_by_name(self):
        """A Harmony bundle member carries no ``short_name``, so an
        unregistered product resolves to "unknown" -- the same condition under
        which ``_cadence_weighted_mean`` degrades to a plain unweighted mean.
        ``build_frame_stack`` RAISES on it, and a raise reaching the tool is a
        crashed plot. The gate has to turn it into a refusal, and inventing a
        bucket width instead would label a scrubber with intervals the product
        never published."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2024-01-01T01:00"])

        refusal = frame_gate(field, time_dim="time", cadence="unknown")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "cadence_unknown")
        self.assertIn("unknown", refusal.detail)

    def test_granules_inside_one_interval_are_not_a_scrub(self):
        """D7's lower bound. Three granules in the same hour are one stop on
        an hourly axis, and a one-stop scrubber is a map with a slider glued to
        it. The count that matters is INTERVALS, not granules -- gating on
        ``n_granules >= 2`` would build a single-frame stack here and pay a
        full extra pass over the bundle for it."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = _field(
            xr, np,
            ["2024-01-01T09:05", "2024-01-01T09:20", "2024-01-01T09:50"],
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "single_interval")

    def test_a_field_with_no_time_axis_is_not_a_scrub(self):
        """A single-granule map has no axis to browse at all -- a different
        fact from three granules inside one hour, and reached before any
        bucketing happens."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((4, 4)),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(30.0, 45.0, 4), "lon": np.linspace(-100.0, -85.0, 4)},
            name="no2",
        )

        refusal = frame_gate(field, time_dim=None, cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "no_time_axis")

    def test_a_time_dimension_with_no_coordinate_is_refused_not_crashed(self):
        """A frame axis is built from TIMESTAMPS, and a dimension without a
        coordinate has none. Real granules do this -- TEMPO O3PROF's ``layer``
        carries no coordinate at all -- and ``_apply_quality_mask`` already
        guards its own per-timestep counters the same way (``time_dim in
        checked.coords``). Reaching the span lookup instead is a ``KeyError``
        out of a gate whose entire job is to not let that happen."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=("time", "lat", "lon"),
            coords={"lat": np.linspace(30.0, 45.0, 4), "lon": np.linspace(-100.0, -85.0, 4)},
            name="no2",
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "no_time_axis")

    def test_two_intervals_pass(self):
        """The smallest thing worth scrubbing, so the gate's job is to let it
        through. Written so a later limit cannot tighten into the ordinary
        case unnoticed."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"])

        self.assertIsNone(frame_gate(field, time_dim="time", cadence="hourly"))

    def test_a_field_with_a_vertical_axis_left_on_it_is_refused(self):
        """The blob is ``(1 + N, ny, nx)`` and the payload's ``shape`` says so.
        A field still carrying a vertical axis would reduce to a 4-D stack that
        reshapes to something plausible and renders as the wrong layer. T56's
        map scrubber over the vertical axis is a separate thing, and this is
        not a partial version of it."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = xr.DataArray(
            np.ones((2, 3, 4, 4)),
            dims=("time", "layer", "lat", "lon"),
            coords={
                "time": np.array(["2024-01-01T09:30", "2024-01-01T10:30"], dtype="datetime64[ns]"),
                "layer": [1000.0, 850.0, 500.0],
                "lat": np.linspace(30.0, 45.0, 4),
                "lon": np.linspace(-100.0, -85.0, 4),
            },
            name="o3",
        )

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "extra_dimensions")

    def test_an_extent_above_the_measured_ceiling_is_refused(self):
        """§7's blocker, and the reason this gate exists at all.

        A production-shaped ``build_frame_stack`` on the full TEMPO domain
        (2950x5771 = 17M native cells, 54 hourly buckets) is OOM-KILLED by the
        kernel in the 3.9 GB container -- measured, twice, via the cgroup's own
        ``memory.events``. D7's ``2 <= N <= 60`` counts FRAMES and bounds no
        spatial extent, so without this the auto-upgrade reaches a reduction
        that dies and takes the researcher's chart with it.

        Refusing costs the full-domain scrub. That is the trade, taken
        deliberately and disclosed, over shipping an unmeasured claim that a
        rechunk makes it fit."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS, frame_gate

        # The full TEMPO domain's shape, without its bytes: the gate reads the
        # sizes, never the values.
        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        oversized = field.isel(lat=[0, 1] * 1475, lon=[0, 1] * 2886)
        self.assertGreater(oversized.sizes["lat"] * oversized.sizes["lon"],
                           MAX_FRAME_NATIVE_CELLS)

        refusal = frame_gate(oversized, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "extent_too_large")
        self.assertIn(f"{MAX_FRAME_NATIVE_CELLS:,}", refusal.detail)

    def test_the_extent_ceiling_admits_the_largest_extent_measured_to_survive(self):
        """The limit is anchored on measurement, not on caution. A CONUS crop
        (1250x3000 = 3.75M native cells, the same 54 hourly buckets) COMPLETED
        at 1,342 MB in the same container the full domain died in, so a gate
        that refused it would be throwing away a scrub that demonstrably
        works."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        conus = field.isel(lat=[0, 1] * 625, lon=[0, 1] * 1500)
        self.assertEqual(conus.sizes["lat"] * conus.sizes["lon"], 1250 * 3000)

        self.assertIsNone(frame_gate(conus, time_dim="time", cadence="hourly"))

    def test_an_extent_that_frames_may_still_not_carry_extra_planes(self):
        """Phase 14's measurement, as a limit.

        ``MAX_FRAME_NATIVE_CELLS`` was measured on a ONE-statistic build, at
        the largest extent that completed: CONUS 1250x3000 at 1,342 MB. Phase
        14 re-measured the same extent asking for three statistics and the
        kernel OOM-killed it, as it did at 1.8M cells and at 1.4M cells. The
        largest extent measured to survive a three-statistic build is 1,050,000
        cells, where it peaks at 1,308 MB against the one-statistic build's
        681.8 MB -- 1.92x, which is Phase 11's own ratio reproduced at 3x the
        extent.

        So the two limits are genuinely different numbers and this is the
        smaller one. A chart between them is not refused: it keeps the mean
        scrubber it has today, because refusing it would charge an existing
        capability for a new one."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import (
            MAX_FRAME_NATIVE_CELLS,
            MAX_PLANE_NATIVE_CELLS,
            frame_gate,
            plane_gate,
        )

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        # Between the two ceilings: framable, and measured not to survive three
        # statistics at anything like this size.
        between = field.isel(lat=[0, 1] * 625, lon=[0, 1] * 1500)
        cells = between.sizes["lat"] * between.sizes["lon"]
        self.assertGreater(cells, MAX_PLANE_NATIVE_CELLS)
        self.assertLessEqual(cells, MAX_FRAME_NATIVE_CELLS)

        # The scrubber this chart has today is untouched...
        self.assertIsNone(frame_gate(between, time_dim="time", cadence="hourly"))
        # ...and the extra planes are what it may not have.
        refusal = plane_gate(between, time_dim="time")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "plane_extent_too_large")
        self.assertIn(f"{MAX_PLANE_NATIVE_CELLS:,}", refusal.detail)

    def test_an_ordinary_regional_extent_may_carry_every_plane(self):
        """The regime every live bundle this project has measured is in.

        Phase 8's and Phase 11's real TEMPO retrievals are 535x658 = 352,181
        native cells, a third of the largest extent measured to survive a
        three-statistic build -- and Phase 11 measured the extra statistics
        there at +32.5 MB and +84.4 MB. A gate that refused these would be
        withholding the tier from every chart anyone actually plots."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import plane_gate

        field = _field(
            xr, np, ["2024-01-01T09:30", "2024-01-01T10:30"], ny=2, nx=2,
        )
        regional = field.isel(lat=[0, 1] * 267, lon=[0, 1] * 329)
        self.assertEqual(regional.sizes["lat"] * regional.sizes["lon"], 534 * 658)

        self.assertIsNone(plane_gate(regional, time_dim="time"))

    def test_a_span_beyond_the_backstop_is_refused_rather_than_coarsened(self):
        """D3's backstop above D14's coarsening.

        Coarsening is what handles a span past 60 frames, and Phase 3 measured
        it as the COMMON case -- a 2.2-day TEMPO retrieval is already 54 hourly
        frames. But coarsening has no ceiling of its own: it will happily fold
        a year of hourly buckets into 60 frames of six days each and label them
        as stops on a scrub. The backstop is where a frame stops being an
        interval anyone asked for."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import MAX_SPAN_BUCKETS, frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2025-01-01T00:00"])

        refusal = frame_gate(field, time_dim="time", cadence="hourly")

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.reason, "span_too_long")
        # The reader is told the shape of the limit, not just that they hit it.
        self.assertIn(f"{MAX_SPAN_BUCKETS:,}", refusal.detail)

    def test_the_span_backstop_sits_above_the_coarsened_tier(self):
        """A month of hourly granules -- 720 buckets, D3's own illustration of
        a slider no human can use -- must still COARSEN to 60 frames rather
        than be refused. Refusing it would revert D3 and cap the scrubber at
        the ~2.5 days §2 measured."""
        import numpy as np
        import xarray as xr
        from tta_backend.preprocessing.frame_stack import frame_gate

        field = _field(xr, np, ["2024-01-01T00:00", "2024-01-31T00:00"])

        self.assertIsNone(frame_gate(field, time_dim="time", cadence="hourly"))


class _PlotHarness:
    """Building a real chart through the real tool, shared by the two classes
    below.

    A mixin rather than a base class with tests on it: ``unittest`` collects
    inherited test methods, so a second class inheriting from the first would
    silently re-run every one of its ~20 tool tests under a second name.
    """

    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.config.settings import Settings, get_settings
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools

        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        # The blob store gets its own throwaway directory, so a test never
        # writes into the deployed named volume.
        store = unittest.mock.patch.dict(
            os.environ, {"FRAME_STORE_DIR": os.path.join(self._tmpdir.name, "frame_store")},
        )
        store.start()
        self.addCleanup(store.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        self.mcp_tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        )

    def add_bundle(self, handle="obs_1", *, times=None, ny=8, nx=8, values=None, flags=None):
        """A TEMPO_NO2-shaped multi-granule bundle: science variable plus the
        sibling QA-flag variable ``col_info_for_short_name`` matches against
        collections.yaml's TEMPO_NO2 entry, so QA masking really runs."""
        import numpy as np
        import xarray as xr

        times = times or ["2024-06-01T09:30", "2024-06-01T10:30", "2024-06-01T11:30"]

        def make_ds():
            shape = (len(times), ny, nx)
            science = np.full(shape, 4.0e15) if values is None else np.asarray(values, dtype=float)
            quality = np.zeros(shape, dtype="int8") if flags is None else np.asarray(flags)
            return xr.Dataset(
                {
                    "vertical_column_troposphere": (
                        ("time", "lat", "lon"), science, {"units": "molecules/cm^2"},
                    ),
                    "main_data_quality_flag": (("time", "lat", "lon"), quality),
                },
                coords={
                    "time": np.array(times, dtype="datetime64[ns]"),
                    "lat": np.linspace(35.0, 42.0, ny),
                    "lon": np.linspace(-80.0, -73.0, nx),
                },
                attrs={"short_name": "TEMPO_NO2_L3"},
            )

        self.volume.add_zarr(handle, make_ds)
        return handle

    async def plot(self, handle="obs_1", **kwargs):
        """Invoke the real tool and return ``(model_summary, full_payload)``."""
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        emitted = {}

        plot_singular = make_plot_singular(self.mcp_tools)
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.setdefault("payload", payload),
        ):
            raw = await plot_singular.ainvoke(
                {"handle": handle, "location": "global", **kwargs}
            )
        summary = json.loads(raw)
        self.assertNotIn("error", summary)
        return summary, emitted["payload"]

    def _uneven_coverage(self, n_times, ny, nx, seed=59):
        """A field whose blocks are seen in DIFFERENT intervals -- the coverage
        pattern that makes the two reductions stop commuting.

        A rotating one-in-four column stripe, which is a crude stand-in for
        what swath tiling does to a real product: within any 2x2 block, one
        native column is observed in a timestep the other is not, so the block
        mean weights each native cell by how many intervals saw it while the
        across-frame mean weights each interval equally. A uniformly covered
        field would agree exactly at any k and would test nothing.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        values = rng.normal(4.0e15, 5.0e14, size=(n_times, ny, nx))
        for step in range(n_times):
            values[step][:, (np.arange(nx) + step) % 4 == 0] = np.nan
        return values


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in TOOL_MODULES),
    "satellite tool integration dependencies are not installed",
)
class PlotSingularFrameWiringTests(_PlotHarness, unittest.IsolatedAsyncioTestCase):
    """D7's auto-upgrade, through the real tool.

    ``plot_singular`` only: not ``plot_multiple``, not comparison panels. A
    tool the agent must CHOOSE to call gets called when someone says "animate"
    and not when they say "show me July" -- T56's own lesson was enforce in the
    composite, don't trust the agent.
    """

    async def test_a_multi_granule_map_gets_a_frame_axis_without_being_asked(self):
        """The whole point of D7: the researcher who retrieved three hours of
        TEMPO and asked for a map can browse those hours without re-asking the
        agent, which is what makes the agent re-retrieve data already on disk.

        One frame per cadence bucket, and the block lands where the Phase 4
        endpoint looks for it -- ``payload["frames"]``."""
        self.add_bundle()

        _summary, payload = await self.plot()

        frames = payload["frames"]
        self.assertEqual(
            [f["t_start"] for f in frames["frames"]],
            ["2024-06-01T09:00:00", "2024-06-01T10:00:00", "2024-06-01T11:00:00"],
        )
        self.assertEqual(frames["cadence"], "hourly")
        self.assertEqual(frames["tier"], "cadence")
        # (1 + N, ny, nx): the period mean is plane 0, D5's frame 0.
        self.assertEqual(frames["shape"][0], 4)

    async def test_the_stack_is_reachable_by_url_wired_after_the_chart_id_exists(self):
        """``_wire_overlay_urls``' convention, followed rather than worked
        around: the stack is built inside ``asyncio.to_thread``, before
        ``_save_chart`` mints a ``chart_id``, so the block leaves the reduction
        holding an internal store key and gains its url at save time.

        The key itself is never handed to the frontend -- it addresses the blob
        store directly, and the route that serves it is ownership-scoped."""
        self.add_bundle()

        _summary, payload = await self.plot()

        frames = payload["frames"]
        self.assertTrue(frames["_key"])
        self.assertEqual(
            frames["url"], f"/chart/{payload['chart_id']}/frames.f32.gz",
        )

    async def test_the_endpoint_serves_the_stack_the_plot_stored(self):
        """The Phase 4 route reads ``payload["frames"]["_key"]`` and the Phase 4
        store guards on ``OPEN_PIPELINE_VERSION``. Both halves of D8's stamp
        have to agree, and the only way to know they do is to read back what
        the plot wrote."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        self.add_bundle()

        _summary, payload = await self.plot()
        frames = payload["frames"]

        self.assertEqual(frames["pipeline_version"], OPEN_PIPELINE_VERSION)
        blob = frame_store.read_frames(
            frames["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
        )
        self.assertIsNotNone(blob)

        planes = np.frombuffer(gzip.decompress(blob.gzipped), dtype="float32")
        self.assertEqual(list(planes.reshape(frames["shape"]).shape), frames["shape"])

    async def test_a_scrubbable_chart_carries_a_max_and_a_min_plane(self):
        """D6a decisions 1, 2 and 5, through the tool rather than through the
        reduction.

        Phases 12 and 13 built the reduction and the transport against hand-made
        fixtures with nothing upstream feeding them. This is the first thing
        that makes a REAL chart carry them: ``_attach_frames`` asks for three
        statistics, and each extra one lands as its own store entry with its own
        key.

        The mean is deliberately NOT among ``planes``. It stays in the block's
        own top-level fields where it has always been -- decision 5's "the mean
        entry keeps its exact current shape, URL and cost", read at the payload
        level."""
        self.add_bundle()

        _summary, payload = await self.plot()

        frames = payload["frames"]
        self.assertEqual(sorted(frames["planes"]), ["max", "min"])
        for statistic in ("max", "min"):
            plane = frames["planes"][statistic]
            self.assertTrue(plane["_key"], statistic)
            self.assertIsNotNone(plane["value_range"], statistic)
            self.assertIn("headline", plane["frame_grid_delta"])
            self.assertIn("basis", plane["frame_grid_delta"])
        # The mean's own entry is not duplicated into the plane map.
        self.assertNotIn("mean", frames["planes"])
        self.assertTrue(frames["_key"])

    async def test_the_max_plane_discloses_how_much_ground_it_paints(self):
        """D6a decision 9, on a real chart, in the regime where it exists.

        Block max paints one native cell's value across every cell its block
        covers, so a rendered peak claims k^2 cells' worth of ground for an
        observation that holds at one. Phase 11 measured that at ~24.7x on both
        live bundles -- ~99% of the k^2=25 ceiling, and the typical case rather
        than a tail.

        Tested at k=(2,2) deliberately. Every other plane test above runs on an
        8x8 grid where the block reduction is a no-op and the figure is
        correctly absent, which is Phase 9's own lesson: an assertion written
        only in the regime where the quantity does not exist pins nothing about
        the regime real charts are in. The min plane carries no such figure --
        "measured, and there is no such quantity here" is a different statement
        from silence."""
        self.add_bundle(ny=200, nx=200, values=self._uneven_coverage(3, 200, 200))

        _summary, payload = await self.plot()

        frames = payload["frames"]
        self.assertEqual(frames["coarsen_k"], [2, 2])

        overstatement = frames["planes"]["max"]["extent_overstatement"]
        self.assertEqual(overstatement["ceiling"], 4)
        self.assertIn("basis", overstatement)
        # Real, and not a placeholder 1.0: a rendered peak here claims more
        # ground than the cell that actually held it.
        self.assertGreater(overstatement["headline"], 1.0)
        self.assertGreaterEqual(
            overstatement["worst_frame"], overstatement["headline"],
        )
        # ``ceiling`` is k^2 and is NOT an upper bound, which this fixture
        # measures at 4.0000014. Both sums are cos(latitude)-weighted per
        # cell, so a block whose max happens to sit on its lowest-weighted row
        # contributes slightly more than k^2, and the pooled figure straddles
        # it. Phase 11's "fractionally under 25, not at it" was an observation
        # about two bundles whose edge blocks held fewer than k^2 real cells,
        # never a bound -- and `_extent_overstatement`'s own docstring is
        # careful to define `ceiling` as what the figure WOULD take under equal
        # weighting rather than as a maximum. Anything rendering this as
        # "up to Nx" would be claiming more than the number supports.
        self.assertAlmostEqual(overstatement["headline"], 4.0, places=3)
        self.assertIsNone(frames["planes"]["min"]["extent_overstatement"])

    async def test_an_extent_too_large_for_planes_keeps_the_scrubber_it_had(self):
        """Phase 14's whole shape, and the reason it is a second limit rather
        than a smaller first one.

        The three-statistic build was measured OOM-killed at 1.4M, 1.8M and
        3.75M native cells -- extents where the one-statistic build completes,
        the last of them being the very extent ``MAX_FRAME_NATIVE_CELLS`` was
        derived from. Lowering that constant instead would have refused every
        chart between 1M and 4M cells outright, taking away a mean scrubber
        that works today to pay for a toggle nobody has yet.

        So: the chart keeps its frames, its key and its url, and loses only the
        planes -- and it is told which, because "no toggle and no reason" is the
        failure ``_DISCLOSED_FRAME_REFUSALS`` exists to prevent, and the fix
        here is the same one ``extent_too_large`` names.

        The ceiling is patched rather than met, so this pays for the wiring and
        not for a million-cell reduction; the real constant is pinned in
        ``FrameGateTests`` against Phase 14's measured table."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        with patch.object(frame_stack_module, "MAX_PLANE_NATIVE_CELLS", 8):
            _summary, payload = await self.plot()

        frames = payload["frames"]
        # The scrubber this chart has today, entirely intact.
        self.assertEqual(len(frames["frames"]), 3)
        self.assertTrue(frames["_key"])
        self.assertEqual(frames["url"], f"/chart/{payload['chart_id']}/frames.f32.gz")
        # And no planes, with a reason rather than a silence.
        self.assertNotIn("planes", frames)
        self.assertEqual(
            frames["planes_unavailable"]["reason"], "plane_extent_too_large",
        )

    async def test_the_max_plane_is_a_field_not_the_readouts_scalar_max(self):
        """The distinction ``PLANE_STATISTICS``' own docstring exists to keep,
        checked on a real chart.

        ``Frame.statistics["max"]`` is the per-frame REGIONAL SCALAR and has
        shipped since Phase 3; the max PLANE is one value per pixel per frame.
        They share three words and nothing else, and Phase 12 measured them as
        different numbers (50.5 against 100.0 on its fixture). A chart that
        wired one into the other's slot would render something plausible.

        The fixture's peak sits in a single pixel of a single hour, so the max
        plane must reach it where the mean plane -- averaging it against 63
        neighbours -- cannot."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        values = np.full((3, 8, 8), 4.0e15)
        values[1, 4, 4] = 9.0e15
        self.add_bundle(values=values)

        _summary, payload = await self.plot()
        frames = payload["frames"]

        def planes_of(key):
            blob = frame_store.read_frames(
                key, pipeline_version=OPEN_PIPELINE_VERSION,
                statistic="max" if key != frames["_key"] else "mean",
            )
            return np.frombuffer(
                gzip.decompress(blob.gzipped), dtype="float32",
            ).reshape(frames["shape"])

        mean_stack = planes_of(frames["_key"])
        max_stack = planes_of(frames["planes"]["max"]["_key"])

        # k=(1,1) here, so the spike survives into both -- what differs is the
        # TEMPORAL reduction, which is the plane's whole definition.
        self.assertAlmostEqual(float(max_stack[2][4, 4]), 9.0e15, delta=1e12)
        self.assertAlmostEqual(float(mean_stack[2][4, 4]), 9.0e15, delta=1e12)
        # Stop 0 is the period max, and it is NOT the period mean (D6a
        # decision 3): the spike is in one hour of three.
        period = frames["period_index"]
        self.assertAlmostEqual(float(max_stack[period][4, 4]), 9.0e15, delta=1e12)
        self.assertLess(float(mean_stack[period][4, 4]), 7.0e15)

    async def test_each_plane_is_reachable_at_a_url_of_its_own(self):
        """A path per statistic (Phase 13's decision 1), minted here for the
        same reason the mean's is: the stack is built inside
        ``asyncio.to_thread`` before ``_save_chart`` exists to mint a
        ``chart_id``, so a block leaves the reduction holding store keys and
        gains its urls at save time.

        The mean's url is unchanged and stays ``frames.f32.gz`` -- decision 5
        asks for it to be untouched outright, not untouched-unless-a-parameter
        is-passed."""
        self.add_bundle()

        _summary, payload = await self.plot()

        frames = payload["frames"]
        chart_id = payload["chart_id"]
        self.assertEqual(frames["url"], f"/chart/{chart_id}/frames.f32.gz")
        self.assertEqual(
            frames["planes"]["max"]["url"],
            f"/chart/{chart_id}/frames.max.f32.gz",
        )
        self.assertEqual(
            frames["planes"]["min"]["url"],
            f"/chart/{chart_id}/frames.min.f32.gz",
        )

    async def test_a_plane_whose_values_did_not_land_gets_no_url(self):
        """An absent url means THAT STATISTIC is unavailable, and must never
        mean the chart is broken.

        ``write_frames`` never raises -- it logs and returns ``None`` -- so a
        plane whose write fails leaves a block entry with no ``_key``, which is
        also exactly what an evicted plane looks like later. Minting a url for
        it would hand the frontend a link that 404s; withholding it is the same
        rule the mean has always followed, applied per plane.

        The failure is injected at the store rather than simulated in the
        block, because the claim under test is that the two agree."""
        import tta_backend.services.frame_store as frame_store_module

        self.add_bundle()

        real_write = frame_store_module.write_frames

        def fail_the_max(source, *, pipeline_version, statistic="mean", **kwargs):
            if statistic == "max":
                return None
            return real_write(
                source, pipeline_version=pipeline_version, statistic=statistic,
                **kwargs,
            )

        with patch.object(frame_store_module, "write_frames", fail_the_max):
            _summary, payload = await self.plot()

        frames = payload["frames"]
        # The chart, the mean scrubber and the surviving plane are all fine.
        self.assertTrue(payload["values"])
        self.assertEqual(frames["url"], f"/chart/{payload['chart_id']}/frames.f32.gz")
        self.assertIn("url", frames["planes"]["min"])
        # The max is present as an axis entry and unreachable, not missing.
        self.assertIn("max", frames["planes"])
        self.assertNotIn("_key", frames["planes"]["max"])
        self.assertNotIn("url", frames["planes"]["max"])

    async def test_the_agent_is_told_which_statistics_it_can_offer(self):
        """D6a decision 7, and Phase 5 decision 2's rule underneath it: the
        agent never sees the payload, so a capability that stops at the payload
        is one nobody is told about.

        The mean is named alongside max and min even though it is not a
        ``planes`` key -- the list answers "what can this chart be scrubbed as",
        and an agent told only "max, min" would reasonably conclude the default
        is not among them."""
        self.add_bundle()

        summary, _payload = await self.plot()

        self.assertEqual(summary["frames"]["statistics"], ["mean", "max", "min"])

    async def test_the_agent_is_not_offered_a_statistic_that_would_404(self):
        """It must say what is REACHABLE, not what was requested.

        A plane whose write failed, or which is evicted later, has a block entry
        and no key. Deriving this list from the requested statistics would let
        the agent offer a toggle that 404s -- a promise the researcher watches
        fail rather than one they were never made."""
        import tta_backend.services.frame_store as frame_store_module

        self.add_bundle()

        real_write = frame_store_module.write_frames

        def fail_the_max(source, *, pipeline_version, statistic="mean", **kwargs):
            if statistic == "max":
                return None
            return real_write(
                source, pipeline_version=pipeline_version, statistic=statistic,
                **kwargs,
            )

        with patch.object(frame_store_module, "write_frames", fail_the_max):
            summary, _payload = await self.plot()

        self.assertEqual(summary["frames"]["statistics"], ["mean", "min"])

    async def test_the_agent_is_told_why_a_chart_offers_only_the_mean(self):
        """The other half of the same rule. A chart above the plane ceiling can
        be scrubbed and cannot be toggled, and the agent relaying "narrow the
        region and you get max and min" is only possible if it is told which
        limit was hit."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        with patch.object(frame_stack_module, "MAX_PLANE_NATIVE_CELLS", 8):
            summary, _payload = await self.plot()

        self.assertEqual(summary["frames"]["statistics"], ["mean"])
        self.assertEqual(
            summary["frames"]["planes_unavailable"], "plane_extent_too_large",
        )

    async def test_an_unregistered_product_says_why_there_is_no_scrubber(self):
        """The refusal above the backstop is disclosed, not silent (D3).

        An off-registry product's cadence is "unknown" -- a fact this backend
        does not have, never a default of "daily" -- and a frame axis has no
        interval width to use. The map is unaffected and still ships; what
        would be wrong is the researcher finding no slider and no reason."""
        import numpy as np
        import xarray as xr

        def make_ds():
            return xr.Dataset(
                {"aod": (
                    ("time", "lat", "lon"), np.full((3, 8, 8), 0.4), {"units": "1"},
                )},
                coords={
                    "time": np.array(
                        ["2024-06-01T09:30", "2024-06-01T10:30", "2024-06-01T11:30"],
                        dtype="datetime64[ns]",
                    ),
                    "lat": np.linspace(35.0, 42.0, 8),
                    "lon": np.linspace(-80.0, -73.0, 8),
                },
            )

        self.volume.add_zarr("obs_unregistered", make_ds)

        summary, payload = await self.plot("obs_unregistered")

        self.assertNotIn("frames", payload)
        self.assertEqual(payload["frames_unavailable"]["reason"], "cadence_unknown")
        # The agent only ever sees the compact summary, so a disclosure absent
        # from it is a disclosure nobody is ever told.
        self.assertEqual(
            summary["frames_unavailable"]["reason"], "cadence_unknown",
        )

    async def test_a_single_granule_map_is_left_alone_without_an_explanation(self):
        """The other kind of no-scrubber, and it is NOT disclosed. Nothing the
        request implied was withheld: one granule has no time axis to browse,
        and saying so on every snapshot map would be noise about a feature
        nobody asked for."""
        self.add_bundle(times=["2024-06-01T09:30"])

        summary, payload = await self.plot()

        self.assertNotIn("frames", payload)
        self.assertNotIn("frames_unavailable", payload)
        self.assertNotIn("frames_unavailable", summary)

    async def test_the_frame_block_is_additive_and_changes_nothing_else(self):
        """D15: ``type`` stays ``"heatmap"`` and frames are an optional extra
        block. Every unupgraded consumer -- ``CHART_TABS``,
        ``_RENDER_TYPE_TO_ARTIFACT_TYPE``, export's ``heatmap_multi`` branches
        -- keeps reading the aggregate and stays correct. That is the fallback,
        not a bug.

        Asserted by building the same chart twice, once with the reduction
        forced to fail, and requiring everything a consumer reads to be
        identical. A new render type would have needed a new artifact prefix, a
        new ``ArtifactMetadata`` model and new export semantics, for zero gain."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()
        _summary, framed = await self.plot()

        def refuse(*args, **kwargs):
            raise RuntimeError("no stack for you")

        with patch.object(frame_stack_module, "build_frame_stack", refuse):
            _summary, plain = await self.plot()

        self.assertEqual(framed["type"], "heatmap")
        self.assertEqual(framed["type"], plain["type"])
        self.assertTrue(framed["chart_id"].startswith("map_"))
        for key in ("lats", "lons", "values", "statistics", "vmin", "vmax", "scale"):
            self.assertEqual(framed[key], plain[key], key)
        # The aggregate is the export, framed or not (D12).
        self.assertEqual(
            {k: v for k, v in framed["export"].items() if k != "frames"},
            {k: v for k, v in plain["export"].items() if k != "frames"},
        )

    async def test_a_failed_reduction_costs_the_scrubber_and_not_the_chart(self):
        """``_render_and_store_overlay``'s posture, applied here: a frame stack
        is a rendering convenience, and nothing about building one may cost a
        researcher the map they actually asked for."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        def refuse(*args, **kwargs):
            raise RuntimeError("no stack for you")

        with patch.object(frame_stack_module, "build_frame_stack", refuse):
            summary, payload = await self.plot()

        self.assertNotIn("frames", payload)
        self.assertEqual(summary["render_type"], "heatmap")
        self.assertTrue(payload["values"])

    async def test_an_hour_qa_rejected_is_distinguishable_from_an_hour_nothing_reached(self):
        """D10 through the wiring. A 90%-masked bucket renders as a clean low
        uniform field, indistinguishable from a calm day to someone scrubbing
        for an event, so the interval's own QA rate has to travel with it.

        The rate can only come from ``MaskedField.counts``' per-timestep areas
        -- finding 12's roll-up sums ``passing_area``/``checked_area`` over the
        bucket and divides ONCE, and a quotient cannot be unmixed back into its
        terms. A caller that forgets to pass them gets ``None`` on every frame:
        "QA never ran", silently substituted for "QA rejected all of it"."""
        import numpy as np

        flags = np.zeros((3, 8, 8), dtype="int8")
        flags[1] = 1                      # hour 2: every pixel fails QA
        self.add_bundle(flags=flags)

        _summary, payload = await self.plot()

        rates = [f["qa_pass_rate"] for f in payload["frames"]["frames"]]
        self.assertEqual(rates[0], 1.0)
        self.assertEqual(rates[1], 0.0)   # a real 0%, not an absent rate
        self.assertEqual(rates[2], 1.0)
        # ...and the frame itself is empty, with no measurement invented for it.
        self.assertEqual(payload["frames"]["frames"][1]["statistics"], {"count": 0})

    async def test_the_reduction_is_handed_the_region_area_the_mask_recorded(self):
        """D5a's other half. ``region_area`` exists in exactly one place -- the
        rasterized geometry mask -- and a frame's coverage denominated on
        anything else describes a different field: the BOUNDING BOX, on which a
        complete retrieval over the continental US reads 60% covered because
        the Atlantic counts as missing observations.

        Asserted at the call rather than through the number, because the two
        denominators coincide on a rectangular region and the failure is
        invisible exactly where a test would most like to see it. The QA
        counters ride the same check: re-deriving either is a second full I/O
        pass over a lazily-opened bundle, with nothing saying so."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        seen = {}
        real = frame_stack_module.build_frame_stack

        def spy(da, **kwargs):
            seen.update(kwargs)
            return real(da, **kwargs)

        with patch.object(frame_stack_module, "build_frame_stack", spy):
            _summary, payload = await self.plot()

        self.assertGreater(seen["region_area"], 0.0)
        self.assertIn("checked_area_by_time", seen["qa_counts"])
        self.assertIn("passing_area_by_time", seen["qa_counts"])
        # The pooled 2-98 histogram cannot choose its bins until the field's
        # extremes are known, and ``apply_quality_mask`` measured them on the
        # walk it was already doing. Without them that is a second full pass.
        self.assertIn("value_min", seen["qa_counts"])
        self.assertIn("value_max", seen["qa_counts"])
        self.assertIsNotNone(payload["frames"]["value_range"])

    async def test_the_frame_spec_makes_the_jsonb_row_a_complete_recipe(self):
        """``_attach_reproducibility`` already writes source_handles, variable,
        region_name and aggregation_meta into ``payload["export"]``. The frame
        spec lands beside them, so the durable row says how the stack was
        reduced rather than only holding what it produced.

        And it records what export actually ships (D12). The 20,000-cell block
        mean is a rendering resolution, not a scientific product -- at k=8 a
        frame has already lost 16% of its own p98 -- so exporting it would put
        a downsampled field into someone's paper."""
        self.add_bundle()

        _summary, payload = await self.plot()

        spec = payload["export"]["frames"]["spec"]
        self.assertEqual(spec["cadence"], "hourly")
        self.assertEqual(spec["tier"], "cadence")
        self.assertEqual(spec["n_frames"], 3)
        self.assertEqual(spec["buckets_per_frame"], 1)
        self.assertEqual(spec["statistic"], "mean")
        self.assertEqual(spec["boundary"], "pad")
        self.assertEqual(spec["dtype"], "float32")
        # The REALIZED cell count, never D5's ceiling: integer coarsening
        # quantizes, so a real grid lands at 70-96% of what was asked for.
        self.assertEqual(spec["cells_per_frame"], payload["frames"]["cells_per_frame"])
        self.assertLessEqual(spec["cells_per_frame"], spec["target_cells"])

        self.assertEqual(payload["export"]["frames"]["exports"], "period aggregate")
        # Nothing else about export moved, and no frame values leaked into it.
        self.assertEqual(payload["export"]["type"], "heatmap")
        self.assertNotIn("values", payload["export"])

    async def test_the_recipe_records_the_export_and_the_planes_separately(self):
        """Two facts, two keys, and the older one does not move.

        ``spec["statistic"]`` describes what the export IS, and D12 says that is
        the period aggregate and is unchanged by any of this -- so it still says
        ``"mean"`` and must keep saying it. That the scrubber now offers three
        planes is a different fact about the same recipe. Repurposing the
        existing key to carry it would change the meaning of a field already on
        the wire, which is the one thing D15 exists to prevent, and would leave
        every archived row ambiguous about which sense it meant.

        ``statistics`` names what was STORED, so a row reproduces the stack it
        actually holds rather than the one that was requested."""
        self.add_bundle()

        _summary, payload = await self.plot()

        spec = payload["export"]["frames"]["spec"]
        # Unmoved: the export is still the period aggregate of the mean.
        self.assertEqual(spec["statistic"], "mean")
        self.assertEqual(payload["export"]["frames"]["exports"], "period aggregate")
        # New, and beside it rather than instead of it.
        self.assertEqual(spec["statistics"], ["mean", "max", "min"])

    async def test_the_recipe_records_only_the_planes_that_were_stored(self):
        """The same rule the agent summary follows, in the durable row: a spec
        naming a plane the row does not hold would describe a stack nobody can
        reproduce from it."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        with patch.object(frame_stack_module, "MAX_PLANE_NATIVE_CELLS", 8):
            _summary, payload = await self.plot()

        spec = payload["export"]["frames"]["spec"]
        self.assertEqual(spec["statistic"], "mean")
        self.assertEqual(spec["statistics"], ["mean"])
        self.assertEqual(payload["export"]["frames"]["exports"], "period aggregate")

    async def test_the_refusal_reaches_the_recipe_as_well_as_the_render(self):
        """A durable row that simply lacks a frame block cannot say whether
        frames were refused or never attempted. Here the span is a year of
        hourly intervals -- past the backstop, where each of 60 frames would
        average more than a day together."""
        self.add_bundle(times=["2024-06-01T09:30", "2025-06-01T09:30"])

        summary, payload = await self.plot()

        self.assertNotIn("frames", payload)
        self.assertEqual(payload["frames_unavailable"]["reason"], "span_too_long")
        self.assertEqual(
            payload["export"]["frames"]["unavailable"]["reason"], "span_too_long",
        )
        self.assertEqual(summary["frames_unavailable"]["reason"], "span_too_long")

    async def test_a_span_past_the_frame_budget_coarsens_and_quantifies_the_cost(self):
        """D14's second tier, which Phase 3 measured as the COMMON case -- a
        2.2-day TEMPO retrieval at hourly cadence is already 54 frames.

        A coarse frame is a DIFFERENT temporal aggregation from the period map,
        and the D16 delta is what keeps that honest rather than silent: on two
        real TEMPO bundles it measured 5.4% and 22.3%, not the sub-1% D16's own
        illustration implies."""
        import numpy as np

        times = [
            f"2024-06-{1 + hour // 24:02d}T{hour % 24:02d}:30" for hour in range(62)
        ]
        self.add_bundle(times=times, ny=4, nx=4, values=np.full((62, 4, 4), 4.0e15))

        _summary, payload = await self.plot()

        frames = payload["frames"]
        self.assertEqual(frames["tier"], "coarsened")
        self.assertEqual(frames["buckets_per_frame"], 2)
        self.assertLessEqual(len(frames["frames"]), 60)
        # Present and self-describing, so the number does not need JSX to
        # explain it (the reason QA_PASS_RATE_BASIS exists).
        self.assertIn("headline", frames["delta"])
        self.assertIn("basis", frames["delta"])

    async def test_plot_multiple_gets_no_scrubber(self):
        """D7 is a scope decision, not an oversight. A comparison has already
        spent its visual axis on the comparison, and ``plot_multiple``'s panels
        are different regions rather than different times -- a slider across
        them would be scrubbing the wrong axis."""
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_multiple

        self.add_bundle("obs_a")
        self.add_bundle("obs_b")

        emitted = {}
        plot_multiple = make_plot_multiple(self.mcp_tools)
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.setdefault("payload", payload),
        ):
            raw = await plot_multiple.ainvoke(
                {"handles": ["obs_a", "obs_b"], "locations": ["global", "global"]}
            )
        self.assertNotIn("error", json.loads(raw))

        payload = emitted["payload"]
        self.assertEqual(payload["type"], "heatmap_multi")
        self.assertNotIn("frames", payload)
        for panel in payload["panels"]:
            self.assertNotIn("frames", panel)

    def _blob_planes(self, frames):
        """The float32 planes the browser downloads, read back through the
        store -- not the in-process arrays. This is the blob Phase 8 fetched
        from ``GET /chart/{id}/frames.f32.gz``, and the whole point of the
        quantity under test is that it describes THESE."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        blob = frame_store.read_frames(
            frames["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
        )
        return np.frombuffer(
            gzip.decompress(blob.gzipped), dtype="float32",
        ).reshape(frames["shape"])

    def _measured_disagreement(self, frames):
        """``probe_t59_phase8_arrays.py``'s calculation, independently: average
        the frame planes, compare to the period plane, weight by cos(lat).

        Written with ``nanmean`` rather than the reduction's own finite-mask
        arithmetic on purpose -- a reference that shares its implementation
        asserts that one thing agrees with itself.
        """
        import warnings

        import numpy as np

        planes = self._blob_planes(frames)
        period = np.asarray(planes[frames["period_index"]], dtype="float64")
        others = np.asarray(
            np.delete(planes, frames["period_index"], axis=0), dtype="float64",
        )
        with warnings.catch_warnings():
            # An all-absent cell has no mean. That is the fixture working.
            warnings.simplefilter("ignore", RuntimeWarning)
            averaged = np.nanmean(others, axis=0)

        both = np.isfinite(averaged) & np.isfinite(period)
        weights = np.cos(np.deg2rad(np.asarray(frames["lats"]))).reshape(-1, 1)
        numerator = float((np.where(both, np.abs(averaged - period), 0.0) * weights).sum())
        denominator = float((np.where(both, np.abs(period), 0.0) * weights).sum())
        return numerator / denominator

    async def test_the_disclosed_disagreement_is_the_one_the_blob_actually_has(self):
        """The test that would have caught Phase 8's finding, in the regime
        real charts are in.

        Every identity test written before this one runs where the block mean
        is the identity function -- the fixture below it pins ``coarsen_k ==
        [1, 1]`` explicitly. A real regional TEMPO retrieval is 352,181 native
        cells and lands at k=(5,5), 25x over the ceiling, and there the two
        arrays that ship disagree by 1.876%. So: a grid that actually
        coarsens, and an assertion that the number in the payload is the number
        a reader gets by averaging the download.

        Both halves matter. That it AGREES is the contract. That it is
        materially non-zero is what makes the assertion capable of failing --
        a quantity hardcoded to zero would satisfy the first half forever.
        """
        self.add_bundle(ny=200, nx=200, values=self._uneven_coverage(3, 200, 200))

        _summary, payload = await self.plot()
        frames = payload["frames"]

        # 40,000 native cells against D5's 20,000-cell ceiling: the regime.
        self.assertEqual(frames["coarsen_k"], [2, 2])
        self.assertEqual(frames["tier"], "cadence")
        # No temporal disagreement to disclose -- and an array disagreement
        # regardless. That pairing is the entire Phase 8 finding.
        self.assertIsNone(frames["delta"])

        disclosed = frames["frame_grid_delta"]["headline"]
        self.assertAlmostEqual(
            disclosed, self._measured_disagreement(frames), places=9,
        )
        self.assertGreater(
            disclosed, 0.001,
            "the fixture no longer breaks the identity, so this asserts nothing",
        )

    async def test_a_coarsened_chart_discloses_two_numbers_and_they_differ(self):
        """Phase 8 §2's mirror: tier two measured its delta natively and
        published it beside arrays that disagreed by a different amount, with
        neither bounding the other and the smaller one on screen.

        61 hourly buckets past the 60-frame budget, on a grid that also
        coarsens -- both reductions active at once, which is what a real
        multi-day regional retrieval is.
        """
        times = [f"2024-06-01T{hour:02d}:30" for hour in range(24)]
        times += [f"2024-06-02T{hour:02d}:30" for hour in range(24)]
        times += [f"2024-06-03T{hour:02d}:30" for hour in range(13)]
        self.add_bundle(
            ny=200, nx=200, times=times,
            values=self._uneven_coverage(len(times), 200, 200),
        )

        _summary, payload = await self.plot()
        frames = payload["frames"]

        self.assertEqual(frames["tier"], "coarsened")
        self.assertGreater(frames["buckets_per_frame"], 1)
        self.assertEqual(frames["coarsen_k"], [2, 2])

        native = frames["delta"]["headline"]
        shipped = frames["frame_grid_delta"]["headline"]
        self.assertAlmostEqual(
            shipped, self._measured_disagreement(frames), places=9,
        )
        # Two measurements, not one copied twice. Publishing only the native
        # figure told a reader something false about the arrays on screen; a
        # `frame_grid_delta` that merely echoed it would do the same thing more
        # convincingly.
        self.assertNotAlmostEqual(native, shipped, places=6)
        self.assertNotEqual(
            frames["delta"]["basis"], frames["frame_grid_delta"]["basis"],
        )

    async def test_frame_zero_is_the_map_the_chart_already_shows(self):
        """D5's frame 0 and D4's guarantee, where the block mean is a no-op.

        Parking the scrubber at frame 0 must reproduce the Map tab exactly --
        that is what makes "the period map is the average of what you scrub"
        structural rather than hopeful. On a grid under the cell ceiling the
        block mean is a no-op (k=1), so the two arrays are directly comparable
        and any disagreement is arithmetic, not resolution.

        Phase 3 measured the residual at 0.000002% on a real bundle: float32
        storage noise, which is the floor.

        **Kept, and its `coarsen_k == [1, 1]` assertion kept too.** It pins a
        real property in a real regime. What it does not pin is the regime a
        real chart is in, which is why
        ``test_the_disclosed_disagreement_is_the_one_the_blob_actually_has``
        sits above it rather than replacing it."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        rng = np.random.default_rng(59)
        values = rng.normal(4.0e15, 5.0e14, size=(3, 8, 8))
        self.add_bundle(values=values)

        _summary, payload = await self.plot()
        frames = payload["frames"]
        self.assertEqual(frames["coarsen_k"], [1, 1])

        blob = frame_store.read_frames(
            frames["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
        )
        planes = np.frombuffer(
            gzip.decompress(blob.gzipped), dtype="float32",
        ).reshape(frames["shape"])
        period = planes[frames["period_index"]]

        drawn = np.array(
            [[np.nan if v is None else v for v in row] for row in payload["values"]],
            dtype="float64",
        )
        np.testing.assert_allclose(period, drawn, rtol=1e-6)
        # ...and averaging the frames themselves lands in the same place.
        np.testing.assert_allclose(
            np.nanmean(planes[1:], axis=0), period, rtol=1e-6,
        )

    async def test_building_a_scrubber_records_its_own_timing_phase(self):
        """The extra graph walk is real -- Phase 3 measured 9.2 s against a
        12-15 s open+mask -- so a plot that gains a scrubber gains that time.
        Charged to its own phase rather than smeared into "aggregate", for the
        reason "qa_pass_rate" got one: what an honesty feature costs has to be
        visible."""
        events = []
        self.add_bundle()

        with patch(
            "tta_backend.utils.phase_timing.record_phase",
            lambda phase, seconds, **context: events.append(phase),
        ):
            _summary, payload = await self.plot()

        self.assertIn("frames", payload)
        self.assertIn("frames", events)

    async def test_every_plane_urls_bytes_are_that_planes_own(self):
        """The first test that crosses all three seams at once.

        Phases 12, 13 and 14 were each tested in isolation: the reduction
        against its own fixtures, the transport against hand-made stacks, the
        wiring against the block. Between them sits the failure none of those
        can see -- a key wired into the wrong slot, which every check on the way
        out passes, because every plane of a chart has the same shape, dtype,
        grid and pipeline version. Phase 13 added the manifest's ``statistic``
        precisely so that mixup is loud; this is what makes it fire.

        Read through each url's own statistic, exactly as the route resolves it
        (segment -> that plane's key -> ``read_frames`` for that statistic), so
        a max key filed under ``min`` is refused here rather than rendering a
        believable field with the wrong numbers in it."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        # A field whose three statistics are genuinely different numbers, so
        # "served the wrong plane" cannot pass by coincidence.
        values = self._uneven_coverage(3, 8, 8)
        self.add_bundle(values=values)

        _summary, payload = await self.plot()
        frames = payload["frames"]

        served = {}
        for statistic, plane in frames["planes"].items():
            self.assertEqual(
                plane["url"],
                f"/chart/{payload['chart_id']}/frames.{statistic}.f32.gz",
            )
            blob = frame_store.read_frames(
                plane["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
                statistic=statistic,
            )
            self.assertIsNotNone(blob, statistic)
            served[statistic] = np.frombuffer(
                gzip.decompress(blob.gzipped), dtype="float32",
            ).reshape(frames["shape"])

        mean_blob = frame_store.read_frames(
            frames["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
        )
        served["mean"] = np.frombuffer(
            gzip.decompress(mean_blob.gzipped), dtype="float32",
        ).reshape(frames["shape"])

        # Each url serves ITS statistic: max >= mean >= min everywhere all
        # three are finite, and not by equality -- which is what a swapped key
        # would produce.
        both = np.isfinite(served["max"]) & np.isfinite(served["min"])
        self.assertTrue(both.any())
        self.assertTrue(np.all(served["max"][both] >= served["mean"][both]))
        self.assertTrue(np.all(served["mean"][both] >= served["min"][both]))
        self.assertTrue(np.any(served["max"][both] > served["min"][both]))

    async def test_the_whole_payload_still_serializes_for_the_jsonb_row(self):
        """The chart payload is persisted as jsonb and streamed to the browser,
        and a numpy scalar or a float32 is neither JSON- nor jsonb-serializable.
        The axis block coerces on the way out of the store; this is the check
        that nothing added here reintroduces one, because the tests capture
        ``emit_chart``'s argument and would otherwise never serialize it."""
        self.add_bundle()

        _summary, payload = await self.plot()

        round_tripped = json.loads(json.dumps(payload))
        self.assertEqual(
            len(round_tripped["frames"]["frames"]), len(payload["frames"]["frames"]),
        )
        self.assertEqual(round_tripped["export"]["frames"]["spec"], payload["export"]["frames"]["spec"])


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None
        for name in TOOL_MODULES + API_MODULES),
    "satellite tool or API dependencies are not installed",
)
class PlotToPlaneEndpointTests(_PlotHarness, unittest.IsolatedAsyncioTestCase):
    """A chart plotted through the tool, then fetched over HTTP at the url the
    tool minted for it (T59 Phases 12+13+14).

    Phase 13's nine endpoint tests drive the route from hand-made stacks, and
    the class above drives the tool without a route. Neither can see the seam
    between them: whether ``_wire_frames_url`` mints a path this router
    actually resolves, and whether the key it filed under a statistic is the
    one the route hands ``read_frames`` for that statistic. Both halves are one
    edit away from disagreeing silently, because a wrong plane still renders.
    """

    async def asyncSetUp(self):
        await super().asyncSetUp()
        import httpx

        import tta_backend.api as api
        from tta_backend.models.user import User

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()
        self.api.app.state.earthdata_mcp_tools = {}
        self.user = User(
            id="user-1", username="tester", password_hash="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        )
        token, _ = self.api.create_access_token(self.user)
        self.auth_headers = {"Authorization": f"Bearer {token}"}

    async def get(self, payload, url):
        """Fetch ``url`` as the chart's owner, with the real router."""
        async def fake_get_user_by_id(user_id):
            return self.user if user_id == self.user.id else None

        async def fake_is_token_revoked(jti):
            return False

        async def fake_get_chart(chart_id):
            return {**payload, "user_id": self.user.id}

        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch("tta_backend.services.auth_service.get_user_by_id", fake_get_user_by_id), \
             patch("tta_backend.services.auth_service.is_token_revoked", fake_is_token_revoked), \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(
                transport=transport, base_url="http://testserver",
            ) as client:
                return await client.get(url, headers=self.auth_headers)

    async def test_the_url_the_plot_minted_serves_that_planes_bytes(self):
        """End to end, over HTTP, on every statistic the chart carries.

        The bytes are compared against the store's own, keyed by statistic, so
        this fails if the route resolves a path to the wrong plane's entry --
        the exact mixup Phase 13's manifest guard exists to make loud, and the
        one that produces a plausible picture rather than an error."""
        import gzip

        import numpy as np
        from tta_backend.services import frame_store
        from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION

        self.add_bundle(values=self._uneven_coverage(3, 8, 8))

        _summary, payload = await self.plot()
        frames = payload["frames"]

        for statistic, plane in frames["planes"].items():
            response = await self.get(payload, plane["url"])

            self.assertEqual(response.status_code, 200, statistic)
            self.assertEqual(response.headers["content-encoding"], "gzip")
            expected = frame_store.read_frames(
                plane["_key"], pipeline_version=OPEN_PIPELINE_VERSION,
                statistic=statistic,
            )
            # httpx inflates a gzipped body transparently, so ``content`` is
            # already the float32 bytes the canvas would read.
            self.assertEqual(
                response.headers["etag"].strip('"'), expected.etag, statistic,
            )
            served = np.frombuffer(
                response.content, dtype="float32",
            ).reshape(frames["shape"])
            np.testing.assert_array_equal(
                served,
                np.frombuffer(
                    gzip.decompress(expected.gzipped), dtype="float32",
                ).reshape(frames["shape"]),
            )

    async def test_the_means_own_url_is_untouched_by_the_planes_beside_it(self):
        """D6a decision 5, at the only place it can finally be checked whole:
        a real chart that HAS planes still serves its mean at
        ``frames.f32.gz``, with the mean's bytes."""
        import numpy as np

        self.add_bundle(values=self._uneven_coverage(3, 8, 8))

        _summary, payload = await self.plot()
        frames = payload["frames"]
        self.assertEqual(sorted(frames["planes"]), ["max", "min"])

        response = await self.get(payload, frames["url"])

        self.assertEqual(response.status_code, 200)
        served = np.frombuffer(
            response.content, dtype="float32",
        ).reshape(frames["shape"])
        # It is the mean and not a plane: the max plane is strictly greater
        # somewhere, so serving that blob here would be visible.
        max_response = await self.get(payload, frames["planes"]["max"]["url"])
        maxes = np.frombuffer(
            max_response.content, dtype="float32",
        ).reshape(frames["shape"])
        both = np.isfinite(served) & np.isfinite(maxes)
        self.assertTrue(np.any(maxes[both] > served[both]))

    async def test_a_statistic_this_chart_never_built_is_a_404(self):
        """A chart above the plane ceiling keeps a working mean url and 404s
        the toggle, rather than the frontend having to infer which it is from a
        broken render. Every unavailable statistic is one answer -- offer the
        ones that are there."""
        import tta_backend.preprocessing.frame_stack as frame_stack_module

        self.add_bundle()

        with patch.object(frame_stack_module, "MAX_PLANE_NATIVE_CELLS", 8):
            _summary, payload = await self.plot()

        mean = await self.get(payload, payload["frames"]["url"])
        missing = await self.get(
            payload, f"/chart/{payload['chart_id']}/frames.max.f32.gz",
        )

        self.assertEqual(mean.status_code, 200)
        self.assertEqual(missing.status_code, 404)
