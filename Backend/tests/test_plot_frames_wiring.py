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
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

TOOL_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]


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


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in TOOL_MODULES),
    "satellite tool integration dependencies are not installed",
)
class PlotSingularFrameWiringTests(unittest.IsolatedAsyncioTestCase):
    """D7's auto-upgrade, through the real tool.

    ``plot_singular`` only: not ``plot_multiple``, not comparison panels. A
    tool the agent must CHOOSE to call gets called when someone says "animate"
    and not when they say "show me July" -- T56's own lesson was enforce in the
    composite, don't trust the agent.
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

    async def test_frame_zero_is_the_map_the_chart_already_shows(self):
        """D5's frame 0 and D4's guarantee, on the two arrays a user can touch.

        Parking the scrubber at frame 0 must reproduce the Map tab exactly --
        that is what makes "the period map is the average of what you scrub"
        structural rather than hopeful. On a grid under the cell ceiling the
        block mean is a no-op (k=1), so the two arrays are directly comparable
        and any disagreement is arithmetic, not resolution.

        Phase 3 measured the residual at 0.000002% on a real bundle: float32
        storage noise, which is the floor."""
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
