"""T51: per-phase wall-clock timing for the retrieval/plot pipeline.

Before this, the only durations recorded anywhere were per-HTTP-request and
whole-turn elapsed, so "that turn felt slow" could never be attributed to a
phase. ``phase_timer`` records a phase's own span to a Prometheus histogram
and a structured ``phase_timing`` log event.

It wraps the hot path, so the governing constraint is that it must never be
the reason a turn fails: every assertion about a broken backend below is
about degrading to a no-op rather than raising.
"""
import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _phase_samples(phase: str) -> tuple[float, float]:
    """(count, sum) currently recorded for ``phase``. Histograms are
    process-wide state, so every test below asserts on a delta."""
    from utils.metrics import PIPELINE_PHASE_DURATION_SECONDS

    count = total = 0.0
    for metric in PIPELINE_PHASE_DURATION_SECONDS.collect():
        for sample in metric.samples:
            if sample.labels.get("phase") != phase:
                continue
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return count, total


class PhaseTimerTests(unittest.TestCase):
    def test_a_phase_of_known_duration_is_recorded_to_the_histogram(self):
        import time

        from utils.phase_timing import phase_timer

        before_count, before_sum = _phase_samples("crop")
        with phase_timer("crop"):
            time.sleep(0.05)
        after_count, after_sum = _phase_samples("crop")

        self.assertEqual(after_count - before_count, 1)
        # Generous upper bound: this asserts the recorded value is the block's
        # own duration, not that the machine is fast.
        self.assertGreaterEqual(after_sum - before_sum, 0.04)
        self.assertLess(after_sum - before_sum, 5.0)

    def test_nested_phases_each_record_their_own_span(self):
        """``open`` contains ``extract``; the outer phase must not be charged
        only for the part outside the inner one, nor the inner for the whole."""
        import time

        from utils.phase_timing import phase_timer

        outer_before = _phase_samples("open")
        inner_before = _phase_samples("extract")
        with phase_timer("open"):
            time.sleep(0.02)
            with phase_timer("extract"):
                time.sleep(0.05)
        outer_after = _phase_samples("open")
        inner_after = _phase_samples("extract")

        self.assertEqual(outer_after[0] - outer_before[0], 1)
        self.assertEqual(inner_after[0] - inner_before[0], 1)
        outer_seconds = outer_after[1] - outer_before[1]
        inner_seconds = inner_after[1] - inner_before[1]
        self.assertGreaterEqual(inner_seconds, 0.04)
        self.assertGreaterEqual(outer_seconds, inner_seconds + 0.015)

    def test_a_phase_that_raises_still_records_and_re_raises(self):
        """A slow phase that then failed is exactly what a "why was that turn
        slow" investigation needs; swallowing either the timing or the
        exception would lose it."""
        from utils.phase_timing import phase_timer

        before = _phase_samples("mask")
        with self.assertRaises(ValueError):
            with phase_timer("mask"):
                raise ValueError("boom")
        after = _phase_samples("mask")

        self.assertEqual(after[0] - before[0], 1)

    def test_context_recorded_inside_the_block_reaches_the_log_event(self):
        from utils.phase_timing import phase_timer

        with self.assertLogs("utils.phase_timing", level="INFO") as captured:
            with phase_timer("crop", cells_in=1000) as ctx:
                ctx["cells_out"] = 40

        record = captured.records[-1]
        self.assertEqual(record._event, "phase_timing")
        self.assertEqual(record._phase, "crop")
        self.assertEqual(record._cells_in, 1000)
        self.assertEqual(record._cells_out, 40)
        self.assertIsInstance(record._duration_seconds, float)
        self.assertIsInstance(record._thread_id, int)

    def test_a_broken_metrics_backend_degrades_to_a_no_op(self):
        """This wraps the hot path: a telemetry failure must never be the
        reason a turn fails."""
        from unittest.mock import patch

        from utils.phase_timing import phase_timer

        with patch("utils.metrics.observe_phase_duration", side_effect=RuntimeError("registry down")):
            with phase_timer("aggregate"):
                pass  # must not raise

    def test_an_unserializable_context_value_does_not_break_the_phase(self):
        from utils.phase_timing import phase_timer

        class Hostile:
            def __repr__(self):
                raise RuntimeError("no repr for you")

        with phase_timer("render", detail=Hostile()):
            pass  # must not raise


class PhaseTimerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_phase_spanning_a_thread_boundary_records_wall_clock(self):
        """The pipeline crosses ``asyncio.to_thread``; a CPU-time clock would
        under-report a phase that spent its span blocked on I/O in a worker
        thread."""
        import asyncio
        import time

        from utils.phase_timing import phase_timer

        before = _phase_samples("open")
        async with phase_timer("open"):
            await asyncio.to_thread(time.sleep, 0.05)
        after = _phase_samples("open")

        self.assertEqual(after[0] - before[0], 1)
        self.assertGreaterEqual(after[1] - before[1], 0.04)


class PhaseLabelsetTests(unittest.TestCase):
    def test_every_pipeline_phase_is_pre_declared_at_zero(self):
        """Prometheus pulls: a phase that has not run yet must still exist as
        a series, or a dashboard reads "no data" and cannot tell a phase that
        was never exercised from one that was removed."""
        from utils.metrics import PIPELINE_PHASES, render_prometheus_metrics

        rendered = render_prometheus_metrics().decode()

        self.assertIn("pipeline_phase_duration_seconds", rendered)
        for phase in PIPELINE_PHASES:
            self.assertIn(
                f'pipeline_phase_duration_seconds_count{{phase="{phase}"}}',
                rendered,
                f"phase '{phase}' has no pre-declared series",
            )

    def test_the_phase_vocabulary_covers_the_instrumented_pipeline(self):
        from utils.metrics import PIPELINE_PHASES

        self.assertEqual(
            set(PIPELINE_PHASES),
            {"export", "extract", "open", "crop", "mask", "aggregate", "render"},
        )


class InstrumentedPipelineTests(unittest.TestCase):
    """The timer is only worth having if the phases are actually wired. These
    exercise each phase through its public entry point -- the scaffolding this
    PRD replaces was dead precisely because nothing asserted it had a caller."""

    def _small_grid(self):
        import numpy as np
        import xarray as xr

        lats = np.arange(20.0, 55.01, 0.5)
        lons = np.arange(-125.0, -60.0 + 0.01, 0.5)
        values = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
        return xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={
                "lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"}),
            },
            name="no2",
        )

    def test_masking_records_a_crop_and_a_mask_phase(self):
        from shapely.geometry import box

        from utils.plotting import mask_data_by_geometry

        crop_before = _phase_samples("crop")
        mask_before = _phase_samples("mask")
        mask_data_by_geometry(self._small_grid(), box(-74.9, 39.1, -73.1, 40.9))

        self.assertEqual(_phase_samples("crop")[0] - crop_before[0], 1)
        self.assertEqual(_phase_samples("mask")[0] - mask_before[0], 1)

    def test_the_crop_phase_logs_the_cells_it_eliminated(self):
        """Duration alone can't separate an I/O-bound phase from a CPU-bound
        one; cells in/out is the cheap size context that can (story 3)."""
        from shapely.geometry import box

        from utils.plotting import mask_data_by_geometry

        with self.assertLogs("utils.phase_timing", level="INFO") as captured:
            mask_data_by_geometry(self._small_grid(), box(-74.9, 39.1, -73.1, 40.9))

        crop = next(r for r in captured.records if r._phase == "crop")
        self.assertGreater(crop._cells_in, crop._cells_out)

    def test_an_uncropped_mask_still_records_the_mask_phase(self):
        """``crop=False`` is the benchmark's case-1 baseline: it must still be
        measurable, or the A/B has nothing to compare."""
        from shapely.geometry import box

        from utils.plotting import mask_data_by_geometry

        before = _phase_samples("mask")
        mask_data_by_geometry(self._small_grid(), box(-74.9, 39.1, -73.1, 40.9), crop=False)

        self.assertEqual(_phase_samples("mask")[0] - before[0], 1)

    def test_building_a_chart_payload_records_a_render_phase(self):
        """The overlay PNG render and grid downsampling are pure CPU on the
        plot path; without this phase they hide inside "whole-turn elapsed"."""
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        before = _phase_samples("render")
        _da_to_heatmap_payload(self._small_grid(), "NO2 over NJ", "no2", "mol/m^2")

        self.assertEqual(_phase_samples("render")[0] - before[0], 1)

    def test_aggregation_records_an_aggregate_phase(self):
        from preprocessing.aggregation_service import AggregationService

        before = _phase_samples("aggregate")
        AggregationService().aggregate(self._small_grid().to_dataset(), variable="no2", stat="mean")

        self.assertEqual(_phase_samples("aggregate")[0] - before[0], 1)


class MetricsEndpointExposesPhaseHistogramTests(unittest.IsolatedAsyncioTestCase):
    """Story 4: an operator sees a regression as a shifted distribution rather
    than an anecdote -- which requires the series to be on ``/metrics``, not
    merely in the process's registry."""

    async def asyncSetUp(self):
        if any(importlib.util.find_spec(name) is None for name in ("fastapi", "httpx")):
            self.skipTest("metrics endpoint test dependencies are not installed")

    async def test_the_metrics_endpoint_serves_every_phase_series(self):
        import httpx

        import api
        from utils.metrics import PIPELINE_PHASES

        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        for phase in PIPELINE_PHASES:
            self.assertIn(f'pipeline_phase_duration_seconds_count{{phase="{phase}"}}', response.text)


@unittest.skipIf(
    any(
        importlib.util.find_spec(name) is None
        for name in ("langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray")
    ),
    "open_handle test dependencies are not installed",
)
class InstrumentedOpenHandleTests(unittest.IsolatedAsyncioTestCase):
    """``export`` (MCP round-trip), ``extract`` (bundle unzip) and ``open``
    (per-member opens + concat) are the three phases a slow retrieval turn
    could be hiding in, and no timing separated them before T51."""

    async def asyncSetUp(self):
        import tempfile

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools
        from fake_earthdata_mcp import FakeEarthdataMCPServer, HandleVolume, build_fake_mcp

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(
            build_fake_mcp(
                {
                    "export_result": self.volume.export_result,
                    "rematerialize": self.volume.rematerialize,
                    "get_retrieval_status": self.volume.get_retrieval_status,
                }
            )
        )
        server.start()
        self.addCleanup(server.stop)
        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None))

    def _add_bundle(self, handle: str, members: int = 2):
        import numpy as np
        import xarray as xr

        def member(day: int):
            def make():
                return xr.Dataset(
                    {"no2": (("time", "lat", "lon"), np.ones((1, 4, 4), dtype="float32"))},
                    coords={
                        "time": [np.datetime64(f"2024-07-{day:02d}T00:00:00")],
                        "lat": np.arange(4.0),
                        "lon": np.arange(4.0),
                    },
                )

            return make

        self.volume.add_netcdf_bundle(
            handle,
            {f"granule_{day:02d}.nc": {None: member(day)} for day in range(1, members + 1)},
        )

    async def test_opening_a_bundle_handle_records_export_extract_and_open(self):
        from services.open_handle import open_handle

        self._add_bundle("obs_phases")
        before = {phase: _phase_samples(phase) for phase in ("export", "extract", "open")}

        await open_handle("obs_phases", self.tools)

        for phase, (count, _) in before.items():
            self.assertEqual(_phase_samples(phase)[0] - count, 1, f"phase '{phase}' was not recorded")

    async def test_the_extract_phase_distinguishes_a_cache_hit_from_a_miss(self):
        """The extract cache is keyed by bundle identity, so a repeat open
        skips the unzip entirely. Without the hit/miss flag the two land in
        one distribution and the cache's effect is invisible."""
        from services.open_handle import open_handle

        self._add_bundle("obs_cached")

        with self.assertLogs("utils.phase_timing", level="INFO") as first:
            await open_handle("obs_cached", self.tools)
        with self.assertLogs("utils.phase_timing", level="INFO") as second:
            await open_handle("obs_cached", self.tools)

        miss = next(r for r in first.records if r._phase == "extract")
        hit = next(r for r in second.records if r._phase == "extract")
        self.assertFalse(miss._cached)
        self.assertTrue(hit._cached)

    async def test_the_open_phase_reports_how_many_members_it_opened(self):
        """Per-member open cost is the thing a many-granule bundle pays; a
        duration with no member count can't be compared across bundles."""
        from services.open_handle import open_handle

        self._add_bundle("obs_members", members=3)

        with self.assertLogs("utils.phase_timing", level="INFO") as captured:
            await open_handle("obs_members", self.tools)

        opened = next(r for r in captured.records if r._phase == "open")
        self.assertEqual(opened._members, 3)


if __name__ == "__main__":
    unittest.main()
