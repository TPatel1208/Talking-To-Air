"""
tests/test_export_event_loop_offload.py
=======================================
The export path used to do all of its heavy work on the event loop.
``build_chart_png`` and ``_export_data_array`` were ``async def`` with
entirely synchronous bodies -- cartopy rendering, ``savefig(dpi=220)``,
full-array ``.values`` materialisation, a dask reduction, and a *synchronous*
Nominatim call containing ``time.sleep`` -- while the module contained zero
``asyncio.to_thread`` calls, against 10 in plot_tools and 4 in stat_tools.

One uvicorn worker means one loop, so while an export was in flight nothing
else progressed: not another user's SSE stream, not the heartbeat that keeps
nginx from timing out, not the ``/health`` check Docker restarts the container
over. Ten concurrent exports of uncached regions compounded through the global
1 rps geocode throttle into tens of seconds of cumulative freeze.

Prior art for the assertion style: ``test_open_handle.py``'s
``OpenHandleEventLoopOffloadTests``. The property asserted is *causal*, not
temporal -- a concurrent ticker's count is sampled from inside the (patched
slow) call itself, so what is measured is what the loop did *while the work
was in flight*, never how long the whole gather took. A wall-clock bound would
measure how busy the host is as much as whether the work was offloaded.
"""
import asyncio
import importlib.util
import os
import sys
import time
import unittest
from unittest.mock import patch


PNG_MAGIC = bytes([0x89]) + b"PNG"

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _profile_payload():
    """A vertical-profile export (T56). Hermetic on purpose: a profile renders
    from its own payload and never re-reads a granule, so this exercises
    ``build_chart_png``'s render-and-save half with no MCP server, no handle
    volume, and no network in the picture."""
    return {
        "type": "profile",
        "title": "ozone_profile vertical profile over New Jersey",
        "export": {
            "type": "profile",
            "variable": "ozone_profile",
            "units": "DU",
            "aggregation": "mean",
            "layers": [0, 1, 2],
            "values": [0.23, 12.0, 30.0],
            "valid_fraction": [0.5, 1.0, 1.0],
            "vertical": {
                "pressure": {
                    "kind": "pressure", "units": "hPa",
                    "values": [0.175, 130.0, 902.0], "spread": [0.0, 56.6, 43.1],
                    "layer_order": "top_down",
                },
            },
            "default_axis": "pressure",
            "layer_order": "top_down",
        },
    }


class _Ticker:
    """A coroutine that counts until it is told to stop, plus the sampling the
    assertion needs.

    ``during(...)`` runs from inside the (slowed) call under test and returns
    how many ticks landed while it was running. Zero means the loop was frozen
    for that whole stretch -- which is exactly what running the work on the
    loop produces.

    Unbounded on purpose. A fixed tick budget looks equivalent and is not: the
    first offloaded call in a process pays a one-time import of the plot_tools
    chain *inside the worker thread*, and a 20-tick ticker is long finished by
    the time the measured call starts. The delta then reads 0 with the loop
    perfectly free -- a false red that says "blocked" about correct code.
    """

    def __init__(self, interval: float = 0.03):
        self.count = 0
        self._interval = interval
        self._stopped = False

    async def run(self):
        while not self._stopped:
            await asyncio.sleep(self._interval)
            self.count += 1

    def stop(self):
        self._stopped = True

    def during(self, seconds: float = 0.6) -> int:
        before = self.count
        time.sleep(seconds)
        return self.count - before

    async def alongside(self, coro):
        """Run ``coro`` to completion with the ticker ticking beside it."""
        async def _run_then_stop():
            try:
                return await coro
            finally:
                self.stop()

        result, _ = await asyncio.gather(_run_then_stop(), self.run())
        return result


@unittest.skipIf(
    importlib.util.find_spec("matplotlib") is None,
    "export offload tests require matplotlib",
)
class ChartPngRenderOffloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_png_render_does_not_freeze_a_concurrent_coroutine(self):
        """``savefig`` is the export's single most expensive synchronous call
        (dpi=220 over a cartopy figure). Slowed here at matplotlib's own
        boundary so the real code path runs, it must not stop the loop."""
        from matplotlib.figure import Figure

        from tta_backend.services.export_service import ExportService

        ticker = _Ticker()
        ticks_during_save = -1
        original_savefig = Figure.savefig

        def slow_savefig(figure, *args, **kwargs):
            nonlocal ticks_during_save
            ticks_during_save = ticker.during()
            return original_savefig(figure, *args, **kwargs)

        with patch.object(Figure, "savefig", slow_savefig):
            png_bytes = await ticker.alongside(
                ExportService().build_chart_png(_profile_payload(), {})
            )

        self.assertGreater(
            ticks_during_save,
            0,
            "the event loop made no progress while savefig was in flight, so "
            "the render ran on the loop instead of being offloaded",
        )
        self.assertTrue(png_bytes.startswith(b"\x89PNG"))


@unittest.skipIf(
    importlib.util.find_spec("xarray") is None,
    "export offload tests require xarray",
)
class ChartCsvOffloadTests(unittest.IsolatedAsyncioTestCase):
    """The CSV download's heavy work is not rendering -- it is
    ``_export_data_array``: resolving the variable, masking by geometry,
    the dask reduction, and the full-array ``.values`` materialisation that
    turns a grid into rows. All of it synchronous, all of it on the loop.
    """

    def _payload(self):
        return {
            "type": "heatmap",
            "title": "no2",
            "export": {
                "variable": "no2",
                "units": "mol/m^2",
                "source_handles": ["obs_offload"],
            },
        }

    def _dataset(self):
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {"no2": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

    async def test_csv_chunks_do_not_freeze_a_concurrent_coroutine(self):
        """Slowed at the aggregation seam -- the dask reduction named in the
        report, and the one call every heatmap CSV passes through. The handle
        open is stubbed rather than slowed because it is already offloaded
        (T16); what is under test is everything *after* it."""
        from unittest.mock import AsyncMock

        from tta_backend.preprocessing.aggregation_service import AggregationService
        from tta_backend.services.export_service import ExportService

        ticker = _Ticker()
        ticks_during_reduction = -1
        original_aggregate = AggregationService.aggregate

        def slow_aggregate(service, *args, **kwargs):
            nonlocal ticks_during_reduction
            if ticks_during_reduction < 0:
                ticks_during_reduction = ticker.during()
            return original_aggregate(service, *args, **kwargs)

        async def collect():
            return [
                chunk
                async for chunk in ExportService().iter_chart_csv_chunks(self._payload(), {})
            ]

        with patch("tta_backend.services.open_handle.open_handle",
                   AsyncMock(return_value=self._dataset())),              patch.object(AggregationService, "aggregate", slow_aggregate):
            chunks = await ticker.alongside(collect())

        self.assertGreater(
            ticks_during_reduction,
            0,
            "the event loop made no progress while the CSV reduction was in "
            "flight, so it ran on the loop instead of being offloaded",
        )
        self.assertIn(b"latitude,longitude", b"".join(chunks))

    async def test_row_materialisation_does_not_freeze_a_concurrent_coroutine(self):
        """The grid-to-rows step is the CSV's other blocking half, and it is
        not covered by the reduction above: turning a full-resolution grid
        into one row per finite cell forces the whole array into memory.

        Nothing is patched at a library boundary here. The granule is
        dask-backed, exactly as a real one is, and its compute is the slow
        part -- so what is measured is the export's own materialisation,
        wherever on the path it happens. Every compute is checked rather than
        the first, because the path materialises more than once (the
        reduction, then the rows) and one offloaded call does not excuse
        another that still runs on the loop.
        """
        from unittest.mock import AsyncMock

        import dask
        import dask.array as dask_array
        import numpy as np
        import xarray as xr

        from tta_backend.services.export_service import ExportService

        ticker = _Ticker()
        ticks_per_compute = []

        def slow_block():
            ticks_per_compute.append(ticker.during(0.3))
            return np.array([[1.0, 2.0], [3.0, 4.0]])

        lazy = dask_array.from_delayed(
            dask.delayed(slow_block)(), shape=(2, 2), dtype=float,
        )
        ds = xr.Dataset(
            {"no2": (("lat", "lon"), lazy)},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        async def collect():
            return [
                row
                async for row in ExportService().iter_chart_csv_rows(self._payload(), {})
            ]

        with patch("tta_backend.services.open_handle.open_handle", AsyncMock(return_value=ds)):
            rows = await ticker.alongside(collect())

        self.assertTrue(ticks_per_compute, "no array materialisation happened at all")
        self.assertGreater(
            min(ticks_per_compute),
            0,
            f"at least one array materialisation froze the event loop "
            f"(ticks per compute: {ticks_per_compute})",
        )
        self.assertEqual(len(rows), 5)  # header + one row per finite cell


    async def test_multi_granule_row_materialisation_does_not_freeze_the_loop(self):
        """The multi-granule CSV is a different code path with the same
        hazard, and a bigger one: it materialises the per-granule cube *and*
        the reduced mean, then writes a column per granule. Covered
        separately because the single-granule path being offloaded says
        nothing about this one."""
        from unittest.mock import AsyncMock

        import dask
        import dask.array as dask_array
        import numpy as np
        import xarray as xr

        from tta_backend.services.export_service import ExportService

        ticker = _Ticker()
        ticks_per_compute = []

        def slow_block():
            ticks_per_compute.append(ticker.during(0.3))
            return np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        lazy = dask_array.from_delayed(
            dask.delayed(slow_block)(), shape=(2, 2, 2), dtype=float,
        )
        ds = xr.Dataset(
            {"no2": (("time", "lat", "lon"), lazy)},
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                "lat": [10.0, 20.0],
                "lon": [30.0, 40.0],
            },
        )

        payload = self._payload()
        payload["export"]["aggregation_meta"] = {"n_granules": 2, "stat": "mean"}

        async def collect():
            return [row async for row in ExportService().iter_chart_csv_rows(payload, {})]

        with patch("tta_backend.services.open_handle.open_handle", AsyncMock(return_value=ds)):
            rows = await ticker.alongside(collect())

        self.assertTrue(ticks_per_compute, "no array materialisation happened at all")
        self.assertGreater(
            min(ticks_per_compute),
            0,
            f"at least one array materialisation froze the event loop "
            f"(ticks per compute: {ticks_per_compute})",
        )
        self.assertIn("mean", rows[0])

    async def test_timeseries_row_building_does_not_freeze_the_loop(self):
        """The time-series CSV reduces every timestep to one number, reading
        each slice in full to do it. Fewer rows out, the same grid in."""
        from unittest.mock import AsyncMock

        import dask
        import dask.array as dask_array
        import numpy as np
        import xarray as xr

        from tta_backend.services.export_service import ExportService

        ticker = _Ticker()
        ticks_per_compute = []

        def slow_block():
            ticks_per_compute.append(ticker.during(0.3))
            return np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        lazy = dask_array.from_delayed(
            dask.delayed(slow_block)(), shape=(2, 2, 2), dtype=float,
        )
        ds = xr.Dataset(
            {"no2": (("time", "lat", "lon"), lazy)},
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                "lat": [10.0, 20.0],
                "lon": [30.0, 40.0],
            },
        )

        payload = self._payload()
        payload["export"]["type"] = "timeseries"
        payload["export"]["aggregation"] = "mean"

        async def collect():
            return [row async for row in ExportService().iter_chart_csv_rows(payload, {})]

        with patch("tta_backend.services.open_handle.open_handle", AsyncMock(return_value=ds)):
            rows = await ticker.alongside(collect())

        self.assertTrue(ticks_per_compute, "no array materialisation happened at all")
        self.assertGreater(
            min(ticks_per_compute),
            0,
            f"at least one array materialisation froze the event loop "
            f"(ticks per compute: {ticks_per_compute})",
        )
        self.assertEqual(len(rows), 3)  # header + one row per timestep



@unittest.skipIf(
    importlib.util.find_spec("xarray") is None,
    "export offload tests require xarray",
)
class ExportRegionLookupOffloadTests(unittest.IsolatedAsyncioTestCase):
    """The export path's region lookup used to call the *synchronous*
    geocoder, whose 1 rps throttle is a bare ``time.sleep`` and whose HTTP
    call is a blocking ``requests.get(timeout=15)``. The geocoder's own
    comment admitted it: "this sync path still runs on the event loop in a few
    legacy callers (export_service)".

    That is the compounding half of the incident. The throttle is global, so
    concurrent exports of uncached regions do not merely block one at a time
    -- each one's wait is stacked on the last, and ten of them serialise into
    tens of seconds during which the single worker serves nobody.
    """

    def _payload(self):
        return {
            "type": "heatmap",
            "title": "no2 over Newark",
            "export": {
                "variable": "no2",
                "units": "mol/m^2",
                "region_name": "T-P0 Export Offload Locale",
                "source_handles": ["obs_region"],
            },
        }

    def _dataset(self):
        """Gridded over the geocoded hit below, so the region mask leaves data
        behind and the render reaches ``savefig`` rather than failing early on
        an empty array."""
        import numpy as np
        import xarray as xr

        return xr.Dataset(
            {"no2": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
            coords={"lat": [40.65, 40.75], "lon": [-74.25, -74.15]},
        )

    async def test_an_uncached_region_lookup_does_not_freeze_the_loop(self):
        """Both twins are slowed identically at the geocoder seam, so this
        cannot be passed by taking a shortcut around the lookup -- only by
        awaiting a lookup that yields the loop while it waits."""
        from unittest.mock import AsyncMock, Mock, patch as patch_

        from tta_backend.services.export_service import ExportService
        from tta_backend.utils.plotting import get_geocoding_service

        hit = {
            "latitude": 40.7,
            "longitude": -74.2,
            "display_name": "T-P0 Export Offload Locale",
            "polygon": None,
            "bbox": [40.6, 40.8, -74.3, -74.1],
        }

        ticker = _Ticker()
        service = get_geocoding_service()
        service.cache.clear()
        self.addCleanup(service.cache.clear)

        ticks_during_lookup = -1

        def slow_geocode(location_name):
            nonlocal ticks_during_lookup
            ticks_during_lookup = ticker.during()
            return hit

        async def slow_ageocode(location_name):
            nonlocal ticks_during_lookup
            before = ticker.count
            await asyncio.sleep(0.6)
            ticks_during_lookup = ticker.count - before
            return hit

        sync_geocode = Mock(side_effect=slow_geocode)
        with (
            patch_("tta_backend.services.open_handle.open_handle",
                   AsyncMock(return_value=self._dataset())),
            patch_.object(service, "geocode", sync_geocode),
            patch_.object(service, "ageocode", AsyncMock(side_effect=slow_ageocode)),
        ):
            png_bytes = await ticker.alongside(
                ExportService().build_chart_png(self._payload(), {})
            )

        self.assertGreater(
            ticks_during_lookup,
            0,
            "the event loop made no progress while the region was being "
            "geocoded, so the export took the blocking sync geocoder",
        )
        self.assertEqual(
            sync_geocode.call_count,
            0,
            "the export called the synchronous geocoder, whose throttle sleeps "
            "on whatever thread reaches it",
        )
        self.assertTrue(png_bytes.startswith(PNG_MAGIC))


@unittest.skipIf(
    importlib.util.find_spec("matplotlib") is None,
    "export offload tests require matplotlib",
)
class RenderUsesNoGlobalFigureRegistryTests(unittest.IsolatedAsyncioTestCase):
    """Moving the render onto a worker thread is only safe if the render owns
    its figure privately.

    ``pyplot`` does not work that way: ``plt.subplots`` files every figure in
    a process-global registry that ``plt.close`` later has to remove it from,
    and neither operation is synchronised. Two exports rendering on two
    threads share that dict. The object-oriented API -- ``Figure`` plus
    ``FigureCanvasAgg`` -- has no registry to share, which is what makes
    per-figure rendering thread-safe rather than merely usually-fine.

    Asserted on the registry's contents rather than by racing threads and
    hoping: a race that reproduces once in fifty runs is not a test.
    """

    async def test_an_in_flight_render_registers_no_global_figure(self):
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        from tta_backend.services.export_service import ExportService

        plt.close("all")
        self.addCleanup(plt.close, "all")

        fignums_during_render = None
        original_savefig = Figure.savefig

        def observing_savefig(figure, *args, **kwargs):
            nonlocal fignums_during_render
            fignums_during_render = plt.get_fignums()
            return original_savefig(figure, *args, **kwargs)

        with patch.object(Figure, "savefig", observing_savefig):
            await ExportService().build_chart_png(_profile_payload(), {})

        self.assertEqual(
            fignums_during_render,
            [],
            "the export's figure was filed in pyplot's process-global "
            "registry, which concurrent renders on worker threads share",
        )

    async def test_a_failing_render_leaves_no_figure_behind(self):
        """The registry's other cost. ``plt.close(fig)`` after ``savefig``
        only runs when ``savefig`` returns, so a render that raises -- an
        empty masked array, a cartopy projection error -- leaves its figure
        registered for the life of the process, and every failed download
        adds another. With no registry there is nothing to leak."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        from tta_backend.services.export_service import ExportService

        plt.close("all")
        self.addCleanup(plt.close, "all")

        def exploding_savefig(figure, *args, **kwargs):
            raise RuntimeError("render blew up")

        with patch.object(Figure, "savefig", exploding_savefig):
            with self.assertRaises(RuntimeError):
                await ExportService().build_chart_png(_profile_payload(), {})

        self.assertEqual(
            plt.get_fignums(),
            [],
            "a failed export left its figure in pyplot's global registry, so "
            "every failed download leaks one for the life of the process",
        )


@unittest.skipIf(
    importlib.util.find_spec("xarray") is None,
    "export offload tests require xarray",
)
class EveryExportTypeStillRendersTests(unittest.IsolatedAsyncioTestCase):
    """The offload rewrote ``build_chart_png`` into a prefetch phase and a
    render phase, and rewrote the render onto a different matplotlib API. Four
    export types go through it and they diverge before they converge: only
    ``heatmap`` reaches cartopy, only ``heatmap_multi`` builds a shared
    colorbar across axes, only ``timeseries`` plots against dates, only
    ``profile`` renders without reading a granule at all.

    ``profile`` and the point-sample time series are covered elsewhere
    (test_vertical_profile, test_export_service); these are the paths that had
    no PNG coverage of their own.
    """

    def _dataset(self, with_time: bool = False):
        import numpy as np
        import xarray as xr

        if with_time:
            return xr.Dataset(
                {"no2": (("time", "lat", "lon"),
                         np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))},
                coords={
                    "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                    "lat": [10.0, 20.0],
                    "lon": [30.0, 40.0],
                },
            )
        return xr.Dataset(
            {"no2": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

    async def _render(self, payload, with_time: bool = False):
        from unittest.mock import AsyncMock

        from tta_backend.services.export_service import ExportService

        with patch("tta_backend.services.open_handle.open_handle",
                   AsyncMock(return_value=self._dataset(with_time))):
            return await ExportService().build_chart_png(payload, {})

    async def test_a_heatmap_renders_through_cartopy(self):
        png_bytes = await self._render({
            "type": "heatmap",
            "title": "no2",
            "export": {"variable": "no2", "units": "mol/m^2", "source_handles": ["obs"]},
        })

        self.assertTrue(png_bytes.startswith(PNG_MAGIC))

    async def test_a_comparison_renders_every_panel_under_one_colorbar(self):
        """The multi-panel path is the one that shares a colorbar across axes
        -- ``fig.colorbar(mesh, ax=axes.ravel().tolist())`` -- which is the
        call most likely to break moving off pyplot, since pyplot's version
        resolves the current figure implicitly and the OO one does not."""
        panel = {"variable": "no2", "units": "mol/m^2", "source_handles": ["obs"]}
        png_bytes = await self._render({
            "type": "heatmap_multi",
            "title": "no2 compared",
            "export": {
                "type": "heatmap_multi",
                "units": "mol/m^2",
                "panels": [dict(panel, region_name=None), dict(panel, region_name=None)],
            },
        })

        self.assertTrue(png_bytes.startswith(PNG_MAGIC))

    async def test_a_comparison_with_no_panels_still_refuses_clearly(self):
        from tta_backend.services.export_service import ExportService

        with self.assertRaises(ValueError):
            await ExportService().build_chart_png(
                {"export": {"type": "heatmap_multi", "panels": []}}, {},
            )

    async def test_a_gridded_timeseries_renders(self):
        png_bytes = await self._render({
            "type": "timeseries",
            "title": "no2 over time",
            "export": {
                "type": "timeseries",
                "variable": "no2",
                "units": "mol/m^2",
                "aggregation": "mean",
                "source_handles": ["obs"],
            },
        }, with_time=True)

        self.assertTrue(png_bytes.startswith(PNG_MAGIC))

    async def test_an_export_with_no_metadata_refuses_before_rendering(self):
        from tta_backend.services.export_service import ExportService

        with self.assertRaises(ValueError):
            await ExportService().build_chart_png({"type": "heatmap"}, {})


@unittest.skipIf(
    importlib.util.find_spec("xarray") is None,
    "export offload tests require xarray",
)
class OffloadedWorkKeepsTheRequestsUserTests(unittest.IsolatedAsyncioTestCase):
    """``current_user_id()`` is a ``ContextVar``, and the export endpoints bind
    it around the download because the workspace-bound MCP tools the export
    reaches read it to decide *whose* data to open.

    Moving work onto a worker thread is exactly the kind of change that
    silently drops it: a plain ``Thread`` starts with an empty context, and an
    export that loses its user does not fail loudly -- it opens the wrong
    workspace or none. ``asyncio.to_thread`` copies the calling context, which
    is why it is the right primitive here and a hand-rolled executor call
    would not be; this pins that, so a later "optimisation" to a shared
    executor cannot quietly take it away.
    """

    async def test_the_offloaded_narrowing_still_sees_the_requests_user(self):
        from unittest.mock import AsyncMock

        import numpy as np
        import xarray as xr

        from tta_backend.services.export_service import ExportService
        from tta_backend.utils.streaming import current_user_id, user_id_context

        seen = []
        service = ExportService()
        original_narrow = ExportService._narrow_data_array

        def recording_narrow(self_, *args, **kwargs):
            seen.append(current_user_id())
            return original_narrow(self_, *args, **kwargs)

        ds = xr.Dataset(
            {"no2": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )
        export = {"variable": "no2", "units": "mol/m^2", "source_handles": ["obs"]}

        with patch("tta_backend.services.open_handle.open_handle", AsyncMock(return_value=ds)),              patch.object(ExportService, "_narrow_data_array", recording_narrow):
            with user_id_context("user-p0-export"):
                await service._export_data_array(export, {}, collapse_to_2d=True)

        self.assertEqual(seen, ["user-p0-export"])


class NoProductionModuleImportsPyplotTests(unittest.TestCase):
    """The structural half of the two classes above, and the replacement for a
    metric that used to watch this from production.

    ``RenderUsesNoGlobalFigureRegistryTests`` proves the *current* export path
    registers no figure. It cannot prove the next one won't: a future
    ``plt.subplots`` in a module those tests do not drive would reintroduce the
    shared registry unnoticed. T45 covered that gap with a
    ``matplotlib_open_figures`` gauge reading ``plt.get_fignums()`` on every
    /metrics scrape, but once the render moved to ``Figure`` +
    ``FigureCanvasAgg`` the gauge was pinned at 0 forever, and importing
    ``pyplot`` to read it left ``utils/metrics.py`` as the only thing in the
    process still loading the registry the render path had just been moved off.

    A source scan gets the canary that gauge was worth keeping for without any
    of that: it fails at merge time rather than on a dashboard, it covers every
    module in the package rather than only the ones some test happens to
    exercise, and it costs the running backend nothing.

    A source scan and not a ``sys.modules`` check on purpose -- several tests in
    this suite import ``pyplot`` themselves, so in a full-suite run that
    assertion would answer according to test ordering rather than according to
    what the backend imports.
    """

    def test_no_module_under_tta_backend_imports_pyplot(self):
        import pathlib
        import re

        package = pathlib.Path(__file__).resolve().parent.parent / "tta_backend"
        self.assertTrue(package.is_dir(), f"{package} is not a directory")

        # Anchored at the start of a line, so that the several *comments* in
        # export_service and plotting explaining why this code avoids pyplot
        # do not themselves trip it.
        pyplot_import = re.compile(
            r"^\s*(?:import\s+matplotlib\.pyplot"
            r"|from\s+matplotlib\.pyplot\s+import"
            r"|from\s+matplotlib\s+import\s+[\w\s,]*\bpyplot\b)"
        )

        offenders = []
        for module in sorted(package.rglob("*.py")):
            source = module.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), start=1):
                if pyplot_import.match(line):
                    rel = module.relative_to(package.parent)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

        self.assertEqual(
            offenders,
            [],
            "pyplot was reintroduced into the backend. Every figure it creates "
            "is filed in a process-global registry that concurrent renders on "
            "worker threads share, and that a render raising before its close "
            "leaks into for the life of the process. Build figures with "
            "Figure + FigureCanvasAgg instead -- see _new_figure in "
            "services/export_service.py. Offending imports: " + "; ".join(offenders),
        )

    def test_the_scan_would_actually_catch_a_reintroduced_import(self):
        """The scan above passes when it finds nothing, which is also what it
        does if the pattern is wrong or the directory is empty. This drives the
        same matcher over the import spellings a reintroduction would realistically
        use, so a silently-broken regex fails here instead of going green forever.
        """
        import re

        pyplot_import = re.compile(
            r"^\s*(?:import\s+matplotlib\.pyplot"
            r"|from\s+matplotlib\.pyplot\s+import"
            r"|from\s+matplotlib\s+import\s+[\w\s,]*\bpyplot\b)"
        )

        for spelling in (
            "import matplotlib.pyplot",
            "import matplotlib.pyplot as plt",
            "    import matplotlib.pyplot as plt",
            "from matplotlib.pyplot import subplots",
            "from matplotlib import pyplot",
            "from matplotlib import pyplot as plt",
            "from matplotlib import cm, pyplot",
        ):
            with self.subTest(spelling=spelling):
                self.assertIsNotNone(pyplot_import.match(spelling))

        for allowed in (
            "from matplotlib.figure import Figure",
            "from matplotlib.backends.backend_agg import FigureCanvasAgg",
            "from matplotlib.colors import ListedColormap",
            "import matplotlib.image as mpimg",
            "import matplotlib as mpl",
            "# The object-oriented API, never pyplot. ``plt.subplots`` files",
        ):
            with self.subTest(allowed=allowed):
                self.assertIsNone(pyplot_import.match(allowed))


if __name__ == "__main__":
    unittest.main()
