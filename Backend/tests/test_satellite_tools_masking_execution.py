"""
tests/test_satellite_tools_masking_execution.py
==================================================
T25 masking-execution fix: the honesty-guard commit (321d507) proved that
before this fix, every real tool path opened a Dataset, extracted just the
science DataArray, and lost the sibling QA-flag variable before it ever
reached AggregationService.aggregate() -- so no tool actually ran QA masking
despite collections.yaml pinning qa_good_values for TEMPO_NO2/TEMPO_HCHO/etc.
The existing test_aggregation_service.py unit tests hid this gap by passing
a full Dataset straight to aggregate(), a shape no production tool call
ever takes.

These are integration tests at the tool layer, mirroring production shape:
open a real Dataset (science var + sibling QA-flag var) through the same
HandleVolume/open_handle seam every tool uses, call the actual plot/stat/
compare tool, and assert (a) bad-quality-flagged pixels are actually
dropped from the computed result, not just from provenance metadata, and
(b) the reported qa_status truthfully says "verified" (a pinned collections.
yaml rule) rather than the honesty guard's "not applied" downgrade.

The registry match is driven by the opened Dataset's global ``short_name``
attribute (datasets/mask_info.py::col_info_for_short_name) -- "TEMPO_NO2_L3"
matches collections.yaml's TEMPO_NO2 entry (quality_flag_var=
main_data_quality_flag, qa_good_values=[0]).
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]


def _tempo_no2_dataset(xr, values, flags, lat=(10.0, 20.0), lon=(30.0, 40.0), time=None):
    """A TEMPO_NO2-shaped Dataset: science var + sibling QA-flag var, with
    the ``short_name`` global attribute col_info_for_short_name matches
    against collections.yaml's TEMPO_NO2 entry."""
    if time is None:
        data_vars = {
            "vertical_column_troposphere": (("lat", "lon"), values, {"units": "molecules/cm^2"}),
            "main_data_quality_flag": (("lat", "lon"), flags),
        }
        coords = {"lat": list(lat), "lon": list(lon)}
    else:
        import numpy as np

        data_vars = {
            "vertical_column_troposphere": (("time", "lat", "lon"), values, {"units": "molecules/cm^2"}),
            "main_data_quality_flag": (("time", "lat", "lon"), flags),
        }
        coords = {"time": np.array(list(time), dtype="datetime64[ns]"), "lat": list(lat), "lon": list(lon)}
    return xr.Dataset(data_vars, coords=coords, attrs={"short_name": "TEMPO_NO2_L3"})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "masking-execution integration test dependencies are not installed",
)
class MaskingExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)
        self._align_handler = None

        async def _align(source_handles, method="outer", workspace_id="default"):
            return await self._align_handler(source_handles)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
            "align": _align,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def test_plot_singular_drops_bad_flag_pixels_and_reports_verified_qa(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        plot_singular = make_plot_singular(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        masking = full["provenance"]["masking"]
        self.assertEqual(masking["qa_status"], "verified")
        self.assertEqual(masking["qa_source"], "collections_yaml")

        # Bad-flag pixels (lon=40.0 column, flag=1) are actually dropped from
        # the rendered grid -- not just disclosed in provenance.
        flat_values = [v for row in full["values"] for v in row if v is not None]
        self.assertTrue(all(v in (1.0, 3.0) for v in flat_values), flat_values)

    async def test_compute_statistic_tool_excludes_bad_flag_pixels_from_the_mean(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_1", "location": "global", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        # Good cells (flag=0): 1.0 at lat=10 and 3.0 at lat=20. The regional
        # mean is cos(latitude) area-weighted (area_weighted_mean) — an
        # unweighted mean (2.0 here) would over-weight the higher latitude.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected_mean = (1.0 * w10 + 3.0 * w20) / (w10 + w20)
        self.assertAlmostEqual(result["mean"], expected_mean)
        self.assertEqual(result["n_pixels"], 2)
        self.assertEqual(result["aggregation_meta"]["masking"]["qa_status"], "verified")

    async def test_compute_statistic_result_discloses_region_type_and_display_name(self):
        """T42: the stats result names *what kind* of region was masked and the
        display_name it resolved, so "mean over the US" is checkable against
        what was actually computed. 'global' is a preset bounding box."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 0], [0, 0]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)
        raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_1", "location": "global", "stats": ["mean"],
        })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertEqual(result["region_type"], "bounding_box")
        self.assertEqual(result["display_name"], "Global")

    async def test_plot_singular_provenance_discloses_region_type_and_display_name(self):
        """T42: heatmap provenance names what kind of region was masked and the
        display_name it resolved (rides alongside region_name via
        _attach_reproducibility)."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 0], [0, 0]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        plot_singular = make_plot_singular(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})

        result = json.loads(raw)
        self.assertNotIn("error", result)

        provenance = emitted["payload"]["provenance"]
        self.assertEqual(provenance["region_type"], "bounding_box")
        self.assertEqual(provenance["display_name"], "Global")

    async def test_compute_statistic_subcell_region_self_heals_to_boundary_cells(self):
        """T42: a region smaller than a grid cell covers no cell center, so
        center-containment masking returns nothing. Instead of "no valid
        data", the tool returns the touched boundary cells and discloses
        region_type: boundary_cells. The stubbed region is a 0.2° box inside
        the cell centered at (lat=20, lon=40) but not containing its center."""
        import xarray as xr
        from shapely.geometry import box
        from unittest.mock import AsyncMock
        from tta_backend.tools.satellite_tools import stat_tools

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 0], [0, 0]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        subcell = {
            "geometry": box(43.0, 23.0, 43.2, 23.2),
            "bounds": (43.0, 23.0, 43.2, 23.2),
            "name": "Tiny Neighborhood",
            "display_name": "Tiny Neighborhood",
            "region_type": "point_buffer",
        }

        compute_statistic_tool = stat_tools.make_compute_statistic_tool(self.mcp_tools)
        with patch.object(stat_tools._resolver, "aresolve_location", AsyncMock(return_value=subcell)):
            raw = await compute_statistic_tool.ainvoke({
                "handle": "obs_1", "location": "Tiny Neighborhood", "stats": ["mean"],
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertEqual(result["region_type"], "boundary_cells")
        self.assertGreaterEqual(result["n_pixels"], 1)

    async def test_find_daily_peak_excludes_a_bad_flag_pixel_even_though_it_is_numerically_highest(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.stat_tools import make_find_daily_peak

        def make_ds():
            # The numerically highest raw value (99.0) carries a bad flag;
            # the true peak once masked is the good-flag 3.0 cell.
            return _tempo_no2_dataset(
                xr, values=[[1.0, 99.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        find_daily_peak = make_find_daily_peak(self.mcp_tools)
        raw = await find_daily_peak.ainvoke({"handle": "obs_1", "location": "global"})
        result = json.loads(raw)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["peak_value"], 3.0)
        self.assertEqual(result["aggregation_meta"]["masking"]["qa_status"], "verified")

    async def test_conduct_temporal_statistic_masks_every_time_step_and_reports_verified_qa(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
                time=["2024-01-01", "2024-01-02"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        # Good cells per step (flag=0, lon=30 column at lat=10/20): step0 ->
        # [1.0, 3.0]; step1 -> [5.0, 7.0]. The plotted mean is cos(latitude)
        # area-weighted (area_weighted_mean), matching the stats tool -- an
        # unweighted mean would be 2.0/6.0 and disagree with it.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        step0 = round((1.0 * w10 + 3.0 * w20) / (w10 + w20), 6)
        step1 = round((5.0 * w10 + 7.0 * w20) / (w10 + w20), 6)
        self.assertEqual(full["values"], [step0, step1])
        self.assertEqual(full["masking"]["qa_status"], "verified")
        self.assertEqual(full["masking"]["qa_source"], "collections_yaml")

    async def test_conduct_temporal_statistic_reports_the_qa_loss_the_plotted_line_cannot_show(self):
        """T55: masking runs before each timestep collapses to one scalar, so a
        timestep the mask gutted still contributes a clean finite value. Here
        every timestep returns -- a naive "timesteps returned" completeness
        signal reads 100% -- while a quarter of the observations were actually
        discarded. Only the realized pass rate can say so."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            # One bad pixel per timestep, at a different latitude each time, so
            # the area-weighted rate is exactly 3/4 rather than a cos(lat) skew.
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[0, 1], [0, 0]], [[0, 0], [1, 0]]],
                time=["2024-01-01", "2024-01-02"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}
        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        self.assertNotIn("error", json.loads(raw))
        full = emitted["payload"]

        # Every timestep survived to the chart -- nothing about `values` reveals
        # the loss.
        self.assertEqual(len(full["values"]), 2)
        self.assertTrue(all(isinstance(v, float) for v in full["values"]))
        # ...but a quarter of the retrievable observations failed QA.
        self.assertAlmostEqual(full["masking"]["qa_pass_rate"], 0.75, places=6)
        self.assertEqual(full["masking"]["qa_checked_pixels"], 8)
        self.assertEqual(full["masking"]["qa_passing_pixels"], 6)

    async def test_a_totally_failed_day_survives_in_the_series_after_dropping_off_the_line(self):
        """One day at 0% and one at 100% average to the same 0.5 as two days at
        50% -- and the second is a data-quality event. The 0% day is dropped
        from the chart entirely (no finite value to plot), which is exactly why
        the companion series must cover every timestep and not only the plotted
        ones: otherwise the worst day is the one the report cannot show."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[1, 1], [1, 1]], [[0, 0], [0, 0]]],
                time=["2024-01-01", "2024-01-02"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}
        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        self.assertNotIn("error", json.loads(raw))
        masking = emitted["payload"]["masking"]

        # The first day never reaches the chart...
        self.assertEqual(len(emitted["payload"]["values"]), 1)
        # ...but the series still reports it, timestamped, as a total loss.
        self.assertEqual(masking["qa_pass_rate_by_time"], [0.0, 1.0])
        self.assertEqual(
            masking["qa_pass_rate_times"],
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        )
        self.assertAlmostEqual(masking["qa_pass_rate"], 0.5, places=6)

    async def test_conduct_temporal_statistic_mean_is_area_weighted_and_agrees_with_stats_tool(self):
        """The per-timestep regional mean must be the SAME cos(latitude)
        area-weighted mean the stats tool computes (area_weighted_mean), not a
        plain np.nanmean over grid cells. On a wide latitude band an unweighted
        mean over-weights high-latitude cells (cells shrink by cos(lat) toward
        the poles), so the trend line and the single-value stats mean for the
        identical region would numerically disagree."""
        import math
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        # A wide latitude band (30..70N) where weighting bites hard, all cells
        # good so masking doesn't confound the comparison.
        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[10.0, 12.0], [30.0, 32.0]], [[20.0, 22.0], [40.0, 42.0]]],
                flags=[[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                lat=(30.0, 70.0),
                lon=(30.0, 40.0),
                time=["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            )

        self.volume.add_zarr("obs_trend", make_ds)
        self.volume.add_zarr("obs_stat", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_trend", "location": "global", "stat": "mean",
            })
        result = json.loads(raw)
        self.assertNotIn("error", result)
        trend_values = emitted["payload"]["values"]

        w30, w70 = math.cos(math.radians(30.0)), math.cos(math.radians(70.0))
        # Step0 cells: lat30 -> [10,12], lat70 -> [30,32].
        step0 = ((10.0 + 12.0) * w30 + (30.0 + 32.0) * w70) / (2 * w30 + 2 * w70)
        # Step1 cells: lat30 -> [20,22], lat70 -> [40,42].
        step1 = ((20.0 + 22.0) * w30 + (40.0 + 42.0) * w70) / (2 * w30 + 2 * w70)
        self.assertAlmostEqual(trend_values[0], round(step0, 6), places=5)
        self.assertAlmostEqual(trend_values[1], round(step1, 6), places=5)

        # Guard against regression to the unweighted mean, which would give a
        # different (larger, high-lat-biased) number.
        unweighted_step0 = (10.0 + 12.0 + 30.0 + 32.0) / 4
        self.assertNotAlmostEqual(trend_values[0], unweighted_step0, places=3)

        # The stats tool, over the same single-step-equivalent region, must
        # return the SAME mean the trend plots for that step -- the two tools
        # can no longer disagree about "the regional mean."
        compute_statistic_tool = make_compute_statistic_tool(self.mcp_tools)

        def make_single():
            return _tempo_no2_dataset(
                xr,
                values=[[10.0, 12.0], [30.0, 32.0]],
                flags=[[0, 0], [0, 0]],
                lat=(30.0, 70.0),
                lon=(30.0, 40.0),
            )

        self.volume.add_zarr("obs_single", make_single)
        stat_raw = await compute_statistic_tool.ainvoke({
            "handle": "obs_single", "location": "global", "stats": ["mean"],
        })
        stat_result = json.loads(stat_raw)
        self.assertNotIn("error", stat_result)
        self.assertAlmostEqual(trend_values[0], round(stat_result["mean"], 6), places=5)

    async def test_conduct_temporal_statistic_attaches_aggregation_meta_like_the_heatmap_path(self):
        """T32: the timeseries chart path never called _attach_reproducibility
        with agg_meta at all, so its Granules/cadence block never rendered
        even though masking info was present. TEMPO_NO2 is registered
        cadence=hourly (collections.yaml), so this also proves cadence is
        threaded through, not just a hardcoded 'daily'."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
                time=["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        agg_meta = full["aggregation_meta"]
        self.assertEqual(agg_meta["n_granules"], 2)
        self.assertEqual(agg_meta["cadence"], "hourly")
        self.assertEqual(len(agg_meta["granule_dates"]), 2)
        self.assertEqual(agg_meta["masking"]["qa_status"], "verified")

        # The same fields land in provenance (T25 Phase 3 convention),
        # never just in internal aggregation_meta.
        self.assertEqual(full["provenance"]["n_granules"], 2)
        self.assertEqual(full["provenance"]["cadence"], "hourly")
        self.assertEqual(len(full["provenance"]["granule_dates"]), 2)

    async def test_conduct_temporal_statistic_granule_dates_match_the_sorted_chart_order(self):
        """The chart's times/values are explicitly sorted chronologically
        before charting (source timesteps aren't guaranteed monotonic, e.g.
        granules from multiple downloads concatenated out of order). The
        aggregation_meta granule_dates/date-range must reflect that same
        sorted order, not the pre-sort loop order -- otherwise the Metadata
        tab's date range can disagree with what's actually plotted."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            # Source timesteps arrive out of chronological order.
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                flags=[[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                time=["2024-01-02T00:00:00", "2024-01-01T00:00:00"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        result = json.loads(raw)
        self.assertNotIn("error", result)

        full = emitted["payload"]
        # Chart is sorted chronologically: Jan 1 (data [[5,6],[7,8]]) before
        # Jan 2 (data [[1,2],[3,4]]). Means are cos(latitude) area-weighted
        # (area_weighted_mean); the unweighted means would be 6.5/2.5.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        jan1 = round(((5.0 + 6.0) * w10 + (7.0 + 8.0) * w20) / (2 * w10 + 2 * w20), 6)
        jan2 = round(((1.0 + 2.0) * w10 + (3.0 + 4.0) * w20) / (2 * w10 + 2 * w20), 6)
        self.assertEqual(full["times"][0][:10], "2024-01-01")
        self.assertEqual(full["times"][1][:10], "2024-01-02")
        self.assertEqual(full["values"], [jan1, jan2])

        agg_meta = full["aggregation_meta"]
        self.assertEqual(agg_meta["granule_dates"], ["2024-01-01", "2024-01-02"])
        self.assertIn("2024-01-01 to 2024-01-02", agg_meta["aggregation_label"])

    async def test_conduct_temporal_statistic_attaches_dataset_and_source_from_registry(self):
        """T32: dataset/source come from the TEMPO_NO2 registry entry matched
        via the opened granule's short_name attribute -- the same match
        col_info_for_variable already performs for masking, not a second
        lookup."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        def make_ds():
            return _tempo_no2_dataset(
                xr,
                values=[[[1.0, 2.0], [3.0, 4.0]]],
                flags=[[[0, 1], [0, 1]]],
                time=["2024-01-01T00:00:00"],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        conduct_temporal_statistic = make_conduct_temporal_statistic(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await conduct_temporal_statistic.ainvoke({
                "handle": "obs_1", "location": "global", "stat": "mean",
            })

        result = json.loads(raw)
        self.assertNotIn("error", result)

        provenance = emitted["payload"]["provenance"]
        self.assertEqual(provenance["dataset"], "TEMPO_NO2_L3")
        self.assertEqual(provenance["provider"], "NASA LARC")
        self.assertEqual(provenance["instrument"], "TEMPO")
        self.assertEqual(provenance["source"], "NASA LARC — TEMPO")
        self.assertEqual(provenance["qa_methodology"]["quality_flag_var"], "main_data_quality_flag")
        self.assertEqual(provenance["qa_methodology"]["qa_good_values"], [0])

    async def test_plot_singular_attaches_dataset_and_source_without_a_second_registry_lookup(self):
        """T32's variable-definition/dataset attach must ride on the same
        col_info the masking pipeline already resolved -- not a second call
        to the registry. Spies on col_info_for_variable (the shared seam every
        satellite tool resolves masking metadata through) and asserts the call
        count is unchanged from before this PRD: exactly one, for the one
        masking resolution the tool already performed."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        def make_ds():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_1", make_ds)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        import tta_backend.tools.satellite_tools.plot_tools as plot_tools_module
        calls = []
        real_lookup = plot_tools_module.col_info_for_variable

        def counting_lookup(da, ds=None):
            calls.append(da.name)
            return real_lookup(da, ds)

        plot_singular = make_plot_singular(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart), \
             patch.object(plot_tools_module, "col_info_for_variable", counting_lookup):
            raw = await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})

        result = json.loads(raw)
        self.assertNotIn("error", result)

        self.assertEqual(len(calls), 1)

        provenance = emitted["payload"]["provenance"]
        self.assertEqual(provenance["dataset"], "TEMPO_NO2_L3")
        self.assertEqual(provenance["source"], "NASA LARC — TEMPO")
        self.assertEqual(provenance["variable_definition"]["mask_note"], "fill values and a valid range are defined")

    async def test_compare_region_mode_masks_bad_flag_pixels_on_both_sides(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools import comparison_tools

        def make_a():
            return _tempo_no2_dataset(
                xr, values=[[1.0, 2.0], [3.0, 4.0]], flags=[[0, 1], [0, 1]],
            )

        def make_b():
            return _tempo_no2_dataset(
                xr, values=[[10.0, 20.0], [30.0, 40.0]], flags=[[0, 1], [0, 1]],
            )

        self.volume.add_zarr("obs_a", make_a)
        self.volume.add_zarr("obs_b", make_b)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        compare = comparison_tools.make_compare(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            raw = await compare.ainvoke({
                "handle_a": "obs_a", "handle_b": "obs_b", "mode": "region",
                "label_a": "A", "label_b": "B",
            })
        result = json.loads(raw)

        self.assertNotIn("error", result)
        full = emitted["payload"]
        # Good cells (flag=0) only: A -> 1.0 at lat=10, 3.0 at lat=20;
        # B is 10x A. Means are cos(latitude) area-weighted.
        import math

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected_a = (1.0 * w10 + 3.0 * w20) / (w10 + w20)
        self.assertAlmostEqual(full["stats"]["A"]["mean"], expected_a)
        self.assertAlmostEqual(full["stats"]["B"]["mean"], 10.0 * expected_a)


if __name__ == "__main__":
    unittest.main()
