import importlib.util
import unittest


REQUIRED_MODULES = ["affine", "cartopy", "langchain", "numpy", "rasterio", "shapely", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite plotting dependencies are not installed",
)
class SatellitePlotPayloadTests(unittest.TestCase):
    def test_geometry_mask_handles_time_lon_lat_dimension_order(self):
        import numpy as np
        import xarray as xr
        from shapely.geometry import box
        from tta_backend.utils.plotting import mask_data_by_geometry

        da = xr.DataArray(
            np.ones((1, 5, 4)),
            dims=("time", "Longitude", "Latitude"),
            coords={
                "time": ["2024-01-01"],
                "Longitude": np.linspace(-100.0, -96.0, 5),
                "Latitude": np.linspace(30.0, 33.0, 4),
            },
        )

        geometry = box(-99.5, 30.5, -96.5, 32.5)
        masked = mask_data_by_geometry(da, geometry)
        uncropped = mask_data_by_geometry(da, geometry, crop=False)

        self.assertEqual(masked.dims, ("time", "Longitude", "Latitude"))
        self.assertTrue(np.isfinite(masked.values).any())
        # Cells outside the geometry are gone -- dropped by the T50 crop where
        # they used to survive as NaN. Same kept values either way.
        self.assertLess(masked.size, da.size)
        np.testing.assert_array_equal(
            np.sort(masked.values[np.isfinite(masked.values)]),
            np.sort(uncropped.values[np.isfinite(uncropped.values)]),
        )

    def test_payload_omits_the_vestigial_points_array(self):
        """``points`` was a second, flattened copy of the same field -- up to
        8,000 lat/lon/value triples, ~270 KB of JSON on every heatmap, stored
        durably in Postgres alongside the grid it duplicated.

        Nothing renders from it. The map draws the server-rendered overlay PNG,
        and falls back to the 2-D ``values`` grid (buildCanvasFallbackFrame);
        the Statistics tab reads ``statistics``. Its last reader was
        chartStats.rawCellValues, and only for payloads predating
        ``statistics`` -- a read-path fallback that stays, because old charts
        are durable rows. New payloads simply stop paying for it.
        """
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

        payload = _da_to_heatmap_payload(da, "Compact", "NO2", "mol/cm2")

        self.assertNotIn("points", payload)
        # the grid the frontend actually renders from is untouched
        self.assertEqual(payload["values"], [[1.0, 2.0], [3.0, 4.0]])

    def test_sparse_valid_cells_survive_thinning_in_the_statistics(self):
        """A sparse field is exactly where the render grid's uniform stride is
        lossiest -- here it steps over both valid cells and the rendered grid
        comes back empty. ``points`` used to be what preserved them.

        Nothing consumed that preservation: what a reader is actually told
        about a sparse scene comes from ``statistics``, computed on the full
        field before ``_downsample_grid`` thins it. That is the guarantee this
        test pins, and it holds without a second copy of the array.
        """
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _MAX_GRID_CELLS, _da_to_heatmap_payload

        arr = np.full((120, 120), np.nan)
        arr[3, 5] = 1.25
        arr[90, 95] = 2.5
        da = xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 120), "lon": np.linspace(-100, -90, 120)},
        )

        payload = _da_to_heatmap_payload(da, "Sparse", "NO2", "mol/cm2")

        rendered = [v for row in payload["values"] for v in row if v is not None]
        self.assertLessEqual(len(rendered), _MAX_GRID_CELLS)
        self.assertEqual(rendered, [])            # the stride really drops both

        self.assertEqual(payload["statistics"]["count"], 2)
        self.assertEqual(payload["statistics"]["min"], 1.25)
        self.assertEqual(payload["statistics"]["max"], 2.5)

    def test_reported_statistics_describe_the_full_field_not_the_rendered_grid(self):
        """The grid in the payload is thinned to _MAX_GRID_CELLS for rendering,
        so a peak that falls between kept rows/columns disappears from it. A
        statistic computed on what survives describes the thumbnail, not the
        analyzed region: measured 2026-08-01 on a real TEMPO NO2 L3 scene, the
        true maximum 9.2418e+17 was reported as 6.9003e+16 -- low by 13.4x."""
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _MAX_GRID_CELLS, _da_to_heatmap_payload

        arr = np.full((200, 200), 1.0)          # 40,000 cells -> thinned
        arr[1, 1] = 1000.0                      # a peak the stride steps over
        da = xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 200), "lon": np.linspace(-100, -90, 200)},
        )

        payload = _da_to_heatmap_payload(da, "Peak", "NO2", "mol/cm2")

        rendered = [v for row in payload["values"] for v in row if v is not None]
        self.assertLessEqual(len(rendered), _MAX_GRID_CELLS)
        self.assertNotIn(1000.0, rendered)      # the peak really is dropped
        self.assertEqual(payload["statistics"]["max"], 1000.0)

    def test_reported_mean_weights_cells_by_latitude(self):
        """Grid cells shrink toward the poles, so a plain cell average
        overweights high-latitude cells. The stats and trend tools already
        report cos(latitude)-weighted means; a map's own reported mean has to
        agree with them or the same scene answers two different numbers."""
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        lats = np.linspace(0.0, 80.0, 100)
        arr = np.tile(np.where(lats < 40.0, 0.0, 100.0)[:, None], (1, 100))  # 50 rows each way
        da = xr.DataArray(
            arr, dims=("lat", "lon"),
            coords={"lat": lats, "lon": np.linspace(-10, 10, 100)},
        )

        payload = _da_to_heatmap_payload(da, "Weighted", "NO2", "mol/cm2")

        self.assertEqual(float(arr.mean()), 50.0)          # the unweighted answer
        weights = np.cos(np.deg2rad(lats))
        expected = float(np.average(arr.mean(axis=1), weights=weights))
        self.assertLess(expected, 50.0)                    # weighting must move it
        self.assertAlmostEqual(payload["statistics"]["mean"], expected, delta=expected * 1e-5)

    def test_payload_normalizes_longitudes_and_sanitizes_values(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.array([[np.inf, 4.0, np.nan]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [350.0, 355.0, 5.0]},
        )

        payload = _da_to_heatmap_payload(da, "Wrapped", "NO2", "mol/cm2")

        self.assertEqual(payload["values"], [[None, 4.0, None]])
        self.assertEqual(payload["lons"], [-10.0, -5.0, 5.0])
        self.assertLess(payload["vmin"], 4.0)
        self.assertGreater(payload["vmax"], 4.0)

    def test_payload_attaches_the_resolved_colormap(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from tta_backend.utils.colormaps import resolve

        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

        payload = _da_to_heatmap_payload(da, "TEMPO over NJ", "NO2", "mol/m^2")

        expected = resolve("NO2")
        self.assertEqual(payload["colormap"]["name"], expected.name)
        self.assertEqual(payload["colormap"]["lut"], expected.lut)

    def test_diverging_payload_attaches_the_diverging_colormap(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from tta_backend.utils.colormaps import resolve

        da = xr.DataArray(
            np.array([[-1.0, 2.0], [3.0, -4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

        payload = _da_to_heatmap_payload(da, "Diff", "NO2", "mol/m^2", diverging=True)

        self.assertEqual(payload["colormap"]["name"], resolve("NO2", diverging=True).name)
        self.assertEqual(payload["colormap"]["name"], "RdBu_r")

    def test_payload_attaches_edge_extended_overlay_bounds_matching_the_rendered_png(self):
        # overlay.bounds must describe the raster's pixel-EDGE extent, because
        # render_overlay_png rasterizes edge-to-edge (left = lons[0] - res/2,
        # etc.). Reporting the pixel-CENTER min/max instead pins an edge-to-edge
        # PNG onto center-to-center bounds, displacing every pixel up to half a
        # cell on coarse grids and pushing edge rows outside the declared box.
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.ones((3, 4)),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 3), "lon": np.linspace(-100, -90, 4)},
        )

        payload = _da_to_heatmap_payload(da, "Extent", "NO2", "mol/m^2")

        # lat step 5 -> half-cell 2.5 -> [7.5, 22.5]; lon step 10/3 -> half-cell
        # 5/3 -> [-101.6667, -88.3333]. These are render_overlay_png's own
        # left/bottom/right/top for this grid.
        minx, miny, maxx, maxy = payload["overlay"]["bounds"]
        self.assertAlmostEqual(minx, -100.0 - 5.0 / 3.0)
        self.assertAlmostEqual(miny, 7.5)
        self.assertAlmostEqual(maxx, -90.0 + 5.0 / 3.0)
        self.assertAlmostEqual(maxy, 22.5)
        self.assertNotIn("_path", payload["overlay"])

    def test_value_range_override_drives_both_reported_bounds_and_overlay_colorization(self):
        import io
        import numpy as np
        import matplotlib.image as mpimg
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from tta_backend.utils.colormaps import resolve

        da = xr.DataArray(
            np.full((6, 8), 5.0),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 6), "lon": np.linspace(-100, -90, 8)},
        )

        # A caller (comparison_tools) overriding the natural percentile bounds
        # with a shared/diverging scale -- the overlay must colorize against
        # *this* range, not the value's own percentile bounds, or the map and
        # its legend would disagree about what the color means.
        payload = _da_to_heatmap_payload(
            da, "Shared scale", "NO2", "mol/m^2", render_overlay=True, value_range=(0.0, 10.0),
        )

        self.assertEqual(payload["vmin"], 0.0)
        self.assertEqual(payload["vmax"], 10.0)

        with open(payload["overlay"]["_path"], "rb") as f:
            decoded = mpimg.imread(io.BytesIO(f.read()), format="png")
        center = np.array(decoded.shape[:2]) // 2
        pixel = tuple((decoded[center[0], center[1]] * 255).round().astype(int))
        expected = tuple(resolve("NO2").lut[128])  # 5.0 is the midpoint of [0, 10]
        self.assertEqual(pixel, expected)

    def test_percentile_derived_bounds_stamp_a_percentile_scale_the_legend_can_disclose(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.linspace(0.0, 1.0, 12).reshape(3, 4),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 3), "lon": np.linspace(-100, -90, 4)},
        )

        payload = _da_to_heatmap_payload(da, "Auto", "NO2", "mol/m^2")

        # The vmin/vmax came from the 2nd–98th percentile clip -> the legend
        # can honestly say the extremes are saturated.
        self.assertEqual(payload["scale"], {"method": "percentile", "p": [2, 98]})

    def test_explicit_value_range_stamps_an_explicit_scale(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.full((6, 8), 5.0),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 6), "lon": np.linspace(-100, -90, 8)},
        )

        # A truly fixed range with no disclosure of how it was derived is not a
        # clip of this array -> no clip disclosure.
        payload = _da_to_heatmap_payload(da, "Shared", "NO2", "mol/m^2", value_range=(0.0, 10.0))

        self.assertEqual(payload["scale"], {"method": "explicit"})

    def test_value_range_with_a_scale_disclosure_stamps_that_disclosure_not_explicit(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.full((6, 8), 5.0),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 6), "lon": np.linspace(-100, -90, 8)},
        )

        # A comparison caller imposes a shared/diverging range that IS a
        # percentile clip (just computed across panels). Passing the method
        # through keeps the legend's saturation warning honest instead of
        # collapsing every imposed range to "explicit" (nothing to disclose).
        payload = _da_to_heatmap_payload(
            da, "Comparison panel", "NO2", "mol/m^2", value_range=(0.0, 10.0),
            scale_disclosure={"method": "percentile", "p": [2, 98]},
        )

        self.assertEqual(payload["scale"], {"method": "percentile", "p": [2, 98]})
        # The imposed range is still what colorizes the map.
        self.assertEqual(payload["vmin"], 0.0)
        self.assertEqual(payload["vmax"], 10.0)

    def test_render_overlay_true_persists_a_png_and_records_its_path(self):
        import os
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.linspace(0.0, 1.0, 12).reshape(3, 4),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 3), "lon": np.linspace(-100, -90, 4)},
        )

        payload = _da_to_heatmap_payload(da, "Extent", "NO2", "mol/m^2", render_overlay=True)

        path = payload["overlay"]["_path"]
        self.assertTrue(os.path.isfile(path))
        with open(path, "rb") as f:
            self.assertTrue(f.read().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_reproducibility_metadata_uses_source_handles(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-01-01T00:00:00Z"},
            name="TEMPO_NO2",
            attrs={"units": "mol/m^2"},
        )
        region = {"bounds": [-75.0, 39.0, -73.0, 41.0]}

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over NJ"},
            ["obs_1"],
            da,
            "New Jersey",
            "single snapshot",
            {"chart_type": "heatmap"},
            region=region,
        )

        self.assertEqual(payload["provenance"]["variable"], "TEMPO_NO2")
        self.assertEqual(payload["provenance"]["region_name"], "New Jersey")
        self.assertEqual(payload["provenance"]["source_handles"], ["obs_1"])
        self.assertEqual(payload["query"]["dataset"], "TEMPO_NO2")
        self.assertEqual(payload["query"]["bbox"], [-75.0, 39.0, -73.0, 41.0])
        self.assertEqual(payload["query"]["aggregation"], "single snapshot")
        self.assertEqual(payload["metadata"]["source_handles"], ["obs_1"])

    def test_provenance_attaches_dataset_and_source_distinct_from_variable(self):
        """T32: `dataset`/`source` are real registry facts about the
        collection, not a fallback that reuses the plotted variable name."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-01-01T00:00:00Z"},
            name="vertical_column_troposphere",
            attrs={"units": "mol/m^2"},
        )
        col_info = {
            "short_name": "TEMPO_NO2_L3",
            "description": "TEMPO tropospheric NO2 vertical column",
            "version": "V04",
            "collection_id": "C3685896708-LARC_CLOUD",
            "provider": "NASA LARC",
            "instrument": "TEMPO",
        }

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over NJ"},
            ["obs_1"], da, "New Jersey", "single snapshot",
            region={"bounds": [-75.0, 39.0, -73.0, 41.0]}, col_info=col_info,
        )

        provenance = payload["provenance"]
        self.assertEqual(provenance["variable"], "vertical_column_troposphere")
        self.assertEqual(provenance["dataset"], "TEMPO_NO2_L3")
        self.assertNotEqual(provenance["dataset"], provenance["variable"])
        self.assertEqual(provenance["dataset_description"], "TEMPO tropospheric NO2 vertical column")
        self.assertEqual(provenance["dataset_version"], "V04")
        self.assertEqual(provenance["collection_id"], "C3685896708-LARC_CLOUD")
        self.assertEqual(provenance["provider"], "NASA LARC")
        self.assertEqual(provenance["instrument"], "TEMPO")
        self.assertEqual(provenance["source"], "NASA LARC — TEMPO")

    def test_provenance_attaches_variable_definition_and_qa_methodology(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-01-01T00:00:00Z"},
            name="vertical_column_troposphere",
            attrs={"units": "mol/m^2", "long_name": "NO2 tropospheric column"},
        )
        col_info = {
            "short_name": "TEMPO_NO2_L3",
            "valid_min": -1.0e15,
            "valid_max": 1.0e18,
            "fill_value": -1.0e30,
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
        }

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over NJ"},
            ["obs_1"], da, "New Jersey", "single snapshot",
            region={"bounds": [-75.0, 39.0, -73.0, 41.0]}, col_info=col_info,
        )

        var_def = payload["provenance"]["variable_definition"]
        self.assertEqual(var_def["long_name"], "NO2 tropospheric column")
        self.assertEqual(var_def["valid_ranges"], {"min": -1.0e15, "max": 1.0e18})
        self.assertEqual(var_def["fill_value"], -1.0e30)
        self.assertEqual(var_def["mask_note"], "fill values and a valid range are defined")
        self.assertEqual(var_def["advisory_notes"], [])

        qa_methodology = payload["provenance"]["qa_methodology"]
        self.assertEqual(qa_methodology["quality_flag_var"], "main_data_quality_flag")
        self.assertEqual(qa_methodology["qa_good_values"], [0])

    def test_provenance_stamps_delivered_scope_from_the_plotted_data(self):
        """T46: the scope actually delivered — region, the data's own date
        span, and cadence — travels in provenance so the disclosure layer (and
        the Metadata tab) can compare it against what was requested."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-07-01T00:00:00Z"},
            name="TEMPO_NO2",
            attrs={"units": "mol/m^2"},
        )
        agg_meta = {
            "aggregation_label": "monthly mean",
            "n_granules": 1,
            "cadence": "monthly",
            "granule_dates": ["2024-07-01"],
        }

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over CA"},
            ["obs_1"], da, "California", "monthly mean",
            agg_meta=agg_meta, region={"bounds": [-124.0, 32.0, -114.0, 42.0]},
        )

        delivered = payload["provenance"]["delivered_scope"]
        self.assertEqual(delivered["region_name"], "California")
        self.assertEqual(delivered["cadence"], "monthly")
        self.assertTrue(delivered["start_date"].startswith("2024-07-01"))

    def test_provenance_stamps_requested_scope_recorded_for_the_handle(self):
        """T46: the requested scope the composite recorded against this handle
        is echoed back into provenance, so a single-day request answered by a
        monthly mean is disclosable end-to-end."""
        import xarray as xr
        from tta_backend.services import scope_registry
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        scope_registry.record_pending("job_x", {"location": "California", "time_range": "2024-07-15/2024-07-15"})
        scope_registry.finalize("job_x", "obs_scope_1")

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-07-01T00:00:00Z"},
            name="TEMPO_NO2",
            attrs={"units": "mol/m^2"},
        )

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over CA"},
            ["obs_scope_1"], da, "California", "monthly mean",
            region={"bounds": [-124.0, 32.0, -114.0, 42.0]},
        )

        requested = payload["provenance"]["requested_scope"]
        self.assertEqual(requested["location"], "California")
        self.assertEqual(requested["time_range"], "2024-07-15/2024-07-15")

    def test_provenance_missing_dataset_facts_render_as_empty_not_error(self):
        """No col_info at all (unregistered collection) must not raise --
        every field the frontend expects is still present, just empty, so
        the UI can render 'Not available' rather than crash on a missing key."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-01-01T00:00:00Z"},
            name="unregistered_var",
            attrs={"units": "mol/m^2"},
        )

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "Unregistered"},
            ["obs_1"], da, "New Jersey", "single snapshot",
            region={"bounds": [-75.0, 39.0, -73.0, 41.0]},
        )

        provenance = payload["provenance"]
        self.assertEqual(provenance["dataset"], "")
        self.assertEqual(provenance["source"], "")
        self.assertEqual(provenance["provider"], "")
        self.assertEqual(provenance["instrument"], "")
        self.assertEqual(provenance["variable_definition"]["long_name"], "")
        self.assertEqual(provenance["variable_definition"]["valid_ranges"], {})
        self.assertEqual(provenance["variable_definition"]["mask_note"], "no fill/range metadata")
        self.assertEqual(provenance["qa_methodology"], {})

    def test_save_chart_mints_a_map_artifact_id_for_a_heatmap_payload(self):
        import json
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility, _save_chart

        da = xr.DataArray(
            [[1.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [-74.0], "time": "2024-01-01T00:00:00Z"},
            name="TEMPO_NO2",
            attrs={"units": "mol/m^2"},
        )
        region = {"bounds": [-75.0, 39.0, -73.0, 41.0]}
        payload = _attach_reproducibility(
            {
                "type": "heatmap",
                "title": "TEMPO over NJ",
                "variable": "TEMPO_NO2",
                "units": "mol/m^2",
                "vmin": 0.0,
                "vmax": 1.0,
                "bounds": region["bounds"],
            },
            ["obs_1"],
            da,
            "New Jersey",
            "single snapshot",
            region=region,
        )

        result = json.loads(_save_chart(payload, "TEMPO_NO2_NJ"))

        self.assertTrue(result["chart_id"].startswith("map_"))
        self.assertEqual(len(result["_artifact_refs"]), 1)
        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["id"], result["chart_id"])
        self.assertEqual(ref["type"], "map")
        self.assertEqual(ref["metadata"]["bbox"], region["bounds"])
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_1"])

    def test_save_chart_emits_the_full_payload_and_returns_a_compact_summary(self):
        import json
        from unittest.mock import patch
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import (
            _attach_reproducibility,
            _da_to_heatmap_payload,
            _save_chart,
        )

        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0], "time": "2024-01-01T00:00:00Z"},
            name="TEMPO_NO2",
            attrs={"units": "mol/m^2"},
        )
        region = {"bounds": [-75.0, 39.0, -73.0, 41.0]}
        payload = _da_to_heatmap_payload(da, "TEMPO over NJ", "TEMPO_NO2", "mol/m^2")
        payload["bounds"] = region["bounds"]
        _attach_reproducibility(payload, ["obs_1"], da, "New Jersey", "single snapshot", region=region)

        emitted = {}

        def fake_emit_chart(full_payload):
            emitted["payload"] = full_payload

        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
            result = json.loads(_save_chart(payload, "TEMPO_NO2_NJ"))

        # (a) the frontend's chart/artifact pipeline still gets the full grid,
        # out-of-band from the model-facing return value.
        self.assertEqual(emitted["payload"]["values"], payload["values"])
        self.assertEqual(emitted["payload"]["lats"], payload["lats"])

        # (b) the model-facing tool result is compact — no raw grid/points/
        # provenance blocks, well under what an 8000-cell grid would cost.
        for bulky_key in ("values", "points", "lats", "lons", "provenance", "query", "export"):
            self.assertNotIn(bulky_key, result)
        self.assertLess(len(json.dumps(result)), 1000)

        # ...but still everything the agent needs to describe and cite it.
        self.assertEqual(result["render_type"], "heatmap")
        self.assertEqual(result["variable"], "TEMPO_NO2")
        self.assertEqual(result["units"], "mol/m^2")
        self.assertEqual(result["grid_dims"], [2, 2])
        self.assertTrue(result["chart_id"].startswith("map_"))
        self.assertEqual(result["source_handles"], ["obs_1"])
        self.assertEqual(result["_artifact_refs"][0]["id"], result["chart_id"])

    def test_save_chart_mints_a_comparison_artifact_id_for_a_heatmap_multi_payload(self):
        import json
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility, _save_chart

        def _panel(name, handle, lon, lat):
            da = xr.DataArray(
                [[1.0]],
                dims=("lat", "lon"),
                coords={"lat": [lat], "lon": [lon], "time": "2024-01-01T00:00:00Z"},
                name="TEMPO_NO2",
                attrs={"units": "mol/m^2"},
            )
            panel = {"type": "heatmap", "title": name, "variable": "TEMPO_NO2", "units": "mol/m^2"}
            _attach_reproducibility(panel, [handle], da, name, "single snapshot")
            return panel

        panels = [_panel("New Jersey", "obs_1", -74.0, 40.0), _panel("New York", "obs_2", -73.9, 40.7)]
        multi_payload = {
            "type": "heatmap_multi",
            "title": "TEMPO NO2 Comparison",
            "panels": panels,
            "metadata": {"source_handles": ["obs_1", "obs_2"]},
        }

        result = json.loads(_save_chart(multi_payload, "TEMPO_NO2_comparison"))

        self.assertTrue(result["chart_id"].startswith("cmp_"))
        ref = result["_artifact_refs"][0]
        self.assertEqual(ref["type"], "comparison")
        self.assertEqual(ref["metadata"]["panels"][0]["handle"], "obs_1")
        self.assertEqual(ref["metadata"]["panels"][1]["handle"], "obs_2")
        self.assertEqual(ref["metadata"]["source_handles"], ["obs_1", "obs_2"])

    def test_save_chart_wires_the_overlay_url_from_the_minted_chart_id(self):
        import json
        from tta_backend.tools.satellite_tools.plot_tools import _save_chart

        payload = {"type": "heatmap", "title": "Has overlay", "overlay": {"bounds": [0, 0, 1, 1], "_path": "/tmp/x.png"}}

        result = json.loads(_save_chart(payload, "n/a"))

        self.assertEqual(payload["overlay"]["url"], f"/chart/{payload['chart_id']}/overlay.png")
        # The internal filesystem path never reaches the model-facing summary.
        self.assertNotIn("overlay", result)

    def test_save_chart_leaves_overlay_url_unset_when_render_failed(self):
        from tta_backend.tools.satellite_tools.plot_tools import _save_chart

        payload = {"type": "heatmap", "title": "No overlay", "overlay": {"bounds": [0, 0, 1, 1]}}

        _save_chart(payload, "n/a")

        self.assertNotIn("url", payload["overlay"])

    def test_save_chart_wires_a_per_panel_overlay_url_for_heatmap_multi(self):
        from tta_backend.tools.satellite_tools.plot_tools import _save_chart

        payload = {
            "type": "heatmap_multi",
            "title": "Comparison",
            "panels": [
                {"title": "A", "overlay": {"bounds": [0, 0, 1, 1], "_path": "/tmp/a.png"}},
                {"title": "B", "overlay": {"bounds": [0, 0, 1, 1]}},  # render failed for B
            ],
        }

        _save_chart(payload, "n/a")

        chart_id = payload["chart_id"]
        self.assertEqual(payload["panels"][0]["overlay"]["url"], f"/chart/{chart_id}/overlay.png?panel=0")
        self.assertNotIn("url", payload["panels"][1]["overlay"])

    def test_save_chart_wires_the_difference_overlay_url_for_heatmap_multi(self):
        from tta_backend.tools.satellite_tools.plot_tools import _save_chart

        payload = {
            "type": "heatmap_multi",
            "mode": "difference",
            "title": "Diff",
            "panels": [{"title": "A"}, {"title": "B"}],
            "difference": {"overlay": {"bounds": [0, 0, 1, 1], "_path": "/tmp/diff.png"}},
        }

        _save_chart(payload, "n/a")

        chart_id = payload["chart_id"]
        self.assertEqual(payload["difference"]["overlay"]["url"], f"/chart/{chart_id}/overlay.png")

    def test_save_chart_omits_artifact_refs_for_an_unmapped_render_type(self):
        import json
        from tta_backend.tools.satellite_tools.plot_tools import _save_chart

        result = json.loads(_save_chart({"type": "error"}, "n/a"))

        self.assertNotIn("_artifact_refs", result)

    @staticmethod
    def _reduced_2d_da():
        """A time-less 2D DataArray, shaped like what reaches _provenance/
        _query_definition after aggregation collapsed (or the granule never
        had) the time dimension."""
        import numpy as np
        import xarray as xr

        return xr.DataArray(
            np.ones((2, 2)),
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
            name="no2",
            attrs={"units": "molec cm-2"},
        )

    def test_provenance_dates_fall_back_to_aggregation_meta_for_a_timeless_reduced_array(self):
        """Regression: the Metadata tab showed "Date Range: Not available" for
        every aggregated map -- _provenance derived start/end from the
        *reduced* array (time dim already collapsed), and monthly L3 granules
        never had a time coordinate at all. The aggregation meta now carries
        the range; provenance must use it."""
        from tta_backend.tools.satellite_tools.plot_tools import _provenance

        agg_meta = {
            "aggregation_label": "Single Snapshot Mean, 1 monthly granule, 2024-01-01 to 2024-01-31",
            "title_suffix": "Single Snapshot Mean (2024, 1 monthly granule)",
            "granule_dates": ["2024-01-01"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "n_granules": 1,
            "cadence": "monthly",
            "stat": "mean",
        }

        provenance = _provenance(["obs_x"], self._reduced_2d_da(), "Houston", "mean", agg_meta)

        self.assertEqual(provenance["start_date"], "2024-01-01")
        self.assertEqual(provenance["end_date"], "2024-01-31")
        self.assertEqual(provenance["granule_dates"], ["2024-01-01"])

    def test_query_definition_dates_fall_back_to_aggregation_meta(self):
        from tta_backend.tools.satellite_tools.plot_tools import _query_definition

        agg_meta = {"start_date": "2024-01-01", "end_date": "2024-01-31", "granule_dates": ["2024-01-01"]}

        query = _query_definition(self._reduced_2d_da(), None, "mean", agg_meta=agg_meta)

        self.assertEqual(query["start_date"], "2024-01-01")
        self.assertEqual(query["end_date"], "2024-01-31")

    def test_time_range_prefers_the_time_coordinate_over_aggregation_meta(self):
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _time_range

        da = xr.DataArray(
            np.ones((2, 1, 1)),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-06-01", "2024-06-02"], "lat": [40.0], "lon": [-75.0]},
        )

        start, end = _time_range(da, {"start_date": "1999-01-01", "end_date": "1999-01-31"})

        self.assertEqual(start, "2024-06-01")
        self.assertEqual(end, "2024-06-02")

    def test_building_a_payload_holds_a_bounded_multiple_of_the_field(self):
        """Building a heatmap must not need an unbounded multiple of the field.

        This is the stage that OOM-killed the backend on 2026-08-05 (full-day
        TEMPO over North America): the retrieval and the materialize both
        succeeded, then uvicorn was SIGKILLed with ~2.8 GB RSS. Measured cause
        was accumulation, not any single allocation -- the reduced field was
        upcast to float64 on arrival and then copied again by every step that
        read it (percentile bounds, overlay rasterization, statistics, the
        point list), all alive at once. 62.8 bytes per cell measured; at TEMPO
        CONUS native resolution (2880x7750) that is ~1.4 GB for one map.

        The ceiling is the guarantee. A satellite retrieval carries nowhere
        near float64 precision and the map is drawn in 8-bit color, so the
        headroom that bought the doubling was never real.
        """
        import tracemalloc
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        def field(nlat, nlon):
            return xr.DataArray(
                np.linspace(0.0, 1.0, nlat * nlon, dtype=np.float32).reshape(nlat, nlon),
                dims=("lat", "lon"),
                coords={"lat": np.linspace(20.0, 55.0, nlat), "lon": np.linspace(-130.0, -60.0, nlon)},
            )

        # Warm up: matplotlib/rasterio import once, and must not be charged
        # to the measured build.
        _da_to_heatmap_payload(field(40, 50), "warmup", "NO2", "mol/cm2", render_overlay=True)

        da = field(600, 800)
        cells = da.size

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            _da_to_heatmap_payload(da, "Peak memory", "NO2", "mol/cm2", render_overlay=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        bytes_per_cell = (peak - before) / cells
        self.assertLess(
            bytes_per_cell, 30.0,
            f"payload build peaked at {bytes_per_cell:.1f} bytes per cell; at TEMPO "
            f"CONUS native resolution (22.3M cells) that is "
            f"{bytes_per_cell * 22.3e6 / 1e9:.2f} GB for a single map",
        )

    def test_statistics_survive_the_working_precision_of_the_field(self):
        """Reported statistics must describe the data, not the buffer it sat in.

        The field is carried at float32 to keep a native-resolution render off
        the OOM killer, and that is safe for pixels but not automatically safe
        for a mean: summing millions of same-signed values in float32 drifts,
        and this mean is a published scientific number that the stats and
        trend tools are required to agree with (see _area_weighted_mean). So
        the reduction accumulates in float64 regardless of how the field is
        stored, and this pins that -- against a float64 field carrying values
        at TEMPO NO2's real magnitude, where naive float32 accumulation is
        visible well inside the six significant digits the payload reports.
        """
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        rng = np.random.default_rng(20260805)
        lats = np.linspace(20.0, 55.0, 400)
        lons = np.linspace(-130.0, -60.0, 500)
        # Real TEMPO NO2 column magnitudes, where float32 has ~7 significant
        # digits total and a large running sum eats them.
        values = 9.0e17 + rng.normal(0.0, 1.0e15, size=(lats.size, lons.size))

        def payload_for(dtype):
            da = xr.DataArray(
                values.astype(dtype), dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            )
            return _da_to_heatmap_payload(da, "Precision", "NO2", "mol/cm2")

        reference = payload_for(np.float64)["statistics"]
        actual = payload_for(np.float32)["statistics"]

        # The area weighting must be reproduced too, so compute the expected
        # mean the way the tool documents it rather than as a flat average.
        weights = np.clip(np.cos(np.deg2rad(lats)), 0.0, None)
        expected_mean = float((values * weights[:, None]).sum() / (weights.sum() * lons.size))

        self.assertAlmostEqual(reference["mean"] / expected_mean, 1.0, places=6)
        self.assertAlmostEqual(actual["mean"] / expected_mean, 1.0, places=6)
        self.assertEqual(actual["count"], values.size)


if __name__ == "__main__":
    unittest.main()
