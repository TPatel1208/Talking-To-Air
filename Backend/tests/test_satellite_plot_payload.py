import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

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
        from utils.plotting import mask_data_by_geometry

        da = xr.DataArray(
            np.ones((1, 5, 4)),
            dims=("time", "Longitude", "Latitude"),
            coords={
                "time": ["2024-01-01"],
                "Longitude": np.linspace(-100.0, -96.0, 5),
                "Latitude": np.linspace(30.0, 33.0, 4),
            },
        )

        masked = mask_data_by_geometry(da, box(-99.5, 30.5, -96.5, 32.5))

        self.assertEqual(masked.dims, ("time", "Longitude", "Latitude"))
        self.assertTrue(np.isfinite(masked.values).any())
        self.assertTrue(np.isnan(masked.values).any())

    def test_payload_preserves_sparse_valid_points(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.plot_tools import _MAX_GRID_CELLS, _da_to_heatmap_payload

        arr = np.full((120, 120), np.nan)
        arr[3, 5] = 1.25
        arr[90, 95] = 2.5
        da = xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 120), "lon": np.linspace(-100, -90, 120)},
        )

        payload = _da_to_heatmap_payload(da, "Sparse", "NO2", "mol/cm2")

        self.assertEqual(payload["points"]["values"], [1.25, 2.5])
        self.assertEqual(len(payload["points"]["values"]), 2)
        self.assertLessEqual(len(payload["points"]["values"]), _MAX_GRID_CELLS)

    def test_payload_normalizes_longitudes_and_sanitizes_values(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.array([[np.inf, 4.0, np.nan]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0], "lon": [350.0, 355.0, 5.0]},
        )

        payload = _da_to_heatmap_payload(da, "Wrapped", "NO2", "mol/cm2")

        self.assertEqual(payload["points"]["values"], [4.0])
        self.assertEqual(payload["lons"], [-10.0, -5.0, 5.0])
        self.assertLess(payload["vmin"], 4.0)
        self.assertGreater(payload["vmax"], 4.0)

    def test_payload_attaches_the_resolved_colormap(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from utils.colormaps import resolve

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
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from utils.colormaps import resolve

        da = xr.DataArray(
            np.array([[-1.0, 2.0], [3.0, -4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

        payload = _da_to_heatmap_payload(da, "Diff", "NO2", "mol/m^2", diverging=True)

        self.assertEqual(payload["colormap"]["name"], resolve("NO2", diverging=True).name)
        self.assertEqual(payload["colormap"]["name"], "RdBu_r")

    def test_payload_attaches_overlay_bounds_from_the_full_resolution_extent(self):
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload

        da = xr.DataArray(
            np.ones((3, 4)),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(10, 20, 3), "lon": np.linspace(-100, -90, 4)},
        )

        payload = _da_to_heatmap_payload(da, "Extent", "NO2", "mol/m^2")

        self.assertEqual(payload["overlay"]["bounds"], [-100.0, 10.0, -90.0, 20.0])
        self.assertNotIn("_path", payload["overlay"])

    def test_value_range_override_drives_both_reported_bounds_and_overlay_colorization(self):
        import io
        import numpy as np
        import matplotlib.image as mpimg
        import xarray as xr
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload
        from utils.colormaps import resolve

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

    def test_render_overlay_true_persists_a_png_and_records_its_path(self):
        import os
        import numpy as np
        import xarray as xr
        from tools.satellite_tools.plot_tools import _da_to_heatmap_payload

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
        from tools.satellite_tools.plot_tools import _attach_reproducibility

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
        from tools.satellite_tools.plot_tools import _attach_reproducibility

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
        from tools.satellite_tools.plot_tools import _attach_reproducibility

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

    def test_provenance_missing_dataset_facts_render_as_empty_not_error(self):
        """No col_info at all (unregistered collection) must not raise --
        every field the frontend expects is still present, just empty, so
        the UI can render 'Not available' rather than crash on a missing key."""
        import xarray as xr
        from tools.satellite_tools.plot_tools import _attach_reproducibility

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
        from tools.satellite_tools.plot_tools import _attach_reproducibility, _save_chart

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
        from tools.satellite_tools.plot_tools import (
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

        with patch("tools.satellite_tools.plot_tools.emit_chart", fake_emit_chart):
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
        from tools.satellite_tools.plot_tools import _attach_reproducibility, _save_chart

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
        from tools.satellite_tools.plot_tools import _save_chart

        payload = {"type": "heatmap", "title": "Has overlay", "overlay": {"bounds": [0, 0, 1, 1], "_path": "/tmp/x.png"}}

        result = json.loads(_save_chart(payload, "n/a"))

        self.assertEqual(payload["overlay"]["url"], f"/chart/{payload['chart_id']}/overlay.png")
        # The internal filesystem path never reaches the model-facing summary.
        self.assertNotIn("overlay", result)

    def test_save_chart_leaves_overlay_url_unset_when_render_failed(self):
        from tools.satellite_tools.plot_tools import _save_chart

        payload = {"type": "heatmap", "title": "No overlay", "overlay": {"bounds": [0, 0, 1, 1]}}

        _save_chart(payload, "n/a")

        self.assertNotIn("url", payload["overlay"])

    def test_save_chart_wires_a_per_panel_overlay_url_for_heatmap_multi(self):
        from tools.satellite_tools.plot_tools import _save_chart

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
        from tools.satellite_tools.plot_tools import _save_chart

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
        from tools.satellite_tools.plot_tools import _save_chart

        result = json.loads(_save_chart({"type": "error"}, "n/a"))

        self.assertNotIn("_artifact_refs", result)


if __name__ == "__main__":
    unittest.main()
