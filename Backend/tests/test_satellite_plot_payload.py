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

    def test_save_chart_omits_artifact_refs_for_an_unmapped_render_type(self):
        import json
        from tools.satellite_tools.plot_tools import _save_chart

        result = json.loads(_save_chart({"type": "error"}, "n/a"))

        self.assertNotIn("_artifact_refs", result)


if __name__ == "__main__":
    unittest.main()
