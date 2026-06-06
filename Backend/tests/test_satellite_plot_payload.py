import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REQUIRED_MODULES = ["affine", "cartopy", "langchain", "numpy", "rasterio", "shapely", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite plotting dependencies are not installed",
)
class SatellitePlotPayloadTests(unittest.TestCase):
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

    def test_reproducibility_metadata_uses_fetch_params(self):
        from tools.satellite_tools.plot_tools import _attach_reproducibility

        data_dict = {
            "variable": "TEMPO_NO2",
            "source": "NASA Harmony — TEMPO tropospheric NO2 vertical column",
            "fetch_params": {
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T23:59:59Z",
                "bbox": [-75.0, 39.0, -73.0, 41.0],
            },
        }

        payload = _attach_reproducibility(
            {"type": "heatmap", "title": "TEMPO over NJ"},
            data_dict,
            "New Jersey",
            "single snapshot",
            {"chart_type": "heatmap"},
        )

        self.assertEqual(payload["provenance"]["dataset"], "TEMPO")
        self.assertEqual(payload["provenance"]["variable"], "TEMPO_NO2")
        self.assertEqual(payload["provenance"]["region_name"], "New Jersey")
        self.assertEqual(payload["query"]["dataset"], "TEMPO_NO2")
        self.assertEqual(payload["query"]["bbox"], [-75.0, 39.0, -73.0, 41.0])
        self.assertEqual(payload["query"]["aggregation"], "single snapshot")


if __name__ == "__main__":
    unittest.main()
