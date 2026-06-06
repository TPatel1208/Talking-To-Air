import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REQUIRED_MODULES = ["xarray", "earthaccess", "harmony_py"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "routing dependencies are not installed",
)
class DataRoutingTests(unittest.TestCase):
    def setUp(self):
        import xarray as xr
        from preprocessing.data_loader import DataLoader

        self.xr = xr
        self.loader = object.__new__(DataLoader)

    def test_explicit_harmony_mode_routes_to_harmony_only(self):
        dataset = self.xr.Dataset()

        with patch.object(self.loader, "_fetch_harmony_fallback", return_value=dataset) as harmony:
            result = self.loader._route(
                mode="harmony",
                provider="GES_DISC",
                col=None,
                collection_id="C1-GES_DISC",
                temporal=("start", "end"),
                bounding_box=None,
                variables=["NO2"],
                max_results=1,
                output_format="application/x-netcdf4",
            )

        self.assertIs(result, dataset)
        harmony.assert_called_once()

    def test_explicit_s3_mode_routes_to_s3(self):
        dataset = self.xr.Dataset()
        col = SimpleNamespace(groups=["product"])

        with patch.object(self.loader, "_fetch_s3", return_value=dataset) as s3:
            result = self.loader._route(
                mode="s3",
                provider="LARC_CLOUD",
                col=col,
                collection_id="C1-LARC_CLOUD",
                temporal=("start", "end"),
                bounding_box=(-1, -1, 1, 1),
                variables=None,
                max_results=1,
                output_format="application/x-netcdf4",
            )

        self.assertIs(result, dataset)
        s3.assert_called_once_with("C1-LARC_CLOUD", ("start", "end"), (-1, -1, 1, 1), "product", 1)

    def test_auto_mode_falls_back_to_opendap_for_ges_disc(self):
        dataset = self.xr.Dataset()

        with patch.object(self.loader, "_fetch_harmony_fallback", side_effect=RuntimeError("harmony down")), \
             patch.object(self.loader, "_fetch_opendap", return_value=dataset) as opendap:
            result = self.loader._route(
                mode="auto",
                provider="GES_DISC",
                col=SimpleNamespace(supports_variable_subsetting=True, variables=["NO2"], groups=[]),
                collection_id="C1-GES_DISC",
                temporal=("start", "end"),
                bounding_box=None,
                variables=None,
                max_results=1,
                output_format="application/x-netcdf4",
            )

        self.assertIs(result, dataset)
        opendap.assert_called_once_with("C1-GES_DISC", ("start", "end"), None, None, 1)

    def test_auto_mode_falls_back_to_s3_for_larc_cloud(self):
        dataset = self.xr.Dataset()
        col = SimpleNamespace(supports_variable_subsetting=False, variables=[], groups=["product"])

        with patch.object(self.loader, "_fetch_harmony_fallback", side_effect=RuntimeError("harmony down")), \
             patch.object(self.loader, "_fetch_s3", return_value=dataset) as s3:
            result = self.loader._route(
                mode="auto",
                provider="LARC_CLOUD",
                col=col,
                collection_id="C1-LARC_CLOUD",
                temporal=("start", "end"),
                bounding_box=None,
                variables=["NO2"],
                max_results=1,
                output_format="application/x-netcdf4",
            )

        self.assertIs(result, dataset)
        s3.assert_called_once_with("C1-LARC_CLOUD", ("start", "end"), None, "product", 1)


if __name__ == "__main__":
    unittest.main()