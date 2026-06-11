import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REQUIRED_MODULES = ["xarray", "harmony"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "routing dependencies are not installed",
)
class DataRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import xarray as xr
        from preprocessing.data_loader import DataLoader

        self.xr = xr
        self.loader = object.__new__(DataLoader)

    async def test_explicit_harmony_mode_routes_to_harmony_only(self):
        dataset = self.xr.Dataset()

        async def fake_harmony(*args, **kwargs):
            return dataset

        with patch.object(self.loader, "_fetch_harmony_fallback", side_effect=fake_harmony) as harmony:
            result = await self.loader._route(
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

    async def test_explicit_s3_mode_routes_to_s3(self):
        dataset = self.xr.Dataset()
        col = SimpleNamespace(groups=["product"])

        with patch.object(self.loader, "_fetch_s3", return_value=dataset) as s3:
            result = await self.loader._route(
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

    async def test_auto_mode_falls_back_to_opendap_for_ges_disc(self):
        dataset = self.xr.Dataset()

        async def fake_harmony(*args, **kwargs):
            raise RuntimeError("harmony down")

        with patch.object(self.loader, "_fetch_harmony_fallback", side_effect=fake_harmony), \
             patch.object(self.loader, "_fetch_opendap", return_value=dataset) as opendap:
            result = await self.loader._route(
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

    async def test_auto_mode_falls_back_to_s3_for_larc_cloud(self):
        dataset = self.xr.Dataset()
        col = SimpleNamespace(supports_variable_subsetting=False, variables=[], groups=["product"])

        async def fake_harmony(*args, **kwargs):
            raise RuntimeError("harmony down")

        with patch.object(self.loader, "_fetch_harmony_fallback", side_effect=fake_harmony), \
             patch.object(self.loader, "_fetch_s3", return_value=dataset) as s3:
            result = await self.loader._route(
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

    async def test_auto_mode_does_not_fallback_for_harmony_operational_errors(self):
        from services.async_harmony_service import HarmonyAuthenticationError

        async def fake_harmony(*args, **kwargs):
            raise HarmonyAuthenticationError("auth expired")

        with patch.object(
            self.loader,
            "_fetch_harmony_fallback",
            side_effect=fake_harmony,
        ), patch.object(self.loader, "_fetch_opendap") as opendap:
            with self.assertRaises(HarmonyAuthenticationError):
                await self.loader._route(
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

        opendap.assert_not_called()

    async def test_harmony_fallback_limits_parallel_granule_parsing(self):
        files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx in range(6):
                path = Path(tmpdir) / f"granule-{idx}.nc"
                path.write_text("placeholder")
                files.append(path)

            self.loader._harmony_service = SimpleNamespace(
                submit_and_download=AsyncMock(return_value=files)
            )
            self.loader._registry_by_id = {"C1-GES_DISC": SimpleNamespace(cadence="daily")}
            self.loader._get_granule_times = lambda *args, **kwargs: {}

            active = 0
            max_active = 0
            lock = threading.Lock()

            def parse_granule(path, granule_times):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.05)
                    idx = int(Path(path).stem.split("-")[-1])
                    return self.xr.Dataset(coords={"time": [idx]})
                finally:
                    with lock:
                        active -= 1

            self.loader._parser = SimpleNamespace(parse_granule=parse_granule)

            with patch("preprocessing.data_loader.get_settings", return_value=SimpleNamespace(granule_parse_max_concurrency=2)):
                result = await self.loader._fetch_harmony_fallback(
                    "C1-GES_DISC",
                    ("start", "end"),
                    None,
                    None,
                    6,
                    "application/x-netcdf4",
                )

            self.assertEqual(max_active, 2)
            self.assertEqual(list(result["time"].values), [0, 1, 2, 3, 4, 5])
            self.assertTrue(all(not path.exists() for path in files))


if __name__ == "__main__":
    unittest.main()
