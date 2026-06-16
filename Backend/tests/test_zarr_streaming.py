import importlib.util
import os
import sys
import tempfile
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["xarray", "zarr", "dask"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "zarr streaming dependencies are not installed",
)
class ZarrStreamingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr

    def _dataset(self, start, *, coord_dtype="float32", value_dtype="float32"):
        lat = self.np.array([40.0, 41.0], dtype=coord_dtype)
        lon = self.np.array([-74.0, -73.0], dtype=coord_dtype)
        values = self.np.full((1, 2, 2), start, dtype=value_dtype)
        return self.xr.Dataset(
            data_vars={
                "no2": (("time", "lat", "lon"), values),
            },
            coords={
                "time": self.np.array([f"2024-01-{start + 1:02d}"], dtype="datetime64[ns]"),
                "lat": lat,
                "lon": lon,
            },
        )

    def test_normalization_casts_to_template_and_reuses_chunks(self):
        from preprocessing.zarr_normalization import normalize_for_zarr_append

        first = normalize_for_zarr_append([self._dataset(0)])
        second = normalize_for_zarr_append(
            [self._dataset(1, coord_dtype="float64", value_dtype="float64")],
            template=first,
        )

        self.assertEqual(second["lat"].dtype, first["lat"].dtype)
        self.assertEqual(second["lon"].dtype, first["lon"].dtype)
        self.assertEqual(second["no2"].dtype, first["no2"].dtype)
        self.assertEqual(second["no2"].encoding["chunks"], first["no2"].encoding["chunks"])

    def test_normalization_rejects_non_append_dimension_drift(self):
        from preprocessing.zarr_normalization import (
            ZarrNormalizationError,
            normalize_for_zarr_append,
        )

        first = normalize_for_zarr_append([self._dataset(0)])
        incompatible = self.xr.Dataset(
            data_vars={
                "no2": (("time", "lat", "lon"), self.np.ones((1, 3, 2), dtype="float32")),
            },
            coords={
                "time": self.np.array(["2024-01-02"], dtype="datetime64[ns]"),
                "lat": self.np.array([40.0, 41.0, 42.0], dtype="float32"),
                "lon": self.np.array([-74.0, -73.0], dtype="float32"),
            },
        )

        with self.assertRaisesRegex(ZarrNormalizationError, "Non-append dimensions"):
            normalize_for_zarr_append([incompatible], template=first)

    def test_zarr_repository_appends_windows_and_returns_lazy_dataset(self):
        from preprocessing.zarr_normalization import normalize_for_zarr_append
        from repositories.zarr_repository import ZarrRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ZarrRepository(os.path.join(tmpdir, "cache.zarr"))
            group_key = "C1/start_end/global"
            first = normalize_for_zarr_append([self._dataset(0), self._dataset(1)])
            second = normalize_for_zarr_append([self._dataset(2)], template=first)

            repo.append_window(first, group_key, first_write=True)
            repo.append_window(second, group_key, first_write=False)

            result = repo.read(group_key)

            self.assertEqual(result.sizes["time"], 3)
            self.assertIsNotNone(result["no2"].chunks)
            self.assertEqual(
                [str(value)[:10] for value in result["time"].values],
                ["2024-01-01", "2024-01-02", "2024-01-03"],
            )


if __name__ == "__main__":
    unittest.main()
