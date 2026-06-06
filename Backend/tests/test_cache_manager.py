import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class FakeZarrRepository:
    def __init__(self, datasets=None):
        self.datasets = datasets or {}
        self.reads = []

    def exists(self, group_key):
        return group_key in self.datasets

    def read(self, group_key):
        self.reads.append(group_key)
        return self.datasets[group_key]

    def write(self, ds, group_key):
        self.datasets[group_key] = ds


class FakeIndexRepository:
    def __init__(self, row=None):
        self.row = row

    def lookup(self, **kwargs):
        return self.row


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CacheManagerTests(unittest.TestCase):
    def test_bbox_normalization_accepts_string_and_nested_values(self):
        from preprocessing.cache_manager import _normalise_bbox, make_group_key

        self.assertEqual(_normalise_bbox("-74,40,-73,41"), (-74.0, 40.0, -73.0, 41.0))
        self.assertEqual(_normalise_bbox([("-74", "40", "-73", "41")]), (-74.0, 40.0, -73.0, 41.0))
        self.assertEqual(
            make_group_key("C1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", ["-74", "40", "-73", "41"]),
            "C1/2024-01-01T00:00:00Z_2024-01-02T00:00:00Z/-74.0_40.0_-73.0_41.0",
        )

    def test_zarr_cache_hit_reads_dataset(self):
        import xarray as xr
        from preprocessing.cache_manager import CacheManager, make_group_key

        temporal = ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
        group_key = make_group_key("C1", temporal[0], temporal[1], ())
        dataset = xr.Dataset()
        zarr = FakeZarrRepository({group_key: dataset})

        result = CacheManager(zarr).lookup("C1", temporal)

        self.assertIs(result, dataset)
        self.assertEqual(zarr.reads, [group_key])

    def test_cache_miss_returns_none(self):
        from preprocessing.cache_manager import CacheManager

        result = CacheManager(FakeZarrRepository()).lookup(
            "C1",
            ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
        )

        self.assertIsNone(result)

    def test_index_hit_takes_precedence_over_zarr_fallback(self):
        import xarray as xr
        from preprocessing.cache_manager import CacheManager

        dataset = xr.Dataset()
        zarr = FakeZarrRepository({"stored": dataset})
        index = FakeIndexRepository({"cache_path": "cache.zarr", "group_key": "stored"})

        result = CacheManager(zarr, index).lookup(
            "C1",
            ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
        )

        self.assertIs(result, dataset)
        self.assertEqual(zarr.reads, ["stored"])


if __name__ == "__main__":
    unittest.main()
