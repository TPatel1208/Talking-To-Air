import importlib.util
import os
import sys
import unittest
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

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

    async def lookup(self, **kwargs):
        return self.row


class FakeCache:
    def __init__(self, dataset):
        self.dataset = dataset
        self.lookups = 0

    async def lookup(self, **kwargs):
        self.lookups += 1
        return self.dataset

    async def store(self, *args, **kwargs):
        return None


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class CacheManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bbox_normalization_accepts_string_and_nested_values(self):
        from preprocessing.cache_manager import make_group_key
        from utils.geo_utils import normalise_bbox

        self.assertEqual(normalise_bbox("-74,40,-73,41"), (-74.0, 40.0, -73.0, 41.0))
        self.assertEqual(normalise_bbox([("-74", "40", "-73", "41")]), (-74.0, 40.0, -73.0, 41.0))
        self.assertEqual(
            make_group_key("C1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", ["-74", "40", "-73", "41"]),
            "C1/2024-01-01T00:00:00Z_2024-01-02T00:00:00Z/-74.0_40.0_-73.0_41.0",
        )

    async def test_zarr_cache_hit_reads_dataset(self):
        import xarray as xr
        from preprocessing.cache_manager import CacheManager, make_group_key

        temporal = ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
        group_key = make_group_key("C1", temporal[0], temporal[1], ())
        dataset = xr.Dataset()
        zarr = FakeZarrRepository({group_key: dataset})

        result = await CacheManager(zarr).lookup("C1", temporal)

        self.assertIs(result, dataset)
        self.assertEqual(zarr.reads, [group_key])

    async def test_cache_miss_returns_none(self):
        from preprocessing.cache_manager import CacheManager

        result = await CacheManager(FakeZarrRepository()).lookup(
            "C1",
            ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
        )

        self.assertIsNone(result)

    async def test_index_hit_takes_precedence_over_zarr_fallback(self):
        import xarray as xr
        from preprocessing.cache_manager import CacheManager

        dataset = xr.Dataset()
        zarr = FakeZarrRepository({"stored": dataset})
        index = FakeIndexRepository({"cache_path": "cache.zarr", "group_key": "stored"})

        result = await CacheManager(zarr, index).lookup(
            "C1",
            ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
        )

        self.assertIs(result, dataset)
        self.assertEqual(zarr.reads, ["stored"])

    @unittest.skipIf(importlib.util.find_spec("netCDF4") is None, "netCDF4 is not installed")
    async def test_data_loader_reuses_in_memory_cache_for_identical_lookup(self):
        import xarray as xr
        from preprocessing.data_loader import DataLoader

        temporal = ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
        dataset = xr.Dataset()
        loader = DataLoader.__new__(DataLoader)
        loader._memory_cache = OrderedDict()
        loader._memory_cache_sizes = {}
        loader._memory_cache_bytes = 0
        loader._cache = FakeCache(dataset)

        kwargs = {
            "collection_id": "C1",
            "temporal": temporal,
            "bounding_box": (-74, 40, -73, 41),
            "max_results": 1,
        }

        first = await loader.download_dataset_harmony_async(**kwargs)
        second = await loader.download_dataset_harmony_async(**kwargs)

        self.assertIs(first, dataset)
        self.assertIs(second, dataset)
        self.assertEqual(loader._cache.lookups, 1)

    async def test_data_loader_memory_cache_concurrent_remember_is_consistent(self):
        import xarray as xr
        from preprocessing.data_loader import DataLoader, _MEMORY_CACHE_MAX_ITEMS

        loader = DataLoader.__new__(DataLoader)
        loader._memory_cache = OrderedDict()
        loader._memory_cache_sizes = {}
        loader._memory_cache_bytes = 0
        loader._memory_cache_lock = threading.Lock()

        datasets = [xr.Dataset(attrs={"idx": idx}) for idx in range(10)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(loader._remember_dataset, f"key-{idx}", dataset)
                for idx, dataset in enumerate(datasets)
            ]
            for future in futures:
                future.result()

        expected_items = min(len(datasets), _MEMORY_CACHE_MAX_ITEMS)
        self.assertEqual(len(loader._memory_cache), expected_items)
        self.assertEqual(len(set(loader._memory_cache.keys())), expected_items)
        self.assertTrue(all(key.startswith("key-") for key in loader._memory_cache))

    async def test_data_loader_memory_cache_evicts_by_bytes(self):
        from unittest.mock import patch
        from preprocessing.data_loader import DataLoader

        class SizedDataset:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        loader = DataLoader.__new__(DataLoader)
        loader._memory_cache = OrderedDict()
        loader._memory_cache_sizes = {}
        loader._memory_cache_bytes = 0
        loader._memory_cache_lock = threading.Lock()

        with patch("preprocessing.data_loader._memory_cache_max_bytes", return_value=10):
            loader._remember_dataset("small-1", SizedDataset(4))
            loader._remember_dataset("small-2", SizedDataset(4))
            loader._remember_dataset("large", SizedDataset(7))

        self.assertEqual(list(loader._memory_cache.keys()), ["large"])
        self.assertEqual(loader._memory_cache_bytes, 7)


if __name__ == "__main__":
    unittest.main()
