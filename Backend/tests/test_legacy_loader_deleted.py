import importlib
import unittest

DELETED_MODULES = (
    "tools.satellite_tools.harmony_api",
    "tools.satellite_tools.models",
    "preprocessing.data_loader",
    "preprocessing.dataset_parser",
    "preprocessing.cache_manager",
    "preprocessing.cache_index",
    "preprocessing.zarr_normalization",
    "repositories.cache_index_repository",
    "repositories.zarr_repository",
    "services.s3_fetch_service",
    "services.opendap_fetch_service",
    "services.async_harmony_service",
    "utils.earthaccess_client",
    "utils.data_utils",
)


class LegacyLoaderDeletedTests(unittest.TestCase):
    def test_legacy_loader_and_cache_modules_no_longer_import(self):
        for module_name in DELETED_MODULES:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
