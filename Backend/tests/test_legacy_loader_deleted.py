import importlib
import unittest

DELETED_MODULES = (
    "tta_backend.tools.satellite_tools.harmony_api",
    "tta_backend.tools.satellite_tools.models",
    "tta_backend.preprocessing.data_loader",
    "tta_backend.preprocessing.dataset_parser",
    "tta_backend.preprocessing.cache_manager",
    "tta_backend.preprocessing.cache_index",
    "tta_backend.preprocessing.zarr_normalization",
    "tta_backend.repositories.cache_index_repository",
    "tta_backend.repositories.zarr_repository",
    "tta_backend.services.s3_fetch_service",
    "tta_backend.services.opendap_fetch_service",
    "tta_backend.services.async_harmony_service",
    "tta_backend.utils.earthaccess_client",
    "tta_backend.utils.data_utils",
)


class LegacyLoaderDeletedTests(unittest.TestCase):
    def test_legacy_loader_and_cache_modules_no_longer_import(self):
        for module_name in DELETED_MODULES:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
