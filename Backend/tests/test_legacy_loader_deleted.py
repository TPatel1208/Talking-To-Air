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
    # The chat system the agent harness replaced.
    "tta_backend.agents",
    "tta_backend.tools",
    "tta_backend.preprocessing",
    "tta_backend.config.supervisor_prompt",
    "tta_backend.config.earthdata_agent_prompt",
    "tta_backend.config.ground_sensor_agent_prompt",
    "tta_backend.services.chat_stream_service",
    "tta_backend.services.dispatch_executor",
    "tta_backend.services.subagent_dispatch",
    "tta_backend.services.intent_router",
    "tta_backend.services.history_service",
    "tta_backend.models.agent_request",
    "tta_backend.models.agent_result",
    "tta_backend.earthdata_mcp.toolset",
)


class LegacyLoaderDeletedTests(unittest.TestCase):
    def test_legacy_loader_and_cache_modules_no_longer_import(self):
        for module_name in DELETED_MODULES:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
