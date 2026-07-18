"""T45: bundle_extract_cache_bytes gauge -- extract_cache_size_bytes() sums
the on-disk size of the TTL-pruned bundle-extract cache
(services/open_handle.py's _extract_bundle_cached), so an unexpectedly large
cache is visible and distinguishable from Python heap growth when chasing a
memory incident.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "xarray"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "extract-cache-size test dependencies are not installed",
)
class ExtractCacheSizeBytesTests(unittest.TestCase):
    def test_sums_the_bytes_of_files_under_the_extract_cache_directory(self):
        from services.open_handle import _EXTRACT_CACHE_DIR_NAME, extract_cache_size_bytes

        root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
        os.makedirs(root, exist_ok=True)
        probe_dir = tempfile.mkdtemp(dir=root)
        self.addCleanup(shutil.rmtree, probe_dir, ignore_errors=True)
        with open(os.path.join(probe_dir, "a.nc"), "wb") as f:
            f.write(b"x" * 100)
        with open(os.path.join(probe_dir, "b.nc"), "wb") as f:
            f.write(b"y" * 250)

        # Other tests/processes may share this process-wide cache directory,
        # so assert a lower bound (our own bytes are always counted) rather
        # than an exact total.
        self.assertGreaterEqual(extract_cache_size_bytes(), 350)

    def test_returns_zero_when_the_cache_directory_does_not_exist(self):
        from services.open_handle import extract_cache_size_bytes

        import services.open_handle as open_handle_module

        original = open_handle_module._EXTRACT_CACHE_DIR_NAME
        open_handle_module._EXTRACT_CACHE_DIR_NAME = "tta_bundle_extract_does_not_exist_probe"
        try:
            self.assertEqual(extract_cache_size_bytes(), 0)
        finally:
            open_handle_module._EXTRACT_CACHE_DIR_NAME = original


if __name__ == "__main__":
    unittest.main()
