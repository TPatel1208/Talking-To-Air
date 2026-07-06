"""
tests/test_data_download_service.py
=====================================
T10: the data-download endpoints' backend seam. Calls the MCP's
``convert_format`` tool to materialize a handle in a downloadable format
(e.g. NetCDF), then streams the converted file's bytes directly — no
parallel download system, just the same file:// export contract
``open_handle``/``export_result`` already use.
"""
import importlib.util
import os
import sys
import tempfile
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class IterConvertedChunksTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_streams_the_converted_files_bytes_with_the_mcps_media_type(self):
        from services.data_download_service import iter_converted_chunks

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fixture:
            fixture.write(b"fake-netcdf-bytes" * 100)
            fixture_path = fixture.name
        self.addCleanup(os.unlink, fixture_path)

        async def convert_format(handle, target_format, workspace_id):
            return {
                "handle": handle,
                "status": "ready",
                "storage_uri": f"file:///{fixture_path.replace(os.sep, '/')}",
                "media_type": "netcdf",
            }

        tools = await self._tools({"convert_format": convert_format})

        chunks = [chunk async for chunk in iter_converted_chunks("obs_1", "netcdf", tools)]
        content = b"".join(chunks)

        self.assertEqual(content, b"fake-netcdf-bytes" * 100)

    async def test_raises_with_the_mcps_message_when_conversion_is_not_ready(self):
        from services.data_download_service import DataDownloadError, iter_converted_chunks

        async def convert_format(handle, target_format, workspace_id):
            return {"handle": handle, "status": "unsupported", "message": "NetCDF export is not available for this handle."}

        tools = await self._tools({"convert_format": convert_format})

        with self.assertRaisesRegex(DataDownloadError, "NetCDF export is not available"):
            async for _ in iter_converted_chunks("obs_1", "netcdf", tools):
                pass


if __name__ == "__main__":
    unittest.main()
