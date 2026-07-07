"""
tests/test_provenance_endpoint.py
===================================
T10: the provenance pane's HTTP surface — provenance, citations, methods
export, and NetCDF download, each scoped to the chart's owner exactly like
the existing /chart/{id}/export.* routes.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "provenance endpoint test dependencies are not installed",
)
class ProvenanceEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import api
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.toolset import load_earthdata_tools
        from config.settings import Settings
        from models.user import User
        from utils.streaming import current_user_id

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "kind": "observation",
                "events": [
                    {"stage": "routed", "at": "2026-07-01T00:00:00Z", "provider": "GES_DISC"},
                    {"stage": "materialized", "at": "2026-07-01T00:12:00Z"},
                ],
                "inputs": [
                    {"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"},
                    {"handle": "aoi_nj", "kind": "aoi", "description": "New Jersey"},
                ],
            }

        async def cite_dataset(dataset_handle, workspace_id):
            return {
                "dataset_handle": dataset_handle,
                "doi": "10.5067/TEMPO/NO2/L3",
                "citation": "NASA, TEMPO NO2 Tropospheric Column L3, doi:10.5067/TEMPO/NO2/L3",
            }

        self.fixture_path = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
        with open(self.fixture_path, "wb") as fixture:
            fixture.write(b"fake-netcdf-bytes")
        self.addCleanup(os.unlink, self.fixture_path)

        async def convert_format(handle, target_format, workspace_id):
            return {
                "handle": handle,
                "status": "ready",
                "storage_uri": f"file:///{self.fixture_path.replace(os.sep, '/')}",
                "media_type": "netcdf",
            }

        self.server = FakeEarthdataMCPServer(build_fake_mcp({
            "get_provenance": get_provenance,
            "cite_dataset": cite_dataset,
            "convert_format": convert_format,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        self.api.app.state.earthdata_mcp_tools = await load_earthdata_tools(settings, current_user_id)
        # Provenance/citations/methods read tools through
        # earthdata_mcp_manager (T17); export.nc stays on the legacy
        # app.state.earthdata_mcp_tools mirror (see test_export_netcdf_*
        # below), so both must be kept in sync in this setUp.
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(
            state="ready", tools=self.api.app.state.earthdata_mcp_tools,
        )

        self.user = User(
            id="user-1", username="tester", password_hash="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        )
        token, _ = self.api.create_access_token(self.user)
        self.auth_headers = {"Authorization": f"Bearer {token}"}

        self.chart_payload = {
            "chart_id": "chart-1",
            "title": "TEMPO NO2 over New Jersey",
            "user_id": self.user.id,
            "provenance": {
                "region_name": "New Jersey",
                "start_date": "2026-06-01T00:00:00",
                "end_date": "2026-06-30T00:00:00",
                "source_handles": ["obs_1"],
            },
            "metadata": {"source_handles": ["obs_1"]},
        }

    def _auth_patch(self):
        async def fake_get_user_by_id(user_id):
            return self.user if user_id == self.user.id else None

        async def fake_is_token_revoked(jti):
            return False

        return patch("services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_provenance_endpoint_returns_the_merged_lineage_for_the_owned_chart(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/provenance", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        handles = {node["handle"] for node in response.json()["nodes"]}
        self.assertEqual(handles, {"obs_1", "dataset_tempo_no2", "aoi_nj"})

    async def test_citations_endpoint_returns_the_deduplicated_dataset_citations(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/citations", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "citations": [{
                "dataset_handle": "dataset_tempo_no2",
                "doi": "10.5067/TEMPO/NO2/L3",
                "citation": "NASA, TEMPO NO2 Tropospheric Column L3, doi:10.5067/TEMPO/NO2/L3",
            }],
        })

    async def test_methods_endpoint_returns_downloadable_markdown_naming_the_real_dataset(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/methods.md", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/markdown; charset=utf-8")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn("TEMPO NO2 L3", response.text)
        self.assertIn("New Jersey", response.text)
        self.assertIn("10.5067/TEMPO/NO2/L3", response.text)

    async def test_export_netcdf_streams_the_converted_file(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/export.nc", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/x-netcdf")
        self.assertIn(".nc", response.headers["content-disposition"])
        self.assertEqual(response.content, b"fake-netcdf-bytes")

    async def test_export_netcdf_422s_when_the_mcp_cannot_convert_the_handle(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.toolset import load_earthdata_tools
        from config.settings import Settings
        from utils.streaming import current_user_id

        async def unsupported_convert_format(handle, target_format, workspace_id):
            return {"handle": handle, "status": "unsupported", "message": "NetCDF export is not available for this handle."}

        server = FakeEarthdataMCPServer(build_fake_mcp({"convert_format": unsupported_convert_format}))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        broken_tools = await load_earthdata_tools(settings, current_user_id)

        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        original_tools = self.api.app.state.earthdata_mcp_tools
        self.api.app.state.earthdata_mcp_tools = broken_tools
        try:
            with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
                async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.get("/chart/chart-1/export.nc", headers=self.auth_headers)
        finally:
            self.api.app.state.earthdata_mcp_tools = original_tools

        self.assertEqual(response.status_code, 422)
        self.assertIn("NetCDF export is not available", response.json()["detail"])

    async def test_provenance_endpoint_fails_honestly_when_the_mcp_is_not_ready(self):
        original_manager = self.api.app.state.earthdata_mcp_manager
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="unavailable", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", original_manager)

        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/provenance", headers=self.auth_headers)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["category"], "provider_unavailable")

    async def test_provenance_endpoint_404s_for_a_chart_owned_by_someone_else(self):
        async def fake_get_chart(chart_id):
            return {**self.chart_payload, "user_id": "someone-else"}

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/provenance", headers=self.auth_headers)

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
