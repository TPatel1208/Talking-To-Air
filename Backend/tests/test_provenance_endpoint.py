"""
tests/test_provenance_endpoint.py
===================================
T10: the provenance pane's HTTP surface — provenance, citations, methods
export, and NetCDF download, each scoped to the chart's owner exactly like
the existing /chart/{id}/export.* routes.
"""
import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "provenance endpoint test dependencies are not installed",
)
class ProvenanceEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.toolset import load_earthdata_tools
        from tta_backend.config.settings import Settings
        from tta_backend.models.user import User
        from tta_backend.utils.streaming import current_user_id

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()

        # Fixtures mirror the REAL MCP contract (src/earthdata_mcp/tools/
        # provenance.py): get_provenance returns {handle, ancestors:
        # [{handle, depth}], events: [{event_type, detail, created_at}]}
        # (events newest-first), and cite_dataset returns CMR's own record
        # ({handle, concept_id, doi, doi_authority, collection_citations,
        # reference_citation_count}). The previous imagined shape (inputs/
        # kind/stage/at, dataset_handle/citation) made methods.md KeyError
        # and citations silently empty against the live server (QA
        # 2026-07-17 blocker).
        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "ancestors": [
                    {"handle": "dataset_tempo_no2", "depth": 1},
                    {"handle": "aoi_nj", "depth": 1},
                ],
                "events": [
                    {"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:12:00Z"},
                    {"event_type": "routed", "detail": {"provider": "GES_DISC"}, "created_at": "2026-07-01T00:00:00Z"},
                ],
            }

        async def cite_dataset(dataset_handle, workspace_id):
            return {
                "handle": dataset_handle,
                "concept_id": "C2930725014-LARC_CLOUD",
                "doi": "10.5067/TEMPO/NO2/L3",
                "doi_authority": "https://doi.org",
                "collection_citations": [{
                    "Creator": "NASA/LARC/SD/ASDC",
                    "Title": "TEMPO NO2 tropospheric and stratospheric columns V03",
                    "Publisher": "NASA Atmospheric Science Data Center",
                    "ReleaseDate": "2024-01-01T00:00:00.000Z",
                }],
                "reference_citation_count": 12,
            }

        self.fixture_path = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
        with open(self.fixture_path, "wb") as fixture:
            fixture.write(b"fake-netcdf-bytes")
        self.addCleanup(os.unlink, self.fixture_path)

        # Real contract (T17 live-verified): convert_format takes
        # source_handle/output_format (not handle/target_format) and mints a
        # new cube_ handle — it never returns storage_uri itself. Resolving
        # bytes requires a follow-up export_result on that new handle, same
        # as any other materialized handle (services/open_handle.py).
        async def convert_format(source_handle, output_format, workspace_id):
            return {"handle": "cube_1", "status": "ready", "output_format": output_format, "operation": "convert_format"}

        async def export_result(handle, workspace_id):
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
            "export_result": export_result,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        self.api.app.state.earthdata_mcp_tools = await load_earthdata_tools(settings, current_user_id)
        # Provenance/citations/methods/export.nc all read tools through
        # earthdata_mcp_manager (T17/T37).
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(
            state="ready", tools=self.api.app.state.earthdata_mcp_tools,
        )
        # app.state is process-global — leave no manager behind, or a later
        # test file that expects "no manager => connecting" fails when this
        # module happens to run first.
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", None)

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

        return patch("tta_backend.services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("tta_backend.services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_provenance_endpoint_returns_the_merged_lineage_for_the_owned_chart(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/provenance", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        nodes = response.json()["nodes"]
        handles = {node["handle"] for node in nodes}
        self.assertEqual(handles, {"obs_1", "dataset_tempo_no2", "aoi_nj"})
        # Kinds are derived from the handle prefix — the real MCP response
        # carries no kind field, and downstream consumers (citations, the
        # methods text) key off it.
        kinds = {node["handle"]: node["kind"] for node in nodes}
        self.assertEqual(kinds["dataset_tempo_no2"], "dataset")
        self.assertEqual(kinds["aoi_nj"], "aoi")
        self.assertEqual(kinds["obs_1"], "observation")
        obs_node = next(node for node in nodes if node["handle"] == "obs_1")
        self.assertEqual(
            [event["event_type"] for event in obs_node["events"]],
            ["routed", "materialized"],
            "events render oldest-first even though the MCP answers newest-first",
        )

    async def test_citations_endpoint_returns_the_deduplicated_dataset_citations(self):
        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/citations", headers=self.auth_headers)

        self.assertEqual(response.status_code, 200)
        citations = response.json()["citations"]
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["handle"], "dataset_tempo_no2")
        self.assertEqual(citations[0]["doi"], "10.5067/TEMPO/NO2/L3")
        self.assertEqual(
            citations[0]["collection_citations"][0]["Title"],
            "TEMPO NO2 tropospheric and stratospheric columns V03",
        )

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
        self.assertIn("TEMPO NO2 tropospheric and stratospheric columns V03", response.text)
        self.assertIn("New Jersey", response.text)
        self.assertIn("10.5067/TEMPO/NO2/L3", response.text)
        self.assertIn("materialized (2026-07-01T00:12:00Z)", response.text)
        self.assertIn("- 2026-07-01", response.text, "retrieval date derives from the materialized event")

    async def test_methods_endpoint_answers_on_taxonomy_when_assembly_crashes(self):
        # QA 2026-07-17 blocker: a KeyError inside the methods assembly
        # escaped as a bare ExceptionGroup ("RuntimeError: No response
        # returned") — the endpoint must classify any assembly surprise as
        # a contract error and answer through the shared taxonomy handler.
        async def fake_get_chart(chart_id):
            return self.chart_payload

        def exploding_build(**kwargs):
            raise KeyError("stage")

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1], \
             patch.object(self.api.chart_service, "get_chart", fake_get_chart), \
             patch.object(self.api, "build_methods_markdown", exploding_build):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/chart/chart-1/methods.md", headers=self.auth_headers)

        self.assertEqual(response.status_code, 500)
        body = response.json()["error"]
        self.assertEqual(body["category"], "contract")
        self.assertNotIn("stage", body["message"], "raw exception text stays in the logs")

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
        from tta_backend.earthdata_mcp.toolset import load_earthdata_tools
        from tta_backend.config.settings import Settings
        from tta_backend.utils.streaming import current_user_id

        async def unsupported_convert_format(source_handle, output_format, workspace_id):
            return {"handle": source_handle, "status": "unsupported", "message": "NetCDF export is not available for this handle."}

        server = FakeEarthdataMCPServer(build_fake_mcp({"convert_format": unsupported_convert_format}))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        broken_tools = await load_earthdata_tools(settings, current_user_id)

        async def fake_get_chart(chart_id):
            return self.chart_payload

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        # T37: export.nc reads tools through the readiness gate (manager.tools).
        original_manager = self.api.app.state.earthdata_mcp_manager
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="ready", tools=broken_tools)
        try:
            with auth_patches[0], auth_patches[1], patch.object(self.api.chart_service, "get_chart", fake_get_chart):
                async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.get("/chart/chart-1/export.nc", headers=self.auth_headers)
        finally:
            self.api.app.state.earthdata_mcp_manager = original_manager

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
