import importlib.util
import os
import sys
import unittest
from datetime import datetime, timezone

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "discovery endpoint test dependencies are not installed",
)
class DiscoveryEndpointTests(unittest.IsolatedAsyncioTestCase):
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

        self.search_calls = []
        self.describe_calls = []
        self.preview_calls = []
        self.coverage_calls = []
        self.aoi_calls = []

        async def search_datasets(query, filters, workspace_id):
            self.search_calls.append((query, filters, workspace_id))
            return {
                "results": [
                    {
                        "dataset_handle": "dataset_smap_l3",
                        "summary": "Soil moisture, L3 daily composite",
                        "variables": ["soil_moisture"],
                        "temporal_extent": "2015-04-01/present",
                        "provider": "NSIDC",
                    }
                ]
            }

        async def describe_dataset(dataset_handle, detail, workspace_id):
            self.describe_calls.append((dataset_handle, detail, workspace_id))
            return {
                "dataset_handle": dataset_handle,
                "summary": "Soil moisture, L3 daily composite",
                "variables": ["soil_moisture"],
                "temporal_extent": "2015-04-01/present",
                "provider": "NSIDC",
            }

        async def define_area_of_interest(location, workspace_id):
            self.aoi_calls.append((location, workspace_id))
            return {"handle": f"aoi_{location.lower().replace(' ', '_')}"}

        async def preview_dataset(dataset_handle, aoi_handle, time_range, layer, workspace_id):
            self.preview_calls.append((dataset_handle, aoi_handle, time_range, layer, workspace_id))
            if dataset_handle == "dataset_no_gibs":
                return {
                    "has_gibs_layer": False,
                    "message": "No GIBS browse layer is available for this dataset.",
                }
            return {
                "has_gibs_layer": True,
                "gibs_url": "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2026-07-01/250m/6/12/34.jpg",
            }

        async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id):
            self.coverage_calls.append((dataset_handle, aoi_handle, time_range, workspace_id))
            return {"has_data": True, "granule_count": 12}

        self.server = FakeEarthdataMCPServer(build_fake_mcp({
            "search_datasets": search_datasets,
            "describe_dataset": describe_dataset,
            "define_area_of_interest": define_area_of_interest,
            "preview_dataset": preview_dataset,
            "check_coverage": check_coverage,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        self.api.app.state.earthdata_mcp_tools = await load_earthdata_tools(settings, current_user_id)

        self.user1 = User(
            id="user-1", username="one", password_hash="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        )
        self.user2 = User(
            id="user-2", username="two", password_hash="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        )
        token1, _ = self.api.create_access_token(self.user1)
        token2, _ = self.api.create_access_token(self.user2)
        self.auth_headers1 = {"Authorization": f"Bearer {token1}"}
        self.auth_headers2 = {"Authorization": f"Bearer {token2}"}

    def _auth_patch(self):
        from unittest.mock import patch
        users = {self.user1.id: self.user1, self.user2.id: self.user2}

        async def fake_get_user_by_id(user_id):
            return users.get(user_id)

        async def fake_is_token_revoked(jti):
            return False

        return patch("services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_search_proxies_the_mcp_scoped_to_the_caller(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/discovery/search", json={"query": "soil moisture"}, headers=self.auth_headers1,
                )

        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(results[0]["dataset_handle"], "dataset_smap_l3")
        self.assertEqual(results[0]["provider"], "NSIDC")
        self.assertEqual(self.search_calls, [("soil moisture", None, "user-user-1")])

    async def test_describe_proxies_the_mcp_scoped_to_the_caller(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get(
                    "/discovery/dataset/dataset_smap_l3", headers=self.auth_headers2,
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["dataset_handle"], "dataset_smap_l3")
        self.assertEqual(body["variables"], ["soil_moisture"])
        self.assertEqual(self.describe_calls, [("dataset_smap_l3", False, "user-user-2")])

    async def test_coverage_resolves_the_aoi_then_checks_coverage_scoped_to_the_caller(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/discovery/dataset/dataset_smap_l3/coverage",
                    json={"location": "Raritan basin", "time_range": "2026-06-01/2026-06-30"},
                    headers=self.auth_headers1,
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"has_data": True, "granule_count": 12})
        self.assertEqual(self.aoi_calls, [("Raritan basin", "user-user-1")])
        self.assertEqual(
            self.coverage_calls,
            [("dataset_smap_l3", "aoi_raritan_basin", "2026-06-01/2026-06-30", "user-user-1")],
        )

    async def test_preview_reports_no_gibs_layer_plainly_instead_of_an_empty_result(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/discovery/dataset/dataset_no_gibs/preview",
                    json={"location": "Raritan basin", "time_range": "2026-06-01/2026-06-30"},
                    headers=self.auth_headers1,
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "has_gibs_layer": False,
            "message": "No GIBS browse layer is available for this dataset.",
        })

    async def test_preview_resolves_the_aoi_when_a_location_is_given(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/discovery/dataset/dataset_smap_l3/preview",
                    json={"location": "Raritan basin", "time_range": "2026-06-01/2026-06-30", "layer": "true_color"},
                    headers=self.auth_headers1,
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["has_gibs_layer"])
        self.assertIn("gibs_url", body)
        self.assertEqual(
            self.preview_calls,
            [("dataset_smap_l3", "aoi_raritan_basin", "2026-06-01/2026-06-30", "true_color", "user-user-1")],
        )

    async def test_preview_skips_aoi_resolution_when_no_location_is_given(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/discovery/dataset/dataset_smap_l3/preview", json={}, headers=self.auth_headers1,
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.aoi_calls, [])
        self.assertEqual(
            self.preview_calls,
            [("dataset_smap_l3", None, None, None, "user-user-1")],
        )

    async def test_discovery_endpoints_require_authentication(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            search_res = await client.post("/discovery/search", json={"query": "no2"})
            describe_res = await client.get("/discovery/dataset/dataset_smap_l3")
            preview_res = await client.post("/discovery/dataset/dataset_smap_l3/preview", json={})
            coverage_res = await client.post(
                "/discovery/dataset/dataset_smap_l3/coverage",
                json={"location": "Raritan basin", "time_range": "2026-06-01/2026-06-30"},
            )

        self.assertEqual(search_res.status_code, 401)
        self.assertEqual(describe_res.status_code, 401)
        self.assertEqual(preview_res.status_code, 401)
        self.assertEqual(coverage_res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
