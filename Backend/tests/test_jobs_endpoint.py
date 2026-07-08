import importlib.util
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

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
    "jobs endpoint test dependencies are not installed",
)
class JobsEndpointTests(unittest.IsolatedAsyncioTestCase):
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

        # Two users, each with jobs scoped to their own "user-{id}" workspace —
        # proves the endpoint never leaks one researcher's jobs to another.
        self.workspace_jobs = {
            "user-user-1": [{"job_handle": "job_1", "dataset": "TEMPO_NO2", "submitted_at": "2026-07-01T00:00:00Z"}],
            "user-user-2": [{"job_handle": "job_2", "dataset": "MOD11A1", "submitted_at": "2026-07-02T00:00:00Z"}],
        }
        self.statuses = {
            "job_1": {"job_handle": "job_1", "status": "ready", "progress": 100, "phase": "done", "obs_handle": "obs_1"},
            "job_2": {"job_handle": "job_2", "status": "failed", "message": "harmony: provider GES_DISC rejected request: invalid bbox"},
        }
        self.cancel_calls = []

        async def list_workspace(workspace_id):
            return {"jobs": self.workspace_jobs.get(workspace_id, [])}

        async def get_retrieval_status(job_handle, workspace_id):
            return self.statuses[job_handle]

        async def cancel_retrieval(job_handle, workspace_id):
            self.cancel_calls.append((job_handle, workspace_id))
            return {"job_handle": job_handle, "status": "cancelled"}

        self.server = FakeEarthdataMCPServer(build_fake_mcp({
            "list_workspace": list_workspace,
            "get_retrieval_status": get_retrieval_status,
            "cancel_retrieval": cancel_retrieval,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        self.api.app.state.earthdata_mcp_tools = await load_earthdata_tools(settings, current_user_id)
        # Jobs endpoints read tools through earthdata_mcp_manager (T17).
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(
            state="ready", tools=self.api.app.state.earthdata_mcp_tools,
        )

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
        users = {self.user1.id: self.user1, self.user2.id: self.user2}

        async def fake_get_user_by_id(user_id):
            return users.get(user_id)

        async def fake_is_token_revoked(jti):
            return False

        return patch("services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def test_get_jobs_composes_list_and_status_scoped_to_the_caller(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                res1 = await client.get("/jobs", headers=self.auth_headers1)
                res2 = await client.get("/jobs", headers=self.auth_headers2)

        self.assertEqual(res1.status_code, 200)
        jobs1 = res1.json()["jobs"]
        self.assertEqual(len(jobs1), 1)
        self.assertEqual(jobs1[0]["job_handle"], "job_1")
        self.assertEqual(jobs1[0]["dataset"], "TEMPO_NO2")
        self.assertEqual(jobs1[0]["status"], "ready")
        self.assertEqual(jobs1[0]["obs_handle"], "obs_1")

        jobs2 = res2.json()["jobs"]
        self.assertEqual(len(jobs2), 1)
        self.assertEqual(jobs2[0]["job_handle"], "job_2")
        self.assertEqual(jobs2[0]["status"], "failed")
        self.assertEqual(jobs2[0]["message"], "harmony: provider GES_DISC rejected request: invalid bbox")

    async def test_cancel_job_proxies_the_mcp_and_scopes_to_the_caller(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post("/jobs/job_1/cancel", headers=self.auth_headers1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"job_handle": "job_1", "status": "cancelled"})
        self.assertEqual(self.cancel_calls, [("job_1", "user-user-1")])

    async def test_jobs_fails_honestly_when_the_mcp_is_not_ready(self):
        original_manager = self.api.app.state.earthdata_mcp_manager
        self.api.app.state.earthdata_mcp_manager = SimpleNamespace(state="incompatible", tools={})
        self.addCleanup(setattr, self.api.app.state, "earthdata_mcp_manager", original_manager)

        transport = self.httpx.ASGITransport(app=self.api.app)
        auth_patches = self._auth_patch()
        with auth_patches[0], auth_patches[1]:
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/jobs", headers=self.auth_headers1)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["category"], "provider_unavailable")

    async def test_jobs_endpoints_require_authentication(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            get_response = await client.get("/jobs")
            cancel_response = await client.post("/jobs/job_1/cancel")

        self.assertEqual(get_response.status_code, 401)
        self.assertEqual(cancel_response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
