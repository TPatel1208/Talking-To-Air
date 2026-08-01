"""
tests/test_session_auth_contract.py
=====================================
T47 findings #1/#2: an auth failure mid-session must read as 401 ("sign in
again"), never as 404 ("this thing is gone"). The frontend routes one to
in-place re-auth and the other to an honest empty state, so the two must never
be confused. These pin the invariant so a future refactor can't quietly turn
an expired/garbage/revoked token into a 404 on a session-scoped route.

The complementary half — a *successfully authenticated* request for a resource
that genuinely belongs to someone else still answers 404 (don't leak
existence) — is pinned here for the session-history route and, for charts,
already by test_provenance_endpoint's foreign-chart test.
"""
import importlib.util
import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "session auth contract test dependencies are not installed",
)
class SessionAuthContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        import jwt
        from tta_backend.config.settings import get_settings
        from tta_backend.models.user import User

        self.httpx = httpx
        self.api = api
        self.jwt = jwt
        self.settings = get_settings()
        self.api.app.state.agent = object()

        self.user = User(
            id="user-1", username="tester", password_hash="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        )
        self.valid_token, _ = self.api.create_access_token(self.user)

    def _token(self, *, exp_delta_minutes: int) -> str:
        payload = {
            "sub": self.user.id,
            "username": self.user.username,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=exp_delta_minutes),
            "jti": str(uuid.uuid4()),
        }
        return self.jwt.encode(payload, self.settings.jwt_secret_key, algorithm=self.settings.jwt_algorithm)

    def _auth_patch(self, *, revoked=False):
        async def fake_get_user_by_id(user_id):
            return self.user if user_id == self.user.id else None

        async def fake_is_token_revoked(jti):
            return revoked

        return patch("tta_backend.services.auth_service.get_user_by_id", fake_get_user_by_id), \
            patch("tta_backend.services.auth_service.is_token_revoked", fake_is_token_revoked)

    async def _client(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        return self.httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_expired_token_answers_401_not_404_on_the_history_route(self):
        expired = self._token(exp_delta_minutes=-1)
        auth = self._auth_patch()
        with auth[0], auth[1]:
            async with await self._client() as client:
                res = await client.get("/session/abc/history", headers={"Authorization": f"Bearer {expired}"})
        self.assertEqual(res.status_code, 401)

    async def test_garbage_token_answers_401_not_404_on_chat(self):
        auth = self._auth_patch()
        with auth[0], auth[1]:
            async with await self._client() as client:
                res = await client.post(
                    "/chat",
                    json={"message": "hello"},
                    headers={"Authorization": "Bearer not-a-real-jwt"},
                )
        self.assertEqual(res.status_code, 401)

    async def test_revoked_token_answers_401_not_404_on_a_chart_route(self):
        auth = self._auth_patch(revoked=True)
        with auth[0], auth[1]:
            async with await self._client() as client:
                res = await client.get(
                    "/chart/chart-1/provenance",
                    headers={"Authorization": f"Bearer {self.valid_token}"},
                )
        self.assertEqual(res.status_code, 401)

    async def test_missing_token_answers_401_not_404_on_the_history_route(self):
        async with await self._client() as client:
            res = await client.get("/session/abc/history")
        self.assertEqual(res.status_code, 401)

    async def test_authenticated_request_for_a_foreign_session_still_answers_404(self):
        auth = self._auth_patch()

        async def fake_session_belongs_to_user(thread_id, user_id):
            return False

        with auth[0], auth[1], patch.object(self.api, "session_belongs_to_user", fake_session_belongs_to_user):
            async with await self._client() as client:
                res = await client.get(
                    "/session/someone-elses-thread/history",
                    headers={"Authorization": f"Bearer {self.valid_token}"},
                )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
