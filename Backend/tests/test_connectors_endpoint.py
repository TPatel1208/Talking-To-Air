"""
tests/test_connectors_endpoint.py
==================================
T30: the Connectors HTTP surface -- list/set-token/disconnect, each scoped to
the caller via request.state.current_user exactly like every other per-user
endpoint (jobs, sessions, charts).
"""
import importlib.util
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "bcrypt", "cryptography", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


class _FakeConnectorStore:
    """In-memory stand-in for repositories.user_connector_repository, keyed
    exactly like the real table's UNIQUE (user_id, connector_type) -- lets
    the route tests prove per-user isolation without a live Postgres."""

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}
        self.secrets: dict[tuple[str, str], str] = {}
        self.delete_calls: list[tuple[str, str]] = []

    async def list_connectors_for_user(self, user_id):
        return [row for (uid, _ctype), row in self.rows.items() if uid == user_id]

    async def upsert_connector(self, user_id, connector_type, auth_method, encrypted_secret, expires_at, status="connected"):
        row = {
            "connector_type": connector_type,
            "auth_method": auth_method,
            "expires_at": expires_at,
            "status": status,
            "connected_at": datetime.now(timezone.utc),
            "last_used_at": None,
        }
        self.rows[(user_id, connector_type)] = row
        self.secrets[(user_id, connector_type)] = encrypted_secret
        return dict(row)

    async def delete_connector(self, user_id, connector_type):
        self.delete_calls.append((user_id, connector_type))
        existed = (user_id, connector_type) in self.rows
        self.rows.pop((user_id, connector_type), None)
        self.secrets.pop((user_id, connector_type), None)
        return existed


def _make_edl_token(exp_delta=timedelta(days=60)):
    import jwt

    exp = datetime.now(timezone.utc) + exp_delta
    return jwt.encode({"sub": "urs-user", "exp": int(exp.timestamp())}, "arbitrary-secret", algorithm="HS256")


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "connectors endpoint test dependencies are not installed",
)
class ConnectorsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import api
        from cryptography.fernet import Fernet
        from models.user import User
        from utils.connector_crypto import build_multi_fernet

        self.httpx = httpx
        self.api = api
        self.store = _FakeConnectorStore()
        self.cipher = build_multi_fernet(Fernet.generate_key().decode())

        self.user1 = User(id="user-1", username="one", password_hash="hash", created_at=datetime.now(timezone.utc), is_active=True)
        self.user2 = User(id="user-2", username="two", password_hash="hash", created_at=datetime.now(timezone.utc), is_active=True)
        token1, _ = self.api.create_access_token(self.user1)
        token2, _ = self.api.create_access_token(self.user2)
        self.auth_headers1 = {"Authorization": f"Bearer {token1}"}
        self.auth_headers2 = {"Authorization": f"Bearer {token2}"}

    def _auth_patches(self):
        users = {self.user1.id: self.user1, self.user2.id: self.user2}

        async def fake_get_user_by_id(user_id):
            return users.get(user_id)

        async def fake_is_token_revoked(jti):
            return False

        return [
            patch("services.auth_service.get_user_by_id", fake_get_user_by_id),
            patch("services.auth_service.is_token_revoked", fake_is_token_revoked),
        ]

    def _repo_patches(self):
        return [
            patch("api.list_connectors_for_user", self.store.list_connectors_for_user),
            patch("api.upsert_connector", self.store.upsert_connector),
            patch("api.delete_connector", self.store.delete_connector),
        ]

    async def _client(self, patches, configured=True):
        cipher_patch = patch("api.get_connector_cipher", return_value=(self.cipher if configured else None))
        for p in patches + [cipher_patch]:
            p.start()
            self.addCleanup(p.stop)
        transport = self.httpx.ASGITransport(app=self.api.app)
        return self.httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_list_connectors_is_registry_driven_and_defaults_to_not_connected(self):
        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.get("/connectors", headers=self.auth_headers1)

        self.assertEqual(res.status_code, 200)
        connectors = res.json()["connectors"]
        self.assertEqual(len(connectors), 1)
        self.assertEqual(connectors[0]["connector_type"], "earthdata")
        self.assertEqual(connectors[0]["status"], "not_connected")
        self.assertIn("token_docs_url", connectors[0])

    async def test_set_token_stores_the_expiry_and_the_response_never_contains_the_token(self):
        raw_token = _make_edl_token()
        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.put(
                "/connectors/earthdata/token", json={"token": raw_token}, headers=self.auth_headers1,
            )

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "connected")
        self.assertIsNotNone(body["expires_at"])

        raw_body = res.text
        self.assertNotIn(raw_token, raw_body)
        stored_encrypted_secret = self.store.secrets[("user-1", "earthdata")]
        self.assertNotIn(stored_encrypted_secret, raw_body)

    async def test_set_token_rejects_an_expired_token_with_422(self):
        raw_token = _make_edl_token(exp_delta=timedelta(days=-1))
        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.put(
                "/connectors/earthdata/token", json={"token": raw_token}, headers=self.auth_headers1,
            )

        self.assertEqual(res.status_code, 422)
        self.assertIn("expired", res.json()["detail"])
        self.assertNotIn(("user-1", "earthdata"), self.store.rows)

    async def test_set_token_rejects_non_jwt_garbage_with_422(self):
        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.put(
                "/connectors/earthdata/token", json={"token": "definitely-not-a-jwt"}, headers=self.auth_headers1,
            )

        self.assertEqual(res.status_code, 422)

    async def test_set_token_rejects_an_unknown_connector_type(self):
        raw_token = _make_edl_token()
        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.put(
                "/connectors/not-a-real-connector/token", json={"token": raw_token}, headers=self.auth_headers1,
            )

        self.assertEqual(res.status_code, 404)

    async def test_a_row_whose_expiry_has_passed_reports_expired_without_a_stored_status_change(self):
        self.store.rows[("user-1", "earthdata")] = {
            "connector_type": "earthdata",
            "auth_method": "token",
            "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
            "status": "connected",  # stored status is untouched -- derived at read time
            "connected_at": datetime.now(timezone.utc) - timedelta(days=61),
            "last_used_at": None,
        }

        async with await self._client(self._auth_patches() + self._repo_patches()) as client:
            res = await client.get("/connectors", headers=self.auth_headers1)

        connector = res.json()["connectors"][0]
        self.assertEqual(connector["status"], "expired")
        self.assertEqual(self.store.rows[("user-1", "earthdata")]["status"], "connected")

    async def test_disconnect_deletes_only_the_callers_row(self):
        raw_token = _make_edl_token()
        patches = self._auth_patches() + self._repo_patches()
        async with await self._client(patches) as client:
            await client.put("/connectors/earthdata/token", json={"token": raw_token}, headers=self.auth_headers1)
            await client.put("/connectors/earthdata/token", json={"token": raw_token}, headers=self.auth_headers2)

            res = await client.delete("/connectors/earthdata", headers=self.auth_headers1)

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "not_connected")
        self.assertEqual(self.store.delete_calls, [("user-1", "earthdata")])
        # user-2's row is untouched by user-1's disconnect.
        self.assertIn(("user-2", "earthdata"), self.store.rows)

    async def test_user_as_connector_never_appears_in_user_bs_list(self):
        raw_token = _make_edl_token()
        patches = self._auth_patches() + self._repo_patches()
        async with await self._client(patches) as client:
            await client.put("/connectors/earthdata/token", json={"token": raw_token}, headers=self.auth_headers1)
            res = await client.get("/connectors", headers=self.auth_headers2)

        self.assertEqual(res.json()["connectors"][0]["status"], "not_connected")

    async def test_all_three_endpoints_answer_a_structured_503_when_unconfigured(self):
        patches = self._auth_patches() + self._repo_patches()
        async with await self._client(patches, configured=False) as client:
            list_res = await client.get("/connectors", headers=self.auth_headers1)
            set_res = await client.put(
                "/connectors/earthdata/token", json={"token": _make_edl_token()}, headers=self.auth_headers1,
            )
            delete_res = await client.delete("/connectors/earthdata", headers=self.auth_headers1)

        for res in (list_res, set_res, delete_res):
            self.assertEqual(res.status_code, 503)
            self.assertIn("not configured", res.json()["detail"])

    async def test_connectors_endpoints_require_authentication(self):
        async with await self._client(self._repo_patches()) as client:
            res = await client.get("/connectors")

        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
