"""
tests/test_session_auth_contract.py
=====================================
T47 findings #1/#2: an auth failure mid-session must read as 401 ("sign in
again"), never as 404 ("this thing is gone"). The frontend routes one to
in-place re-auth and the other to an honest empty state, so the two must never
be confused. These pin the invariant so a future refactor can't quietly turn
an expired/garbage/unsigned token into a 404 on a session-scoped route.

The complementary half -- a *successfully authenticated* request for a resource
that genuinely belongs to someone else still answers 404 (don't leak
existence) -- is pinned here for the session-history route and, for charts,
already by test_provenance_endpoint's foreign-chart test.

T61 note: the revoked-token case this file used to carry is **gone, not
forgotten**. There is no denylist any more -- an access token stays
cryptographically valid until its own `exp`, which is the accepted consequence
of verifying locally against a JWKS (logout revokes the *refresh* token, so
sign-out latency is bounded by the 45-minute access-token TTL, not by us).
Testing revocation would mean asserting behaviour the system no longer has.
It is replaced below by the wrong-signing-key case: a well-formed token
carrying every correct claim, signed by a key we never published. That is the
forgery the old shared-secret scheme could not have caught at all, and it must
read as 401 for exactly the same reason the expired one does.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above


REQUIRED_MODULES = ["fastapi", "httpx", "jwt", "langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in REQUIRED_MODULES),
    "session auth contract test dependencies are not installed",
)
class SessionAuthContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api
        self.api.app.state.agent = object()

        self.user = auth_helpers.user("user-1", email="tester@example.com")
        self.valid_token = auth_helpers.make_token(self.user.id, email=self.user.email)

    def _auth_patch(self):
        return auth_helpers.patch_verifier()

    async def _client(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        return self.httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_expired_token_answers_401_not_404_on_the_history_route(self):
        expired = auth_helpers.make_token(self.user.id, expires_in_minutes=-1)
        with self._auth_patch():
            async with await self._client() as client:
                res = await client.get("/session/abc/history", headers={"Authorization": f"Bearer {expired}"})
        self.assertEqual(res.status_code, 401)

    async def test_garbage_token_answers_401_not_404_on_chat(self):
        with self._auth_patch():
            async with await self._client() as client:
                res = await client.post(
                    "/chat",
                    json={"message": "hello"},
                    headers={"Authorization": "Bearer not-a-real-jwt"},
                )
        self.assertEqual(res.status_code, 401)

    async def test_token_signed_by_an_unpublished_key_answers_401_not_404(self):
        # Every claim correct, signed by a key absent from the JWKS. Replaces
        # the old revoked-token case (see the module docstring): this is what
        # a forged token looks like now, and the reply must still be 401.
        forged = auth_helpers.make_token(
            self.user.id, private_key=auth_helpers.OTHER_PRIVATE_KEY
        )
        with self._auth_patch():
            async with await self._client() as client:
                res = await client.get(
                    "/chart/chart-1/provenance",
                    headers={"Authorization": f"Bearer {forged}"},
                )
        self.assertEqual(res.status_code, 401)

    async def test_missing_token_answers_401_not_404_on_the_history_route(self):
        async with await self._client() as client:
            res = await client.get("/session/abc/history")
        self.assertEqual(res.status_code, 401)

    async def test_identity_provider_outage_answers_503_not_401(self):
        # The other half of the contract, new in T61. A 401 sends the frontend
        # into T47's re-auth modal; signing in again cannot reach an identity
        # provider that is down, so it would hand the user a login form that
        # also fails. A verifier that has never fetched a key set raises
        # IdentityProviderUnavailable, and that must surface as 503.
        dead = auth_helpers.build_verifier()
        dead.fetch_jwks = lambda: (_ for _ in ()).throw(ConnectionError("supabase unreachable"))
        with patch("tta_backend.api.supabase_verifier", dead):
            async with await self._client() as client:
                res = await client.get(
                    "/session/abc/history",
                    headers={"Authorization": f"Bearer {self.valid_token}"},
                )
        self.assertEqual(res.status_code, 503)
        self.assertNotIn("WWW-Authenticate", res.headers)

    async def test_auth_rejections_carry_cors_headers(self):
        """A 401 the browser cannot read is a 401 the frontend cannot act on.

        The auth middleware answers without calling call_next, so any middleware
        registered outside it never sees these responses. With CORSMiddleware
        registered inside, a cross-origin 401 arrived with no
        Access-Control-Allow-Origin and fetch rejected with a TypeError -- the
        status never reached shouldPromptReauth(401), and the 401-vs-503 split
        was invisible to the only client that consumes it.
        """
        origin = self.api.settings.cors_origins[0]
        async with await self._client() as client:
            unauthenticated = await client.get("/session/abc/history", headers={"Origin": origin})
        self.assertEqual(unauthenticated.status_code, 401)
        self.assertEqual(unauthenticated.headers.get("access-control-allow-origin"), origin)

    async def test_authenticated_request_for_a_foreign_session_still_answers_404(self):
        async def fake_session_belongs_to_user(thread_id, user_id):
            return False

        with self._auth_patch(), patch.object(self.api, "session_belongs_to_user", fake_session_belongs_to_user):
            async with await self._client() as client:
                res = await client.get(
                    "/session/someone-elses-thread/history",
                    headers={"Authorization": f"Bearer {self.valid_token}"},
                )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
