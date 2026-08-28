"""The rate limiter's two load-bearing properties: who shares a bucket, and
what a throttled client is told.

Everything here is about ``api._rate_limit_key``. The limits themselves are
slowapi's to enforce and are not re-tested; what *is* ours is the decision that
a signed-in request counts against its user rather than its address. That
decision is invisible until two users share an address -- a campus or corporate
NAT, which is the normal case for this app's users -- and at that point getting
it wrong means the first heavy user locks out everyone behind the same egress.

``/debug/heap-snapshot`` is the endpoint under test purely because 1/minute is
the cheapest limit in the API to exhaust: two requests, no waiting. Nothing here
depends on what it returns.

The suite runs with the limiter disabled (see ``cache_isolation``), so these
tests arm it explicitly through :func:`rate_limiting_enabled` and hand it back
disabled -- which is also why the last test asserts the disabled default still
holds, rather than trusting that the opt-in cleaned up after itself.
"""
import dataclasses
import importlib.util
import os
import sys
import tracemalloc
import unittest
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402 -- needs the TESTS_DIR insert above
from cache_isolation import rate_limiting_enabled  # noqa: E402 -- same

_REQUIRED = ["fastapi", "httpx", "jwt", "slowapi"]

_SKIP = unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "rate limiting test dependencies are not installed",
)


def _request(host: str, user=None):
    """A Request carrying just the two things the key function reads."""
    from starlette.requests import Request

    request = Request({"type": "http", "client": (host, 5555), "headers": [], "state": {}})
    if user is not None:
        request.state.current_user = user
    return request


@_SKIP
class RateLimitKeyTests(unittest.TestCase):
    """The bucket a request is counted against."""

    def setUp(self):
        import tta_backend.api as api

        self.key = api._rate_limit_key

    def test_a_signed_in_request_is_keyed_by_user_not_address(self):
        key = self.key(_request("203.0.113.7", auth_helpers.user("user-a")))
        self.assertEqual(key, "user:user-a")

    def test_two_users_behind_one_address_do_not_share_a_bucket(self):
        # The whole point of the user key. Same egress address, different people.
        a = self.key(_request("203.0.113.7", auth_helpers.user("user-a")))
        b = self.key(_request("203.0.113.7", auth_helpers.user("user-b")))
        self.assertNotEqual(a, b)

    def test_one_user_on_two_networks_keeps_a_single_allowance(self):
        # The converse failure: keying by address would hand the same person a
        # fresh budget every time they changed network.
        office = self.key(_request("203.0.113.7", auth_helpers.user("user-a")))
        home = self.key(_request("198.51.100.4", auth_helpers.user("user-a")))
        self.assertEqual(office, home)

    def test_an_anonymous_request_falls_back_to_the_address(self):
        self.assertEqual(self.key(_request("203.0.113.7")), "ip:203.0.113.7")

    def test_the_two_namespaces_cannot_collide(self):
        # A user id that happens to look like an address must not inherit that
        # address's count, so the prefixes are load-bearing rather than cosmetic.
        as_user = self.key(_request("10.0.0.1", auth_helpers.user("203.0.113.7")))
        as_address = self.key(_request("203.0.113.7"))
        self.assertNotEqual(as_user, as_address)


@_SKIP
class RateLimitEnforcementTests(unittest.IsolatedAsyncioTestCase):
    """End to end, through the middleware that supplies the user."""

    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api
        tracemalloc.start()
        self.addCleanup(tracemalloc.stop)

    async def _get(self, client, sub):
        return await client.get(
            "/debug/heap-snapshot",
            headers={"Authorization": f"Bearer {auth_helpers.make_token(sub)}"},
        )

    def _enabled_app(self):
        return patch.object(
            self.api,
            "settings",
            dataclasses.replace(self.api.settings, debug_heap_profiling_enabled=True),
        )

    async def test_a_user_who_exceeds_the_limit_is_refused(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        with auth_helpers.patch_verifier(), self._enabled_app(), rate_limiting_enabled():
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                first = await self._get(client, "user-a")
                second = await self._get(client, "user-a")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    async def test_one_users_exhaustion_does_not_refuse_another_user(self):
        """The regression this whole change exists to prevent.

        Both requests arrive from the same address -- ASGITransport gives every
        request the same client host -- so under address keying the second user
        would be refused a request they never made.
        """
        transport = self.httpx.ASGITransport(app=self.api.app)
        with auth_helpers.patch_verifier(), self._enabled_app(), rate_limiting_enabled():
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                await self._get(client, "user-a")
                exhausted = await self._get(client, "user-a")
                other_user = await self._get(client, "user-b")

        self.assertEqual(exhausted.status_code, 429)
        self.assertEqual(other_user.status_code, 200, "a second user inherited the first user's count")

    async def test_a_refusal_tells_the_client_when_to_come_back(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        with auth_helpers.patch_verifier(), self._enabled_app(), rate_limiting_enabled():
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                await self._get(client, "user-a")
                refused = await self._get(client, "user-a")

        self.assertEqual(refused.status_code, 429)
        retry_after = refused.headers.get("Retry-After")
        self.assertIsNotNone(retry_after, "a 429 without Retry-After leaves the client guessing")
        self.assertGreaterEqual(int(retry_after), 1)
        self.assertLessEqual(int(retry_after), 60)

    async def test_a_refusal_is_shaped_like_every_other_error_this_api_returns(self):
        # The frontend reads `detail` on any non-ok response; slowapi's own
        # handler uses `error`, which would surface as a bare "HTTP 429".
        transport = self.httpx.ASGITransport(app=self.api.app)
        with auth_helpers.patch_verifier(), self._enabled_app(), rate_limiting_enabled():
            async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                await self._get(client, "user-a")
                refused = await self._get(client, "user-a")

        self.assertIn("detail", refused.json())


@_SKIP
class RateLimiterSuiteIsolationTests(unittest.TestCase):
    """The limiter must be off for everyone who did not ask for it.

    Without this the suite fails on its own traffic: the per-minute limits are
    production policy, and a test file legitimately makes far more calls as one
    user inside one minute than a human would.
    """

    def test_the_limiter_is_disabled_by_default(self):
        import tta_backend.api as api

        self.assertFalse(api.limiter.enabled)


if __name__ == "__main__":
    unittest.main()
