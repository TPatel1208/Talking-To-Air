import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_REQUIRED = ["fastapi", "httpx", "jwt", "bcrypt", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "capabilities endpoint dependencies are not installed",
)
class CapabilitiesStartersEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import api

        self.httpx = httpx
        self.api = api

    async def test_starters_endpoint_is_reachable_without_authentication(self):
        from config.starter_prompts import STARTER_PROMPTS

        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/capabilities/starters")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), len(STARTER_PROMPTS))
        for entry in body:
            self.assertIn("id", entry)
            self.assertIn("label", entry)
            self.assertIn("prompt", entry)
            self.assertIn("category", entry)


if __name__ == "__main__":
    unittest.main()
