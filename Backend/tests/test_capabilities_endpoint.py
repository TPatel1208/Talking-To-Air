import importlib.util
import unittest


_REQUIRED = ["fastapi", "httpx", "jwt", "langchain", "langgraph"]


@unittest.skipIf(
    any(importlib.util.find_spec(m) is None for m in _REQUIRED),
    "capabilities endpoint dependencies are not installed",
)
class CapabilitiesStartersEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api

    async def test_starters_endpoint_is_reachable_without_authentication(self):
        from tta_backend.config.starter_prompts import STARTER_PROMPTS

        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/capabilities/starters")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), len(STARTER_PROMPTS))
        for entry in body:
            self.assertEqual(set(entry), {"id", "label", "prompt", "category"})
        self.assertEqual([entry["category"] for entry in body], [s["category"] for s in STARTER_PROMPTS])


if __name__ == "__main__":
    unittest.main()
