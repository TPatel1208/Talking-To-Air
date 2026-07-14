"""
tests/test_earthdata_mcp_edl_injection.py
===========================================
T31: bind_workspace's per-user Earthdata credential injection. The
workspace-binding wrapper (earthdata_mcp/workspace.py) is the single seam
every model-facing and composite MCP call already passes through (T19/T26
prior art) -- this extends it to feature-detect an advertised edl_token
parameter and inject the calling user's decrypted, unexpired token, never
require the parameter, and degrade cleanly against tools/MCPs that don't
advertise it.

Uses the same real, in-process FastMCP fixture as test_earthdata_mcp_
workspace.py (fake_earthdata_mcp.build_fake_mcp) so schema feature-detection
is exercised against a genuine JSON-schema-advertising tool, not a mock.
"""
import importlib.util
import json
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


class _FakeInjector:
    """A minimal stand-in for services.connector_credential_service.
    EdlCredentialInjector -- bind_workspace only depends on this shape
    (earthdata_mcp.workspace.EdlCredentialInjector, a Protocol)."""

    def __init__(self, token=None):
        self._token = token
        self.resolve_calls = []
        self.mark_used_calls = []
        self.mark_invalid_calls = []

    async def resolve(self, user_id):
        self.resolve_calls.append(user_id)
        return self._token

    def mark_used(self, user_id):
        self.mark_used_calls.append(user_id)

    async def mark_invalid(self, user_id):
        self.mark_invalid_calls.append(user_id)


def _build_mcp_with_edl_advertising_search(handler):
    """Every REQUIRED_TOOL_NAMES tool build_fake_mcp already provides, with
    search_datasets swapped for a variant whose schema additionally
    advertises edl_token -- models an MCP that has adopted PRD-022's
    per-call contract for at least one tool. describe_dataset is left
    untouched (no edl_token) as the non-advertising control."""
    from fake_earthdata_mcp import build_fake_mcp

    mcp = build_fake_mcp({"search_datasets": handler})
    mcp.local_provider.remove_tool("search_datasets")

    @mcp.tool(name="search_datasets")
    async def search_datasets(
        query: str, filters: dict | None = None, workspace_id: str = "default", edl_token: str | None = None
    ) -> dict:
        return await handler(query=query, filters=filters, workspace_id=workspace_id, edl_token=edl_token)

    return mcp


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class EdlTokenInjectionMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import FakeEarthdataMCPServer

        received = {}

        async def search_datasets(query, filters=None, workspace_id="default", edl_token=None):
            received["edl_token"] = edl_token
            received["workspace_id"] = workspace_id
            return {"datasets": [], "count": 0}

        self.received = received
        mcp = _build_mcp_with_edl_advertising_search(search_datasets)
        self.server = FakeEarthdataMCPServer(mcp)
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_connected_unexpired_token_is_injected_when_the_schema_advertises_it(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertEqual(self.received["edl_token"], "decrypted-token-abc")
        self.assertEqual(injector.mark_used_calls, ["17"])
        self.assertEqual(injector.resolve_calls, ["17"])

    async def test_no_connector_sends_nothing_and_never_marks_used(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token=None)
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertIsNone(self.received["edl_token"])
        self.assertEqual(injector.mark_used_calls, [])

    async def test_non_advertising_schema_never_receives_edl_token_even_with_a_valid_connector(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})

        # never even asked -- feature detection happened once at bind time.
        self.assertEqual(injector.resolve_calls, [])
        self.assertEqual(injector.mark_used_calls, [])

    async def test_no_injector_bound_is_inert(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")  # edl_injector omitted

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertIsNone(self.received["edl_token"])

    async def test_edl_token_and_workspace_id_are_both_stripped_from_the_model_facing_schema(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        schema = bound["search_datasets"].args_schema
        properties = schema["properties"] if isinstance(schema, dict) else schema.schema()["properties"]

        self.assertNotIn("edl_token", properties)
        self.assertNotIn("workspace_id", properties)
        self.assertIn("query", properties)

    async def test_missing_user_context_fails_loud_regardless_of_the_injector(self):
        from earthdata_mcp.workspace import MissingUserContextError, bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: None, edl_injector=injector)

        with self.assertRaises(MissingUserContextError):
            await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertNotIn("edl_token", self.received)

    async def test_contract_check_still_reaches_ready_with_no_edl_token_in_the_required_param_contract(self):
        """T31: edl_token is feature-detected, never required -- an
        un-upgraded MCP (every tool here except search_datasets) must still
        pass the connect-time schema check (T17)."""
        from earthdata_mcp.connection import check_tool_schemas

        mismatches = check_tool_schemas(self.tools)

        self.assertEqual(mismatches, {})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class EdlTokenErrorClassificationTests(unittest.IsolatedAsyncioTestCase):
    """PRD-022 (soft dependency): the three per-user credential error
    classes route through the existing structured-error pipeline
    (earthdata_mcp/results.py's category-generic _classify_dict) with fixed,
    actionable message templates -- and, since those templates never echo
    the MCP's own message/raw_preview, this is also where the secret-hygiene
    requirement (never log/return the injected token) is structurally
    satisfied, not bolted on."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import FakeEarthdataMCPServer

        async def search_datasets(query, filters=None, workspace_id="default", edl_token=None):
            if query == "trigger-invalid":
                return {"error": {"category": "token_invalid", "message": f"EDL rejected token {edl_token!r}"}}
            if query == "trigger-expired":
                return {"error": {"category": "token_expired", "message": "EDL token expired"}}
            if query == "trigger-eula":
                return {
                    "error": {
                        "category": "eula_not_accepted",
                        "message": "EULA not accepted",
                        "resolution_url": "https://urs.earthdata.nasa.gov/eula",
                    }
                }
            return {"datasets": [], "count": 0}

        mcp = _build_mcp_with_edl_advertising_search(search_datasets)
        self.server = FakeEarthdataMCPServer(mcp)
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_token_invalid_flips_connector_status_and_never_echoes_the_token(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="super-secret-token-value")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        raw = await bound["search_datasets"].ainvoke({"query": "trigger-invalid"})
        payload = json.loads(raw)

        self.assertEqual(payload["error"]["category"], "token_invalid")
        self.assertIn("reconnect", (payload["error"]["message"] + payload["error"].get("suggestion", "")).lower())
        self.assertNotIn("super-secret-token-value", raw)
        self.assertEqual(injector.mark_invalid_calls, ["17"])

    async def test_token_expired_degrades_without_flipping_stored_status(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="super-secret-token-value")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        raw = await bound["search_datasets"].ainvoke({"query": "trigger-expired"})
        payload = json.loads(raw)

        self.assertEqual(payload["error"]["category"], "token_expired")
        self.assertIn("expired", payload["error"]["message"].lower())
        self.assertEqual(injector.mark_invalid_calls, [])
        self.assertNotIn("super-secret-token-value", raw)

    async def test_eula_not_accepted_leaves_status_untouched_and_carries_the_resolution_link(self):
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="super-secret-token-value")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        raw = await bound["search_datasets"].ainvoke({"query": "trigger-eula"})
        payload = json.loads(raw)

        self.assertEqual(payload["error"]["category"], "eula_not_accepted")
        self.assertIn("urs.earthdata.nasa.gov/eula", payload["error"]["suggestion"])
        self.assertEqual(injector.mark_invalid_calls, [])

    async def test_token_invalid_on_the_shared_credential_path_never_flips_status(self):
        """No connector resolved (injected is False) -- a shared-credential
        call bouncing as token_invalid must never flip a connector row that
        this call never used."""
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token=None)
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["search_datasets"].ainvoke({"query": "trigger-invalid"})

        self.assertEqual(injector.mark_invalid_calls, [])


if __name__ == "__main__":
    unittest.main()
