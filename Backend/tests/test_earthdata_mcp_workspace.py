import importlib.util
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


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class WorkspaceBindingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        received = {}

        async def search_datasets(query, filters, workspace_id):
            received["workspace_id"] = workspace_id
            return {"datasets": [], "count": 0}

        self.received = received
        self.server = FakeEarthdataMCPServer(build_fake_mcp({"search_datasets": search_datasets}))
        self.server.start()
        self.addCleanup(self.server.stop)

        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_bind_workspace_injects_workspace_id_at_call_time(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertEqual(self.received["workspace_id"], "user-17")

    async def test_bind_workspace_hides_workspace_id_from_the_schema(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        schema = bound["search_datasets"].args_schema
        properties = schema["properties"] if isinstance(schema, dict) else schema.schema()["properties"]

        self.assertNotIn("workspace_id", properties)
        self.assertIn("query", properties)

    async def test_search_datasets_emits_a_search_stage_status(self):
        """T19: one wrapper covers all curated discovery tools without
        touching the MCP — search_datasets narrates the "search" stage."""
        from earthdata_mcp.workspace import bind_workspace
        import utils.streaming as streaming

        bound = bind_workspace(self.tools, lambda: "17")

        seen = []
        token = streaming._status_emitter.set(
            lambda message, *, stage=None, detail=None: seen.append({"stage": stage, "detail": detail})
        )
        try:
            await bound["search_datasets"].ainvoke({"query": "no2"})
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["search"])


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class WorkspaceBindingStageStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        async def define_area_of_interest(location, workspace_id):
            return {"aoi_handle": "aoi_1", "location": location}

        async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"granule_count": 14, "coverage_pct": 100}

        self.server = FakeEarthdataMCPServer(build_fake_mcp({
            "define_area_of_interest": define_area_of_interest,
            "check_coverage": check_coverage,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_define_area_of_interest_emits_an_aoi_stage_status(self):
        from earthdata_mcp.workspace import bind_workspace
        import utils.streaming as streaming

        bound = bind_workspace(self.tools, lambda: "17")

        seen = []
        token = streaming._status_emitter.set(
            lambda message, *, stage=None, detail=None: seen.append({"stage": stage, "detail": detail})
        )
        try:
            await bound["define_area_of_interest"].ainvoke({"location": "New Jersey"})
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["aoi"])

    async def test_check_coverage_surfaces_the_granule_count_as_detail(self):
        """T19 story #3: granule count surfaced when coverage is checked, so
        a researcher understands why their request is small or large before
        the wait — a second stage="coverage" status carrying the count once
        the MCP's own response is known, alongside the pre-call one."""
        from earthdata_mcp.workspace import bind_workspace
        import utils.streaming as streaming

        bound = bind_workspace(self.tools, lambda: "17")

        seen = []
        token = streaming._status_emitter.set(
            lambda message, *, stage=None, detail=None: seen.append({"stage": stage, "detail": detail})
        )
        try:
            await bound["check_coverage"].ainvoke({
                "dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "2024-01-01/2024-01-02",
            })
        finally:
            streaming._status_emitter.reset(token)

        self.assertEqual([s["stage"] for s in seen], ["coverage", "coverage"])
        self.assertEqual(seen[-1]["detail"], 14)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class WorkspaceBindingClassifiedErrorTests(unittest.IsolatedAsyncioTestCase):
    """T18: bind_workspace is the single place every model-facing MCP tool
    passes through — a classified MCPToolError (raised by a tool through the
    MCP, or by a transport failure) never leaks out as a raw exception; the
    model (and any direct backend caller) always gets back a string, either
    the tool's normal content or the structured error envelope."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        async def define_area_of_interest(location, workspace_id):
            raise ValueError(f"Nominatim found no results for location {location!r}")

        self.server = FakeEarthdataMCPServer(build_fake_mcp({"define_area_of_interest": define_area_of_interest}))
        self.server.start()
        self.addCleanup(self.server.stop)

        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_a_classified_tool_error_comes_back_as_the_structured_json_envelope(self):
        import json

        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        raw = await bound["define_area_of_interest"].ainvoke({"location": "zzzzqqqq nowhere"})

        payload = json.loads(raw)
        self.assertEqual(payload["error"]["category"], "user_input")
        self.assertIn("zzzzqqqq nowhere", payload["error"]["message"])
        self.assertIn("suggestion", payload["error"])

    async def test_a_backend_composite_calling_parse_tool_result_on_that_same_output_recovers_the_typed_error(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        raw = await bound["define_area_of_interest"].ainvoke({"location": "zzzzqqqq nowhere"})

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class WorkspaceMissingUserContextTests(unittest.IsolatedAsyncioTestCase):
    """T26: a None user id must never mint a shared "user-None" workspace —
    that pooled every caller's retrievals together (113 orphaned rows found
    live). bind_workspace refuses instead, raising a typed error the model
    never silently swallows into a default."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        received = {}

        async def search_datasets(query, filters, workspace_id):
            received["workspace_id"] = workspace_id
            return {"datasets": [], "count": 0}

        self.received = received
        self.server = FakeEarthdataMCPServer(build_fake_mcp({"search_datasets": search_datasets}))
        self.server.start()
        self.addCleanup(self.server.stop)

        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None))

    async def test_a_none_user_id_raises_instead_of_minting_user_none(self):
        from earthdata_mcp.workspace import MissingUserContextError, bind_workspace

        bound = bind_workspace(self.tools, lambda: None)

        with self.assertRaises(MissingUserContextError):
            await bound["search_datasets"].ainvoke({"query": "no2"})

        # The MCP itself must never have been reached — the guard fires
        # before workspace_id is ever constructed, let alone with "None".
        self.assertNotIn("workspace_id", self.received)


if __name__ == "__main__":
    unittest.main()
