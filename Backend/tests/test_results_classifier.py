"""
tests/test_results_classifier.py
==================================
PRD T18: parse_tool_result is the single classification point turning every
raw MCP tool outcome into either a parsed dict or a typed MCPToolError. This
is the table-driven test the PRD calls for — it pins the raw shapes the
adapter is known to produce (clean JSON, the MCP's structured not_found
passthrough, FastMCP validation-error prose, a tool-raised ValueError's
prose, unrecognized garbage) so a future adapter upgrade that changes error
text fails here, not in production.
"""
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


class ParseToolResultClassifierTests(unittest.TestCase):
    """Hermetic table-driven tests: feed a raw shape straight into
    parse_tool_result and assert what comes back — a dict, or a typed
    MCPToolError with the right category. No adapter/network involved here;
    the live-round-trip case (a genuine Nominatim-raised ValueError crossing
    the wire) is covered separately below."""

    def test_clean_json_dict_passes_through_unchanged(self):
        from earthdata_mcp.results import parse_tool_result

        result = parse_tool_result({"dataset_handle": "dataset_1", "variables": ["no2"]})

        self.assertEqual(result, {"dataset_handle": "dataset_1", "variables": ["no2"]})

    def test_clean_json_string_is_parsed(self):
        from earthdata_mcp.results import parse_tool_result

        result = parse_tool_result('{"handle": "aoi_1"}')

        self.assertEqual(result, {"handle": "aoi_1"})

    def test_content_block_list_is_unwrapped_and_parsed(self):
        from earthdata_mcp.results import parse_tool_result

        raw = [{"type": "text", "text": '{"job_handle": "job_1", "status": "ready"}'}]

        result = parse_tool_result(raw)

        self.assertEqual(result, {"job_handle": "job_1", "status": "ready"})

    def test_structured_not_found_dict_passes_through_as_a_result_not_an_error(self):
        from earthdata_mcp.results import parse_tool_result

        raw = {"handle": "obs_1", "status": "not_found", "message": "Unknown handle 'obs_1'."}

        result = parse_tool_result(raw)

        self.assertEqual(result["status"], "not_found")

    def test_structured_expired_dict_passes_through_as_a_result_not_an_error(self):
        from earthdata_mcp.results import parse_tool_result

        raw = {"handle": "obs_1", "status": "expired", "message": "Handle evicted."}

        result = parse_tool_result(raw)

        self.assertEqual(result["status"], "expired")

    def test_fastmcp_validation_error_prose_classifies_as_contract(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        # The exact shape pydantic's ValidationError.__str__ produces, and
        # the exact string that crashed json.loads live on 2026-07-07 (F3).
        raw = "1 validation error for check_coverageArguments\naoi_handle\n  Input should be a valid string"

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "contract")
        self.assertIn("aoi_handle", ctx.exception.raw_preview)

    def test_nominatim_miss_prose_classifies_as_user_input_with_a_suggestion(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = "Nominatim found no results for location 'zzzzqqqq nowhere'"

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")
        self.assertIn("zzzzqqqq nowhere", ctx.exception.message)
        self.assertIsNotNone(ctx.exception.suggestion)

    def test_ambiguous_location_prose_classifies_as_user_input(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = "Ambiguous location '02': valid as both a HUC watershed code and a US FIPS code."

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")

    def test_malformed_time_range_prose_classifies_as_user_input(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = (
            "time_range 'not-a-range' has no '/' or ',' delimiter. time_range must be "
            "'start/end' or 'start,end' with ISO-8601 endpoints"
        )

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")

    def test_adapter_error_calling_tool_prefix_is_stripped_before_classification(self):
        # Live-verified 2026-07-08 against the real MCP: langchain_mcp_adapters
        # wraps a tool-raised exception's text in "Error calling tool 'X': "
        # before it ever reaches this classifier — that boilerplate carries
        # no information a researcher needs and must not leak into the
        # user-facing message or defeat the known-prefix prose match.
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = (
            "Error calling tool 'define_area_of_interest': Neither Nominatim nor "
            "USGS WBD found results for location 'zzzzqqqq nowhere'"
        )

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")
        self.assertNotIn("Error calling tool", ctx.exception.message)
        self.assertIn("zzzzqqqq nowhere", ctx.exception.message)

    def test_unknown_variable_prose_classifies_as_user_input_preserving_candidates(self):
        """Live 2026-07-11: the earthdata agent guessed 'ozone_total_column'
        for a collection whose real variable is 'column_amount_o3'. The MCP's
        ValueError names the guess *and* the closest real matches — swallowing
        it into the generic contract message left the model unable to
        self-correct, so it guessed again and eventually reported
        "configuration errors" for four pollutants in a row."""
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = (
            "Error calling tool 'retrieve_subset': Unknown variable(s) for this "
            "collection: 'ozone_total_column' (closest matches: ['column_amount_o3', 'fc'])"
        )

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")
        self.assertIn("ozone_total_column", ctx.exception.message)
        self.assertIn("column_amount_o3", ctx.exception.message)
        self.assertIsNotNone(ctx.exception.suggestion)

    def test_unknown_variable_prose_without_close_matches_still_surfaces_the_name(self):
        # The no-matches shape ("'formaldehyde_total_column'" bare, live
        # 2026-07-11) must classify the same way — the pattern anchors on the
        # message prefix, not on the optional closest-matches suffix.
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = (
            "Error calling tool 'retrieve_subset': Unknown variable(s) for this "
            "collection: 'formaldehyde_total_column'"
        )

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")
        self.assertIn("formaldehyde_total_column", ctx.exception.message)
        self.assertIsNotNone(ctx.exception.suggestion)

    def test_transient_dns_failure_prose_classifies_as_provider_unavailable(self):
        """Live 2026-07-11: describe_dataset died inside the MCP with a DNS
        ConnectError ("[Errno -5] No address associated with hostname") at the
        exact moment the agent needed it to recover from a failed retrieval.
        A transient network failure raised server-side arrives as prose (not
        as a client-side ConnectError, which call_tool already classifies) —
        it must read as retryable, not as the dead-end contract message."""
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = (
            "Error calling tool 'describe_dataset': [Errno -5] No address "
            "associated with hostname"
        )

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "provider_unavailable")
        self.assertIsNotNone(ctx.exception.suggestion)

    def test_unknown_garbage_string_classifies_as_contract_never_crashes(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result("something a future MCP version might say that we've never seen")

        self.assertEqual(ctx.exception.category, "contract")

    def test_self_produced_error_envelope_reraises_as_mcp_tool_error(self):
        """bind_workspace (T18) serializes a classified MCPToolError into this
        exact JSON envelope for model consumption; a downstream backend
        composite calling parse_tool_result on that same string must recover
        the typed exception, not treat it as a normal result."""
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        raw = '{"error": {"category": "provider_unavailable", "message": "MCP is down.", "suggestion": "Try later."}}'

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "provider_unavailable")
        self.assertEqual(ctx.exception.message, "MCP is down.")
        self.assertEqual(ctx.exception.suggestion, "Try later.")

    def test_to_tool_json_round_trips_through_parse_tool_result(self):
        from earthdata_mcp.results import MCPToolError, parse_tool_result

        original = MCPToolError("no_data", "No granules for this AOI/period.", suggestion="Check availability first.")

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(original.to_tool_json())

        self.assertEqual(ctx.exception.category, "no_data")
        self.assertEqual(ctx.exception.message, "No granules for this AOI/period.")
        self.assertEqual(ctx.exception.suggestion, "Check availability first.")


class CallToolConnectionFailureTests(unittest.IsolatedAsyncioTestCase):
    """call_tool is the thin seam that catches a connection-raised exception
    (never returned as content, per langchain_mcp_adapters — see
    earthdata_mcp/results.py) and reclassifies it as MCPToolError instead of
    letting httpcore/mcp exception types leak past this module."""

    async def test_connect_error_raised_by_ainvoke_classifies_as_provider_unavailable(self):
        import httpcore

        from earthdata_mcp.results import MCPToolError, call_tool

        class _BrokenTool:
            name = "search_datasets"

            async def ainvoke(self, kwargs):
                raise httpcore.ConnectError("Connection refused")

        with self.assertRaises(MCPToolError) as ctx:
            await call_tool(_BrokenTool(), {"query": "no2"})

        self.assertEqual(ctx.exception.category, "provider_unavailable")

    async def test_mcp_error_raised_by_ainvoke_classifies_as_provider_unavailable(self):
        from mcp.shared.exceptions import ErrorData, McpError

        from earthdata_mcp.results import MCPToolError, call_tool

        class _BrokenTool:
            name = "search_datasets"

            async def ainvoke(self, kwargs):
                raise McpError(ErrorData(code=-32000, message="session closed"))

        with self.assertRaises(MCPToolError) as ctx:
            await call_tool(_BrokenTool(), {"query": "no2"})

        self.assertEqual(ctx.exception.category, "provider_unavailable")

    async def test_connect_error_wrapped_in_an_exception_group_classifies_as_provider_unavailable(self):
        # Live-verified 2026-07-08 (MCP stopped mid-session): the
        # streamable-HTTP transport's anyio task groups wrap the actual
        # httpx.ConnectError in a BaseExceptionGroup rather than raising it
        # bare — a bare `except httpcore.ConnectError` never catches this
        # and the request 500s with no body. This is the regression test.
        import httpx

        from earthdata_mcp.results import MCPToolError, call_tool

        class _BrokenTool:
            name = "search_datasets"

            async def ainvoke(self, kwargs):
                raise ExceptionGroup(
                    "unhandled errors in a TaskGroup",
                    [httpx.ConnectError("[Errno -5] No address associated with hostname")],
                )

        with self.assertRaises(MCPToolError) as ctx:
            await call_tool(_BrokenTool(), {"query": "no2"})

        self.assertEqual(ctx.exception.category, "provider_unavailable")

    async def test_successful_call_returns_the_raw_result_unclassified(self):
        from earthdata_mcp.results import call_tool

        class _WorkingTool:
            name = "search_datasets"

            async def ainvoke(self, kwargs):
                return '{"results": []}'

        result = await call_tool(_WorkingTool(), {"query": "no2"})

        self.assertEqual(result, '{"results": []}')


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "live-round-trip classifier test dependencies are not installed",
)
class LiveRoundTripProseClassificationTests(unittest.IsolatedAsyncioTestCase):
    """One genuine round trip through the real adapter (fake_earthdata_mcp's
    real FastMCP-over-HTTP server), proving the classifier handles what the
    adapter actually hands back for a tool-raised error — not just a
    hand-typed string standing in for it."""

    async def test_tool_raised_value_error_reaches_parse_tool_result_as_prose(self):
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.results import MCPToolError, parse_tool_result
        from config.settings import Settings
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        async def define_area_of_interest(location, workspace_id="default"):
            raise ValueError(f"Nominatim found no results for location {location!r}")

        server = FakeEarthdataMCPServer(build_fake_mcp({"define_area_of_interest": define_area_of_interest}))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        tools = await load_raw_mcp_tools(settings)

        raw = await tools["define_area_of_interest"].ainvoke({
            "location": "zzzzqqqq nowhere", "workspace_id": "default",
        })

        with self.assertRaises(MCPToolError) as ctx:
            parse_tool_result(raw)

        self.assertEqual(ctx.exception.category, "user_input")
        self.assertIn("zzzzqqqq nowhere", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
