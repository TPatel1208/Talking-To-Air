"""
tests/test_tool_cache.py
=========================
T53: the discovery-metadata cache at the ``bind_workspace`` seam. Every
model-facing MCP tool call already passes through that one wrapper for
workspace injection, T31 credential injection and T18 error classification —
so the cache lives there too, as an *allowlist* of effectively-immutable
metadata calls.

The load-bearing part of the PRD is the exclusion list: coverage and
availability are never cached (new granules land continuously and the agent's
availability answers are tool-grounded on purpose), retrieval tools have side
effects, and errors are never stored (a ``provider_unavailable`` during an MCP
restart window must not become a sticky failure for the rest of the TTL).

Exercised against the same real, in-process FastMCP fixture as
test_earthdata_mcp_workspace.py, so "issued exactly one MCP call" is counted
at a genuine server handler, not a mock.
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


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class ToolCacheAllowlistTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from earthdata_mcp.tool_cache import clear_tool_cache
        from fake_earthdata_mcp import FakeEarthdataMCPServer, build_fake_mcp

        clear_tool_cache()
        self.addCleanup(clear_tool_cache)

        self.calls = []

        async def describe_dataset(dataset_handle, detail=False, workspace_id="default"):
            self.calls.append(("describe_dataset", dataset_handle))
            return {"dataset_handle": dataset_handle, "variables": [{"name": "no2"}]}

        async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id="default"):
            self.calls.append(("check_coverage", dataset_handle))
            return {"granule_count": 14, "coverage_pct": 100}

        self.server = FakeEarthdataMCPServer(build_fake_mcp({
            "describe_dataset": describe_dataset,
            "check_coverage": check_coverage,
        }))
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools

        self.tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        )

    async def test_describe_dataset_twice_with_identical_args_issues_one_mcp_call(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        first = await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        second = await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second, first)

    async def test_check_coverage_twice_still_issues_two_mcp_calls(self):
        """The load-bearing exclusion. New granules land continuously and the
        agent's availability answers are tool-grounded on purpose — a cache
        that told a researcher data exists when it doesn't (or the reverse)
        would quietly defeat a rule built deliberately."""
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")
        args = {"dataset_handle": "d1", "aoi_handle": "aoi_1", "time_range": "2024-01-01/2024-01-02"}

        await bound["check_coverage"].ainvoke(args)
        await bound["check_coverage"].ainvoke(args)

        self.assertEqual(len(self.calls), 2)

    def test_the_include_and_exclude_lists_are_asserted_not_merely_commented(self):
        from earthdata_mcp.tool_cache import is_cacheable

        for name in ("describe_dataset", "search_datasets", "define_area_of_interest", "preview_dataset"):
            with self.subTest(cached=name):
                self.assertTrue(is_cacheable(name))

        never_cached = (
            # Freshness: availability changes continuously.
            "check_coverage",
            "check_availability",
            # Side effects, or state that changes.
            "retrieve_subset",
            "estimate_retrieval_size",
            "export_result",
            "rematerialize",
            "convert_format",
            "align",
            "retrieve_timeseries",
            "cancel_retrieval",
            # Already cached correctly (terminal-only) by jobs_service.
            "get_retrieval_status",
        )
        for name in never_cached:
            with self.subTest(never_cached=name):
                self.assertFalse(is_cacheable(name))

    async def test_two_workspaces_never_cross_serve_each_others_entries(self):
        """workspace_id participates in the key even though CMR metadata is
        public, so the T31 entitlement question never has to be answered."""
        from earthdata_mcp.workspace import bind_workspace

        await bind_workspace(self.tools, lambda: "17")["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        await bind_workspace(self.tools, lambda: "99")["describe_dataset"].ainvoke({"dataset_handle": "d1"})

        self.assertEqual(len(self.calls), 2)

    async def test_different_args_are_different_entries(self):
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        await bound["describe_dataset"].ainvoke({"dataset_handle": "d2"})
        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})

        self.assertEqual([handle for _, handle in self.calls], ["d1", "d2"])

    async def test_the_cache_is_invisible_to_the_model_facing_compaction_wrapper(self):
        """The **raw** result is what is cached, so T13's compaction still runs
        normally and a hit is indistinguishable from a miss downstream."""
        from earthdata_mcp.workspace import bind_workspace, model_view_describe_dataset

        bound = bind_workspace(self.tools, lambda: "17")
        model_facing = model_view_describe_dataset(bound["describe_dataset"])

        on_miss = await model_facing.ainvoke({"dataset_handle": "d1"})
        on_hit = await model_facing.ainvoke({"dataset_handle": "d1"})

        self.assertEqual(on_hit, on_miss)
        self.assertEqual(len(self.calls), 1)

    async def test_hit_and_miss_are_counted_per_tool_name(self):
        """Story #6: the cache's value must be measurable rather than
        assumed — and per tool, so it is visible *which* call is repeating."""
        from earthdata_mcp import tool_cache
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        await bound["describe_dataset"].ainvoke({"dataset_handle": "d2"})
        await bound["check_coverage"].ainvoke(
            {"dataset_handle": "d1", "aoi_handle": "a1", "time_range": "2024-01-01/2024-01-02"}
        )

        stats = tool_cache.tool_cache_stats()

        self.assertEqual(stats["describe_dataset"], {"hits": 1, "misses": 2})
        # An excluded tool is not a miss — it never consults the cache at all.
        self.assertNotIn("check_coverage", stats)

    async def test_a_hit_leaves_a_greppable_log_event_naming_the_tool(self):
        """What makes the cache verifiable in a live session: counters live in
        Prometheus, but "did this describe_dataset actually hit?" has to be
        answerable from the backend log."""
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})
        with self.assertLogs("earthdata_mcp.tool_cache", level="INFO") as cm:
            await bound["describe_dataset"].ainvoke({"dataset_handle": "d1"})

        hits = [r for r in cm.records if getattr(r, "_event", None) == "mcp_tool_cache_hit"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]._tool, "describe_dataset")

    def test_a_newly_added_mcp_tool_is_uncached_until_deliberately_allowlisted(self):
        """Allowlist, not denylist — writing a new call site must never be
        enough to start caching something."""
        from earthdata_mcp.tool_cache import is_cacheable

        self.assertFalse(is_cacheable("some_tool_the_mcp_grows_next"))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class ToolCacheNeverStoresErrorsTests(unittest.IsolatedAsyncioTestCase):
    """A transient failure must be retried on the next attempt, never
    remembered. Replaying a provider_unavailable for the rest of the TTL would
    turn an MCP restart window (~10-15s, documented by the compare-flakiness
    investigation) into a sticky wall for the whole session."""

    async def asyncSetUp(self):
        from earthdata_mcp.tool_cache import clear_tool_cache
        from fake_earthdata_mcp import FakeEarthdataMCPServer, build_fake_mcp

        clear_tool_cache()
        self.addCleanup(clear_tool_cache)

        self.attempts = []

        async def define_area_of_interest(location, workspace_id="default"):
            self.attempts.append(location)
            if len(self.attempts) == 1:
                raise RuntimeError("upstream service temporarily unavailable")
            return {"aoi_handle": "aoi_1", "location": location}

        self.server = FakeEarthdataMCPServer(
            build_fake_mcp({"define_area_of_interest": define_area_of_interest})
        )
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools

        self.tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        )

    async def test_a_failed_call_is_not_stored_and_the_next_identical_call_can_succeed(self):
        import json

        from earthdata_mcp.results import parse_tool_result
        from earthdata_mcp.workspace import bind_workspace

        bound = bind_workspace(self.tools, lambda: "17")

        failed = json.loads(await bound["define_area_of_interest"].ainvoke({"location": "Newark"}))
        recovered = parse_tool_result(await bound["define_area_of_interest"].ainvoke({"location": "Newark"}))

        self.assertIn("error", failed)
        self.assertEqual(recovered["aoi_handle"], "aoi_1")
        self.assertEqual(len(self.attempts), 2)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class ToolCacheCredentialHygieneTests(unittest.IsolatedAsyncioTestCase):
    """The cache sits at the same seam as T31 credential injection, so it must
    not become a place a decrypted Earthdata token comes to rest — and a hit
    used no credential, so it must not report one as used."""

    async def asyncSetUp(self):
        from earthdata_mcp.tool_cache import clear_tool_cache
        from fake_earthdata_mcp import FakeEarthdataMCPServer
        from test_earthdata_mcp_edl_injection import _build_mcp_with_edl_advertising_search

        clear_tool_cache()
        self.addCleanup(clear_tool_cache)

        self.calls = []

        async def search_datasets(query, filters=None, workspace_id="default", edl_token=None):
            self.calls.append(query)
            return {"datasets": [], "count": 0}

        self.server = FakeEarthdataMCPServer(_build_mcp_with_edl_advertising_search(search_datasets))
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools

        self.tools = await load_raw_mcp_tools(
            Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)
        )

    async def test_a_hit_serves_without_marking_the_credential_used(self):
        from test_earthdata_mcp_edl_injection import _FakeInjector

        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["search_datasets"].ainvoke({"query": "no2"})
        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(injector.mark_used_calls, ["17"])

    async def test_no_decrypted_token_is_retained_anywhere_in_the_cache(self):
        from test_earthdata_mcp_edl_injection import _FakeInjector

        from earthdata_mcp import tool_cache
        from earthdata_mcp.workspace import bind_workspace

        injector = _FakeInjector(token="decrypted-token-abc")
        bound = bind_workspace(self.tools, lambda: "17", edl_injector=injector)

        await bound["search_datasets"].ainvoke({"query": "no2"})

        self.assertNotIn("decrypted-token-abc", repr(tool_cache._CACHE))

    def test_a_call_carrying_a_token_shares_the_entry_of_one_without(self):
        """The token is neither keyed on nor stored, so injection state never
        splits the cache — and no key can carry a credential."""
        from earthdata_mcp import tool_cache

        tool_cache.store("search_datasets", "user-17", {"query": "no2", "edl_token": "secret"}, "body")

        self.assertEqual(tool_cache.lookup("search_datasets", "user-17", {"query": "no2"}), "body")
        self.assertNotIn("secret", repr(tool_cache._CACHE))


class ToolCacheTtlAndBoundTests(unittest.TestCase):
    """The TTL is the safety net under "effectively immutable", and the entry
    cap is what stops a long-running process growing unbounded on search-query
    variety. Both are exercised directly against the store — the seam
    bind_workspace reads and writes — with the clock under test control."""

    def setUp(self):
        from earthdata_mcp.tool_cache import clear_tool_cache

        clear_tool_cache()
        self.addCleanup(clear_tool_cache)

    @staticmethod
    def _args(n):
        return {"dataset_handle": f"d{n}"}

    def test_an_entry_past_its_ttl_is_not_served(self):
        from unittest.mock import patch

        from earthdata_mcp import tool_cache

        def lookup(n):
            return tool_cache.lookup("describe_dataset", "user-17", self._args(n))

        with patch.dict(os.environ, {"MCP_METADATA_CACHE_TTL_SECONDS": "60"}, clear=False):
            self._reload_settings()
            with patch("earthdata_mcp.tool_cache.time.monotonic", return_value=1000.0):
                tool_cache.store("describe_dataset", "user-17", self._args(1), "cached-body")
                self.assertEqual(lookup(1), "cached-body")
            with patch("earthdata_mcp.tool_cache.time.monotonic", return_value=1059.0):
                self.assertEqual(lookup(1), "cached-body")
            with patch("earthdata_mcp.tool_cache.time.monotonic", return_value=1061.0):
                self.assertIsNone(lookup(1))

    def test_exceeding_the_entry_cap_evicts_the_least_recently_used_entry(self):
        from unittest.mock import patch

        from earthdata_mcp import tool_cache

        def store(n, body):
            tool_cache.store("describe_dataset", "user-17", self._args(n), body)

        def lookup(n):
            return tool_cache.lookup("describe_dataset", "user-17", self._args(n))

        with patch.dict(os.environ, {"MCP_METADATA_CACHE_MAX_ENTRIES": "2"}, clear=False):
            self._reload_settings()
            store(1, "one")
            store(2, "two")
            # Reading d1 makes d2 the least recently used one.
            lookup(1)
            store(3, "three")

            self.assertEqual(lookup(1), "one")
            self.assertIsNone(lookup(2))
            self.assertEqual(lookup(3), "three")

    def _reload_settings(self):
        from config.settings import get_settings

        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)


if __name__ == "__main__":
    unittest.main()
