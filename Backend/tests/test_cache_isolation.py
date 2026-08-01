"""Process-global cache isolation between tests, under any runner.

Two caches here outlive a single test by design — T53's discovery-metadata
cache (``earthdata_mcp.tool_cache``) and T-jobs' terminal-status cache
(``services.jobs_service``). Both are process-lifetime state, so without
something clearing them between tests, one test's call is served to the next
test's identical call and that test's handler never runs. The symptom is not a
visible cache error: the later test simply observes that its fake was never
invoked.

That is not hypothetical. It cost CI four failures
(``test_earthdata_mcp_workspace``, ``test_earthdata_mcp_edl_injection``,
``test_discovery_endpoint``), and it went unseen because the only thing clearing
the caches was an autouse fixture in ``conftest.py`` — which pytest loads and
``unittest`` does not. The suite was isolated under one runner and not the
other, and nothing said so.

So this module asserts the *property*, not the mechanism: a test that populates
a cache must not leave it populated for the next one. It is deliberately
runner-agnostic — it fails under any runner whose isolation is missing, which is
exactly the signal that was absent before.

The ``test_a_``/``test_b_`` prefixes are load-bearing: ``unittest`` orders test
methods alphabetically and pytest uses definition order, so both runners execute
the writer before the reader.
"""

import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import ProcessCacheIsolation, clear_process_caches  # noqa: E402 -- needs the TESTS_DIR insert above

_WORKSPACE = "user-cache-isolation"
_ARGS = {"query": "isolation-probe"}


class ToolCacheIsolationTests(ProcessCacheIsolation, unittest.TestCase):
    """T53's discovery-metadata cache must not survive into the next test."""

    def test_a_a_test_can_populate_the_tool_cache(self) -> None:
        from tta_backend.earthdata_mcp import tool_cache

        tool_cache.store("search_datasets", _WORKSPACE, _ARGS, "cached-payload")

        assert tool_cache.lookup("search_datasets", _WORKSPACE, _ARGS) == "cached-payload"

    def test_b_the_next_test_starts_on_a_cold_tool_cache(self) -> None:
        from tta_backend.earthdata_mcp import tool_cache

        assert tool_cache.lookup("search_datasets", _WORKSPACE, _ARGS) is None, (
            "the previous test's cached result leaked into this one — process-global "
            "cache isolation is not active under this runner"
        )


class TerminalStatusCacheIsolationTests(ProcessCacheIsolation, unittest.TestCase):
    """The jobs terminal-status cache, same contract.

    Cleared today by two test modules that remember to do it themselves
    (``test_jobs_endpoint``, ``test_jobs_service``) — which is evidence it
    leaks, not evidence it is handled.
    """

    def test_a_a_test_can_populate_the_terminal_status_cache(self) -> None:
        from tta_backend.services import jobs_service

        jobs_service._TERMINAL_STATUS_CACHE["job_isolation_probe"] = {"status": "ready"}

        assert "job_isolation_probe" in jobs_service._TERMINAL_STATUS_CACHE

    def test_b_the_next_test_starts_on_a_cold_terminal_status_cache(self) -> None:
        from tta_backend.services import jobs_service

        assert "job_isolation_probe" not in jobs_service._TERMINAL_STATUS_CACHE, (
            "the previous test's cached status leaked into this one — process-global "
            "cache isolation is not active under this runner"
        )


class ClearProcessCachesTests(unittest.TestCase):
    """The shared policy itself: one function, every process-global cache.

    Both runners' hooks delegate here, so if a third cache is ever added this is
    the one place that has to learn about it.
    """

    def test_it_clears_every_process_global_cache(self) -> None:
        from tta_backend.earthdata_mcp import tool_cache
        from tta_backend.services import jobs_service

        tool_cache.store("describe_dataset", _WORKSPACE, _ARGS, "payload")
        jobs_service._TERMINAL_STATUS_CACHE["job_probe"] = {"status": "ready"}

        clear_process_caches()

        assert tool_cache.lookup("describe_dataset", _WORKSPACE, _ARGS) is None
        assert jobs_service._TERMINAL_STATUS_CACHE == {}


if __name__ == "__main__":
    unittest.main()
