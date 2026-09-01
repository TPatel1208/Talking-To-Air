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
import tempfile
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import (  # noqa: E402 -- needs the TESTS_DIR insert above
    ProcessCacheIsolation,
    clear_process_caches,
    deployment_cube_store_dir,
)

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


class CubeStoreIsolationTests(unittest.TestCase):
    """T52's cube store leaks the other way: not stale memory, stale *disk*.

    The two caches above are process-global and are fixed by clearing them. The
    cube store cannot be — it is a directory, and its default
    (``CUBE_STORE_DIR``, falling back to the deployment volume) names a real,
    shared, long-lived location. Under Docker that is the production named
    volume; on a Windows checkout the same POSIX default resolves against the
    current drive, and the suite was observed writing cubes and rewriting
    ``handle_index.json`` there.

    Two things follow, and both are bugs. Tests mutate a store a developer or a
    deployment also uses; and they *read* it, so a run's outcome depends on what
    earlier runs left behind — which is how a cube-cache test came to pass on a
    clean machine and fail on one that had run the suite before.

    ``test_cube_cache.StoreTestCase`` redirects per test, but that only covers
    classes that opt in, and it unwinds its redirect at teardown — so a
    background cube write still in flight lands in whatever the root is *then*.
    The fix has to move the floor: the default itself must not be a real store
    for the whole test process. These assert that property, which is why they
    check the resolved root rather than any particular fixture.
    """

    @staticmethod
    def _resolved_root() -> str:
        from tta_backend.services import cube_cache

        return os.path.abspath(cube_cache._store_root())

    @staticmethod
    def _deployment_default() -> str:
        return os.path.abspath(deployment_cube_store_dir())

    def test_the_store_root_is_never_the_deployment_volume(self) -> None:
        assert self._resolved_root() != self._deployment_default(), (
            "the cube store resolves to the deployment volume during tests — the "
            "suite is reading and writing the same store the application uses"
        )

    def test_the_store_root_is_outside_the_app_root(self) -> None:
        """The other real location this has landed in. ``Backend/cube_store/``
        is gitignored, so a polluted store is invisible to ``git status`` and
        survives every branch switch — the worst possible place for state a test
        result depends on.

        Anchored on ``APP_ROOT`` rather than ``TESTS_DIR/../..``. That expression
        gives the repo root on a developer machine, but in the container
        ``TESTS_DIR`` is ``/app/tests``, so ``../..`` resolves to ``/`` — the
        filesystem root, which contains every path including the tempdir this
        test is meant to bless. The assertion could therefore never pass under
        ``docker compose --profile test``, and reported a correctly-isolated
        store as pollution. ``APP_ROOT`` is ``Backend/`` in a checkout and
        ``/app`` in the image: exactly the directory the store must stay out of,
        in both.
        """
        from tta_backend import APP_ROOT

        app_root = os.path.abspath(APP_ROOT)
        root = self._resolved_root()
        try:
            inside = os.path.commonpath([root, app_root]) == app_root
        except ValueError:  # different drives — trivially outside
            inside = False
        assert not inside, f"the cube store resolves inside the app root: {root}"

    def test_a_the_store_root_can_be_recorded(self) -> None:
        """Paired with the next one — see the module docstring on the ``a``/``b``
        prefixes, which make both runners execute the writer first."""
        type(self)._seen_root = self._resolved_root()

    def test_b_the_store_root_is_the_same_one_the_previous_test_saw(self) -> None:
        """Isolation must not mean a *fresh* store per test. Cubes are written in
        the background and read back later, and ``handle_index.json`` is only
        useful across calls, so a root that moved underneath the suite would turn
        every hit into a miss and quietly stop exercising the cache at all —
        while still looking green."""
        assert self._resolved_root() == getattr(type(self), "_seen_root", None)

    def test_the_deployment_path_is_still_answerable_from_inside_the_suite(self) -> None:
        """Redirecting the store must not cost us the ability to *ask* where the
        real one is. ``test_cube_store_persistence`` asserts the deployment
        contract — that the path is on a persisted named volume, and is created
        before the Dockerfile's chown — against this value. Were it to follow the
        redirect, those tests would be checking a tempdir: passing, and guarding
        nothing."""
        deployment = self._deployment_default()
        assert deployment != self._resolved_root()
        assert not deployment.startswith(os.path.abspath(tempfile.gettempdir())), (
            f"the deployment cube store reads as a temp directory ({deployment}) — "
            "the redirect has leaked into the answer the contract tests rely on"
        )


class FrameStoreIsolationTests(unittest.TestCase):
    """T59's frame store leaks the same way the cube store does, and worse.

    Same default shape — ``FRAME_STORE_DIR`` falling back to a deployment
    volume, which on a Windows checkout resolves against the current drive — so
    the same two bugs follow: the suite mutates a store the application also
    uses, and reads it back.

    The "worse" is eviction. The cube store's failure mode was a test *adding*
    cubes a developer would later hit; this store has an LRU sweeper, so a test
    that writes past ``FRAME_STORE_MAX_BYTES`` **deletes** whatever is coldest.
    Pointed at a real store, a single eviction test would silently strip the
    scrubber off charts nobody touched — and D8 forbids rebuilding them.
    """

    @staticmethod
    def _resolved_root() -> str:
        from tta_backend.services import frame_store

        return os.path.abspath(frame_store._store_root())

    def test_the_store_root_is_never_the_deployment_volume(self) -> None:
        from cache_isolation import deployment_frame_store_dir

        assert self._resolved_root() != os.path.abspath(deployment_frame_store_dir()), (
            "the frame store resolves to the deployment volume during tests — an "
            "eviction test would delete stacks the application is serving"
        )

    def test_the_store_root_is_outside_the_app_root(self) -> None:
        from tta_backend import APP_ROOT

        app_root = os.path.abspath(APP_ROOT)
        root = self._resolved_root()
        try:
            inside = os.path.commonpath([root, app_root]) == app_root
        except ValueError:  # different drives — trivially outside
            inside = False
        assert not inside, f"the frame store resolves inside the app root: {root}"

    def test_the_deployment_path_is_still_answerable_from_inside_the_suite(self) -> None:
        """``test_frame_store_persistence`` asserts the volume and ownership
        contracts against this value; were it to follow the redirect those tests
        would be checking a tempdir — passing, and guarding nothing."""
        from cache_isolation import deployment_frame_store_dir

        deployment = os.path.abspath(deployment_frame_store_dir())
        assert deployment != self._resolved_root()
        assert not deployment.startswith(os.path.abspath(tempfile.gettempdir())), (
            f"the deployment frame store reads as a temp directory ({deployment}) — "
            "the redirect has leaked into the answer the contract tests rely on"
        )


class ClearProcessCachesTests(unittest.TestCase):
    """The shared policy itself: one function, every process-global cache.

    Both runners' hooks delegate here, so if a third cache is ever added this is
    the one place that has to learn about it.
    """

    def test_it_clears_every_process_global_cache(self) -> None:
        from tta_backend.earthdata_mcp import tool_cache
        from tta_backend.services import admission, jobs_service

        tool_cache.store("describe_dataset", _WORKSPACE, _ARGS, "payload")
        jobs_service._TERMINAL_STATUS_CACHE["job_probe"] = {"status": "ready"}
        # Admission control is the one entry here that does not merely serve a
        # stale answer when it leaks. A test that dies holding a permit removes
        # capacity permanently, so the next test that needs one *hangs* instead
        # of failing -- no name, no traceback, ~29 minutes into a suite. Set
        # directly rather than by acquiring, because this case is precisely the
        # one where the release never ran.
        admission.heavy_limit()  # force a generation to exist
        admission._limiter.in_flight = 3

        clear_process_caches()

        assert tool_cache.lookup("describe_dataset", _WORKSPACE, _ARGS) is None
        assert jobs_service._TERMINAL_STATUS_CACHE == {}
        assert admission.in_flight() == 0

    def test_it_also_neutralises_the_api_rate_limiter(self) -> None:
        """Not a cache, but the same leak and the same hook.

        A test that re-armed the limiter and then failed would otherwise leave
        it armed for everything after it, and the symptom -- a later file's
        requests refused on a limit it never asked to exercise -- would look
        nothing like its cause.
        """
        import tta_backend.api as api

        api.limiter.enabled = True

        clear_process_caches()

        assert api.limiter.enabled is False


if __name__ == "__main__":
    unittest.main()
