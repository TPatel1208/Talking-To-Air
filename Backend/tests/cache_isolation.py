"""Cache isolation, independent of which runner is executing.

Several caches and stores outlive a single test by design — T53's
discovery-metadata cache (``earthdata_mcp.tool_cache``), the jobs
terminal-status cache (``services.jobs_service``), T52's Zarr cube store, T23's
overlay PNG store, and the public chart-output directory. The first two are
process-lifetime *memory*, so unless something clears them between tests, one
test's call is served to the next test's identical call and that test's handler
never runs. The last three are *disk*, and need the opposite treatment — see
:func:`isolate_cube_store` and :func:`_isolate_store`.

For the two in-memory ones, the only thing clearing them used to be an autouse
fixture in ``Backend/conftest.py``. pytest loads that file; ``unittest`` does
not. So the suite was isolated under one runner and silently order-dependent
under the other, and CI — which ran ``unittest discover`` — paid for it with
four failures nobody could reproduce locally.

This module is the runner-agnostic half of the fix. :func:`clear_process_caches`
is the single definition of *what leaks*; ``conftest.py``'s fixture calls it, and
:class:`ProcessCacheIsolation` gives any ``TestCase`` the same guarantee through
``setUp``, which every runner honours because it is the framework's own contract
rather than one runner's extension point.

The three store redirects are called from the same two places, but from
``conftest.py``'s *import*, not its fixture: a store root has to be wrong-proof
before the first module is imported, not merely before each test. For the
output dir that is not belt-and-braces — ``api.py`` hands it to a
``StaticFiles`` mount at import, so nothing later would be early enough. Tests
that need a deployment's real path — rather than this process's sandbox — ask
:func:`deployment_cube_store_dir`, :func:`deployment_overlay_store_dir` or
:func:`deployment_output_dir`.

The API's rate limiter (``api.limiter``) is a fourth piece of process-global
state and rides along on the same hook, though it is not a cache: it counts
requests per user in process memory, so an un-neutralised limiter makes the
suite fail on its own traffic rather than merely serve a stale answer. See
:func:`disable_api_rate_limiter`, and :func:`rate_limiting_enabled` for the
opt-in a test uses when the limit *is* the thing under test.

If another process-global cache is ever added, :func:`clear_process_caches` is
the one place that has to learn about it — and ``test_cache_isolation.py``
asserts that it actually does.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sys
import tempfile

CUBE_STORE_ENV = "CUBE_STORE_DIR"
OVERLAY_STORE_ENV = "OVERLAY_STORE_DIR"
OUTPUT_DIR_ENV = "OUTPUT_DIR"
FRAME_STORE_ENV = "FRAME_STORE_DIR"

_cube_store_root: str | None = None
_isolated_roots: dict[str, str] = {}


def _isolate_store(env_var: str, prefix: str) -> str:
    """Point ``env_var`` at a throwaway directory for this whole process.

    The generalisation of :func:`isolate_cube_store` to the other two on-disk
    stores. Same reasoning throughout: the problem is *where the store is*, not
    what is in it, so clearing between tests cannot fix it and only moving the
    default can.

    Idempotent, and one directory per process rather than per call — a root that
    moved underneath the suite would strand whatever was already written to the
    old one.
    """
    if env_var in _isolated_roots:
        return _isolated_roots[env_var]

    root = tempfile.mkdtemp(prefix=prefix)
    _isolated_roots[env_var] = root
    os.environ[env_var] = root
    # ignore_errors for the same reason as the cube store: a still-open handle
    # on Windows would otherwise turn cleanup into an interpreter-exit traceback.
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    # Settings are lru_cached, so anything that read them before this point is
    # holding the real root. Local import: this module is imported by
    # ``conftest.py``, and at conftest import time the application packages are
    # not necessarily importable yet.
    try:
        from tta_backend.config.settings import get_settings

        get_settings.cache_clear()
    except Exception:  # noqa: BLE001 — isolation must not gate collection
        pass

    return root


def _deployment_dir(env_var: str, attr: str) -> str:
    """The path the *deployment* uses, ignoring the test redirect.

    Same contract as :func:`deployment_cube_store_dir`: once the redirect is in
    place, ``Settings()`` answers "where is this process's sandbox", not "where
    does this live in production". The deployment-contract tests need the
    latter, and asserting a volume contract against the sandbox would pass while
    checking nothing.

    Derived from the setting's own default rather than hard-coded, so it keeps
    telling the truth if the deployment path ever moves.
    """
    import unittest.mock

    from tta_backend.config.settings import Settings

    with unittest.mock.patch.dict(os.environ):
        os.environ.pop(env_var, None)
        return getattr(Settings(), attr)


def isolate_overlay_store() -> str:
    """Redirect T23's overlay PNG store (``plot_tools``).

    Until this existed the store was an ``APP_ROOT``-relative constant with an
    ``os.makedirs`` beside it, so merely *importing* ``plot_tools`` created
    ``Backend/overlay_store/`` in the checkout — a directory the developer uses
    and Docker backs with a named volume. Gitignored, so the pollution survived
    branch switches and never showed up in ``git status``.
    """
    return _isolate_store(OVERLAY_STORE_ENV, "tta-overlay-store-")


def deployment_overlay_store_dir() -> str:
    """The overlay store path the deployment uses — see :func:`_deployment_dir`."""
    return _deployment_dir(OVERLAY_STORE_ENV, "overlay_store_dir")


def isolate_frame_store() -> str:
    """Redirect T59's frame blob store (``services/frame_store.py``).

    Same reasoning as the cube store: its default is the deployment volume
    (``/app/frame_store``), which under Docker is production's named volume and
    on a Windows checkout resolves against the current drive as
    ``C:\\app\\frame_store``. Left alone, the suite would write stacks into a
    store it does not own — and this one *evicts*, so a test writing past the
    cap would delete a developer's or a deployment's entries, not merely add to
    them.
    """
    return _isolate_store(FRAME_STORE_ENV, "tta-frame-store-")


def deployment_frame_store_dir() -> str:
    """The frame store path the deployment uses — see :func:`_deployment_dir`."""
    return _deployment_dir(FRAME_STORE_ENV, "frame_store_dir")


def isolate_output_dir() -> str:
    """Redirect the public chart-output directory (``/app/outputs``).

    Unlike the overlay store this one is genuinely needed at import: ``api.py``
    hands it to a ``StaticFiles`` mount, which resolves the directory when it is
    mounted. So the ``os.makedirs`` there stays and only its *location* moves —
    which is why this, like :func:`isolate_cube_store`, has to run from
    ``conftest``'s import rather than a fixture.
    """
    return _isolate_store(OUTPUT_DIR_ENV, "tta-outputs-")


def deployment_output_dir() -> str:
    """The output dir the deployment uses — see :func:`_deployment_dir`."""
    return _deployment_dir(OUTPUT_DIR_ENV, "output_dir")


def isolate_cube_store() -> str:
    """Point T52's cube store at a throwaway directory for this whole process.

    Unlike the two in-memory caches, the cube store cannot be fixed by clearing
    it between tests, because the problem is *where it is*. Its default is the
    deployment volume (``/app/cube_store``) — a real, shared, long-lived
    directory. Under Docker that is production's named volume; on a Windows
    checkout the same POSIX default resolves against the current drive
    (``C:\\app\\cube_store``). Either way the suite was writing cubes and
    rewriting ``handle_index.json`` in a store it does not own, and reading it
    back, so a run's result depended on what earlier runs had left there.

    ``test_cube_cache.StoreTestCase`` already redirects per test, and still
    should — a test that asserts on store contents wants a store no other test
    has touched. But that is opt-in, and it *unwinds* at teardown: a cube write
    still in flight when the patch stops resolves the root afterwards and lands
    in the real store. That is how a ``handle_index.json`` outside any tempdir
    came to hold an ``obs_cubed`` entry pointing at a cube directory that does
    not exist beside it — a torn write, split across two stores.

    So this moves the floor rather than adding another opt-in: for the lifetime
    of the test process the *default* is a tempdir, and StoreTestCase's
    teardown now reverts to that instead of to somewhere real.

    Idempotent, and deliberately one directory per process rather than per call:
    cubes are written in the background and read back later, so a root that
    moved underneath the suite would turn every hit into a miss and quietly stop
    exercising the cache at all. Returns the root, mostly so a caller can say
    where it went.
    """
    global _cube_store_root
    if _cube_store_root is not None:
        return _cube_store_root

    _cube_store_root = tempfile.mkdtemp(prefix="tta-cube-store-")
    os.environ[CUBE_STORE_ENV] = _cube_store_root
    # Cubes are small here but not free, and a crashed run should not leave them
    # behind either; ignore_errors because a still-open Zarr handle on Windows
    # would otherwise turn cleanup into a spurious interpreter-exit traceback.
    atexit.register(shutil.rmtree, _cube_store_root, ignore_errors=True)

    # Settings are lru_cached, so anything that read them before this point is
    # holding the real root. Local import: this module is imported by
    # ``conftest.py``, and at conftest import time the application packages are
    # not necessarily importable yet.
    try:
        from tta_backend.config.settings import get_settings

        get_settings.cache_clear()
    except Exception:  # noqa: BLE001 — isolation must not gate collection
        pass

    return _cube_store_root


def deployment_cube_store_dir() -> str:
    """The cube store path the *deployment* uses, ignoring the test redirect.

    :func:`isolate_cube_store` sets ``CUBE_STORE_DIR``, so from inside the suite
    ``Settings().cube_store_dir`` no longer answers "where do cubes live in
    production" — it answers "where is this process's sandbox". The
    deployment-contract tests need the former (is that path on a named volume,
    is it created before the Dockerfile's chown), so they ask here instead.

    Derived from the setting's own default rather than hard-coded, so it keeps
    telling the truth if the deployment path ever moves.
    """
    import unittest.mock

    from tta_backend.config.settings import Settings

    with unittest.mock.patch.dict(os.environ):
        os.environ.pop(CUBE_STORE_ENV, None)
        return Settings().cube_store_dir


def disable_api_rate_limiter() -> None:
    """Take the API's rate limiter out of the suite's way.

    ``api.limiter`` counts against a process-lifetime ``MemoryStorage`` keyed by
    user id, so it is process-global state in exactly the sense this module
    exists for — but it leaks *forward* in a way the caches above do not. A
    cache leak serves a stale answer; this one refuses a request outright. The
    endpoints carry per-minute limits and a test file makes many calls as one
    user inside one minute, so without this the suite fails on its own traffic:
    ``/debug/heap-snapshot`` allows 1/minute and its third test was already
    getting a 429 rather than the 200 it asserts.

    Disabled rather than merely reset between tests. Resetting would fix the
    across-tests case only, and would still leave any single test that calls a
    tight endpoint twice failing on a limit it never asked to exercise. The
    limits are production policy; a test that wants to exercise them says so
    explicitly with :func:`rate_limiting_enabled`.

    Looked up through ``sys.modules`` instead of imported. Importing
    ``tta_backend.api`` here would force the whole application graph — and its
    import-time ``CRS.from_epsg`` — on every ``TestCase`` that mixes in
    :class:`ProcessCacheIsolation`, including under bare ``unittest``, which
    never loads ``conftest.py`` and so has not wired PROJ. If nothing has
    imported the API yet, the environment variable above is what disarms the
    limiter it will build when something finally does -- which is the case that
    matters, since the fixture runs before the lazy import in most setUps here.
    """
    # Set unconditionally, and first. Under pytest conftest.py has already done
    # this before collection; under bare unittest nothing has, and a module
    # imported later in this process would otherwise build an armed limiter.
    os.environ["RATE_LIMITING_ENABLED"] = "0"
    api = sys.modules.get("tta_backend.api")
    if api is None:
        return
    api.limiter.enabled = False
    api.limiter.reset()


@contextlib.contextmanager
def rate_limiting_enabled():
    """Turn the limiter back on for one test, on a cleared counter.

    The opt-in half of :func:`disable_api_rate_limiter`. Clears on the way in so
    the test starts from a known count rather than from whatever the file's
    earlier requests left, and restores the disabled default on the way out even
    if the test fails — otherwise one failing test would re-arm the limiter for
    everything that ran after it, which is the exact cross-test coupling the
    default is there to prevent.
    """
    from tta_backend import api

    api.limiter.reset()
    api.limiter.enabled = True
    try:
        yield api.limiter
    finally:
        api.limiter.enabled = False
        api.limiter.reset()


def clear_process_caches() -> None:
    """Drop every process-global cache a test could leave behind.

    Imports are local: this module is imported by test modules *and* by
    ``conftest.py``, and at ``conftest`` import time the application packages
    are not necessarily importable yet.

    Clearing a cache can only ever cause a miss, so this is safe to call
    unconditionally — including for tests that never touch either cache. The
    rate limiter is here for the same reason but is not a cache; see
    :func:`disable_api_rate_limiter`.
    """
    from tta_backend.earthdata_mcp.tool_cache import clear_tool_cache
    from tta_backend.services.jobs_service import clear_terminal_status_cache

    clear_tool_cache()
    clear_terminal_status_cache()
    disable_api_rate_limiter()


class ProcessCacheIsolation:
    """Mixin giving a ``TestCase`` a cold cache, whatever the runner.

    Mix in **before** the ``TestCase`` base so this ``setUp`` runs first::

        class MyTests(ProcessCacheIsolation, unittest.IsolatedAsyncioTestCase):
            ...

    Clears on the way in *and* registers a cleanup for the way out: the entry
    clear is what protects this test from whatever ran before it, and the exit
    clear is what protects the next test from this one. Either alone would leave
    a gap at one end of the suite.

    ``setUp`` rather than an autouse fixture on purpose — ``setUp`` is
    ``unittest``'s own contract, so pytest, ``unittest discover`` and a bare
    ``python -m unittest`` all honour it identically. That is the whole point:
    the isolation must not depend on the runner having loaded ``conftest.py``.

    ``setUp`` alone covers async cases too: ``IsolatedAsyncioTestCase`` calls
    ``setUp`` *and then* ``asyncSetUp``, so the cache is already cold before a
    subclass's ``asyncSetUp`` builds anything — and hooking only ``setUp``
    keeps this working for the many classes here whose ``asyncSetUp`` does not
    call ``super()``.

    A subclass that overrides ``setUp`` must call ``super().setUp()``, or it
    opts itself back out of isolation.
    """

    def setUp(self) -> None:  # noqa: D102 - contract documented on the class
        isolate_cube_store()
        isolate_overlay_store()
        isolate_output_dir()
        isolate_frame_store()
        clear_process_caches()
        self.addCleanup(clear_process_caches)
        super().setUp()
