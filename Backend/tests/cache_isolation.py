"""Cache isolation, independent of which runner is executing.

Three caches outlive a single test by design — T53's discovery-metadata cache
(``earthdata_mcp.tool_cache``), the jobs terminal-status cache
(``services.jobs_service``), and T52's Zarr cube store. The first two are
process-lifetime *memory*, so unless something clears them between tests, one
test's call is served to the next test's identical call and that test's handler
never runs. The third is *disk*, and needs the opposite treatment — see
:func:`isolate_cube_store`.

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

:func:`isolate_cube_store` is called from the same two places, but from
``conftest.py``'s *import*, not its fixture: a store root has to be wrong-proof
before the first module is imported, not merely before each test. Tests that
need the deployment's real path — rather than this process's sandbox — ask
:func:`deployment_cube_store_dir`.

If another process-global cache is ever added, :func:`clear_process_caches` is
the one place that has to learn about it — and ``test_cache_isolation.py``
asserts that it actually does.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

CUBE_STORE_ENV = "CUBE_STORE_DIR"

_cube_store_root: str | None = None


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


def clear_process_caches() -> None:
    """Drop every process-global cache a test could leave behind.

    Imports are local: this module is imported by test modules *and* by
    ``conftest.py``, and at ``conftest`` import time the application packages
    are not necessarily importable yet.

    Clearing a cache can only ever cause a miss, so this is safe to call
    unconditionally — including for tests that never touch either cache.
    """
    from tta_backend.earthdata_mcp.tool_cache import clear_tool_cache
    from tta_backend.services.jobs_service import clear_terminal_status_cache

    clear_tool_cache()
    clear_terminal_status_cache()


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
        clear_process_caches()
        self.addCleanup(clear_process_caches)
        super().setUp()
