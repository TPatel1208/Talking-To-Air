"""Process-global cache isolation, independent of which runner is executing.

Two caches outlive a single test by design — T53's discovery-metadata cache
(``earthdata_mcp.tool_cache``) and the jobs terminal-status cache
(``services.jobs_service``). Both are process-lifetime state, so unless
something clears them between tests, one test's call is served to the next
test's identical call and that test's handler never runs.

Until now the only thing clearing them was an autouse fixture in
``Backend/conftest.py``. pytest loads that file; ``unittest`` does not. So the
suite was isolated under one runner and silently order-dependent under the
other, and CI — which ran ``unittest discover`` — paid for it with four
failures nobody could reproduce locally.

This module is the runner-agnostic half of the fix. :func:`clear_process_caches`
is the single definition of *what leaks*; ``conftest.py``'s fixture calls it, and
:class:`ProcessCacheIsolation` gives any ``TestCase`` the same guarantee through
``setUp``, which every runner honours because it is the framework's own contract
rather than one runner's extension point.

If a third process-global cache is ever added, :func:`clear_process_caches` is
the one place that has to learn about it — and ``test_cache_isolation.py``
asserts that it actually does.
"""

from __future__ import annotations



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
        clear_process_caches()
        self.addCleanup(clear_process_caches)
        super().setUp()
