"""
Backend/conftest.py
-------------------
Pytest bootstrap. Three jobs, all of which must happen before any test module
is imported.

**1. Put ``Backend/`` on ``sys.path``** so ``import tta_backend...`` resolves
in a checkout where the package has not been ``pip install -e``'d. This lived
copy-pasted at the top of 102 test files, each carrying a stale "TODO: remove
after pyproject.toml install" — stale because pyproject *did* land, but its
flat include list omitted ``api`` and ``earthdata_mcp``, so the hack stayed
load-bearing anyway. The tta_backend.* refactor fixed the packaging; this is
the one remaining reason the path needs help, and one copy is enough now that
pytest is the runner everywhere (CI, ``--profile test``, and locally).

Consequence worth knowing: ``python Backend/tests/test_x.py`` no longer works
on its own. Use ``python -m pytest Backend/tests/test_x.py``. The ``__main__``
blocks those files still carry are vestigial from the unittest era.

**2. Point PROJ at rasterio's bundled data directory** before any test module
(transitively) imports ``tta_backend.utils.overlay_render``, which builds a
CRS at import time (``CRS.from_epsg(4326)``).

On this Windows checkout ``PROJ_LIB`` is inherited from a *different* Python
(a conda ``airPollution`` env) than the pip-installed rasterio the tests run
under, so GDAL follows that stale path and fails collection-wide with
``rasterio.errors.CRSError: Cannot find proj.db``. We therefore don't just
fill in an *unset* PROJ path — we also override one that points somewhere
without a usable ``proj.db``. When the inherited path is already valid, or
rasterio's own data dir can't be located, this is a no-op. The Docker image,
where PROJ resolves correctly on its own, is never touched.

**3. Redirect the on-disk stores to tempdirs** so the suite stops reading and
writing the real ones — T52's cube store, T23's overlay store, and the public
chart-output directory. See :func:`_isolate_on_disk_stores`.
"""
import importlib.util
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# tta_backend.api caches get_settings() in a module-level variable at import
# time, so whichever test file happens to import it first (collection order
# is filesystem-dependent, not alphabetical-guaranteed) permanently decides
# what the Supabase settings are for the whole run. This used to be
# JWT_SECRET_KEY, set only as a side effect of test_chat_endpoint.py importing
# before test_auth_endpoints.py -- fragile, and silently masked locally by a
# real value in .env. The reasoning survives T61 unchanged: set these here,
# before any test module is imported, so it can't depend on import order again.
#
# The host must be syntactically real but must never be reached. Verification
# runs against a fixture keypair injected in place of the verifier, so nothing
# in the suite should ever resolve this name -- if a test starts making DNS
# queries for it, something is talking to the live identity provider.
os.environ.setdefault("SUPABASE_URL", "https://test-project.supabase.co")
os.environ.setdefault("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")

# Same "before any test module is imported" reasoning, for the same reason:
# api.py decides at import whether its limiter is armed. Set here rather than
# from the fixture because a test file that imports api.py lazily (inside
# asyncSetUp, which most of the endpoint tests do) imports it *after* the
# fixture has run, and by then an armed limiter is already counting. Assigned,
# not setdefault: the suite must not inherit an ambient value that re-arms it.
os.environ["RATE_LIMITING_ENABLED"] = "0"


def _has_proj_db(path: str | None) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "proj.db"))


def _wire_proj_data() -> None:
    # Respect an already-valid PROJ path (e.g. the Docker image's own setup).
    if _has_proj_db(os.environ.get("PROJ_DATA")) or _has_proj_db(os.environ.get("PROJ_LIB")):
        return
    # Locate rasterio's data dir WITHOUT importing the package — importing it
    # here would initialize GDAL/PROJ before the env var is set, caching the
    # wrong (undiscoverable) data dir and defeating the whole point.
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.submodule_search_locations:
        return
    proj_data = os.path.join(list(spec.submodule_search_locations)[0], "proj_data")
    if _has_proj_db(proj_data):
        # Override the inherited-but-broken PROJ_LIB, not just an unset one.
        os.environ["PROJ_DATA"] = proj_data
        os.environ["PROJ_LIB"] = proj_data


_wire_proj_data()


def _isolate_on_disk_stores() -> None:
    """Redirect every on-disk store before any test module is imported.

    At import time, not in a fixture, and for the same reason as PROJ above:
    by the time a fixture runs, a module-scope or session-scope fixture may
    already have resolved the store root, and a cube write started under one
    test can still be in flight during the next. There is no point in the
    session at which it is safe for the default to be a real directory.

    Left un-redirected, T52's cube store defaults to the deployment volume —
    production's named volume under Docker, ``C:\\app\\cube_store`` on a Windows
    checkout — which the suite then reads back, making a run's outcome depend on
    what earlier runs left behind.

    T23's overlay store and the public output dir were worse: ``APP_ROOT``-
    relative constants with an ``os.makedirs`` beside them, so they landed
    *inside the checkout* and a bare import was enough to create them. For the
    output dir the import-time call is load-bearing rather than merely early —
    ``api.py`` hands it to a ``StaticFiles`` mount, which resolves the directory
    when it is mounted, so a fixture would already be too late.

    The policy lives in ``tests/cache_isolation.py`` with the rest of it; this
    is just the call site early enough to matter.
    """
    tests_dir = os.path.join(BACKEND_DIR, "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from cache_isolation import (
        isolate_cube_store,
        isolate_frame_store,
        isolate_output_dir,
        isolate_overlay_store,
    )

    isolate_cube_store()
    isolate_overlay_store()
    isolate_output_dir()
    isolate_frame_store()


_isolate_on_disk_stores()


@pytest.fixture(autouse=True)
def _isolate_process_caches():
    """Blanket cache isolation, for pytest.

    T53's discovery-metadata cache is process-global by design, so without this
    one test's ``search_datasets(query="no2")`` would be served to the next
    test's identical call and its handler would never run — a shared-state leak,
    not a behavior change.

    The *policy* — which caches are process-global and therefore leak — lives in
    ``tests/cache_isolation.py``, not here, because this file is pytest-only and
    the leak is not: ``unittest`` never loads a conftest, so CI's
    ``unittest discover`` ran without this fixture and four tests failed on state
    earlier ones had left. Anything that must hold under every runner cannot live
    in this file. This fixture is now just pytest's adapter onto the shared
    policy; a ``TestCase`` gets the same guarantee runner-independently by mixing
    in ``cache_isolation.ProcessCacheIsolation``.
    """
    import sys

    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from cache_isolation import clear_process_caches

    clear_process_caches()
    yield
    clear_process_caches()
