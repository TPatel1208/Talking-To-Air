"""
Backend/conftest.py
-------------------
Pytest bootstrap. Two jobs, both of which must happen before any test module
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
"""
import importlib.util
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


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
