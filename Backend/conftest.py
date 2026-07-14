"""
Backend/conftest.py
-------------------
Pytest bootstrap. Point PROJ at rasterio's bundled data directory before any
test module (transitively) imports ``utils.overlay_render``, which builds a
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
