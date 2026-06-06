"""
preprocessing/__init__.py
=========================
Factory that assembles the production data pipeline.

Usage
-----
    from preprocessing import build_data_pipeline

    cache_manager = build_data_pipeline()
    ds = cache_manager.lookup(collection_id, temporal, bbox)
"""

import logging

logger = logging.getLogger(__name__)


def build_data_pipeline(zarr_store: str = "./data/cache.zarr"):
    """
    Assemble and return a CacheManager wired to the real Zarr and PostGIS
    repositories.

    The PostGIS index is optional: if the database is unreachable,
    CacheIndexRepository silently disables itself and CacheManager falls
    back to Zarr-only lookup.

    Parameters
    ----------
    zarr_store : str
        Path to the Zarr store directory (created on first write).

    Returns
    -------
    CacheManager
    """
    from repositories.zarr_repository import ZarrRepository
    from repositories.cache_index_repository import CacheIndexRepository
    from preprocessing.cache_manager import CacheManager

    zarr_repo = ZarrRepository(zarr_store)

    try:
        index_repo = CacheIndexRepository()
    except Exception as exc:
        logger.warning("build_data_pipeline: index repo unavailable — %s", exc)
        index_repo = None

    return CacheManager(zarr_repo, index_repo)


__all__ = ["build_data_pipeline"]
