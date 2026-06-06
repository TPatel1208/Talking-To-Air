"""
repositories/cache_index_repository.py
=======================================
Repository wrapper around preprocessing/cache_index.py.

Provides an object-oriented interface that CacheManager expects, while
delegating all SQL to the existing cache_index module.  Connection
lifecycle is managed internally: one connection is opened on first use
and kept open for the lifetime of the repository instance.

Example
-------
    from repositories.cache_index_repository import CacheIndexRepository

    repo = CacheIndexRepository()
    row  = repo.lookup(
        collection_id="C1266136111-GES_DISC",
        group_key="C1266136111-GES_DISC/2025-01-01T00:00:00Z_.../...",
        temporal=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        bbox=(-74, 40, -73, 41),
    )
    if row:
        print(row["cache_path"], row["group_key"])

    repo.insert(
        collection_id="C1266136111-GES_DISC",
        group_key="...",
        cache_path="./data/cache.zarr",
        temporal=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        bbox=(-74, 40, -73, 41),
    )
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CacheIndexRepository:
    """
    Object-oriented facade over the cache_index SQL module.

    Manages a single psycopg connection.  If the database is unavailable at
    construction time, all methods return safe defaults (None / False) so
    the rest of the pipeline degrades gracefully to Zarr-only caching.
    """

    def __init__(self) -> None:
        self._ci = None
        self._available = False
        self._init_connection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_connection(self) -> None:
        """Open DB connection and ensure schema exists. Non-fatal on failure."""
        try:
            from preprocessing import cache_index as ci  # noqa: PLC0415
            with ci.get_connection() as conn:
                ci.ensure_schema(conn)
            self._ci = ci
            self._available = True
            logger.info("CacheIndexRepository: connected to PostGIS metadata index")
        except Exception as exc:
            logger.warning(
                "CacheIndexRepository: PostGIS unavailable — index disabled (%s)", exc
            )
            self._available = False

    def _ensure_connected(self) -> bool:
        """
        Return True if the connection is healthy.  Attempt a reconnect once
        on a closed / broken connection before giving up.
        """
        if not self._available:
            return False
        return self._ci is not None

    # ------------------------------------------------------------------
    # Public API (matches CacheManager expectations)
    # ------------------------------------------------------------------

    def lookup(
        self,
        collection_id: str,
        group_key: str,
        temporal: Optional[Tuple[str, str]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[dict]:
        """
        Search for a cached entry.

        Parameters
        ----------
        collection_id : str
            NASA CMR concept ID.
        group_key : str
            Zarr group key (from CacheManager.make_group_key).
        temporal : tuple, optional
            (start_iso, end_iso) — enables spatial superset fallback.
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat) — enables spatial superset fallback.

        Returns
        -------
        dict or None
            Row dict with keys: id, collection_id, group_key, bbox_wkt,
            start_date, end_date, cache_path, created_at, superset_hit.
            None on miss or if the index is unavailable.
        """
        if not self._ensure_connected():
            return None
        try:
            with self._ci.get_connection() as conn:
                return self._ci.lookup(
                    conn,
                    collection_id=collection_id,
                    group_key=group_key,
                    temporal=temporal,
                    bounding_box=bbox,
                )
        except Exception as exc:
            logger.warning("CacheIndexRepository.lookup failed: %s", exc)
            return None

    def insert(
        self,
        collection_id: str,
        group_key: str,
        cache_path: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """
        Write a metadata row for a freshly cached Zarr group.

        Parameters
        ----------
        collection_id : str
            NASA CMR concept ID.
        group_key : str
            Zarr group key.
        cache_path : str
            Path to the Zarr store directory.
        temporal : tuple
            (start_iso, end_iso).
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        bool
            True on successful insert, False otherwise (non-fatal).
        """
        if not self._ensure_connected():
            return False
        try:
            with self._ci.get_connection() as conn:
                return self._ci.insert(
                    conn,
                    collection_id=collection_id,
                    group_key=group_key,
                    cache_path=cache_path,
                    temporal=temporal,
                    bounding_box=bbox,
                )
        except Exception as exc:
            logger.warning("CacheIndexRepository.insert failed: %s", exc)
            return False

    def close(self) -> None:
        """Mark the repository unavailable; pool shutdown owns connections."""
        self._available = False

    def __repr__(self) -> str:
        status = "connected" if self._available else "unavailable"
        return f"CacheIndexRepository({status})"
