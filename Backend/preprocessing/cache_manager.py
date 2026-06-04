"""
preprocessing/cache_manager.py
==============================
Orchestrates three-tier cache lookup: PostGIS → Zarr → Harmony (on miss).

Depends on repositories (ZarrRepository, CacheIndexRepository) but not on
Harmony or file parsing logic — those are injected as needed.

Example
-------
    from preprocessing.cache_manager import CacheManager
    from repositories.zarr_repository import ZarrRepository

    zarr_repo = ZarrRepository("cache.zarr")
    cache_mgr = CacheManager(zarr_repo, index_repo=None)

    # Lookup
    ds = cache_mgr.lookup(
        collection_id="C1266136111-GES_DISC",
        temporal=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        bbox=(-74, 40, -73, 41),
    )

    # Store
    cache_mgr.store(ds, collection_id, temporal, bbox, cache_path="cache.zarr")
"""

import hashlib
import logging
from typing import Optional, Tuple

import xarray as xr

logger = logging.getLogger(__name__)


def _normalise_bbox(bbox) -> Tuple[float, float, float, float]:
    """Return bbox as (min_lon, min_lat, max_lon, max_lat)."""
    while isinstance(bbox, (list, tuple)) and len(bbox) == 1:
        bbox = bbox[0]

    if isinstance(bbox, str):
        parts = [float(x) for x in bbox.split(",")]
    else:
        parts = [float(x) for x in bbox]

    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}: {bbox!r}")
    return parts[0], parts[1], parts[2], parts[3]


def make_group_key(
    collection_id: str,
    start_time: str,
    end_time: str,
    bbox: Tuple[float, ...] = (),
) -> str:
    """
    Generate a stable Zarr group key from collection and constraints.

    Format: {collection_id}/{temporal}/{spatial}
    where temporal and spatial are hashed for brevity.

    Example
    -------
    >>> make_group_key("C1266136111-GES_DISC", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", (-74, 40, -73, 41))
    'C1266136111-GES_DISC/2025-01-01T00:00:00Z_2025-01-02T00:00:00Z/-74_40_-73_41'
    """
    temporal_key = f"{start_time}_{end_time}"
    bbox = _normalise_bbox(bbox) if bbox else ()
    spatial_key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}" if bbox else "global"
    return f"{collection_id}/{temporal_key}/{spatial_key}"


class CacheManager:
    """Coordinate three-tier cache: PostGIS index → Zarr store → Harmony (on miss)."""

    def __init__(self, zarr_repository, index_repository=None):
        """
        Parameters
        ----------
        zarr_repository : ZarrRepository
            Zarr store accessor.
        index_repository : CacheIndexRepository, optional
            PostGIS metadata index. If None, only Zarr-level lookups work.
        """
        self.zarr_repo = zarr_repository
        self.index_repo = index_repository

    def lookup(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[xr.Dataset]:
        """
        Three-tier cache lookup.

        1. PostGIS exact / superset hit (B-tree, fast)
        2. Zarr group key check (fallback)
        3. None (cache miss)

        Parameters
        ----------
        collection_id : str
            NASA CMR concept ID.
        temporal : tuple
            (start_iso, end_iso) in 'YYYY-MM-DDTHH:MM:SSZ' format.
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat). None = global.

        Returns
        -------
        xr.Dataset or None
            The cached dataset if found, else None.
        """
        group_key = make_group_key(
            collection_id,
            temporal[0],
            temporal[1],
            bbox if bbox else (),
        )

        logger.info("Cache lookup: collection=%s temporal=%s bbox=%s", collection_id, temporal, bbox)

        # ─────────────────────────────────────────────────────────────────
        # Tier 1: PostGIS index (if available)
        # ─────────────────────────────────────────────────────────────────
        if self.index_repo is not None:
            try:
                row = self.index_repo.lookup(
                    collection_id=collection_id,
                    group_key=group_key,
                    temporal=temporal,
                    bbox=bbox,
                )
                if row:
                    stored_path = row["cache_path"]
                    cached_group_key = row["group_key"]
                    is_superset = row.get("superset_hit", False)

                    logger.info(
                        "Index hit (%s): %s group=%s",
                        "superset" if is_superset else "exact",
                        stored_path,
                        cached_group_key,
                    )

                    try:
                        ds = self.zarr_repo.read(cached_group_key)
                        if is_superset and bbox:
                            ds = self._subset_bbox(ds, bbox)
                        return ds
                    except Exception as exc:
                        logger.warning(
                            "Index pointed to stale entry (%s): %s — falling through",
                            stored_path,
                            exc,
                        )

            except Exception as exc:
                logger.warning("Index lookup error (non-fatal): %s", exc)

        # ─────────────────────────────────────────────────────────────────
        # Tier 2: Zarr store (if DB unavailable or miss)
        # ─────────────────────────────────────────────────────────────────
        if self.zarr_repo.exists(group_key):
            logger.info("Zarr hit: group=%s", group_key)
            return self.zarr_repo.read(group_key)

        logger.info("Cache miss")
        return None

    def store(
        self,
        ds: xr.Dataset,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]],
        cache_path: str,
    ) -> None:
        """
        Write dataset to Zarr and metadata row to PostGIS index.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to cache.
        collection_id : str
            NASA CMR concept ID.
        temporal : tuple
            (start_iso, end_iso).
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat). None = global.
        cache_path : str
            Path to Zarr store.
        """
        group_key = make_group_key(
            collection_id,
            temporal[0],
            temporal[1],
            bbox if bbox else (),
        )

        logger.info("Storing to cache: collection=%s group=%s", collection_id, group_key)

        # Write Zarr
        try:
            self.zarr_repo.write(ds, group_key)
        except Exception as exc:
            logger.error("Zarr write failed: %s", exc)
            raise

        # Insert metadata row
        if self.index_repo is not None:
            try:
                self.index_repo.insert(
                    collection_id=collection_id,
                    group_key=group_key,
                    cache_path=cache_path,
                    temporal=temporal,
                    bbox=bbox,
                )
            except Exception as exc:
                logger.warning(
                    "Index insert failed (non-fatal — data is safely cached): %s", exc
                )

    @staticmethod
    def _subset_bbox(
        ds: xr.Dataset, bbox: Tuple[float, float, float, float]
    ) -> xr.Dataset:
        """
        Subset a cached dataset to a smaller bounding box.

        Only safe when lat/lon are 1-D and monotonic.

        Parameters
        ----------
        ds : xr.Dataset
            Cached dataset (likely a superset).
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        xr.Dataset
            Subset dataset.
        """
        min_lon, min_lat, max_lon, max_lat = _normalise_bbox(bbox)

        try:
            if "latitude" in ds and "longitude" in ds:
                ds = ds.sel(
                    latitude=slice(min_lat, max_lat),
                    longitude=slice(min_lon, max_lon),
                )
            elif "lat" in ds and "lon" in ds:
                ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

            logger.info("Subset cached dataset to bbox=%s", bbox)
            return ds
        except Exception as exc:
            logger.warning("Subsetting failed: %s — returning full dataset", exc)
            return ds
