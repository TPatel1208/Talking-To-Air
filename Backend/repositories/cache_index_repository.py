"""
Async repository for the satellite Zarr cache metadata index.

All database work goes through the shared async connection pool in utils.db.
The repository does not retain a connection between calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from utils.db import pg_connection

logger = logging.getLogger(__name__)


def _row_to_dict(cursor, row) -> dict:
    columns = [column.name for column in cursor.description]
    return dict(zip(columns, row))


def _normalise_bbox(bbox) -> Tuple[float, float, float, float]:
    while isinstance(bbox, (list, tuple)) and len(bbox) == 1:
        bbox = bbox[0]

    if isinstance(bbox, str):
        parts = [float(x) for x in bbox.split(",")]
    else:
        parts = [float(x) for x in bbox]

    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(parts)}: {bbox!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _bbox_to_polygon_wkt(bbox) -> str:
    min_lon, min_lat, max_lon, max_lat = _normalise_bbox(bbox)
    return (
        "POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, "
        f"{min_lon} {max_lat}, "
        f"{min_lon} {min_lat}"
        "))"
    )


class CacheIndexRepository:
    """Pool-backed async facade over zarr_cache_entries."""

    async def lookup(
        self,
        collection_id: str,
        group_key: str,
        temporal: Optional[Tuple[str, str]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[dict]:
        try:
            async with pg_connection() as conn:
                row = await self._lookup_exact(conn, group_key)
                if row is None and temporal is not None and bbox is not None:
                    row = await self._lookup_spatial(conn, collection_id, temporal, bbox)
                if row is not None:
                    await self._mark_accessed(conn, int(row["id"]))
                return row
        except Exception as exc:
            logger.warning("CacheIndexRepository.lookup failed: %s", exc)
            return None

    async def insert(
        self,
        collection_id: str,
        group_key: str,
        cache_path: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        sql = """
            INSERT INTO zarr_cache_entries
                (collection_id, group_key, bbox, start_date, end_date, cache_path)
            VALUES
                (%s, %s,
                 CASE WHEN %s::text IS NOT NULL
                      THEN ST_GeomFromText(%s::text, 4326)
                      ELSE NULL
                 END,
                 %s::timestamptz, %s::timestamptz, %s)
            ON CONFLICT (group_key) DO UPDATE
            SET cache_path = EXCLUDED.cache_path,
                last_accessed_at = now()
            RETURNING id
        """
        bbox_wkt = _bbox_to_polygon_wkt(bbox) if bbox else None
        start_iso, end_iso = temporal
        try:
            async with pg_connection() as conn:
                cursor = await conn.execute(
                    sql,
                    (
                        collection_id,
                        group_key,
                        bbox_wkt,
                        bbox_wkt,
                        start_iso,
                        end_iso,
                        cache_path,
                    ),
                )
                returned = await cursor.fetchone()
            return returned is not None
        except Exception as exc:
            logger.warning("CacheIndexRepository.insert failed: %s", exc)
            return False

    async def list_prunable(self, older_than_days: int) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        sql = """
            SELECT id, group_key, cache_path, created_at, last_accessed_at
            FROM zarr_cache_entries
            WHERE COALESCE(last_accessed_at, created_at) < %s
            ORDER BY COALESCE(last_accessed_at, created_at) ASC
        """
        async with pg_connection() as conn:
            cursor = await conn.execute(sql, (cutoff,))
            rows = await cursor.fetchall()
        return [_row_to_dict(cursor, row) for row in rows]

    async def delete_entries(self, entry_ids: list[int]) -> int:
        if not entry_ids:
            return 0
        async with pg_connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM zarr_cache_entries WHERE id = ANY(%s) RETURNING id",
                (entry_ids,),
            )
            rows = await cursor.fetchall()
        return len(rows)

    async def _lookup_exact(self, conn, group_key: str) -> Optional[dict]:
        cursor = await conn.execute(
            """
            SELECT
                id, collection_id, group_key,
                ST_AsText(bbox) AS bbox_wkt,
                start_date, end_date, cache_path, created_at, last_accessed_at
            FROM zarr_cache_entries
            WHERE group_key = %s
            LIMIT 1
            """,
            (group_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        result = _row_to_dict(cursor, row)
        result["superset_hit"] = False
        return result

    async def _lookup_spatial(
        self,
        conn,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Tuple[float, float, float, float],
    ) -> Optional[dict]:
        start_iso, end_iso = temporal
        cursor = await conn.execute(
            """
            SELECT
                id, collection_id, group_key,
                ST_AsText(bbox) AS bbox_wkt,
                start_date, end_date, cache_path, created_at, last_accessed_at
            FROM zarr_cache_entries
            WHERE collection_id = %s
              AND start_date <= %s::timestamptz
              AND end_date >= %s::timestamptz
              AND (bbox IS NULL OR ST_Covers(bbox, ST_GeomFromText(%s, 4326)))
            ORDER BY COALESCE(last_accessed_at, created_at) DESC
            LIMIT 1
            """,
            (collection_id, start_iso, end_iso, _bbox_to_polygon_wkt(bbox)),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        result = _row_to_dict(cursor, row)
        result["superset_hit"] = True
        return result

    async def _mark_accessed(self, conn, entry_id: int) -> None:
        await conn.execute(
            "UPDATE zarr_cache_entries SET last_accessed_at = now() WHERE id = %s",
            (entry_id,),
        )

    async def close(self) -> None:
        """Compatibility hook; pooled connections are owned by utils.db."""

    def __repr__(self) -> str:
        return "CacheIndexRepository(async)"
