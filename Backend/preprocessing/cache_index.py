"""
cache_index.py
==============
Metadata catalog for the satellite Zarr cache, backed by PostgreSQL + PostGIS.

Each row describes one cached Zarr group:
  - what collection it belongs to
  - which group_key addresses it inside the Zarr store
  - the geographic bounding box  (PostGIS geometry, EPSG:4326)
  - the temporal extent
  - the filesystem path to the Zarr store

Public API
----------
  get_connection()          -> psycopg connection (caller must close / use as ctx-mgr)
  ensure_schema(conn)       -> creates the table + indexes if they don't exist yet
  lookup(conn, ...)         -> find a cached entry; returns row dict or None
  insert(conn, ...)         -> write a metadata row after a successful cache write
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from config.settings import get_settings
from utils.geo_utils import normalise_bbox

logger = logging.getLogger(__name__)


_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS zarr_cache_entries (
    id            SERIAL PRIMARY KEY,
    collection_id TEXT        NOT NULL,
    group_key     TEXT        NOT NULL,
    bbox          geometry(Polygon, 4326),
    start_date    TIMESTAMPTZ NOT NULL,
    end_date      TIMESTAMPTZ NOT NULL,
    cache_path    TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE zarr_cache_entries
    ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- spatial index for bbox overlap queries
CREATE INDEX IF NOT EXISTS idx_zarr_cache_bbox
    ON zarr_cache_entries USING GIST (bbox);

-- range index for temporal queries
CREATE INDEX IF NOT EXISTS idx_zarr_cache_dates
    ON zarr_cache_entries (start_date, end_date);

-- uniqueness guard: one row per logical dataset slice
CREATE UNIQUE INDEX IF NOT EXISTS idx_zarr_cache_group_key
    ON zarr_cache_entries (group_key);

CREATE INDEX IF NOT EXISTS idx_zarr_cache_last_accessed
    ON zarr_cache_entries (last_accessed_at);
"""

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _dsn() -> str:
    """Build a libpq DSN from the same DB * env vars used by the rest of the app."""
    settings = get_settings()
    host = settings.db_host
    port = settings.db_port
    dbname = settings.db_name
    user = settings.db_user
    password = settings.db_password or ""
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_connection():
    """
    Return an open psycopg (v3) connection.

    The caller is responsible for closing the connection when done.
    Use conn.close() explicitly, or wrap in contextlib.closing().

    Note: in psycopg v3, using a connection as a context manager (``with conn:``)
    commits/rolls back the current transaction AND closes the connection on exit.
    cache_index avoids that pattern — it uses ``with conn.cursor() as cur:`` only,
    and leaves connection lifecycle to the caller (data_loader.py).
    """
    try:
        import psycopg
        return psycopg.connect(**get_settings().db_kwargs)
    except Exception as exc:
        logger.error("cache_index: could not connect to PostgreSQL — %s", exc)
        raise


def ensure_schema(conn=None) -> None:
    """
    Create the zarr_cache_entries table and its indexes if they don't exist.
    Accepts an existing connection or opens (and closes) one internally.
    """
    if conn is None:
        with get_connection() as owned_conn:
            ensure_schema(owned_conn)
        return

    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
        logger.info("cache_index: schema ensured")
    except Exception as exc:
        logger.error("cache_index: schema creation failed — %s", exc)
        if conn:
            conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _bbox_to_polygon_wkt(bbox) -> str:
    """
    Convert (min_lon, min_lat, max_lon, max_lat) to a WKT polygon string
    suitable for ST_GeomFromText(..., 4326).
    """
    min_lon, min_lat, max_lon, max_lat = normalise_bbox(bbox)
    return (
        f"POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, "
        f"{min_lon} {max_lat}, "
        f"{min_lon} {min_lat}"
        f"))"
    )


def lookup(
    conn,
    collection_id: str,
    group_key: str,
    temporal: Optional[Tuple[str, str]] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[dict]:
    """
    Search for a cached entry that matches the given parameters.

    The lookup strategy is:
      1. Exact match on ``group_key``  (cheapest — uses the unique B-tree index)
      2. Spatial superset search via ``lookup_spatial`` when (a) the exact match
         misses, (b) both ``temporal`` and ``bounding_box`` are provided.

    Returns a row dict with keys matching the table columns, or None if no
    matching entry is found.  When the result comes from a spatial superset hit
    the dict contains an extra key ``"superset_hit": True`` so the caller can
    trim the dataset to the originally requested bbox.

    Parameters
    ----------
    conn          : open psycopg connection
    collection_id : NASA CMR concept-id
    group_key     : MD5 key computed by DataLoader.make_group_key()
    temporal      : (start_iso, end_iso) — required for the spatial fallback
    bounding_box  : (min_lon, min_lat, max_lon, max_lat) — required for spatial fallback
    """
    _SELECT = """
        SELECT
            id, collection_id, group_key,
            ST_AsText(bbox) AS bbox_wkt,
            start_date, end_date, cache_path, created_at
        FROM zarr_cache_entries
        WHERE group_key = %s
        LIMIT 1
    """
    try:
        with conn.cursor() as cur:
            cur.execute(_SELECT, (group_key,))
            row = cur.fetchone()
        if row is not None:
            col_names = ["id", "collection_id", "group_key", "bbox_wkt",
                         "start_date", "end_date", "cache_path", "created_at"]
            result = dict(zip(col_names, row))
            result["superset_hit"] = False
            logger.info(
                "cache_index: exact hit for group_key=%s  collection=%s  cache_path=%s",
                group_key, result["collection_id"], result["cache_path"],
            )
            return result

        logger.debug("cache_index: exact miss for group_key=%s", group_key)

        # ------------------------------------------------------------------
        # Second pass: look for a cached superset that spatially covers the
        # requested bbox and fully spans the requested temporal range.
        # ------------------------------------------------------------------
        if temporal is not None and bounding_box is not None:
            spatial_row = lookup_spatial(conn, collection_id, temporal, bounding_box)
            if spatial_row is not None:
                spatial_row["superset_hit"] = True
                return spatial_row

        return None

    except Exception as exc:
        logger.error("cache_index: lookup failed — %s", exc)
        return None


def lookup_spatial(
    conn,
    collection_id: str,
    temporal: Tuple[str, str],
    bounding_box: Tuple[float, float, float, float],
) -> Optional[dict]:
    """
    Broader spatial+temporal search — useful for finding a *superset* cache entry
    that covers the requested bbox and time range.

    Returns the most-recently cached matching row, or None.
    """
    _SPATIAL_SELECT = """
        SELECT
            id, collection_id, group_key,
            ST_AsText(bbox) AS bbox_wkt,
            start_date, end_date, cache_path, created_at
        FROM zarr_cache_entries
        WHERE collection_id = %s
          AND start_date    <= %s::timestamptz
          AND end_date      >= %s::timestamptz
          AND (bbox IS NULL OR ST_Covers(bbox, ST_GeomFromText(%s, 4326)))
        ORDER BY created_at DESC
        LIMIT 1
    """
    try:
        bbox_wkt  = _bbox_to_polygon_wkt(bounding_box)
        start_iso, end_iso = temporal
        with conn.cursor() as cur:
            cur.execute(_SPATIAL_SELECT, (collection_id, start_iso, end_iso, bbox_wkt))
            row = cur.fetchone()
        if row is None:
            logger.debug(
                "cache_index: spatial miss  collection=%s  start=%s  end=%s",
                collection_id, start_iso, end_iso,
            )
            return None
        col_names = ["id", "collection_id", "group_key", "bbox_wkt",
                     "start_date", "end_date", "cache_path", "created_at"]
        result = dict(zip(col_names, row))
        logger.info(
            "cache_index: spatial hit  group_key=%s  collection=%s",
            result["group_key"], result["collection_id"],
        )
        return result
    except Exception as exc:
        logger.error("cache_index: spatial lookup failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Insert helper
# ---------------------------------------------------------------------------

def insert(
    conn,
    collection_id: str,
    group_key: str,
    cache_path: str,
    temporal: Tuple[str, str],
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    """
    Write a metadata row for a freshly cached Zarr group.

    Uses ``ON CONFLICT DO NOTHING`` so duplicate inserts (e.g. from a retry)
    are silently ignored.

    Returns True on a successful insert, False if the row already existed or
    if the insert failed (error is logged but not re-raised so the caller can
    proceed without the metadata layer).
    """
    _INSERT = """
        INSERT INTO zarr_cache_entries
            (collection_id, group_key, bbox, start_date, end_date, cache_path, created_at)
        VALUES
            (%s, %s,
             CASE WHEN %s::text IS NOT NULL
                  THEN ST_GeomFromText(%s::text, 4326)
                  ELSE NULL
             END,
             %s::timestamptz, %s::timestamptz,
             %s,
             %s)
        ON CONFLICT (group_key) DO NOTHING
        RETURNING id
    """
    bbox_wkt  = _bbox_to_polygon_wkt(bounding_box) if bounding_box else None
    start_iso, end_iso = temporal
    now = datetime.now(tz=timezone.utc).isoformat()

    try:
        with conn.cursor() as cur:
            cur.execute(
                _INSERT,
                (
                    collection_id,
                    group_key,
                    bbox_wkt,   # for the CASE IS NOT NULL check
                    bbox_wkt,   # for ST_GeomFromText
                    start_iso,
                    end_iso,
                    cache_path,
                    now,
                ),
            )
            returned = cur.fetchone()
        conn.commit()
        if returned:
            logger.info(
                "cache_index: inserted metadata  id=%s  group_key=%s",
                returned[0], group_key,
            )
            return True
        else:
            logger.debug("cache_index: row already existed for group_key=%s", group_key)
            return False
    except Exception as exc:
        logger.error("cache_index: insert failed — %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
