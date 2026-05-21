-- =============================================================================
--  Talking to Air — Zarr cache metadata index
--  File: sql/init_cache_index.sql
--
--  Mounted into the PostGIS container at:
--    /docker-entrypoint-initdb.d/01_cache_index.sql
--  and executed automatically on first container start.
--
--  Safe to run multiple times (all statements are idempotent).
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- -----------------------------------------------------------------------------
--  Main table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zarr_cache_entries (
    id            SERIAL PRIMARY KEY,

    -- Which NASA CMR collection this entry belongs to
    collection_id TEXT        NOT NULL,

    -- MD5-based key that addresses the group inside the Zarr store
    -- Computed by DataLoader.make_group_key(collection_id, start, end, bbox)
    group_key     TEXT        NOT NULL,

    -- Geographic bounding box of the cached subset (EPSG:4326 polygon)
    -- NULL when no spatial subsetting was requested
    bbox          geometry(Polygon, 4326),

    -- Temporal extent of the cached data
    start_date    TIMESTAMPTZ NOT NULL,
    end_date      TIMESTAMPTZ NOT NULL,

    -- Filesystem path to the Zarr store (e.g. ./data/cache.zarr)
    cache_path    TEXT        NOT NULL,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
--  Indexes
-- -----------------------------------------------------------------------------

-- Spatial index — accelerates ST_Covers / ST_Intersects bbox queries
CREATE INDEX IF NOT EXISTS idx_zarr_cache_bbox
    ON zarr_cache_entries
    USING GIST (bbox);

-- Temporal range index — accelerates date-range overlap queries
CREATE INDEX IF NOT EXISTS idx_zarr_cache_dates
    ON zarr_cache_entries (start_date, end_date);

-- Unique key index — primary fast-path lookup + duplicate guard
CREATE UNIQUE INDEX IF NOT EXISTS idx_zarr_cache_group_key
    ON zarr_cache_entries (group_key);