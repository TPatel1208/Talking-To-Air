# Talking to Air — Storage Architecture

Everything this stack persists data to: one PostgreSQL database, four Docker
named volumes, and one ephemeral OS-tempdir cache. No Redis, no queue, no
second database — Postgres is the only database in the stack.

## 1. PostgreSQL + PostGIS

Container `db` (`postgis/postgis:16-3.4`), database `talking_to_air_memory`,
backed by named volume `pg_data`. Connected to via a shared async connection
pool ([Backend/utils/db.py](../Backend/utils/db.py)).

### `agent_charts`
Source: [sql/init_agent_charts.sql](../sql/init_agent_charts.sql)

```sql
CREATE TABLE agent_charts (
    id          TEXT PRIMARY KEY,   -- content-hash uuid5, or prefixed ids like map_52fd40b2e418
    thread_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT '__legacy__',
    payload     JSONB NOT NULL,     -- full chart payload (grid/statistics/overlay._path/etc.)
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_agent_charts_thread_created ON agent_charts (thread_id, created_at);
CREATE INDEX idx_agent_charts_user_id        ON agent_charts (user_id);
```

Rows are durable and never rewritten, so the payload shape is append-only in
practice: readers must tolerate every shape ever written, not just the current
one. Heatmaps written before 2026-08-08 also carry a `points` key — a flattened
lat/lon/value duplicate of the same field, ~270 KB per chart — which the backend
no longer emits. `chartStats.rawCellValues` still reads it as a fallback for
those rows; nothing else does.
Holds every chart/plot the agent has emitted, keyed by an id the frontend
cites back to fetch it. `payload` includes the `overlay._path` pointer that
the `/chart/{id}/overlay.png` route resolves against `overlay_store`.

### `agent_artifacts`
Source: [sql/init_agent_artifacts.sql](../sql/init_agent_artifacts.sql), also
ensure-created on startup by `ensure_artifact_table()`
([Backend/repositories/artifact_repository.py](../Backend/repositories/artifact_repository.py)).

```sql
CREATE TABLE agent_artifacts (
    id          TEXT PRIMARY KEY,   -- ArtifactStore's tbl_<hex12> id
    user_id     TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    columns     JSONB NOT NULL,
    rows        JSONB NOT NULL,     -- full table payload; hundreds of rows at current sizes
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at  TIMESTAMPTZ         -- NULL until claimed; rows are only ever written at claim time
);
CREATE INDEX idx_agent_artifacts_thread_created ON agent_artifacts (thread_id, created_at);
CREATE INDEX idx_agent_artifacts_user_id        ON agent_artifacts (user_id);
CREATE INDEX idx_agent_artifacts_unclaimed      ON agent_artifacts (created_at) WHERE claimed_at IS NULL;
```
(T39) Durable home for the ground/EPA tools' table artifacts. `ArtifactStore`
(`Backend/services/artifact_store.py`) mints a table into an in-memory dict
only (the tool that mints it doesn't yet know who owns it); the stream
service's `claim()` moments later is the durability boundary -- it upserts
the row here so the artifact survives a restart and is readable from any
worker. Reads check memory first and rehydrate from Postgres on a miss, so
memory is a hot cache, not the system of record, for anything already
claimed. Deleted alongside `agent_charts` when a session is deleted
(`SessionRepository.delete_session`).

### `session_metadata`
Source: [Backend/repositories/session_metadata_repository.py](../Backend/repositories/session_metadata_repository.py)

```sql
CREATE TABLE session_metadata (
    thread_id              TEXT PRIMARY KEY,
    title                  TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id                TEXT NOT NULL DEFAULT '__legacy__',
    ground_monitor_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    satellite_context      JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_session_metadata_user_id ON session_metadata(user_id);
```
One row per chat thread: auto-generated title (first message, truncated to
60 chars) plus per-session UI context (which ground monitors / satellite
layers are active) so a reload can restore it.

### LangGraph checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`)
Not defined by this repo — auto-created by `AsyncPostgresSaver.setup()` in
[Backend/utils/db.py:176-181](../Backend/utils/db.py) (`get_checkpointer()`),
from the `langgraph-checkpoint-postgres>=2.0.0` package. This is the actual
"conversation memory" the docker-compose comment refers to: LangGraph's own
serialized graph-state snapshots per `thread_id`, not application data this
repo's code reads directly. Approximate shape (confirm against the live DB —
exact columns are the library's to change):

```sql
-- checkpoints: one row per graph-state snapshot
(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint JSONB, metadata JSONB)
  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)

-- checkpoint_blobs: large channel values stored out-of-line
(thread_id, checkpoint_ns, channel, version, type, blob BYTEA)
  PRIMARY KEY (thread_id, checkpoint_ns, channel, version)

-- checkpoint_writes: pending writes not yet folded into a checkpoint
(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob BYTEA, task_path)
  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
```

## 2. Docker named volumes (file storage)

### `plot_outputs` — public, shared
Backend writes to `/app/outputs`; the same volume is mounted into the
frontend nginx container at `/usr/share/nginx/html/outputs` and served
**unauthenticated** at `/outputs` (`StaticFiles` mount,
[Backend/tta_backend/api.py:210](../Backend/tta_backend/api.py:210), path from
`OUTPUT_DIR`). Holds matplotlib chart PNGs.

The `StaticFiles` mount resolves its directory at import, so unlike the overlay
store this one legitimately has to exist before any request arrives — the
`os.makedirs` in `api.py` stays, and `OUTPUT_DIR` is what keeps it from landing
in the checkout during a test run
([Backend/tests/cache_isolation.py](../Backend/tests/cache_isolation.py)).
`plot_tools` and `stat_tools` each used to carry their own `APP_ROOT`-relative
`OUTPUT_DIR` with an import-time `os.makedirs`, neither of which was ever read;
both were deleted rather than redirected, so `api.py` is now the only
definition.

### `overlay_store` — private, backend-only
Backend writes to `/app/overlay_store/overlays`
([Backend/tta_backend/tools/satellite_tools/plot_tools.py:100](../Backend/tta_backend/tools/satellite_tools/plot_tools.py:100),
path from `OVERLAY_STORE_DIR`).
**Not** mounted into the frontend; only reachable through the authenticated
`GET /chart/{chart_id}/overlay.png` route, which checks chart ownership
against the requesting user before streaming bytes. Holds server-rendered
MapLibre heatmap overlay PNGs (T23). Kept out of `plot_outputs` on purpose —
see the difference table below.

`OVERLAY_STORE_DIR` exists for the same reason as `CUBE_STORE_DIR`: so the test
suite can redirect the store at a per-process tempdir. Until it did, the path
was `APP_ROOT`-relative and created at import, so the suite created and wrote
`Backend/overlay_store/` inside the checkout — gitignored, so the pollution
survived branch switches and never showed up in `git status`.
[test_store_isolation.py](../Backend/tests/test_store_isolation.py) guards both
stores; the deployment mounts are guarded by
[test_overlay_store_persistence.py](../Backend/tests/test_overlay_store_persistence.py),
which reads the container paths through `deployment_overlay_store_dir()` /
`deployment_output_dir()` precisely because the isolation has taken the live
settings away from it.

### `cube_store` — private, backend-only
Backend writes to `/app/cube_store`
([Backend/services/cube_cache.py](../Backend/services/cube_cache.py), path from
`CUBE_STORE_DIR`). **Not** mounted into the frontend and not served by any
route — nothing outside the backend reads it. Holds the T52 Zarr cube cache:
one Zarr store per cached retrieval, each with a sidecar `manifest.json`
written last as its completion marker, plus `refused.json` negative-cache
entries for sources that cannot round-trip.

A named volume rather than a tempdir on purpose. A cube costs minutes to build
(the full open pipeline plus a read/compress/write of the whole dataset) and
`tta-backend` is rebuilt constantly, so a tempdir store would be empty every
time anyone looked — the `overlay_store` failure mode exactly, except it would
present as "the cache doesn't help" rather than as a broken deployment.
[test_cube_store_persistence.py](../Backend/tests/test_cube_store_persistence.py)
guards the mount.

Bounded by `CUBE_STORE_MAX_BYTES` (default 4 GiB), evicted LRU by last access
before each write. A fixed byte cap, not a share of free space: a percentage
silently expands to fill any disk. A single cube is separately capped at
`CUBE_WRITE_MAX_STORE_FRACTION` (default 0.5) of the store's total limit — a
share rather than an absolute byte count, so it does not need retuning
whenever the store is resized, and so one cube can never evict the whole
store to fit itself only to be evicted by the next write. Every entry is
disposable — a cube that fails its per-hit integrity sweep is deleted and the
answer comes from the lazy path instead.

### `frame_store` — private, backend-only
Backend writes to `/app/frame_store`
([Backend/tta_backend/services/frame_store.py](../Backend/tta_backend/services/frame_store.py),
path from `FRAME_STORE_DIR`). **Not** mounted into the frontend and served by
no route directly — the float32 values behind a chart's T59 time scrubber.
Deliberately its own volume rather than a corner of `overlay_store`: that
store has no eviction policy and grows forever, and an LRU sweeper sharing a
directory with an unbounded store would evict frames to make room for PNGs
that never leave.

Split by durability (D13): the frame **axis** and every per-frame disclosure
(valid fraction, QA pass rate, statistics) live in the chart's Postgres jsonb
row and always survive; only the gzipped float32 **values** live here. An
eviction degrades a chart to a labeled but unscrubbable axis rather than
breaking it — the axis was never at risk. Bounded by `FRAME_STORE_MAX_BYTES`
(default 1 GiB), evicted LRU by last access, same policy shape as
`cube_store` but measured in real gzipped bytes on disk rather than
`ds.nbytes`. Each entry's key is a random UUID, not content-addressed —
charts are ownership-scoped, so two charts are never allowed to share one
blob.

### `earthdata_data` — external, read-only
Declared `external: true` in [docker-compose.yml:154](../docker-compose.yml).
Owned and written by the separate `harmony-retrieval-mcp` stack; this repo
mounts it read-only at `/data` so `export_result`'s `file://` URIs resolve as
a plain filesystem read, without this stack ever writing into it.

## 3. Ephemeral (not a Docker volume, not persisted)

### OS tempdir extract cache
[Backend/services/open_handle.py:41](../Backend/services/open_handle.py:41),
under `tempfile.gettempdir()`. Staging area for extracting members out of
multi-file data bundles before they're opened lazily with dask. TTL-pruned
(`_prune_extract_cache`) and size-gated by `BUNDLE_OPEN_MAX_UNCOMPRESSED_BYTES`
(default 2 GiB) — this cache and its gate are what fixed the earlier
[bundle-open OOM crash](bundle-open-oom-crash.md): eager `.load()` on large
bundles was killing the container before this lazy/TTL/size-capped approach
existed. Wiped on container restart; nothing here is meant to survive one.

## `plot_outputs` vs `overlay_store`

| | `plot_outputs` | `overlay_store` |
|---|---|---|
| Contents | matplotlib chart PNGs (timeseries, comparisons, static plots) | server-rendered MapLibre heatmap overlay PNGs |
| Written by | Backend | Backend (same source file, separate dir) |
| Served by | nginx, directly off disk | Backend route `/chart/{chart_id}/overlay.png` |
| Auth | **none** — anyone with the URL can fetch it | **checked** — verifies the requester owns that chart |
| Mounted in frontend container? | yes | no |

The split exists because `plot_outputs` has to stay unauthenticated (nginx
serves it with no app logic in front), so anything needing per-user access
control was deliberately kept in a separate volume the frontend never
touches.

## Image-generation / hand-off prompt

Paste this into an image-gen model (Midjourney, DALL·E, etc.) or hand it to a
designer — they can't read the code, so this spells out every fact needed:

> Draw a clean, flat, technical architecture diagram (whiteboard/system-design
> style, not photorealistic) with three horizontal tiers, top to bottom:
>
> **Tier 1 — Application:** two boxes side by side, connected by a
> bidirectional arrow: "Frontend (React/Vite, nginx)" and "Backend (FastAPI +
> LangGraph)".
>
> **Tier 2 — Persistent storage** (Docker named volumes), six boxes below
> tier 1, each with an arrow up to Backend:
> 1. **PostgreSQL + PostGIS** (volume `pg_data`) — list inside:
>    `session_metadata`, `agent_charts`, `agent_artifacts`, `user_connectors`, and LangGraph's own
>    `checkpoints`/`checkpoint_blobs`/`checkpoint_writes` tables.
> 2. **plot_outputs** volume — chart PNGs; arrows from *both* Frontend and
>    Backend (nginx serves it directly and unauthenticated at `/outputs`;
>    backend writes to it).
> 3. **overlay_store** volume — server-rendered map overlay PNGs; Backend-only,
>    reachable solely via an authenticated route, deliberately separate from
>    `plot_outputs` because that one is public.
> 4. **cube_store** volume — cached Zarr cubes of opened datasets; Backend-only,
>    served by no route at all, size-capped and LRU-evicted.
> 5. **frame_store** volume — float32 values behind a chart's time scrubber;
>    Backend-only, size-capped and LRU-evicted, its own volume separate from
>    `overlay_store` specifically because that one has no eviction policy.
> 6. **earthdata_data** volume — dotted/foreign-styled box, "external,
>    read-only, owned by a different repo/stack (harmony-retrieval-mcp)";
>    mounted read-only at `/data`.
>
> **Tier 3 — Ephemeral (not persisted):** one dashed box, "OS temp-dir
> extract cache" — bundle-extraction scratch space, TTL-pruned and
> size-gated.
>
> Legend: solid border = persisted, owned by this repo; dashed = ephemeral;
> dotted = persisted but owned by another stack. Restrained palette — blue for
> app tier, green for owned persistent storage, orange for ephemeral, purple
> for external.
