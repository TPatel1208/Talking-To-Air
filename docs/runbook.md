# Operational Runbook

## Health Checks

Check the service with:

```bash
curl -i https://localhost/api/health
```

The backend publishes no host port -- nginx is the only way in, so every
command here goes through `/api/`. Against a local stack serving a
mkcert-minted certificate, add `-k`: browsers trust that CA after
`mkcert -install`, but curl on Windows fails the revocation check
(`CRYPT_E_NO_REVOCATION_CHECK`) rather than the trust check. To skip TLS
entirely, address the backend from inside the stack:

```bash
docker compose exec backend curl -i http://localhost:8000/health
```

A healthy service returns HTTP 200:

```json
{"status":"ok","db":true,"agent":true,"earthdata_mcp":"ready"}
```

A degraded service returns HTTP 503 and names the failed dependency:

```json
{"status":"degraded","db":false,"agent":true,"earthdata_mcp":"connecting","db_error":"connection refused"}
```

`db=false` means the backend could not run `SELECT 1` through the PostgreSQL pool within the health timeout. `agent=false` means the FastAPI process has not successfully initialized the supervisor agent. `earthdata_mcp` is `connecting` / `ready` / `unavailable` / `incompatible` — it does not affect the HTTP status code, since the satellite path degrades independently of ground/EPA features (see the main README's MCP-joining doc).

## Metrics

Prometheus-compatible metrics are available at:

```bash
curl https://localhost/api/metrics
```

Key metrics:

- `http_requests_total`: request volume by method, route path, and status code. A normal local development baseline is low and bursty.
- `http_request_duration_seconds`: request latency by method and route path. Health and metrics should usually stay well below 1 second.
- `agent_requests_total`: subagent calls by `agent_type` and `outcome`. `failure` and `timeout` should be rare.
- `envelope_salvaged_total`: sub-agent final messages recovered from prose after failing structured-envelope parsing, by `agent_type`. Nonzero is tolerable; a sustained rise means a provider/prompt drift is breaking the structured output.
- `harmony_fetch_duration_seconds`: end-to-end Harmony submission, polling, and download duration. Remote data jobs can take seconds to minutes depending on NASA service load and granule size.
- `harmony_timeouts_total`: Harmony jobs that exceeded the configured processing timeout. Normal value is 0.
- `cache_hits_total` / `cache_misses_total`: hits by `cache_level`, misses that fell through to a remote fetch. Repeated identical satellite requests should produce hits; new collection/time/bbox requests raise misses.
- `cube_store_bytes` / `cube_evictions_total`: on-disk size of the T52 cube cache and how often it evicts to stay under `CUBE_STORE_MAX_BYTES` (see "Cube cache" below).
- `cube_index_hits_total` / `cube_index_misses_total` / `cube_index_invalidations_total`: T54 handle→cube index — cubes served with no re-verify round-trip, lookups that fell through to verify-first, and indexed cubes dropped because a re-verify delivered different content.
- `pipeline_phase_duration_seconds`: wall-clock duration of one retrieval/visualization pipeline phase, by `phase`.
- `db_pool_connections_active`: active PostgreSQL connections in the shared backend pool. It should stay below `DB_POOL_MAX_SIZE`.

## Harmony Thread Pool Exhaustion

Look for repeated Harmony timeout warnings with the structured event `harmony_job_timeout`, especially when `elapsed_seconds` is near the configured Harmony processing timeout. If these appear alongside long-running requests and no successful `harmony_fetch_duration_seconds` observations, the Harmony wait/download worker may be saturated or stalled.

Useful fields:

- `job_url`: Harmony job status URL.
- `thread_id`: application conversation thread affected by the stalled request.
- `elapsed_seconds`: time spent waiting before the timeout.

## Canceling Stalled Requests

A long-running satellite retrieval surfaces as a job and can be cancelled directly — from the Jobs panel, or:

```bash
curl -X POST "https://localhost/api/jobs/<job_handle>/cancel" \
  -H "Authorization: Bearer <token>"
```

There is no separate endpoint to cancel an in-flight chat turn that is not a job — but every turn is automatically bounded by `CHAT_TURN_TIMEOUT_SECONDS` (default 1800s), so it cannot hang indefinitely. To force-stop immediately, restart the backend process:

```bash
docker compose restart backend
```

This interrupts in-flight requests. Conversation history already committed to PostgreSQL remains available after restart.

## Cube Cache

The T52 cube cache (`cube_store` volume) holds opened-and-reduced Zarr cubes, keyed so it self-invalidates when the underlying export changes. It evicts on its own — LRU by last access, run before every write — to stay under `CUBE_STORE_MAX_BYTES` (default 4 GiB); no manual pruning endpoint exists or is needed. Watch `cube_store_bytes` and `cube_evictions_total` in `/metrics` to see it working.

To force a full reset (e.g. after a schema-incompatible code change), stop the stack and drop the volume:

```bash
docker compose down
docker volume ls --filter name=cube_store   # find the actual name (prefixed with the compose project)
docker volume rm <name-from-above>
```

`docker compose down -v` wipes this along with every other named volume (Postgres, overlay store, frame store) — prefer the single `docker volume rm` above unless you actually want a clean slate everywhere.
