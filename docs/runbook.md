# Operational Runbook

## Health Checks

Check the service with:

```bash
curl -i http://localhost:8000/health
```

A healthy service returns HTTP 200:

```json
{"status":"ok","db":true,"agent":true}
```

A degraded service returns HTTP 503 and names the failed dependency:

```json
{"status":"degraded","db":false,"agent":true,"db_error":"connection refused"}
```

`db=false` means the backend could not run `SELECT 1` through the PostgreSQL pool within the health timeout. `agent=false` means the FastAPI process has not successfully initialized the supervisor agent.

## Metrics

Prometheus-compatible metrics are available at:

```bash
curl http://localhost:8000/metrics
```

Key metrics:

- `http_requests_total`: request volume by method, route path, and status code. A normal local development baseline is low and bursty.
- `http_request_duration_seconds`: request latency by method and route path. Health and metrics should usually stay well below 1 second.
- `agent_requests_total`: subagent calls by `agent_type` and `outcome`. `failure` and `timeout` should be rare.
- `harmony_fetch_duration_seconds`: end-to-end Harmony submission, polling, and download duration. Remote data jobs can take seconds to minutes depending on NASA service load and granule size.
- `harmony_timeouts_total`: Harmony jobs that exceeded the configured processing timeout. Normal value is 0.
- `cache_hits_total`: hits by `memory`, `zarr`, and `postgis`. Repeated identical satellite requests should produce memory or Zarr hits.
- `cache_misses_total`: remote fetches after all cache levels miss. This rises for new collection, time, or bounding-box requests.
- `db_pool_connections_active`: active PostgreSQL connections in the shared backend pool. It should stay below `DB_POOL_MAX_SIZE`.

## Harmony Thread Pool Exhaustion

Look for repeated Harmony timeout warnings with the structured event `harmony_job_timeout`, especially when `elapsed_seconds` is near the configured Harmony processing timeout. If these appear alongside long-running requests and no successful `harmony_fetch_duration_seconds` observations, the Harmony wait/download worker may be saturated or stalled.

Useful fields:

- `job_url`: Harmony job status URL.
- `thread_id`: application conversation thread affected by the stalled request.
- `elapsed_seconds`: time spent waiting before the timeout.

## Canceling Stalled Requests

There is no per-request cancellation endpoint yet. To manually cancel a stalled backend request, restart the backend process:

```bash
docker compose restart backend
```

This interrupts in-flight requests. Conversation history already committed to PostgreSQL remains available after restart.

## Pruning The Zarr Cache

Preferred application path:

```bash
curl -X DELETE "http://localhost:8000/admin/cache/prune?older_than_days=30" \
  -H "Authorization: Bearer <token>"
```

Manual fallback inside Docker:

```bash
docker compose exec db psql -U postgres -d talking_to_air_memory \
  -c "DELETE FROM cache_index WHERE created_at < now() - interval '30 days' RETURNING group_key;"
docker compose exec backend find /app/data/cache.zarr -mindepth 1 -mtime +30 -exec rm -rf {} +
```

Use the application endpoint when possible so database rows and filesystem data stay in sync.
