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
- `llm_tokens_total`: provider tokens billed for chat completions, by `model`, `agent_type` and `kind`. `agent_type` uses the same values as `agent_requests_total` (`satellite`, `ground_sensor`) plus `supervisor`, so tokens-per-agent-call joins across the two; both sub-agents default to the same model id, so `model` alone cannot separate them. A series under `agent_type="unknown"` means a chat model was built without declaring one. `cache_read` is the **subset** of `input` that hit the provider prompt cache and billed at the cached rate — it is not additional to `input`, so do not sum the kinds. The prompt-cache hit rate for a model is `cache_read / input`; the sub-agents re-send a large constant prefix (system prompt plus bound tool schemas) on every call, so a persistently low ratio there is the signal that something is breaking the cached prefix. A model that has never been called has no series at all — unlike the metrics above, this labelset is not pre-declared, because the model ids are environment configuration rather than a closed vocabulary.
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

A chat turn that is not a job is stopped by thread, which is what the Stop button calls. The turn is cancelled wherever it is running — including on another replica — and any provider retrievals it orphaned are cancelled with it:

```bash
curl -X POST "https://localhost/api/chat/<thread_id>/stop"   -H "Authorization: Bearer <token>"
```

A 404 means no turn is running on that thread. Every turn is bounded by `CHAT_TURN_TIMEOUT_SECONDS` (default 1800s) regardless, so nothing hangs indefinitely.

Restarting the backend also ends in-flight turns, but no longer silently: shutdown drains, writing an `interrupted` terminal entry to every live stream so readers are told rather than left spinning.

```bash
docker compose restart backend
```

Conversation history already committed to PostgreSQL remains available after restart.

## Detached Chat Turns

A chat turn runs independently of the connection that started it: `POST /api/chat` returns 202 with a `turn_id`, and all narration arrives over `GET /api/chat/<thread_id>/stream`. Switching sessions, reloading the page or sleeping a laptop costs a reader its place in the stream, never the turn.

This depends on the `redis` service. While Redis is unreachable chat returns **503** — deliberately, since Redis carries every event, the one-turn-per-thread claim and the stop signal. Check it first when chat is down but the rest of the app is up:

```bash
docker compose exec redis redis-cli ping
```

**One turn per thread.** A second send while a turn runs is answered 409 naming the running turn, so a second tab joins it rather than forking a second agent run onto one conversation. A thread whose owning replica was killed outright holds its claim for up to `CLAIM_TTL_SECONDS` (30s) before it lapses on its own; nothing needs doing.

**Rolling back.** Set `CHAT_DETACHED_TURNS_ENABLED=0` and restart the backend. The frontend branches on the response — 200 with a stream against 202 with JSON — so the shipped bundle serves either and no image is rebuilt.

```bash
CHAT_DETACHED_TURNS_ENABLED=0 docker compose up -d backend
```

**Deploy ordering matters in the other direction.** A backend running the new protocol against a frontend bundle that predates it leaves the chat bubble spinning forever: the old bundle reads the 202's JSON body into an SSE parser and finds no events. Deploy the frontend first, or set `CHAT_DETACHED_TURNS_ENABLED=0` on the backend until it has:

```bash
docker compose build frontend && docker compose up -d frontend
docker compose build backend  && docker compose up -d backend
```

Note that `docker compose up -d <service>` also recreates any *dependency* whose
image has changed, so `up -d frontend` can silently redeploy the backend under
you. Confirm what is actually running rather than what you asked for:

```bash
docker inspect tta-backend --format '{{.State.StartedAt}} {{.Image}}'
```

**What a deploy does to a turn that is running.** On SIGTERM the backend stops
accepting, waits up to `--timeout-graceful-shutdown` (5s) for in-flight
requests, then runs the drain: every turn still producing is marked
`interrupted` with reason `shutdown`, its thread's claim is released and its
provider jobs are deliberately left alone (the user will retry, and the cached
retrievals are worth more to that retry than the provider capacity). The whole
sequence is budgeted by `stop_grace_period: 30s`; a full shutdown under load
was measured at **8.6s**, so Docker's 10s default is not enough.

Three settings make this work and all three are load-bearing:

| Setting | Where | Why |
|---|---|---|
| `exec uvicorn` | `Backend/Dockerfile` | Without it `sh` is PID 1 and never forwards SIGTERM, so uvicorn is killed outright and none of the below runs |
| `--timeout-graceful-shutdown 5` | `Backend/Dockerfile` | The GET carrying a turn is in-flight for the whole turn and ends only when the drain marks it, which happens *after* this wait. Unbounded, the two wait for each other |
| `stop_grace_period: 30s` | `docker-compose.yml` | The sequence above is a sum, and Docker's 10s default SIGKILLs it partway |

A reader watching a turn when the backend goes down has its connection severed
before the `interrupted` entry is written, so it shows "Connection lost" with a
**Reload session** button; reloading reattaches and delivers the real ending
("The server restarted before this answer finished"). Deploys log one
`CancelledError: Task cancelled, timeout graceful shutdown exceeded` traceback
per open stream — that is uvicorn cutting the SSE connections at the 5s bound,
and it is expected.

**A replica that is killed outright** (SIGKILL, OOM) writes nothing. Its
readers are told `interrupted` with reason `stale` about 30s after the last
frame, and the thread's claim lapses on its own at `CLAIM_TTL_SECONDS` —
measured at **31s** — after which the thread accepts new messages normally.
Nothing needs doing.

**Why a thread can be missing from "Recent analyses".** Two different facts
about a thread are recorded at two different times. `session_metadata` is
written when the message is posted, because it is the only record of who owns
the thread and both the reattach stream and stop refuse without it;
`first_frame_at` is stamped when that thread's turn first produces a frame,
and `GET /sessions` lists only threads that have one. A turn that was stopped,
failed or died before narrating therefore leaves an unlisted thread — still
reachable by id, still deletable, just not in the sidebar. Before this, every
such thread showed up as a titled row over an empty conversation, because the
fast path writes the transcript once, at the end of the turn.

How many threads are hidden this way:

```bash
docker compose exec db psql -U postgres -d talking_to_air_memory   -c "SELECT count(*) FROM session_metadata WHERE first_frame_at IS NULL"
```

The column is added on startup and backfilled from `created_at` **once**, on
the run that adds it — so threads that predate it all keep listing. The
backfill is guarded on the column not already existing: making it
unconditional would relist every empty thread on the next restart.

## Cube Cache

The T52 cube cache (`cube_store` volume) holds opened-and-reduced Zarr cubes, keyed so it self-invalidates when the underlying export changes. It evicts on its own — LRU by last access, run before every write — to stay under `CUBE_STORE_MAX_BYTES` (default 4 GiB); no manual pruning endpoint exists or is needed. Watch `cube_store_bytes` and `cube_evictions_total` in `/metrics` to see it working.

To force a full reset (e.g. after a schema-incompatible code change), stop the stack and drop the volume:

```bash
docker compose down
docker volume ls --filter name=cube_store   # find the actual name (prefixed with the compose project)
docker volume rm <name-from-above>
```

`docker compose down -v` wipes this along with every other named volume (Postgres, overlay store, frame store) — prefer the single `docker volume rm` above unless you actually want a clean slate everywhere.
