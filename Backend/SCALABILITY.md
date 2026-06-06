# Backend Scalability Notes

## Runtime lifecycle

FastAPI initializes shared resources in `api.lifespan`:

- validates required configuration (`DB_PASSWORD`, `GOOGLE_API_KEY`)
- creates the shared PostgreSQL connection pool
- ensures chart tables exist
- builds the supervisor agent
- closes database resources on shutdown

EarthAccess is intentionally excluded from startup. Authentication happens only
when an S3 or CMR satellite path calls `get_earthaccess_auth()`.

## Async request boundary

`POST /chat` is an async endpoint. The LangGraph streaming iterator and chart
persistence calls are still synchronous libraries, so the route advances them
with `asyncio.to_thread(...)`. This keeps blocking I/O off the FastAPI event
loop while preserving the existing SSE contract.

## Load testing

Start the API, then run:

```bash
python Backend/scripts/load_chat.py --url http://localhost:8000 --concurrency 10
python Backend/scripts/load_chat.py --url http://localhost:8000 --concurrency 20
```

Record latency, throughput, and database connection count from Postgres during
the run. The pool size is controlled with `DB_POOL_MIN_SIZE` and
`DB_POOL_MAX_SIZE`.
