# Talking to Air

**Talking to Air** is an AI-powered conversational interface for querying, visualizing, and analyzing atmospheric data from NASA satellite missions and EPA ground sensors. Ask natural-language questions about air quality and get interactive maps, trend plots, vertical profiles, and statistical summaries drawn from real observations.

## How it works

A React frontend talks to a FastAPI backend over a streamed (SSE) `/chat` endpoint. A **supervisor** agent routes each query to one of two subagents: a **satellite** agent that retrieves NASA data on demand through the [earthdata-retrieval MCP](https://github.com/TPatel1208/harmony-retrieval-mcp) (a separate stack, with size-gated "safe retrieval"), and a **ground-sensor** agent that queries the EPA AQS API. PostgreSQL holds conversation memory (one thread per session) and a chart/artifact index. Each agent's provider and model are configuration entries resolved through a single model factory, so switching providers is an environment change, not a code change.

That's the whole picture you need to run it. For internals — storage layout, operational runbook, design notes — see [`docs/`](docs/).

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [mkcert](https://github.com/FiloSottile/mkcert#installation) — mints the locally-trusted cert the frontend serves over HTTPS
- [Google AI Studio API key](https://ai.google.dev/) — `GOOGLE_API_KEY` (every agent uses this by default)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/) — username + password
- [EPA AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html) — email + key
- The [harmony-retrieval-mcp](https://github.com/TPatel1208/harmony-retrieval-mcp) stack, for satellite data — see [`docs/mcp-setup.md`](docs/mcp-setup.md). Ground/EPA features work without it.
- Optional: [LangSmith API key](https://smith.langchain.com/) for tracing

---

## Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/TPatel1208/Talking-To-Air.git
   cd Talking-To-Air
   cp .env.example .env
   ```
   Fill in `.env` — see [Environment Variables](#environment-variables) below.

2. **If you want satellite data**, bring up the `harmony-retrieval-mcp` stack once first — it creates a Docker network and volume this stack depends on. Details: [`docs/mcp-setup.md`](docs/mcp-setup.md). Skip this if you only need ground/EPA features, or if that stack is already running somewhere.

3. **Set up local HTTPS (one-time, required):**
   ```bash
   ./scripts/setup-tls.sh
   ```
   Trusts a local CA via mkcert and writes `Frontend/localhost+2.pem` / `Frontend/localhost+2-key.pem`. The frontend's Docker build copies these into the nginx image, so it won't build without them. Re-run any time if those files go missing.

4. **Build and start:**
   ```bash
   docker compose up --build
   ```

5. **Open the chat interface** — https://localhost. It opens on a sign-in screen; these are app-level accounts in this stack's own Postgres, unrelated to the Earthdata/EPA credentials above. Click "Create account" the first time.

   Also available: API docs at `/docs`, health check at `/health`, Prometheus metrics at `/metrics` (all on port 8000).

6. **Subsequent starts** (no rebuild unless dependencies changed): `docker compose up`
7. **Stop and wipe volumes:** `docker compose down -v`

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below — it's the exhaustive reference, documenting every tunable with its default.

### Required

| Variable | Description |
|---|---|
| `DB_PASSWORD` | PostgreSQL password (any string you choose) |
| `SUPABASE_URL` | Supabase project URL — the identity provider. Access tokens are verified locally against its published JWKS |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase publishable (anon) key. Served to the browser at runtime by `GET /config/auth` |
| `GOOGLE_API_KEY` | Google AI Studio key — every agent defaults to the `google` provider |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | NASA Earthdata credentials |
| `AQS_API_EMAIL` / `AQS_API_KEY` | EPA AQS credentials. `AQS_API_EMAIL` also becomes the contact address in the Nominatim geocoder's User-Agent, so it must be a **real** email — Nominatim rejects placeholder `@example.com`/`.org`/`.net` addresses, and location lookups (ground *and* satellite) fail without it |
| `EARTHDATA_MCP_URL` / `EARTHDATA_MCP_TOKEN` | Endpoint and bearer token of the MCP stack (satellite path only) |

The backend refuses to boot without `DB_PASSWORD`, `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `GOOGLE_API_KEY`. Earthdata/AQS/MCP values aren't checked at boot but are required for the corresponding features to work.

### Worth knowing about

| Variable | Default | Description |
|---|---|---|
| Per-agent provider/model | all `google` | Supervisor: `SUPERVISOR_MODEL_PROVIDER`/`LLM_MODEL` (default `gemma-4-31b-it`). Satellite: `EARTHDATA_AGENT_PROVIDER`/`EARTHDATA_AGENT_MODEL`. Ground: `GROUND_AGENT_PROVIDER`/`GROUND_AGENT_MODEL` (both default `gemini-3.1-flash-lite`). Set a provider to `groq` to route that agent through Groq instead — then `GROQ_API_KEY` becomes required too. |
| `RETRIEVAL_SOFT_CAP_BYTES` / `RETRIEVAL_HARD_CAP_BYTES` | 2 GiB / 10 GiB | A retrieval estimated at/below the soft cap proceeds automatically; above it (up to the hard cap) it pauses for in-chat confirmation; above the hard cap it's refused with guidance to narrow the request. |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | Set `LOG_FORMAT=json` for structured logs suitable for aggregators. |
| `CONNECTOR_ENCRYPTION_KEY` | — | Fernet key(s) encrypting per-user connector tokens (bring-your-own Earthdata token). Unset: the feature answers a structured 503 instead of blocking boot. |

---

## Features

- **Conversational querying** — ask in natural language; the supervisor routes to satellite or ground-sensor agents automatically.
- **Interactive maps** — MapLibre heatmap panels with light/dark basemaps and terrain, single- and multi-panel layouts, plus a time scrubber to step or animate through a multi-day map.
- **Compare mode** — put two regions, pollutants, or time windows side by side in a compare grid.
- **Data discovery** — search NASA collections by pollutant/region, preview a dataset's coverage and variables before plotting anything.
- **Variable guidance** — an inventory of a dataset's variables, related-variable suggestions (QA flags, companion bands), and a picker when a request is ambiguous.
- **Jobs panel** — long-running satellite retrievals surface as cancellable jobs you can watch to completion.
- **Trend plots, vertical profiles & statistics** — time-series charts, altitude/pressure profiles for layered products, and statistical summaries over a region and date range.
- **Provenance & export** — every chart carries its data provenance and methods; export as CSV, PNG, or NetCDF.
- **Persistent sessions** — full conversation history in PostgreSQL, one thread per session, revisitable from the sidebar.
- **Per-user connectors** — optionally link your own NASA Earthdata token instead of using the shared one in `.env`.
- **Degrade-don't-die** — the satellite path heals in the background when the MCP comes up; ground/EPA features never wait on it.
- **Observability** — structured JSON logging, `/health`, `/metrics`, and optional LangSmith tracing.

Beyond a set of built-in presets (OMI, TROPOMI, TEMPO NO₂/O₃/HCHO), the satellite path supports **any gridded NASA collection** — lat/lon are identified from CF metadata rather than a hard-coded list.

---

## Usage

The empty chat screen shows six starter prompts pulled live from the backend (`config/starter_prompts.py`) — each is tied to a real eval task, so these are the questions most likely to work end-to-end:

```
What NASA datasets are available for NO2 column density over New Jersey?
Plot TROPOMI NO2 over New Jersey for 2024-01-15.
Show me how NO2 changed over Newark NJ during January 2024.
Compare TEMPO NO2 with EPA ground monitors over Newark NJ for the first week of January 2024.
Compare TEMPO NO2 over New Jersey between June 2025 and June 2026 — did it change?
What was the NO2 level in Newark, New Jersey yesterday?
```

A date/region with no granules in the source collection returns a refusal, not an error — that's by design. Prefer the starter chips, or a recent well-covered date, over guessing one from scratch.

Generated maps and plots appear inline and open in a lightbox on click. The left sidebar starts a new conversation or reopens a previous one — all history is persisted.

---

## Development

Run test suites through Docker so results match CI exactly — the image bakes in the native geospatial stack (PROJ, GEOS, GDAL) and CI-pinned dependencies. Host `python -m pytest` is discouraged for the same reason: those system libraries are only guaranteed to line up inside the image. **Always pass `--build`** — the test services bake source into the image at build time and won't see your edits otherwise.

```bash
docker compose --profile test run --build --rm backend-test    # pytest + coverage
docker compose --profile test run --build --rm frontend-test   # frontend tests
```

Run a subset while iterating:
```bash
docker compose --profile test run --build --rm backend-test sh -c "pytest tests/test_subagent_dispatch.py -q"
```

`.github/workflows/backend-ci.yml` runs on every push and PR to `main`: backend installs system geo deps, syntax-checks with `compileall`, lints with `ruff`, type-checks selected packages with `mypy`, then runs the test suite with coverage; frontend runs `npm ci`, `npm run lint`, `npm test`, `npm run build`, and a Docker image build.

Postgres schema changes go in `sql/init_agent_charts.sql` / `sql/init_agent_artifacts.sql` — these only run against a fresh volume, so apply a local change with `docker compose down -v` then `docker compose up --build`.

---

## Operations

`/health` (dependency-aware readiness) and `/metrics` (Prometheus text format) are exposed by the backend — `/metrics` is intentionally exempt from API-key auth so a scraper can reach it; bind or proxy it on a private interface in production. See [`docs/runbook.md`](docs/runbook.md) for response interpretation, key metrics, and diagnosing stalled requests or timeouts.

---

## More docs

- [`docs/mcp-setup.md`](docs/mcp-setup.md) — joining the earthdata-retrieval MCP stack
- [`docs/runbook.md`](docs/runbook.md) — health checks, metrics, stalled requests, the cube cache
- [`docs/storage-architecture.md`](docs/storage-architecture.md) — every Postgres table and Docker volume, and why each exists
