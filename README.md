# Talking to Air

**Talking to Air** is an AI-powered conversational interface for querying, visualizing, and analyzing atmospheric data from NASA satellite missions and EPA ground sensors. Ask natural-language questions about air quality and get interactive maps, trend plots, and statistical summaries drawn from real observations.

## How it works

A React frontend talks to a FastAPI backend over a streamed (SSE) `/chat` endpoint. A **supervisor** agent routes each query to one of two subagents: a **satellite** agent that retrieves NASA data on demand through the [earthdata-retrieval MCP](https://github.com/TPatel1208/harmony-retrieval-mcp) (a separate stack, with size-gated "safe retrieval"), and a **ground-sensor** agent that queries the EPA AQS API. PostgreSQL holds conversation memory (one thread per session) and a chart/artifact index. Each agent's provider and model are configuration entries resolved through a single model factory, so switching providers is an environment change, not a code change.

That's the whole picture you need to run it — this README is setup-focused. For internals, see [`docs/`](docs/) and the PRDs in [`docs/prds/`](docs/prds/).

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Google AI Studio API key](https://ai.google.dev/) — `GOOGLE_API_KEY` (supervisor and satellite agents)
- [Groq API key](https://console.groq.com/) — `GROQ_API_KEY` (ground-sensor agent)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/) — username + password
- [EPA AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html) — email + key
- The [harmony-retrieval-mcp](https://github.com/TPatel1208/harmony-retrieval-mcp) stack, for satellite data (see [Joining the MCP stack](#joining-the-earthdata-retrieval-mcp-stack) below). Ground/EPA features work without it.
- Optional: [LangSmith API key](https://smith.langchain.com/) for tracing

---

## Quick Start

1. **Clone:**
   ```bash
   git clone https://github.com/TPatel1208/Talking-To-Air.git
   cd Talking-To-Air
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in the required values (see [Environment Variables](#environment-variables)).

3. **Build and start:**
   ```bash
   docker compose up --build
   ```

4. **Open the app:**
   - Chat interface — http://localhost:5173
   - API docs (Swagger) — http://localhost:8000/docs
   - Health check — http://localhost:8000/health
   - Prometheus metrics — http://localhost:8000/metrics

5. **Subsequent starts** (no rebuild unless dependencies change):
   ```bash
   docker compose up
   ```

6. **Stop and wipe volumes:**
   ```bash
   docker compose down -v
   ```

---

## Joining the earthdata-retrieval MCP stack

The satellite path retrieves NASA data through the [harmony-retrieval-mcp](https://github.com/TPatel1208/harmony-retrieval-mcp) stack. The two stacks connect over a shared external Docker network and share the MCP's materialized data volume (read-only), so a retrieved file's `file://` URI resolves as a plain filesystem read in both containers.

**Startup order does not matter.** The backend boots without the MCP — ground/EPA features work immediately, and a background task connects to the MCP (with capped retry) and heals the satellite path without a restart once it's up. `/health`'s `earthdata_mcp` field reports `connecting` / `ready` / `unavailable` / `incompatible`. The one exception is a genuinely fresh machine: Docker's `external: true` network (`earthdata_net`) and volume (`earthdata_data`) must exist before this stack's `docker compose up` will succeed at all, so the MCP stack has to run **once** first to create them.

1. **First time only** — in the `harmony-retrieval-mcp` repo (with `EARTHDATA_MCP_TRANSPORT=http` in its `.env`), bring the stack up to create the shared network and volume:
   ```bash
   docker compose up --build
   ```

2. **Set `EARTHDATA_MCP_URL` and `EARTHDATA_MCP_TOKEN`** in this repo's `.env` to match that stack's HTTP endpoint and token.

3. **Start this stack** (either order relative to the MCP stack, from here on):
   ```bash
   docker compose up --build
   ```

4. **Smoke check:**
   ```bash
   docker compose exec backend ls /data       # lists what the MCP has materialized
   docker compose exec backend curl http://localhost:8000/health   # earthdata_mcp: ready
   ```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below. **`.env.example` is the exhaustive reference** — every tunable (retrieval poll bounds, granule concurrency, CSV/export caps, map basemap URLs, etc.) is documented there with defaults. The tables below cover what you actually need to set.

### Required

| Variable | Description |
|---|---|
| `DB_PASSWORD` | PostgreSQL password (any string you choose) |
| `JWT_SECRET_KEY` | Long random secret for session auth tokens |
| `GOOGLE_API_KEY` | Google AI Studio key — required if any agent uses the `google` provider (supervisor and satellite agents do by default) |
| `GROQ_API_KEY` | Groq key — required if any agent uses the `groq` provider (the ground-sensor agent does by default) |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | NASA Earthdata credentials |
| `EDL_USERNAME` / `EDL_PASSWORD` | Earth Data Login credentials (same account as Earthdata) |
| `AQS_API_EMAIL` / `AQS_API_KEY` | EPA AQS credentials. `AQS_API_EMAIL` is **also** the contact address in the Nominatim geocoder's User-Agent, so it must be a **real** email — Nominatim rejects placeholder `@example.com`/`.org`/`.net` addresses, and location lookups (ground *and* satellite) fail without it |
| `EARTHDATA_MCP_URL` / `EARTHDATA_MCP_TOKEN` | Endpoint and bearer token of the MCP stack (satellite path only — see above) |

> The backend refuses to boot without `DB_PASSWORD`, `JWT_SECRET_KEY`, and the provider key(s) for whichever agents are configured. Earthdata/AQS/MCP values aren't checked at boot but are required for the corresponding features to work.

### Common tuning

| Variable | Default | Description |
|---|---|---|
| `SUPERVISOR_MODEL_PROVIDER` / `LLM_MODEL` | `google` / `gemini-2.5-flash` | Supervisor provider + model |
| `EARTHDATA_AGENT_PROVIDER` / `EARTHDATA_AGENT_MODEL` | `google` / `gemini-3.1-flash-lite` | Satellite subagent provider + model |
| `GROUND_AGENT_PROVIDER` / `GROUND_AGENT_MODEL` | `groq` / `openai/gpt-oss-20b` | Ground-sensor subagent provider + model |
| `RETRIEVAL_SOFT_CAP_BYTES` | `2147483648` | Safe-retrieval gate — estimates at/below this proceed automatically; above it (up to the hard cap) pause for in-chat confirmation |
| `RETRIEVAL_HARD_CAP_BYTES` | `10737418240` | Retrievals estimated above this are refused with guidance to narrow the request |
| `BUNDLE_OPEN_MAX_UNCOMPRESSED_BYTES` | `2147483648` | Refuse to open a result bundle whose *uncompressed* size exceeds this (guards against OOM at open time) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | Set `LOG_FORMAT=json` for structured logs suitable for aggregators |
| `LANGSMITH_API_KEY` | — | Enables LangSmith tracing when set |

Provider for each agent is `google` (Gemini) or `groq`; the model factory (`config/model_factory.py`) turns provider + model into a chat model, so either can change without touching code.

---

## Features

- **Conversational querying** — ask in natural language; the supervisor routes to satellite or ground-sensor agents automatically.
- **Interactive maps** — MapLibre heatmap panels with light/dark basemaps and terrain, single-panel and multi-panel layouts.
- **Compare mode** — put two regions, pollutants, or time windows side by side in a compare grid.
- **Jobs panel** — long-running satellite retrievals surface as cancellable jobs you can watch to completion.
- **Trend plots & statistics** — time-series charts and statistical summaries over a region and date range.
- **Provenance & export** — every chart carries its data provenance and methods; export as CSV, PNG, or NetCDF.
- **Persistent sessions** — full conversation history stored in PostgreSQL, one thread per session, revisitable from the sidebar.
- **Degrade-don't-die** — the satellite path heals in the background when the MCP comes up; ground/EPA features never wait on it.
- **Observability** — structured JSON logging, named log events, `/health`, `/metrics`, and optional LangSmith tracing.

### Datasets

Beyond a set of built-in presets (OMI, TROPOMI, and TEMPO NO₂ / O₃ / HCHO), the satellite path supports **any gridded NASA collection** through universal coordinate discovery — lat/lon are identified from CF metadata rather than a hard-coded list.

---

## Usage

Try these after startup as a smoke test:

```
Plot NO2 levels in Texas on April 8, 2024
What was the mean NO2 in Los Angeles in March 2024?
Compare NO2 between California and New York
Show the NO2 trend over Greece for the last 18 months
What are the current PM2.5 readings in Chicago?
```

In the interface, type your question in the input at the bottom; generated maps and plots appear inline and open in a lightbox on click. The left sidebar starts a new conversation or reopens a previous one — all history is persisted.

---

## Development

### Running tests (use Docker)

Run the suites through Docker so results match CI exactly — the image bakes in the native geospatial stack (PROJ, GEOS, GDAL) and CI-pinned dependencies. **Always pass `--build`**; the test services bake source into the image at build time and won't see your edits otherwise.

```bash
docker compose --profile test run --build --rm backend-test    # pytest + coverage
docker compose --profile test run --build --rm frontend-test   # frontend tests
```

Run a subset while iterating:

```bash
docker compose --profile test run --build --rm backend-test sh -c "pytest tests/test_subagent_dispatch.py -q"
```

See [`CLAUDE.md`](CLAUDE.md) for why host `python -m pytest` is discouraged (PROJ / `proj.db` and optional-dependency traps).

### CI

`.github/workflows/backend-ci.yml` runs on every push and PR to `main`:

- **Backend** — installs system geo deps, syntax-checks with `compileall`, lints with `ruff`, type-checks selected packages with `mypy`, then runs the test suite with coverage.
- **Frontend** — `npm ci`, `npm test`, `npm run build`, and a Docker image build.

### Database schema

Fresh PostgreSQL volumes are initialized from SQL mounted into `docker-entrypoint-initdb.d`:

- `sql/init_agent_charts.sql` creates the `agent_charts` table.

Make schema changes in that file. To apply init-script changes locally, recreate the volume: `docker compose down -v` then `docker compose up --build`.

---

## Operations

The backend exposes `/health` for dependency-aware readiness and `/metrics` in Prometheus text format. `/metrics` is intentionally exempt from API-key auth so a scraper can reach it — in production, bind or proxy it on a private interface only.

See [`docs/runbook.md`](docs/runbook.md) for health-response interpretation, key metrics, retrieval/timeout diagnosis, stalled-request handling, and cache pruning.

---

## Project Structure

```
.
├── Backend/            # FastAPI app (api.py), agents, services, tools, tests
│   ├── agents/         # Supervisor, satellite (earthdata), ground-sensor agents
│   ├── earthdata_mcp/  # Client + toolset for the earthdata-retrieval MCP
│   ├── services/       # Retrieval, discovery, jobs, export, chart, provenance
│   ├── datasets/       # Dataset registry + universal coordinate discovery
│   ├── config/         # Settings, model factory, system prompts
│   └── tests/
├── Frontend/           # React + Vite app (MapLibre, compare grid, jobs panel)
├── sql/                # Postgres init scripts
├── docs/               # Runbook, design notes, PRDs
├── docker-compose.yml
├── .env.example
└── README.md
```
