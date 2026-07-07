# Talking to Air

**Talking to Air** is an AI-powered conversational interface for querying, visualizing, and analyzing atmospheric data from NASA satellite missions and EPA ground sensors. Ask natural-language questions about air quality and get maps, trend plots, and statistical summaries drawn from real observations.

---

## Features

- **Multi-agent architecture** — a stateful supervisor agent routes queries to specialized satellite and ground-sensor subagents
- **NASA Harmony integration** — fetches data on demand from OMI, TROPOMI, and TEMPO missions via the NASA Harmony API
- **EPA AQS integration** — queries the EPA Air Quality System API for ground-level measurements
- **Intelligent data caching** — downloaded granules are stored in Zarr format on disk and indexed in PostgreSQL to avoid redundant fetches
- **Flexible fetch routing** — `DATA_FETCH_MODE` supports `auto`, `harmony`, `opendap`, and `s3` strategies
- **Persistent sessions** — full conversation history is stored in PostgreSQL (one thread per session) using a LangGraph Postgres checkpointer
- **Streaming responses** — the `/chat` endpoint streams Server-Sent Events so the UI updates progressively
- **Production-ready observability** — structured JSON logging, named log events, and LangSmith tracing support

---

## Supported Datasets

| Dataset | Sensor | Variable | Resolution | Coverage |
|---|---|---|---|---|
| `OMI_NO2` | OMI / Aura | NO₂ | Daily | Global |
| `TROPOMI_NO2` | Sentinel-5P | NO₂ | Monthly | Global |
| `TEMPO_NO2` | TEMPO | NO₂ | Hourly | North America |
| `TEMPO_O3TOT` | TEMPO | O₃ | Hourly | North America |
| `OMI_O3` | OMI / Aura | O₃ | Daily | Global |
| `TEMPO_HCHO` | TEMPO | HCHO | Hourly | North America |
| `OMI_HCHO` | OMI / Aura | HCHO | Daily | Global |

---

## Architecture

```
Frontend (React + Vite)
        │  SSE stream
        ▼
Backend (FastAPI)
        │
        ▼
Supervisor Agent  ──── Postgres checkpointer (conversation memory)
   ├── Satellite Agent  ──  NASA Harmony / OPeNDAP / S3
   └── Ground Sensor Agent  ──  EPA AQS API
        │
        ▼
Cache Layer  ──  Zarr files on disk + PostgreSQL cache index
```

**Supervisor** — stateful LangGraph agent on Google Gemini 2.5 Flash (its own free-tier rate budget, separate from the Groq subagents). Owns the Postgres checkpointer and maintains one conversation thread per session. Trims its own context window to stay within the model's token budget.

**Satellite Agent** — stateless LangGraph agent on Groq (the large tool-use model, with the full Groq budget to itself). Delegates data fetching to whichever strategy is configured (Harmony, OPeNDAP CE, or S3 direct). Produces maps and time-series plots using Cartopy and Matplotlib.

**Ground Sensor Agent** — stateless LangGraph agent on Groq (the small model). Queries the EPA AQS API for PM₂.₅, O₃, NO₂, and other pollutants at monitoring stations.

Each agent's provider + model is a configuration entry, resolved through one model factory (`config/model_factory.py`) — switching providers is an environment change, not a code change. See `SUPERVISOR_MODEL_PROVIDER`, `EARTHDATA_AGENT_PROVIDER`, and `GROUND_AGENT_PROVIDER` below.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Google AI Studio API key](https://ai.google.dev/) (`GOOGLE_API_KEY`)
- [Groq API key](https://console.groq.com/) (`GROQ_API_KEY`)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/) (username + password)
- [EPA AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html) (email + key)
- Optional: [LangSmith API key](https://smith.langchain.com/) for tracing

---

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/talking-to-air.git
   cd talking-to-air
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in all required values (see [Environment Variables](#environment-variables) below).

3. **Build and start:**
   ```bash
   docker compose up --build
   ```

4. **Open the app:**
   - Chat interface: http://localhost:5173
   - API docs (Swagger): http://localhost:8000/docs
   - Health check: http://localhost:8000/health
   - Prometheus metrics: http://localhost:8000/metrics

5. **Subsequent starts** (no rebuild needed unless dependencies change):
   ```bash
   docker compose up
   ```

6. **Stop and wipe volumes:**
   ```bash
   docker compose down -v
   ```

---

## Joining the earthdata-retrieval MCP stack

This stack connects to the [harmony-retrieval-mcp](https://github.com/your-username/harmony-retrieval-mcp) stack over a shared external Docker network, and reads that stack's materialized data volume directly (read-only) so `export_result`'s `file://` URIs resolve as a plain filesystem read in both containers.

1. **Start the MCP stack first** (in the harmony-retrieval-mcp repo, with `EARTHDATA_MCP_TRANSPORT=http` set in its `.env`):
   ```bash
   docker compose up --build
   ```
   This creates the external network `earthdata_net` and the external volume `earthdata_data` that this stack attaches to.

2. **Set `EARTHDATA_MCP_URL` and `EARTHDATA_MCP_TOKEN`** in this repo's `.env` to match that stack's HTTP endpoint and token.

3. **Start this stack:**
   ```bash
   docker compose up --build
   ```
   If `earthdata_net` doesn't exist yet, compose will fail with a "network not found" error — bring up the MCP stack first.

4. **Smoke check** — confirm the shared mount and network both resolve:
   ```bash
   docker compose exec backend ls /data
   docker compose exec backend curl -H "Authorization: Bearer $EARTHDATA_MCP_TOKEN" http://mcp:8765/mcp
   ```
   The first command should list whatever the MCP stack has materialized; the second should get a response from the MCP's HTTP endpoint rather than a DNS/connection error.

---

## Operations

The backend exposes `/health` for dependency-aware readiness and `/metrics` in Prometheus text format. `/metrics` is intentionally exempt from API key authentication so a scraper can collect it, but production deployments should bind or proxy it only on a private, non-public interface.

See [docs/runbook.md](docs/runbook.md) for health response interpretation, key metrics, Harmony timeout diagnosis, stalled request handling, and Zarr cache pruning.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below.

### Required

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio key — used by the supervisor agent |
| `GROQ_API_KEY` | Groq key — used by the satellite and ground-sensor subagents |
| `DB_PASSWORD` | PostgreSQL password (any string you choose) |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | NASA Earthdata credentials |
| `EDL_USERNAME` / `EDL_PASSWORD` | Earth Data Login credentials (same account as Earthdata) |
| `AQS_API_EMAIL` / `AQS_API_KEY` | EPA AQS API credentials — `AQS_API_EMAIL` is also reused as the contact address in the Nominatim geocoder's User-Agent, so it must be a real email (Nominatim rejects placeholder `@example.com`/`.org`/`.net` addresses) |

### Optional / Tuning

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-2.5-flash` | Supervisor model |
| `SUPERVISOR_MODEL_PROVIDER` | `google` | Supervisor provider (`google` or `groq`) |
| `EARTHDATA_AGENT_MODEL` | `openai/gpt-oss-120b` | Satellite (earthdata) subagent model |
| `EARTHDATA_AGENT_PROVIDER` | `groq` | Satellite subagent provider (`groq` or `google`) |
| `GROUND_AGENT_MODEL` | `openai/gpt-oss-20b` | Ground sensor subagent model |
| `GROUND_AGENT_PROVIDER` | `groq` | Ground sensor subagent provider (`groq` or `google`) |
| `DATA_FETCH_MODE` | `auto` | Fetch strategy: `auto`, `harmony`, `opendap`, or `s3` |
| `S3_FORCE_FETCH` | `0` | Set to `1` to bypass the us-west-2 region check for S3 fetches |
| `SATELLITE_MAX_RESULTS_CAP` | `20` | Maximum granule results per satellite query |
| `MEMORY_CACHE_MAX_BYTES` | `524288000` | In-memory satellite dataset cache limit in bytes |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `text` | Set to `json` for structured production logs |
| `LONG_REQUEST_SECONDS` | `30` | Threshold (seconds) for `long_running_request` log events |
| `DB_POOL_MIN_SIZE` | `1` | Minimum PostgreSQL connection pool size |
| `DB_POOL_MAX_SIZE` | `10` | Maximum PostgreSQL connection pool size |
| `LANGSMITH_API_KEY` | — | Enables LangSmith tracing when set |
| `LANGCHAIN_PROJECT` | `talking_to_air_monitoring` | LangSmith project name |
| `EARTHDATA_MCP_URL` | `http://mcp:8765/mcp` | URL of the earthdata-retrieval MCP's HTTP endpoint (see [Joining the earthdata-retrieval MCP stack](#joining-the-earthdata-retrieval-mcp-stack)) |
| `EARTHDATA_MCP_TOKEN` | — | Bearer token for the MCP endpoint; must match that stack's `EARTHDATA_MCP_TOKEN` |

---

## Usage

### Example Queries

**Single location:**
```
Plot NO2 levels in Texas on April 8, 2024
What was the mean NO2 in Los Angeles in March 2024?
Where was NO2 highest in California today?
```

**Comparisons:**
```
Compare NO2 between California and New York
Show formaldehyde levels in London vs Tokyo
How does O3 in Texas compare to Florida?
```

**Time series:**
```
Show the NO2 trend over Greece for the last 18 months
Plot HCHO levels over Florida over the past year
How has ozone changed in New York since January?
```

**Ground sensors:**
```
What are the current PM2.5 readings in Chicago?
Show EPA air quality data for Houston last week
```

### Interface Guide

- **Chat window** — type your question in the input at the bottom of the screen
- **Images** — generated maps and plots appear inline and open in a lightbox on click
- **Tool badges** — yellow badges beside responses show which tools the agent invoked
- **Sessions** — use the left sidebar to start a new conversation or revisit a previous one; all history is persisted

### Dataset Constraints

- **TEMPO datasets** cover **North America only** (hourly). For other regions use OMI or TROPOMI.
- **TROPOMI_NO2** is **monthly resolution only** — single-day queries are not supported.
- **Date formats** — natural language ("March 2024", "last 18 months") and ISO format ("2024-03-15") both work.
- **Large bounding boxes** — queries over very large regions take longer and may hit granule caps.

---

## Development

### Running Tests

```bash
# From the repo root
docker compose exec backend python -m unittest discover -s tests -p "test_*.py"
```

### Coverage

```bash
docker compose exec backend coverage run -m unittest discover -s tests -p "test_*.py"
docker compose exec backend coverage report
```

The CI pipeline (`.github/workflows/backend-ci.yml`) runs on every push and PR to `main`. It installs system dependencies (PROJ, GEOS), lints Python syntax with `compileall`, runs the full test suite, and enforces a minimum 60% coverage threshold.

### Load Testing

```bash
python Backend/scripts/load_chat.py --url http://localhost:8000 --concurrency 10
python Backend/scripts/load_chat.py --url http://localhost:8000 --concurrency 20
```

Tune `DB_POOL_MIN_SIZE` and `DB_POOL_MAX_SIZE` based on observed connection counts during load.

### Database Schema

Fresh PostgreSQL volumes are initialized from SQL scripts mounted into `docker-entrypoint-initdb.d`:

- `sql/init_agent_charts.sql` creates `agent_charts`.

Schema changes should be made in these SQL files. To apply init-script changes to a local fresh database, stop the stack and recreate the database volume with `docker compose down -v`, then start it again with `docker compose up --build`.

### Logging

Set `LOG_FORMAT=json` in `.env` for structured logs suitable for log aggregators. Key event names emitted by the backend:

| Event | Meaning |
|---|---|
| `startup_complete` | Application ready |
| `shutdown_complete` | Clean shutdown |
| `cache_hit` | Zarr data served from local cache |
| `cache_miss` | Data fetched from remote source |
| `agent_failure` | Subagent returned an error |
| `response_truncated` | Context trimming removed messages |
| `database_reconnect` | Pool recovered a lost connection |
| `long_running_request` | Request exceeded `LONG_REQUEST_SECONDS` |

---

## Project Structure

```
.
├── Backend/
│   ├── agents/             # Supervisor, satellite, and ground-sensor agents
│   ├── config/             # Settings, system prompts
│   ├── datasets/           # Dataset registry (collections.yaml) and collection tools
│   ├── models/             # Pydantic models for agent results and chart payloads
│   ├── preprocessing/      # Data loader, cache manager, cache index
│   ├── repositories/       # PostgreSQL repositories for cache index and charts
│   ├── scripts/            # Load testing and utility scripts
│   ├── services/           # Harmony, OPeNDAP, and S3 fetch services
│   ├── tests/              # Unit and integration tests
│   ├── tools/
│   │   ├── ground_sensor_tools/   # EPA AQS tools
│   │   └── satellite_tools/       # Harmony API, plot, stat, and date tools
│   ├── utils/              # DB pool, logging, plotting, streaming helpers
│   ├── api.py              # FastAPI application entry point
│   ├── Dockerfile
│   └── requirements.txt
├── Frontend/
│   ├── src/
│   │   ├── components/     # Chat, ChartMessage, Dashboard, ImageViewer, etc.
│   │   ├── hooks/          # useChat (SSE client)
│   │   └── utils/          # SSE parser
│   ├── tests/
│   ├── Dockerfile
│   └── package.json
├── sql/
│   ├── init_cache_index.sql
│   └── init_agent_charts.sql
├── docker-compose.yml
├── .env.example
└── README.md
```

---
