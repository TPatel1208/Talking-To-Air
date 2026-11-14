# Testing, Configuration, and Observability

## Configuration

Backend configuration is centralized in `config/settings.py`.

Use `get_settings()` instead of reading environment variables directly in runtime code. The settings object loads `.env` once, normalizes known option sets, and exposes typed values for model names, database settings, CORS origins, fetch routing, logging, and external service credentials.

Required startup values:

- `DB_PASSWORD`
- `GOOGLE_API_KEY`

Useful operational values:

- `LLM_MODEL`
- `GROUND_AGENT_MODEL`
- `SATELLITE_AGENT_MODEL`
- `DATA_FETCH_MODE`
- `LOG_LEVEL`
- `LOG_FORMAT=json`
- `LONG_REQUEST_SECONDS`

## Logging

Logging is configured by `utils/logging.py`.

Set `LOG_FORMAT=json` for production-style JSON logs:

```json
{"timestamp":"...","level":"INFO","module":"api","message":"request_completed"}
```

Important event names include:

- `startup_complete`
- `shutdown_complete`
- `cache_hit`
- `cache_miss`
- `agent_failure`
- `response_truncated`
- `database_reconnect`
- `long_running_request`

## Tests

Install the locked environment and run the backend tests locally (pytest, not
`unittest discover`: the suite's bootstrap lives in `Backend/conftest.py`):

```bash
cd Backend && uv sync
cd Backend && uv run pytest tests
```

Run coverage locally (from the repo root, where `.coveragerc` lives):

```bash
PYTHONPATH=Backend Backend/.venv/bin/coverage run -m pytest Backend/tests
Backend/.venv/bin/coverage report
```

Coverage is measured over all of `tta_backend` and CI requires at least 87% (`.coveragerc`). The Redis-backed turn tests skip on a host with no Redis; set `TTA_REQUIRE_REDIS=1` to make that a failure, as CI and the `backend-test` compose profile do.

The Postgres-backed tests (`tests/postgres_helpers.py`) run against a real database, each inside its own throwaway schema that is dropped afterwards, so they never touch `public` or another test's tables. The default URL is `postgresql://postgres:test@127.0.0.1:55432/tta_test`; override it with `TTA_TEST_DATABASE_URL`. Start one locally with `docker run -d --rm --name tta-test-pg -e POSTGRES_PASSWORD=test -e POSTGRES_DB=tta_test -p 55432:5432 postgres:16`. They skip when it is unreachable; `TTA_REQUIRE_POSTGRES=1` makes that a failure, as CI and the `backend-test` profile (which starts a `test-db` service) do.

## The benchmark as a required gate

`Backend/benchmark/` holds 20 benchmark cases, each a question, the data the
fake services return for it, and checks on the **outcome**: what reached the
data boundary (calls to the fake earthdata service and to EPA AQS) and what the
user was shown (charts, artifacts, the answer text, the narrated stages). A
case never asserts which tools an agent called, so the same cases can run
against any system through a `SystemAdapter`. The current system's adapter
(`benchmark/adapters/current.py`) drives every case through
`ChatStreamService.stream_chat_events` with the real intent router, the real
checkpointed supervisor (it needs Postgres), the real ground agent and the real
satellite agent, whose tools are the workspace-bound production set pointed at
the fake earthdata service.

It is opt-in (the `eval` pytest marker) because it spends real model tokens, and
it is the required before/after gate for any prompt, model, or routing change.
Run it before and after the change and compare. The `benchmark` compose
profile runs it in a Linux container against the working tree, reading `.env`
and starting Postgres and Redis for it:

```bash
docker compose --profile benchmark run --rm benchmark
```

On a host that can import the plotting stack, with Postgres up, it also runs
directly:

```bash
pytest Backend/tests/test_benchmark.py -v -s -m eval
```

The models are whatever the environment configures (`*_PROVIDER` and `*_MODEL`,
Portkey included), and the run is skipped when a configured provider has no real
key. It needs Postgres, since the supervisor checkpoints there. The rest of
`tests/test_benchmark.py` (checks, scoring, outcome parsing, case structure) runs
in the normal suite and spends nothing.

The run must hold on three counts:

- **Pass threshold.** At least `PASS_THRESHOLD` (17) of `TOTAL_CASES` (20) pass.
  The threshold is provisional until a live run confirms it. The cases are
  13 discovery, retrieval, plotting, comparison and ground-validation cases,
  5 end-to-end cases (a ground lookup, a satellite plot, both sources, and two
  routing checks that must not add a ground request), and 2
  region-refusal cases (a refused region must never become a retrieval of
  another one).
- **Latency budgets.** Every case stays under its category's budget in
  `CATEGORY_BUDGETS` (15s to 90s, against the fake MCP; the cross-source case,
  which runs both agents, has the longest). They were measured on the current
  system entered through the full chat stream, as described beside the table.
  A case whose checks all pass but that blows its budget still fails.
- **Zero rate-limit evidence.** The run fails outright if any provider
  retry/429 evidence was logged during a single-user run
  (`capture_rate_limit_evidence`).

The printed results table (case, category, PASS/FAIL, seconds, model tokens,
and the first failing check's reason) is what a human reads, so a regression's
location and cause are visible without re-reading this doc. The token columns
are input, output, and cached (the share of input read from the provider's
prompt cache), summed over every model call in the case and totalled for the
run. They are the counts the app's own `record_llm_tokens` metric receives; a
dash means the provider reported none.

The `est cost` column prices those tokens per model with the table in
`benchmark/pricing.py`: Anthropic's first-party list prices, with cache reads
at the cache-read price. It is an estimate, not the bill. The models run through
Portkey on Vertex, which prices Claude separately, and cache writes are charged
at the plain input price because they are not counted separately. A model with
no entry in the table is listed under the table and its cost carries a `?`; add
its price there when a new model is configured.

Relative dates are measured against the run's own clock, which the outcome
records (the real UTC date at the start and end of the run). The "yesterday"
case accepts an EPA AQS call dated the day before either, so a run that
crosses midnight UTC still passes, but a "yesterday" sent as today fails.
