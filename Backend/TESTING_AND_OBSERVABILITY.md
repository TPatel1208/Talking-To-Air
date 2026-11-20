# Testing, Configuration, and Observability

## Configuration

Backend configuration is centralized in `infrastructure/settings.py`.

Use `get_settings()` instead of reading environment variables directly in runtime code. The settings object loads `.env` once, normalizes known option sets, and exposes typed values for model names, database settings, CORS origins, fetch routing, logging, and external service credentials.

Required startup values:

- `DB_PASSWORD`
- `GOOGLE_API_KEY`

Useful operational values:

- `LLM_MODEL` (the supervisor and exploration)
- `EARTHDATA_AGENT_MODEL` (a workflow's agent steps)
- `NOMINATIM_CONTACT_EMAIL` (the geocoder's contact address; must be a real email)
- `DATA_FETCH_MODE`
- `LOG_LEVEL`
- `LOG_FORMAT=json`
- `LONG_REQUEST_SECONDS`

## Logging

Logging is configured by `infrastructure/logging.py`.

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
case never asserts which tools a system called, so the same cases run against
any system through a `SystemAdapter`.

The gate system is the agent harness's supervisor end to end
(`benchmark/adapters/harness.py`): every case's prompt goes in verbatim as the
one message of a fresh conversation, on the fake data boundary in
`benchmark/adapters/harness_boundary.py` (the earthdata fake behind the app's
own `McpEarthdataSession`, EPA AQS behind a capturing fetch with a benchmark
user's own connector, an offline geocoder, memory stores and an in-memory
checkpointer). It needs no Postgres or web server.

It spends real model tokens, so it is run by hand, and it is the required
before/after gate for any prompt, model, workflow or routing change. Run it
before and after the change and compare:

```bash
cd Backend
uv run python -m benchmark.runner
docker compose --profile benchmark run --rm benchmark
```

The models are whatever the environment configures (`*_PROVIDER` and `*_MODEL`,
Portkey included). Results are saved under `benchmark-results/harness/`;
`--results-dir` overrides the folder (`BENCHMARK_RESULTS_DIR` also works). The
rest of `tests/test_benchmark*.py` (checks, scoring, outcomes, case structure,
the adapters on scripted models) runs in the normal suite and spends nothing.

The run must hold on three counts:

- **Pass threshold.** At least `PASS_THRESHOLD` (17) of `TOTAL_CASES` (20)
  pass, less any case waived in the design's cutover gate table. The cases are
  13 discovery, retrieval, plotting, comparison and ground-validation cases,
  5 end-to-end cases (a ground lookup, a satellite plot, both sources, and two
  routing checks that must not add a ground request), and 2
  region-refusal cases (a refused region must never become a retrieval of
  another one).
- **Latency budgets.** Every case stays under its category's budget in
  `CATEGORY_BUDGETS` (against the fake MCP). A case whose checks all pass but
  that blows its budget still fails.
- **Zero rate-limit evidence.** The run fails outright if any provider
  retry/429 evidence was logged during a single-user run
  (`capture_rate_limit_evidence`).

The printed results table (case, category, PASS/FAIL, seconds, model tokens,
and the first failing check's reason) is what a human reads, so a regression's
location and cause are visible without re-reading this doc. The token columns
are input, output, and cached (the share of input read from the provider's
prompt cache), summed over every model call in the case and totalled for the
run; a dash means the provider reported none.

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

### The benchmark through the harness workflows (no model)

`benchmark/adapters/workflow.py` runs the same cases through the harness's
workflow runner instead of a model. A case opts in with two optional fields:
`workflow` (a workflow name) and `workflow_inputs` (a dict, or a function of the
run's date for relative dates such as "yesterday"). The adapter runs that
workflow directly on the same fake data boundary and fills the `Outcome` the
checks read. The answer text is composed from the result envelope's fixed
wording (mode label, findings, disclosures, gaps, and the reason when the run
is not done). A case that names no workflow is not run and is listed as not
covered, never as failed.

```bash
cd Backend
uv run python -m benchmark.runner --system workflow
docker compose --profile benchmark run --rm benchmark python -m benchmark.runner --system workflow
```

It needs no model key or network, and prints a table per workflow (cases
covered, passed, the first failed check of each failing case) followed by the
uncovered case ids. Its table is not checked against the gate. Results are
saved under `benchmark-results/workflow/`.

A needs_input result counts as a completed run (a refused area is an answer),
except when the adapter left out a connector or an input, which is recorded as
an error. `tests/test_benchmark_workflow_adapter.py` pins what each covered
case does today, failures included.

`--system explore` runs each covered case's prompt as one exploration task with
the case's bundle and real models; it is exploratory, costs tokens, and is not
checked against the gate either. Results are saved under
`benchmark-results/explore/`.
