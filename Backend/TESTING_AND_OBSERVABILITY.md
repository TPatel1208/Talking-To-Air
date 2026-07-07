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
- `SATELLITE_MAX_RESULTS_CAP`
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

Run the backend tests locally:

```bash
python -m unittest discover -s Backend/tests -p "test_*.py"
```

Run coverage locally:

```bash
coverage run -m unittest discover -s Backend/tests -p "test_*.py"
coverage report
```

The CI workflow runs syntax linting, tests, and coverage reporting. Coverage is focused on backend API/config/logging/cache/routing/helper logic and currently requires at least 60%.

## The scripted eval as a required gate

`Backend/tests/eval_harness.py` (run via `Backend/tests/test_eval_harness.py`,
opt-in `eval` pytest marker — spends real model tokens) is the required
before/after gate for any prompt, model, or routing change. Run it before
and after the change and compare:

```bash
pytest Backend/tests/test_eval_harness.py -v -m eval
```

The run must hold on both counts:

- **Pass threshold** — at least `PASS_THRESHOLD` of `TOTAL_TASKS` tasks pass
  (13 direct-agent tasks against the fake MCP, plus 3 end-to-end tasks
  entering through `ChatStreamService.stream_chat_events` with the real
  router and real sub-agents).
- **Latency budgets** — every task stays under its category's budget in
  `CATEGORY_BUDGETS` (ground-tier tasks under 15s, satellite-tier tasks
  under 45s against the fake MCP). A task that passes its tool-trace and
  outcome checks but blows its budget still fails — speed is a scored
  dimension, not a side observation.
- **Zero rate-limit evidence** — the run fails outright if any provider
  retry/429 evidence was logged during a single-user run
  (`capture_rate_limit_evidence`), since rate-limit pressure was the
  original outage mode this eval exists to catch.

The printed per-task table (name, category, pass/fail, trace verdict,
seconds) is the deliverable a human reads — a regression's location should
be obvious from the output alone, without re-reading this doc.
