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
