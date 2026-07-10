# Talking to Air

## Commits

Do not add a `Co-Authored-By` trailer (or any Claude/AI attribution) to commit
messages.

## Running tests — use Docker

Run the backend test suite through Docker, not the host Python. The Docker
image installs the native geospatial stack (PROJ, GEOS, GDAL) and every Python
dependency at the versions CI uses, so results match CI exactly:

```bash
docker compose --profile test run --rm backend-test   # backend: pytest + coverage
docker compose --profile test run --rm frontend-test  # frontend: vitest
```

To run a subset while iterating:

```bash
docker compose --profile test run --rm backend-test sh -c "pytest tests/test_subagent_dispatch.py -q"
```

### Why not host `python -m pytest`

The host Windows checkout has two traps that Docker sidesteps:

- **PROJ / `proj.db`.** Anything importing `utils/overlay_render.py` builds a
  CRS at import time. A stale system-wide `PROJ_LIB` (pointing at a conda
  `airPollution` env that doesn't match the pip-installed rasterio) makes GDAL
  fail collection-wide with `CRSError: Cannot find proj.db`. `Backend/conftest.py`
  works around this for host runs by overriding a broken `PROJ_LIB` with
  rasterio's bundled data dir — but Docker avoids the problem entirely.
- **Optional deps / `.env` bleed.** Host runs miss optional packages (e.g.
  `langchain_google_genai`) and pick up a local `.env`, so a couple of tests
  (`test_model_factory` Gemini, `test_config_logging` token default) fail on the
  host that pass in Docker.

If you must run on the host anyway, `Backend/conftest.py` wires PROJ
automatically — just be aware of the two known host-only failures above.
