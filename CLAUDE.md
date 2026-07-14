# Talking to Air

## Commits

Do not add a `Co-Authored-By` trailer (or any Claude/AI attribution) to commit
messages.

## Running tests — use Docker

Run the backend test suite through Docker, not the host Python. The Docker
image installs the native geospatial stack (PROJ, GEOS, GDAL) and every Python
dependency at the versions CI uses, so results match CI exactly:

```bash
docker compose --profile test run --build --rm backend-test   # backend: pytest + coverage
docker compose --profile test run --build --rm frontend-test  # frontend: vitest
```

To run a subset while iterating:

```bash
docker compose --profile test run --build --rm backend-test sh -c "pytest tests/test_subagent_dispatch.py -q"
```

**Always pass `--build`.** The test services bake the source into the image at
build time (`build: context: ./Backend`) — they do *not* bind-mount your working
tree, on purpose, so the run is hermetic and matches CI. But `docker compose run`
reuses the last-built image unless told to rebuild, so **without `--build` you
silently test stale code** — a green run that never saw your edits. The rebuild
is cheap: deps install before the source `COPY`, so a code-only change only
re-runs the final layer (~seconds). Do not add a source bind-mount to make runs
faster — that reintroduces the host/image divergence (see the PROJ and
optional-deps traps below) that this hermetic build exists to avoid.

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
