# â”€â”€ Stage 1: Builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Debian 12 (bookworm), not 11 (bullseye). Bullseye's LTS window closed and
# its security archive stopped being republished, so the Release file for
# bullseye-security went past its Valid-Until on 2026-09-07 and `apt-get
# update` now exits 100 on every build from a cold cache. Nothing in the repo
# changed to cause it and nothing in the repo could have prevented it -- the
# base image simply aged out. Pinning an older digest would not help: the
# expiry is in the archive apt fetches at build time, not in the image.
# Keep the CI comment in .github/workflows/backend-ci.yml in step with this.
FROM python:3.12-bookworm AS builder

WORKDIR /app

# Geospatial / scientific build deps (Cartopy, HDF5, NetCDF, Zarr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libproj-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies come from uv.lock, which pins every package, direct and
# indirect, with hashes. `--locked` refuses to build if the lock no longer
# matches pyproject.toml rather than quietly re-resolving. Pinned uv version;
# keep it in step with the CI workflow's setup-uv.
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies alone first, so a source-only change reuses this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . .
RUN uv sync --locked --no-dev


# â”€â”€ Stage 2: Dev tooling (tests/lint only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# pytest/ruff/mypy/coverage live in their own stage so they never reach the
# runtime venv: stage 3 copies /opt/venv from `builder`, which this stage adds
# to without mutating. `docker compose --profile test run backend-test` targets
# this stage; nothing else should.
FROM builder AS devtools

RUN uv sync --locked


# â”€â”€ Stage 3: Runtime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FROM python:3.12-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libproj-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .
RUN rm -rf tests .pytest_cache

# Pre-create volume mount points with correct ownership. This must stay ABOVE
# the chown below: Docker initializes a fresh named volume from whatever the
# image has at that path, so a mount point that doesn't exist here produces a
# root-owned volume that uid 1001 cannot write to. For cube_store that failure
# is silent â€” the background writer catches the PermissionError and the cache
# appears to work while never storing a cube.
RUN mkdir -p /app/data /app/outputs /app/overlay_store /app/cube_store /app/frame_store

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE ${PORT}

# Entry point is tta_backend/api.py, app object is named "app".
#
# `exec` is load-bearing, not tidiness. The shell is here only to expand
# ${PORT}; without exec it stays alive as PID 1 with uvicorn as its child, and
# Docker sends SIGTERM to PID 1 and to nothing else. `sh` does not forward it,
# so uvicorn never runs its lifespan shutdown -- which is where T63's drain
# lives, and with it the `interrupted` frame every reader of an in-flight turn
# is owed on a deploy.
#
# --timeout-graceful-shutdown bounds the wait for in-flight requests. It has
# to: the GET carrying a chat turn is in-flight for the whole turn, and it
# ends when the turn writes its terminal entry -- which the drain writes,
# in the lifespan shutdown that runs *after* this wait. Unbounded, the two
# wait for each other and the drain never runs at all.
CMD ["sh", "-c", "exec uvicorn tta_backend.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-graceful-shutdown 5"]
