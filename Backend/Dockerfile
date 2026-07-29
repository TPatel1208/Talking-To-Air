# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-bullseye AS builder

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

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-bullseye AS runtime

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
# is silent — the background writer catches the PermissionError and the cache
# appears to work while never storing a cube.
RUN mkdir -p /app/data /app/outputs /app/overlay_store /app/cube_store

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE ${PORT}

# Entry point is api.py, app object is named "app"
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
