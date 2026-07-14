"""Deployment-contract guard for the server-rendered map overlays (T23).

The overlay PNGs are the *native* high-quality map layer: the frontend
(MapLibreHeatmapPanel) shows them when /chart/{id}/overlay.png resolves, and
degrades to a blocky client canvas render when it 404s. The chart payload —
including ``overlay.url`` — persists durably in Postgres, so it always reloads
on refresh/restart. The overlay PNG must therefore persist across a container
recreate too, or a restart silently downgrades every prior chart to the
canvas fallback ("chart quality lowered on refresh or restart").

Overlays are deliberately stored OUTSIDE ``/app/outputs`` (that dir is served
unauthenticated at /outputs), so they cannot ride the ``plot_outputs`` volume
and need their own named volume. This test asserts the deployment gives them
one.
"""
from __future__ import annotations

import os

import yaml

# Bind-mounted into the backend-test container (see docker-compose.yml),
# because docker-compose.yml lives at the repo root, outside the ./Backend
# build context.
COMPOSE_PATH = "/compose/docker-compose.yml"


def _covers(mount_target: str, path: str) -> bool:
    """True if a mount at ``mount_target`` persists everything under ``path``."""
    mount_target = mount_target.rstrip("/")
    path = path.rstrip("/")
    return path == mount_target or path.startswith(mount_target + "/")


def _persisted_named_volume_targets(service: dict, top_level_volumes: dict) -> list[str]:
    """Container mount targets in ``service`` backed by a *named* volume, which
    survives ``docker compose up --build`` / down+up — unlike a bind mount or
    the ephemeral container layer."""
    targets: list[str] = []
    for entry in service.get("volumes", []) or []:
        if not isinstance(entry, str):
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        if source in top_level_volumes:  # bare name => named volume
            targets.append(target)
    return targets


def test_overlay_store_is_backed_by_a_persisted_volume():
    if not os.path.isfile(COMPOSE_PATH):
        import pytest

        pytest.skip(f"{COMPOSE_PATH} not mounted (run via docker compose backend-test)")

    from tools.satellite_tools.plot_tools import OVERLAY_STORE_DIR

    # In the container, plot_tools resolves this to /app/overlay_store/overlays.
    overlay_container_path = os.path.abspath(OVERLAY_STORE_DIR)

    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    backend = compose["services"]["backend"]
    top_level_volumes = compose.get("volumes", {}) or {}
    persisted = _persisted_named_volume_targets(backend, top_level_volumes)

    assert any(_covers(target, overlay_container_path) for target in persisted), (
        f"overlay store {overlay_container_path!r} is not covered by any persisted "
        f"named volume on the backend service (persisted targets: {persisted}). "
        "A container recreate wipes every rendered overlay PNG while its chart "
        "payload persists in Postgres, so /chart/{id}/overlay.png 404s and the "
        "map silently degrades to the canvas fallback."
    )
