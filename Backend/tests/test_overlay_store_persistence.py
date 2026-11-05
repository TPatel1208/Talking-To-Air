"""Deployment-contract guard for the server-rendered map overlays (T23).

The overlay PNGs are the *native* high-quality map layer: the frontend
(MapLibreHeatmapPanel) shows them when /chart/{id}/overlay.png resolves, and
degrades to a blocky client canvas render when it 404s. The chart payload —
including ``overlay.url`` — persists durably in Postgres, so it always reloads
on refresh/restart. The overlay PNG must therefore persist across a container
recreate too, or a restart silently downgrades every prior chart to the
canvas fallback ("chart quality lowered on refresh or restart").

The overlays need their own named volume, and it must not be one the
frontend also mounts: nginx would then serve the PNGs without the ownership
check the overlay route performs.
"""
from __future__ import annotations

import os
import sys

import yaml

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import deployment_overlay_store_dir  # noqa: E402 -- needs the TESTS_DIR insert above

# Bind-mounted into the backend-test container (see docker-compose.yml),
# because docker-compose.yml lives at the repo root, outside the ./Backend
# build context.
COMPOSE_PATH = "/compose/docker-compose.yml"


def _covers(mount_target: str, path: str) -> bool:
    """True if a mount at ``mount_target`` persists everything under ``path``."""
    mount_target = mount_target.rstrip("/")
    path = path.rstrip("/")
    return path == mount_target or path.startswith(mount_target + "/")


def _named_volume_mounts(service: dict, top_level_volumes: dict) -> list[tuple[str, str]]:
    """``(source, target)`` for each mount in ``service`` backed by a *named*
    volume, which survives ``docker compose up --build`` / down+up — unlike a
    bind mount or the ephemeral container layer."""
    mounts: list[tuple[str, str]] = []
    for entry in service.get("volumes", []) or []:
        if not isinstance(entry, str):
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        if source in top_level_volumes:  # bare name => named volume
            mounts.append((source, target))
    return mounts


def _persisted_named_volume_targets(service: dict, top_level_volumes: dict) -> list[str]:
    return [target for _source, target in _named_volume_mounts(service, top_level_volumes)]


def _load_compose():
    if not os.path.isfile(COMPOSE_PATH):
        import pytest

        pytest.skip(f"{COMPOSE_PATH} not mounted (run via docker compose backend-test)")
    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_overlay_store_is_backed_by_a_persisted_volume():
    if not os.path.isfile(COMPOSE_PATH):
        import pytest

        pytest.skip(f"{COMPOSE_PATH} not mounted (run via docker compose backend-test)")

    # The *deployment* path (/app/overlay_store/overlays), not the live setting:
    # the suite redirects the overlay store at a tempdir for hermeticity
    # (cache_isolation.isolate_overlay_store), and asserting the volume contract
    # against that tempdir would pass while checking nothing.
    overlay_container_path = deployment_overlay_store_dir()

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


def test_the_overlay_store_is_not_on_a_volume_the_frontend_mounts():
    """Overlays are authenticated (``/chart/{id}/overlay.png`` checks chart
    ownership). A volume shared with nginx would make every overlay PNG
    world-readable while both paths kept working exactly as before.
    """
    compose = _load_compose()
    overlay_path = deployment_overlay_store_dir().rstrip("/")

    top_level = compose.get("volumes", {}) or {}
    frontend_sources = {
        source
        for source, _target in _named_volume_mounts(
            compose["services"].get("frontend", {}), top_level
        )
    }
    exposed = [
        source
        for source, target in _named_volume_mounts(compose["services"]["backend"], top_level)
        if source in frontend_sources and _covers(target, overlay_path)
    ]

    assert not exposed, (
        f"the overlay store {overlay_path!r} rides volume(s) {exposed}, which the "
        "frontend also mounts — the PNGs would be reachable without authentication"
    )
