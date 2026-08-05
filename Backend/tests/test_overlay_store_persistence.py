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

The public ``/app/outputs`` dir is covered here too, and the separation between
the two is asserted at the *deployment* level rather than only in settings —
a volume layout that quietly put the private store inside the public one would
be an access-control regression that no unit test of either path would catch.
"""
from __future__ import annotations

import os
import sys

import yaml

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import (  # noqa: E402 -- needs the TESTS_DIR insert above
    deployment_output_dir,
    deployment_overlay_store_dir,
)

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


def test_the_output_dir_is_backed_by_the_volume_the_frontend_serves():
    """``/app/outputs`` must land on the shared ``plot_outputs`` volume.

    This directory had no deployment-contract test at all until now — the
    overlay store and the cube store each had one, and the public output dir,
    the *oldest* of the three, was never covered. It became testable when
    ``OUTPUT_DIR`` stopped being an ``APP_ROOT``-relative constant, because
    ``deployment_output_dir()`` can now name the container path.

    The property asserted is functional rather than "a volume called
    plot_outputs": the volume backing ``/app/outputs`` must be the same one
    mounted into the frontend, because that shared mount is precisely how a
    chart PNG written by the backend becomes reachable at ``/outputs`` without
    the backend serving it. A named volume the frontend did not mount would
    persist the files and still 404 every one of them.
    """
    compose = _load_compose()
    output_path = deployment_output_dir()
    top_level = compose.get("volumes", {}) or {}

    backend_mounts = _named_volume_mounts(compose["services"]["backend"], top_level)
    frontend_sources = {
        source
        for source, _target in _named_volume_mounts(
            compose["services"].get("frontend", {}), top_level
        )
    }

    covering = [source for source, target in backend_mounts if _covers(target, output_path)]

    assert covering, (
        f"output dir {output_path!r} is not covered by any persisted named volume on "
        f"the backend service (named-volume targets: "
        f"{[t for _s, t in backend_mounts]}). Chart PNGs would be wiped on every "
        "container recreate while the chart payloads referencing them persist in "
        "Postgres."
    )
    assert frontend_sources.intersection(covering), (
        f"output dir {output_path!r} is on volume(s) {covering} but the frontend "
        f"mounts {sorted(frontend_sources)}. nginx serves /outputs straight off the "
        "shared volume, so a backend-only volume persists the PNGs and still 404s "
        "every one of them."
    )


def test_the_overlay_store_is_not_inside_the_publicly_served_output_volume():
    """The two stores must not collapse onto one directory *in the deployment*.

    Overlays are authenticated (``/chart/{id}/overlay.png`` checks chart
    ownership); ``/outputs`` is not served by the backend at all, it is handed to
    nginx wholesale. If the overlay store ever resolved inside the output dir,
    every overlay PNG would become world-readable — an access-control regression
    that no unit test of either path would notice, because both would keep
    working exactly as before.
    """
    compose = _load_compose()
    output_path = deployment_output_dir().rstrip("/")
    overlay_path = deployment_overlay_store_dir().rstrip("/")

    assert not _covers(output_path, overlay_path), (
        f"the overlay store {overlay_path!r} is inside the public output dir "
        f"{output_path!r}, so nginx would serve every overlay PNG unauthenticated"
    )

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
