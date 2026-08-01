"""Deployment-contract guard for the T52 Zarr cube store.

A cube costs minutes to build — the full open pipeline plus a read, compress
and write of the whole dataset — and `tta-backend` is rebuilt constantly
(CLAUDE.md). Without a named volume the store is wiped on every container
recreate, so it would be empty every time anyone looked and the feature would
surface as "the cache doesn't help" rather than as a broken deployment. That
is the `overlay_store` failure mode exactly ("chart quality lowered on refresh
or restart"), which is why this test exists before anyone reports it.

The volume must also be backend-only and read-write: the frontend has no
business reading cubes, and a read-only mount would fail every write silently
into the background writer's catch-all.
"""
from __future__ import annotations

import os
import sys

import yaml

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import deployment_cube_store_dir  # noqa: E402 -- needs the TESTS_DIR insert above

# Bind-mounted into the backend-test container (see docker-compose.yml),
# because docker-compose.yml lives at the repo root, outside the ./Backend
# build context.
COMPOSE_PATH = "/compose/docker-compose.yml"


def _load_compose():
    if not os.path.isfile(COMPOSE_PATH):
        import pytest

        pytest.skip(f"{COMPOSE_PATH} not mounted (run via docker compose backend-test)")
    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _named_volume_mounts(service: dict, top_level_volumes: dict) -> list[tuple[str, str, str]]:
    """``(source, target, mode)`` for each mount backed by a *named* volume,
    which survives ``docker compose up --build`` / down+up — unlike a bind
    mount or the ephemeral container layer."""
    mounts = []
    for entry in service.get("volumes", []) or []:
        if not isinstance(entry, str):
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        mode = parts[2] if len(parts) > 2 else "rw"
        if source in top_level_volumes:  # bare name => named volume
            mounts.append((source, target.rstrip("/"), mode))
    return mounts


def _covers(mount_target: str, path: str) -> bool:
    path = path.rstrip("/")
    return path == mount_target or path.startswith(mount_target + "/")


def test_the_cube_store_is_backed_by_a_persisted_named_volume():
    compose = _load_compose()
    # The deployment's path, not the ambient one: the suite redirects
    # CUBE_STORE_DIR to a tempdir so tests stop reading and writing the real
    # store, and a tempdir is of course on no named volume.
    cube_dir = os.path.abspath(deployment_cube_store_dir())
    mounts = _named_volume_mounts(compose["services"]["backend"], compose.get("volumes", {}) or {})

    covering = [m for m in mounts if _covers(m[1], cube_dir)]

    assert covering, (
        f"cube store {cube_dir!r} is not covered by any persisted named volume on the "
        f"backend service (named-volume targets: {[m[1] for m in mounts]}). Cubes cost "
        "minutes to build and tta-backend is rebuilt constantly, so without a volume the "
        "store is empty every time you look and the cache silently never helps."
    )
    assert all(m[2] != "ro" for m in covering), (
        f"cube store {cube_dir!r} is mounted read-only; every cube write would fail "
        "silently inside the background writer's catch-all."
    )


def test_the_cube_store_mount_point_is_owned_by_the_runtime_user():
    """Docker initializes a fresh named volume from the image path it covers —
    and if that path does not exist in the image, the volume is created
    **root-owned**. The container runs as `appuser` (uid 1001), so every cube
    write would then fail with PermissionError, swallowed by the background
    writer's catch-all: the cache would appear to work and never store a
    single cube. Caught live on the first deploy of this feature.

    The fix is to create the directory in the Dockerfile *before* the
    ``chown -R appuser:appuser /app``, so the fresh volume inherits that
    ownership. This test asserts the ordering, since getting it wrong is
    invisible until someone checks a volume that has already been created.
    """
    dockerfile = "/app/Dockerfile"
    if not os.path.isfile(dockerfile):
        import pytest

        pytest.skip("Dockerfile not present in this image")

    with open(dockerfile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    cube_dir = deployment_cube_store_dir().rstrip("/")
    created_at = next(
        (i for i, line in enumerate(lines) if line.startswith("RUN mkdir") and cube_dir in line),
        None,
    )
    chowned_at = next((i for i, line in enumerate(lines) if "chown -R appuser" in line), None)

    assert created_at is not None, (
        f"{cube_dir} is never created in the Dockerfile, so Docker will initialize its "
        "named volume root-owned and every cube write will fail with PermissionError "
        "inside the writer's catch-all — a cache that silently never stores anything."
    )
    assert chowned_at is not None and created_at < chowned_at, (
        f"{cube_dir} is created after the chown, so it stays root-owned and uid 1001 "
        "cannot write into it."
    )


def test_the_cube_store_is_not_mounted_on_the_frontend():
    compose = _load_compose()
    frontend = compose["services"].get("frontend", {})
    top_level = compose.get("volumes", {}) or {}

    sources = [source for source, _target, _mode in _named_volume_mounts(frontend, top_level)]

    assert not any("cube" in source for source in sources), (
        f"the cube store is mounted on the frontend ({sources}); unlike plot_outputs it is "
        "not public content and nginx has no reason to serve it."
    )
