"""Deployment-contract guard for the T59 frame blob store.

D8 says a frame stack is **never regenerated in place**: once a chart's blob is
gone, the slider is disabled and the axis is all that survives. So a store with
no named volume does not degrade slowly — every container recreate silently
strips the scrubber off every chart ever plotted, while the frame axis persists
in Postgres and goes on advertising a scrub that 404s. The overlay PNG store
shipped exactly that way once ("chart quality lowered on refresh or restart"),
and the cube store nearly did.

The store is also bounded, which is the other half of why it is separate:
``overlay_store`` has no eviction policy and grows forever (PRD finding 9).
That is real and tracked elsewhere and deliberately **not** fixed here — but it
is precisely why frames were not put in it.
"""
from __future__ import annotations

import os
import sys

import yaml

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import (  # noqa: E402 -- needs the TESTS_DIR insert above
    deployment_frame_store_dir,
    deployment_output_dir,
)

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


def test_the_frame_store_is_backed_by_a_persisted_named_volume():
    compose = _load_compose()
    # The deployment's path, not the ambient one: the suite redirects
    # FRAME_STORE_DIR at a tempdir for hermeticity, and asserting a volume
    # contract against that tempdir would pass while checking nothing.
    frame_dir = deployment_frame_store_dir()
    mounts = _named_volume_mounts(compose["services"]["backend"], compose.get("volumes", {}) or {})

    covering = [m for m in mounts if _covers(m[1], frame_dir)]

    assert covering, (
        f"frame store {frame_dir!r} is not covered by any persisted named volume on the "
        f"backend service (named-volume targets: {[m[1] for m in mounts]}). A container "
        "recreate would wipe every stored frame stack while the frame axis persists in "
        "Postgres, so every chart would keep advertising a scrubber whose blob 404s — and "
        "D8 forbids rebuilding it."
    )
    assert all(m[2] != "ro" for m in covering), (
        f"frame store {frame_dir!r} is mounted read-only; every write would fail into "
        "write_frames' catch-all and no chart would ever get a scrubber."
    )


def test_the_frame_store_has_its_own_volume_rather_than_riding_the_overlay_store():
    """Frames are bounded; overlays are not.

    ``overlay_store`` has no eviction policy and grows forever (PRD finding 9),
    which is *why* frames get their own volume rather than joining it: an LRU
    sweeper on a directory shared with an unbounded store would evict frames to
    make room for PNGs that never leave. Sharing would also make one store's
    growth the other's eviction pressure, with no disclosure surface anywhere.
    """
    compose = _load_compose()
    frame_dir = deployment_frame_store_dir()
    top_level = compose.get("volumes", {}) or {}
    mounts = _named_volume_mounts(compose["services"]["backend"], top_level)

    sources = {source for source, target, _mode in mounts if _covers(target, frame_dir)}

    assert sources and not sources.intersection({"overlay_store", "plot_outputs"}), (
        f"the frame store rides {sorted(sources)}; it needs a bounded volume of its own"
    )


def test_the_frame_store_mount_point_is_owned_by_the_runtime_user():
    """Docker initializes a fresh named volume from the image path it covers,
    and if that path does not exist in the image the volume is created
    **root-owned**. The container runs as ``appuser`` (uid 1001), so every
    write would fail with PermissionError — swallowed by ``write_frames``'
    catch-all, which by design never costs the caller their chart. The store
    would appear to work and never hold a single stack. Caught live on the
    cube store's first deploy; this is the same trap one store over.
    """
    dockerfile = "/app/Dockerfile"
    if not os.path.isfile(dockerfile):
        import pytest

        pytest.skip("Dockerfile not present in this image")

    with open(dockerfile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    frame_dir = deployment_frame_store_dir().rstrip("/")
    created_at = next(
        (i for i, line in enumerate(lines) if line.startswith("RUN mkdir") and frame_dir in line),
        None,
    )
    chowned_at = next((i for i, line in enumerate(lines) if "chown -R appuser" in line), None)

    assert created_at is not None, (
        f"{frame_dir} is never created in the Dockerfile, so Docker will initialize its "
        "named volume root-owned and every write will fail with PermissionError inside "
        "write_frames' catch-all — a store that silently never holds anything."
    )
    assert chowned_at is not None and created_at < chowned_at, (
        f"{frame_dir} is created after the chown, so it stays root-owned and uid 1001 "
        "cannot write into it."
    )


def test_the_frame_store_is_not_reachable_without_authentication():
    """Frames are served only through ``/chart/{id}/frames.f32.gz``, which
    checks chart ownership first. ``/app/outputs`` is handed to nginx wholesale
    and served unauthenticated, so a frame store resolving inside it would make
    every researcher's field world-readable — an access-control regression
    neither path's own tests would notice, because both would keep working.
    """
    compose = _load_compose()
    output_path = deployment_output_dir().rstrip("/")
    frame_dir = deployment_frame_store_dir().rstrip("/")

    assert not _covers(output_path, frame_dir), (
        f"the frame store {frame_dir!r} is inside the public output dir {output_path!r}, "
        "so nginx would serve every stored stack unauthenticated"
    )

    top_level = compose.get("volumes", {}) or {}
    frontend_sources = {
        source
        for source, _target, _mode in _named_volume_mounts(
            compose["services"].get("frontend", {}), top_level
        )
    }
    exposed = [
        source
        for source, target, _mode in _named_volume_mounts(compose["services"]["backend"], top_level)
        if source in frontend_sources and _covers(target, frame_dir)
    ]

    assert not exposed, (
        f"the frame store rides volume(s) {exposed}, which the frontend also mounts — "
        "the stacks would be reachable without authentication"
    )
