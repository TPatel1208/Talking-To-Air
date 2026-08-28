#!/usr/bin/env bash
# Creates the shared Docker resources this stack joins as `external`.
#
# Usage: ./scripts/provision.sh
#
# docker-compose.yml declares two resources it does not own:
#
#   networks: earthdata_net   -- how the backend reaches the MCP stack's `mcp`
#   volumes:  earthdata_data  -- the MCP's materialized files, mounted read-only
#
# Both are `external: true`, which means compose refuses to start the stack
# until they exist -- it will not create them. Until now the only thing that
# created them was the harmony-retrieval-mcp stack's first `docker compose up`,
# so standing this stack up on a fresh host meant cloning, configuring and
# building an entirely separate repo to obtain an empty volume and a bridge
# network. That is a hard dependency for the *ground/EPA-only* path too, which
# never touches the MCP at all.
#
# This script creates them directly. Idempotent: safe to re-run, and a no-op if
# the MCP stack already made them.
set -euo pipefail

NETWORK="${EARTHDATA_NET:-earthdata_net}"
VOLUME="${EARTHDATA_VOLUME:-earthdata_data}"

# The key `earthdata_net` appears under `networks:` in the harmony-retrieval-mcp
# compose file. See the long note below for why this script has to care.
NET_COMPOSE_KEY="${EARTHDATA_NET_COMPOSE_KEY:-earthdata_net}"

if ! docker info > /dev/null 2>&1; then
  echo "Cannot talk to the Docker daemon. Is Docker Desktop running?" >&2
  exit 1
fi

# ── Network ───────────────────────────────────────────────────────────────────
#
# Created WITH a compose ownership label, which looks odd for a resource this
# script owns. The reason is measured, not theoretical:
#
#   * This stack joins the network as `external: true`, so it does not care
#     about labels at all -- any network of that name satisfies it.
#   * The harmony-retrieval-mcp stack declares the same network *non*-external
#     (fixed `name:`, so it is the creator). When compose finds a network that
#     already exists under a name it means to create, it checks the label
#     `com.docker.compose.network` against its own key for that network and
#     HARD FAILS on a mismatch:
#
#       network earthdata_net was found but has incorrect label
#       com.docker.compose.network set to "" (expected: "earthdata_net")
#
#     An unlabeled network -- i.e. a plain `docker network create` -- therefore
#     bricks `docker compose up` in the *other* repo. Provisioning this stack
#     must not break that one.
#
# Labelling it with the key the other stack expects makes that stack adopt the
# network silently. Only this one label matters; the project label is not
# checked. Volumes are more forgiving -- an unlabeled one is adopted with a
# warning -- but the network is fatal, which is why only this half is careful.
#
# If harmony-retrieval-mcp ever renames its network key, its compose will print
# the expected value in exactly the error above; re-run with
# EARTHDATA_NET_COMPOSE_KEY=<that value>.
if docker network inspect "$NETWORK" > /dev/null 2>&1; then
  existing_key="$(docker network inspect "$NETWORK" \
    --format '{{index .Labels "com.docker.compose.network"}}' 2>/dev/null || true)"
  echo "network  $NETWORK already exists (compose key: '${existing_key}') -- leaving it alone"
  if [ -n "$existing_key" ] && [ "$existing_key" != "$NET_COMPOSE_KEY" ]; then
    echo "  note: that differs from EARTHDATA_NET_COMPOSE_KEY='$NET_COMPOSE_KEY'." >&2
    echo "        Harmless for this stack (it joins as external), but the stack that" >&2
    echo "        owns the network may refuse to start. Nothing to do unless it does." >&2
  fi
else
  docker network create --label "com.docker.compose.network=${NET_COMPOSE_KEY}" "$NETWORK" > /dev/null
  echo "network  $NETWORK created"
fi

# ── Volume ────────────────────────────────────────────────────────────────────
#
# Starts empty, which is correct and not a degraded state: it is the MCP stack's
# materialization target, mounted here read-only so this stack is never a second
# writer. With no MCP running there is nothing to read, and the backend boots
# fine without it (T17) -- /health reports earthdata_mcp: unavailable and the
# ground/EPA path works. Files appear as the MCP materializes them.
if docker volume inspect "$VOLUME" > /dev/null 2>&1; then
  echo "volume   $VOLUME already exists -- leaving it alone"
else
  docker volume create "$VOLUME" > /dev/null
  echo "volume   $VOLUME created"
fi

echo
echo "Shared resources ready. Next: docker compose up --build"
