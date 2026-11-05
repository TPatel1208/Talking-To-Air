#!/usr/bin/env bash
# Boots a built backend image against a throwaway Postgres and Redis and waits
# for /health to answer 200.
#
# Usage: ./scripts/smoke-backend-image.sh <image>
#
# CI builds the image but nothing ever started it, so an image that cannot boot
# (a missing runtime dependency, an import that only fails in the slim runtime
# stage, a lifespan step that raises) still went green. The test suite runs on
# the host interpreter, not inside the image, so it cannot catch any of those.
#
# The earthdata MCP is deliberately absent: the backend boots without it (T17)
# and /health reports it as connecting. Every credential is a placeholder; no
# outbound call is needed to reach "ok".
set -euo pipefail

image="${1:?usage: $0 <image>}"
tag="tta-smoke-$$"
net="$tag-net"
timeout_s="${SMOKE_TIMEOUT_SECONDS:-120}"

cleanup() {
  docker rm -f "$tag-backend" "$tag-db" "$tag-redis" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create "$net" >/dev/null
docker run -d --name "$tag-db" --network "$net" --network-alias db \
  -e POSTGRES_DB=talking_to_air_memory -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=smoke \
  postgres:16 >/dev/null
docker run -d --name "$tag-redis" --network "$net" --network-alias redis redis:7-alpine >/dev/null

# Postgres restarts once during first-boot init, so wait for it to accept
# connections before the backend's pool tries.
for _ in $(seq 1 60); do
  if docker exec "$tag-db" pg_isready -U postgres -d talking_to_air_memory >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker run -d --name "$tag-backend" --network "$net" \
  -e DB_HOST=db -e DB_PORT=5432 -e DB_NAME=talking_to_air_memory \
  -e DB_USER=postgres -e DB_PASSWORD=smoke \
  -e REDIS_URL=redis://redis:6379/0 \
  -e SUPABASE_URL=https://smoke-test.supabase.co \
  -e SUPABASE_PUBLISHABLE_KEY=smoke-publishable-key \
  -e GROQ_API_KEY=smoke -e GOOGLE_API_KEY=smoke \
  -e AQS_API_KEY=smoke -e AQS_API_EMAIL=smoke@example.com \
  "$image" >/dev/null

deadline=$((SECONDS + timeout_s))
while (( SECONDS < deadline )); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$tag-backend")" != "true" ]]; then
    echo "backend container exited before becoming healthy:" >&2
    docker logs "$tag-backend" >&2 || true
    exit 1
  fi
  # From inside the container: the image has Python, and nothing else is
  # guaranteed to be on its PATH.
  if body="$(docker exec "$tag-backend" python -c \
      'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).read().decode())' \
      2>/dev/null)"; then
    echo "backend healthy: $body"
    exit 0
  fi
  sleep 2
done

echo "backend did not answer /health with 200 within ${timeout_s}s:" >&2
docker logs "$tag-backend" >&2 || true
exit 1
