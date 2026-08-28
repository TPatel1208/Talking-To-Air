#!/usr/bin/env bash
# One-time local HTTPS setup: mints a locally-trusted cert for the frontend.
#
# Usage: ./scripts/setup-tls.sh
#
# Installs a local CA into your OS/browser trust stores (mkcert -install), then
# writes Frontend/certs/tls.crt + Frontend/certs/tls.key. docker-compose.yml
# bind-mounts that directory to /etc/nginx/certs, which is where
# Frontend/nginx.conf points ssl_certificate/ssl_certificate_key.
#
# The cert is no longer copied into the image at build time, so this is now a
# *runtime* prerequisite, not a build one: `docker compose build frontend`
# succeeds on a clean checkout without ever running this script, and only
# `docker compose up` needs the files to exist. Safe to re-run; mkcert reuses
# the existing CA and overwrites the pair.
set -euo pipefail

if ! command -v mkcert &> /dev/null; then
  echo "mkcert not found. Install it first, then re-run this script:" >&2
  echo "  macOS:   brew install mkcert" >&2
  echo "  Windows: choco install mkcert   (or: scoop install mkcert)" >&2
  echo "  Linux:   https://github.com/FiloSottile/mkcert#installation" >&2
  exit 1
fi

# Trusts mkcert's local CA in your OS/browser stores. No-op if already trusted.
mkcert -install

CERT_DIR="$(cd "$(dirname "$0")/.." && pwd)/Frontend/certs"
mkdir -p "$CERT_DIR"

# tls.crt / tls.key, not mkcert's default "localhost+2.pem" naming: the pair is
# mounted rather than baked, and these are the names Kubernetes already projects
# from a kubernetes.io/tls secret -- so the same nginx.conf works unchanged from
# a laptop to a cluster.
mkcert -cert-file "$CERT_DIR/tls.crt" -key-file "$CERT_DIR/tls.key" localhost 127.0.0.1 ::1

# The key is world-readable as mkcert leaves it; nginx reads it as root before
# dropping privileges, so it does not need to be.
chmod 600 "$CERT_DIR/tls.key" 2>/dev/null || true

echo
echo "Wrote Frontend/certs/tls.crt and Frontend/certs/tls.key."
echo "Now run: docker compose up --build"
