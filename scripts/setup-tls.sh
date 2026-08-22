#!/usr/bin/env bash
# One-time local HTTPS setup: mints a locally-trusted cert for the frontend.
#
# Usage: ./scripts/setup-tls.sh
#
# Installs a local CA into your OS/browser trust stores (mkcert -install),
# then generates Frontend/localhost+2.pem + Frontend/localhost+2-key.pem —
# the exact filenames Frontend/Dockerfile copies into the nginx image and
# Frontend/nginx.conf points ssl_certificate/ssl_certificate_key at. Safe to
# re-run; mkcert reuses the existing CA and overwrites the cert files.
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

cd "$(dirname "$0")/../Frontend"
mkcert localhost 127.0.0.1 ::1

echo
echo "Wrote Frontend/localhost+2.pem and Frontend/localhost+2-key.pem."
echo "Now run: docker compose up --build"
