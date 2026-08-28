# ── Stage 1: Build the React/Vite app ────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN \
  if [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
  elif [ -f pnpm-lock.yaml ]; then npm install -g pnpm && pnpm install --frozen-lockfile; \
  else npm ci; fi

COPY . .

# In production the frontend calls /api/* which nginx proxies to the backend.
# Override at build time if deploying frontend separately.
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build


# ── Stage 2: Serve with nginx ─────────────────────────────────────────────────
FROM nginx:1.25-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

# The TLS keypair is deliberately NOT baked in. It used to be two
# `COPY localhost+2*.pem` lines, which made this image unbuildable from a clean
# checkout -- those files are mkcert output, `.gitignore`d as `*.pem`, and so
# never present on a fresh clone. CI papered over it by minting a throwaway
# self-signed pair just to satisfy the COPY, which is the tell: the build was
# depending on a file that only ever existed on a developer's machine.
#
# Baking it was also wrong on its own terms. A private key in an image layer is
# a private key in every registry, cache and `docker history` that image passes
# through, and it pins one image to one hostname -- rebuilding to renew a cert
# or to serve a different domain is not something a release artifact should
# need. The cert is now deployment configuration: mount it at /etc/nginx/certs
# (see docker-compose.yml and scripts/setup-tls.sh).
RUN mkdir -p /etc/nginx/certs

# nginx's entrypoint runs /docker-entrypoint.d/*.sh before starting the server.
# Without this guard a missing mount surfaces as nginx failing to open a path
# nobody mounted, which reads like an image bug; this names the actual cause and
# the command that fixes it. chmod because the entrypoint skips files that are
# not executable -- silently, with only an "Ignoring" line -- and the bit does
# not survive a checkout on every platform.
COPY docker-entrypoint.d/10-require-tls-cert.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/10-require-tls-cert.sh

EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
