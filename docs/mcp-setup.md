# Joining the earthdata-retrieval MCP stack

The satellite path retrieves NASA data through the [harmony-retrieval-mcp](https://github.com/TPatel1208/harmony-retrieval-mcp) stack — a separate repo/stack. The two stacks connect over a shared external Docker network and share the MCP's materialized data volume (read-only), so a retrieved file's `file://` URI resolves as a plain filesystem read in both containers.

**Startup order does not matter.** The backend boots without the MCP — ground/EPA features work immediately, and a background task connects to the MCP (with capped retry) and heals the satellite path without a restart once it's up. `/health`'s `earthdata_mcp` field reports `connecting` / `ready` / `unavailable` / `incompatible`.

There used to be one exception: the `external: true` network (`earthdata_net`) and volume (`earthdata_data`) must exist before this stack's `docker compose up` will start anything, and the only thing that created them was the MCP stack's first run — so a fresh machine had to clone, configure and build an entire second repo to obtain a bridge network and an empty volume, even for ground/EPA-only use. `scripts/provision.sh` in this repo creates both directly, so that is no longer a prerequisite.

1. **First time only** — create the shared network and volume. Either works:
   ```bash
   ./scripts/provision.sh                      # from this repo, no MCP stack needed
   ```
   ```bash
   docker compose up --build                   # or from harmony-retrieval-mcp,
                                               # with EARTHDATA_MCP_TRANSPORT=http in its .env
   ```
   `provision.sh` is idempotent and a no-op if the resources already exist. It labels the network `com.docker.compose.network=earthdata_net`, which matters: `harmony-retrieval-mcp` declares that network non-external, and compose *hard-fails* when it finds a network it means to create carrying a different value for that label (`network earthdata_net was found but has incorrect label ... (expected: "earthdata_net")`). An unlabelled `docker network create` here would therefore break `docker compose up` over there. If that stack ever renames its network key, its error message names the expected value — re-run with `EARTHDATA_NET_COMPOSE_KEY=<that value>`.

2. **Set `EARTHDATA_MCP_URL` and `EARTHDATA_MCP_TOKEN`** in this repo's `.env` to match that stack's HTTP endpoint and token.

3. **Start this stack** (either order relative to the MCP stack, from here on):
   ```bash
   docker compose up --build
   ```

4. **Smoke check:**
   ```bash
   docker compose exec backend ls /data       # lists what the MCP has materialized
   docker compose exec backend curl http://localhost:8000/health   # earthdata_mcp: ready
   ```
