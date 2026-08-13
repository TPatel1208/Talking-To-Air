# Joining the earthdata-retrieval MCP stack

The satellite path retrieves NASA data through the [harmony-retrieval-mcp](https://github.com/TPatel1208/harmony-retrieval-mcp) stack — a separate repo/stack. The two stacks connect over a shared external Docker network and share the MCP's materialized data volume (read-only), so a retrieved file's `file://` URI resolves as a plain filesystem read in both containers.

**Startup order does not matter, with one exception.** The backend boots without the MCP — ground/EPA features work immediately, and a background task connects to the MCP (with capped retry) and heals the satellite path without a restart once it's up. `/health`'s `earthdata_mcp` field reports `connecting` / `ready` / `unavailable` / `incompatible`. The exception is a genuinely fresh machine: Docker's `external: true` network (`earthdata_net`) and volume (`earthdata_data`) must exist before this stack's `docker compose up` will succeed at all, so the MCP stack has to run **once** first to create them.

1. **First time only** — in the `harmony-retrieval-mcp` repo (with `EARTHDATA_MCP_TRANSPORT=http` in its `.env`), bring the stack up to create the shared network and volume:
   ```bash
   docker compose up --build
   ```

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
