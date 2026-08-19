import asyncio
import json
import logging
import os
import time
import tracemalloc
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Optional

import psycopg
from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.routing import Match

from tta_backend.agents.earthdata_agent import LazySatelliteAgent, build_earthdata_agent, refresh_live_tools
from tta_backend.agents.ground_sensor_agent import build_ground_agent
from tta_backend.agents.supervisor_agent import build_agent
from tta_backend.config.connectors import CONNECTOR_REGISTRY, CONNECTOR_REGISTRY_BY_TYPE
from tta_backend.config.settings import get_settings
from tta_backend.config.starter_prompts import STARTER_PROMPTS
from tta_backend.earthdata_mcp.connection import STATE_CONNECTING, STATE_READY, EarthdataMCPConnectionManager
from tta_backend.earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_NOT_FOUND,
    CATEGORY_NO_DATA,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_TOO_LARGE,
    CATEGORY_USER_INPUT,
    MCPToolError,
)
from tta_backend.repositories.chart_repository import ensure_chart_table
from tta_backend.repositories.session_metadata_repository import (
    ensure_session_metadata_table,
    get_session_metadata,
    save_session_metadata_once,
    session_belongs_to_user,
)
from tta_backend.repositories.revoked_token_repository import ensure_revoked_token_table, revoke_token
from tta_backend.repositories.session_repository import SessionRepository
from tta_backend.repositories.user_connector_repository import (
    delete_connector,
    ensure_user_connector_table,
    list_connectors_for_user,
    upsert_connector,
)
from tta_backend.repositories.user_repository import create_user, ensure_user_table, get_user_by_username
from tta_backend.repositories.artifact_repository import ensure_artifact_table
from tta_backend.services import cube_cache, frame_store, warmup
from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION
from tta_backend.services.auth_service import authenticate_request, create_access_token, hash_password, verify_password
from tta_backend.services.connector_credential_service import EdlCredentialInjector
from tta_backend.services.connector_token_service import TokenValidationError, decode_token_expiry
from tta_backend.services.artifact_store import artifact_store
from tta_backend.services.chat_stream_service import ChatStreamService
from tta_backend.services.chart_service import ChartService
from tta_backend.services.export_service import ExportService, materialize_first_chunk
from tta_backend.services.history_service import HistoryService
from tta_backend.services.data_download_service import DataDownloadError, export_converted, iter_file_chunks
from tta_backend.services.discovery_service import (
    check_coverage,
    describe_dataset,
    inspect_granules,
    preview_dataset,
    search_datasets,
)
from tta_backend.services.jobs_service import cancel_job, list_jobs
from tta_backend.services.methods_export_service import build_methods_markdown
from tta_backend.services.provenance_service import get_citations, get_lineage
from tta_backend.utils.connector_crypto import encrypt_secret, get_connector_cipher
from tta_backend.utils.db import active_pool_connections, check_db_pool, close_db_pool, init_db_pool, validate_config
from tta_backend.utils.logging import configure_logging
from tta_backend.utils.metrics import (
    observe_http_request,
    prometheus_content_type,
    refresh_process_gauges,
    render_prometheus_metrics,
    set_db_pool_connections_active,
)
from tta_backend.utils.streaming import current_user_id, iter_with_user_id, user_id_context

agent = None
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

chart_service = ChartService()
export_service = ExportService(settings.csv_export_max_granules)
history_service = HistoryService(chart_service)
session_repository = SessionRepository()


async def _on_earthdata_mcp_ready(tools: dict) -> None:
    """earthdata_mcp_manager's on_ready hook (T17): refreshes the persistent
    app.state.earthdata_mcp_tools dict (read directly by the unmigrated chart
    export.png endpoint; export.csv/.nc moved to the readiness gate in T37) and
    rebuilds the real satellite agent into whatever LazySatelliteAgent the
    current lifespan cycle assigned to app.state.satellite_agent — see
    agents/earthdata_agent.py for why a mutable placeholder, not a reassigned
    reference, is what makes this visible to the supervisor's already-built
    ask_earthdata_agent tool closure.

    In-flight recovery (2026-07-21): the MCP server crash-restarts, killing the
    long-lived session; the manager reconnects and re-fires this hook with a new
    session's bound tools. Refresh the tools dict IN PLACE (refresh_live_tools)
    and rebuild the agent bound to that SAME dict — so a chat turn compiled
    against it before the restart (an in-flight compare mid-retry) reads the
    reconnected tools on its next call instead of retrying the dead session. See
    refresh_live_tools for the call-time-indexing contract this relies on."""
    live = app.state.earthdata_mcp_tools
    if live is None:  # a late callback racing lifespan shutdown — nothing to refresh
        return
    refresh_live_tools(live, tools)
    app.state.satellite_agent.set_real(build_earthdata_agent(mcp_tools=live))
    logger.info("earthdata_mcp_satellite_agent_ready", extra={"_event": "earthdata_mcp_satellite_agent_ready"})


# T31: one shared instance -- its in-process cache and per-turn
# last_used_at coalescing are only meaningful as process-lifetime state, and
# the connector endpoints below invalidate it directly on re-paste/disconnect.
edl_credential_injector = EdlCredentialInjector(settings)
earthdata_mcp_manager = EarthdataMCPConnectionManager(
    settings, current_user_id, on_ready=_on_earthdata_mcp_ready, edl_injector=edl_credential_injector,
)
chat_stream_service = ChatStreamService(
    chart_service, settings.long_request_seconds, earthdata_mcp_manager, settings.chat_turn_timeout_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    validate_config()
    await init_db_pool()
    await ensure_user_table()
    await ensure_revoked_token_table()
    await ensure_session_metadata_table()
    await ensure_user_connector_table()
    await ensure_artifact_table()
    await ensure_chart_table()

    # T52: reclaim staging dirs and manifest-less entries a crash mid-write
    # left behind. Neither is ever served (the manifest is the completion
    # marker), so this is space, not correctness. T54: this also rebuilds the
    # handle->cube index from the surviving manifests, which is what makes that
    # index derived state — losing the file costs a boot's scan, never an
    # answer.
    cube_cache.sweep_store(OPEN_PIPELINE_VERSION)

    # T59 D8: the same reclaim for the frame store. Frames are never served or
    # rebuilt across a version change, so after a bump every stack in there is
    # unreachable — but its manifest is still valid, so nothing else would ever
    # drop it and its bytes would go on counting against the cap.
    frame_store.sweep_store(OPEN_PIPELINE_VERSION)

    # Pay the render path's one-time lazy-import cost here rather than charging
    # it to whoever asks the first question: measured at ~0.44 s of dask and
    # ~0.10 s of PIL, pulled in by the aggregate-and-render chain the first
    # time any plot, statistic or comparison runs. Synchronous on purpose --
    # nothing is being served yet, and a request that arrives mid-warm is no
    # worse off than it was before this existed.
    warmup.warm_render_path()

    logger.info("startup_begin", extra={"_model": settings.llm_model})
    # T17: the backend boots without the earthdata-retrieval MCP — ground/EPA
    # features work immediately. earthdata_mcp_manager runs a background
    # connect loop (capped exponential backoff) instead of the old hard
    # boot-time raise; misconfiguration (a malformed URL) still fails loud
    # from validate_config() above, before any of this runs. satellite_agent
    # is a LazySatelliteAgent placeholder — services.subagent_dispatch
    # .run_satellite gates on earthdata_mcp_manager.state before ever
    # touching it, so it's never invoked before _on_earthdata_mcp_ready
    # (module scope) fills it in.
    app.state.earthdata_mcp_tools = {}
    app.state.earthdata_mcp_manager = earthdata_mcp_manager
    app.state.satellite_agent = LazySatelliteAgent()
    earthdata_mcp_manager.start()

    # Built once here (not inside build_agent) so the supervisor's tool
    # wrappers and the router fast path (services/chat_stream_service.py,
    # T14) invoke the identical sub-agent instances.
    ground_agent = build_ground_agent()
    app.state.ground_agent = ground_agent
    agent = await build_agent(
        settings.llm_model,
        ground_agent=ground_agent,
        satellite_agent=app.state.satellite_agent,
        mcp_manager=earthdata_mcp_manager,
    )
    app.state.agent = agent
    logger.info("startup_complete")
    try:
        yield
    finally:
        await earthdata_mcp_manager.stop()
        agent = None
        app.state.agent = None
        app.state.ground_agent = None
        app.state.satellite_agent = None
        app.state.earthdata_mcp_tools = None
        app.state.earthdata_mcp_manager = None
        await close_db_pool()
        logger.info("shutdown_complete")


app = FastAPI(title="Talking to Air API", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The one live consumer of the public output dir. StaticFiles resolves and
# checks the directory when it is mounted, so this genuinely has to exist at
# import — which is why the makedirs stays here while the two dead copies of
# this constant (plot_tools, stat_tools) were simply deleted. Resolved from
# settings so the test suite lands in a tempdir instead of creating
# `Backend/outputs/` inside the checkout.
OUTPUT_DIR = settings.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

PUBLIC_ENDPOINTS = {
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/capabilities/starters"),
    ("GET", "/config/map-tiles"),
    ("POST", "/auth/login"),
    ("POST", "/auth/register"),
}
ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]
JobHandle = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]
DatasetHandle = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]
ConnectorType = Annotated[str, Path(pattern=r"^[a-z0-9_-]+$")]


def _route_path(request: Request) -> str:
    for route in app.routes:
        match, _ = route.matches(request.scope)
        if match != Match.NONE:
            return getattr(route, "path", request.url.path)
    return request.url.path


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    path = _route_path(request)
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        set_db_pool_connections_active(active_pool_connections())
        # A streaming response (StreamingResponse, or the equivalent
        # call_next builds when another BaseHTTPMiddleware wraps this one)
        # exposes body_iterator; call_next already returned by the time
        # headers exist, well before an SSE stream's slow work runs, so
        # observing here would put every /chat turn at ~0ms. Defer to the
        # iterator's own close instead; non-streaming routes are unaffected.
        if response is not None and hasattr(response, "body_iterator"):
            _observe_request_metrics_at_stream_close(response, request.method, path, status_code, started)
        else:
            observe_http_request(request.method, path, status_code, time.perf_counter() - started)


def _observe_request_metrics_at_stream_close(
    response, method: str, path: str, status_code: int, started: float,
) -> None:
    original_iterator = response.body_iterator

    async def _timed_iterator():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            observe_http_request(method, path, status_code, time.perf_counter() - started)

    response.body_iterator = _timed_iterator()


@app.middleware("http")
async def require_authentication(request: Request, call_next):
    if request.method == "OPTIONS" or (request.method, request.url.path) in PUBLIC_ENDPOINTS:
        return await call_next(request)
    if not any(route.matches(request.scope)[0] != Match.NONE for route in app.routes):
        return await call_next(request)
    try:
        request.state.current_user = await authenticate_request(request)
    except HTTPException as exc:
        return Response(
            content=json.dumps({"detail": exc.detail}),
            status_code=exc.status_code,
            media_type="application/json",
            headers=exc.headers,
        )
    return await call_next(request)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    thread_id: Optional[str] = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9-]+$")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=1, max_length=1024)

class RegisterRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls,value: str) -> str:
        if len(value) < 3 or len(value) > 64:
            raise ValueError("Username must be between 3 and 64 characters")
        if " " in value:
            raise ValueError("Username cannot contain spaces")

        return value
    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value) < 8 or len(value) > 1024:
            raise ValueError("Password must be between 8 and 1024 characters")
        if " " in value:
            raise ValueError("Password cannot contain spaces")
        if not any(c.isupper() for c in value):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in value):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in value):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in value):
            raise ValueError("Password must contain at least one special character")
        return value

class DiscoverySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    filters: Optional[dict] = None


class DiscoveryPreviewRequest(BaseModel):
    location: Optional[str] = Field(default=None, min_length=1, max_length=200)
    time_range: Optional[str] = Field(default=None, min_length=1, max_length=200)
    layer: Optional[str] = Field(default=None, min_length=1, max_length=200)


class DiscoveryCoverageRequest(BaseModel):
    location: str = Field(min_length=1, max_length=200)
    time_range: str = Field(min_length=1, max_length=200)


class DiscoveryGranulesRequest(BaseModel):
    location: str = Field(min_length=1, max_length=200)
    time_range: str = Field(min_length=1, max_length=200)
    limit: Optional[int] = Field(default=None, ge=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: str
    username: str
    is_active: bool


class SetConnectorTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=8192)


class ConnectorStatusView(BaseModel):
    connector_type: str
    display_name: str
    auth_method: str
    description: str
    token_docs_url: str
    status: str
    connected_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@app.post("/auth/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register(req: RegisterRequest):
    password_hash = hash_password(req.password)
    try:
        user = await create_user(req.username, password_hash)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    return UserResponse(id=user.id, username=user.username, is_active=user.is_active)


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = await get_user_by_username(req.username)
    if user is None or not user.is_active or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user)
    return TokenResponse(access_token=token, expires_in=expires_in)


@app.post("/auth/logout")
async def logout(request: Request):
    payload = getattr(request.state, "jwt_payload", {})
    jti = payload.get("jti")
    exp = payload.get("exp")
    if not jti or exp is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    await revoke_token(jti, datetime.fromtimestamp(exp, tz=timezone.utc))
    return {"detail": "Logged out"}


# T30: per-user connector (e.g. Earthdata Login) token storage. Registry-
# driven -- CONNECTOR_REGISTRY is the single source of what cards the
# frontend can render and what connector_type values these endpoints accept.
# No endpoint here ever selects or returns encrypted_secret (see
# repositories/user_connector_repository.py); this phase only stores the
# token, nothing consumes it yet.
def _connector_status(row: dict | None) -> str:
    if row is None:
        return "not_connected"
    expires_at = row.get("expires_at")
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        return "expired"
    if row.get("status") == "error":
        return "error"
    return "connected"


def _connector_view(entry: dict, row: dict | None) -> ConnectorStatusView:
    return ConnectorStatusView(
        connector_type=entry["connector_type"],
        display_name=entry["display_name"],
        auth_method=entry["auth_method"],
        description=entry["description"],
        token_docs_url=entry["token_docs_url"],
        status=_connector_status(row),
        connected_at=row.get("connected_at") if row else None,
        expires_at=row.get("expires_at") if row else None,
    )


def _require_connector_cipher():
    cipher = get_connector_cipher(settings)
    if cipher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Connectors are not configured on this deployment.",
        )
    return cipher


def _connector_registry_entry(connector_type: str) -> dict:
    entry = CONNECTOR_REGISTRY_BY_TYPE.get(connector_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown connector type")
    return entry


@app.get("/connectors")
async def list_connectors_endpoint(request: Request):
    _require_connector_cipher()
    rows = {row["connector_type"]: row for row in await list_connectors_for_user(request.state.current_user.id)}
    return {"connectors": [_connector_view(entry, rows.get(entry["connector_type"])) for entry in CONNECTOR_REGISTRY]}


@app.put("/connectors/{connector_type}/token")
async def set_connector_token_endpoint(connector_type: ConnectorType, req: SetConnectorTokenRequest, request: Request):
    cipher = _require_connector_cipher()
    entry = _connector_registry_entry(connector_type)
    try:
        expires_at = decode_token_expiry(req.token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))

    encrypted_secret = encrypt_secret(cipher, req.token)
    row = await upsert_connector(
        request.state.current_user.id, connector_type, entry["auth_method"], encrypted_secret, expires_at,
    )
    # T31: a re-paste must never be shadowed by the injector's short-TTL
    # cache of the previous (possibly invalid/expired) row.
    edl_credential_injector.invalidate(request.state.current_user.id)
    return _connector_view(entry, row)


@app.delete("/connectors/{connector_type}")
async def disconnect_connector_endpoint(connector_type: ConnectorType, request: Request):
    _require_connector_cipher()
    entry = _connector_registry_entry(connector_type)
    await delete_connector(request.state.current_user.id, connector_type)
    # T31: a disconnect must never leave a stale cached token injectable
    # for the rest of the cache's TTL window.
    edl_credential_injector.invalidate(request.state.current_user.id)
    return _connector_view(entry, None)


@app.get("/health")
async def health():
    db_ok, db_error = await check_db_pool(timeout_seconds=2.0)
    active_agent = getattr(app.state, "agent", None) or agent
    agent_ok = active_agent is not None
    # T17 story #6: the data layer's connection state, not just db/agent —
    # so an MCP outage or schema mismatch is visible to monitoring the same
    # way it's visible to a researcher.
    manager = getattr(app.state, "earthdata_mcp_manager", None)
    earthdata_mcp_state = manager.state if manager is not None else STATE_CONNECTING
    if db_ok and agent_ok:
        return {"status": "ok", "db": True, "agent": True, "earthdata_mcp": earthdata_mcp_state}

    body = {"status": "degraded", "db": db_ok, "agent": agent_ok, "earthdata_mcp": earthdata_mcp_state}
    if db_error:
        body["db_error"] = db_error
    if not agent_ok:
        body["agent_error"] = "agent is not initialized"
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)


@app.get("/metrics")
def metrics():
    refresh_process_gauges()
    return Response(content=render_prometheus_metrics(), media_type=prometheus_content_type())


@app.get("/debug/heap-snapshot")
async def heap_snapshot(limit: int = 25):
    """T45: tracemalloc top-allocations snapshot for chasing a specific
    memory incident (the 2026-07-17 QA jump-and-plateau) -- gated behind
    DEBUG_HEAP_PROFILING_ENABLED, off by default, since tracemalloc adds
    per-allocation overhead this deployment shouldn't pay standingly.
    Requires auth like every other route (not in PUBLIC_ENDPOINTS)."""
    if not settings.debug_heap_profiling_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        return {"top": [], "note": "tracemalloc just started; call again after some traffic for history."}
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")[:limit]
    return {
        "top": [
            {
                "file": stat.traceback[0].filename,
                "line": stat.traceback[0].lineno,
                "size_bytes": stat.size,
                "count": stat.count,
            }
            for stat in top_stats
        ],
    }


@app.get("/config/map-tiles")
def config_map_tiles():
    """T23: basemap/terrain tile sources as configuration, not code, so a
    keyed or self-hosted provider can be swapped in without a redeploy.
    Unauthenticated -- these are non-sensitive, static URLs the map needs
    before the chart underneath it can even render."""
    return {
        "basemap_light_url": settings.map_basemap_light_url,
        "basemap_dark_url": settings.map_basemap_dark_url,
        "terrain_dem_url": settings.map_terrain_dem_url,
        "basemap_attribution": settings.map_basemap_attribution,
        "terrain_attribution": settings.map_terrain_attribution,
    }


@app.get("/capabilities/starters")
def capabilities_starters():
    """T22: the empty-chat's example questions — unauthenticated so a
    first-time visitor sees them before signing in. The single backend-owned
    constant (config.starter_prompts) is also what the eval harness's
    task-coverage assertion checks, so nothing here can drift into a broken
    promise (story #11)."""
    return STARTER_PROMPTS


# T18: one exception handler for every classified MCP tool outcome — pane
# endpoints, agent tools, and chat answers all trace back to the same
# taxonomy (story #11: one JSON error shape across every endpoint). T17's
# unavailable/incompatible states render through this same handler (story
# #13) via _earthdata_tools raising MCPToolError below, rather than a
# second, differently-shaped 503.
_CATEGORY_STATUS_CODES = {
    CATEGORY_USER_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    CATEGORY_TOO_LARGE: status.HTTP_422_UNPROCESSABLE_CONTENT,
    CATEGORY_NO_DATA: status.HTTP_200_OK,
    CATEGORY_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    CATEGORY_PROVIDER_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    CATEGORY_CONTRACT: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


@app.exception_handler(MCPToolError)
async def _handle_mcp_tool_error(request: Request, exc: MCPToolError) -> JSONResponse:
    status_code = _CATEGORY_STATUS_CODES.get(exc.category, status.HTTP_500_INTERNAL_SERVER_ERROR)
    body: dict = {"category": exc.category, "message": exc.message}
    if exc.suggestion:
        body["suggestion"] = exc.suggestion
    return JSONResponse(status_code=status_code, content={"error": body})


def _earthdata_tools(request: Request) -> dict:
    """Discovery/jobs/provenance endpoints' MCP tools, read through
    earthdata_mcp_manager (T17) rather than app.state.earthdata_mcp_tools
    directly, so a not-ready connection answers with the shared structured
    503 instead of proxying a bare 500 from an empty/absent tool dict."""
    manager = getattr(request.app.state, "earthdata_mcp_manager", None)
    state = manager.state if manager is not None else STATE_CONNECTING
    if manager is None or state != STATE_READY:
        raise MCPToolError(
            CATEGORY_PROVIDER_UNAVAILABLE,
            f"The satellite data layer is temporarily unavailable (earthdata_mcp: {state}).",
            suggestion="Ground/EPA endpoints are unaffected. Try again in a moment.",
        )
    return manager.tools


@app.get("/chart/{chart_id}/export.csv")
async def export_chart_csv(chart_id: str, request: Request):
    # T37: resolved through the T17 readiness gate (shared structured 503),
    # never app.state.earthdata_mcp_tools directly — a not-ready MCP must
    # fail here, before any 200 header is committed, not inside the
    # StreamingResponse generator.
    tools = _earthdata_tools(request)
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    if not payload.get("export"):
        raise HTTPException(status_code=422, detail="This chart does not include full-resolution export metadata.")

    # T37: materialize the first chunk before committing to a 200 — the
    # common failures (missing handle, evicted export) raise here as a clean
    # 4xx/5xx instead of truncating the download mid-stream. An MCPToolError
    # propagates to the shared taxonomy handler.
    #
    # iter_with_user_id wraps the row generator rather than a `with
    # user_id_context(...)` wrapping only this await: StreamingResponse pulls
    # every chunk after the first one *outside* the handler, and this export
    # opens a handle per panel — a heatmap_multi comparison reaches panel B
    # long after the first 64 KiB chunk went out. Bound at the innermost seam,
    # so every pull, first or last, carries the request's user.
    try:
        stream = await materialize_first_chunk(
            iter_with_user_id(
                request.state.current_user.id,
                export_service.iter_chart_csv_chunks(payload, tools),
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StreamingResponse(
        stream,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{export_service.safe_export_name(payload, "csv")}"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chart/{chart_id}/export.png")
async def export_chart_png(chart_id: str, request: Request):
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    try:
        with user_id_context(request.state.current_user.id):
            content = await export_service.build_chart_png(payload, request.app.state.earthdata_mcp_tools)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(
        content=content,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{export_service.safe_export_name(payload, "png")}"'},
    )


def _chart_overlay_path(payload: dict, panel: int | None) -> str | None:
    """Resolve the stored overlay PNG path for a chart, or its Nth panel /
    difference panel for a heatmap_multi comparison (T23)."""
    if payload.get("type") == "heatmap_multi":
        if panel is not None:
            panels = payload.get("panels") or []
            if 0 <= panel < len(panels):
                return (panels[panel].get("overlay") or {}).get("_path")
            return None
        return (payload.get("difference") or {}).get("overlay", {}).get("_path")
    return (payload.get("overlay") or {}).get("_path")


def _read_overlay_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@app.get("/chart/{chart_id}/overlay.png")
async def chart_overlay_png(chart_id: str, request: Request, panel: int | None = None):
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    overlay_path = _chart_overlay_path(payload, panel)
    if not overlay_path or not os.path.isfile(overlay_path):
        raise HTTPException(status_code=404, detail="This chart has no rendered overlay.")
    content = await asyncio.to_thread(_read_overlay_bytes, overlay_path)
    return Response(content=content, media_type="image/png", headers={"Cache-Control": "no-store"})


# T59 D13. A deliberate divergence from overlay.png's `no-store` above, and it
# rests entirely on D8: a frame blob is never regenerated in place, so this
# URL's bytes cannot change and a cache that keeps them forever can never serve
# a stale field. `private` because charts are ownership-scoped — every route
# here goes through `_get_owned_chart`, there is no shared cache to warm, and
# no cross-user entry for one researcher's region to leak into.
_FRAME_CACHE_CONTROL = "private, immutable, max-age=31536000"


def _if_none_match(request: Request) -> set[str]:
    header = request.headers.get("if-none-match", "")
    return {tag.strip().removeprefix("W/") for tag in header.split(",") if tag.strip()}


@app.get("/chart/{chart_id}/frames.f32.gz")
async def chart_frames(chart_id: str, request: Request):
    """The float32 frame stack behind a chart's scrubber (T59).

    Serves the stored bytes still gzipped, so the browser inflates them
    natively and no client-side decompressor stands between the checked bytes
    and the canvas. Everything needed to *label* the scrub — the axis, the
    intervals, coverage, QA rates, per-frame statistics — travels in the chart
    payload instead, which is why a 404 here degrades the slider rather than
    breaking the chart (D8: an evicted stack disables the scrubber with the
    axis still drawn, and is never rebuilt in place).
    """
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    frames = payload.get("frames")
    key = frames.get("_key") if isinstance(frames, dict) else None
    if not key:
        raise HTTPException(status_code=404, detail="This chart has no stored frame stack.")

    # Off the event loop for the same reason the overlay read is (T45): a slow
    # volume must not stall every other in-flight stream for the read's
    # duration, and this one hashes what it reads as well.
    blob = await asyncio.to_thread(
        frame_store.read_frames, key, pipeline_version=OPEN_PIPELINE_VERSION
    )
    if blob is None:
        raise HTTPException(
            status_code=404, detail="The frame values for this chart are no longer available."
        )

    return _frame_blob_response(blob, request)


@app.get("/chart/{chart_id}/frames.{statistic}.f32.gz")
async def chart_frame_plane(chart_id: str, statistic: str, request: Request):
    """One additional statistic's frame stack (T59 D6a decision 5).

    A path per statistic rather than a query parameter on the mean's URL. A
    parameter would make one URL serve several bodies, which every cache
    between the browser and this app then has to be told about with `Vary`, and
    would turn `_FRAME_CACHE_CONTROL`'s `immutable` into a claim about a URL
    whose content is no longer fixed. A distinct path is a distinct cache entry
    needing no coordination at all — and it leaves `frames.f32.gz` untouched
    outright rather than untouched-as-long-as-nobody-passes-the-parameter,
    which is what decision 5's "the mean entry keeps its exact shape, URL and
    cost" actually asks for.

    Every failure here is a 404: an unknown statistic, one this chart never
    computed, and one whose entry has been evicted are all "there is no such
    blob", and the frontend's answer to each is the same — offer the statistics
    that are there. The axis is in Postgres either way, so the scrubber stays
    drawn and labeled (D8).
    """
    # Constrained to the statistics the reduction actually produces, so the
    # segment can never become an arbitrary key lookup into the chart's block.
    if statistic not in frame_store.STORABLE_STATISTICS:
        raise HTTPException(status_code=404, detail="Unknown frame statistic.")

    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    frames = payload.get("frames")
    planes = frames.get("planes") if isinstance(frames, dict) else None
    plane = planes.get(statistic) if isinstance(planes, dict) else None
    # `mean` is never a key in there — its blob is the chart's own, at the URL
    # above — so it falls out as a miss with no special case for it.
    key = plane.get("_key") if isinstance(plane, dict) else None
    if not key:
        raise HTTPException(
            status_code=404, detail="This chart has no stored frames for that statistic."
        )

    blob = await asyncio.to_thread(
        frame_store.read_frames, key,
        pipeline_version=OPEN_PIPELINE_VERSION, statistic=statistic,
    )
    if blob is None:
        raise HTTPException(
            status_code=404,
            detail="The frame values for that statistic are no longer available.",
        )
    return _frame_blob_response(blob, request)


def _frame_blob_response(blob, request: Request) -> Response:
    """The stored bytes, still gzipped, under the ETag they were stored with.

    Shared by the mean's route and every plane's, so the two cannot drift into
    serving the same kind of thing under different cache rules.
    """
    etag = f'"{blob.etag}"'
    headers = {"ETag": etag, "Cache-Control": _FRAME_CACHE_CONTROL}
    if etag in _if_none_match(request):
        return Response(status_code=304, headers=headers)
    return Response(
        content=blob.gzipped,
        media_type="application/octet-stream",
        headers={**headers, "Content-Encoding": "gzip"},
    )


@app.get("/chart/{chart_id}/provenance")
async def chart_provenance_endpoint(chart_id: str, request: Request):
    tools = _earthdata_tools(request)
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    source_handles = _chart_source_handles(payload)
    with user_id_context(request.state.current_user.id):
        return await get_lineage(source_handles, tools)


@app.get("/chart/{chart_id}/citations")
async def chart_citations_endpoint(chart_id: str, request: Request):
    tools = _earthdata_tools(request)
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    source_handles = _chart_source_handles(payload)
    with user_id_context(request.state.current_user.id):
        return {"citations": await get_citations(source_handles, tools)}


@app.get("/chart/{chart_id}/methods.md")
async def chart_methods_endpoint(chart_id: str, request: Request):
    tools = _earthdata_tools(request)
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    source_handles = _chart_source_handles(payload)
    with user_id_context(request.state.current_user.id):
        lineage = await get_lineage(source_handles, tools)
        citations = await get_citations(source_handles, tools)

    provenance = payload.get("provenance") or {}
    try:
        markdown = build_methods_markdown(
            artifact_title=payload.get("title") or "Untitled artifact",
            aoi_description=provenance.get("region_name") or "the study area",
            time_window=_methods_time_window(provenance),
            lineage=lineage,
            citations=citations,
            maturity=provenance.get("maturity"),
            maturity_note=provenance.get("maturity_note"),
            # T59 D12: the whole payload, not the provenance sub-dict — the
            # frames disclosure reads two blocks that live at the top level
            # (``frames``/``frames_unavailable`` and ``export.frames``), and
            # the precedence rule between them belongs in one place rather
            # than split across this endpoint.
            chart=payload,
        )
    except MCPToolError:
        raise
    except Exception:
        # Any surprise inside the methods assembly (e.g. a KeyError on a
        # shifted provenance shape) is a contract failure, not a stack trace
        # to leak: classify it through the shared taxonomy handler with a
        # generic message — the raw exception text stays in the logs. Without
        # this the KeyError escaped as a bare ExceptionGroup 500 (QA
        # 2026-07-17 blocker).
        logger.exception("methods_markdown_assembly_failed", extra={"_chart_id": chart_id})
        raise MCPToolError(
            CATEGORY_CONTRACT,
            "The methods document could not be assembled for this chart.",
        )
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="methods-{export_service.safe_export_name(payload, "md")}"'},
    )


@app.get("/chart/{chart_id}/export.nc")
async def export_chart_netcdf(chart_id: str, request: Request):
    # T37: same readiness gate (shared structured 503) as every other
    # MCP-backed endpoint, instead of a bespoke bare-detail 503.
    tools = _earthdata_tools(request)
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    source_handles = _chart_source_handles(payload)
    if not source_handles:
        raise HTTPException(status_code=422, detail="This chart does not include a source handle to export.")

    try:
        with user_id_context(request.state.current_user.id):
            export = await export_converted(source_handles[0], "netcdf", tools)
    except DataDownloadError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # T37: materialize the first chunk before committing to a 200 — an
    # evicted/vanished converted file raises here instead of streaming a
    # truncated "successful" download.
    try:
        stream = await materialize_first_chunk(iter_file_chunks(export["storage_uri"]))
    except OSError:
        logger.exception("export_netcdf_file_unreadable", extra={"_chart_id": chart_id})
        raise HTTPException(status_code=422, detail="The converted export is no longer available. Please retry the export.")

    return StreamingResponse(
        stream,
        media_type="application/x-netcdf",
        headers={
            "Content-Disposition": f'attachment; filename="{export_service.safe_export_name(payload, "nc")}"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        return await artifact_store.get_page(artifact_id, request.state.current_user.id, offset, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")


@app.get("/artifacts/{artifact_id}/csv")
async def export_artifact_csv(artifact_id: str, request: Request):
    try:
        artifact = await artifact_store.reference(artifact_id)
        await artifact_store.get_page(artifact_id, request.state.current_user.id, 0, 1)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    filename = _safe_artifact_filename(artifact.title or artifact.id)
    return StreamingResponse(
        artifact_store.iter_csv_chunks(artifact_id, request.state.current_user.id),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.csv"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/jobs")
async def get_jobs(request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        jobs = await list_jobs(tools)
    return {"jobs": jobs}


@app.post("/jobs/{job_handle}/cancel")
async def cancel_job_endpoint(job_handle: JobHandle, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await cancel_job(job_handle, tools)


@app.post("/discovery/search")
async def discovery_search_endpoint(req: DiscoverySearchRequest, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await search_datasets(req.query, req.filters, tools)


@app.get("/discovery/dataset/{dataset_handle}")
async def discovery_describe_endpoint(dataset_handle: DatasetHandle, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await describe_dataset(dataset_handle, tools)


@app.post("/discovery/dataset/{dataset_handle}/preview")
async def discovery_preview_endpoint(dataset_handle: DatasetHandle, req: DiscoveryPreviewRequest, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await preview_dataset(dataset_handle, req.location, req.time_range, req.layer, tools)


@app.post("/discovery/dataset/{dataset_handle}/coverage")
async def discovery_coverage_endpoint(dataset_handle: DatasetHandle, req: DiscoveryCoverageRequest, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await check_coverage(dataset_handle, req.location, req.time_range, tools)


@app.post("/discovery/dataset/{dataset_handle}/granules")
async def discovery_granules_endpoint(dataset_handle: DatasetHandle, req: DiscoveryGranulesRequest, request: Request):
    tools = _earthdata_tools(request)
    with user_id_context(request.state.current_user.id):
        return await inspect_granules(dataset_handle, req.location, req.time_range, req.limit, tools)


def _safe_artifact_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.lower())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:80] or "artifact"


async def _get_owned_chart(chart_id: str, user_id: str):
    payload = await chart_service.get_chart(chart_id)
    if not payload or payload.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Chart not found")
    return payload


def _chart_source_handles(payload: dict) -> list[str]:
    return (payload.get("metadata") or {}).get("source_handles") or (payload.get("provenance") or {}).get("source_handles", [])


def _methods_time_window(provenance: dict) -> str:
    start, end = provenance.get("start_date"), provenance.get("end_date")
    if start and end:
        return f"{start}/{end}"
    return "the analyzed period"


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    user = request.state.current_user
    active_agent = getattr(app.state, "agent", None) or agent
    if active_agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    ground_agent = getattr(app.state, "ground_agent", None)
    satellite_agent = getattr(app.state, "satellite_agent", None)
    thread_id = await _resolve_thread(req, user.id)
    request_id = str(uuid.uuid4())
    await _save_session_metadata(thread_id, req.message, user.id, request_id)
    return StreamingResponse(
        chat_stream_service.stream_chat_events(
            active_agent, ground_agent, satellite_agent, req.message, thread_id, user.id, request_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _resolve_thread(req: ChatRequest, user_id: str) -> str:
    thread_id = req.thread_id or str(uuid.uuid4())
    if req.thread_id:
        metadata = await get_session_metadata(thread_id)
        if metadata is not None and metadata["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
    return thread_id


async def _save_session_metadata(thread_id: str, message: str, user_id: str, request_id: str) -> None:
    try:
        await save_session_metadata_once(thread_id, message, user_id)
    except Exception:
        logger.exception("session_metadata_save_failed", extra={"_request_id": request_id, "_thread_id": thread_id})

# T37: session endpoint catch-alls answer with a fixed generic detail — the
# real exception goes to the logs with request context, never to the client.
_INTERNAL_ERROR_DETAIL = "Internal server error"


@app.get("/sessions")
async def get_sessions(request: Request):
    try:
        return {"sessions": await session_repository.list_sessions(request.state.current_user.id)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("sessions_list_failed", extra={"_user_id": request.state.current_user.id})
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL)

@app.get("/session/{thread_id}/history")
async def get_history(thread_id: ThreadId, request: Request):
    try:
        user_id = request.state.current_user.id
        if not await session_belongs_to_user(thread_id, user_id):
            raise HTTPException(status_code=404, detail="Session not found")
        active_agent = getattr(app.state, "agent", None) or agent
        if active_agent is None:
            raise HTTPException(status_code=503, detail="Agent is not ready")
        return {"messages": await history_service.build_history(active_agent, thread_id, user_id)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("session_history_failed", extra={"_thread_id": thread_id})
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL)


@app.delete("/session/{thread_id}")
async def remove_session(thread_id: ThreadId, request: Request):
    try:
        deleted = await session_repository.delete_session(thread_id, request.state.current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": thread_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("session_delete_failed", extra={"_thread_id": thread_id})
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL)
