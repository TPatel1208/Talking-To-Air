import json
import logging
import os
import time
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
from pydantic import BaseModel, Field
from starlette.routing import Match

from agents.earthdata_agent import LazySatelliteAgent, build_earthdata_agent
from agents.ground_sensor_agent import build_ground_agent
from agents.supervisor_agent import build_agent
from config.settings import get_settings
from config.starter_prompts import STARTER_PROMPTS
from earthdata_mcp.connection import STATE_CONNECTING, STATE_READY, EarthdataMCPConnectionManager
from earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_NOT_FOUND,
    CATEGORY_NO_DATA,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_TOO_LARGE,
    CATEGORY_USER_INPUT,
    MCPToolError,
)
from repositories.session_metadata_repository import (
    ensure_session_metadata_table,
    get_session_metadata,
    save_session_metadata_once,
    session_belongs_to_user,
)
from repositories.revoked_token_repository import ensure_revoked_token_table, revoke_token
from repositories.session_repository import SessionRepository
from repositories.user_repository import create_user, ensure_user_table, get_user_by_username
from services.auth_service import authenticate_request, create_access_token, hash_password, verify_password
from services.artifact_store import artifact_store
from services.chat_stream_service import ChatStreamService
from services.chart_service import ChartService
from services.export_service import ExportService
from services.history_service import HistoryService
from services.data_download_service import DataDownloadError, export_converted, iter_file_chunks
from services.discovery_service import (
    check_coverage,
    describe_dataset,
    inspect_granules,
    preview_dataset,
    search_datasets,
)
from services.jobs_service import cancel_job, list_jobs
from services.methods_export_service import build_methods_markdown
from services.provenance_service import get_citations, get_lineage
from utils.db import active_pool_connections, check_db_pool, close_db_pool, init_db_pool, validate_config
from utils.logging import configure_logging
from utils.metrics import (
    observe_http_request,
    prometheus_content_type,
    render_prometheus_metrics,
    set_db_pool_connections_active,
)
from utils.streaming import current_user_id, user_id_context

agent = None
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

chart_service = ChartService()
export_service = ExportService(settings.csv_export_max_granules)
history_service = HistoryService(chart_service)
session_repository = SessionRepository()


async def _on_earthdata_mcp_ready(tools: dict) -> None:
    """earthdata_mcp_manager's on_ready hook (T17): populates the legacy
    app.state.earthdata_mcp_tools mirror (still read directly by the
    unmigrated chart export.csv/.png/.nc endpoints) and rebuilds the real
    satellite agent into whatever LazySatelliteAgent the current lifespan
    cycle assigned to app.state.satellite_agent — see
    agents/earthdata_agent.py for why a mutable placeholder, not a
    reassigned reference, is what makes this visible to the supervisor's
    already-built ask_earthdata_agent tool closure."""
    app.state.earthdata_mcp_tools = tools
    app.state.satellite_agent.set_real(build_earthdata_agent(mcp_tools=tools))
    logger.info("earthdata_mcp_satellite_agent_ready", extra={"_event": "earthdata_mcp_satellite_agent_ready"})


earthdata_mcp_manager = EarthdataMCPConnectionManager(settings, current_user_id, on_ready=_on_earthdata_mcp_ready)
chat_stream_service = ChatStreamService(chart_service, settings.long_request_seconds, earthdata_mcp_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    validate_config()
    await init_db_pool()
    await ensure_user_table()
    await ensure_revoked_token_table()
    await ensure_session_metadata_table()

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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

PUBLIC_ENDPOINTS = {
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/capabilities/starters"),
    ("POST", "/auth/login"),
    ("POST", "/auth/register"),
}
ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]
JobHandle = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]
DatasetHandle = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]


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
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - started
        set_db_pool_connections_active(active_pool_connections())
        observe_http_request(request.method, path, status_code, duration)


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


class AuthRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=1, max_length=1024)


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


@app.post("/auth/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register(req: AuthRequest):
    password_hash = hash_password(req.password)
    try:
        user = await create_user(req.username, password_hash)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    return UserResponse(id=user.id, username=user.username, is_active=user.is_active)


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: AuthRequest):
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
    return Response(content=render_prometheus_metrics(), media_type=prometheus_content_type())


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
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    try:
        if not payload.get("export"):
            raise ValueError("This chart does not include full-resolution export metadata.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StreamingResponse(
        export_service.iter_chart_csv_chunks_async(payload, request.app.state.earthdata_mcp_tools),
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
        content = await export_service.build_chart_png_async(payload, request.app.state.earthdata_mcp_tools)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(
        content=content,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{export_service.safe_export_name(payload, "png")}"'},
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
    markdown = build_methods_markdown(
        artifact_title=payload.get("title") or "Untitled artifact",
        aoi_description=provenance.get("region_name") or "the study area",
        time_window=_methods_time_window(provenance),
        lineage=lineage,
        citations=citations,
    )
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="methods-{export_service.safe_export_name(payload, "md")}"'},
    )


@app.get("/chart/{chart_id}/export.nc")
async def export_chart_netcdf(chart_id: str, request: Request):
    tools = request.app.state.earthdata_mcp_tools
    if not tools:
        raise HTTPException(status_code=503, detail="Earthdata MCP is not ready")
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    source_handles = _chart_source_handles(payload)
    if not source_handles:
        raise HTTPException(status_code=422, detail="This chart does not include a source handle to export.")

    try:
        with user_id_context(request.state.current_user.id):
            export = await export_converted(source_handles[0], "netcdf", tools)
    except DataDownloadError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StreamingResponse(
        iter_file_chunks(export["storage_uri"]),
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
        return artifact_store.get_page(artifact_id, request.state.current_user.id, offset, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")


@app.get("/artifacts/{artifact_id}/csv")
async def export_artifact_csv(artifact_id: str, request: Request):
    try:
        artifact = artifact_store.reference(artifact_id)
        artifact_store.get_page(artifact_id, request.state.current_user.id, 0, 1)
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

@app.get("/sessions")
async def get_sessions(request: Request):
    try:
        return {"sessions": await session_repository.list_sessions(request.state.current_user.id)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{thread_id}")
async def remove_session(thread_id: ThreadId, request: Request):
    try:
        deleted = await session_repository.delete_session(thread_id, request.state.current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
