import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Optional

import psycopg
from fastapi import FastAPI, HTTPException, Path, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.routing import Match

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.supervisor_agent import build_agent
from config.settings import get_settings
from preprocessing.data_loader import DataLoader
from repositories.chart_repository import ensure_chart_table
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
from services.chat_stream_service import ChatStreamService
from services.chart_service import ChartService
from services.export_service import ExportService
from services.history_service import HistoryService
from tools.satellite_tools.harmony_api import set_data_loader
from utils.db import close_db_pool, init_db_pool, validate_config
from utils.logging import configure_logging
from utils.metrics import snapshot_metrics

agent = None
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

chart_service = ChartService()
export_service = ExportService(settings.csv_export_max_granules)
history_service = HistoryService(chart_service)
chat_stream_service = ChatStreamService(chart_service, settings.long_request_seconds)
session_repository = SessionRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    validate_config()
    await init_db_pool()
    await ensure_user_table()
    await ensure_revoked_token_table()
    await ensure_chart_table()
    await ensure_session_metadata_table()

    logger.info("startup_begin", extra={"_model": settings.llm_model})
    data_loader = DataLoader()
    app.state.data_loader = data_loader
    set_data_loader(data_loader)
    agent = await build_agent(settings.llm_model)
    app.state.agent = agent
    logger.info("startup_complete")
    try:
        yield
    finally:
        agent = None
        app.state.agent = None
        app.state.data_loader = None
        set_data_loader(None)
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
    ("POST", "/auth/login"),
    ("POST", "/auth/register"),
}
ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]


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
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {"counters": snapshot_metrics()}


@app.get("/chart/{chart_id}/export.csv")
async def export_chart_csv(chart_id: str, request: Request):
    payload = await _get_owned_chart(chart_id, request.state.current_user.id)
    try:
        if not payload.get("export"):
            raise ValueError("This chart does not include full-resolution export metadata.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StreamingResponse(
        export_service.iter_chart_csv_chunks(payload),
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
        content = await asyncio.to_thread(export_service.build_chart_png, payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(
        content=content,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{export_service.safe_export_name(payload, "png")}"'},
    )


async def _get_owned_chart(chart_id: str, user_id: str):
    payload = await chart_service.get_chart(chart_id)
    if not payload or payload.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Chart not found")
    return payload


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    user = request.state.current_user
    active_agent = getattr(app.state, "agent", None) or agent
    if active_agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    thread_id = await _resolve_thread(req, user.id)
    request_id = str(uuid.uuid4())
    await _save_session_metadata(thread_id, req.message, user.id, request_id)
    return StreamingResponse(
        chat_stream_service.stream_chat_events(active_agent, req.message, thread_id, user.id, request_id),
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
