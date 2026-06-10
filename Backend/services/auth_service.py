from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any
import uuid

import bcrypt
import jwt
from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param

from config.settings import get_settings
from models.user import User
from repositories.revoked_token_repository import is_token_revoked
from repositories.user_repository import get_user_by_id


def _password_bytes(password: str) -> bytes:
    # bcrypt 5 rejects inputs over 72 bytes. Prehashing keeps password handling
    # deterministic for long or non-ASCII passwords without storing plaintext.
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(_password_bytes(password), password_hash.encode("utf-8"))


def create_access_token(user: User) -> tuple[str, int]:
    settings = get_settings()
    expires_delta = timedelta(minutes=settings.jwt_expiration_minutes)
    expires_at = datetime.now(timezone.utc) + expires_delta
    payload: dict[str, Any] = {
        "sub": user.id,
        "username": user.username,
        "exp": expires_at,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, int(expires_delta.total_seconds())


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def authenticate_request(request: Request) -> User:
    authorization = request.headers.get("Authorization")
    scheme, token = get_authorization_scheme_param(authorization)
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized()

    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "username", "exp", "jti"]},
        )
    except jwt.PyJWTError:
        raise _unauthorized()

    user_id = payload.get("sub")
    jti = payload.get("jti")
    if not isinstance(user_id, str) or not user_id or not isinstance(jti, str) or not jti:
        raise _unauthorized()
    if await is_token_revoked(jti):
        raise _unauthorized()

    user = await get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    request.state.jwt_payload = payload
    return user
