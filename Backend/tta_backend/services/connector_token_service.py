"""T30: paste-time validation for a connector token.

The pasted EDL user token is a JWT, but we are not its audience -- decoding
verifies shape and a future `exp` only, never a signature, and never round-
trips to EDL. Liveness (does EDL actually still honor it) is proven at first
real use in PRD-022/T31, not here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import jwt


class TokenValidationError(ValueError):
    """Raised for a paste that is not a usable, not-yet-expired JWT."""


def decode_token_expiry(token: str) -> datetime:
    token = (token or "").strip()
    if not token:
        raise TokenValidationError("Paste a token before saving.")

    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False, "require": ["exp"]},
        )
    except jwt.PyJWTError:
        raise TokenValidationError("That doesn't look like a valid Earthdata Login token.") from None

    try:
        expires_at = datetime.fromtimestamp(float(payload["exp"]), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        raise TokenValidationError("That doesn't look like a valid Earthdata Login token.") from None

    if expires_at <= datetime.now(timezone.utc):
        raise TokenValidationError("This token has already expired. Generate a new one at Earthdata Login.")

    return expires_at
