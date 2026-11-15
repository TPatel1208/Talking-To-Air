"""The EPA AQS HTTP client: one paced request per credential, and a shared response cache.

Every request carries the credential it is given; nothing here looks one up.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

AQS_BASE_URL = "https://aqs.epa.gov/data/api"

# An async (endpoint, params) -> AQS response body.
Fetch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

# Response cache keyed by (endpoint, sorted params), credentials excluded.
# Values are (expires_at, payload).
_response_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}

# Settled measurements never change; unsettled ones lapse so data EPA
# publishes later is fetched.
_SETTLED_TTL_SECONDS = 7 * 24 * 60 * 60
_UNSETTLED_TTL_SECONDS = 15 * 60
_MAX_CACHE_ENTRIES = 256

# EPA AQS allows 10 requests per minute per credential; nine leaves a margin.
_AQS_MAX_REQUESTS_PER_WINDOW = 9
_AQS_WINDOW_SECONDS = 60.0
_AQS_MIN_SPACING_SECONDS = 1.0

# A window ending within this many days of today is likely not yet published
# (AQS releases data with a lag of about two months).
RECENT_DATA_WINDOW_DAYS = 90

_TIMEOUT_SECONDS = 30


def _now() -> float:
    """Monotonic clock for the cache and the rate limiter; tests replace it."""
    return time.monotonic()


@dataclass(frozen=True)
class AqsCredential:
    email: str
    key: str = field(repr=False)


class AqsCredentialsRejected(RuntimeError):
    """EPA answered that the email/key pair is not valid."""

    category = "token_invalid"


class AqsUnavailable(RuntimeError):
    """EPA could not be reached or gave no verdict."""

    category = "provider_unavailable"


class AqsRequestFailed(RuntimeError):
    """EPA answered with an error other than a verdict on the credentials."""

    def __init__(self, message: str, *, category: str = "provider_unavailable") -> None:
        super().__init__(message)
        self.category = category


class AqsNoDataError(RuntimeError):
    """EPA answered, and has no rows matching the query."""

    category = "no_data"


def credential_from_secret(secret: str) -> AqsCredential:
    """The pair stored by the epa_aqs connector as ``{"email": ..., "key": ...}``.

    Raises AqsCredentialsRejected when the secret is not such a pair.
    """
    try:
        parsed = json.loads(secret)
        email, key = parsed["email"], parsed["key"]
    except (ValueError, TypeError, KeyError):
        raise AqsCredentialsRejected("the EPA AQS credential is malformed") from None
    if not isinstance(email, str) or not isinstance(key, str) or not email or not key:
        raise AqsCredentialsRejected("the EPA AQS credential is malformed")
    return AqsCredential(email=email, key=key)


class _SlidingWindowLimiter:
    """Paces requests to at most ``max_requests`` per ``window_seconds``,
    delaying a caller only once the window is full."""

    def __init__(self, max_requests: int, window_seconds: float, min_spacing_seconds: float):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._min_spacing_seconds = min_spacing_seconds
        # Held for arithmetic only; callers wait after release.
        self._lock = threading.Lock()
        self._reserved: deque[float] = deque()

    def reserve(self) -> float:
        """Claim the next send slot and return how long to wait for it."""
        with self._lock:
            now = _now()
            horizon = now - self._window_seconds
            while self._reserved and self._reserved[0] <= horizon:
                self._reserved.popleft()

            send_at = now
            if self._reserved:
                send_at = max(send_at, self._reserved[-1] + self._min_spacing_seconds)
            if len(self._reserved) >= self._max_requests:
                send_at = max(send_at, self._reserved[-self._max_requests] + self._window_seconds)

            self._reserved.append(send_at)
            return max(0.0, send_at - now)

    def idle(self) -> bool:
        """Whether every reserved slot has aged out of the window."""
        with self._lock:
            return not self._reserved or self._reserved[-1] <= _now() - self._window_seconds


# One limiter per credential, since EPA's cap is per credential. Keyed by a
# digest of the key so the map holds no secret.
_rate_limiters: dict[str, _SlidingWindowLimiter] = {}
_rate_limiter_registry_lock = threading.Lock()
_MAX_RATE_LIMITERS = 1024


def credential_digest(key: str) -> str:
    """The rate-limiter map's identity for a credential key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _prune_limiters() -> None:
    """Make room for one limiter: drop limiters idle for a whole window, which
    hold no pacing state, then the least recently used."""
    idle = [d for d, limiter in _rate_limiters.items() if limiter.idle()]
    for digest in idle:
        del _rate_limiters[digest]
    while len(_rate_limiters) >= _MAX_RATE_LIMITERS:
        del _rate_limiters[next(iter(_rate_limiters))]


def _limiter_for(key: str) -> _SlidingWindowLimiter:
    """The limiter of the credential with this key; most recently used last."""
    digest = credential_digest(key)
    with _rate_limiter_registry_lock:
        limiter = _rate_limiters.pop(digest, None)
        if limiter is None:
            if len(_rate_limiters) >= _MAX_RATE_LIMITERS:
                _prune_limiters()
            limiter = _SlidingWindowLimiter(
                _AQS_MAX_REQUESTS_PER_WINDOW,
                _AQS_WINDOW_SECONDS,
                _AQS_MIN_SPACING_SECONDS,
            )
        _rate_limiters[digest] = limiter
        return limiter


def _cache_ttl_seconds(params: dict[str, Any]) -> float:
    """How long a response may be reused: long once EPA has settled the
    window's data, short while it may still be published."""
    edate = str(params.get("edate") or "").strip()
    if not edate:
        return _SETTLED_TTL_SECONDS
    try:
        edate_obj = date(int(edate[0:4]), int(edate[4:6]), int(edate[6:8]))
    except (ValueError, IndexError):
        return _UNSETTLED_TTL_SECONDS
    if edate_obj < date.today() - timedelta(days=RECENT_DATA_WINDOW_DAYS):
        return _SETTLED_TTL_SECONDS
    return _UNSETTLED_TTL_SECONDS


def _cache_key(endpoint: str, params: dict[str, Any]) -> tuple[Any, ...]:
    """AQS data is public, so the cache identity leaves the credential out."""
    return (endpoint, tuple(sorted((k, str(v)) for k, v in params.items())))


def cached_response(endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """A live cached response for the request, or None."""
    cache_key = _cache_key(endpoint, params)
    cached = _response_cache.get(cache_key)
    if cached is None:
        return None
    expires_at, payload = cached
    if _now() < expires_at:
        return payload
    del _response_cache[cache_key]
    return None


def remember_response(endpoint: str, params: dict[str, Any], data: dict[str, Any]) -> None:
    """Cache a response, dropping expired entries and then the oldest when full."""
    now = _now()
    if len(_response_cache) >= _MAX_CACHE_ENTRIES:
        for key in [k for k, (expires_at, _) in _response_cache.items() if expires_at <= now]:
            del _response_cache[key]
        while len(_response_cache) >= _MAX_CACHE_ENTRIES:
            del _response_cache[next(iter(_response_cache))]
    _response_cache[_cache_key(endpoint, params)] = (now + _cache_ttl_seconds(params), data)


def reset_request_state() -> None:
    """Drop cached responses and rate-limit history."""
    _response_cache.clear()
    with _rate_limiter_registry_lock:
        _rate_limiters.clear()


def _scrub(text: str, credential: AqsCredential) -> str:
    """EPA echoes the request URL, email and key included, in its Header."""
    for secret in (credential.key, credential.email):
        if secret:
            text = text.replace(secret, "REDACTED")
    return text


def _header_error_text(header: dict[str, Any]) -> str:
    detail = header.get("error") or header.get("status") or ""
    if isinstance(detail, (list, tuple)):
        detail = "; ".join(str(item) for item in detail)
    return str(detail)


def _is_credential_rejection(detail: str) -> bool:
    # EPA's wording for a bad pair: "Email and/or key are invalid."
    return "email and/or key" in detail.lower()


def _http_category(status_code: int) -> str:
    return "provider_unavailable" if status_code == 429 or status_code >= 500 else "contract"


async def request(
    endpoint: str,
    params: dict[str, Any],
    credential: AqsCredential,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """One uncached GET with ``credential``, paced by that credential's limiter.

    Raises AqsCredentialsRejected when EPA refuses the pair, AqsRequestFailed
    (a RuntimeError) for other EPA errors; transport errors propagate.
    """
    full_params = {**params, "email": credential.email, "key": credential.key}
    wait_seconds = _limiter_for(credential.key).reserve()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=transport) as client:
        resp = await client.get(f"{AQS_BASE_URL}/{endpoint}", params=full_params)
    if resp.status_code >= 400:
        try:
            body = resp.json()
            aqs_header = body.get("Header", [{}])
            aqs_msg = _header_error_text(aqs_header[0])
        except (ValueError, KeyError, IndexError, AttributeError):
            resp.raise_for_status()
            raise
        if _is_credential_rejection(aqs_msg):
            raise AqsCredentialsRejected(f"AQS HTTP {resp.status_code} on {endpoint}: {aqs_msg}")
        raise AqsRequestFailed(
            f"AQS HTTP {resp.status_code} on {endpoint}: {_scrub(aqs_msg or resp.text[:300], credential)}",
            category=_http_category(resp.status_code),
        )
    data: dict[str, Any] = resp.json()
    header = data.get("Header", [{}])
    status = header[0].get("status", "").lower()
    rows = len(data.get("Data", data.get("Body", [])))
    # The query is in httpx's request line, which logging redacts.
    logger.debug(
        "EPA response endpoint=%s http=%s header_status=%r rows=%s",
        endpoint,
        resp.status_code,
        header[0].get("status", ""),
        rows,
    )
    # "No data matched your selection" is a valid empty result.
    if status not in ("success", "no data matched your selection", ""):
        detail = _header_error_text(header[0]) or "unexpected response"
        logged_header = {k: v for k, v in header[0].items() if k != "url"}
        logger.warning("aqs_error endpoint=%s header=%r", endpoint, logged_header)
        if _is_credential_rejection(detail):
            raise AqsCredentialsRejected(f"EPA AQS request failed: {detail}")
        raise AqsRequestFailed(f"EPA AQS request failed: {_scrub(detail, credential)}", category="contract")
    return data


async def validate_credentials(
    credential: AqsCredential, *, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    """Prove EPA accepts the pair with one live call, never from the cache.

    Raises AqsCredentialsRejected for a bad pair, AqsUnavailable when EPA
    gives no verdict.
    """
    try:
        await request("list/states", {}, credential, transport=transport)
    except AqsCredentialsRejected:
        raise
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise AqsUnavailable("EPA AQS did not answer the credential check") from exc


def fetcher(credential: AqsCredential, *, transport: httpx.AsyncBaseTransport | None = None) -> Fetch:
    """A cached, paced fetch on ``credential`` alone. A transport failure or an
    unreadable answer raises AqsUnavailable."""

    async def fetch(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cached = cached_response(endpoint, params)
        if cached is not None:
            return cached
        try:
            data = await request(endpoint, params, credential, transport=transport)
        except (httpx.HTTPError, ValueError) as exc:
            raise AqsUnavailable("EPA AQS could not be reached") from exc
        remember_response(endpoint, params, data)
        return data

    return fetch
