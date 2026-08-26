import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import jwt
import threading
import time
from typing import Any
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None

# Both unknown-kid rejections use this, throttled or not -- see _signing_key.
UNKNOWN_KID_MESSAGE = "Kid Lookup Failed"


class AuthenticationError(Exception):
    """The token is bad. Maps to 401 -- signing in again can fix it."""


class IdentityProviderUnavailable(Exception):
    """
    We cannot verify anyone right now. Maps to 503 -- signing in cannot fix it.
    """

@dataclass
class JwksCacheEntry:
    # The *parsed* key set, not the raw dict: parsing is where a useless
    # response (no keys, wrong shape) announces itself, and it must do so
    # before anything is allowed to replace a working entry. Parsing once
    # per fetch instead of once per request is a bonus, not the reason.
    keyset: jwt.PyJWKSet
    expiration_time: float

class SupabaseJwtVerifier:
    def __init__(self,
                  fetch_jwks: Callable[[], dict[str, Any]],
                  issuer: str,
                  cache_ttl: int = 900,
                  retry_interval: int = 60,
                  kid_refresh_cooldown: int = 60,
                  now: Callable[[], float] = time.monotonic,
                ):
        self.fetch_jwks = fetch_jwks
        self.issuer = issuer
        self._cache_entry: JwksCacheEntry | None = None
        self.cache_ttl = cache_ttl
        self.retry_interval = retry_interval
        self.kid_refresh_cooldown = kid_refresh_cooldown
        self._last_kid_refresh: float | None = None
        # The warm-cache branch defers its next attempt by parking a deadline on
        # the cache entry. A cold cache has no entry to park one on, so it needs
        # its own: without it every request during a cold outage re-attempts the
        # blocking fetch. See _serve_stale_or_fail.
        self._cold_retry_deadline: float | None = None
        self._throttled_kid_refreshes = 0
        self._warm_task: asyncio.Task | None = None
        # warm() fetches on a worker thread while requests run on the event
        # loop, so every read-modify-write of the state above is guarded.
        # The fetch itself deliberately happens outside the lock: holding
        # it across a 30s blocking HTTP call would stall the loop far worse
        # than the duplicate fetch it would prevent.
        self._lock = threading.Lock()
        self.now = now

    def verify(self, token: str) -> AuthenticatedUser:

        try:
            kid = jwt.get_unverified_header(token).get('kid')
            if not kid:
                raise AuthenticationError("Kid missing from token header")
            signing_key = self._signing_key(kid)
            decoded_payload = jwt.decode(
                token,
                signing_key,
                algorithms=['ES256'],
                audience='authenticated',
                issuer=self.issuer,
                options={"require": ["sub", "exp", "aud", "iss"]},
            )
        except jwt.InvalidIssuerError as e:
            raise AuthenticationError("Invalid issuer.") from e
        except jwt.InvalidAudienceError as e:
            raise AuthenticationError("Invalid audience.") from e
        except jwt.InvalidAlgorithmError as e:
            raise AuthenticationError("Invalid algorithm.") from e
        except jwt.ExpiredSignatureError as e:
            raise AuthenticationError("token is expired") from e
        except jwt.InvalidSignatureError as e:
            raise AuthenticationError("Signature verification failed! The public key does not match.") from e
        except jwt.MissingRequiredClaimError as e:
            # str(e) names the claim ('Token is missing the "sub" claim'), which
            # the generic branch below would flatten into "Invalid token format."
            raise AuthenticationError(str(e)) from e
        except jwt.InvalidTokenError as e :
            raise AuthenticationError("Invalid token format.") from e
        except jwt.PyJWTError as e:
            raise AuthenticationError("An error occurred while decoding the token.") from e
        # Not the main defence, and deliberately kept anyway: options={"require"}
        # above rejects an absent sub, and PyJWT rejects a non-string one, so
        # the only case left here is the empty string -- which PyJWT accepts and
        # which would otherwise become AuthenticatedUser(id="").
        user_id = decoded_payload.get("sub")
        if not user_id:
            raise AuthenticationError("Missing 'sub' claim in token payload.")
        return AuthenticatedUser(
            id=user_id,
            email=decoded_payload.get("email"),
        )

    def start_warm(self) -> None:
        """Kick the warm off in the background. Deliberately not a coroutine.

        Boot calls this and moves on -- the T17 doctrine: the backend serves
        while a remote dependency is still coming up. Awaiting here would let a
        Supabase blip during a deploy stop the container starting at all.
        """
        if self._warm_task is not None:
            return
        self._warm_task = asyncio.create_task(self.warm())
        self._warm_task.add_done_callback(self._log_if_warm_died)

    async def stop_warm(self) -> None:
        """Cancel the warm task. Safe to call when it already finished."""
        task, self._warm_task = self._warm_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _log_if_warm_died(self, task: asyncio.Task) -> None:
        # warm() swallows its own failures, so reaching here means something
        # escaped it -- a bug, not an outage. Without this the exception is
        # retrievable only from a task nobody ever inspects.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "jwks_warm_task_died",
                extra={"_event": "jwks_warm_task_died"},
                exc_info=exc,
            )

    async def warm(
        self,
        *,
        max_attempts: int = 5,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> bool:
        """Populate the key cache at boot so no request pays for the fetch.

        Returns whether the cache ended up warm. **Never raises** -- a boot that
        dies because the identity provider was briefly unreachable is worse than
        a first request that is slow, and the request path already handles a
        cold cache (see IdentityProviderUnavailable).

        The fetch is blocking urllib, so it runs in a thread: awaiting it
        directly would freeze the event loop for the whole timeout and make
        this "background" warm anything but.
        """
        backoff = initial_backoff_seconds
        for attempt in range(1, max_attempts + 1):
            try:
                # force_refresh so the cold-start cooldown in _keyset cannot
                # swallow attempts 2..N of this loop: the capped backoff above is
                # already this path's rate limit, and it is a single background
                # task rather than one call per inbound request.
                await asyncio.to_thread(self._keyset, True)
            except Exception:
                logger.warning(
                    "jwks_warm_attempt_failed",
                    extra={
                        "_event": "jwks_warm_attempt_failed",
                        "_attempt": attempt,
                        "_max_attempts": max_attempts,
                    },
                    exc_info=True,
                )
                if attempt == max_attempts:
                    return False
                await sleep(backoff)
                backoff = min(backoff * 2, max_backoff_seconds)
                continue
            logger.info(
                "jwks_warmed",
                extra={"_event": "jwks_warmed", "_attempts": attempt},
            )
            return True
        return False

    def _signing_key(self, kid: str) -> jwt.PyJWK:
        """Find the key this token claims to be signed by.

        A miss is not necessarily an attack -- it is what a key rotation looks
        like from here -- so look again before giving up.
        """
        try:
            return self._keyset()[kid]
        except KeyError:
            pass
        if not self._claim_kid_refresh():
            # Word for word what the refetched-and-still-missing path says
            # below. A different string here would tell an unauthenticated
            # caller whether the cooldown window was open, letting them time
            # bogus kids to land exactly one outbound fetch per window.
            raise AuthenticationError(UNKNOWN_KID_MESSAGE)
        try:
            return self._keyset(force_refresh=True)[kid]
        except KeyError as e:
            raise AuthenticationError(UNKNOWN_KID_MESSAGE) from e

    def _claim_kid_refresh(self) -> bool:
        """Reserve the right to refetch for an unknown kid, or refuse.

        `kid` is read from the token header -- unsigned and unverified -- and
        this runs before any rate limiting, so an unauthenticated caller could
        otherwise turn every invented kid into an outbound HTTPS request.
        Measured at 6 fetches for 5 bogus kids before this existed.

        Claiming the window and counting the refusals are one atomic step, so
        a concurrent caller cannot slip a second fetch through the same window.

        The window is claimed *before* the refetch is attempted, so a refetch
        that fails still burns it. That is deliberate -- the cooldown has to
        bound attempts, not successes, or an unreachable provider would let
        every bogus kid through to a fetch -- but it means a rotation that
        coincides with a network blip is not picked up until the next window.
        Bounded by kid_refresh_cooldown, and the TTL refresh recovers it
        regardless.
        """
        with self._lock:
            current_time = self.now()
            last = self._last_kid_refresh
            if last is not None and current_time - last < self.kid_refresh_cooldown:
                self._throttled_kid_refreshes += 1
                return False
            throttled, self._throttled_kid_refreshes = self._throttled_kid_refreshes, 0
            self._last_kid_refresh = current_time
        if throttled:
            # Reported when the window closes rather than per refusal: this
            # path is reachable in a loop by anyone, so logging each one would
            # hand an attacker control of our log volume. One line per
            # cooldown, carrying the count, says the same thing.
            logger.warning(
                "jwks_kid_refresh_throttled",
                extra={
                    "_event": "jwks_kid_refresh_throttled",
                    "_throttled_requests": throttled,
                    "_cooldown_seconds": self.kid_refresh_cooldown,
                },
            )
        return True

    def _keyset(self, force_refresh: bool = False) -> jwt.PyJWKSet:
        with self._lock:
            entry = self._cache_entry
            if (
                not force_refresh
                and entry is not None
                and entry.expiration_time > self.now()
            ):
                return entry.keyset
            if (
                not force_refresh
                and entry is None
                and self._cold_retry_deadline is not None
                and self._cold_retry_deadline > self.now()
            ):
                # Nothing has ever been cached and the last attempt just failed.
                # Refuse without touching the network: the request path runs one
                # of these per inbound request, on a worker thread, so fetching
                # here would put the whole default executor behind a queue of
                # blocking calls that are all going to fail anyway -- and hand an
                # unauthenticated caller one outbound HTTPS request each. warm()
                # passes force_refresh so its own capped backoff still governs.
                raise IdentityProviderUnavailable(
                    "identity provider unreachable and no usable signing keys are cached"
                )

        try:
            # Parsed here, inside the guard: a 200 response carrying no usable
            # keys raises PyJWKSetError and is then treated exactly like an
            # unreachable provider. Caching the raw dict instead would let a
            # useless response evict a working key set and 401 every user for a
            # full cache_ttl -- outlasting the provider's own recovery.
            keyset = jwt.PyJWKSet.from_dict(self.fetch_jwks())
        except Exception as e:
            return self._serve_stale_or_fail(e)

        with self._lock:
            # Re-fetched on a schedule rather than per request: these are
            # public keys that rotate approximately never, so the TTL only
            # bounds how long a key Supabase has *withdrawn* stays trusted.
            self._cache_entry = JwksCacheEntry(
                keyset=keyset,
                expiration_time=self.now() + self.cache_ttl,
            )
            self._cold_retry_deadline = None
        return keyset

    def _serve_stale_or_fail(self, exc: Exception) -> jwt.PyJWKSet:
        """Keep serving the last-good keys, or admit we cannot verify anyone."""
        with self._lock:
            entry = self._cache_entry
            if entry is None:
                # Cold start: nothing usable has ever been fetched, so there is
                # no fallback and no token can be verified at all. Defer the next
                # attempt the same way the stale branch below does -- the reason
                # is identical, and this branch is the one the request path takes
                # when a restart coincides with a provider outage.
                self._cold_retry_deadline = self.now() + self.retry_interval
                raise IdentityProviderUnavailable(
                    "identity provider unreachable and no usable signing keys are cached"
                ) from exc
            # Push the next attempt out by retry_interval rather than leaving
            # the entry expired, otherwise every request during an outage
            # re-attempts a blocking fetch -- measured at 100 attempts for 100
            # requests. max() so a failure can only ever *defer* the next
            # attempt: a kid-triggered refresh can land here while the entry is
            # still fresh, and pulling its deadline in would let hostile input
            # force a fetch every retry_interval.
            entry.expiration_time = max(
                entry.expiration_time, self.now() + self.retry_interval
            )
            keyset = entry.keyset
        logger.warning(
            "jwks_refresh_failed_serving_stale",
            extra={"_event": "jwks_refresh_failed_serving_stale"},
            exc_info=True,
        )
        return keyset

def _jwks_url(supabase_url: str) -> str:
    """Where Supabase publishes this project's public signing keys."""
    return f"{supabase_url}/auth/v1/.well-known/jwks.json"


def make_jwks_fetcher(supabase_url: str, timeout: float = 10) -> Callable[[], dict[str, Any]]:
    """Build the production fetch_jwks callable for SupabaseJwtVerifier.

    PyJWKClient is here for its urllib and SSL handling, nothing else.
    cache_jwk_set=False is load-bearing rather than incidental: its cache
    clears itself on a failed refresh (fetch_data's finally puts None back),
    which is precisely the behaviour this module's own cache was written to
    avoid. Leaving it on would stack a cache with the wrong failure semantics
    in front of the one with the right ones.

    fetch_data returns the parsed JWKS dict and lets urllib and JSON errors
    out, which is the contract _keyset wants -- it treats any exception as an
    unreachable provider and serves the last-good key set instead.
    """
    client = jwt.PyJWKClient(
        _jwks_url(supabase_url),
        cache_jwk_set=False,
        timeout=timeout,
    )
    return client.fetch_data
