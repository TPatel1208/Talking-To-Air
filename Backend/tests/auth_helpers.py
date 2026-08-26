"""
tests/auth_helpers.py
=====================
Fixture identity for the tests that exercise authenticated endpoints.

T61 replaced the local ``users`` table with Supabase-issued **ES256** access
tokens, verified locally against a JWKS. These helpers mint tokens the real
verifier accepts and hand the app a verifier holding the matching public key.

**Why patching, not ``dependency_overrides``.** The auth check is a *middleware*
(``api.require_authentication``), deliberately so: a middleware is fail-closed,
meaning a route added tomorrow is authenticated unless ``PUBLIC_ENDPOINTS``
says otherwise, where a ``Depends`` is fail-open and silently leaves a forgotten
route wide open. FastAPI's ``app.dependency_overrides`` cannot reach middleware,
so injection happens by patching the module-level verifier instead. That is also
why ``api.supabase_verifier`` is built at import rather than inside ``lifespan``
-- moving it would leave these tests nowhere to inject.

**Not a pytest fixture on purpose.** Every consumer is a
``unittest.IsolatedAsyncioTestCase``, and pytest cannot inject fixtures as
arguments into unittest test methods. A plain importable module works from both.

The keypair is generated **once per process**. Key generation is the slow part,
and no test here varies the key -- except the wrong-signature case, which is
what :data:`OTHER_PRIVATE_KEY` exists for.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

from tta_backend.config.settings import get_settings
from tta_backend.services.supabase_jwt import AuthenticatedUser, SupabaseJwtVerifier

# Derived from the same setting api.py builds its own verifier's issuer from
# (conftest.py sets SUPABASE_URL before any test module is imported), so the
# two cannot drift apart. A mismatch here would fail every test with "Invalid
# issuer." and look like a bug in the code under test.
ISSUER = f"{get_settings().supabase_url}/auth/v1"

KID = "test-kid"
DEFAULT_SUB = "user-1"
DEFAULT_EMAIL = "tester@example.com"

PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
# A second, unpublished key. A token signed with this one is well-formed and
# carries the right claims, so it isolates *signature* rejection from every
# other reason a token can be refused.
OTHER_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())

_JWK = ECAlgorithm.to_jwk(PRIVATE_KEY.public_key(), as_dict=True)
_JWK.update({"kid": KID, "use": "sig", "alg": "ES256"})
JWKS = {"keys": [_JWK]}


def make_token(
    sub: str = DEFAULT_SUB,
    *,
    email: str | None = DEFAULT_EMAIL,
    private_key=None,
    kid: str = KID,
    expires_in_minutes: int = 10,
    **claims,
) -> str:
    """Mint an access token shaped like the ones Supabase issues.

    ``expires_in_minutes`` accepts a negative value to produce an already-expired
    token. Anything in ``**claims`` overrides a default, so a test can drop or
    corrupt a single claim without restating the rest.
    """
    payload = {
        "sub": sub,
        "email": email,
        "aud": "authenticated",
        "iss": ISSUER,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
    }
    payload.update(claims)
    return jwt.encode(
        payload,
        private_key if private_key is not None else PRIVATE_KEY,
        algorithm="ES256",
        headers={"kid": kid},
    )


def auth_header(sub: str = DEFAULT_SUB, **kwargs) -> dict[str, str]:
    """The Authorization header for a freshly minted token."""
    return {"Authorization": f"Bearer {make_token(sub, **kwargs)}"}


def user(sub: str = DEFAULT_SUB, email: str | None = DEFAULT_EMAIL) -> AuthenticatedUser:
    """The identity the middleware will attach for a token minted with the same sub."""
    return AuthenticatedUser(id=sub, email=email)


def build_verifier() -> SupabaseJwtVerifier:
    """A verifier holding the fixture public key, reachable without network."""
    return SupabaseJwtVerifier(fetch_jwks=lambda: JWKS, issuer=ISSUER)


def patch_verifier():
    """Install a fixture-key verifier in place of the one api.py built at import.

    Returns the patcher itself, so it works either as a context manager or via
    ``.start()`` / ``.stop()``.
    """
    return patch("tta_backend.api.supabase_verifier", build_verifier())
