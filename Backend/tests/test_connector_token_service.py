import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _make_token(exp=None, **extra_claims):
    import jwt

    payload = dict(extra_claims)
    if exp is not None:
        payload["exp"] = exp
    # Signed with an arbitrary secret -- decode_token_expiry never verifies
    # the signature (EDL, not us, is the audience), so any secret works here.
    return jwt.encode(payload, "arbitrary-test-secret", algorithm="HS256")


class ConnectorTokenServiceTests(unittest.TestCase):
    def test_accepts_a_future_dated_token_and_returns_its_exp(self):
        from services.connector_token_service import decode_token_expiry

        future = datetime.now(timezone.utc) + timedelta(days=60)
        token = _make_token(exp=int(future.timestamp()), sub="user")

        expires_at = decode_token_expiry(token)

        self.assertAlmostEqual(expires_at.timestamp(), future.timestamp(), delta=1)

    def test_rejects_an_expired_token_with_a_specific_message(self):
        from services.connector_token_service import TokenValidationError, decode_token_expiry

        past = datetime.now(timezone.utc) - timedelta(days=1)
        token = _make_token(exp=int(past.timestamp()))

        with self.assertRaisesRegex(TokenValidationError, "already expired"):
            decode_token_expiry(token)

    def test_rejects_non_jwt_garbage(self):
        from services.connector_token_service import TokenValidationError, decode_token_expiry

        with self.assertRaises(TokenValidationError):
            decode_token_expiry("this is not a jwt at all")

    def test_rejects_a_token_missing_the_exp_claim(self):
        from services.connector_token_service import TokenValidationError, decode_token_expiry

        token = _make_token(sub="user")  # no exp

        with self.assertRaises(TokenValidationError):
            decode_token_expiry(token)

    def test_rejects_an_empty_paste(self):
        from services.connector_token_service import TokenValidationError, decode_token_expiry

        with self.assertRaisesRegex(TokenValidationError, "Paste a token"):
            decode_token_expiry("   ")


if __name__ == "__main__":
    unittest.main()
