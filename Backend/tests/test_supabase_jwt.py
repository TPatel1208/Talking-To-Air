import threading
import unittest
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

from tta_backend.services.supabase_jwt import (
    AuthenticationError,
    IdentityProviderUnavailable,
    SupabaseJwtVerifier,
    _jwks_url,
    make_jwks_fetcher,
)
ISSUER = "https://nzzetaoojoopkfjrkqyk.supabase.co/auth/v1"

class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, sec: int): self.t += sec
class TestSupabaseJWT(unittest.TestCase):
    def setUp(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        jwk = ECAlgorithm.to_jwk(self.private_key.public_key(), as_dict=True)
        jwk.update({"kid": "test-kid", "use": "sig", "alg" : "ES256"})
        self.jwks = {"keys": [jwk]}

    def _make_token(self, *, private_key=None, kid="test-kid", **claims):
        payload = {
            "sub": "11111111-2222-3333-4444-555555555555",
            "email": "me@example.com",
            "aud": "authenticated",
            "iss": ISSUER,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
        }
        payload.update(claims)
        return jwt.encode(
            payload,
            private_key if private_key is not None else self.private_key,
            algorithm="ES256",
            headers={"kid": kid},
        )

    def test_valid_token_returns_identity(self):
        token = jwt.encode(
            {
                "sub": "11111111-2222-3333-4444-555555555555",
                "email": "me@example.com",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
            },
            self.private_key,
            algorithm="ES256",
            headers={"kid": "test-kid"},
        )

        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        user = verifier.verify(token)

        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")
        self.assertEqual(user.email, "me@example.com")

    def test_expired_token(self):
        token = jwt.encode(
            {
                "sub": "11111111-2222-3333-4444-555555555555",
                "email": "me@example.com",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": datetime.now(timezone.utc) - timedelta(minutes=10),

            },
            self.private_key,
            algorithm="ES256",
            headers={"kid": "test-kid"},
        )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("expired", str(ctx.exception))

    def test_different_keypair(self):
        imposter_private_key = ec.generate_private_key(ec.SECP256R1())
        token = jwt.encode(
            {
                "sub": "11111111-2222-3333-4444-555555555555",
                "email": "me@example.com",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
            },
            imposter_private_key,
            algorithm="ES256",
            headers={"kid": "test-kid"},
        )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("Signature verification failed", str(ctx.exception))

    def test_unsupported_algorithm(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        imposter_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    imposter_private_key,
                    algorithm="RS256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("algorithm", str(ctx.exception))

    def test_audience_is_authenticated(self):
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "unauthenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("audience", str(ctx.exception))

    def  test_valid_issuer(self):
        false_issuer = "https://fake-issuer.com"
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": false_issuer,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("issuer", str(ctx.exception))

    def test_subject_is_required(self):
        token = jwt.encode(
                    {
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("sub", str(ctx.exception))

    def test_subject_must_be_a_string(self):
        token = jwt.encode(
                    {
                        "sub": None,
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError):
            verifier.verify(token)

    def test_expiry_is_required(self):
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("exp", str(ctx.exception))

    def test_cache_empty_fetch(self):
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        calls = []
        def fake_fetch():
            calls.append(1)
            return self.jwks
        verifier = SupabaseJwtVerifier(fetch_jwks=fake_fetch, issuer=ISSUER)

        verifier.verify(token)
        verifier.verify(token)

        self.assertEqual(len(calls), 1)


    def test_cache_present_younger_than_ttl(self):
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        calls = []
        def fake_fetch():
            calls.append(1)
            return self.jwks
        verifier = SupabaseJwtVerifier(fetch_jwks=fake_fetch, issuer=ISSUER)

        verifier.verify(token)
        verifier.verify(token)
        verifier.verify(token)

        self.assertEqual(len(calls), 1)
    def test_cache_present_older_than_ttl(self):

        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        calls = []
        def fake_fetch():
            calls.append(1)
            return self.jwks
        fake_clock = FakeClock()
        cache_ttl = 900
        verifier = SupabaseJwtVerifier(
            fetch_jwks=fake_fetch, issuer=ISSUER, cache_ttl=cache_ttl, now=fake_clock
        )

        verifier.verify(token)
        fake_clock.advance(cache_ttl + 1)
        user = verifier.verify(token)

        self.assertEqual(len(calls), 2)
        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")

    def test_supabase_outage_stale_tolerance(self):

        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        calls = []
        def flaky_fetch():
            calls.append(1)
            if len(calls) > 1:
                raise ConnectionError("supabase unreachable")
            return self.jwks
        fake_clock = FakeClock()
        cache_ttl = 900
        verifier = SupabaseJwtVerifier(
            fetch_jwks=flaky_fetch, issuer=ISSUER, cache_ttl=cache_ttl, now=fake_clock
        )

        verifier.verify(token)
        fake_clock.advance(cache_ttl + 1)
        user = verifier.verify(token)

        self.assertEqual(len(calls), 2)
        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")
    def test_outage_does_not_refetch_on_every_request(self):
        """An outage must cost one fetch per retry_interval, not one per request.

        Without this, a lapsed cache entry stays lapsed, so every authenticated
        request re-attempts a blocking fetch -- turning a Supabase outage into a
        stall of our own event loop. The fallback would "work" while the process
        drowned in retries.
        """
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )
        calls = []
        def flaky_fetch():
            calls.append(1)
            if len(calls) > 1:
                raise ConnectionError("supabase unreachable")
            return self.jwks
        fake_clock = FakeClock()
        cache_ttl = 900
        retry_interval = 60
        verifier = SupabaseJwtVerifier(
            fetch_jwks=flaky_fetch,
            issuer=ISSUER,
            cache_ttl=cache_ttl,
            retry_interval=retry_interval,
            now=fake_clock,
        )

        verifier.verify(token)                      # warm the cache
        fake_clock.advance(cache_ttl + 1)           # TTL lapses; outage begins

        for _ in range(10):
            user = verifier.verify(token)

        # One warm-up fetch plus exactly one failed refresh -- the other nine
        # requests were served from the stale entry without touching the network.
        self.assertEqual(len(calls), 2)
        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")

        # ...and once the retry interval passes, it does try again.
        fake_clock.advance(retry_interval + 1)
        verifier.verify(token)
        self.assertEqual(len(calls), 3)

    def test_unknown_kid_refetches_again_once_the_cooldown_has_passed(self):
        """The other edge: throttled, not switched off.

        Without this, an implementation that simply never refetched after the
        first miss would satisfy the cooldown test above -- and a rotation that
        happened during the cooldown would never be picked up.
        """
        calls = []
        def counting_fetch():
            calls.append(1)
            return self.jwks

        fake_clock = FakeClock()
        cooldown = 60
        verifier = SupabaseJwtVerifier(
            fetch_jwks=counting_fetch,
            issuer=ISSUER,
            kid_refresh_cooldown=cooldown,
            now=fake_clock,
        )
        verifier.verify(self._make_token())              # warm: 1 fetch

        with self.assertRaises(AuthenticationError):     # miss: refetch -> 2
            verifier.verify(self._make_token(kid="bogus-1"))
        with self.assertRaises(AuthenticationError):     # inside cooldown -> 2
            verifier.verify(self._make_token(kid="bogus-2"))
        self.assertEqual(len(calls), 2)

        fake_clock.advance(cooldown + 1)
        with self.assertRaises(AuthenticationError):     # cooldown over -> 3
            verifier.verify(self._make_token(kid="bogus-3"))
        self.assertEqual(len(calls), 3)

    def test_throttled_kid_refreshes_are_reported_once_per_window(self):
        """A burst of bogus kids must be visible in the logs -- but only once.

        Logging each refusal would hand whoever is generating the kids control
        of our log volume, so the count is reported when the window closes.
        """
        verifier = SupabaseJwtVerifier(
            fetch_jwks=lambda: self.jwks,
            issuer=ISSUER,
            kid_refresh_cooldown=60,
            now=FakeClock(),
        )
        verifier.verify(self._make_token())

        for i in range(4):
            with self.assertRaises(AuthenticationError):
                verifier.verify(self._make_token(kid=f"bogus-{i}"))

        # Nothing logged yet: the window is still open.
        fake_clock = verifier.now
        fake_clock.advance(61)
        with self.assertLogs("tta_backend.services.supabase_jwt", level="WARNING") as logs:
            with self.assertRaises(AuthenticationError):
                verifier.verify(self._make_token(kid="bogus-last"))

        self.assertEqual(len(logs.records), 1)
        record = logs.records[0]
        self.assertEqual(record.getMessage(), "jwks_kid_refresh_throttled")
        # 4 bogus kids, minus the first which was allowed through to a refetch.
        self.assertEqual(record._throttled_requests, 3)

    def test_cold_start_with_unreachable_provider_is_not_an_auth_error(self):
        """No keys have ever been fetched: we cannot verify anyone.

        This must NOT be an AuthenticationError. A 401 sends the frontend's
        re-auth modal at a user whose credentials are fine, and no amount of
        signing in will fix an unreachable identity provider.
        """
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                    headers={"kid": "test-kid"},
                )

        def dead_fetch():
            raise ConnectionError("supabase unreachable")

        verifier = SupabaseJwtVerifier(fetch_jwks=dead_fetch, issuer=ISSUER)
        with self.assertRaises(IdentityProviderUnavailable):
            verifier.verify(token)

    def test_identity_provider_unavailable_is_not_an_authentication_error(self):
        # Pins the class relationship itself: if IdentityProviderUnavailable
        # ever subclasses AuthenticationError, Phase 2's middleware silently
        # turns every provider outage back into a 401.
        self.assertFalse(issubclass(IdentityProviderUnavailable, AuthenticationError))

    def test_throttled_and_unthrottled_kid_rejections_are_indistinguishable(self):
        """The rejection text must not say whether the cooldown was open.

        Phase 2 returns this message as the 401 body. If the throttled and
        unthrottled paths word it differently, an unauthenticated caller can
        watch for the change to learn exactly when the window resets and time
        bogus kids to land one outbound fetch per window.
        """
        verifier = SupabaseJwtVerifier(
            fetch_jwks=lambda: self.jwks,
            issuer=ISSUER,
            kid_refresh_cooldown=60,
            now=FakeClock(),
        )
        verifier.verify(self._make_token())

        with self.assertRaises(AuthenticationError) as refetched:
            verifier.verify(self._make_token(kid="bogus-1"))
        with self.assertRaises(AuthenticationError) as throttled:
            verifier.verify(self._make_token(kid="bogus-2"))

        self.assertEqual(str(refetched.exception), str(throttled.exception))

    # ------------------------------------------------------------------
    # A 200 response carrying a useless key set. None of these raise, so the
    # stale-tolerance path above cannot see them -- they have to be caught by
    # validating what came back before it is allowed into the cache.
    # ------------------------------------------------------------------

    def test_empty_key_set_does_not_evict_good_keys(self):
        """Supabase answers 200 with no keys; we must keep serving.

        Without validation at fetch time the empty set is cached as if it were
        good, and every request 401s for a full cache_ttl -- outlasting the
        provider's own recovery.
        """
        published = {"payload": self.jwks}
        verifier = SupabaseJwtVerifier(
            fetch_jwks=lambda: published["payload"],
            issuer=ISSUER,
            cache_ttl=900,
            now=FakeClock(),
        )
        verifier.verify(self._make_token())          # warm on good keys

        published["payload"] = {"keys": []}          # keys disabled upstream
        verifier.now.advance(901)                    # TTL lapses -> refresh

        user = verifier.verify(self._make_token())
        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")

    def test_empty_key_set_on_cold_start_is_not_an_auth_error(self):
        """Nothing cached and nothing usable returned: 503, not 401.

        A 401 opens the re-auth modal at users whose credentials are fine.
        """
        verifier = SupabaseJwtVerifier(
            fetch_jwks=lambda: {"keys": []}, issuer=ISSUER, now=FakeClock()
        )
        with self.assertRaises(IdentityProviderUnavailable):
            verifier.verify(self._make_token())

    def test_failed_refresh_never_shortens_a_fresh_entry(self):
        """A kid-triggered refresh that fails must not pull the deadline in.

        kid is attacker-controlled, so otherwise one bogus token during an
        outage drops a 900s entry to 60s and puts the verifier back on a
        per-minute blocking-fetch cadence.
        """
        calls = []
        def flaky_fetch():
            calls.append(1)
            if len(calls) > 1:
                raise ConnectionError("supabase unreachable")
            return self.jwks

        fake_clock = FakeClock()
        verifier = SupabaseJwtVerifier(
            fetch_jwks=flaky_fetch,
            issuer=ISSUER,
            cache_ttl=900,
            retry_interval=60,
            now=fake_clock,
        )
        verifier.verify(self._make_token())                    # entry good to t=900

        with self.assertRaises(AuthenticationError):           # forces a refresh
            verifier.verify(self._make_token(kid="bogus"))
        self.assertEqual(len(calls), 2)

        # t=100: the entry still had 800s left, so nothing should refetch.
        fake_clock.advance(100)
        verifier.verify(self._make_token())
        self.assertEqual(len(calls), 2)

    # ------------------------------------------------------------------
    # Malformed / hostile tokens. None of these carry a usable key id, so
    # they never reach jwt.decode -- they are rejected on the way in. Each
    # one used to escape verify() as a KeyError or DecodeError, which the
    # Phase 2 middleware would surface as a 500 instead of a 401.
    # ------------------------------------------------------------------

    def test_token_without_kid_header_is_rejected(self):
        token = jwt.encode(
                    {
                        "sub": "11111111-2222-3333-4444-555555555555",
                        "email": "me@example.com",
                        "aud": "authenticated",
                        "iss": ISSUER,
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
                    },
                    self.private_key,
                    algorithm="ES256",
                )
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(token)
        self.assertIn("Kid", str(ctx.exception))

    def test_unknown_kid_refetches_once_then_rejects(self):
        """A miss must look again before giving up -- but only look once.

        The cache is warm and inside its TTL here, so the second fetch can only
        have been triggered by the unknown kid. Without that refetch, a Supabase
        key rotation would 401 every user until the TTL lapsed.
        """
        good_token = self._make_token()
        unknown_kid_token = self._make_token(kid="a-kid-the-keyset-has-never-heard-of")

        calls = []
        def counting_fetch():
            calls.append(1)
            return self.jwks

        verifier = SupabaseJwtVerifier(
            fetch_jwks=counting_fetch, issuer=ISSUER, now=FakeClock()
        )

        verifier.verify(good_token)          # warms the cache
        self.assertEqual(len(calls), 1)

        with self.assertRaises(AuthenticationError) as ctx:
            verifier.verify(unknown_kid_token)

        self.assertIn("Kid", str(ctx.exception))
        self.assertEqual(len(calls), 2)      # the miss forced one refetch

    def test_rotated_key_is_picked_up_without_waiting_for_the_ttl(self):
        """Supabase rotates; a token signed by the new key must verify at once.

        The cached set is still fresh, so nothing but the unknown kid can
        prompt the refetch that makes this succeed.
        """
        rotated_private_key = ec.generate_private_key(ec.SECP256R1())
        rotated_jwk = ECAlgorithm.to_jwk(rotated_private_key.public_key(), as_dict=True)
        rotated_jwk.update({"kid": "rotated-kid", "use": "sig", "alg": "ES256"})

        published = {"keys": list(self.jwks["keys"])}

        calls = []
        def counting_fetch():
            calls.append(1)
            return {"keys": list(published["keys"])}

        verifier = SupabaseJwtVerifier(
            fetch_jwks=counting_fetch, issuer=ISSUER, now=FakeClock()
        )
        verifier.verify(self._make_token())          # warm on the old key

        # Supabase publishes the new key and starts signing with it.
        published["keys"].append(rotated_jwk)
        rotated_token = self._make_token(
            private_key=rotated_private_key, kid="rotated-kid"
        )

        user = verifier.verify(rotated_token)

        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")
        self.assertEqual(len(calls), 2)

    def test_unknown_kids_inside_the_cooldown_cost_one_fetch(self):
        """The kid is attacker-controlled, so the refetch above must be throttled.

        `kid` comes from the token header -- unsigned, unverified, free for
        anyone to invent -- and this middleware runs before any rate limiting.
        Without a cooldown, each bogus kid becomes an outbound HTTPS request,
        every one of them a blocking fetch on the event loop.
        """
        calls = []
        def counting_fetch():
            calls.append(1)
            return self.jwks

        verifier = SupabaseJwtVerifier(
            fetch_jwks=counting_fetch, issuer=ISSUER, now=FakeClock()
        )
        verifier.verify(self._make_token())          # warms the cache
        self.assertEqual(len(calls), 1)

        for i in range(5):
            with self.assertRaises(AuthenticationError):
                verifier.verify(self._make_token(kid=f"bogus-kid-{i}"))

        # One refetch for the first miss; the other four were refused without
        # touching the network.
        self.assertEqual(len(calls), 2)

    def test_garbage_string_is_rejected(self):
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError):
            verifier.verify("hello.world.garbage")

    def test_empty_string_is_rejected(self):
        # Needs no signature, no key, no valid structure at all -- reachable
        # by anyone who can reach the endpoint.
        verifier = SupabaseJwtVerifier(fetch_jwks = lambda: self.jwks, issuer = ISSUER)
        with self.assertRaises(AuthenticationError):
            verifier.verify("")


class TestSupabaseJwtWarm(unittest.IsolatedAsyncioTestCase):
    """T61 Phase 1: the boot-time warm.

    Boot must not block on Supabase (T17 degrade-don't-die), but the first real
    request should not be paying for a blocking fetch either.
    """

    def setUp(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        jwk = ECAlgorithm.to_jwk(self.private_key.public_key(), as_dict=True)
        jwk.update({"kid": "test-kid", "use": "sig", "alg": "ES256"})
        self.jwks = {"keys": [jwk]}

    def _token(self):
        return jwt.encode(
            {
                "sub": "11111111-2222-3333-4444-555555555555",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
            },
            self.private_key,
            algorithm="ES256",
            headers={"kid": "test-kid"},
        )

    async def test_warm_populates_the_cache_before_any_request(self):
        calls = []
        def counting_fetch():
            calls.append(1)
            return self.jwks

        verifier = SupabaseJwtVerifier(
            fetch_jwks=counting_fetch, issuer=ISSUER, now=FakeClock()
        )
        await verifier.warm()
        self.assertEqual(len(calls), 1)

        # The first real request is now served from cache -- no extra fetch.
        user = verifier.verify(self._token())
        self.assertEqual(user.id, "11111111-2222-3333-4444-555555555555")
        self.assertEqual(len(calls), 1)

    async def test_warm_retries_with_capped_backoff_then_succeeds(self):
        calls = []
        def flaky_fetch():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("supabase unreachable")
            return self.jwks

        slept = []
        async def fake_sleep(seconds):
            slept.append(seconds)

        verifier = SupabaseJwtVerifier(
            fetch_jwks=flaky_fetch, issuer=ISSUER, now=FakeClock()
        )
        warmed = await verifier.warm(sleep=fake_sleep, max_backoff_seconds=1.5)

        self.assertTrue(warmed)
        self.assertEqual(len(calls), 3)
        # Doubling, but capped -- a long outage must not push the retry out
        # past the point where recovery is noticed.
        self.assertEqual(slept, [1.0, 1.5])

    async def test_warm_gives_up_without_raising(self):
        calls = []
        def dead_fetch():
            calls.append(1)
            raise ConnectionError("supabase unreachable")

        async def fake_sleep(seconds):
            pass

        verifier = SupabaseJwtVerifier(
            fetch_jwks=dead_fetch, issuer=ISSUER, now=FakeClock()
        )
        warmed = await verifier.warm(sleep=fake_sleep, max_attempts=3)

        # Boot survives. The request path still handles the cold cache.
        self.assertFalse(warmed)
        self.assertEqual(len(calls), 3)

    async def test_warm_fetches_off_the_event_loop(self):
        """The fetch is blocking urllib -- it must not run on the loop thread.

        Awaiting it directly would stall every concurrent request for the whole
        HTTP timeout, which is the opposite of what a background warm is for.
        """
        fetch_thread = []
        def recording_fetch():
            fetch_thread.append(threading.get_ident())
            return self.jwks

        verifier = SupabaseJwtVerifier(
            fetch_jwks=recording_fetch, issuer=ISSUER, now=FakeClock()
        )
        await verifier.warm()

        self.assertNotEqual(fetch_thread[0], threading.get_ident())

    async def test_start_does_not_block_boot_and_stop_cancels(self):
        """start_warm() hands back control immediately; stop_warm() cleans up.

        The fetch runs in a worker thread, so anything it touches must be
        thread-safe -- an asyncio.Event here would raise "non-thread-safe
        operation invoked on an event loop other than the current one" from
        inside the fetch, which warm() would then mistake for an outage.
        """
        fetched = threading.Event()

        def fetch():
            fetched.set()
            return self.jwks

        verifier = SupabaseJwtVerifier(
            fetch_jwks=fetch, issuer=ISSUER, now=FakeClock()
        )

        verifier.start_warm()                       # sync call, no await
        task = verifier._warm_task
        self.assertIsNotNone(task)
        self.assertFalse(task.done())               # boot did not wait for it

        self.assertTrue(await task)                 # now let it finish
        self.assertTrue(fetched.is_set())

        await verifier.stop_warm()
        self.assertIsNone(verifier._warm_task)

    async def test_start_warm_is_idempotent(self):
        """A second start must not leave an orphaned task nobody cancels."""
        verifier = SupabaseJwtVerifier(
            fetch_jwks=lambda: self.jwks, issuer=ISSUER, now=FakeClock()
        )
        verifier.start_warm()
        first = verifier._warm_task
        verifier.start_warm()

        self.assertIs(verifier._warm_task, first)
        await verifier.stop_warm()


class ColdStartCooldownTests(unittest.TestCase):
    """A cold cache during an outage must not fetch once per request.

    The warm-cache branch defers by parking a deadline on the cache entry. The
    cold branch has no entry, and before this it re-attempted the blocking fetch
    for every caller -- which the request path runs on a worker thread, one per
    inbound request, so a restart that coincided with a Supabase outage put the
    whole default executor behind doomed 10s calls and turned each unauthenticated
    request into an outbound HTTPS attempt.
    """

    def setUp(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        jwk = ECAlgorithm.to_jwk(self.private_key.public_key(), as_dict=True)
        jwk.update({"kid": "test-kid", "use": "sig", "alg": "ES256"})
        self.jwks = {"keys": [jwk]}

    def _token(self):
        return jwt.encode(
            {
                "sub": "11111111-2222-3333-4444-555555555555",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
            },
            self.private_key,
            algorithm="ES256",
            headers={"kid": "test-kid"},
        )

    def test_a_cold_outage_costs_one_fetch_per_retry_interval(self):
        calls = []

        def dead_fetch():
            calls.append(1)
            raise ConnectionError("supabase unreachable")

        clock = FakeClock()
        verifier = SupabaseJwtVerifier(
            fetch_jwks=dead_fetch, issuer=ISSUER, retry_interval=60, now=clock
        )

        for _ in range(25):
            with self.assertRaises(IdentityProviderUnavailable):
                verifier.verify(self._token())
        self.assertEqual(len(calls), 1, "every request re-attempted the blocking fetch")

        clock.advance(61)
        with self.assertRaises(IdentityProviderUnavailable):
            verifier.verify(self._token())
        self.assertEqual(len(calls), 2, "the cooldown never reopened")

    def test_recovery_clears_the_cooldown_and_serves_normally(self):
        state = {"up": False}

        def flaky_fetch():
            if not state["up"]:
                raise ConnectionError("supabase unreachable")
            return self.jwks

        clock = FakeClock()
        verifier = SupabaseJwtVerifier(
            fetch_jwks=flaky_fetch, issuer=ISSUER, retry_interval=60, now=clock
        )
        with self.assertRaises(IdentityProviderUnavailable):
            verifier.verify(self._token())

        state["up"] = True
        clock.advance(61)
        self.assertEqual(
            verifier.verify(self._token()).id, "11111111-2222-3333-4444-555555555555"
        )
        # A later failure must fall to the *stale* branch, not the cold one.
        state["up"] = False
        clock.advance(10_000)
        self.assertEqual(
            verifier.verify(self._token()).id, "11111111-2222-3333-4444-555555555555"
        )


class MakeJwksFetcherTests(unittest.TestCase):
    """Cover the one link in the chain the tests above never touch.

    Every test in this file hands the verifier a ``lambda: self.jwks``, so
    make_jwks_fetcher itself -- and the URL it builds -- is exercised for the
    first time by a real request at boot. That is a bad place to find a typo:
    warm() never raises, so a wrong path does not crash anything. The backend
    comes up, every authenticated request 503s, and the only evidence is a
    jwks_warm_attempt_failed log line.
    """

    def test_url_is_the_supabase_jwks_endpoint(self):
        self.assertEqual(
            _jwks_url("https://abc.supabase.co"),
            "https://abc.supabase.co/auth/v1/.well-known/jwks.json",
        )

    def test_fetcher_is_bound_to_a_client_for_that_url(self):
        # The client is a local inside the factory; the returned bound method
        # is the only handle on it.
        fetcher = make_jwks_fetcher("https://abc.supabase.co")
        self.assertEqual(
            fetcher.__self__.uri,
            "https://abc.supabase.co/auth/v1/.well-known/jwks.json",
        )

    def test_pyjwkclient_caching_stays_off(self):
        # Reaches into PyJWT's internals deliberately. Turning cache_jwk_set
        # back on silently reinstates a cache that clears itself on a failed
        # refresh, sitting in front of the one this module wrote to avoid
        # exactly that. If a PyJWT upgrade renames this attribute, repair the
        # assertion -- do not delete it.
        fetcher = make_jwks_fetcher("https://abc.supabase.co")
        self.assertIsNone(fetcher.__self__.jwk_set_cache)

    def test_timeout_reaches_the_client(self):
        # PyJWKClient's own default is 30s, long enough to matter to warm()'s
        # backoff schedule, so an unpassed timeout would not be obvious.
        fetcher = make_jwks_fetcher("https://abc.supabase.co", timeout=3)
        self.assertEqual(fetcher.__self__.timeout, 3)
