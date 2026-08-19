import unittest
from unittest.mock import patch
from pydantic import ValidationError
from types import SimpleNamespace
import psycopg

from tta_backend.api import RegisterRequest

class RegisterRequestValidationTests(unittest.TestCase):
    def test_rejects_password_missing_uppercase(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="lowercase1!")
        self.assertIn("uppercase",str(ctx.exception))

    def test_rejects_password_missing_lowercase(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="UPPERCASE1!")
        self.assertIn("lowercase",str(ctx.exception))

    def test_rejects_password_missing_digit(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="NoDigits!")
        self.assertIn("digit",str(ctx.exception))

    def test_rejects_password_missing_special_character(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="NoSpecial1")
        self.assertIn("special character",str(ctx.exception))

    def test_rejects_password_with_spaces(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="Valid 1!")
        self.assertIn("spaces",str(ctx.exception))

    def test_rejects_password_too_short(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="Shrt1!")
        self.assertIn("between 8 and 1024",str(ctx.exception))

    def test_rejects_password_too_long(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="validuser", password="A"*1025)
        self.assertIn("between 8 and 1024",str(ctx.exception))

    def test_rejects_username_too_short(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="ab", password="Validpw1!")
        self.assertIn("between 3 and 64",str(ctx.exception))

    def test_rejects_username_too_long(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="a"*65, password="Validpw1!")
        self.assertIn("between 3 and 64",str(ctx.exception))

    def test_rejects_username_with_spaces(self):
        with self.assertRaises(ValidationError) as ctx:
            RegisterRequest(username="user name", password="Validpw1!")
        self.assertIn("spaces",str(ctx.exception))

    def test_accepts_valid_username_and_password(self):
        req = RegisterRequest(
            username="validuser",
            password="Validpw1!"
        )
        self.assertEqual(req.username, "validuser")
        self.assertEqual(req.password, "Validpw1!")

class RegisterEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import tta_backend.api as api
        self.httpx = httpx
        self.api = api

    async def test_registerendpoint_rejects_weak_password_over_http(self):
        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:
            response = await client.post("/auth/register", json={
                "username": "validuser",
                "password": "x"
            })
        self.assertEqual(response.status_code, 422)

    async def test_register_successful_with_valid_password(self):
        async def fake_create_user(username, password_hash):
            return SimpleNamespace(id="user-1", username=username, is_active=True)
        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "create_user", fake_create_user):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/auth/register",
                    json={"username": "newuser", "password": "Valid1Pass!"},
                )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["username"], "newuser")

    async def test_duplicate_user(self):
        async def fake_create_user(username, password_hash):
            raise psycopg.errors.UniqueViolation("duplicate key")
        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "create_user", fake_create_user):
            async with self.httpx.AsyncClient(transport=transport, base_url="https://test") as client:
                response = await client.post(
                    "/auth/register",
                    json={"username": "newuser", "password": "Valid1Pass!"},
                )

        self.assertEqual(response.status_code,409)

    async def test_login_accepts_legacy_accounts(self):
        password_hash = self.api.hash_password("1234")
        legacy_user = SimpleNamespace(
            id = "user-1",
            username="legacyuser",
            password_hash=password_hash,
            is_active=True,
        )
        async def fake_get_user_by_username(username):
            return legacy_user if username == "legacyuser" else None
        transport = self.httpx.ASGITransport(app=self.api.app)
        with patch.object(self.api, "get_user_by_username", fake_get_user_by_username):
            async with self.httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/auth/login",
                    json={"username": "legacyuser", "password": "1234"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["access_token"])
