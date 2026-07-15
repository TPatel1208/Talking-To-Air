"""
tests/test_connector_credential_service.py
============================================
T31: EdlCredentialInjector -- the injection-policy resolver bound into
bind_workspace (earthdata_mcp/workspace.py) as edl_injector. Exercises the
policy (connected ∧ unexpired), the short-TTL encrypted-row cache and its
invalidation, and the fire-and-forget/coalesced last_used_at bookkeeping --
independent of any live MCP or Postgres (repository calls are patched, same
seam test_user_connector_repository.py already proves against a fake conn).
"""
import asyncio
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _settings_with_key():
    from config.settings import Settings
    from cryptography.fernet import Fernet

    return Settings(connector_encryption_key=Fernet.generate_key().decode())


def _connected_row(*, expires_in=timedelta(days=1), encrypted_secret="x", status="connected"):
    return {
        "encrypted_secret": encrypted_secret,
        "expires_at": datetime.now(timezone.utc) + expires_in,
        "status": status,
    }


class ResolveTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_none_when_the_encryption_key_is_unconfigured(self):
        from config.settings import Settings
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(Settings(connector_encryption_key=None))

        self.assertIsNone(await injector.resolve("user-1"))

    async def test_returns_none_when_no_connector_row_exists(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        with patch("services.connector_credential_service.get_connector_secret_row", AsyncMock(return_value=None)):
            self.assertIsNone(await injector.resolve("user-1"))

    async def test_returns_none_when_status_is_not_connected(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        row = _connected_row(status="error")
        with patch("services.connector_credential_service.get_connector_secret_row", AsyncMock(return_value=row)):
            self.assertIsNone(await injector.resolve("user-1"))

    async def test_returns_none_when_expired(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        row = _connected_row(expires_in=timedelta(days=-1))
        with patch("services.connector_credential_service.get_connector_secret_row", AsyncMock(return_value=row)):
            self.assertIsNone(await injector.resolve("user-1"))

    async def test_returns_the_decrypted_token_when_connected_and_unexpired(self):
        from services.connector_credential_service import EdlCredentialInjector
        from utils.connector_crypto import encrypt_secret, get_connector_cipher

        settings = _settings_with_key()
        encrypted = encrypt_secret(get_connector_cipher(settings), "raw-edl-token")
        row = _connected_row(encrypted_secret=encrypted)

        injector = EdlCredentialInjector(settings)
        with patch("services.connector_credential_service.get_connector_secret_row", AsyncMock(return_value=row)):
            token = await injector.resolve("user-1")

        self.assertEqual(token, "raw-edl-token")

    async def test_a_row_undecryptable_under_the_current_key_resolves_to_none_not_a_raised_error(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        row = _connected_row(encrypted_secret="not-actually-fernet-ciphertext")
        with patch("services.connector_credential_service.get_connector_secret_row", AsyncMock(return_value=row)):
            self.assertIsNone(await injector.resolve("user-1"))


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_second_resolve_within_the_ttl_does_not_refetch(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key(), cache_ttl_seconds=60.0)
        fetch = AsyncMock(return_value=None)
        with patch("services.connector_credential_service.get_connector_secret_row", fetch):
            await injector.resolve("user-1")
            await injector.resolve("user-1")

        self.assertEqual(fetch.call_count, 1)

    async def test_invalidate_forces_a_refetch_on_the_next_resolve(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key(), cache_ttl_seconds=60.0)
        fetch = AsyncMock(return_value=None)
        with patch("services.connector_credential_service.get_connector_secret_row", fetch):
            await injector.resolve("user-1")
            injector.invalidate("user-1")
            await injector.resolve("user-1")

        self.assertEqual(fetch.call_count, 2)

    async def test_caching_is_scoped_per_user(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key(), cache_ttl_seconds=60.0)
        fetch = AsyncMock(return_value=None)
        with patch("services.connector_credential_service.get_connector_secret_row", fetch):
            await injector.resolve("user-1")
            await injector.resolve("user-2")

        self.assertEqual(fetch.call_count, 2)


class MarkUsedTests(unittest.IsolatedAsyncioTestCase):
    async def test_coalesces_repeated_calls_for_the_same_user_into_one_write(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        write = AsyncMock()
        with patch("services.connector_credential_service.touch_last_used_at", write):
            injector.mark_used("user-1")
            injector.mark_used("user-1")
            injector.mark_used("user-1")
            await asyncio.sleep(0)  # let the fire-and-forget task(s) run

        write.assert_awaited_once_with("user-1", "earthdata")

    async def test_a_failing_write_is_swallowed_not_raised(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        write = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("services.connector_credential_service.touch_last_used_at", write):
            injector.mark_used("user-1")  # must not raise synchronously
            await asyncio.sleep(0)

        write.assert_awaited()


class MarkInvalidTests(unittest.IsolatedAsyncioTestCase):
    async def test_flips_status_to_error_and_invalidates_the_cache(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key(), cache_ttl_seconds=60.0)
        fetch = AsyncMock(return_value=_connected_row())
        set_status = AsyncMock()
        with patch("services.connector_credential_service.get_connector_secret_row", fetch), \
                patch("services.connector_credential_service.set_connector_status", set_status):
            await injector.resolve("user-1")
            await injector.mark_invalid("user-1")
            await injector.resolve("user-1")

        set_status.assert_awaited_once_with("user-1", "earthdata", "error")
        self.assertEqual(fetch.call_count, 2)  # the second resolve refetched, not cached

    async def test_a_failing_status_write_is_swallowed_not_raised(self):
        from services.connector_credential_service import EdlCredentialInjector

        injector = EdlCredentialInjector(_settings_with_key())
        set_status = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("services.connector_credential_service.set_connector_status", set_status):
            await injector.mark_invalid("user-1")  # must not raise

        set_status.assert_awaited()


if __name__ == "__main__":
    unittest.main()
