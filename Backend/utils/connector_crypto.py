"""T30: encryption for per-user connector secrets (e.g. pasted EDL tokens).

MultiFernet from day one -- CONNECTOR_ENCRYPTION_KEY is a comma-separated key
list (first key encrypts, all keys are tried on decrypt), so rotating the key
is "prepend a new one and redeploy", never a flag day that strands every
stored secret. Decryption only ever happens in-process; nothing here crosses
the API boundary.
"""
from __future__ import annotations

from cryptography.fernet import Fernet, MultiFernet


class ConnectorCryptoError(ValueError):
    """Raised when CONNECTOR_ENCRYPTION_KEY is set but malformed."""


def parse_encryption_keys(raw: str) -> list[bytes]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ConnectorCryptoError("CONNECTOR_ENCRYPTION_KEY is set but empty.")
    return [key.encode("utf-8") for key in keys]


def build_multi_fernet(raw: str) -> MultiFernet:
    try:
        return MultiFernet([Fernet(key) for key in parse_encryption_keys(raw)])
    except ConnectorCryptoError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConnectorCryptoError(f"CONNECTOR_ENCRYPTION_KEY is malformed: {exc}") from exc


def get_connector_cipher(settings) -> MultiFernet | None:
    """None means the feature is unconfigured on this deployment -- callers
    turn that into the structured 503, never an exception."""
    if not settings.connector_encryption_key:
        return None
    return build_multi_fernet(settings.connector_encryption_key)


def encrypt_secret(cipher: MultiFernet, secret: str) -> str:
    return cipher.encrypt(secret.encode("utf-8")).decode("utf-8")


def decrypt_secret(cipher: MultiFernet, token: str) -> str:
    return cipher.decrypt(token.encode("utf-8")).decode("utf-8")
