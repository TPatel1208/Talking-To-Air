import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class ConnectorCryptoTests(unittest.TestCase):
    def test_round_trips_a_secret_through_encrypt_and_decrypt(self):
        from cryptography.fernet import Fernet

        from utils.connector_crypto import build_multi_fernet, decrypt_secret, encrypt_secret

        key = Fernet.generate_key().decode()
        cipher = build_multi_fernet(key)

        token = encrypt_secret(cipher, "my-edl-token")

        self.assertNotIn("my-edl-token", token)
        self.assertEqual(decrypt_secret(cipher, token), "my-edl-token")

    def test_a_secret_encrypted_under_an_old_key_still_decrypts_after_rotation(self):
        from cryptography.fernet import Fernet

        from utils.connector_crypto import build_multi_fernet, decrypt_secret, encrypt_secret

        old_key = Fernet.generate_key().decode()
        new_key = Fernet.generate_key().decode()

        old_cipher = build_multi_fernet(old_key)
        stored = encrypt_secret(old_cipher, "pre-rotation-secret")

        # Rotation: prepend the new key, keep the old one for decrypt.
        rotated_cipher = build_multi_fernet(f"{new_key},{old_key}")

        self.assertEqual(decrypt_secret(rotated_cipher, stored), "pre-rotation-secret")

    def test_a_row_is_unreadable_without_any_matching_key(self):
        from cryptography.fernet import Fernet, InvalidToken

        from utils.connector_crypto import build_multi_fernet, decrypt_secret, encrypt_secret

        cipher = build_multi_fernet(Fernet.generate_key().decode())
        stored = encrypt_secret(cipher, "secret")

        wrong_cipher = build_multi_fernet(Fernet.generate_key().decode())
        with self.assertRaises(InvalidToken):
            decrypt_secret(wrong_cipher, stored)

    def test_build_multi_fernet_rejects_a_malformed_key(self):
        from utils.connector_crypto import ConnectorCryptoError, build_multi_fernet

        with self.assertRaises(ConnectorCryptoError):
            build_multi_fernet("not-a-valid-fernet-key")

    def test_build_multi_fernet_rejects_an_empty_string(self):
        from utils.connector_crypto import ConnectorCryptoError, build_multi_fernet

        with self.assertRaises(ConnectorCryptoError):
            build_multi_fernet("")

    def test_get_connector_cipher_returns_none_when_unset(self):
        from types import SimpleNamespace

        from utils.connector_crypto import get_connector_cipher

        self.assertIsNone(get_connector_cipher(SimpleNamespace(connector_encryption_key=None)))

    def test_get_connector_cipher_builds_a_cipher_when_set(self):
        from types import SimpleNamespace

        from cryptography.fernet import Fernet, MultiFernet

        from utils.connector_crypto import get_connector_cipher

        cipher = get_connector_cipher(SimpleNamespace(connector_encryption_key=Fernet.generate_key().decode()))
        self.assertIsInstance(cipher, MultiFernet)


class SettingsConnectorKeyValidationTests(unittest.TestCase):
    def test_validate_startup_passes_when_connector_encryption_key_is_unset(self):
        from config.settings import Settings

        loaded = Settings(
            db_password="x", jwt_secret_key="x", google_api_key="x", groq_api_key="x",
            connector_encryption_key=None,
        )
        loaded.validate_startup()  # must not raise

    def test_validate_startup_accepts_a_well_formed_connector_encryption_key(self):
        from cryptography.fernet import Fernet

        from config.settings import Settings

        loaded = Settings(
            db_password="x", jwt_secret_key="x", google_api_key="x", groq_api_key="x",
            connector_encryption_key=Fernet.generate_key().decode(),
        )
        loaded.validate_startup()  # must not raise

    def test_validate_startup_accepts_a_comma_separated_rotation_pair(self):
        from cryptography.fernet import Fernet

        from config.settings import Settings

        loaded = Settings(
            db_password="x", jwt_secret_key="x", google_api_key="x", groq_api_key="x",
            connector_encryption_key=f"{Fernet.generate_key().decode()},{Fernet.generate_key().decode()}",
        )
        loaded.validate_startup()  # must not raise

    def test_validate_startup_fails_loudly_on_a_malformed_connector_encryption_key(self):
        from config.settings import ConfigurationError, Settings

        loaded = Settings(
            db_password="x", jwt_secret_key="x", google_api_key="x", groq_api_key="x",
            connector_encryption_key="not-a-valid-fernet-key",
        )
        with self.assertRaisesRegex(ConfigurationError, "CONNECTOR_ENCRYPTION_KEY"):
            loaded.validate_startup()


if __name__ == "__main__":
    unittest.main()
