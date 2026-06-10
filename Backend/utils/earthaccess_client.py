"""
Lazy EarthAccess authentication helpers.

Importing this module does not authenticate. The first call to
get_earthaccess_auth() performs environment-based login, and later calls reuse
that session unless force=True is requested.
"""

from __future__ import annotations

import logging
import os
import threading

import earthaccess
from config.settings import get_settings

logger = logging.getLogger(__name__)

_auth = None
_auth_lock = threading.Lock()


def ensure_earthdata_environment_from_edl() -> None:
    """
    Let libraries that only understand EARTHDATA_* env names use EDL settings.

    The application-level configuration is EDL_USERNAME / EDL_PASSWORD. Some
    NASA clients, including earthaccess paths, still look for
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD when using environment login. Copy
    the values into process-local environment variables when the EARTHDATA_*
    names are absent, so users do not have to configure duplicate secrets.
    """
    settings = get_settings()
    if settings.edl_username and not os.getenv("EARTHDATA_USERNAME"):
        os.environ["EARTHDATA_USERNAME"] = settings.edl_username
    if settings.edl_password and not os.getenv("EARTHDATA_PASSWORD"):
        os.environ["EARTHDATA_PASSWORD"] = settings.edl_password


def get_earthaccess_auth(force: bool = False):
    global _auth
    if _auth is not None and not force:
        return _auth

    with _auth_lock:
        if _auth is not None and not force:
            return _auth
        ensure_earthdata_environment_from_edl()
        try:
            auth = earthaccess.login(strategy="environment", force=force)
        except TypeError as exc:
            if "force" not in str(exc):
                raise
            logger.debug("earthaccess.login does not support force=; retrying without it")
            auth = earthaccess.login(strategy="environment")
        if not auth:
            raise RuntimeError("earthaccess login returned no credentials")
        _auth = auth
        logger.info("EarthAccess authenticated")
        return _auth


def reset_earthaccess_auth() -> None:
    """Test helper for clearing the cached EarthAccess session."""
    global _auth
    with _auth_lock:
        _auth = None
