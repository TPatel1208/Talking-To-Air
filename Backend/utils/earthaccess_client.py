"""
Lazy EarthAccess authentication helpers.

Importing this module does not authenticate. The first call to
get_earthaccess_auth() performs environment-based login, and later calls reuse
that session unless force=True is requested.
"""

from __future__ import annotations

import logging
import threading

import earthaccess

logger = logging.getLogger(__name__)

_auth = None
_auth_lock = threading.Lock()


def get_earthaccess_auth(force: bool = False):
    global _auth
    if _auth is not None and not force:
        return _auth

    with _auth_lock:
        if _auth is not None and not force:
            return _auth
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
