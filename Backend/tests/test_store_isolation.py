"""Hermeticity guard for the overlay store.

Importing ``plot_tools`` or ``api`` must not create ``Backend/overlay_store/``
or ``Backend/outputs/`` inside the checkout. That is the cube-store
non-hermeticity again (:func:`cache_isolation.isolate_cube_store`), with two
extra twists:

  * the suite writes into directories a developer also uses, and that Docker
    backs with named volumes; and
  * a stray directory would be untracked, so the polluted state survives a
    branch switch, and an *empty* one never shows up in ``git status``.

These tests assert the *property* — no test-suite write lands inside the
checkout — rather than the mechanism, so they keep failing if a redirect is
later moved, renamed, or dropped from ``conftest.py``. The deployment paths are
covered separately by ``test_overlay_store_persistence.py``, which reads the
real values through the ``deployment_*_dir`` helpers precisely because this
isolation has taken the live settings away from it.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

from tta_backend import APP_ROOT

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from cache_isolation import (  # noqa: E402 -- needs the TESTS_DIR insert above
    deployment_overlay_store_dir,
    isolate_overlay_store,
)

# Where the old APP_ROOT-relative constants landed. Nothing in the suite may
# create or write into either of these, under any runner.
CHECKOUT_OVERLAY_STORE = os.path.join(APP_ROOT, "overlay_store")
CHECKOUT_OUTPUTS = os.path.join(APP_ROOT, "outputs")
IN_CHECKOUT_STORES = (CHECKOUT_OVERLAY_STORE, CHECKOUT_OUTPUTS)

# `APP_ROOT` is a source checkout on a developer machine and `/app` in the
# container — and in the *runtime* image `/app/overlay_store` legitimately
# exists, because the Dockerfile pre-creates it so its named volume inherits
# appuser's ownership. "This directory must not exist" is
# therefore a statement about a checkout, not about every environment; asserting
# it unconditionally would either fail in a runtime image or pass only by the
# accident that `backend-test` builds the `builder` target, which does not run
# that mkdir.
#
# So the strict form is scoped by an explicit marker rather than left to chance,
# and the property asserted everywhere is the one that is true everywhere: this
# run neither created these directories nor wrote anything into them.
IS_SOURCE_CHECKOUT = os.path.exists(os.path.join(APP_ROOT, os.pardir, ".git"))

RENDER_MODULES = ["affine", "matplotlib", "numpy", "rasterio"]


def _is_inside(path: str, root: str) -> bool:
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


def _store_state(path: str) -> tuple[bool, tuple[str, ...]]:
    """Existence plus contents, so a test can prove it neither created nor wrote.

    Absent and empty are deliberately distinguishable: an *empty* directory is
    invisible to ``git status``.
    """
    if not os.path.exists(path):
        return (False, ())
    return (True, tuple(sorted(os.listdir(path))))


# Captured at import, which is after conftest has installed the redirects — so
# in a clean checkout both of these are (False, ()).
_INITIAL_STORE_STATE = {path: _store_state(path) for path in IN_CHECKOUT_STORES}


class StoreStateAssertions(unittest.TestCase):
    """Shared assertions for "the suite did not touch this directory"."""

    def assertSurvivesIsolation(self, deployment: str, isolated: str) -> None:
        """A ``deployment_*_dir()`` helper still reports the container's path.

        Deliberately *not* "the deployment path is outside APP_ROOT". That reads
        correctly on a developer machine and is flatly wrong in the container,
        where APP_ROOT is ``/app`` and the deployment path —
        ``/app/overlay_store/overlays`` — is supposed to live inside it.

        No ``os.path.isabs`` check either: the deployment value is a POSIX
        container path, and ``ntpath.isabs('/app/overlay_store')`` is False on the
        Windows development host, so that would test the host's path parser
        rather than the deployment value.

        What matters, and holds in both environments, is that the helper did not
        hand back isolation's tempdir — the whole failure mode it exists to
        prevent, because a deployment-contract test fed the tempdir would assert
        the volume contract against the harness's own value and pass while
        checking nothing.
        """
        self.assertNotEqual(os.path.abspath(deployment), os.path.abspath(isolated))
        self.assertFalse(
            _is_inside(deployment, isolated),
            f"{deployment!r} is inside the isolation tempdir {isolated!r}",
        )

    def assertUntouched(self, path: str) -> None:
        self.assertEqual(
            _store_state(path),
            _INITIAL_STORE_STATE[path],
            f"{path!r} was created or written to during this run — the suite is "
            "polluting a directory the developer and the Docker named volume "
            "also use",
        )
        if IS_SOURCE_CHECKOUT:
            self.assertFalse(
                os.path.exists(path),
                f"{path!r} exists inside a source checkout. Being gitignored is "
                "what makes this bad rather than harmless: the state survives a "
                "branch switch and never shows up in `git status`.",
            )


class OverlayStoreIsolationTests(StoreStateAssertions):
    def test_the_overlay_store_resolves_outside_the_checkout(self) -> None:
        from tta_backend.tools.satellite_tools import plot_tools

        store = plot_tools.overlay_store_dir()

        self.assertFalse(
            _is_inside(store, APP_ROOT),
            f"the overlay store resolves to {store!r}, inside the checkout — the "
            "suite is writing into a directory the developer and the Docker named "
            "volume also use, and because it is gitignored the pollution survives "
            "a branch switch invisibly",
        )

    def test_the_overlay_store_is_never_the_deployment_volume(self) -> None:
        """The overlay store evicts, so a test that writes past its cap deletes
        the coldest overlays. Pointed at the deployment volume, that would
        degrade charts nobody touched. Checked separately from the checkout
        test because on a Windows host the deployment path resolves outside
        the checkout."""
        from tta_backend.tools.satellite_tools import plot_tools

        self.assertNotEqual(
            os.path.abspath(plot_tools.overlay_store_dir()),
            os.path.abspath(deployment_overlay_store_dir()),
            "the overlay store resolves to the deployment volume during tests",
        )

    def test_importing_plot_tools_does_not_create_the_checkout_store(self) -> None:
        """Import must be free of filesystem side effects.

        Asserting on import alone is the sharp version of this test: the
        directory appeared with no test having rendered anything, which is why a
        fixture that cleaned up after each *test* would never have caught it.
        """
        importlib.import_module("tta_backend.tools.satellite_tools.plot_tools")

        self.assertUntouched(CHECKOUT_OVERLAY_STORE)

    @unittest.skipIf(
        any(importlib.util.find_spec(name) is None for name in RENDER_MODULES),
        "overlay rendering dependencies are not installed",
    )
    def test_storing_an_overlay_writes_outside_the_checkout(self) -> None:
        import numpy as np

        from tta_backend.tools.satellite_tools import plot_tools
        from tta_backend.utils.colormaps import resolve

        lats = np.linspace(30.0, 33.0, 8)
        lons = np.linspace(-100.0, -96.0, 10)
        values = np.linspace(0.0, 1.0, lats.size * lons.size).reshape(lats.size, lons.size)

        path = plot_tools._render_and_store_overlay(
            lats, lons, values, resolve("NO2").lut, 0.0, 1.0
        )

        self.assertIsNotNone(path, "the overlay render failed, so this proves nothing")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(
            _is_inside(path, APP_ROOT), f"a stored overlay landed at {path!r}, in the checkout"
        )
        self.assertUntouched(CHECKOUT_OVERLAY_STORE)

    def test_the_deployment_path_survives_the_isolation(self) -> None:
        self.assertSurvivesIsolation(
            deployment_overlay_store_dir(), isolate_overlay_store()
        )

    def test_isolate_overlay_store_is_idempotent(self) -> None:
        """conftest calls it once, but ``ProcessCacheIsolation.setUp`` calls it
        again per test; two calls must not strand a second tempdir or move the
        store out from under something already holding a path."""
        first = isolate_overlay_store()
        second = isolate_overlay_store()

        self.assertEqual(first, second)
        self.assertEqual(os.environ["OVERLAY_STORE_DIR"], first)

    def test_the_store_is_configurable_through_settings(self) -> None:
        import tempfile
        import unittest.mock

        from tta_backend.config.settings import get_settings
        from tta_backend.tools.satellite_tools import plot_tools

        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"OVERLAY_STORE_DIR": tmp}):
                get_settings.cache_clear()
                self.addCleanup(get_settings.cache_clear)

                self.assertEqual(
                    os.path.abspath(plot_tools.overlay_store_dir()), os.path.abspath(tmp)
                )


class OutputsDirRemovedTests(StoreStateAssertions):
    """``/outputs`` was a public static mount that nothing wrote to. Importing
    the API must not recreate the directory or the route."""

    def test_importing_the_api_does_not_create_the_checkout_outputs(self) -> None:
        importlib.import_module("tta_backend.api")

        self.assertUntouched(CHECKOUT_OUTPUTS)

    def test_the_api_serves_nothing_at_outputs(self) -> None:
        api = importlib.import_module("tta_backend.api")

        paths = [getattr(route, "path", "") for route in api.app.routes]
        self.assertFalse(
            [path for path in paths if path == "/outputs" or path.startswith("/outputs/")],
            f"a route is mounted under /outputs: {paths}",
        )


if __name__ == "__main__":
    unittest.main()
