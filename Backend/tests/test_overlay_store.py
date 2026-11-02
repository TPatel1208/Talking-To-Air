"""The map-overlay PNG store.

Overlays are server-rendered PNGs served only through the authenticated
``/chart/{id}/overlay.png`` route. The store is bounded by a fixed byte cap and
evicts least-recently-read entries first. A missing overlay is a normal state:
the chart payload still carries the grid the frontend's canvas fallback draws.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
import unittest.mock


def png(seed: int, size: int = 4096) -> bytes:
    """Distinct PNG-looking bytes of a known size."""
    header = b"\x89PNG\r\n\x1a\n"
    body = seed.to_bytes(4, "big") * (size // 4)
    return header + body[: size - len(header)]


class OverlayStoreTestCase(unittest.TestCase):
    """Points the overlay store at a throwaway directory for each test."""

    def setUp(self):
        from tta_backend.config.settings import get_settings

        self._store = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._store.cleanup)
        self.store_root = os.path.join(self._store.name, "overlays")

        self.set_env(OVERLAY_STORE_DIR=self.store_root)
        self.addCleanup(get_settings.cache_clear)

    def set_env(self, **env):
        from tta_backend.config.settings import get_settings

        patcher = unittest.mock.patch.dict(
            os.environ, {k: str(v) for k, v in env.items()}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()


class RoundTripTests(OverlayStoreTestCase):
    def test_a_written_overlay_reads_back_byte_for_byte(self):
        from tta_backend.services.overlay_store import read_overlay, write_overlay

        stored = write_overlay(png(1))

        self.assertEqual(read_overlay(stored), png(1))


class PathResolutionTests(OverlayStoreTestCase):
    """A chart payload records the overlay's path at render time. The store
    resolves it by file name inside the store as currently configured."""

    def test_a_path_recorded_under_another_store_directory_still_reads(self):
        from tta_backend.services.overlay_store import read_overlay, write_overlay

        stored = write_overlay(png(1))
        recorded = "/app/overlay_store/overlays/" + os.path.basename(stored)

        self.assertEqual(read_overlay(recorded), png(1))

    def test_a_path_outside_the_store_is_never_read(self):
        from tta_backend.services.overlay_store import read_overlay, write_overlay

        write_overlay(png(1))  # the store exists and is not empty
        outside = os.path.join(self._store.name, "elsewhere.png")
        with open(outside, "wb") as f:
            f.write(png(2))

        self.assertIsNone(read_overlay(outside))
        self.assertIsNone(read_overlay(os.path.join(self.store_root, "..", "elsewhere.png")))


class SweepTests(OverlayStoreTestCase):
    def test_a_write_a_crash_cut_short_is_never_served_and_is_swept_at_startup(self):
        """Writes are staged and renamed into place, so a process killed
        mid-write leaves a staging file rather than a truncated overlay. A
        truncated PNG would otherwise be served as if it were whole."""
        from tta_backend.services.overlay_store import (
            read_overlay, sweep_store, write_overlay,
        )

        kept = write_overlay(png(1))
        staging = os.path.join(self.store_root, "staging-" + "a" * 32 + ".png")
        with open(staging, "wb") as f:
            f.write(png(2)[:100])

        self.assertIsNone(read_overlay(staging))

        sweep_store()

        self.assertFalse(os.path.exists(staging))
        self.assertEqual(read_overlay(kept), png(1))

    def test_sweeping_a_store_that_does_not_exist_yet_is_harmless(self):
        from tta_backend.services.overlay_store import sweep_store

        sweep_store()

        self.assertFalse(os.path.exists(self.store_root))


class StartupSweepWiringTests(unittest.TestCase):
    """A source-level check, the same kind test_extract_cache_bound uses for
    the other on-disk stores: a sweep that exists but is never called reclaims
    nothing, and a boot test would need the whole lifespan."""

    def test_lifespan_sweeps_the_overlay_store(self):
        here = os.path.dirname(os.path.abspath(__file__))
        api_py = os.path.normpath(os.path.join(here, "..", "tta_backend", "api.py"))
        with open(api_py, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertTrue("frame_store.sweep_store(" in source, "this test's premise is stale")
        self.assertTrue(
            "overlay_store.sweep_store(" in source,
            "lifespan does not sweep the overlay store, so staging files left "
            "by an interrupted write are never removed",
        )


class EvictionTests(OverlayStoreTestCase):
    def test_a_write_past_the_cap_evicts_the_least_recently_read_overlay(self):
        from tta_backend.services.overlay_store import (
            read_overlay, store_size_bytes, write_overlay,
        )

        cold = write_overlay(png(1))
        warm = write_overlay(png(2))
        read_overlay(warm)  # reading counts as use, so "cold" is the coldest

        # Room for the two resident overlays and not a third.
        self.set_env(OVERLAY_STORE_MAX_BYTES=store_size_bytes() + 1024)
        new = write_overlay(png(3))

        self.assertIsNone(read_overlay(cold))
        self.assertEqual(read_overlay(warm), png(2))
        self.assertEqual(read_overlay(new), png(3))

    def test_the_cap_is_a_fixed_1_gib_and_eviction_holds_the_store_under_it(self):
        """A fixed byte count, never a share of free space, which would expand
        to fill whatever disk the volume is on. Checked through eviction as
        well as the setting, since a cap nothing enforces is only a comment."""
        from tta_backend.config.settings import get_settings
        from tta_backend.services.overlay_store import store_size_bytes, write_overlay

        self.assertEqual(get_settings().overlay_store_max_bytes, 1024 ** 3)

        budget = 3 * 4096
        self.set_env(OVERLAY_STORE_MAX_BYTES=budget)
        for seed in range(10):
            write_overlay(png(seed))

        self.assertLessEqual(store_size_bytes(), budget)
        self.assertGreater(store_size_bytes(), 0)


RENDER_MODULES = ["affine", "matplotlib", "numpy", "rasterio"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in RENDER_MODULES),
    "overlay rendering dependencies are not installed",
)
class RenderPathTests(OverlayStoreTestCase):
    def render(self, offset: float):
        import numpy as np

        from tta_backend.tools.satellite_tools import plot_tools
        from tta_backend.utils.colormaps import resolve

        lats = np.linspace(30.0, 33.0, 8)
        lons = np.linspace(-100.0, -96.0, 10)
        values = np.linspace(offset, offset + 1.0, lats.size * lons.size).reshape(lats.size, lons.size)
        fields = plot_tools._render_and_store_overlay(lats, lons, values, resolve("NO2").lut, 0.0, 2.0)
        self.assertIn("_path", fields, "the overlay render failed, so this proves nothing")
        return fields["_path"]

    def test_rendered_overlays_are_held_to_the_cap(self):
        from tta_backend.services.overlay_store import read_overlay, store_size_bytes

        first = self.render(0.0)
        self.assertIsNotNone(read_overlay(first))

        # Room for one overlay of this size, so the next render must evict.
        self.set_env(OVERLAY_STORE_MAX_BYTES=store_size_bytes() * 3 // 2)
        second = self.render(1.0)

        self.assertIsNone(read_overlay(first))
        self.assertIsNotNone(read_overlay(second))

    def test_a_render_stores_its_value_raster_and_the_url_says_so(self):
        import numpy as np

        from tta_backend.services.overlay_store import read_overlay
        from tta_backend.tools.satellite_tools import plot_tools
        from tta_backend.utils.colormaps import resolve

        lats = np.linspace(30.0, 33.0, 8)
        lons = np.linspace(-100.0, -96.0, 10)
        values = np.linspace(0.5, 1.5, lats.size * lons.size).reshape(lats.size, lons.size)
        fields = plot_tools._render_and_store_overlay(lats, lons, values, resolve("NO2").lut, 0.0, 2.0)

        self.assertIsNotNone(read_overlay(fields["_values"]["_path"]))
        # The raster holds the reprojected grid, whose bilinear resampling can
        # only pull the extremes inward.
        self.assertTrue(0.5 <= fields["_values"]["lo"] < 0.51, fields["_values"])
        self.assertTrue(1.49 < fields["_values"]["hi"] <= 1.5, fields["_values"])

        payload = {"chart_id": "c1", "type": "heatmap", "overlay": fields}
        # A chart drawn before rasters were stored.
        old_payload = {"chart_id": "c2", "type": "heatmap", "overlay": {"_path": fields["_path"]}}
        plot_tools._wire_overlay_urls(payload)
        plot_tools._wire_overlay_urls(old_payload)

        self.assertTrue(payload["overlay"]["rescalable"])
        self.assertEqual(old_payload["overlay"]["url"], "/chart/c2/overlay.png")
        self.assertNotIn("rescalable", old_payload["overlay"])


class ChartOverlayPathsTests(unittest.TestCase):
    """Every overlay a chart payload records, whatever its shape. Missing one
    here means deleting the chart leaves that PNG on disk."""

    def test_a_single_map_names_its_overlay(self):
        from tta_backend.services.overlay_store import chart_overlay_paths

        payload = {"type": "heatmap", "overlay": {"_path": "/s/a.png", "url": "/chart/x/overlay.png"}}

        self.assertEqual(chart_overlay_paths(payload), ["/s/a.png"])

    def test_a_comparison_names_every_panel_and_the_difference(self):
        from tta_backend.services.overlay_store import chart_overlay_paths

        payload = {
            "type": "heatmap_multi",
            "panels": [{"overlay": {"_path": "/s/p0.png"}}, {"overlay": {"_path": "/s/p1.png"}}],
            "difference": {"overlay": {"_path": "/s/d.png"}},
        }

        self.assertEqual(sorted(chart_overlay_paths(payload)), ["/s/d.png", "/s/p0.png", "/s/p1.png"])

    def test_value_rasters_are_named_beside_their_overlays(self):
        from tta_backend.services.overlay_store import chart_overlay_paths

        single = {"type": "heatmap", "overlay": {"_path": "/s/a.png", "_values": {"_path": "/s/a-v.png"}}}
        comparison = {
            "type": "heatmap_multi",
            "panels": [{"overlay": {"_path": "/s/p0.png", "_values": {"_path": "/s/p0-v.png"}}}],
            "difference": {"overlay": {"_path": "/s/d.png", "_values": {"_path": "/s/d-v.png"}}},
        }

        self.assertEqual(sorted(chart_overlay_paths(single)), ["/s/a-v.png", "/s/a.png"])
        self.assertEqual(
            sorted(chart_overlay_paths(comparison)), ["/s/d-v.png", "/s/d.png", "/s/p0-v.png", "/s/p0.png"],
        )

    def test_charts_without_a_rendered_overlay_name_nothing(self):
        from tta_backend.services.overlay_store import chart_overlay_paths

        for payload in (
            {"type": "timeseries"},
            {"type": "heatmap", "overlay": {}},
            {"type": "heatmap", "overlay": None},
            {"type": "heatmap_multi", "panels": [None, {"overlay": {}}], "difference": None},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(chart_overlay_paths(payload), [])


class DeleteOverlaysTests(OverlayStoreTestCase):
    def test_deletes_the_named_overlays_and_leaves_the_rest(self):
        from tta_backend.services.overlay_store import delete_overlays, read_overlay, write_overlay

        doomed = [write_overlay(png(1)), write_overlay(png(2))]
        kept = write_overlay(png(3))

        self.assertEqual(delete_overlays(doomed), 2)

        for path in doomed:
            self.assertIsNone(read_overlay(path))
        self.assertEqual(read_overlay(kept), png(3))

    def test_an_already_missing_overlay_is_not_an_error(self):
        from tta_backend.services.overlay_store import delete_overlays, write_overlay

        path = write_overlay(png(1))
        delete_overlays([path])

        self.assertEqual(delete_overlays([path]), 0)

    def test_never_deletes_a_file_that_is_not_a_store_entry(self):
        """Paths come from chart payloads in the database. Like reads, a
        delete resolves them by entry name inside the store, so nothing
        outside it can be named."""
        from tta_backend.services.overlay_store import delete_overlays

        os.makedirs(self.store_root, exist_ok=True)
        outside = os.path.join(self._store.name, "notes.png")
        inside_not_entry = os.path.join(self.store_root, "notes.png")
        for path in (outside, inside_not_entry):
            with open(path, "wb") as f:
                f.write(b"keep")

        self.assertEqual(delete_overlays([outside, inside_not_entry, "../notes.png", ""]), 0)

        self.assertTrue(os.path.exists(outside))
        self.assertTrue(os.path.exists(inside_not_entry))


if __name__ == "__main__":
    unittest.main()
