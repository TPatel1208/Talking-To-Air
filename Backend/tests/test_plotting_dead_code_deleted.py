"""T45: plot_diff_maps' "symmetric" diverging scale hardcodes vmin=0, which
would clip every negative difference to the bottom color — a live landmine.
It's unreachable from the tool surface today (the compare tool renders via
_da_to_heatmap_payload, not this), and grep confirms its only caller is
RegionResolver.plot_multiple, which itself has no caller either. Following
the test_legacy_loader_deleted.py precedent (dead code with a bug is still a
bug): delete both rather than fix the bug in unreachable code.

plot_map is NOT part of this deletion — export_service.py's PNG export path
still calls it directly, so it (and RegionResolver.plot_singular, which
wraps it) stays.
"""
import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["cartopy", "shapely", "rasterio"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "plotting dead-code test dependencies are not installed",
)
class PlotDiffMapsDeletedTests(unittest.TestCase):
    def test_plot_diff_maps_is_deleted(self):
        import utils.plotting as plotting

        self.assertFalse(hasattr(plotting, "plot_diff_maps"))

    def test_region_resolver_plot_multiple_is_deleted(self):
        from utils.plotting import RegionResolver

        self.assertFalse(hasattr(RegionResolver, "plot_multiple"))

    def test_plot_map_and_plot_singular_survive(self):
        """plot_map stays live via export_service.py's PNG export path."""
        import utils.plotting as plotting
        from utils.plotting import RegionResolver

        self.assertTrue(hasattr(plotting, "plot_map"))
        self.assertTrue(hasattr(RegionResolver, "plot_singular"))


if __name__ == "__main__":
    unittest.main()
