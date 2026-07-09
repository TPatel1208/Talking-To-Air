import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["affine", "matplotlib", "numpy", "rasterio"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "overlay rendering dependencies are not installed",
)
class OverlayRenderTests(unittest.TestCase):
    def test_renders_valid_png_bytes_for_an_in_range_grid(self):
        import io
        import numpy as np
        import matplotlib.image as mpimg
        from utils.colormaps import resolve
        from utils.overlay_render import render_overlay_png

        lats = np.linspace(30.0, 33.0, 12)
        lons = np.linspace(-100.0, -96.0, 16)
        values = np.linspace(0.0, 1.0, lats.size * lons.size).reshape(lats.size, lons.size)
        lut = resolve("NO2").lut

        png_bytes = render_overlay_png(lats, lons, values, lut, vmin=0.0, vmax=1.0)

        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        decoded = mpimg.imread(io.BytesIO(png_bytes), format="png")
        self.assertEqual(decoded.shape[-1], 4)
        self.assertGreater(decoded.shape[0] * decoded.shape[1], 0)
        alpha = decoded[..., 3]
        self.assertTrue(np.any(alpha > 0))

    def test_no_data_regions_stay_transparent(self):
        import io
        import numpy as np
        import matplotlib.image as mpimg
        from utils.colormaps import resolve
        from utils.overlay_render import render_overlay_png

        lats = np.linspace(30.0, 33.0, 12)
        lons = np.linspace(-100.0, -96.0, 16)
        values = np.full((lats.size, lons.size), 5.0)
        values[:, : lons.size // 2] = np.nan  # left half is no-data
        lut = resolve("NO2").lut

        png_bytes = render_overlay_png(lats, lons, values, lut, vmin=0.0, vmax=10.0)

        decoded = mpimg.imread(io.BytesIO(png_bytes), format="png")
        alpha = decoded[..., 3]
        width = alpha.shape[1]

        # Deep within the no-data half (away from the reprojected/resampled
        # boundary) every pixel must stay fully transparent -- the overlay
        # must never invent structure across a real gap.
        left_quarter = alpha[:, : width // 4]
        self.assertTrue(np.all(left_quarter == 0))

        # Deep within the valid half, pixels must be opaque.
        right_quarter = alpha[:, 3 * width // 4 :]
        self.assertTrue(np.all(right_quarter > 0))

    def test_extreme_values_map_to_the_colormaps_endpoint_colors(self):
        import io
        import numpy as np
        import matplotlib.image as mpimg
        from utils.colormaps import resolve
        from utils.overlay_render import render_overlay_png

        lats = np.linspace(30.0, 33.0, 12)
        lons = np.linspace(-100.0, -96.0, 16)
        lut = resolve("NO2").lut

        def center_rgba(value):
            values = np.full((lats.size, lons.size), value)
            png_bytes = render_overlay_png(lats, lons, values, lut, vmin=0.0, vmax=10.0)
            decoded = mpimg.imread(io.BytesIO(png_bytes), format="png")
            center = np.array(decoded.shape[:2]) // 2
            pixel = decoded[center[0], center[1]]
            return tuple((pixel * 255).round().astype(int))

        min_rgba = center_rgba(0.0)
        max_rgba = center_rgba(10.0)

        self.assertEqual(min_rgba, tuple(lut[0]))
        self.assertEqual(max_rgba, tuple(lut[-1]))
        self.assertNotEqual(min_rgba, max_rgba)


if __name__ == "__main__":
    unittest.main()
