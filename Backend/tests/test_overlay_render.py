import importlib.util
import unittest


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
        from tta_backend.utils.colormaps import resolve
        from tta_backend.utils.overlay_render import render_overlay_png

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
        from tta_backend.utils.colormaps import resolve
        from tta_backend.utils.overlay_render import render_overlay_png

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
        from tta_backend.utils.colormaps import resolve
        from tta_backend.utils.overlay_render import render_overlay_png

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


    def test_render_holds_a_bounded_multiple_of_the_grid_in_memory(self):
        """Rendering must not need an unbounded multiple of the grid it draws.

        The render stage OOM-killed the backend (2026-08-05, full-day TEMPO
        over North America): retrieval and materialization both succeeded, then
        uvicorn was SIGKILLed by the kernel with ~2.8 GB RSS while rasterizing.
        Measured cause: the rasterizer allocated 54 bytes per grid cell -- it
        upcast the caller's float32 grid to float64, reprojected into another
        float64 buffer, and colorized through six more full-size float64/int64
        temporaries. At 22.3M cells (TEMPO CONUS native) that is ~1.2 GB for
        one panel, on top of the aggregation that produced it.

        The ceiling, not the byte count, is the guarantee: a render may hold a
        handful of working buffers, never a dozen. Expressed per cell so it
        stays meaningful as grids grow -- which is the whole failure mode.
        """
        import tracemalloc
        import numpy as np
        from tta_backend.utils.colormaps import resolve
        from tta_backend.utils.overlay_render import render_overlay_png

        lut = resolve("NO2").lut

        def grid(nlat, nlon):
            lats = np.linspace(20.0, 55.0, nlat)
            lons = np.linspace(-130.0, -60.0, nlon)
            # float32 is what the plot path hands us; a satellite retrieval is
            # nowhere near float64-precision to begin with.
            values = np.linspace(0.0, 1.0, nlat * nlon, dtype=np.float32).reshape(nlat, nlon)
            return lats, lons, values

        # Warm up: matplotlib/rasterio import and their module-level caches
        # allocate once, and must not be charged to the measured render.
        render_overlay_png(*grid(40, 50), lut, vmin=0.0, vmax=1.0)

        lats, lons, values = grid(600, 800)
        cells = values.size

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            render_overlay_png(lats, lons, values, lut, vmin=0.0, vmax=1.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        bytes_per_cell = (peak - before) / cells
        self.assertLess(
            bytes_per_cell, 24.0,
            f"overlay render peaked at {bytes_per_cell:.1f} bytes per grid cell; "
            f"at TEMPO CONUS native resolution (22.3M cells) that is "
            f"{bytes_per_cell * 22.3e6 / 1e9:.2f} GB for a single panel",
        )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "overlay rendering dependencies are not installed",
)
class ValueRasterTests(unittest.TestCase):
    """The value raster stored beside each overlay, which the overlay route
    recolors at a scale chosen after the chart was drawn (compare mode's
    shared scale)."""

    lut = None

    def setUp(self):
        import numpy as np
        from tta_backend.utils.colormaps import resolve

        self.lut = resolve("NO2").lut
        self.lats = np.linspace(30.0, 33.0, 40)
        self.lons = np.linspace(-100.0, -96.0, 50)
        values = np.linspace(2.0, 9.0, self.lats.size * self.lons.size, dtype=np.float32)
        self.values = values.reshape(self.lats.size, self.lons.size)
        self.values[:5, :] = np.nan

    def decode(self, png_bytes):
        import io
        from PIL import Image
        import numpy as np

        with Image.open(io.BytesIO(png_bytes)) as image:
            return np.asarray(image.convert("RGBA"))

    def assert_same_picture(self, recolored, direct):
        """Identical transparency, and colors that differ by at most one LUT
        step: the raster rounds to 1/65534 of the range, which can only tip
        a value sitting on a LUT boundary into the neighbouring color."""
        import numpy as np

        a, b = self.decode(recolored), self.decode(direct)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a[..., 3] == 0, b[..., 3] == 0)
        lut = np.asarray(self.lut, dtype=np.int16)
        def lut_index(rgba):
            opaque = rgba[rgba[..., 3] > 0][:, :3].astype(np.int16)
            return np.abs(opaque[:, None, :] - lut[None, :, :3]).sum(axis=2).argmin(axis=1)
        self.assertLessEqual(int(np.abs(lut_index(a) - lut_index(b)).max()), 1)

    def test_recoloring_at_the_drawn_scale_reproduces_the_overlay(self):
        from tta_backend.utils.overlay_render import render_overlay, render_value_raster_png

        rendered = render_overlay(self.lats, self.lons, self.values, self.lut, 3.0, 8.0)
        raster = rendered.value_raster

        recolored = render_value_raster_png(raster.png, raster.lo, raster.hi, self.lut, 3.0, 8.0)

        self.assert_same_picture(recolored, rendered.png)

    def test_recoloring_at_another_scale_matches_drawing_at_that_scale(self):
        """The point of the raster: a shared scale wider than the panel's own
        clip, at full resolution, including the tails the first drawing
        saturated."""
        from tta_backend.utils.overlay_render import render_overlay, render_overlay_png, render_value_raster_png

        raster = render_overlay(self.lats, self.lons, self.values, self.lut, 3.0, 8.0).value_raster

        recolored = render_value_raster_png(raster.png, raster.lo, raster.hi, self.lut, 0.0, 12.0)
        direct = render_overlay_png(self.lats, self.lons, self.values, self.lut, 0.0, 12.0)

        self.assert_same_picture(recolored, direct)

    def test_the_raster_spans_the_grids_finite_range(self):
        import numpy as np
        from tta_backend.utils.overlay_render import encode_value_raster

        raster = encode_value_raster(np.array([[1.5, np.nan], [4.0, 2.0]], dtype=np.float32))

        self.assertEqual((raster.lo, raster.hi), (1.5, 4.0))

    def test_a_grid_with_no_finite_value_has_no_raster(self):
        import numpy as np
        from tta_backend.utils.overlay_render import encode_value_raster

        self.assertIsNone(encode_value_raster(np.full((3, 4), np.nan, dtype=np.float32)))

    def test_a_constant_grid_recolors_to_its_one_value(self):
        import numpy as np
        from tta_backend.utils.overlay_render import encode_value_raster, render_value_raster_png

        raster = encode_value_raster(np.full((3, 4), 5.0, dtype=np.float32))
        rgba = self.decode(render_value_raster_png(raster.png, raster.lo, raster.hi, self.lut, 0.0, 10.0))

        self.assertTrue(np.all(rgba == rgba[0, 0]))
        self.assertEqual(tuple(rgba[0, 0]), tuple(self.lut[128]))

    def test_recoloring_holds_a_bounded_multiple_of_the_raster_in_memory(self):
        """The route recolors on request, so it must stay as lean as the
        render it stands in for (see the 24 bytes/cell ceiling above)."""
        import tracemalloc
        import numpy as np
        from tta_backend.utils.overlay_render import encode_value_raster, render_value_raster_png

        values = np.linspace(0.0, 1.0, 600 * 800, dtype=np.float32).reshape(600, 800)
        raster = encode_value_raster(values)
        render_value_raster_png(raster.png, raster.lo, raster.hi, self.lut, 0.0, 1.0)  # warm up

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            render_value_raster_png(raster.png, raster.lo, raster.hi, self.lut, 0.0, 1.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        bytes_per_cell = (peak - before) / values.size
        self.assertLess(bytes_per_cell, 24.0, f"recolor peaked at {bytes_per_cell:.1f} bytes per cell")

    def test_rendering_with_the_raster_holds_the_same_memory_ceiling(self):
        """render_overlay is what the plot path calls; storing the raster
        must not push it past the ceiling render_overlay_png is held to."""
        import tracemalloc
        import numpy as np
        from tta_backend.utils.overlay_render import render_overlay

        lats = np.linspace(20.0, 55.0, 600)
        lons = np.linspace(-130.0, -60.0, 800)
        values = np.linspace(0.0, 1.0, 600 * 800, dtype=np.float32).reshape(600, 800)
        render_overlay(lats[:40], lons[:50], values[:40, :50], self.lut, 0.0, 1.0)  # warm up

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            render_overlay(lats, lons, values, self.lut, 0.0, 1.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        bytes_per_cell = (peak - before) / values.size
        self.assertLess(bytes_per_cell, 24.0, f"render peaked at {bytes_per_cell:.1f} bytes per cell")


if __name__ == "__main__":
    unittest.main()
