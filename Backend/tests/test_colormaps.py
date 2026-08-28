import unittest


from tta_backend.utils.colormaps import resolve


class ColormapRegistryTests(unittest.TestCase):
    def test_no2_resolves_to_the_omi_style_sequential_colormap(self):
        resolution = resolve("NO2")

        self.assertEqual(resolution.name, "no2_omi")
        self.assertTrue(resolution.lut)
        for stop in resolution.lut:
            self.assertEqual(len(stop), 4)
            for channel in stop:
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_diverging_fields_resolve_to_rdbu_r_regardless_of_variable(self):
        resolution = resolve("NO2", diverging=True)

        self.assertEqual(resolution.name, "RdBu_r")
        self.assertTrue(resolution.lut)

    def test_unrecognized_variable_falls_back_to_viridis_rather_than_erroring(self):
        resolution = resolve("SOME_UNMAPPED_VARIABLE")

        self.assertEqual(resolution.name, "viridis")
        self.assertTrue(resolution.lut)

    def test_missing_variable_falls_back_to_viridis_rather_than_erroring(self):
        resolution = resolve(None)

        self.assertEqual(resolution.name, "viridis")


class ColormapExportAntiDriftTests(unittest.TestCase):
    """Proves export_service renders with the same registry entry the
    payload ships as `colormap.lut`, so the map, the legend, and the export
    can never disagree on what a value looks like.

    Asserted against the render path the PNG export actually takes (api.py ->
    build_chart_png -> _render_png). A sync twin this once covered could not
    render at all -- it fetched through a stub that unconditionally raised --
    so the anti-drift guarantee was being proven on code no download ever ran.
    Since the render moved off the event loop the drawing half is synchronous
    and takes its array as an argument, so this no longer has to stub the
    fetch to reach it: what runs below is the download's own code.
    """

    def test_export_heatmap_axis_uses_the_same_registry_entry_as_the_payload(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr
        from tta_backend.services.export_service import ExportService

        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [10.0, 11.0], "lon": [-100.0, -99.0]},
        )
        export = {"variable": "NO2"}
        service = ExportService()
        fig, ax = plt.subplots()

        try:
            mesh = service._plot_heatmap_axis(ax, export, da, "Panel")
        finally:
            plt.close(fig)

        self.assertEqual(mesh.cmap.name, resolve("NO2").name)


if __name__ == "__main__":
    unittest.main()
