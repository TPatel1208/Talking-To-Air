"""
tests/test_export_service.py
==============================
PRD T20 gap fix: export_service.py's timeseries CSV/PNG export unconditionally
routed every "timeseries" chart's export through AggregationService.to_dataarray,
which assumes an xarray Dataset/DataArray. point_timeseries's export.source_handles
points at a Parquet handle (an Arrow Table via services.open_handle), so exporting
one of its charts raised AttributeError instead of producing a CSV/PNG.

Prior art: test_open_handle.py (Parquet fixtures via HandleVolume),
test_point_timeseries.py (the point-sample export payload shape).
"""
import importlib.util
import os
import sys
import tempfile
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "pyarrow"]


def _make_point_series_table():
    import pyarrow as pa

    table = pa.table({
        "time": ["2024-01-02", "2024-01-01"],
        "no2": [2.0, 1.0],
    })
    return table.replace_schema_metadata({b"units": b"mol/m^2"})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "export_service point-sample test dependencies are not installed",
)
class PointSampleTimeseriesExportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)
        self.volume.add_parquet("obs_pt_ts_1", _make_point_series_table)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.tools = await load_raw_mcp_tools(settings)

        self.payload = {
            "type": "timeseries",
            "title": "no2 at Newark, NJ",
            "export": {
                "type": "timeseries",
                "variable": "no2",
                "units": "mol/m^2",
                "region_name": "Newark, NJ",
                "aggregation": "point sample",
                "chart_parameters": {"chart_type": "timeseries", "location": "Newark, NJ"},
                "source_handles": ["obs_pt_ts_1"],
            },
        }

    async def test_timeseries_rows_reads_the_parquet_table_directly(self):
        from services.export_service import ExportService

        service = ExportService()
        rows = await service._timeseries_rows_async(self.payload["export"], self.tools)

        self.assertEqual(rows, [
            ["no2", "2024-01-01T00:00:00", "point sample", 1.0, "mol/m^2"],
            ["no2", "2024-01-02T00:00:00", "point sample", 2.0, "mol/m^2"],
        ])

    async def test_csv_rows_export_a_point_sample_timeseries_without_crashing(self):
        from services.export_service import ExportService

        service = ExportService()
        rows = [row async for row in service.iter_chart_csv_rows_async(self.payload, self.tools)]

        self.assertEqual(rows[0], ["variable", "time", "stat", "value", "units"])
        self.assertEqual(rows[1:], [
            ["no2", "2024-01-01T00:00:00", "point sample", 1.0, "mol/m^2"],
            ["no2", "2024-01-02T00:00:00", "point sample", 2.0, "mol/m^2"],
        ])

    async def test_csv_chunks_export_a_point_sample_timeseries_without_crashing(self):
        from services.export_service import ExportService

        service = ExportService()
        chunks = [
            chunk async for chunk in service.iter_chart_csv_chunks_async(self.payload, self.tools)
        ]
        csv_text = b"".join(chunks).decode("utf-8")

        self.assertIn("no2,2024-01-01T00:00:00,point sample,1.0,mol/m^2", csv_text)
        self.assertIn("no2,2024-01-02T00:00:00,point sample,2.0,mol/m^2", csv_text)

    async def test_png_export_renders_a_point_sample_timeseries_without_crashing(self):
        from services.export_service import ExportService

        service = ExportService()
        png_bytes = await service.build_chart_png_async(self.payload, self.tools)

        self.assertGreater(len(png_bytes), 0)
        self.assertTrue(png_bytes.startswith(b"\x89PNG"))

    async def test_timeseries_rows_raises_a_clear_error_with_no_source_handle(self):
        from services.export_service import ExportService

        service = ExportService()
        export = dict(self.payload["export"])
        export["source_handles"] = []

        with self.assertRaises(ValueError):
            await service._timeseries_rows_async(export, self.tools)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES)
    or importlib.util.find_spec("xarray") is None
    or (importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None),
    "export_service multi-variable NetCDF test dependencies are not installed",
)
class MultiVariableExportResolutionTests(unittest.IsolatedAsyncioTestCase):
    """T25 review #3: the export path calls to_dataarray with the payload's
    stored ``variable`` and no handle. With the silent-first-variable fallback
    deleted, a stored payload whose ``variable`` is None/empty over a multi-
    variable file used to raise a structured MCPToolError -- uncaught on the
    CSV export path, where the streaming response has already started. The
    export path now threads the source handle (so a recorded choice or a
    lone science-vs-flag pair resolves) and, failing that, surfaces a clean
    ValueError instead of letting the MCPToolError escape mid-stream."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.tools = await load_raw_mcp_tools(settings)

    def _add_science_plus_flag_handle(self, handle):
        import xarray as xr

        def make_root():
            return xr.Dataset()

        def make_product_group():
            ds = xr.Dataset(
                {
                    "vertical_column_troposphere": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
                    "main_data_quality_flag": (("lat", "lon"), [[0, 0], [0, 0]]),
                },
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )
            return ds

        self.volume.add_netcdf(handle, {None: make_root, "product": make_product_group})

    def _add_two_science_var_handle(self, handle):
        import xarray as xr

        def make_root():
            return xr.Dataset()

        def make_product_group():
            return xr.Dataset(
                {
                    "no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
                    "hcho": (("lat", "lon"), [[5.0, 6.0], [7.0, 8.0]]),
                },
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_netcdf(handle, {None: make_root, "product": make_product_group})

    async def test_export_with_variable_none_resolves_the_science_var_over_a_science_plus_flag_file(self):
        from services.export_service import ExportService

        self._add_science_plus_flag_handle("obs_multi")
        export = {"variable": None, "units": "mol/m^2", "source_handles": ["obs_multi"]}

        da = await ExportService()._export_data_array_async(export, self.tools, collapse_to_2d=False)

        self.assertEqual(da.name, "vertical_column_troposphere")

    async def test_export_with_variable_none_fails_cleanly_over_a_multi_science_var_file(self):
        """No recorded choice and two genuine science variables: it can't be
        resolved, but it must surface as the export path's own ValueError
        (a clean 422) rather than an MCPToolError escaping mid-stream."""
        from services.export_service import ExportService

        self._add_two_science_var_handle("obs_ambiguous")
        export = {"variable": None, "units": "mol/m^2", "source_handles": ["obs_ambiguous"]}

        with self.assertRaises(ValueError):
            await ExportService()._export_data_array_async(export, self.tools, collapse_to_2d=False)

    async def test_export_inherits_a_recorded_choice_via_the_source_handle(self):
        """The handle is threaded through, so a choice recorded at retrieval
        time resolves the export even when the payload's ``variable`` is None."""
        from services import variable_choice_registry
        from services.export_service import ExportService

        self._add_two_science_var_handle("obs_recorded")
        variable_choice_registry._choices.clear()
        variable_choice_registry._choices["obs_recorded"] = ("hcho", float("inf"))
        self.addCleanup(variable_choice_registry._choices.clear)

        export = {"variable": None, "units": "mol/m^2", "source_handles": ["obs_recorded"]}

        da = await ExportService()._export_data_array_async(export, self.tools, collapse_to_2d=False)

        self.assertEqual(da.name, "hcho")


if __name__ == "__main__":
    unittest.main()
