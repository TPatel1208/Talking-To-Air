"""T51: the cache benchmark harness.

T52 (the L4 Zarr cube) cannot have its chunking designed without numbers that
did not exist -- specifically, whether the AOI crop's slice pushes down to an
h5netcdf hyperslab read or whether dask materializes the whole single chunk
(``chunks={}`` -> one chunk per variable per file) and slices in memory. That
is a measurement, not an argument, and this harness is what takes it.

These tests run the harness end-to-end against a fixture bundle on a tiny
synthetic grid: enough to prove the harness reports what it claims to and
that case 3 skips cleanly until T52 lands, without pretending a 4x4 grid
says anything about production timings.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# A small AOI on a continental grid -- the shape T50 exists for, and the only
# shape where cropping can show anything at all.
SMALL_AOI = "-74.9,39.1,-73.1,40.9"


def _write_bundle(directory: str, members: int = 2) -> str:
    """A zip of NetCDF granule subsets on a coarse continental grid, in the
    shape :func:`services.open_handle._open_netcdf_bundle` expects."""
    import numpy as np
    import xarray as xr

    lats = np.arange(20.0, 55.01, 1.0)
    lons = np.arange(-125.0, -60.0 + 0.01, 1.0)
    bundle = os.path.join(directory, "bundle.zip")
    with zipfile.ZipFile(bundle, "w") as zf:
        for day in range(1, members + 1):
            values = np.full((1, len(lats), len(lons)), float(day), dtype="float32")
            ds = xr.Dataset(
                {"no2": (("time", "lat", "lon"), values, {"units": "mol m-2"})},
                coords={
                    "time": [np.datetime64(f"2024-07-{day:02d}T00:00:00")],
                    "lat": ("lat", lats, {"units": "degrees_north"}),
                    "lon": ("lon", lons, {"units": "degrees_east"}),
                },
                attrs={"ShortName": "TEST_NO2_L3"},
            )
            member = os.path.join(directory, f"granule_{day:02d}.nc")
            ds.to_netcdf(member)
            zf.write(member, arcname=f"granule_{day:02d}.nc")
    return bundle


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in ("xarray", "rasterio", "shapely")),
    "benchmark harness dependencies are not installed",
)
class BenchPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.bundle = _write_bundle(self._tmpdir.name)

    def _report(self, **kwargs):
        import bench_pipeline

        return bench_pipeline.run_benchmark(
            self.bundle, variable="no2", aoi=SMALL_AOI, runs=kwargs.pop("runs", 2), **kwargs
        )

    def test_the_baseline_and_crop_cases_both_produce_a_timed_result(self):
        report = self._report()

        baseline = report.case(1)
        cropped = report.case(2)
        for result in (baseline, cropped):
            self.assertIsNone(result.skipped_reason, f"case {result.number} unexpectedly skipped")
            self.assertGreater(result.median_seconds, 0.0)

    def test_the_crop_case_reduces_over_fewer_cells_than_the_baseline(self):
        """The A/B's whole premise: case 2 does the same reduction over a
        smaller array. If this ever stops holding, the timings are comparing
        something other than what the table claims."""
        report = self._report()

        self.assertLess(report.case(2).cells_reduced, report.case(1).cells_reduced)

    def test_the_two_cases_agree_on_the_answer(self):
        """The harness's own guard. T50's claim is that cropping moves no
        number; a benchmark that silently compared two different answers
        would be measuring the wrong thing fast."""
        report = self._report()

        self.assertAlmostEqual(report.case(1).value, report.case(2).value, places=10)

    def test_case_three_runs_against_a_real_cube_and_agrees_on_the_answer(self):
        """T52 landed, so case 3 is no longer a stub: it cubes the opened
        Dataset through the shipped writer and reduces over what a cache *hit*
        serves. Agreeing with cases 1 and 2 to full precision is the guard that
        matters -- a cache that changed a scientific result would be worse than
        no cache, and a benchmark that didn't check would report it as a win."""
        report = self._report()

        case3 = report.case(3)
        self.assertIsNone(case3.skipped_reason, f"case 3 unexpectedly skipped: {case3.skipped_reason}")
        self.assertGreater(case3.median_seconds, 0.0)
        self.assertAlmostEqual(report.case(1).value, case3.value, places=10)

    def test_case_three_reduces_over_the_same_cells_as_the_crop_case(self):
        """The cube is a faithful mirror, so cropping it must select the same
        window cropping the uncubed open selects. A different cell count would
        mean the round-trip moved the grid."""
        report = self._report()

        self.assertEqual(report.case(3).cells_reduced, report.case(2).cells_reduced)

    def test_the_report_names_the_product_and_grid_it_ran_against(self):
        """Story 6: a result nobody can reproduce, against a product nobody
        can name, is an anecdote."""
        report = self._report()

        self.assertEqual(report.members, 2)
        self.assertIn("TEST_NO2_L3", report.products)
        self.assertEqual(report.grid_shape[-2:], (36, 66))
        self.assertEqual(report.variable, "no2")

    def test_the_report_records_bytes_read_alongside_duration(self):
        """Story 3: bytes-read is what separates an I/O-bound phase from a
        CPU-bound one. It is unavailable off Linux, and reported as such
        rather than as a fabricated zero."""
        report = self._report()

        for number in (1, 2):
            read = report.case(number).bytes_read
            self.assertTrue(read is None or read >= 0)

    def test_the_rendered_table_shows_every_case(self):
        import bench_pipeline

        table = bench_pipeline.format_report(self._report())

        self.assertIn("case", table)
        for fragment in ("baseline", "crop-before-mask", "T52"):
            self.assertIn(fragment, table)

    def test_a_variable_that_is_not_in_the_bundle_fails_with_the_names_that_are(self):
        import bench_pipeline

        with self.assertRaises(SystemExit) as ctx:
            bench_pipeline.run_benchmark(self.bundle, variable="nope", aoi=SMALL_AOI, runs=1)

        self.assertIn("no2", str(ctx.exception))

    def test_an_unknown_aoi_names_the_two_accepted_forms(self):
        import bench_pipeline

        with self.assertRaises(SystemExit) as ctx:
            bench_pipeline.run_benchmark(self.bundle, variable="no2", aoi="Sasquatch County", runs=1)

        self.assertIn("minx,miny,maxx,maxy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
