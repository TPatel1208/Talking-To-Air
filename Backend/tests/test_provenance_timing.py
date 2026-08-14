"""Timing for the provenance build, the plot path's last unmeasured stretch.

Two live traces on 2026-08-07 left a hole that no phase owned: 107s of a 372s
turn (18 granules), then 303s of an 870s turn (36 granules) -- 34.8%, and more
than the retrieval wait's share of the second turn. ``llm_call``/``agent_step``
narrowed it to the window between the ``render`` timer closing and the plot
tool returning, which contains exactly one thing: ``_attach_reproducibility``.

The doubling with granule count is what these pin down. ``_evidence`` and
``_related_variables`` both read the *unaggregated* Dataset -- every granule,
every band -- rather than the reduced array the chart was drawn from, and only
their split says which one to change. So the counts matter as much as the
durations: a duration alone cannot separate "reads too many bands" from "reads
each band over too much data".
"""
import importlib.util
import unittest


REQUIRED_MODULES = [
    "numpy", "xarray", "shapely", "rasterio", "cartopy", "affine", "matplotlib",
]


def _phase_samples(phase: str) -> tuple[float, float]:
    """(count, sum) currently recorded for ``phase``. Histograms are
    process-wide state, so every test below asserts on a delta."""
    from tta_backend.utils.metrics import PIPELINE_PHASE_DURATION_SECONDS

    count = total = 0.0
    for metric in PIPELINE_PHASE_DURATION_SECONDS.collect():
        for sample in metric.samples:
            if sample.labels.get("phase") != phase:
                continue
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return count, total


def _global_region():
    from shapely.geometry import box

    return {"name": "global", "geometry": box(-180, -90, 180, 90), "bounds": (-180, -90, 180, 90)}


def _tempo_shaped_dataset(xr):
    """A TEMPO-O3-shaped file: one science var, two context bands, a QA flag.
    The same shape tests/test_companion_evidence.py exercises, so the timing
    assertions here describe the path those tests already pin the meaning of."""
    return xr.Dataset(
        {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            "radiative_cloud_frac": (("lat", "lon"), [[0.0, 0.1], [0.2, 0.1]], {"units": "1"}),
            "uv_aerosol_index": (("lat", "lon"), [[1.0, 2.0], [3.0, 2.0]], {"units": "1"}),
            "main_data_quality_flag": (("lat", "lon"), [[0, 1], [0, 0]]),
        },
        coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
    )


_COL_INFO = {
    "short_name": "TEMPO_O3TOT_L3",
    "primary_var": "column_amount_o3",
    "quality_flag_var": "main_data_quality_flag",
    "qa_good_values": [0],
}


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "provenance-timing test dependencies are not installed",
)
class EvidencePhaseTests(unittest.TestCase):
    def test_computing_evidence_records_an_evidence_phase(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _tempo_shaped_dataset(xr)
        before = _phase_samples("evidence")
        _evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())

        self.assertEqual(_phase_samples("evidence")[0] - before[0], 1)

    def test_the_evidence_phase_reports_how_many_bands_it_read(self):
        """The number that turns a duration into a per-band cost, and so
        decides whether the fix is fewer bands or less data per band."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _tempo_shaped_dataset(xr)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            facts = _evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())

        record = next(r for r in captured.records if r._phase == "evidence")
        self.assertEqual(record._bands_read, len(facts))
        self.assertGreater(record._bands_read, 0)
        self.assertGreaterEqual(record._bands_classified, record._bands_read)

    def test_the_evidence_phase_reports_the_cells_it_read(self):
        """The whole finding is that this reads the unaggregated Dataset. Cells
        read is what makes that visible next to the science array's own size --
        and what would show a fix having landed."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _tempo_shaped_dataset(xr)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            _evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())

        record = next(r for r in captured.records if r._phase == "evidence")
        self.assertEqual(record._science_cells, 4)
        # Two context bands of four cells each are read beside the science var.
        self.assertEqual(record._band_cells, 8)

    def test_a_file_with_no_companions_still_records_the_phase(self):
        """A MODIS-AOD-shaped file yields no facts. The phase must still be
        recorded, or the histogram silently describes only rich files and its
        percentiles stop meaning "what a plot pays for provenance"."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = xr.Dataset(
            {"AOD": (("lat", "lon"), [[0.1, 0.2], [0.3, 0.4]], {"units": "1"})},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )
        before = _phase_samples("evidence")
        facts = _evidence(ds, ds["AOD"], {"short_name": "MODIS_AOD", "primary_var": "AOD"}, _global_region())

        self.assertEqual(facts, [])
        self.assertEqual(_phase_samples("evidence")[0] - before[0], 1)

    def test_an_absent_dataset_records_nothing(self):
        """The early return does no work; timing it would put a near-zero
        sample in the histogram and drag the percentiles toward a case that
        never touches a band."""
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        before = _phase_samples("evidence")
        self.assertEqual(_evidence(None, None, {}, _global_region()), [])

        self.assertEqual(_phase_samples("evidence")[0] - before[0], 0)

    def test_the_evidence_facts_are_unchanged_by_the_timing_split(self):
        """The body moved into a helper so the timer could wrap it. That is a
        refactor, and this is the guard that it stayed one."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _tempo_shaped_dataset(xr)
        facts = _evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())
        names = {f["name"] for f in facts}

        self.assertIn("radiative_cloud_frac", names)
        self.assertIn("uv_aerosol_index", names)
        self.assertNotIn("column_amount_o3", names)
        self.assertNotIn("main_data_quality_flag", names)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "provenance-timing test dependencies are not installed",
)
@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "provenance-timing test dependencies are not installed",
)
class DeferredScienceMeanTests(unittest.TestCase):
    """The 226s fix.

    A live trace (TEMPO NO2, 36 granules, 2026-08-07) classified 2 bands, read
    0 and produced 0 facts, yet spent 226s -- 26% of an 870s turn -- computing
    a science mean that nothing then read. The mean is only consumed by
    ``pct_of_science`` on an *uncertainty* band, and it is expensive because
    the aggregation result stays dask-backed, so each pass re-runs the graph.
    """

    def _counting_area_weighted_mean(self):
        """Wraps the real function, counting calls. The waste being fixed is
        measured in *passes over a lazy array*, so calls are the honest unit --
        a wall-clock assertion here would just measure the machine."""
        from tta_backend.preprocessing import aggregation_service

        real = aggregation_service.area_weighted_mean
        calls = []

        def counting(da):
            calls.append(da)
            return real(da)

        return counting, calls

    def test_a_file_with_no_uncertainty_band_never_computes_the_science_mean(self):
        """The measured case. Nothing reads the mean, so nothing should pay
        for it -- this is the whole 226s."""
        from unittest.mock import patch

        import xarray as xr
        from tta_backend.tools.satellite_tools import plot_tools

        ds = _tempo_shaped_dataset(xr)
        counting, calls = self._counting_area_weighted_mean()
        with patch.object(plot_tools, "area_weighted_mean", counting):
            plot_tools._evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())

        # The context bands legitimately take their own means -- that is the
        # facts being produced. What must NOT happen is a pass over the science
        # variable, which is the value nothing here reads.
        science_passes = [da for da in calls if getattr(da, "name", None) == "column_amount_o3"]
        self.assertEqual(
            science_passes, [],
            "the science mean was computed with no uncertainty band to read it",
        )

    def test_the_deferral_is_visible_in_the_phase_log(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _tempo_shaped_dataset(xr)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as captured:
            _evidence(ds, ds["column_amount_o3"], _COL_INFO, _global_region())

        record = next(r for r in captured.records if r._phase == "evidence")
        self.assertFalse(record._science_mean_computed)

    def test_an_uncertainty_band_still_gets_its_pct_of_science(self):
        """Deferring must not mean dropping: the fact that DOES need the mean
        must still carry it, or the fix has silently removed a disclosure."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = xr.Dataset(
            {
                "vertical_column_troposphere": (
                    ("lat", "lon"), [[100.0, 200.0], [300.0, 400.0]], {"units": "mol/m^2"},
                ),
                "vertical_column_troposphere_uncertainty": (
                    ("lat", "lon"), [[10.0, 20.0], [30.0, 40.0]], {"units": "mol/m^2"},
                ),
            },
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )
        facts = _evidence(
            ds, ds["vertical_column_troposphere"],
            {"short_name": "TEMPO_NO2_L3", "primary_var": "vertical_column_troposphere"},
            _global_region(),
        )

        uncertainty = [f for f in facts if "uncertainty" in f["name"]]
        self.assertTrue(uncertainty, "the uncertainty band produced no fact")
        self.assertIn("pct_of_science", uncertainty[0])
        self.assertIsNotNone(uncertainty[0]["pct_of_science"])

    def test_the_science_mean_is_computed_at_most_once(self):
        """Memoized, not merely deferred: two uncertainty bands must not mean
        two passes over a lazy array."""
        from tta_backend.tools.satellite_tools.plot_tools import _DeferredScienceMean

        import xarray as xr

        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )
        deferred = _DeferredScienceMean(da)
        first, second = deferred(), deferred()

        self.assertEqual(first, second)
        self.assertTrue(deferred.computed)

    def test_an_all_nan_field_yields_none_rather_than_raising(self):
        """``area_weighted_mean`` raises when nothing is finite. The previous
        code pre-checked to avoid that; this one catches it. Same contract --
        evidence is additive and must never cost the chart."""
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _DeferredScienceMean

        da = xr.DataArray(
            [[np.nan, np.nan], [np.nan, np.nan]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        self.assertIsNone(_DeferredScienceMean(da)())

    def test_a_lazy_array_is_materialized_once_before_the_weighted_mean(self):
        """``area_weighted_mean`` makes two passes internally. Against a dask
        graph those are two full re-executions back to the source granules,
        which is what made this cost 226s; it must receive materialized data."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _DeferredScienceMean

        try:
            import dask  # noqa: F401
        except ImportError:
            self.skipTest("dask is not installed")

        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        ).chunk({"lat": 1})
        self.assertIsNotNone(da.chunks, "test precondition: the array must be lazy")

        seen = {}

        def spy(passed):
            seen["chunks"] = getattr(passed, "chunks", None)
            return 2.5

        deferred = _DeferredScienceMean(da)
        import tta_backend.tools.satellite_tools.plot_tools as pt
        from unittest.mock import patch

        with patch.object(pt, "area_weighted_mean", spy):
            deferred()

        self.assertIsNone(seen["chunks"], "area_weighted_mean received a still-lazy array")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "provenance-timing test dependencies are not installed",
)
class RelatedVariablesPhaseTests(unittest.TestCase):
    def test_building_related_variables_records_its_own_phase(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _related_variables

        ds = _tempo_shaped_dataset(xr)
        before = _phase_samples("related_variables")
        _related_variables(ds["column_amount_o3"], _COL_INFO, ds=ds)

        self.assertEqual(_phase_samples("related_variables")[0] - before[0], 1)

    def test_the_phase_records_whether_it_read_the_dataset_or_the_registry(self):
        """Two different costs behind one name: an inventory built from the
        opened file, or the registry's curated subset. Only the flag separates
        them, and only the Dataset path scales with the retrieval."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _related_variables

        ds = _tempo_shaped_dataset(xr)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as from_ds:
            _related_variables(ds["column_amount_o3"], _COL_INFO, ds=ds)
        with self.assertLogs("tta_backend.utils.phase_timing", level="INFO") as from_registry:
            _related_variables(ds["column_amount_o3"], _COL_INFO, ds=None)

        opened = next(r for r in from_ds.records if r._phase == "related_variables")
        curated = next(r for r in from_registry.records if r._phase == "related_variables")
        self.assertTrue(opened._from_dataset)
        self.assertFalse(curated._from_dataset)
        self.assertEqual(opened._variables, len(ds.data_vars))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "provenance-timing test dependencies are not installed",
)
class ProvenancePhaseTests(unittest.TestCase):
    """The outer span -- the one that maps directly onto the unattributed
    window in the live traces."""

    def test_attaching_reproducibility_records_a_provenance_phase(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        ds = _tempo_shaped_dataset(xr)
        before = _phase_samples("provenance")
        _attach_reproducibility(
            {}, ["obs_x"], ds["column_amount_o3"], "global", "single snapshot",
            col_info=_COL_INFO, region=_global_region(), ds=ds,
        )

        self.assertEqual(_phase_samples("provenance")[0] - before[0], 1)

    def test_the_provenance_phase_contains_the_evidence_phase(self):
        """Nesting is the point: provenance minus evidence minus
        related_variables is what remains unexplained, and that residual is the
        next thing to chase if the two known passes do not account for it."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        ds = _tempo_shaped_dataset(xr)
        outer = _phase_samples("provenance")
        evidence = _phase_samples("evidence")
        related = _phase_samples("related_variables")
        _attach_reproducibility(
            {}, ["obs_x"], ds["column_amount_o3"], "global", "single snapshot",
            col_info=_COL_INFO, region=_global_region(), ds=ds,
        )

        self.assertEqual(_phase_samples("provenance")[0] - outer[0], 1)
        self.assertEqual(_phase_samples("evidence")[0] - evidence[0], 1)
        self.assertEqual(_phase_samples("related_variables")[0] - related[0], 1)

    def test_the_payload_is_unchanged_by_the_timing(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _attach_reproducibility

        ds = _tempo_shaped_dataset(xr)
        payload = _attach_reproducibility(
            {"type": "heatmap"}, ["obs_x"], ds["column_amount_o3"], "global", "single snapshot",
            col_info=_COL_INFO, region=_global_region(), ds=ds,
        )

        self.assertIn("provenance", payload)
        self.assertIn("evidence", payload["provenance"])
        self.assertIn("related_variables", payload["provenance"])
        self.assertEqual(payload["metadata"]["source_handles"], ["obs_x"])


class PhaseVocabularyTests(unittest.TestCase):
    def test_the_provenance_phases_are_pre_declared_at_zero(self):
        from tta_backend.utils.metrics import render_prometheus_metrics

        rendered = render_prometheus_metrics().decode()

        for phase in ("provenance", "evidence", "related_variables"):
            self.assertIn(
                f'pipeline_phase_duration_seconds_count{{phase="{phase}"}}',
                rendered,
                f"phase '{phase}' has no pre-declared series",
            )


if __name__ == "__main__":
    unittest.main()
