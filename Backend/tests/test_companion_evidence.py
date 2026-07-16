"""
tests/test_companion_evidence.py
==================================
PRD T36 Phase 2 -- deterministic companion evidence. ``_evidence`` reads the
quality/context bands sitting unused beside the plotted science variable in the
same opened Dataset and summarizes them as co-located facts (QA pass rate,
retrieval uncertainty, cloud fraction, aerosol index), each carrying an honest
``coverage`` valid-fraction. These tests exercise the computation directly on
synthetic TEMPO-O3 / TEMPO-NO2 / MODIS-AOD-shaped Datasets:

- a rich file yields context means + a QA-pass-rate fact, none for the science
  var or unclassified bands;
- fill values over part of the region push ``coverage`` below 1.0 and are
  excluded from the stat (guards the valid-pct-after-null-stripping trap);
- a science-only file yields ``[]`` (no invented companions -- MODIS AOD);
- the QA-pass-rate fact reuses the SAME flag var/good values masking resolves;
- ``_provenance`` attaches ``evidence`` additively, leaving ``masking`` /
  ``related_variables`` intact.
"""
import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REQUIRED_MODULES = [
    "numpy", "xarray", "shapely", "rasterio", "cartopy", "affine", "matplotlib",
]


def _global_region():
    from shapely.geometry import box

    return {"name": "global", "geometry": box(-180, -90, 180, 90), "bounds": (-180, -90, 180, 90)}


def _dataset(xr, data_vars, lat=(10.0, 20.0), lon=(30.0, 40.0), attrs=None):
    return xr.Dataset(
        data_vars,
        coords={"lat": list(lat), "lon": list(lon)},
        attrs=attrs or {},
    )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "companion-evidence test dependencies are not installed",
)
class EvidenceComputationTests(unittest.TestCase):
    def _fact(self, facts, name):
        for f in facts:
            if f["name"] == name:
                return f
        return None

    def test_tempo_o3_shaped_file_yields_context_and_qa_facts(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            "radiative_cloud_frac": (("lat", "lon"), [[0.0, 0.1], [0.2, 0.1]], {"units": "1"}),
            "uv_aerosol_index": (("lat", "lon"), [[1.0, 2.0], [3.0, 2.0]], {"units": "1"}),
            "main_data_quality_flag": (("lat", "lon"), [[0, 1], [0, 0]]),
        })
        da = ds["column_amount_o3"]
        col_info = {
            "short_name": "TEMPO_O3TOT_L3",
            "primary_var": "column_amount_o3",
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
        }
        facts = _evidence(ds, da, col_info, _global_region())
        names = {f["name"] for f in facts}

        # Context bands present as mean facts; QA flag present as a pass-rate fact.
        self.assertIn("radiative_cloud_frac", names)
        self.assertIn("uv_aerosol_index", names)
        self.assertIn("main_data_quality_flag", names)
        # Never the science variable itself.
        self.assertNotIn("column_amount_o3", names)

        cloud = self._fact(facts, "radiative_cloud_frac")
        self.assertEqual(cloud["role"], "context")
        self.assertEqual(cloud["stat"], "mean")
        self.assertAlmostEqual(cloud["value"], 0.1)  # mean of 0.0,0.1,0.2,0.1
        self.assertEqual(cloud["units"], "1")
        self.assertAlmostEqual(cloud["coverage"], 1.0)

        qa = self._fact(facts, "main_data_quality_flag")
        self.assertEqual(qa["role"], "quality")
        self.assertEqual(qa["stat"], "pass_rate")
        # good=[0]: 3 of 4 pixels flag==0 -> 0.75 pass rate, full coverage.
        self.assertAlmostEqual(qa["value"], 0.75)
        self.assertAlmostEqual(qa["coverage"], 1.0)

    def test_context_band_fill_lowers_coverage_and_is_excluded_from_the_mean(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            # Two of four cells are fill (-9999); the honest mean is over the
            # two real cells (0.1, 0.3) -> 0.2, and coverage is 0.5.
            "radiative_cloud_frac": (
                ("lat", "lon"), [[0.1, -9999.0], [0.3, -9999.0]], {"units": "1", "_FillValue": -9999.0},
            ),
        })
        da = ds["column_amount_o3"]
        col_info = {"short_name": "TEMPO_O3TOT_L3", "primary_var": "column_amount_o3"}
        facts = _evidence(ds, da, col_info, _global_region())

        cloud = self._fact(facts, "radiative_cloud_frac")
        self.assertIsNotNone(cloud)
        self.assertAlmostEqual(cloud["value"], 0.2)   # fill excluded, not averaged in
        self.assertLess(cloud["coverage"], 1.0)
        self.assertAlmostEqual(cloud["coverage"], 0.5)

    def test_modis_aod_shaped_file_yields_no_evidence(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "AOD": (("lat", "lon"), [[0.1, 0.2], [0.3, 0.4]], {"units": "1"}),
        }, attrs={"short_name": "MODIS_AOD"})
        da = ds["AOD"]
        facts = _evidence(ds, da, {"short_name": "MODIS_AOD", "primary_var": "AOD"}, _global_region())
        self.assertEqual(facts, [])

    def test_unclassified_band_is_not_invented_as_evidence(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            # No positive evidence for any role -> unclassified, never evidence.
            "processing_version": (("lat", "lon"), [[1.0, 1.0], [1.0, 1.0]], {}),
        })
        da = ds["column_amount_o3"]
        col_info = {"short_name": "TEMPO_O3TOT_L3", "primary_var": "column_amount_o3"}
        facts = _evidence(ds, da, col_info, _global_region())
        self.assertEqual([f["name"] for f in facts], [])

    def test_tempo_no2_shaped_file_yields_qa_and_uncertainty_but_no_invented_context(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "vertical_column_troposphere": (
                ("lat", "lon"), [[2.0, 4.0], [6.0, 8.0]], {"units": "molecules/cm^2"},
            ),
            "vertical_column_troposphere_uncertainty": (
                ("lat", "lon"), [[0.2, 0.4], [0.6, 0.8]], {"units": "molecules/cm^2"},
            ),
            "main_data_quality_flag": (("lat", "lon"), [[0, 0], [0, 0]]),
        })
        da = ds["vertical_column_troposphere"]
        col_info = {
            "short_name": "TEMPO_NO2_L3",
            "primary_var": "vertical_column_troposphere",
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
        }
        facts = _evidence(ds, da, col_info, _global_region())
        by_role = {f["name"]: f["role"] for f in facts}

        self.assertEqual(by_role.get("main_data_quality_flag"), "quality")
        # Uncertainty band summarized as a quality mean, with pct-of-science.
        unc = self._fact(facts, "vertical_column_troposphere_uncertainty")
        self.assertIsNotNone(unc)
        self.assertEqual(unc["role"], "quality")
        self.assertAlmostEqual(unc["value"], 0.5)          # mean of 0.2,0.4,0.6,0.8
        self.assertIn("pct_of_science", unc)
        self.assertAlmostEqual(unc["pct_of_science"], 0.1)  # 0.5 / 5.0 (science mean)
        # No context band exists -> none invented.
        self.assertFalse(any(f["role"] == "context" for f in facts))

    def test_qa_pass_rate_reuses_the_same_flag_and_good_values_masking_resolves(self):
        """The pass-rate fact must be computed with the exact flag var and good
        values ``resolve_and_mask`` masks with -- not a divergent second QA
        path. Compare directly against the shared resolution."""
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence, _aggregation_service
        from datasets.qa_flags import resolve_qa_info

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            "main_data_quality_flag": (("lat", "lon"), [[0, 1], [1, 1]]),
        })
        da = ds["column_amount_o3"]
        col_info = {
            "short_name": "TEMPO_O3TOT_L3",
            "primary_var": "column_amount_o3",
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
        }

        # The masking pipeline's own resolution, for comparison.
        qf_var, flag_attrs = _aggregation_service._resolve_qa_flag_var(ds, da, col_info)
        qa_col_info, _ = resolve_qa_info(yaml_info=col_info, flag_attrs=flag_attrs)
        self.assertEqual(qf_var, "main_data_quality_flag")
        self.assertEqual(qa_col_info["qa_good_values"], [0])

        facts = _evidence(ds, da, col_info, _global_region())
        qa = self._fact(facts, "main_data_quality_flag")
        # good=[0]: 1 of 4 pixels passes -> 0.25, over the same flag var.
        self.assertAlmostEqual(qa["value"], 0.25)

    def test_provenance_attaches_evidence_additively(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _provenance

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            "radiative_cloud_frac": (("lat", "lon"), [[0.1, 0.2], [0.3, 0.4]], {"units": "1"}),
            "main_data_quality_flag": (("lat", "lon"), [[0, 0], [0, 1]]),
        })
        da = ds["column_amount_o3"]
        col_info = {
            "short_name": "TEMPO_O3TOT_L3",
            "primary_var": "column_amount_o3",
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
        }
        agg_meta = {
            "aggregation_label": "Single Snapshot Mean, 1 daily granule",
            "n_granules": 1,
            "cadence": "daily",
            "granule_dates": ["2024-01-01"],
            "masking": {"qa_status": "verified", "qa_source": "collections_yaml"},
        }
        prov = _provenance(
            ["obs_1"], da, "global", "single snapshot",
            agg_meta=agg_meta, col_info=col_info, ds=ds, region=_global_region(),
        )

        # Additive: evidence present, and the pre-existing keys are undisturbed.
        self.assertIn("evidence", prov)
        self.assertTrue(prov["evidence"])
        self.assertEqual(prov["masking"]["qa_status"], "verified")
        self.assertIn("related_variables", prov)
        self.assertEqual(prov["related_variables"]["role"], "science")

    def test_evidence_is_empty_and_safe_without_a_source_dataset(self):
        import xarray as xr
        from tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
        })
        da = ds["column_amount_o3"]
        # No Dataset / no region -> empty, never an exception.
        self.assertEqual(_evidence(None, da, {}, _global_region()), [])
        self.assertEqual(_evidence(ds, da, {}, None), [])


if __name__ == "__main__":
    unittest.main()
