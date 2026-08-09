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
import unittest


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

    def test_tempo_o3_shaped_file_yields_context_facts_but_not_a_duplicate_qa_fact(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

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

        # Context bands present as mean facts.
        self.assertIn("radiative_cloud_frac", names)
        self.assertIn("uv_aerosol_index", names)
        # Never the science variable itself.
        self.assertNotIn("column_amount_o3", names)
        # T55: the QA pass rate is no longer an evidence fact. It is counted
        # where the mask is applied and reported once, as masking provenance --
        # a second computation here could legitimately disagree with it (this
        # path re-crops the flag band spatially, ignorant of any dim_selector
        # applied to the plotted array), and two disagreeing quality numbers on
        # one page are worse than one.
        self.assertNotIn("main_data_quality_flag", names)

        cloud = self._fact(facts, "radiative_cloud_frac")
        self.assertEqual(cloud["role"], "context")
        self.assertEqual(cloud["stat"], "mean")
        # cos-lat area-weighted mean of 0.0,0.1 (lat 10) and 0.2,0.1 (lat 20).
        self.assertAlmostEqual(cloud["value"], 0.098828, places=6)
        self.assertEqual(cloud["units"], "1")
        self.assertAlmostEqual(cloud["coverage"], 1.0)

    def test_context_band_mean_is_cos_latitude_area_weighted(self):
        """Finding #13: the companion band mean must be cos(latitude)
        area-weighted, the SAME regional-mean definition as the headline
        science mean -- an unweighted grid-cell mean over-weights poleward
        pixels. Over a band with 10.0 at the equator and 0.0 at 80N, the
        unweighted mean is 5.0; the area-weighted mean is ~8.52 (the equatorial
        cells carry ~5.8x the weight of the shrunken 80N cells)."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(
            xr,
            {
                "column_amount_o3": (("lat", "lon"), [[300.0, 300.0], [300.0, 300.0]], {"units": "DU"}),
                "radiative_cloud_frac": (("lat", "lon"), [[10.0, 10.0], [0.0, 0.0]], {"units": "1"}),
            },
            lat=(0.0, 80.0),
        )
        da = ds["column_amount_o3"]
        col_info = {"short_name": "TEMPO_O3TOT_L3", "primary_var": "column_amount_o3"}

        facts = _evidence(ds, da, col_info, _global_region())
        cloud = self._fact(facts, "radiative_cloud_frac")

        self.assertIsNotNone(cloud)
        self.assertAlmostEqual(cloud["value"], 8.52, places=2)

    def test_context_band_fill_lowers_coverage_and_is_excluded_from_the_mean(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

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
        # Fill excluded; area-weighted mean of the two real cells 0.1 (lat 10)
        # and 0.3 (lat 20) -- weighting nudges it just below the plain 0.2.
        self.assertAlmostEqual(cloud["value"], 0.197656, places=6)
        self.assertLess(cloud["coverage"], 1.0)
        self.assertAlmostEqual(cloud["coverage"], 0.5)

    def test_modis_aod_shaped_file_yields_no_evidence(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "AOD": (("lat", "lon"), [[0.1, 0.2], [0.3, 0.4]], {"units": "1"}),
        }, attrs={"short_name": "MODIS_AOD"})
        da = ds["AOD"]
        facts = _evidence(ds, da, {"short_name": "MODIS_AOD", "primary_var": "AOD"}, _global_region())
        self.assertEqual(facts, [])

    def test_unclassified_band_is_not_invented_as_evidence(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            # No positive evidence for any role -> unclassified, never evidence.
            "processing_version": (("lat", "lon"), [[1.0, 1.0], [1.0, 1.0]], {}),
        })
        da = ds["column_amount_o3"]
        col_info = {"short_name": "TEMPO_O3TOT_L3", "primary_var": "column_amount_o3"}
        facts = _evidence(ds, da, col_info, _global_region())
        self.assertEqual([f["name"] for f in facts], [])

    def test_tempo_no2_shaped_file_yields_uncertainty_but_no_invented_context(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

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

        # T55: the flag itself is excluded from the evidence list -- its pass
        # rate is reported once, as masking provenance.
        self.assertNotIn("main_data_quality_flag", by_role)
        # Uncertainty band summarized as a quality mean, with pct-of-science.
        unc = self._fact(facts, "vertical_column_troposphere_uncertainty")
        self.assertIsNotNone(unc)
        self.assertEqual(unc["role"], "quality")
        # Area-weighted mean of 0.2,0.4,0.6,0.8 over lat 10/20.
        self.assertAlmostEqual(unc["value"], 0.495311, places=6)
        self.assertIn("pct_of_science", unc)
        # Weighted unc / weighted science (4.953115) -- both weighted the same
        # way, so the ratio is exactly 0.1 (0.2:2, 0.4:4, ... uncertainty is a
        # flat 10% of science in every cell).
        self.assertAlmostEqual(unc["pct_of_science"], 0.1)
        # No context band exists -> none invented.
        self.assertFalse(any(f["role"] == "context" for f in facts))

    def test_group_stamped_band_classifies_by_its_group_prior(self):
        """open_handle stamps each merged variable's source group as a
        ``group_path`` attr (flat names would otherwise strand the group
        priors). A markerless band under geolocation/ must classify as
        context (High) in the evidence path and yield a mean fact — the same
        verdict the describe_dataset inventory reaches from its
        slash-qualified name."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
            # No name marker at all — only the stamped group says context.
            "pixel_weight": (
                ("lat", "lon"), [[0.5, 0.5], [0.5, 0.5]], {"units": "1", "group_path": "geolocation"},
            ),
        })
        da = ds["column_amount_o3"]
        col_info = {"short_name": "TEMPO_O3TOT_L3", "primary_var": "column_amount_o3"}
        facts = _evidence(ds, da, col_info, _global_region())

        weight = self._fact(facts, "pixel_weight")
        self.assertIsNotNone(weight)
        self.assertEqual(weight["role"], "context")

    def test_time_dimensioned_companion_band_is_reduced_over_the_plotted_window(self):
        """The plotted science field is a per-pixel time-mean 2-D map; an
        evidence fact must summarize the SAME time-reduced field, not the raw
        space-time stack. ``_crop_band_to_region`` only crops spatially, so a
        band that still carries a time dimension would be space-time-averaged
        and (for uncertainty) divided by the science *time-mean* -- two
        incommensurable aggregations. The two answers diverge whenever valid
        coverage varies across time, which is exactly this fixture."""
        import numpy as np
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        unc = np.array([
            [[2.0, 2.0], [2.0, 2.0]],             # t0: all four cells valid
            [[10.0, np.nan], [np.nan, np.nan]],   # t1: only one cell valid
        ])
        sci = np.array([
            [[5.0, 5.0], [5.0, 5.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ])
        ds = xr.Dataset(
            {
                "vertical_column_troposphere": (
                    ("time", "lat", "lon"), sci, {"units": "molecules/cm^2"},
                ),
                "vertical_column_troposphere_uncertainty": (
                    ("time", "lat", "lon"), unc, {"units": "molecules/cm^2"},
                ),
            },
            coords={"time": [0, 1], "lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )
        da = ds["vertical_column_troposphere"].mean("time")  # the plotted 2-D field
        col_info = {"short_name": "TEMPO_NO2_L3", "primary_var": "vertical_column_troposphere"}

        facts = _evidence(ds, da, col_info, _global_region())
        unc_fact = self._fact(facts, "vertical_column_troposphere_uncertainty")
        self.assertIsNotNone(unc_fact)
        # per-pixel time-mean: cell(0,0)=mean(2,10)=6, the rest=mean(2,NaN)=2;
        # cos-lat area-weighted space-mean over the region ~= 3.023 -- NOT the
        # raw stack mean of 18/5 = 3.6, and (with weighting) just off a flat 3.0.
        self.assertAlmostEqual(unc_fact["value"], 3.023443, places=6)
        # And pct-of-science divides that time-mean by the science time-mean 5.0.
        self.assertAlmostEqual(unc_fact["pct_of_science"], 0.6047, places=4)

    def test_zero_science_mean_keeps_the_uncertainty_fact_without_pct(self):
        """An anomaly/difference-style science field can legitimately average
        exactly 0.0 — the pct-of-science ratio is then undefined (division by
        zero), so the key is honestly omitted, but the uncertainty mean fact
        itself must survive (a naive presence-only guard would raise inside
        the best-effort except and silently lose the whole fact)."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            # Symmetric within each latitude row so the area-weighted mean is
            # exactly 0.0 regardless of the cos-lat weights (each row sums to 0).
            "vertical_column_troposphere": (
                ("lat", "lon"), [[-2.0, 2.0], [-4.0, 4.0]], {"units": "molecules/cm^2"},
            ),
            "vertical_column_troposphere_uncertainty": (
                ("lat", "lon"), [[0.2, 0.4], [0.6, 0.8]], {"units": "molecules/cm^2"},
            ),
        })
        da = ds["vertical_column_troposphere"]
        col_info = {"short_name": "TEMPO_NO2_L3", "primary_var": "vertical_column_troposphere"}
        facts = _evidence(ds, da, col_info, _global_region())

        unc = self._fact(facts, "vertical_column_troposphere_uncertainty")
        self.assertIsNotNone(unc)
        self.assertAlmostEqual(unc["value"], 0.495311, places=6)  # area-weighted
        self.assertNotIn("pct_of_science", unc)

    def test_provenance_attaches_evidence_additively(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _provenance

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

    def test_related_variables_reflect_the_opened_dataset_not_just_the_curated_list(self):
        """The registry's curated ``variables:`` subset can be far narrower
        than the retrieved file (TEMPO_NO2 curates 2 of its 38 real
        variables). related_variables and evidence must read the SAME source
        — the opened Dataset — so the chart page can never say "no
        companions" while showing companion-derived evidence facts below."""
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _provenance

        ds = _dataset(xr, {
            "vertical_column_troposphere": (
                ("lat", "lon"), [[2.0, 4.0], [6.0, 8.0]], {"units": "molecules/cm^2"},
            ),
            "vertical_column_troposphere_uncertainty": (
                ("lat", "lon"), [[0.2, 0.4], [0.6, 0.8]], {"units": "molecules/cm^2"},
            ),
            "eff_cloud_fraction": (("lat", "lon"), [[0.0, 0.1], [0.2, 0.1]], {"units": "1"}),
            "main_data_quality_flag": (("lat", "lon"), [[0, 0], [0, 0]]),
        })
        da = ds["vertical_column_troposphere"]
        col_info = {
            "short_name": "TEMPO_NO2_L3",
            "primary_var": "vertical_column_troposphere",
            "quality_flag_var": "main_data_quality_flag",
            "qa_good_values": [0],
            # The curated subset omits the uncertainty and context bands the
            # file actually carries — the panel must still see them.
            "variables": ["product/vertical_column_troposphere", "product/main_data_quality_flag"],
        }
        prov = _provenance(
            ["obs_1"], da, "global", "single snapshot",
            col_info=col_info, ds=ds, region=_global_region(),
        )

        rv = prov["related_variables"]
        self.assertEqual(rv["uncertainty_sibling"], "vertical_column_troposphere_uncertainty")
        self.assertIn("eff_cloud_fraction", rv["context_siblings"])
        # And the evidence section is grounded in the same bands.
        evidence_names = {f["name"] for f in prov["evidence"]}
        self.assertIn("vertical_column_troposphere_uncertainty", evidence_names)
        self.assertIn("eff_cloud_fraction", evidence_names)

    def test_related_variables_fall_back_to_the_curated_list_without_a_dataset(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _provenance

        ds = _dataset(xr, {
            "vertical_column": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "molecules/cm^2"}),
        })
        da = ds["vertical_column"]
        col_info = {
            "primary_var": "vertical_column",
            "quality_flag_var": "main_data_quality_flag",
            "variables": [
                "product/vertical_column", "product/vertical_column_uncertainty",
                "product/main_data_quality_flag",
            ],
        }
        prov = _provenance(["obs_1"], da, "global", "single snapshot", col_info=col_info)

        rv = prov["related_variables"]
        self.assertEqual(rv["uncertainty_sibling"], "vertical_column_uncertainty")
        self.assertEqual(rv["qa_sibling"], "main_data_quality_flag")

    def test_multi_panel_merge_drops_panel_specific_evidence_and_related_variables(self):
        """A compare chart's top-level provenance inherits panel 0's — but
        evidence facts (QA pass rate, cloud fraction) and related-variables
        are region-specific: presenting one panel's as the whole comparison's
        is misleading for exactly the trust judgment they exist to support.
        They stay per-panel; the merged view omits them."""
        from tta_backend.tools.satellite_tools.plot_tools import _merged_multi_provenance

        panels = [
            {"provenance": {
                "variable": "vertical_column_troposphere",
                "region_name": "New Jersey",
                "evidence": [{"name": "main_data_quality_flag", "value": 0.96}],
                "related_variables": {"role": "science"},
                "masking": {"qa_status": "verified"},
            }},
            {"provenance": {
                "variable": "vertical_column_troposphere",
                "region_name": "Los Angeles",
                "evidence": [{"name": "main_data_quality_flag", "value": 0.41}],
                "related_variables": {"role": "science"},
            }},
        ]
        merged = _merged_multi_provenance(panels)

        self.assertNotIn("evidence", merged)
        self.assertNotIn("related_variables", merged)
        self.assertEqual(merged["region_name"], "New Jersey, Los Angeles")
        self.assertEqual(merged["aggregation"], "single snapshot comparison")
        # Non-panel-specific keys still inherit from panel 0 (pre-existing shape).
        self.assertEqual(merged["variable"], "vertical_column_troposphere")
        self.assertEqual(merged["masking"]["qa_status"], "verified")
        # The panels themselves keep their own evidence untouched.
        self.assertEqual(panels[0]["provenance"]["evidence"][0]["value"], 0.96)
        self.assertEqual(panels[1]["provenance"]["evidence"][0]["value"], 0.41)

    def test_geometry_mask_is_the_single_rasterization_mask_data_applies(self):
        """geometry_mask is the one rasterization seam: applying it via
        ``where()`` must equal ``mask_data_by_geometry``, so the evidence
        crop can derive both its crop and its region-footprint count from a
        single rasterize call instead of rasterizing an all-ones twin."""
        import xarray as xr
        from shapely.geometry import box
        from tta_backend.utils.plotting import geometry_mask, mask_data_by_geometry

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
        })
        band = ds["column_amount_o3"]
        geometry = box(25.0, 5.0, 36.0, 16.0)  # covers only the lower-left cell

        mask = geometry_mask(band, geometry)
        # ``assert_equal``, not ``assert_identical``: the seam additionally
        # RECORDS what it rasterized (``region_cells``, and ``region_type``
        # when the mask self-heals), which a bare ``where()`` has no way to
        # know. The claim under test is that the two keep the same cells.
        uncropped = mask_data_by_geometry(band, geometry, crop=False)
        xr.testing.assert_equal(band.where(mask), uncropped)
        # With T50's crop the seam returns that same array narrowed to the
        # mask's footprint -- the same rasterization, fewer cells carried.
        cropped = mask_data_by_geometry(band, geometry)
        xr.testing.assert_equal(band.where(mask).sel(cropped.coords), cropped)
        # The footprint count is readable straight off the mask -- and the seam
        # reports the same number, so a caller needing the region's own cell
        # count (per-layer coverage, T56) never re-rasterizes on cropped axes,
        # whose float32 step differs in the 8th digit and flips boundary cells.
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(uncropped.attrs["region_cells"], 1)
        self.assertEqual(cropped.attrs["region_cells"], 1)

    def test_evidence_is_empty_and_safe_without_a_source_dataset(self):
        import xarray as xr
        from tta_backend.tools.satellite_tools.plot_tools import _evidence

        ds = _dataset(xr, {
            "column_amount_o3": (("lat", "lon"), [[300.0, 310.0], [320.0, 330.0]], {"units": "DU"}),
        })
        da = ds["column_amount_o3"]
        # No Dataset / no region -> empty, never an exception.
        self.assertEqual(_evidence(None, da, {}, _global_region()), [])
        self.assertEqual(_evidence(ds, da, {}, None), [])


if __name__ == "__main__":
    unittest.main()
