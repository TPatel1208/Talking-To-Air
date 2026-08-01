import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class VariableResolverTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr

    def _var(self, values, attrs=None):
        return (("lat", "lon"), self.np.array(values, dtype=float), attrs or {})

    def test_resolve_picks_the_mean_field_over_its_standard_deviation_sibling(self):
        """Tracer: the statistical mean of a geophysical quantity IS the
        science field; its Standard_Deviation sibling is plumbing (category 1)
        and is never the pick. discover -> classify -> score -> rank -> decide,
        end to end."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "Terra_MODIS_NASADarkTarget_AOD_550/Mean": self._var([[0.1, 0.2]], {"units": "1"}),
            "Terra_MODIS_NASADarkTarget_AOD_550/Standard_Deviation": self._var([[0.01, 0.02]]),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "Terra_MODIS_NASADarkTarget_AOD_550/Mean")

    def test_cf_flag_variable_is_classified_implementation_by_metadata(self):
        """A QA flag carries CF flag_values + flag_meanings -- the metadata
        signal (before any name heuristic) marks it category 1, so it is never
        offered as a science variable even when its name looks innocuous."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "aod_550/Mean": self._var([[0.1, 0.2]], {"units": "1"}),
            "aod_550/quality": self._var(
                [[0, 1]], {"flag_values": [0, 1], "flag_meanings": "good bad"},
            ),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "aod_550/Mean")
        self.assertNotIn("aod_550/quality", [c.name for c in res.candidates])

    def test_geometry_reflectance_mask_ndvi_cloud_are_implementation_by_name(self):
        """When metadata is absent, name heuristics catch the plumbing the PRD
        taxonomy enumerates -- geometry angles, reflectances, land/ocean masks,
        NDVI, cloud fraction -- so none is ever the science pick."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "aod_550/Mean": self._var([[0.1, 0.2]], {"units": "1"}),
            "solar_zenith_angle": self._var([[30.0, 31.0]]),
            "scattering_angle": self._var([[120.0, 121.0]]),
            "TOA_Reflectance_550": self._var([[0.2, 0.3]]),
            "Land_Ocean_Mask": self._var([[0, 1]]),
            "NDVI": self._var([[0.4, 0.5]]),
            "cloud_fraction": self._var([[0.1, 0.2]]),
        })

        res = resolve(ds)

        surfaced = [c.name for c in res.candidates]
        self.assertEqual(surfaced, ["aod_550/Mean"])

    def test_explicitly_requesting_an_implementation_leaf_surfaces_it(self):
        """Category-1 plumbing is hidden from the default choice but still
        reachable by explicit request. When the user names ``Standard_Deviation``
        (which the earlier exact-match tier can't resolve because collision-
        renaming qualified it away from a bare leaf), the resolver surfaces the
        matching implementation var instead of excluding it."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "aod_550/Mean": self._var([[0.1, 0.2]], {"units": "1"}),
            "aod_550/Standard_Deviation": self._var([[0.01, 0.02]]),
        })

        res = resolve(ds, requested="Standard_Deviation")

        self.assertEqual(res.name, "aod_550/Standard_Deviation")

    def test_scoring_ranks_a_metadata_rich_geophysical_field_above_a_bare_one(self):
        """The score combines weak signals (Mean-leaf, units, geophysical name,
        standard_name, long_name). A metadata-rich AOD Mean outscores a bare
        field with no units or CF metadata, and candidates come back sorted
        best-first."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "bare/Value": self._var([[1.0, 2.0]]),
            "rich/Mean": self._var(
                [[0.1, 0.2]],
                {"units": "1", "standard_name": "aerosol_optical_depth", "long_name": "AOD 550 nm"},
            ),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "rich/Mean")
        names = [c.name for c in res.candidates]
        self.assertEqual(names[0], "rich/Mean")
        self.assertGreater(res.candidates[0].score, res.candidates[1].score)

    def test_all_nan_candidate_is_dropped_and_never_returned(self):
        """A globally empty field (0% valid -- the AERDA case: 27 of 49 AOD
        Mean groups) is dropped outright, so 'success' always means data on the
        map. Two otherwise-identical Mean fields: the empty one must never be
        the pick nor even a surfaced candidate."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "empty_group/Mean": self._var([[self.np.nan, self.np.nan]], {"units": "1"}),
            "populated_group/Mean": self._var([[0.1, 0.2]], {"units": "1"}),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "populated_group/Mean")
        self.assertNotIn("empty_group/Mean", [c.name for c in res.candidates])

    def test_a_sparsely_populated_field_ranks_below_a_well_populated_one(self):
        """Among populated candidates, materially higher coverage outranks the
        score: an 87%-valid field is chosen over a 0.2%-valid one even when
        both are geophysical Means. (The pipeline is healthy on the populated
        field -- the whole failure was picking an empty/near-empty one.)"""
        from tta_backend.preprocessing.variable_resolver import resolve

        high = self.np.full(100, 0.2)
        high[:13] = self.np.nan  # 87% valid
        low = self.np.full(100, 0.2)
        low[2:] = self.np.nan  # 2% valid
        # Names chosen so the lexical tiebreak would pick the SPARSE field --
        # only the coverage band can make the dense one win.
        ds = self.xr.Dataset({
            "aaa_sparse/Mean": (("cell",), low, {"units": "1"}),
            "zzz_dense/Mean": (("cell",), high, {"units": "1"}),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "zzz_dense/Mean")
        self.assertEqual([c.name for c in res.candidates], ["zzz_dense/Mean", "aaa_sparse/Mean"])

    def _sensor_ds(self, terra_vf, aqua_vf):
        def frac(vf):
            a = self.np.full(100, 0.2)
            a[int(round(vf * 100)):] = self.np.nan
            return (("cell",), a, {"units": "1"})
        return self.xr.Dataset({
            "Terra_MODIS_DarkTarget_AOD_550/Mean": frac(terra_vf),
            "Aqua_MODIS_DarkTarget_AOD_550/Mean": frac(aqua_vf),
        })

    def test_swapped_within_band_coverage_is_reproducible_but_a_cross_band_gap_reorders(self):
        """Reproducibility (PRD): ranking on the quantized coverage BAND, not
        raw valid_fraction, means day-to-day valid-pixel jitter within a band
        can't flip the chosen sensor -- two datasets identical but for
        swapped-within-band coverage resolve to the SAME name. A materially
        larger (cross-band) gap still reorders, so coverage genuinely wins when
        it matters."""
        from tta_backend.preprocessing.variable_resolver import resolve

        a = resolve(self._sensor_ds(0.85, 0.80))  # both band 8
        b = resolve(self._sensor_ds(0.80, 0.85))  # swapped, both still band 8
        self.assertEqual(a.name, b.name)

        c = resolve(self._sensor_ds(0.85, 0.20))  # Terra band 8, Aqua band 2
        d = resolve(self._sensor_ds(0.20, 0.85))  # Terra band 2, Aqua band 8
        self.assertEqual(c.name, "Terra_MODIS_DarkTarget_AOD_550/Mean")
        self.assertEqual(d.name, "Aqua_MODIS_DarkTarget_AOD_550/Mean")

    def test_within_a_tied_band_the_sensor_table_not_lexical_name_breaks_the_tie(self):
        """When coverage band AND score tie, the documented sensor/algorithm
        ordering decides -- NOT the lexical name. Terra and Aqua Dark Target
        AOD with identical coverage: 'Aqua' sorts first alphabetically, but the
        sensor table prefers Terra as the deterministic default, so Terra is
        chosen. (The table is a stable tiebreak, not a quality claim.)"""
        from tta_backend.preprocessing.variable_resolver import resolve

        res = resolve(self._sensor_ds(0.85, 0.85))  # both band 8, identical score

        self.assertEqual(res.name, "Terra_MODIS_DarkTarget_AOD_550/Mean")

    def test_aerda_shape_high_ambiguity_picks_a_populated_field_and_discloses_alternatives(self):
        """The origin case: several populated, scientifically-distinct AOD Mean
        products (different sensors/algorithms). Resolution confidence is high
        (a real populated science field exists) but scientific ambiguity is
        high (the sensor choice is a genuine fork) -- so the resolver auto-picks
        the top populated field AND emits a disclosure that names the chosen
        product and its alternatives, rather than refusing or silently guessing."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "Terra_MODIS_DarkTarget_AOD_550/Mean": (
                ("cell",), self.np.full(50, 0.19), {"units": "1", "long_name": "Terra MODIS Dark Target AOD 550"},
            ),
            "Aqua_MODIS_DarkTarget_AOD_550/Mean": (
                ("cell",), self.np.full(50, 0.21), {"units": "1", "long_name": "Aqua MODIS Dark Target AOD 550"},
            ),
            "SNPP_VIIRS_DeepBlue_AOD_550/Mean": (
                ("cell",), self.np.full(50, 0.18), {"units": "1", "long_name": "SNPP VIIRS Deep Blue AOD 550"},
            ),
        })

        res = resolve(ds)

        self.assertEqual(res.scientific_ambiguity, "high")
        self.assertEqual(res.resolution_confidence, "high")
        self.assertEqual(res.name, "Terra_MODIS_DarkTarget_AOD_550/Mean")
        self.assertIsNotNone(res.disclosure)
        # Names the chosen product and at least one alternative.
        self.assertIn("Terra MODIS Dark Target AOD 550", res.disclosure)
        self.assertTrue(
            "Aqua MODIS Dark Target AOD 550" in res.disclosure
            or "SNPP VIIRS Deep Blue AOD 550" in res.disclosure
        )

    def test_only_weak_signalless_candidates_refuses_with_no_name(self):
        """Genuinely low confidence: several populated but signal-less fields
        (no units, no CF metadata, no geophysical name, not a Mean) -- the
        resolver has nothing to resolve on, so it returns name=None rather than
        guessing. (The to_dataarray tail turns that into the P1-bounded refusal
        that asks the researcher to choose.)"""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "field_alpha": self._var([[1.0, 2.0]]),
            "field_beta": self._var([[3.0, 4.0]]),
        })

        res = resolve(ds)

        self.assertIsNone(res.name)
        self.assertEqual(res.resolution_confidence, "low")
        # Candidates are still surfaced for the refusal to list.
        self.assertEqual(len(res.candidates), 2)

    def test_a_single_clear_science_field_is_picked_silently(self):
        """High confidence, low ambiguity: one populated geophysical Mean (its
        Standard_Deviation sibling excluded as plumbing) is auto-picked with NO
        disclosure -- no fork to surface, so no note to nag with."""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "aod_550/Mean": self._var([[0.1, 0.2]], {"units": "1", "standard_name": "aerosol_optical_depth"}),
            "aod_550/Standard_Deviation": self._var([[0.01, 0.02]]),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "aod_550/Mean")
        self.assertEqual(res.resolution_confidence, "high")
        self.assertEqual(res.scientific_ambiguity, "low")
        self.assertIsNone(res.disclosure)

    def test_multiple_weak_name_only_contenders_refuse_rather_than_guess(self):
        """Conservative doctrine (keeps T25 where it is load-bearing): when two+
        candidates are distinguished ONLY by a weak 'name looks geophysical'
        heuristic -- no units, no aggregated Mean leaf, no CF standard_name --
        the resolver has no strong signal to choose on, so it refuses (name=None)
        and lets the researcher pick, rather than inventing a scientific choice.
        (A single such candidate is still pickable; contention is what forces
        the refusal.)"""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": self._var([[0.1, 0.2]]),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),
        })

        res = resolve(ds)

        self.assertIsNone(res.name)
        self.assertEqual(res.resolution_confidence, "low")

    def test_a_single_weak_geophysical_candidate_is_still_picked_with_a_note(self):
        """The refusal is about CONTENTION, not weakness: when only one science
        candidate survives (its plumbing siblings excluded), a weak-but-real
        geophysical field is still auto-picked -- at medium confidence, with a
        brief note -- rather than refused. (MOD08-style Cloud_Fraction excluded,
        Aerosol_Optical_Depth chosen.)"""
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "Cloud_Fraction": self._var([[0.5, 0.6]]),
            "Aerosol_Optical_Depth": self._var([[0.1, 0.2]]),
        })

        res = resolve(ds)

        self.assertEqual(res.name, "Aerosol_Optical_Depth")
        self.assertEqual(res.resolution_confidence, "medium")
        self.assertIsNotNone(res.disclosure)


if __name__ == "__main__":
    unittest.main()
