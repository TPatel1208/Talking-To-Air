import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class AggregationServiceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr
        self.col_info = {
            "primary_var": "no2",
            "cadence": "daily",
            "fill_value": -999.0,
            "valid_min": 0.0,
            "valid_max": 100.0,
        }

    def test_aggregate_counts_only_valid_time_steps(self):
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "no2": (
                    ("time", "lat", "lon"),
                    self.np.array([
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[-999.0, -999.0], [-999.0, -999.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]),
                )
            },
            coords={
                "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            attrs={"cadence": "daily"},
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)
        da = result.ds["no2"]

        self.assertEqual(result.meta["n_granules"], 2)
        self.assertEqual(result.meta["granule_dates"], ["2024-01-01", "2024-01-03"])
        self.assertEqual(float(da.sel(lat=40.0, lon=-75.0)), 3.0)

    def test_all_stats_supported_and_invalid_stat_raises(self):
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([
                [[1.0, 3.0]],
                [[5.0, 7.0]],
            ]),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-01-01", "2024-01-02"], "lat": [40.0], "lon": [-75.0, -74.0]},
            name="no2",
        )
        service = AggregationService()

        self.assertEqual(float(service.aggregate(da, stat="max", col_info=self.col_info).ds["no2"].values[0, 1]), 7.0)
        self.assertEqual(float(service.aggregate(da, stat="min", col_info=self.col_info).ds["no2"].values[0, 1]), 3.0)
        self.assertEqual(float(service.aggregate(da, stat="median", col_info=self.col_info).ds["no2"].values[0, 0]), 3.0)
        self.assertAlmostEqual(float(service.aggregate(da, stat="std", col_info=self.col_info).ds["no2"].values[0, 0]), 2.0)
        with self.assertRaises(ValueError):
            service.aggregate(da, stat="mode", col_info=self.col_info)

    def test_apply_quality_mask_falls_back_to_dataset_attrs_when_col_info_empty(self):
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            [[-999.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 100.0},
        )

        masked = AggregationService().apply_quality_mask(da, col_info={})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))
        self.assertTrue(self.np.isnan(values[1, 0]))
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)

    def test_aggregate_meta_discloses_cf_attrs_masking_when_no_registry_or_umm_var_info(self):
        """T25 Phase 1: an unregistered collection with no UMM-Var facts still
        gets a masking-provenance disclosure — the CF-attrs tier — instead of
        aggregate() staying silent about where fill/valid came from."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-999.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="unregistered_var",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 500.0},
        )

        result = AggregationService().aggregate(da, variable="unregistered_var")

        self.assertIn("masking", result.meta)
        self.assertEqual(result.meta["masking"]["fill_value_source"], "cf_attrs")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "cf_attrs")
        self.assertTrue(result.meta["masking"]["applied"])

    def test_aggregate_uses_umm_var_facts_when_no_yaml_col_info_supplied(self):
        """describe_dataset's per-variable UMM-Var facts mask an unregistered
        collection correctly even though the file's own CF attrs are wrong."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-9999.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="tempo_so2",
            attrs={"_FillValue": -1.0, "valid_min": -1e6, "valid_max": 1e6},  # wrong per-file attrs
        )
        umm_var_facts = [
            {
                "name": "tempo_so2",
                "fill_values": [{"value": -9999.0}],
                "valid_ranges": [{"min": 0.0, "max": 500.0}],
                "units": "molecules/cm^2",
            }
        ]

        result = AggregationService().aggregate(da, variable="tempo_so2", umm_var_facts=umm_var_facts)

        values = result.ds["tempo_so2"].values
        self.assertTrue(self.np.isnan(values[0, 0]))  # masked by UMM-Var fill, not the wrong CF fill
        self.assertTrue(self.np.isnan(values[1, 0]))  # 600 > UMM-Var valid_max of 500
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(result.meta["masking"]["fill_value_source"], "umm_var")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "umm_var")

    def test_aggregate_col_info_override_still_wins_over_umm_var_facts(self):
        """Registry/quirk-ledger col_info stays the top precedence tier even
        when UMM-Var facts are also supplied."""
        from preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-1.0, 10.0], [600.0, 20.0]]),
            dims=("lat", "lon"),
            name="no2",
        )
        umm_var_facts = [{"name": "no2", "fill_values": [{"value": -9999.0}], "valid_ranges": [{"min": 0.0, "max": 5000.0}]}]

        result = AggregationService().aggregate(
            da, variable="no2", col_info=self.col_info, umm_var_facts=umm_var_facts,
        )

        self.assertEqual(result.meta["masking"]["fill_value_source"], "collections_yaml")
        self.assertEqual(result.meta["masking"]["valid_range_source"], "collections_yaml")

    def test_to_dataarray_returns_the_single_data_var_with_no_choice_needed(self):
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({"no2": (("lat", "lon"), [[1.0]])})

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "no2")

    def test_to_dataarray_raises_a_candidate_listing_error_for_a_multi_var_file_with_no_choice(self):
        """T25: the next(iter(data.data_vars)) silent-first-var fallback is
        deleted -- MOD08_D3-style multi-variable files with no explicit
        variable, and no retrieval-recorded choice, must refuse with a
        structured error naming the candidates rather than guess."""
        from earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[1.0]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[2.0]]),
        })

        with self.assertRaises(MCPToolError) as ctx:
            AggregationService().to_dataarray(ds)

        self.assertEqual(ctx.exception.category, CATEGORY_VARIABLE_CHOICE_REQUIRED)
        self.assertIn("Cloud_Fraction", ctx.exception.message)
        self.assertIn("Aerosol_Optical_Depth", ctx.exception.message)

    def test_to_dataarray_explicit_variable_param_wins_on_a_multi_var_file(self):
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[1.0]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, variable="Aerosol_Optical_Depth")

        self.assertEqual(da.name, "Aerosol_Optical_Depth")
        self.assertEqual(float(da.values[0, 0]), 2.0)

    def test_to_dataarray_inherits_the_retrieval_recorded_choice_via_handle(self):
        from preprocessing.aggregation_service import AggregationService
        from services import variable_choice_registry

        variable_choice_registry._choices.clear()
        variable_choice_registry._choices["obs_1"] = ("Cloud_Fraction", float("inf"))
        self.addCleanup(variable_choice_registry._choices.clear)

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[1.0]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, handle="obs_1")

        self.assertEqual(da.name, "Cloud_Fraction")

    def test_to_dataarray_explicit_variable_wins_over_a_recorded_choice(self):
        from preprocessing.aggregation_service import AggregationService
        from services import variable_choice_registry

        variable_choice_registry._choices.clear()
        variable_choice_registry._choices["obs_1"] = ("Cloud_Fraction", float("inf"))
        self.addCleanup(variable_choice_registry._choices.clear)

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[1.0]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, handle="obs_1", variable="Aerosol_Optical_Depth")

        self.assertEqual(da.name, "Aerosol_Optical_Depth")

    def test_to_dataarray_resolves_a_group_qualified_recorded_choice_against_merged_leaf_names(self):
        """T25 review #1: retrieval records the model's choice group-qualified
        (``product/vertical_column_troposphere``), but open_handle merges HDF
        groups down to bare leaf names -- so resolution has to match on the
        leaf, or a registered TEMPO retrieval refuses its own recorded choice."""
        from preprocessing.aggregation_service import AggregationService
        from services import variable_choice_registry

        variable_choice_registry._choices.clear()
        variable_choice_registry._choices["obs_1"] = ("product/vertical_column_troposphere", float("inf"))
        self.addCleanup(variable_choice_registry._choices.clear)

        ds = self.xr.Dataset({
            "vertical_column_troposphere": (("lat", "lon"), [[1.0]]),
            "weight": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, handle="obs_1")

        self.assertEqual(da.name, "vertical_column_troposphere")

    def test_to_dataarray_resolves_a_group_qualified_explicit_variable_against_merged_leaf_names(self):
        """An explicit ``variable`` may arrive group-qualified too (the
        registry's own variable list is): it must match the merged bare leaf."""
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "vertical_column_troposphere": (("lat", "lon"), [[1.0]]),
            "weight": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, variable="product/vertical_column_troposphere")

        self.assertEqual(float(da.values[0, 0]), 1.0)

    def test_to_dataarray_resolves_a_science_plus_flag_pair_to_the_science_var_without_refusal(self):
        """T25 review #2: a standard TEMPO retrieval opens science +
        main_data_quality_flag (2 data_vars). The QA flag is not a science-
        variable candidate, so resolution picks the sole science var rather
        than raising CATEGORY_VARIABLE_CHOICE_REQUIRED and offering the flag."""
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "vertical_column_troposphere": (("lat", "lon"), [[1.0]]),
            "main_data_quality_flag": (("lat", "lon"), [[0]]),
        })

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "vertical_column_troposphere")

    def test_to_dataarray_excludes_a_cf_flag_var_identified_by_flag_attrs(self):
        """A sibling flag var need not be registry-pinned: CF ``flag_values``
        + ``flag_meanings`` mark it as a flag, so a science + CF-flag pair
        still resolves to the science var."""
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aerosol_optical_depth": (("lat", "lon"), [[1.0]]),
            "qa": (("lat", "lon"), [[0]]),
        })
        ds["qa"].attrs["flag_values"] = [0, 1]
        ds["qa"].attrs["flag_meanings"] = "good bad"

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "aerosol_optical_depth")

    def test_to_dataarray_never_offers_a_qa_flag_as_a_candidate_in_the_choice_error(self):
        """Even when a real choice is still required (2+ science vars), the QA
        flag riding along is excluded from the candidate list -- offering
        main_data_quality_flag as a 'science variable' to pick would be wrong."""
        from earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "vertical_column_troposphere": (("lat", "lon"), [[1.0]]),
            "vertical_column_stratosphere": (("lat", "lon"), [[2.0]]),
            "main_data_quality_flag": (("lat", "lon"), [[0]]),
        })

        with self.assertRaises(MCPToolError) as ctx:
            AggregationService().to_dataarray(ds)

        self.assertEqual(ctx.exception.category, CATEGORY_VARIABLE_CHOICE_REQUIRED)
        self.assertIn("vertical_column_troposphere", ctx.exception.message)
        self.assertIn("vertical_column_stratosphere", ctx.exception.message)
        self.assertNotIn("main_data_quality_flag", ctx.exception.message)

    def test_aggregate_recognizes_a_valid_time_dim_as_time(self):
        """T25: a MERRA-2-style `valid_time` dim (no CF standard_name, just
        the bare name) must still auto-reduce like a literal 'time' dim,
        rather than surviving into _normalize_to_2d as an unrecognized extra
        dimension that used to be silently averaged."""
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("valid_time", "lat", "lon"), [[[1.0, 2.0]], [[3.0, 4.0]]])},
            coords={
                "valid_time": ("valid_time", ["2024-01-01", "2024-01-02"], {"standard_name": "time"}),
                "lat": [40.0],
                "lon": [-75.0, -74.0],
            },
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="no2")

        self.assertNotIn("valid_time", result.ds["no2"].dims)
        self.assertEqual(result.meta["n_granules"], 2)

    def test_aggregate_applies_tier1_pinned_qa_rule_and_tags_verified(self):
        """T25 Phase 3: a pinned collections.yaml quality_flag_var + qa_good_values
        rule masks deterministically with no CF flag_meanings needed at all,
        and result.meta["masking"]["qa_status"] discloses it as "verified"."""
        from datasets.qa_flags import QA_VERIFIED
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "no2": (("lat", "lon"), self.np.array([[1.0, 2.0], [3.0, 4.0]])),
            "main_data_quality_flag": (("lat", "lon"), self.np.array([0, 1], dtype="int64").reshape(1, 2).repeat(2, axis=0)),
        })

        col_info = {**self.col_info, "quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}
        result = AggregationService().aggregate(ds, variable="no2", col_info=col_info)

        self.assertEqual(result.meta["masking"]["qa_status"], QA_VERIFIED)
        self.assertEqual(result.meta["masking"]["qa_source"], "collections_yaml")
        values = result.ds["no2"].values
        self.assertEqual(values[0, 0], 1.0)  # flag 0 (good) kept
        self.assertTrue(self.np.isnan(values[0, 1]))  # flag 1 (not in qa_good_values) masked

    def test_aggregate_discovers_qa_flag_var_via_ancillary_variables_attr(self):
        """T25 Phase 3: for an unregistered collection with no pinned
        quality_flag_var, the CF `ancillary_variables` attribute on the
        science variable is a real machine-readable pointer to the QA
        variable -- resolved without any registry entry."""
        from datasets.qa_flags import QA_CF_DETERMINISTIC
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "so2_column": (
                    ("lat", "lon"),
                    self.np.array([[1.0, 2.0], [3.0, 4.0]]),
                    {"ancillary_variables": "quality_flags"},
                ),
                "quality_flags": (
                    ("lat", "lon"),
                    self.np.array([[0, 1], [1, 0]], dtype="int64"),
                    {"flag_values": [0, 1], "flag_meanings": "good_quality bad_quality"},
                ),
            }
        )

        result = AggregationService().aggregate(ds, variable="so2_column")

        self.assertEqual(result.meta["masking"]["qa_status"], QA_CF_DETERMINISTIC)
        values = result.ds["so2_column"].values
        self.assertEqual(values[0, 0], 1.0)
        self.assertTrue(self.np.isnan(values[0, 1]))
        self.assertTrue(self.np.isnan(values[1, 0]))
        self.assertEqual(values[1, 1], 4.0)

    def test_aggregate_discovers_the_sole_sibling_flag_var_with_no_ancillary_attr(self):
        """T25 Phase 3: with no pinned name and no `ancillary_variables` CF
        attr, a single sibling data var carrying flag_values+flag_meanings is
        still discoverable deterministically -- there is nothing ambiguous
        about picking the only candidate."""
        from datasets.qa_flags import QA_CF_DETERMINISTIC
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aod": (("lat", "lon"), self.np.array([[1.0, 2.0]])),
            "qa": (
                ("lat", "lon"),
                self.np.array([[0, 1]], dtype="int64"),
                {"flag_values": [0, 1], "flag_meanings": "good_quality bad_quality"},
            ),
        })

        result = AggregationService().aggregate(ds, variable="aod")

        self.assertEqual(result.meta["masking"]["qa_status"], QA_CF_DETERMINISTIC)
        values = result.ds["aod"].values
        self.assertEqual(values[0, 0], 1.0)
        self.assertTrue(self.np.isnan(values[0, 1]))

    def test_aggregate_ambiguous_cf_tokens_with_agent_proposal_masks_and_tags_inferred(self):
        """T25 Phase 3: ambiguous flag_meanings tokens are never auto-applied
        -- only the agent's proposed good-token list resolves them, and the
        result is tagged "inferred, not verified", not silently folded into
        "verified" or "cf-deterministic"."""
        from datasets.qa_flags import QA_INFERRED
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aod": (("lat", "lon"), self.np.array([[1.0, 2.0, 3.0]])),
            "qa": (
                ("lat", "lon"),
                self.np.array([[0, 1, 2]], dtype="int64"),
                {"flag_values": [0, 1, 2], "flag_meanings": "good_quality partially_cloudy_usable missing"},
            ),
        })

        result = AggregationService().aggregate(ds, variable="aod", qa_good_tokens=["partially_cloudy_usable"])

        masking = result.meta["masking"]
        self.assertEqual(masking["qa_status"], QA_INFERRED)
        self.assertEqual(masking["qa_ambiguous_tokens"], ["partially_cloudy_usable"])
        self.assertEqual(masking["qa_inferred_tokens"], ["partially_cloudy_usable"])
        values = result.ds["aod"].values
        self.assertEqual(values[0, 0], 1.0)  # good_quality
        self.assertEqual(values[0, 1], 2.0)  # inferred good via agent proposal
        self.assertTrue(self.np.isnan(values[0, 2]))  # missing, still bad

    def test_aggregate_ambiguous_cf_tokens_without_proposal_applies_no_mask(self):
        from datasets.qa_flags import QA_AMBIGUOUS_PENDING
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aod": (("lat", "lon"), self.np.array([[1.0, 2.0, 3.0]])),
            "qa": (
                ("lat", "lon"),
                self.np.array([[0, 1, 2]], dtype="int64"),
                {"flag_values": [0, 1, 2], "flag_meanings": "good_quality partially_cloudy_usable missing"},
            ),
        })

        result = AggregationService().aggregate(ds, variable="aod")

        masking = result.meta["masking"]
        self.assertEqual(masking["qa_status"], QA_AMBIGUOUS_PENDING)
        values = result.ds["aod"].values
        # No mask applied yet -- every value survives untouched.
        self.assertEqual(list(values[0]), [1.0, 2.0, 3.0])

    def test_aggregate_no_qa_metadata_anywhere_discloses_not_applied(self):
        """T25 Phase 3: a prose-only-QA product (e.g. MOD08_D3) has no
        pinned rule and no CF flag_values/flag_meanings anywhere -- masking
        stays off, and meta says so explicitly rather than staying silent."""
        from datasets.qa_flags import QA_NOT_APPLIED
        from preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Aerosol_Optical_Depth_Land_Ocean_Mean": (("lat", "lon"), self.np.array([[1.0, 2.0]])),
        })

        result = AggregationService().aggregate(ds, variable="Aerosol_Optical_Depth_Land_Ocean_Mean")

        self.assertEqual(result.meta["masking"]["qa_status"], QA_NOT_APPLIED)
        values = result.ds["Aerosol_Optical_Depth_Land_Ocean_Mean"].values
        self.assertEqual(list(values[0]), [1.0, 2.0])

    def test_apply_quality_mask_col_info_override_wins_over_dataset_attrs(self):
        from preprocessing.aggregation_service import AggregationService

        # Dataset's own attrs are wrong (a known CMR/UMM-Var quirk); the
        # override in col_info must take precedence.
        da = self.xr.DataArray(
            [[-1.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -1.0, "valid_min": -1000.0, "valid_max": 1000.0},
        )

        masked = AggregationService().apply_quality_mask(
            da, col_info={"fill_value": -1.0, "valid_min": 0.0, "valid_max": 100.0}
        )

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # fill
        self.assertTrue(self.np.isnan(values[1, 0]))  # 200 > override valid_max of 100
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)


if __name__ == "__main__":
    unittest.main()
