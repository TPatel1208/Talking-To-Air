import importlib.util
import unittest


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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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

    def test_clustered_granules_are_cadence_weighted_not_over_counted(self):
        """Finding #11: 'average over the period' must weight each cadence
        bucket equally, not each granule. Three granules on 2024-01-01 (value
        10) and one on 2024-01-02 (value 0): the naive per-granule mean is
        (10+10+10+0)/4 = 7.5, over-weighting the dense first day. The
        cadence-weighted (per-day) mean is (10 + 0)/2 = 5.0."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("time", "lat", "lon"), self.np.array([
                [[10.0]], [[10.0]], [[10.0]], [[0.0]],
            ]))},
            coords={
                "time": [
                    "2024-01-01T06:00:00", "2024-01-01T12:00:00",
                    "2024-01-01T18:00:00", "2024-01-02T12:00:00",
                ],
                "lat": [40.0], "lon": [-75.0],
            },
            attrs={"cadence": "daily"},
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)

        self.assertAlmostEqual(float(result.ds["no2"].sel(lat=40.0, lon=-75.0)), 5.0)

    def test_evenly_cadenced_granules_are_unchanged_by_weighting(self):
        """Finding #11 guard: one granule per cadence bucket weights every day
        equally already, so the cadence-weighted mean equals the plain mean --
        the fix only moves clustered sampling, never evenly-spaced series."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("time", "lat", "lon"), self.np.array([[[3.0]], [[6.0]], [[9.0]]]))},
            coords={
                "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "lat": [40.0], "lon": [-75.0],
            },
            attrs={"cadence": "daily"},
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info)

        self.assertAlmostEqual(float(result.ds["no2"].sel(lat=40.0, lon=-75.0)), 6.0)

    def test_all_stats_supported_and_invalid_stat_raises(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        # Sample std (ddof=1) over the two granules [1, 5]: sqrt(((1-3)^2 +
        # (5-3)^2) / (2-1)) = sqrt(8). Population std (ddof=0) would understate
        # this as 2.0 — few-granule variability is a *sample*, not the whole.
        self.assertAlmostEqual(
            float(service.aggregate(da, stat="std", col_info=self.col_info).ds["no2"].values[0, 0]),
            self.np.sqrt(8.0),
        )
        with self.assertRaises(ValueError):
            service.aggregate(da, stat="mode", col_info=self.col_info)

    def test_sample_std_of_a_single_granule_is_nan_not_a_fabricated_zero(self):
        """n=1 has no sample spread to estimate: ddof=1 makes the honest answer
        NaN, not the 0.0 that ddof=0 would fabricate (and never a crash)."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[[5.0, 5.0]]]),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-01-01"], "lat": [40.0], "lon": [-75.0, -74.0]},
            name="no2",
        )

        result = AggregationService().aggregate(da, stat="std", col_info=self.col_info)

        self.assertTrue(self.np.isnan(float(result.ds["no2"].values[0, 0])))

    def test_apply_quality_mask_falls_back_to_dataset_attrs_when_col_info_empty(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            [[-999.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -999.0, "valid_min": 0.0, "valid_max": 100.0},
        )

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))
        self.assertTrue(self.np.isnan(values[1, 0]))
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)

    def test_aggregate_meta_discloses_cf_attrs_masking_when_no_registry_or_umm_var_info(self):
        """T25 Phase 1: an unregistered collection with no UMM-Var facts still
        gets a masking-provenance disclosure — the CF-attrs tier — instead of
        aggregate() staying silent about where fill/valid came from."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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

    def test_timeseries_aggregation_meta_matches_aggregate_meta_shape(self):
        """T32: conduct_temporal_statistic keeps every timestep instead of
        reducing over time, so it can't call aggregate() directly -- this
        wrapper must build the identical aggregation_label/granule_dates/
        n_granules/cadence summary aggregate() derives internally, from the
        caller's own record of which timesteps survived masking."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]),
            dims=("time", "lat", "lon"),
            coords={
                "time": ["2024-01-01", "2024-01-02"],
                "lat": [40.0, 41.0],
                "lon": [-75.0, -74.0],
            },
            name="no2",
        )
        service = AggregationService()

        meta = service.timeseries_aggregation_meta(
            da, valid_indices=[0, 1], stat="mean", time_dim="time", col_info=self.col_info,
        )
        reduced_meta = service.aggregate(da, stat="mean", col_info=self.col_info).meta

        self.assertEqual(meta["n_granules"], 2)
        self.assertEqual(meta["granule_dates"], ["2024-01-01", "2024-01-02"])
        self.assertEqual(meta["cadence"], "daily")
        self.assertEqual(meta["aggregation_label"], reduced_meta["aggregation_label"])
        self.assertNotIn("masking", meta)  # caller merges its own masking_provenance in

    def test_timeseries_aggregation_meta_counts_only_the_valid_indices_given(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]),
            dims=("time", "lat", "lon"),
            coords={"time": ["2024-01-01", "2024-01-02", "2024-01-03"], "lat": [40.0], "lon": [-75.0, -74.0]},
            name="no2",
        )

        meta = AggregationService().timeseries_aggregation_meta(
            da, valid_indices=[0, 2], stat="mean", time_dim="time", col_info=self.col_info,
        )

        self.assertEqual(meta["n_granules"], 2)
        self.assertEqual(meta["granule_dates"], ["2024-01-01", "2024-01-03"])

    def test_aggregate_falls_back_to_ecs_range_attrs_when_no_time_coordinate_exists(self):
        """Monthly L3 means (e.g. HAQ_TROPOMI_NO2_GLOBAL_M_L3) have lat/lon
        dims only -- the month lives exclusively in RangeBeginningDate/
        RangeEndingDate global attrs. The Metadata tab showed "Date Range:
        Not available"/"Granule Dates: Not available" for these charts because
        _build_meta only read time coordinates (regression)."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
            attrs={
                "cadence": "monthly",
                "RangeBeginningDate": "2024-01-01",
                "RangeBeginningTime": "00:00:00.000000Z",
                "RangeEndingDate": "2024-01-31",
                "RangeEndingTime": "23:59:59.999999Z",
            },
        )

        meta = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info).meta

        self.assertEqual(meta["n_granules"], 1)
        self.assertEqual(meta["start_date"], "2024-01-01")
        self.assertEqual(meta["end_date"], "2024-01-31")
        self.assertEqual(meta["granule_dates"], ["2024-01-01"])
        self.assertIn("2024-01-01 to 2024-01-31", meta["aggregation_label"])

    def test_aggregate_reads_range_attrs_from_source_ds_when_data_is_an_extracted_dataarray(self):
        """Every real tool path passes an already-extracted DataArray as
        ``data`` (variable attrs only -- no global attrs) plus the opened
        Dataset as ``source_ds``: the temporal fallback must look there."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        source_ds = self.xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
            attrs={"time_coverage_start": "2024-01-01T00:00:00Z", "time_coverage_end": "2024-01-31T23:59:59Z"},
        )
        da = source_ds["no2"]

        meta = AggregationService().aggregate(
            da, stat="mean", variable="no2", col_info=self.col_info, source_ds=source_ds,
        ).meta

        self.assertEqual(meta["start_date"], "2024-01-01")
        self.assertEqual(meta["end_date"], "2024-01-31")
        self.assertEqual(meta["granule_dates"], ["2024-01-01"])

    def test_aggregate_meta_start_and_end_come_from_the_time_coordinate_when_present(self):
        """The explicit meta start_date/end_date facts (new) must agree with
        the granule_dates the time coordinate already produced -- attrs are a
        fallback, never an override."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("time", "lat", "lon"), [[[1.0]], [[2.0]]])},
            coords={"time": ["2024-06-01", "2024-06-02"], "lat": [40.0], "lon": [-75.0]},
            attrs={"cadence": "daily", "RangeBeginningDate": "1999-01-01", "RangeEndingDate": "1999-01-31"},
        )

        meta = AggregationService().aggregate(ds, stat="mean", variable="no2", col_info=self.col_info).meta

        self.assertEqual(meta["start_date"], "2024-06-01")
        self.assertEqual(meta["end_date"], "2024-06-02")
        self.assertEqual(meta["granule_dates"], ["2024-06-01", "2024-06-02"])

    def test_twelve_clustered_monthly_granules_are_not_labeled_annual(self):
        """Finding #14: "Annual" is inferred from a granule *count* of 12, not
        the date *span*. Twelve monthly-cadence granules clustered inside a
        single month (reprocessed/overlapping) span days, not a year -- calling
        that "Annual" is a false provenance claim. The label must reflect the
        real span instead."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        clustered = [f"2024-01-{d:02d}" for d in range(1, 13)]  # 12 granules, 11-day span
        da = self.xr.DataArray(
            self.np.arange(12.0).reshape(12, 1, 1),
            dims=("time", "lat", "lon"),
            coords={"time": clustered, "lat": [40.0], "lon": [-75.0]},
            name="no2",
            attrs={"cadence": "monthly"},
        )

        meta = AggregationService().timeseries_aggregation_meta(
            da, valid_indices=list(range(12)), stat="mean", time_dim="time", col_info=self.col_info,
        )

        self.assertNotIn("Annual", meta["aggregation_label"])
        self.assertIn("2024-01-01 to 2024-01-12", meta["aggregation_label"])

    def test_twelve_monthly_granules_spanning_a_year_are_labeled_annual(self):
        """Finding #14 guard: a real year of monthly means (span ~a year) must
        still read "Annual" -- the fix narrows the label to genuine spans, it
        doesn't remove it."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        annual = [f"2023-{m:02d}-01" for m in range(6, 13)] + [f"2024-{m:02d}-01" for m in range(1, 6)]
        da = self.xr.DataArray(
            self.np.arange(12.0).reshape(12, 1, 1),
            dims=("time", "lat", "lon"),
            coords={"time": annual, "lat": [40.0], "lon": [-75.0]},
            name="no2",
            attrs={"cadence": "monthly"},
        )

        meta = AggregationService().timeseries_aggregation_meta(
            da, valid_indices=list(range(12)), stat="mean", time_dim="time", col_info=self.col_info,
        )

        self.assertIn("Annual", meta["aggregation_label"])

    def test_to_dataarray_returns_the_single_data_var_with_no_choice_needed(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({"no2": (("lat", "lon"), [[1.0]])})

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "no2")

    def test_to_dataarray_picks_the_science_var_over_plumbing_in_a_multi_var_file(self):
        """T48: the VariableResolver replaces the old blanket refusal. A
        MOD08_D3-style file pairs a geophysical field with plumbing
        (Cloud_Fraction is a category-1 implementation variable) -- so instead
        of refusing and dumping both, the resolver excludes the plumbing and
        auto-picks the single populated science field. The choice rides out on
        the returned array's resolution attrs."""
        from tta_backend.preprocessing.aggregation_service import VARIABLE_RESOLUTION_ATTR, AggregationService

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[0.5]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[0.2]]),
        })

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "Aerosol_Optical_Depth")
        facts = da.attrs[VARIABLE_RESOLUTION_ATTR]
        self.assertEqual(facts["chosen"], "Aerosol_Optical_Depth")

    def test_to_dataarray_resolves_a_wide_unregistered_mean_file_to_a_populated_field(self):
        """P2 win (AERDA_D3_VIIRS_MODIS, dataset_a88593edb7246c9b): a
        Yori-aggregated L3 merges to hundreds of ``group/Mean`` data_vars with
        no registry default. The old tail refused with an unusable candidate
        dump; the resolver now scores the Mean fields, drops empties, ranks, and
        auto-picks a populated one -- a real map on the first try instead of a
        refusal. The resolution facts (chosen + disclosure) ride out on attrs."""
        from tta_backend.preprocessing.aggregation_service import VARIABLE_RESOLUTION_ATTR, AggregationService

        data_vars = {
            f"group_{i:03d}/Mean": (("lat", "lon"), [[float(i) / 100.0]], {"units": "1"})
            for i in range(200)
        }
        ds = self.xr.Dataset(data_vars)

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.attrs[VARIABLE_RESOLUTION_ATTR]["chosen"], da.name)
        self.assertTrue(da.name.endswith("/Mean"))
        # A wide distinct-product fork must disclose (never silently guess).
        self.assertIsNotNone(da.attrs[VARIABLE_RESOLUTION_ATTR]["disclosure"])

    def test_ambiguous_variable_error_is_bounded_for_a_pathologically_wide_unresolvable_file(self):
        """P1 bound, still load-bearing for the files that genuinely refuse: a
        wide file whose candidates carry no strong disambiguating signal (no
        units, no Mean leaf, no CF metadata) is a real fork the resolver won't
        guess -- it refuses. That refusal message must be bounded in size
        regardless of candidate count: name a capped sample, disclose the true
        total, and say how many are hidden -- an O(1) error, never the 20 KB
        O(N) dump that drove the original context blowup."""
        from tta_backend.earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired

        data_vars = {f"field_{i:03d}": (("lat", "lon"), [[float(i)]]) for i in range(200)}
        ds = self.xr.Dataset(data_vars)

        with self.assertRaises(VariableChoiceRequired) as ctx:
            AggregationService().to_dataarray(ds)

        # T49: the LLM-facing compact tool result the signal carries is still
        # P1-bounded (the full, uncapped candidate list rides out-of-band to
        # the picker instead of being re-fed to the model).
        exc = ctx.exception.mcp_error
        self.assertEqual(exc.category, CATEGORY_VARIABLE_CHOICE_REQUIRED)
        # O(1) size: the whole point of P1 -- must not scale with 200 vars.
        self.assertLess(len(exc.message), 4000)
        self.assertLess(len(exc.to_tool_json()), 4000)
        # Still honest about the true total and that candidates are hidden.
        self.assertIn("200", exc.message)
        self.assertIn("more", exc.message.lower())
        # Still actionable: names at least a few real candidates.
        self.assertIn("field_000", exc.message)

    def test_ambiguous_variable_error_lists_all_candidates_when_few(self):
        """Bounding must not truncate a small refusal: two weak, name-only
        distinct products (no units, no Mean leaf) are a genuine fork the
        resolver refuses -- and the refusal names both candidates in full (no
        spurious 'and N more')."""
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": (("lat", "lon"), [[0.1]]),
            "COMBINE_AOD_550_AVG": (("lat", "lon"), [[0.3]]),
        })

        with self.assertRaises(VariableChoiceRequired) as ctx:
            AggregationService().to_dataarray(ds)

        msg = ctx.exception.mcp_error.message
        self.assertIn("DT_AOD_550_AVG", msg)
        self.assertIn("COMBINE_AOD_550_AVG", msg)
        self.assertNotIn("more", msg.lower())
        # The signal carries the resolver's full Resolution so the tool can
        # build the (uncapped) picker from it.
        names = {c.name for c in ctx.exception.resolution.candidates}
        self.assertEqual(names, {"DT_AOD_550_AVG", "COMBINE_AOD_550_AVG"})

    def test_to_dataarray_resolves_a_registered_multi_var_file_to_its_pinned_primary_var(self):
        """AOD misrouting follow-up (2026-07-12): a registered collection's
        pinned collections.yaml primary_var is a curated choice, not the
        deleted first-var guess -- so AER_DBDT AOD (multiple AOD science vars,
        primary COMBINE_AOD_550_AVG) resolves to that variable instead of
        raising. Uses the CF/ACDD ``ShortName`` spelling the real AER_DBDT
        export actually carries (live-verified 2026-07-12), not lowercase."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": (("lat", "lon"), [[1.0]]),
            "DB_AOD_550_AVG": (("lat", "lon"), [[2.0]]),
            "COMBINE_AOD_550_AVG": (("lat", "lon"), [[3.0]]),
        })
        ds.attrs["ShortName"] = "AER_DBDT_D10KM_L3_MODIS_TERRA"

        da = AggregationService().to_dataarray(ds)

        self.assertEqual(da.name, "COMBINE_AOD_550_AVG")
        self.assertEqual(float(da.values[0, 0]), 3.0)

    def test_to_dataarray_still_refuses_a_multi_var_file_when_the_collection_is_unregistered(self):
        """The primary_var tier is registry-gated: the same shape whose
        short_name matches no registered collection must still refuse, so the
        never-guess doctrine holds for genuinely unknown files."""
        from tta_backend.earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": (("lat", "lon"), [[1.0]]),
            "COMBINE_AOD_550_AVG": (("lat", "lon"), [[3.0]]),
        })
        ds.attrs["ShortName"] = "NOT_A_REGISTERED_COLLECTION"

        with self.assertRaises(VariableChoiceRequired) as ctx:
            AggregationService().to_dataarray(ds)

        self.assertEqual(ctx.exception.mcp_error.category, CATEGORY_VARIABLE_CHOICE_REQUIRED)

    def test_aggregate_surfaces_the_variable_resolution_disclosure_in_meta(self):
        """T48 disclosure plumbing: when aggregate() resolves an ambiguous
        multi-product file itself (Dataset input), the resolver's chosen
        variable and its disclosure must ride out in meta -- the channel the
        tool layer copies into chart provenance and the dispatch layer appends
        to the answer. A silent auto-pick of a genuine sensor fork would hide
        the choice."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Terra_MODIS_DarkTarget_AOD_550/Mean": (
                ("lat", "lon"), [[0.19, 0.2]], {"units": "1", "long_name": "Terra MODIS Dark Target AOD 550"},
            ),
            "Aqua_MODIS_DarkTarget_AOD_550/Mean": (
                ("lat", "lon"), [[0.21, 0.22]], {"units": "1", "long_name": "Aqua MODIS Dark Target AOD 550"},
            ),
        })

        meta = AggregationService().aggregate(ds, stat="mean").meta

        self.assertIn("variable_resolution", meta)
        self.assertEqual(meta["variable_resolution"]["chosen"], "Terra_MODIS_DarkTarget_AOD_550/Mean")
        self.assertIsNotNone(meta["variable_resolution"]["disclosure"])
        self.assertIn("Aqua MODIS Dark Target AOD 550", meta["variable_resolution"]["disclosure"])

    def test_aggregate_omits_variable_resolution_when_no_resolver_choice_was_made(self):
        """A single-variable file (or an explicit/registry pick) never runs the
        resolver, so meta carries no variable_resolution -- no note to nag with."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

        meta = AggregationService().aggregate(ds, stat="mean", variable="no2").meta

        self.assertNotIn("variable_resolution", meta)

    def test_to_dataarray_explicit_variable_param_wins_on_a_multi_var_file(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Cloud_Fraction": (("lat", "lon"), [[1.0]]),
            "Aerosol_Optical_Depth": (("lat", "lon"), [[2.0]]),
        })

        da = AggregationService().to_dataarray(ds, variable="Aerosol_Optical_Depth")

        self.assertEqual(da.name, "Aerosol_Optical_Depth")
        self.assertEqual(float(da.values[0, 0]), 2.0)

    def test_to_dataarray_inherits_the_retrieval_recorded_choice_via_handle(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService
        from tta_backend.services import variable_choice_registry

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
        from tta_backend.preprocessing.aggregation_service import AggregationService
        from tta_backend.services import variable_choice_registry

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
        from tta_backend.preprocessing.aggregation_service import AggregationService
        from tta_backend.services import variable_choice_registry

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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED
        from tta_backend.preprocessing.aggregation_service import AggregationService, VariableChoiceRequired

        ds = self.xr.Dataset({
            "vertical_column_troposphere": (("lat", "lon"), [[1.0]]),
            "vertical_column_stratosphere": (("lat", "lon"), [[2.0]]),
            "main_data_quality_flag": (("lat", "lon"), [[0]]),
        })

        with self.assertRaises(VariableChoiceRequired) as ctx:
            AggregationService().to_dataarray(ds)

        exc = ctx.exception.mcp_error
        self.assertEqual(exc.category, CATEGORY_VARIABLE_CHOICE_REQUIRED)
        self.assertIn("vertical_column_troposphere", exc.message)
        self.assertIn("vertical_column_stratosphere", exc.message)
        self.assertNotIn("main_data_quality_flag", exc.message)
        # The QA flag is excluded from the picker candidates too, not just the
        # bounded message.
        cand_names = {c.name for c in ctx.exception.resolution.candidates}
        self.assertNotIn("main_data_quality_flag", cand_names)

    def test_aggregate_recognizes_a_valid_time_dim_as_time(self):
        """T25: a MERRA-2-style `valid_time` dim (no CF standard_name, just
        the bare name) must still auto-reduce like a literal 'time' dim,
        rather than surviving into _normalize_to_2d as an unrecognized extra
        dimension that used to be silently averaged."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.datasets.qa_flags import QA_VERIFIED
        from tta_backend.preprocessing.aggregation_service import AggregationService

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

    def test_aggregate_applies_tier1_qa_rule_via_source_ds_when_data_is_an_already_extracted_dataarray(self):
        """T25 masking-execution fix: production never passes aggregate() a
        Dataset -- every tool extracts the science DataArray first (crops it
        to a region, normalizes longitude, etc.) and only the *opened*
        Dataset still carries the sibling QA-flag variable. Before
        source_ds, aggregate(cropped_da, ...) always saw qf_source=None and
        the honesty guard downgraded to "not applied" no matter what
        collections.yaml pinned. This proves source_ds restores real masking
        for that exact shape."""
        from tta_backend.datasets.qa_flags import QA_VERIFIED
        from tta_backend.preprocessing.aggregation_service import AggregationService

        source_ds = self.xr.Dataset({
            "no2": (("lat", "lon"), self.np.array([[1.0, 2.0], [3.0, 4.0]])),
            "main_data_quality_flag": (("lat", "lon"), self.np.array([[0, 1], [0, 1]], dtype="int64")),
        })
        # Mirrors the real tool shape: `da` is an already-extracted, standalone
        # DataArray -- not the Dataset carrying the flag variable.
        da = source_ds["no2"]

        col_info = {**self.col_info, "quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}
        result = AggregationService().aggregate(da, variable="no2", col_info=col_info, source_ds=source_ds)

        self.assertEqual(result.meta["masking"]["qa_status"], QA_VERIFIED)
        values = result.ds["no2"].values
        self.assertEqual(values[0, 0], 1.0)  # flag 0 (good) kept
        self.assertTrue(self.np.isnan(values[0, 1]))  # flag 1 (bad) masked

    def test_aggregate_still_downgrades_to_not_applied_when_source_ds_is_omitted(self):
        """The honesty guard still fires when a caller genuinely has no
        Dataset available at all -- source_ds is opt-in, not a silent
        always-on assumption."""
        from tta_backend.datasets.qa_flags import QA_NOT_APPLIED
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("lat", "lon"), name="no2",
        )
        col_info = {**self.col_info, "quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}

        result = AggregationService().aggregate(da, variable="no2", col_info=col_info)

        self.assertEqual(result.meta["masking"]["qa_status"], QA_NOT_APPLIED)

    def test_resolve_and_mask_is_the_shared_seam_aggregate_and_conduct_temporal_statistic_both_use(self):
        """conduct_temporal_statistic (plot_tools.py) calls resolve_and_mask
        directly instead of hand-rolling apply_quality_mask -- this pins its
        contract: masked data + honest provenance, same shape aggregate()
        gets internally."""
        from tta_backend.datasets.qa_flags import QA_VERIFIED
        from tta_backend.preprocessing.aggregation_service import AggregationService

        source_ds = self.xr.Dataset({
            "no2": (("time", "lat", "lon"), self.np.array([[[1.0, 2.0]], [[3.0, 4.0]]])),
            "main_data_quality_flag": (("time", "lat", "lon"), self.np.array([[[0, 1]], [[0, 1]]], dtype="int64")),
        })
        da = source_ds["no2"]
        col_info = {**self.col_info, "quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}

        masked, provenance = AggregationService().resolve_and_mask(
            da, variable="no2", col_info=col_info, source_ds=source_ds,
        )

        self.assertEqual(provenance["qa_status"], QA_VERIFIED)
        # Shape preserved (no time reduction) -- unlike aggregate(), which reduces.
        self.assertEqual(masked.dims, da.dims)
        values = masked.values
        self.assertEqual(values[0, 0, 0], 1.0)
        self.assertTrue(self.np.isnan(values[0, 0, 1]))

    def test_aggregate_discovers_qa_flag_var_via_ancillary_variables_attr(self):
        """T25 Phase 3: for an unregistered collection with no pinned
        quality_flag_var, the CF `ancillary_variables` attribute on the
        science variable is a real machine-readable pointer to the QA
        variable -- resolved without any registry entry."""
        from tta_backend.datasets.qa_flags import QA_CF_DETERMINISTIC
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.datasets.qa_flags import QA_CF_DETERMINISTIC
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.datasets.qa_flags import QA_INFERRED
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        from tta_backend.datasets.qa_flags import QA_AMBIGUOUS_PENDING
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
        stays off, and meta says so explicitly rather than staying silent.

        Since 2026-08-02 that disclosure is the more precise "publishes no
        quality flag variable": nothing here is unknown, there is simply no
        flag band to interpret. MODIS L3 AOD is quality-screened at L2 before
        gridding, so this is a permanent product property, not a gap to pin."""
        from tta_backend.datasets.qa_flags import QA_NO_FLAG_VARIABLE
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "Aerosol_Optical_Depth_Land_Ocean_Mean": (("lat", "lon"), self.np.array([[1.0, 2.0]])),
        })

        result = AggregationService().aggregate(ds, variable="Aerosol_Optical_Depth_Land_Ocean_Mean")

        self.assertEqual(result.meta["masking"]["qa_status"], QA_NO_FLAG_VARIABLE)
        values = result.ds["Aerosol_Optical_Depth_Land_Ocean_Mean"].values
        self.assertEqual(list(values[0]), [1.0, 2.0])

    def test_apply_quality_mask_col_info_override_wins_over_dataset_attrs(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        # Dataset's own attrs are wrong (a known CMR/UMM-Var quirk); the
        # override in col_info must take precedence.
        da = self.xr.DataArray(
            [[-1.0, 10.0], [200.0, 20.0]],
            dims=("y", "x"),
            name="no2",
            attrs={"_FillValue": -1.0, "valid_min": -1000.0, "valid_max": 1000.0},
        )

        masked, _ = AggregationService()._apply_quality_mask(
            da, col_info={"fill_value": -1.0, "valid_min": 0.0, "valid_max": 100.0}
        )

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # fill
        self.assertTrue(self.np.isnan(values[1, 0]))  # 200 > override valid_max of 100
        self.assertEqual(values[0, 1], 10.0)
        self.assertEqual(values[1, 1], 20.0)

    def test_aggregate_all_bad_cf_flags_do_not_wipe_the_whole_variable(self):
        """Follow-up review #1: a flag var whose flag_meanings are all
        bad-quality tokens yields good_values=[]. Now that masking actually
        runs (source_ds threaded), keying ``qf.isin([])`` would NaN every
        pixel. The empty good-set must instead disable masking (go ambiguous),
        never wipe the variable."""
        from tta_backend.datasets.qa_flags import QA_AMBIGUOUS_PENDING
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aod": (("lat", "lon"), self.np.array([[1.0, 2.0]])),
            "qa": (
                ("lat", "lon"),
                self.np.array([[1, 2]], dtype="int64"),
                {"flag_values": [1, 2], "flag_meanings": "missing cloudy"},
            ),
        })

        result = AggregationService().aggregate(ds, variable="aod")

        self.assertEqual(result.meta["masking"]["qa_status"], QA_AMBIGUOUS_PENDING)
        values = result.ds["aod"].values
        self.assertEqual(list(values[0]), [1.0, 2.0])  # nothing wiped

    def test_apply_quality_mask_zero_fill_masks_only_exact_fill_not_near_zero(self):
        """Follow-up review #2: a 0 fill (now reachable via the widened
        UMM-Var fill tier) must mask only exact-fill cells. The old
        magnitude-scaled tolerance collapsed to atol=0 here; the exact-match
        contract keeps legitimate near-zero measurements alive."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            [[0.0, 0.4], [1.0, 2.0]],
            dims=("y", "x"),
            name="conc",
        )

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={"fill_value": 0.0})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # exact 0 fill masked
        self.assertEqual(values[0, 1], 0.4)  # legit near-zero measurement kept
        self.assertEqual(values[1, 0], 1.0)
        self.assertEqual(values[1, 1], 2.0)

    def test_apply_quality_mask_uses_exact_fill_match_not_a_magnitude_band(self):
        """Follow-up review #2: the old ``atol=abs(fill)*1e-3`` band wrongly
        masked legitimate values *near* a small integer fill (49.99 and 50.05
        against a 50.0 fill fall inside the old 0.05 band). Exact matching
        masks only the true fill."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            [[50.0, 49.99], [50.05, 10.0]],
            dims=("y", "x"),
            name="v",
        )

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={"fill_value": 50.0})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # exact fill masked
        self.assertEqual(values[0, 1], 49.99)  # inside old band -> now kept
        self.assertEqual(values[1, 0], 50.05)  # inside old band -> now kept
        self.assertEqual(values[1, 1], 10.0)

    def test_apply_quality_mask_bad_values_drops_unknown_quality_flag_pixels(self):
        """Follow-up review #3: the bad_values path must drop pixels whose
        flag is NaN/unknown (uncomputed quality), symmetric with the
        good_values path -- not silently count them as good. Before, ``~isin``
        alone kept every unknown-flag pixel."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "aod": (("lat", "lon"), self.np.array([[1.0, 2.0, 3.0]])),
            # flag 0 = good/known, 2 = bad, NaN = quality not computed
            "qa": (("lat", "lon"), self.np.array([[0.0, 2.0, self.np.nan]])),
        })

        masked, _ = AggregationService()._apply_quality_mask(
            ds["aod"], ds=ds, col_info={"quality_flag_var": "qa", "qa_bad_values": [2]},
        )

        values = masked.values
        self.assertEqual(values[0, 0], 1.0)  # flag 0 -> known, not bad -> kept
        self.assertTrue(self.np.isnan(values[0, 1]))  # flag 2 -> bad -> masked
        self.assertTrue(self.np.isnan(values[0, 2]))  # flag NaN -> unknown -> masked

    def test_apply_quality_mask_drops_flag_own_integer_fill_sentinel_pixels(self):
        """The QA-flag variable's OWN fill sentinel must be honored before the
        good/bad test. A flag stored as an undecoded integer sentinel (255)
        that xarray never turned to NaN would otherwise satisfy
        ``notnull() & ~isin(bad)`` and let its science pixel through as a real
        'good' observation. Only the already-NaN-flag case was covered before."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset({
            "hcho": (("lat", "lon"), self.np.array([[1.0, 2.0, 3.0]])),
            # flag 0 = good, 2 = bad, 255 = fill sentinel (quality not computed),
            # declared via the flag var's own _FillValue -- never decoded to NaN.
            "qa": (("lat", "lon"), self.np.array([[0, 2, 255]], dtype="int16")),
        })
        ds["qa"].attrs["_FillValue"] = 255

        masked, _ = AggregationService()._apply_quality_mask(
            ds["hcho"], ds=ds, col_info={"quality_flag_var": "qa", "qa_bad_values": [2]},
        )

        values = masked.values
        self.assertEqual(values[0, 0], 1.0)  # flag 0 -> good -> kept
        self.assertTrue(self.np.isnan(values[0, 1]))  # flag 2 -> bad -> masked
        self.assertTrue(self.np.isnan(values[0, 2]))  # flag 255 fill -> unknown -> masked

    def test_apply_quality_mask_honors_a_combined_cf_valid_range_attr(self):
        """CF allows ``valid_range: [min, max]`` INSTEAD of valid_min/
        valid_max, and xarray does not apply it on decode — ignoring it meant
        no range mask at all for products publishing only that spelling."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[-5.0, 1.0], [2.0, 150.0]]),
            dims=("lat", "lon"),
            attrs={"valid_range": [0.0, 100.0]},
            name="aod",
        )

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={})

        values = masked.values
        self.assertTrue(self.np.isnan(values[0, 0]))  # below valid_range min
        self.assertEqual(values[0, 1], 1.0)
        self.assertEqual(values[1, 0], 2.0)
        self.assertTrue(self.np.isnan(values[1, 1]))  # above valid_range max

    def test_explicit_valid_min_max_attrs_win_over_valid_range(self):
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[5.0, 15.0]]),
            dims=("lat", "lon"),
            attrs={"valid_min": 0.0, "valid_max": 10.0, "valid_range": [0.0, 100.0]},
            name="aod",
        )

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={})

        self.assertEqual(masked.values[0, 0], 5.0)
        self.assertTrue(self.np.isnan(masked.values[0, 1]))  # 15 > valid_max 10

    def test_cadence_defaults_to_unknown_for_an_unregistered_collection(self):
        """An off-registry product's cadence is a fact this backend does not
        have. The old "daily" default stamped a false provenance claim ("12
        daily granules" over a year of monthly means) and fed the period-
        label heuristics wrong inputs — the honest answer is "unknown", and
        the granule label says just "granules"."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "some_unregistered_var": (
                    ("time", "lat", "lon"),
                    self.np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
                )
            },
            coords={
                "time": ["2024-01-01", "2024-02-01"],
                "lat": [40.0],
                "lon": [-75.0, -74.0],
            },
        )

        result = AggregationService().aggregate(ds, stat="mean", variable="some_unregistered_var")

        self.assertEqual(result.meta["cadence"], "unknown")
        self.assertIn("2 granules", result.meta["aggregation_label"])
        self.assertNotIn("daily", result.meta["aggregation_label"])

    def test_cf_valid_range_from_packed_attrs_is_scaled_before_masking_decoded_data(self):
        """A scaled int product publishes ``valid_range`` in PACKED units, but
        xarray's mask_and_scale decode leaves that attr untouched while decoding
        the DATA to physical units and moving scale_factor/add_offset into
        ``.encoding``. Comparing the physical field against the packed bound
        (``da <= 30000``) wipes the entire real field. The CF bound must be
        scaled to physical space first, so only genuinely out-of-range cells
        drop."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        # scale_factor 2e13, valid_range packed [0, 30000] -> physical [0, 6e17].
        da = self.xr.DataArray(
            self.np.array([[2.0e13, 4.0e13], [6.0e13, 1.0e18]]),  # already decoded (physical)
            dims=("lat", "lon"),
            name="scaled_no2",
            attrs={"valid_range": [0, 30000]},
        )
        da.encoding["scale_factor"] = 2.0e13
        da.encoding["add_offset"] = 0.0

        result = AggregationService().aggregate(da, variable="scaled_no2")
        values = result.ds["scaled_no2"].values

        # The three in-range physical values survive (packed bound scaled up).
        self.assertEqual(values[0, 0], 2.0e13)
        self.assertEqual(values[0, 1], 4.0e13)
        self.assertEqual(values[1, 0], 6.0e13)
        # Only the genuinely-too-large cell (> 6e17 physical) is masked.
        self.assertTrue(self.np.isnan(values[1, 1]))

    def test_cf_valid_range_scales_via_source_ds_encoding_when_the_working_array_lost_it(self):
        """The real plot/stat path masks geometry with ``.where()`` *before*
        aggregate, and ``.where()`` strips ``.encoding`` off the working array —
        so scale_factor is only still reachable through the unmasked opened
        Dataset (source_ds). resolve_and_mask must read the encoding from there,
        or a decoded scaled product is wiped exactly where users plot it."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        source = self.xr.Dataset(
            {"scaled_no2": (
                ("lat", "lon"),
                self.np.array([[2.0e13, 4.0e13], [6.0e13, 1.0e18]]),
                {"valid_range": [0, 30000]},  # packed -> physical [0, 6e17]
            )},
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
        )
        source["scaled_no2"].encoding["scale_factor"] = 2.0e13
        source["scaled_no2"].encoding["add_offset"] = 0.0

        # The working array is a .where()-derived view: same values, but its
        # own .encoding has been dropped.
        working = source["scaled_no2"].where(source["scaled_no2"].notnull())
        self.assertNotIn("scale_factor", working.encoding)

        masked, _ = AggregationService().resolve_and_mask(working, source_ds=source)
        values = masked.values

        self.assertEqual(values[0, 0], 2.0e13)  # survives — bound scaled to physical
        self.assertTrue(self.np.isnan(values[1, 1]))  # 1e18 > 6e17

    def test_apply_quality_mask_scales_packed_cf_valid_range_from_da_attrs(self):
        """The direct apply_quality_mask fallback (no resolved col_info) reads
        ``valid_range`` straight off ``da.attrs`` — also packed. It must scale
        those bounds by the encoding's scale/offset too, or a decoded scaled
        product is wiped."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[2.0e13, 6.0e13], [1.0e18, 4.0e13]]),  # decoded/physical
            dims=("y", "x"),
            name="scaled_v",
            attrs={"valid_range": [0, 30000]},  # packed -> physical [0, 6e17]
        )
        da.encoding["scale_factor"] = 2.0e13
        da.encoding["add_offset"] = 0.0

        masked, _ = AggregationService()._apply_quality_mask(da, col_info={})
        values = masked.values

        self.assertEqual(values[0, 0], 2.0e13)
        self.assertEqual(values[0, 1], 6.0e13)
        self.assertEqual(values[1, 1], 4.0e13)
        self.assertTrue(self.np.isnan(values[1, 0]))  # 1e18 > 6e17 physical

    def test_apply_quality_mask_does_not_scale_physical_col_info_bounds(self):
        """A scaled product can still carry a curated *physical* valid range in
        col_info (registry/UMM-Var). Those bounds are already in the decoded
        field's units — scaling them by the encoding would be a double-apply
        that wrongly nukes the field. col_info bounds are used verbatim."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[2.0e13, 6.0e13], [8.0e17, 4.0e13]]),  # decoded/physical
            dims=("y", "x"),
            name="scaled_v",
        )
        da.encoding["scale_factor"] = 2.0e13
        da.encoding["add_offset"] = 0.0

        masked, _ = AggregationService()._apply_quality_mask(
            da, col_info={"valid_min": 0.0, "valid_max": 6.0e17},  # already physical
        )
        values = masked.values

        self.assertEqual(values[0, 0], 2.0e13)
        self.assertEqual(values[0, 1], 6.0e13)
        self.assertEqual(values[1, 1], 4.0e13)
        self.assertTrue(self.np.isnan(values[1, 0]))  # 8e17 > physical valid_max


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class QaPassRateCounterTests(unittest.TestCase):
    """T55: the QA pass rate reported to the researcher is counted where the
    mask is actually applied -- from the same boolean condition
    ``apply_quality_mask`` masks with -- so it is structurally incapable of
    disagreeing with the plotted data."""

    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr

    def _tempo_ds(self, values, flags, lat=(10.0, 20.0), lon=(30.0, 40.0)):
        """A TEMPO_NO2-shaped Dataset: science var + sibling QA flag."""
        return self.xr.Dataset(
            {
                "no2": (("lat", "lon"), self.np.array(values, dtype="float64")),
                "main_data_quality_flag": (("lat", "lon"), self.np.array(flags, dtype="int64")),
            },
            coords={"lat": list(lat), "lon": list(lon)},
        )

    @property
    def _pinned(self):
        return {"quality_flag_var": "main_data_quality_flag", "qa_good_values": [0]}

    def test_apply_quality_mask_counts_the_pixels_it_checked_and_the_pixels_that_passed(self):
        """The counters come back from the same ``.where()`` condition the mask
        uses: 3 of 4 pixels carry a good flag."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [0, 0]])

        _, counts = AggregationService()._apply_quality_mask(
            ds["no2"], ds, self._pinned,
        )

        self.assertEqual(counts["checked"], 4)
        self.assertEqual(counts["passing"], 3)

    def test_checked_excludes_cells_already_masked_as_fill_or_out_of_range(self):
        """The denominator is "pixels that had a retrievable value at all",
        not the raw grid size: a fill pixel is not a QA failure and must not
        be counted as one. Of 4 pixels one is fill and one is out of range,
        so only 2 were ever QA-checkable -- and both pass."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds([[-999.0, 500.0], [3.0, 4.0]], [[0, 0], [0, 0]])
        col_info = {**self._pinned, "fill_value": -999.0, "valid_min": 0.0, "valid_max": 100.0}

        _, counts = AggregationService()._apply_quality_mask(ds["no2"], ds, col_info)

        self.assertEqual(counts["checked"], 2)
        self.assertEqual(counts["passing"], 2)

    def test_a_pixel_whose_flag_is_itself_fill_is_reported_as_unknown_not_just_failed(self):
        """Three counters, not two. ``_decode_flag_fill`` nulls the flag's own
        sentinel so an unknown-quality pixel reads as absent, not good -- correct
        masking, but in the *report* it collapses "failed QA" into "QA unknown".
        The third counter keeps them separable: of 4 checked pixels, 2 pass, 1
        genuinely failed, and 1 had no usable flag at all."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "no2": (("lat", "lon"), self.np.array([[1.0, 2.0], [3.0, 4.0]])),
                "main_data_quality_flag": (
                    ("lat", "lon"),
                    self.np.array([[0, 1], [0, 255]], dtype="int64"),
                    {"_FillValue": 255},
                ),
            },
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        _, counts = AggregationService()._apply_quality_mask(
            ds["no2"], ds, self._pinned,
        )

        self.assertEqual(counts["checked"], 4)
        self.assertEqual(counts["passing"], 2)
        self.assertEqual(counts["flag_missing"], 1)

    def test_the_pass_rate_is_cos_latitude_area_weighted_not_a_raw_pixel_ratio(self):
        """Finding #13's doctrine, applied to quality: an unweighted pixel
        count sitting in the same stat grid as an area-weighted mean is an
        inconsistency. Two good pixels at the equator and two bad ones at 80N
        are 50% of the pixels but ~85% of the *area* -- the shrunken poleward
        cells must not be counted as equals."""
        import math

        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds(
            [[1.0, 2.0], [3.0, 4.0]], [[0, 0], [1, 1]], lat=(0.0, 80.0),
        )

        _, counts = AggregationService()._apply_quality_mask(
            ds["no2"], ds, self._pinned,
        )

        w_eq, w_80 = 1.0, math.cos(math.radians(80.0))
        expected = (2 * w_eq) / (2 * w_eq + 2 * w_80)
        self.assertAlmostEqual(counts["pass_rate"], expected, places=9)
        self.assertNotAlmostEqual(counts["pass_rate"], 0.5, places=2)
        # The raw integer counts stay available as the honest "how many
        # observations" fact -- they are not replaced by the weighted fraction.
        self.assertEqual(counts["checked"], 4)
        self.assertEqual(counts["passing"], 2)

    def test_counting_leaves_the_masked_array_exactly_what_the_qa_condition_dictates(self):
        """An honesty feature must not change the science. The counters are
        derived from the same condition the mask applies, never from a
        re-derived mask that could re-enter the masking path -- so the surviving
        cells are exactly the good-flagged ones and nothing else moved."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        # Good flags on the diagonal: [0][0] and [1][1] survive, the rest go.
        ds = self._tempo_ds([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [1, 0]])

        masked, counts = AggregationService()._apply_quality_mask(
            ds["no2"], ds, self._pinned,
        )

        self.np.testing.assert_array_equal(
            masked.values, self.np.array([[1.0, self.np.nan], [self.np.nan, 4.0]]),
        )
        self.assertEqual(counts["passing"], 2)

    def test_qa_flag_variable_names_the_sibling_flag_without_masking_anything(self):
        """plot_tools needs to *identify* the QA flag so it can exclude it from
        the evidence band loop -- its pass rate is already reported once, from
        the mask itself (T55), so a second computation there could only
        disagree. A narrow public accessor for that question, so the caller
        stops reaching across a module seam into a private method.

        Answers with a name the caller can actually use: a flag pinned in
        col_info but absent from the opened view is not reachable, so it is not
        an answer.
        """
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [0, 0]])
        service = AggregationService()

        self.assertEqual(
            service.qa_flag_variable(ds, ds["no2"], self._pinned),
            "main_data_quality_flag",
        )
        self.assertIsNone(
            service.qa_flag_variable(
                ds, ds["no2"], {"quality_flag_var": "not_in_this_file"},
            ),
        )

    def test_provenance_states_the_extent_the_counters_actually_covered(self):
        """A pass rate is only interpretable against the extent it was counted
        over: ``qa_checked_pixels: 4`` says nothing about whether that was a
        whole grid or one cell until the extent says so too.

        Derived from the aligned array the reductions actually ran on, never
        declared by a caller -- a caller cannot state an extent that disagrees
        with what was counted, which is the failure this closes.
        """
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [0, 0]])

        _, provenance = AggregationService().resolve_and_mask(
            ds["no2"], col_info=self._pinned, source_ds=ds,
        )

        self.assertEqual(provenance["qa_counted_extent"], {"lat": 2, "lon": 2})

    def test_a_lazy_array_stays_lazy_and_its_graph_is_computed_exactly_once(self):
        """T51's finding applied here: counting adds an eager reduction to the
        masking path, *before* the reduction the ``aggregate`` phase already
        pays for, on lazily-opened multi-granule bundles. Two guarantees earn
        the feature its keep -- the returned array is still lazy (masking did
        not force the graph), and every counter comes out of ONE
        ``compute``, not one per counter."""
        dask = self._require_dask()
        from tta_backend.preprocessing.aggregation_service import AggregationService

        loads = []

        def _load_science():
            loads.append("science")
            return self.np.array([[1.0, 2.0], [3.0, 4.0]])

        science = self.xr.DataArray(
            dask.array.from_delayed(
                dask.delayed(_load_science)(), shape=(2, 2), dtype="float64",
            ),
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            name="no2",
        )
        ds = self.xr.Dataset({
            "no2": science,
            "main_data_quality_flag": (
                ("lat", "lon"), self.np.array([[0, 1], [0, 0]], dtype="int64"),
            ),
        })
        masked, counts = AggregationService()._apply_quality_mask(
            ds["no2"], ds, self._pinned,
        )

        self.assertEqual(counts["checked"], 4)
        self.assertEqual(len(loads), 1, f"source graph materialized {len(loads)} times")
        self.assertIsNotNone(masked.chunks, "masking forced a lazy array to load")

    def _lazy_bundle(self, granules, flags, loads):
        """A lazily-opened multi-granule bundle: ONE dask chunk per granule,
        each backed by a loader that records its read.

        The single-chunk ``from_delayed`` array above proves the masking path
        alone computes once; it cannot see the cost this models. A real
        HDF5/Zarr-backed bundle is chunked along time, so a second walk of the
        graph is a second *I/O pass* over every granule -- the thing the
        counter's own comment measured at +0.32s synthetically and flagged as
        larger on real storage.
        """
        dask = self._require_dask()

        def _chunk(name, index, values, dtype):
            block = self.np.asarray(values, dtype=dtype)[None, ...]

            def _load():
                loads.append(f"{name}[{index}]")
                return block

            return dask.array.from_delayed(
                dask.delayed(_load)(), shape=block.shape, dtype=dtype,
            )

        science = dask.array.concatenate(
            [_chunk("no2", i, g, "float64") for i, g in enumerate(granules)], axis=0,
        )
        flag = dask.array.concatenate(
            [_chunk("flag", i, f, "int64") for i, f in enumerate(flags)], axis=0,
        )
        return self.xr.Dataset(
            {
                "no2": (("time", "lat", "lon"), science),
                "main_data_quality_flag": (("time", "lat", "lon"), flag),
            },
            coords={
                "time": self.np.array(
                    ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]",
                ),
                "lat": [10.0, 20.0],
                "lon": [30.0, 40.0],
            },
        )

    def test_the_bundle_is_read_once_per_pass_and_there_are_only_two_passes(self):
        """Separate ``compute`` calls share no graph, so each one is a fresh
        I/O pass over a lazily-opened bundle -- the cost the counter's own
        comment flagged as larger than its synthetic +0.32s on real storage.

        Measured through a full ``aggregate()``, the source was read THREE
        times per granule: the counter compute, ``_valid_time_indices``'
        per-timestep ``.values`` scan, and the temporal reduction. The scan is
        now folded into the counter compute, leaving two.

        Two is the floor, not a way-station: the reduction graph cannot be
        built until the valid timesteps are known (cadence-bucket weights and
        the reported granule dates both depend on them), so scan-then-reduce is
        a real data dependency. A third pass reappearing here means someone
        added a compute that could have ridden an existing graph walk.
        """
        from tta_backend.preprocessing.aggregation_service import AggregationService

        loads = []
        ds = self._lazy_bundle(
            granules=[[[1.0, 2.0], [3.0, 4.0]]] * 3,
            flags=[[[0, 1], [0, 0]]] * 3,
            loads=loads,
        )

        result = AggregationService().aggregate(
            ds, variable="no2", col_info=self._pinned,
        )
        # aggregate() must not have forced the reduction -- collapsing the
        # passes by materializing eagerly is the RAM trade this must not make.
        self.assertIsNotNone(
            result.ds["no2"].chunks, "aggregate() forced its result eagerly",
        )
        reduced = result.ds["no2"].values  # now force the temporal reduction

        self.assertEqual(reduced.shape, (2, 2))
        self.assertEqual(result.meta["masking"]["qa_checked_pixels"], 12)
        self.assertEqual(result.meta["masking"]["qa_passing_pixels"], 9)
        for name in ("no2", "flag"):
            reads = [entry for entry in loads if entry.startswith(name)]
            self.assertEqual(
                len(reads), 6,
                f"{name} bundle read {len(reads)} times across 3 granules, "
                f"want 6 (2 passes x 3 granules): {loads}",
            )

    def test_the_folded_scan_drops_the_same_timesteps_the_per_step_scan_did(self):
        """The fused reduction has to BE the scan, not an approximation of it:
        a granule the QA mask emptied must still be dropped from the temporal
        mean and from the reported granule count. Granule 1 is entirely
        bad-flagged, so two of three survive -- and the surviving mean must not
        have averaged in a timestep of NaN."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        loads = []
        ds = self._lazy_bundle(
            granules=[[[2.0, 2.0], [2.0, 2.0]]] * 3,
            flags=[[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]],
            loads=loads,
        )

        result = AggregationService().aggregate(
            ds, variable="no2", col_info=self._pinned,
        )

        self.assertEqual(result.meta["n_granules"], 2)
        self.assertEqual(result.meta["granule_dates"], ["2024-01-01", "2024-01-03"])
        self.np.testing.assert_allclose(
            result.ds["no2"].values, self.np.full((2, 2), 2.0),
        )

    def test_a_timestep_left_holding_only_infinities_is_not_a_valid_timestep(self):
        """``_valid_time_indices`` tested finiteness, not non-nullness, and the
        folded reduction has to keep that distinction: ``notnull()`` counts an
        inf as present, so reusing the ``checked`` counter here would quietly
        promote an all-infinite granule into the temporal mean."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        loads = []
        ds = self._lazy_bundle(
            granules=[
                [[1.0, 1.0], [1.0, 1.0]],
                [[self.np.inf, self.np.inf], [self.np.inf, self.np.inf]],
                [[3.0, 3.0], [3.0, 3.0]],
            ],
            flags=[[[0, 0], [0, 0]]] * 3,
            loads=loads,
        )

        result = AggregationService().aggregate(
            ds, variable="no2", col_info=self._pinned,
        )

        self.assertEqual(result.meta["n_granules"], 2)
        self.np.testing.assert_allclose(
            result.ds["no2"].values, self.np.full((2, 2), 2.0),
        )

    def test_resolve_and_mask_reports_the_realized_pass_rate_as_provenance(self):
        """The rate is a first-class masking-provenance fact, so both chart
        types inherit it through the ``masking`` dict they already carry --
        no new wiring, and no second computation to drift from this one."""
        import math

        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [0, 0]])

        _, provenance = AggregationService().resolve_and_mask(
            ds["no2"], variable="no2", col_info=self._pinned, source_ds=ds,
        )

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        expected = (w10 + 2 * w20) / (2 * w10 + 2 * w20)  # 3 of 4, area-weighted
        self.assertEqual(provenance["qa_checked_pixels"], 4)
        self.assertEqual(provenance["qa_passing_pixels"], 3)
        self.assertEqual(provenance["qa_flag_missing_pixels"], 0)
        self.assertAlmostEqual(provenance["qa_pass_rate"], expected, places=6)
        # The number explains its own denominator wherever it travels (CSV and
        # metadata export, not only JSX).
        self.assertIn("retrievable", provenance["qa_pass_rate_basis"])

    def test_the_pass_rate_is_absent_not_null_when_qa_masking_never_ran(self):
        """Consistent with the existing downgrade-to-not-applied honesty guard:
        a key that isn't there says "QA didn't run", which must stay
        distinguishable from a real 0% pass rate."""
        from tta_backend.datasets.qa_flags import QA_NOT_APPLIED
        from tta_backend.preprocessing.aggregation_service import AggregationService

        da = self.xr.DataArray(
            self.np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            name="no2",
        )

        _, provenance = AggregationService().resolve_and_mask(
            da, variable="no2", col_info=self._pinned,
        )

        self.assertEqual(provenance["qa_status"], QA_NOT_APPLIED)
        self.assertNotIn("qa_pass_rate", provenance)
        self.assertNotIn("qa_checked_pixels", provenance)

    def test_a_scene_with_no_retrievable_pixels_reports_zero_checked_not_a_missing_run(self):
        """A fully fill-covered scene really did have nothing to QA-check.
        That is diagnosable ("no retrievable pixels to check") only if it is
        distinguishable from QA never running -- so the count is reported as a
        real 0 while the undefined rate stays absent."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds(
            [[-999.0, -999.0], [-999.0, -999.0]], [[0, 0], [0, 0]],
        )
        col_info = {**self._pinned, "fill_value": -999.0}

        _, provenance = AggregationService().resolve_and_mask(
            ds["no2"], variable="no2", col_info=col_info, source_ds=ds,
        )

        self.assertEqual(provenance["qa_checked_pixels"], 0)
        self.assertEqual(provenance["qa_passing_pixels"], 0)
        self.assertNotIn("qa_pass_rate", provenance)

    def test_a_per_timestep_series_keeps_a_bad_day_from_averaging_away(self):
        """A single cumulative fraction cannot distinguish a uniform 50% every
        day from one clean day plus one totally lost day -- and the second is a
        data-quality event. The companion series carries every timestep, not
        only the ones that survived to be plotted: a 0%-pass day drops off the
        chart line entirely, which is exactly when it most needs to be visible.
        Timestamps travel alongside so the series can be aligned to a chart."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "no2": (("time", "lat", "lon"), self.np.array([[[1.0, 2.0]], [[3.0, 4.0]]])),
                "main_data_quality_flag": (
                    ("time", "lat", "lon"), self.np.array([[[1, 1]], [[0, 0]]], dtype="int64"),
                ),
            },
            coords={
                "time": self.np.array(
                    ["2024-01-01", "2024-01-02"], dtype="datetime64[ns]",
                ),
                "lat": [10.0],
                "lon": [30.0, 40.0],
            },
        )

        _, provenance = AggregationService().resolve_and_mask(
            ds["no2"], variable="no2", col_info=self._pinned, source_ds=ds,
        )

        self.assertEqual(provenance["qa_pass_rate_by_time"], [0.0, 1.0])
        self.assertEqual(
            provenance["qa_pass_rate_times"],
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        )
        # The cumulative rate is unchanged by the split -- the series is extra
        # resolution, not a redefinition.
        self.assertAlmostEqual(provenance["qa_pass_rate"], 0.5)

    def test_the_count_is_region_scoped_because_callers_narrow_before_masking(self):
        """Every real tool path narrows to its analyzed region *before* masking
        -- an AOI crop on the plot/stat paths, the monitor cell on the
        validation path -- which is the only reason one pass-rate card means the
        same thing on a heatmap, a timeseries and a point series. Pin it: a
        narrowed input must count strictly fewer pixels than the full grid, so a
        future caller that masks before narrowing fails here instead of silently
        reporting a continental rate for a city.

        The extent rides along so the number says which of the two it was --
        reading ``checked`` alone cannot distinguish a 2-cell strip from a whole
        small grid."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self._tempo_ds(
            [[1.0, 2.0], [3.0, 4.0]], [[0, 0], [0, 0]], lat=(10.0, 20.0), lon=(30.0, 40.0),
        )
        service = AggregationService()

        _, full = service._apply_quality_mask(ds["no2"], ds, self._pinned)
        _, cropped = service._apply_quality_mask(
            ds["no2"].sel(lat=[10.0]), ds, self._pinned,
        )

        self.assertEqual(full["checked"], 4)
        self.assertLess(cropped["checked"], full["checked"])
        self.assertEqual(cropped["checked"], 2)
        self.assertEqual(full["counted_extent"], {"lat": 2, "lon": 2})
        self.assertEqual(cropped["counted_extent"], {"lat": 1, "lon": 2})

    def test_a_collection_nobody_pinned_still_gets_a_real_pass_rate(self):
        """collections.yaml pins a QA rule for a minority of collections, but
        ``_resolve_qa_flag_var`` discovers flag variables generically at
        runtime -- so masking already works where no registry entry exists.
        Because the rate is counted at the mask, reporting follows the masking
        rather than the registry, for every present and future collection."""
        from tta_backend.datasets.qa_flags import QA_CF_DETERMINISTIC
        from tta_backend.preprocessing.aggregation_service import AggregationService

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
            },
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        result = AggregationService().aggregate(ds, variable="so2_column")
        masking = result.meta["masking"]

        self.assertEqual(masking["qa_status"], QA_CF_DETERMINISTIC)
        self.assertEqual(masking["qa_checked_pixels"], 4)
        self.assertEqual(masking["qa_passing_pixels"], 2)
        self.assertAlmostEqual(masking["qa_pass_rate"], 0.5, places=6)

    def test_a_product_with_no_flag_band_says_so_instead_of_semantics_unknown(self):
        """TEMPO O3TOT publishes no quality flag variable at all -- granule-
        verified 2026-08-02, 16 bands, none carrying CF flag attributes. Saying
        "semantics unknown" there describes a flag we could not interpret, when
        the truth is there is no flag to interpret. The two cases lead a reader
        to different actions (go pin it / nothing to pin), so they read
        differently."""
        from tta_backend.datasets.qa_flags import QA_NO_FLAG_VARIABLE
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {"column_amount_o3": (("lat", "lon"), self.np.array([[300.0, 310.0], [320.0, 330.0]]))},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

        _masked, provenance = AggregationService().resolve_and_mask(
            ds["column_amount_o3"], variable="column_amount_o3", col_info={}, source_ds=ds,
        )

        self.assertEqual(provenance["qa_status"], QA_NO_FLAG_VARIABLE)

    def _require_dask(self):
        if importlib.util.find_spec("dask") is None:  # pragma: no cover
            self.skipTest("dask is not installed")
        import dask
        import dask.array  # noqa: F401  -- registers the .array attribute

        return dask


if __name__ == "__main__":
    unittest.main()
