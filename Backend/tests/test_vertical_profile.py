"""PRD T56 — the vertical axis as a first-class scientific concept.

Every fact pinned here was measured against a real granule
(``TEMPO_O3PROF_L3_V04_20251001T120743Z_S002.nc``, spike 2026-08-08) rather than
read out of documentation, which is incomplete for this L3 product. Where a
number looks arbitrary, the spike finding it came from is named.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]

# The granule's own vertical grid, top-first, sampled from the spike (finding
# 3): layer 0 sits at ~0.175 hPa / 60 km and the last layer at the surface.
# Six layers instead of 24 keeps the fixtures readable; nothing under test
# depends on the count.
PRESSURES_TOP_DOWN = [0.175, 1.5, 27.0, 130.0, 500.0, 902.0]
ALTITUDES_TOP_DOWN = [60.0, 44.0, 27.0, 13.0, 5.0, 1.0]


class ProfileProductIsRegisteredWithItsMeasuredFactsTests(unittest.TestCase):
    """Phase 1 gate: the profile feature is unreachable until the collection is
    registered, because an unregistered collection has no pinned masking rule
    and therefore masks nothing (see test_collection_registry_qa_coverage)."""

    def _cfg(self):
        from tta_backend.datasets.registry import load_registry

        registry = load_registry()
        self.assertIn("TEMPO_O3PROF", registry, "the ozone profile product is not registered")
        return registry["TEMPO_O3PROF"]

    def test_the_profile_product_is_registered_against_its_larc_collection(self):
        cfg = self._cfg()
        self.assertEqual(cfg.collection_id, "C3685896402-LARC_CLOUD")
        self.assertEqual(cfg.short_name, "TEMPO_O3PROF_L3")
        self.assertEqual(cfg.cadence, "hourly")

    def test_the_science_variable_is_the_layered_ozone_profile(self):
        cfg = self._cfg()
        self.assertEqual(cfg.primary_var, "ozone_profile")
        self.assertEqual(cfg.units, "DU")

    def test_the_vertical_coordinate_siblings_are_requested_alongside_the_science_variable(self):
        """The profile needs three variables retrieved in lockstep: the partial
        columns and the two per-pixel vertical axes. A retrieval that subsets to
        the science variable alone comes back with no axis to plot against."""
        cfg = self._cfg()
        self.assertEqual(
            cfg.variables,
            [
                "product/ozone_profile",
                "support_data/ozone_profile_pressure",
                "support_data/ozone_profile_altitude",
            ],
        )
        self.assertIn("support_data", cfg.groups)
        self.assertIn("product", cfg.groups)

    def test_the_valid_range_is_permissive_enough_to_survive_a_single_layer(self):
        """Finding 7/D7: a partial column runs 0.23-32.9 DU by layer, while the
        sibling O3TOT product's ``valid_min: 50.0`` describes a TOTAL column.
        Copying it here would silently wipe every one of the 24 layers."""
        cfg = self._cfg()
        self.assertLessEqual(cfg.valid_min, 0.0)
        self.assertGreaterEqual(cfg.valid_max, 150.0)
        self.assertAlmostEqual(cfg.fill_value, -1.2676506e30)

    def test_the_product_is_recorded_as_publishing_no_quality_flag(self):
        """D5/finding 5: the L3 producer pre-filtered on fit_RMS, avg_residual
        and retrieval_exit_status before gridding, so there is no per-pixel flag
        band by design -- a fact that must be recorded, not merely absent."""
        from tta_backend.datasets.registry import COLLECTIONS_WITHOUT_QUALITY_FLAG

        cfg = self._cfg()
        self.assertIsNone(cfg.quality_flag_var)
        self.assertIn("TEMPO_O3PROF", COLLECTIONS_WITHOUT_QUALITY_FLAG)


class NarrowingWorksOnAFieldThatHasAVerticalAxisTests(unittest.TestCase):
    """Narrowing selects WHERE to look and must not care how many non-spatial
    axes ride along. ``mask_data_by_geometry`` aligns its rasterized mask by
    dimension NAME, so a (time, lat, lon, layer) field is no harder than a
    (time, lat, lon) one -- but a rank check refused it outright, which is what
    a profile product hits on its first narrowing."""

    def _field(self):
        import numpy as np
        import xarray as xr

        # 2x2 degree grid straddling a box that keeps only the western column.
        return xr.DataArray(
            np.arange(1 * 2 * 2 * 3, dtype="float64").reshape(1, 2, 2, 3),
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": [40.0, 41.0],
                "longitude": [-75.0, -74.0],
                "layer": [0, 1, 2],
            },
            name="ozone_profile",
        )

    def test_a_four_dimensional_field_narrows_to_the_region_with_every_layer_intact(self):
        import numpy as np
        from shapely.geometry import box

        from tta_backend.utils.plotting import mask_data_by_geometry

        field = self._field()
        narrowed = mask_data_by_geometry(field, box(-75.4, 39.6, -74.6, 41.4))

        self.assertIn("layer", narrowed.dims)
        self.assertEqual(narrowed.sizes["layer"], 3)
        # Only the western column is in the region, and T50's crop-before-mask
        # narrows the field to that footprint rather than carrying an
        # all-NaN column into the reduction.
        self.assertEqual(list(narrowed["longitude"].values), [-75.0])
        self.assertTrue(
            np.isfinite(narrowed.values).all(),
            f"kept column lost values: {narrowed.values}",
        )

    def test_narrowing_carries_the_vertical_coordinate_siblings_along(self):
        """The per-pixel pressure/altitude axes arrive as CF auxiliary
        coordinates on the science variable itself, so they must be narrowed by
        the same ``.where``/crop -- not left on the full grid, where they would
        describe pixels the answer is not about."""
        import numpy as np
        from shapely.geometry import box

        from tta_backend.utils.plotting import mask_data_by_geometry

        field = self._field()
        field = field.assign_coords(
            ozone_profile_pressure=(field.dims, np.full(field.shape, 500.0)),
        )
        narrowed = mask_data_by_geometry(field, box(-75.4, 39.6, -74.6, 41.4))

        self.assertIn("ozone_profile_pressure", narrowed.coords)
        self.assertEqual(narrowed["ozone_profile_pressure"].dims, narrowed.dims)


class ReducingOverTheAnalyzedRegionKeepsOneNamedAxisTests(unittest.TestCase):
    """Phase 2: the timeseries and the profile differ only in WHICH axis
    survives the spatial reduction. The reduction itself -- cos(latitude)
    weighting, the per-cell fallback for non-mean statistics, the absent-not-
    zero treatment of an empty slice -- is one piece of previously-corrected
    math and must not fork."""

    def _field(self, values, *, lats=(10.0, 70.0), dims=("time", "lat", "lon")):
        import numpy as np
        import xarray as xr

        arr = np.asarray(values, dtype="float64")
        coords = {"lat": np.linspace(lats[0], lats[1], arr.shape[dims.index("lat")]),
                  "lon": np.linspace(30.0, 40.0, arr.shape[dims.index("lon")])}
        if "time" in dims:
            coords["time"] = np.array(
                ["2024-01-01", "2024-01-02"][: arr.shape[dims.index("time")]],
                dtype="datetime64[ns]",
            )
        if "layer" in dims:
            coords["layer"] = np.arange(arr.shape[dims.index("layer")])
        return xr.DataArray(arr, dims=dims, coords=coords, name="ozone_profile")

    def test_keeping_the_time_axis_reproduces_the_area_weighted_regional_mean(self):
        """The number the timeseries chart plots today. A plain cell mean would
        answer 2.5 and 6.5 here; cos(latitude) weighting is what makes the trend
        line agree with the statistics tool for the identical region."""
        import math

        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

        field = self._field([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        reduced = reduce_keeping_axes(field, keep=("time",), stat="mean")

        w10, w70 = math.cos(math.radians(10.0)), math.cos(math.radians(70.0))
        expected = [
            (1.5 * w10 + 3.5 * w70) / (w10 + w70),
            (5.5 * w10 + 7.5 * w70) / (w10 + w70),
        ]
        self.assertEqual(reduced.dims, ("time",))
        for got, want in zip(reduced.values, expected):
            self.assertAlmostEqual(float(got), want, places=9)

    def test_keeping_time_and_the_vertical_axis_yields_the_profile_intermediate(self):
        """D4's space-then-time order: one area-weighted mean per (timestep,
        layer) BEFORE time is collapsed. Reducing time first would weight a
        densely-sampled hour as heavily as a whole cadence bucket."""
        import math

        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

        # (time=2, layer=2, lat=2, lon=1)
        field = self._field(
            [[[[1.0], [3.0]], [[10.0], [30.0]]],
             [[[2.0], [4.0]], [[20.0], [40.0]]]],
            dims=("time", "layer", "lat", "lon"),
        )
        reduced = reduce_keeping_axes(field, keep=("time", "layer"), stat="mean")

        w10, w70 = math.cos(math.radians(10.0)), math.cos(math.radians(70.0))
        self.assertEqual(reduced.dims, ("time", "layer"))
        self.assertEqual(reduced.shape, (2, 2))
        self.assertAlmostEqual(
            float(reduced.sel(time="2024-01-01", layer=0)),
            (1.0 * w10 + 3.0 * w70) / (w10 + w70), places=9,
        )
        self.assertAlmostEqual(
            float(reduced.sel(time="2024-01-02", layer=1)),
            (20.0 * w10 + 40.0 * w70) / (w10 + w70), places=9,
        )

    def test_a_slice_the_mask_emptied_reads_as_absent_rather_than_zero(self):
        """A timestep or layer with nothing retrievable left has no statistic --
        not a zero. The caller drops it from the series; a zero would plot as a
        real, very clean measurement."""
        import numpy as np

        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

        field = self._field([[[1.0, 2.0], [3.0, 4.0]],
                             [[np.nan, np.nan], [np.nan, np.nan]]])
        reduced = reduce_keeping_axes(field, keep=("time",), stat="mean")

        self.assertTrue(np.isfinite(float(reduced.values[0])))
        self.assertTrue(np.isnan(float(reduced.values[1])))

    def test_statistics_other_than_the_mean_stay_per_cell(self):
        """max/min/median/std describe the cells themselves, not an area, so
        they are deliberately NOT area-weighted -- the same split
        ``conduct_temporal_statistic`` has always made."""
        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

        field = self._field([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        self.assertEqual(list(reduce_keeping_axes(field, keep=("time",), stat="max").values), [4.0, 8.0])
        self.assertEqual(list(reduce_keeping_axes(field, keep=("time",), stat="min").values), [1.0, 5.0])
        self.assertEqual(list(reduce_keeping_axes(field, keep=("time",), stat="median").values), [2.5, 6.5])


def _o3prof_dataset(xr, np, *, layer_values, lat=(10.0, 20.0), lon=(30.0, 40.0), times=("2025-10-01T12:00:00",)):
    """A TEMPO_O3PROF-shaped Dataset, in the shape a real retrieval delivers.

    Two things here are not incidental. The vertical axes arrive as CF
    AUXILIARY COORDINATES, not data variables -- that is how the granule
    declares them and how ``open_handle`` now surfaces them. And layer 0 is the
    TOP of the atmosphere, so any code that assumes index increases upward
    renders the atmosphere inverted.

    ``layer_values`` is the per-layer partial column, broadcast to every pixel
    and timestep unless it is already fully shaped.
    """
    n_layers = len(PRESSURES_TOP_DOWN)
    shape = (len(times), len(lat), len(lon), n_layers)
    values = np.asarray(layer_values, dtype="float64")
    if values.shape != shape:
        values = np.broadcast_to(values, shape).copy()

    axis = np.broadcast_to(np.asarray(PRESSURES_TOP_DOWN), shape).copy()
    altitude = np.broadcast_to(np.asarray(ALTITUDES_TOP_DOWN), shape).copy()
    dims = ("time", "latitude", "longitude", "layer")
    return xr.Dataset(
        {"ozone_profile": (dims, values, {"units": "DU"})},
        coords={
            "time": np.array(list(times), dtype="datetime64[ns]"),
            "latitude": list(lat),
            "longitude": list(lon),
            "ozone_profile_pressure": (dims, axis, {"units": "hPa"}),
            "ozone_profile_altitude": (dims, altitude, {"units": "km"}),
        },
        attrs={"short_name": "TEMPO_O3PROF_L3"},
    )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "vertical profile integration test dependencies are not installed",
)
class VerticalProfileToolTests(unittest.IsolatedAsyncioTestCase):
    """Phase 3: the tool a researcher actually calls. Reduces lat/lon and time
    away, keeps the vertical axis, and reports the physical axis alongside the
    values so the chart never has to guess which end of the array is the sky."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

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
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def _profile(self, handle="obs_prof", location="global", **kwargs):
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_vertical_profile

        emitted = {}
        tool = make_plot_vertical_profile(self.mcp_tools)
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.update(payload=payload),
        ):
            raw = await tool.ainvoke({"handle": handle, "location": location, **kwargs})
        return json.loads(raw), emitted.get("payload")

    def _add(self, handle, **kwargs):
        import numpy as np
        import xarray as xr

        self.volume.add_zarr(handle, lambda: _o3prof_dataset(xr, np, **kwargs))

    async def test_the_profile_reports_one_value_per_vertical_layer(self):
        self._add("obs_prof", layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result, payload = await self._profile()

        self.assertNotIn("error", result)
        self.assertEqual(payload["type"], "profile")
        self.assertEqual(payload["values"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(payload["units"], "DU")

    async def test_the_payload_says_which_end_of_the_array_is_the_sky(self):
        """Risk 2, the single most likely wrong-picture bug: layer 0 is the TOP.
        A chart that assumes index increases upward draws the atmosphere upside
        down, and nothing about the numbers looks wrong. The ordering is
        MEASURED off the pressure axis rather than pinned, so a product ordered
        the other way is drawn correctly by the same code."""
        self._add("obs_prof", layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        _result, payload = await self._profile()

        self.assertEqual(payload["layer_order"], "top_down")
        self.assertEqual(payload["default_axis"], "pressure")
        self.assertEqual(payload["vertical"]["pressure"]["units"], "hPa")
        self.assertEqual(payload["vertical"]["pressure"]["values"], PRESSURES_TOP_DOWN)
        self.assertEqual(payload["vertical"]["altitude"]["units"], "km")
        self.assertEqual(payload["vertical"]["altitude"]["values"], ALTITUDES_TOP_DOWN)
        # Both axes must agree about which way is up, or the toggle flips the
        # picture.
        self.assertEqual(payload["vertical"]["altitude"]["layer_order"], "top_down")

    async def test_a_bottom_up_product_is_reported_bottom_up(self):
        """The ordering key is only worth anything if it can say the other
        thing. Reversing the axis must reverse the answer -- otherwise
        ``top_down`` is a constant with a plausible name."""
        import numpy as np
        import xarray as xr

        def make_ds():
            ds = _o3prof_dataset(xr, np, layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            return ds.isel(layer=slice(None, None, -1))

        self.volume.add_zarr("obs_flipped", make_ds)

        _result, payload = await self._profile(handle="obs_flipped")

        self.assertEqual(payload["layer_order"], "bottom_up")
        self.assertEqual(payload["vertical"]["altitude"]["layer_order"], "bottom_up")

    async def test_the_layers_still_sum_to_the_total_column(self):
        """Finding 7's free correctness invariant. On the real granule the 24
        partial columns sum to 278.71698 DU, exactly the file's own
        total_ozone_column -- so if the reduction drops a layer, double-counts
        one, or mixes up the axis order, the sum stops matching. The reduction
        is linear, so the invariant survives it: the sum of the regional means
        must equal the regional mean of the per-pixel sums."""
        import math

        import numpy as np

        # Distinct values per pixel so an unweighted mean would give a
        # different (wrong) answer than the area-weighted one.
        base = np.array([[[1.0], [4.0]], [[2.0], [8.0]]])  # (lat=2, lon=2, 1)
        layers = np.arange(1, len(PRESSURES_TOP_DOWN) + 1, dtype="float64")
        values = (base * layers)[None, ...]  # (time=1, lat, lon, layer)
        self._add("obs_prof", layer_values=values)

        _result, payload = await self._profile()

        w10, w20 = math.cos(math.radians(10.0)), math.cos(math.radians(20.0))
        # Per-pixel total column, then the SAME cos-lat weighted regional mean.
        totals = base[..., 0] * layers.sum()
        expected_total = (
            (totals[0, 0] + totals[0, 1]) * w10 + (totals[1, 0] + totals[1, 1]) * w20
        ) / (2 * w10 + 2 * w20)

        # Compared as a ratio, not an absolute difference: the payload publishes
        # six significant digits (as every chart payload does), so summing six
        # rounded layers cannot reproduce the total to more than that.
        self.assertAlmostEqual(sum(payload["values"]) / expected_total, 1.0, places=5)

    async def test_the_temporal_mean_weights_cadence_buckets_not_granules(self):
        """D4: 'the average over the period' means each cadence bucket counts
        once. Two scans inside one hour and one in the next must not let the
        busy hour outvote the quiet one -- the same Finding #11 correction the
        map and timeseries paths already carry. A plain granule mean answers
        4.667 here; the bucket-weighted mean answers 6.0."""
        import numpy as np

        values = np.array([1.0, 3.0, 10.0]).reshape(3, 1, 1, 1) * np.ones(
            (1, 1, 1, len(PRESSURES_TOP_DOWN))
        )
        self._add(
            "obs_prof",
            layer_values=values,
            times=("2025-10-01T12:00:00", "2025-10-01T12:30:00", "2025-10-01T13:10:00"),
        )

        _result, payload = await self._profile()

        for value in payload["values"]:
            self.assertAlmostEqual(value, 6.0, places=6)

    async def test_the_product_discloses_that_it_publishes_no_quality_flag(self):
        """D5: this is the rare case where 'no QA mask' is honest rather than a
        gap -- the L3 producer screened the retrievals before gridding. The
        disclosure has to say exactly that, not the ambiguous 'not applied'
        that also covers 'we never looked'."""
        from tta_backend.datasets.qa_flags import QA_NO_FLAG_VARIABLE

        self._add("obs_prof", layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        _result, payload = await self._profile()

        self.assertEqual(payload["masking"]["qa_status"], QA_NO_FLAG_VARIABLE)
        self.assertEqual(payload["provenance"]["masking"]["qa_status"], QA_NO_FLAG_VARIABLE)

    async def test_the_permissive_valid_range_does_not_wipe_the_profile(self):
        """Risk 3: the sibling O3TOT entry's ``valid_min: 50.0`` describes a
        TOTAL column. Applied to partial columns of 0.23-32.9 DU it masks every
        layer, and a fully-masked variable is indistinguishable downstream from
        a scene with no data. Real per-layer magnitudes must survive masking."""
        self._add("obs_prof", layer_values=[0.23, 1.2, 10.0, 29.7, 12.9, 6.4])

        _result, payload = await self._profile()

        self.assertEqual(payload["values"], [0.23, 1.2, 10.0, 29.7, 12.9, 6.4])
        self.assertTrue(all(f == 1.0 for f in payload["valid_fraction"]))

    async def test_a_layer_the_mask_thinned_reports_a_lower_valid_fraction(self):
        """A profile drawn from one surviving pixel at 60 km and every pixel at
        the surface is two measurements sharing an axis, and the line itself
        cannot say so."""
        import numpy as np

        values = np.tile(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (1, 2, 2, 1),
        ).astype("float64")
        # Fill-sentinel the top layer at three of the four pixels.
        values[0, 0, 0, 0] = -1.2676506e30
        values[0, 0, 1, 0] = -1.2676506e30
        values[0, 1, 0, 0] = -1.2676506e30
        self._add("obs_prof", layer_values=values)

        _result, payload = await self._profile()

        self.assertAlmostEqual(payload["valid_fraction"][0], 0.25, places=6)
        self.assertAlmostEqual(payload["valid_fraction"][-1], 1.0, places=6)
        # The surviving pixel still produces a value -- thinned, not absent.
        self.assertAlmostEqual(payload["values"][0], 1.0, places=6)

    async def test_the_per_layer_axis_spread_is_measured_for_this_region(self):
        """Finding 4: the vertical grid is fixed aloft and terrain-following
        near the surface, so a regional-mean axis is exact for the upper layers
        and approximate below. How approximate depends entirely on the region,
        so it is measured per request rather than assumed."""
        import numpy as np
        import xarray as xr

        def make_ds():
            ds = _o3prof_dataset(xr, np, layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            pressure = ds["ozone_profile_pressure"].values.copy()
            # Terrain-following only at the bottom layer: 902 hPa at one pixel,
            # 945 hPa at another -- a 43 hPa spread, matching the spike.
            pressure[0, 1, 1, -1] = 945.0
            return ds.assign_coords(
                ozone_profile_pressure=(ds["ozone_profile_pressure"].dims, pressure, {"units": "hPa"}),
            )

        self.volume.add_zarr("obs_terrain", make_ds)

        _result, payload = await self._profile(handle="obs_terrain")

        spread = payload["vertical"]["pressure"]["spread"]
        self.assertEqual(spread[0], 0.0, "an upper layer's pressure is constant across the region")
        self.assertAlmostEqual(spread[-1], 43.0, places=4)

    async def test_a_variable_with_no_vertical_axis_is_refused_with_an_alternative(self):
        """Pointing this tool at an ordinary 2-D product must say what to use
        instead, not produce a one-point 'profile'."""
        import numpy as np
        import xarray as xr

        def make_ds():
            return xr.Dataset(
                {"column_amount_o3": (("latitude", "longitude"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"units": "DU"})},
                coords={"latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
                attrs={"short_name": "TEMPO_O3TOT_L3"},
            )

        self.volume.add_zarr("obs_flat", make_ds)

        result, payload = await self._profile(handle="obs_flat")

        self.assertIsNone(payload)
        self.assertIn("no vertical dimension", result["error"])
        self.assertIn("conduct_temporal_statistic", result["error"])

    async def test_the_profile_is_a_citable_artifact_of_its_own_kind(self):
        """A profile is not a time series (D2). Sharing the ``ts_`` prefix or
        the timeseries artifact type would let the export and compare paths
        treat pressures as timestamps."""
        self._add("obs_prof", layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result, payload = await self._profile()

        self.assertTrue(payload["chart_id"].startswith("prof_"), payload["chart_id"])
        self.assertEqual(payload["_artifact_refs"][0]["type"], "profile")
        self.assertEqual(result["render_type"], "profile")


class VerticalAxisIdentificationTests(unittest.TestCase):
    """Which variable IS the vertical axis, decided by CF metadata rather than
    by name -- the same doctrine T24 applied to lat/lon. There is exactly one
    of these decisions in the codebase; the profile tool and the
    dimension-refusal suggestion both read it."""

    def _var(self, **attrs):
        import numpy as np
        import xarray as xr

        return xr.DataArray(np.zeros(3), dims=("layer",), attrs=attrs)

    def test_pressure_and_altitude_units_are_recognized(self):
        from tta_backend.utils.geo_utils import vertical_axis_kind

        self.assertEqual(vertical_axis_kind(self._var(units="hPa")), "pressure")
        self.assertEqual(vertical_axis_kind(self._var(units="Pa")), "pressure")
        self.assertEqual(vertical_axis_kind(self._var(units="km")), "altitude")
        self.assertEqual(vertical_axis_kind(self._var(units="m")), "altitude")

    def test_a_variable_that_declares_itself_something_else_is_not_an_axis(self):
        """"m" is a length whether it measures height above the surface or an
        easting on a projected grid. A variable carrying a standard_name has
        already said which; believing its units over its own declaration would
        plot a map projection as an atmosphere."""
        from tta_backend.utils.geo_utils import vertical_axis_kind

        self.assertIsNone(
            vertical_axis_kind(self._var(units="m", standard_name="projection_y_coordinate")),
        )
        self.assertEqual(
            vertical_axis_kind(self._var(units="m", standard_name="altitude")), "altitude",
        )

    def test_a_placeholder_standard_name_counts_as_no_declaration(self):
        """Producers write "none"/"N/A"/"unknown" into an unset standard_name,
        and a placeholder is not a declaration. Treating it as one short-
        circuits the units check and loses a real axis: an hPa coordinate
        labelled ``standard_name: none`` classified as not-vertical at all,
        which reads to the caller as "this product has no pressure axis".

        Fails safe rather than wrong, but it fails on a real file (found while
        reviewing T58's conversion seam)."""
        from tta_backend.utils.geo_utils import vertical_axis_kind

        for placeholder in ("none", "None", "N/A", "n/a", "NA", "unknown", "unset", "-", "   "):
            with self.subTest(standard_name=placeholder):
                self.assertEqual(
                    vertical_axis_kind(self._var(units="hPa", standard_name=placeholder)),
                    "pressure",
                    f"standard_name={placeholder!r} was read as a declaration",
                )
                self.assertEqual(
                    vertical_axis_kind(self._var(units="km", standard_name=placeholder)),
                    "altitude",
                )

    def test_a_placeholder_does_not_make_a_non_vertical_variable_vertical(self):
        """The other direction still has to hold: falling through to units when
        nothing was declared must not start reading every metre-valued variable
        as an altitude. A variable with a placeholder AND non-vertical units is
        still not an axis."""
        from tta_backend.utils.geo_utils import vertical_axis_kind

        self.assertIsNone(vertical_axis_kind(self._var(units="degrees", standard_name="none")))
        self.assertIsNone(vertical_axis_kind(self._var(units="DU", standard_name="N/A")))

    def test_an_unlabelled_axis_is_not_guessed_to_be_vertical(self):
        from tta_backend.utils.geo_utils import vertical_axis_kind

        self.assertIsNone(vertical_axis_kind(self._var()))
        self.assertIsNone(vertical_axis_kind(self._var(units="nm")))

    def test_both_spellings_of_the_vertical_axis_are_classified_alike(self):
        """The classifier already read ``ozone_profile_pressure`` as context (a
        coordinate, like latitude and time) but left ``ozone_profile_altitude``
        unclassified -- the same axis in different units landing in different
        buckets purely because one name happened to match a substring. The
        visible effect is a related-variables panel that offers one half of a
        toggle."""
        from tta_backend.datasets.variable_roles import ROLE_CONTEXT, classify_variable

        for name in ("support_data/ozone_profile_altitude", "support_data/ozone_profile_pressure"):
            role, _confidence = classify_variable(name, group="support_data")
            self.assertEqual(role, ROLE_CONTEXT, name)

    def test_there_is_one_definition_shared_by_both_readers(self):
        """The profile tool and the dimension-refusal suggestion must agree
        about what "vertical" means, or a refusal points at a tool that then
        says the dimension isn't vertical after all.

        Asserted on BEHAVIOUR over real arrays. This used to compare a module
        alias against the function it was assigned from -- a tautology that
        stayed green no matter what the two readers actually did, and which
        pinned an alias nothing used."""
        import numpy as np
        import xarray as xr

        from tta_backend.utils.geo_utils import is_vertical_dim, vertical_axes_for_dim

        def field(axis_attrs):
            da = xr.DataArray(
                np.zeros((2, 2, 3)), dims=("latitude", "longitude", "layer"),
                coords={"latitude": [40.0, 41.0], "longitude": [-75.0, -74.0]},
                name="ozone",
            )
            return da.assign_coords(p=(da.dims, np.zeros((2, 2, 3)), axis_attrs))

        vertical = field({"units": "hPa"})
        self.assertTrue(is_vertical_dim(vertical, "layer"))
        self.assertEqual(vertical_axes_for_dim(vertical, "layer"), {"pressure": "p"})

        # A wavelength axis: the refusal must NOT point at the profile tool, and
        # the profile tool must NOT claim an axis for it.
        spectral = field({"units": "nm"})
        self.assertFalse(is_vertical_dim(spectral, "layer"))
        self.assertEqual(vertical_axes_for_dim(spectral, "layer"), {})


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "vertical profile integration test dependencies are not installed",
)
class ProfileDescribesTheRegionAndNothingElseTests(unittest.IsolatedAsyncioTestCase):
    """Everything a profile reports must describe the analyzed region. The two
    ways that quietly stops being true are a coverage figure whose denominator
    includes the bounding box's empty corners, and a vertical axis averaged
    over the whole granule because it was reached through the unnarrowed
    Dataset."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

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
        self.mcp_tools = await load_raw_mcp_tools(settings)

    async def _profile(self, handle, location):
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_vertical_profile

        emitted = {}
        tool = make_plot_vertical_profile(self.mcp_tools)
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.update(payload=payload),
        ):
            raw = await tool.ainvoke({"handle": handle, "location": location})
        return json.loads(raw), emitted.get("payload")

    async def test_coverage_counts_the_region_not_its_bounding_box(self):
        """"Continental US" is a polygon whose bounding box holds a lot of
        ocean and a slice of Canada. Those cells were never observations this
        answer is about, so counting them as missing turns a complete
        retrieval into a 60%-coverage one -- and a reader sees a data problem
        that does not exist. (The same valid-pct trap the evidence band crop
        was fixed for.)"""
        import numpy as np
        import xarray as xr

        def make_ds():
            lats = np.arange(26.0, 49.1, 1.0)
            lons = np.arange(-124.0, -66.9, 1.0)
            shape = (1, len(lats), len(lons), len(PRESSURES_TOP_DOWN))
            dims = ("time", "latitude", "longitude", "layer")
            return xr.Dataset(
                {"ozone_profile": (dims, np.full(shape, 5.0), {"units": "DU"})},
                coords={
                    "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                    "latitude": lats,
                    "longitude": lons,
                    "ozone_profile_pressure": (
                        dims, np.broadcast_to(np.asarray(PRESSURES_TOP_DOWN), shape).copy(),
                        {"units": "hPa"},
                    ),
                },
                attrs={"short_name": "TEMPO_O3PROF_L3"},
            )

        self.volume.add_zarr("obs_conus", make_ds)

        result, payload = await self._profile("obs_conus", "Continental US")

        self.assertNotIn("error", result)
        for index, fraction in enumerate(payload["valid_fraction"]):
            self.assertAlmostEqual(
                fraction, 1.0, places=6,
                msg=f"layer {index}: every in-region cell held a value, so coverage is complete",
            )

    async def test_a_vertical_axis_published_as_a_data_variable_is_narrowed_too(self):
        """Not every product will declare its axes as CF auxiliary coordinates.
        One that publishes them as ordinary data variables has to be co-located
        with the science variable first -- reached straight off the opened
        Dataset it is still on the FULL granule grid, and the reported axis
        would describe pixels thousands of kilometres from the region."""
        import numpy as np
        import xarray as xr

        def make_ds():
            lats = np.arange(26.0, 49.1, 1.0)
            lons = np.arange(-124.0, -66.9, 1.0)
            shape = (1, len(lats), len(lons), len(PRESSURES_TOP_DOWN))
            dims = ("time", "latitude", "longitude", "layer")
            # Surface pressure that varies hugely with longitude, so a
            # full-grid mean and an in-region mean cannot be confused.
            pressure = np.broadcast_to(np.asarray(PRESSURES_TOP_DOWN), shape).copy()
            pressure[..., -1] = 900.0 + (lons[None, None, :, None] + 124.0)[..., 0] * 2.0
            return xr.Dataset(
                {
                    "ozone_profile": (dims, np.full(shape, 5.0), {"units": "DU"}),
                    "ozone_profile_pressure": (dims, pressure, {"units": "hPa"}),
                },
                coords={
                    "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                    "latitude": lats,
                    "longitude": lons,
                },
                attrs={"short_name": "TEMPO_O3PROF_L3"},
            )

        self.volume.add_zarr("obs_datavar_axis", make_ds)

        _result, payload = await self._profile("obs_datavar_axis", "Continental US")

        surface = payload["vertical"]["pressure"]["values"][-1]
        full_grid_mean = float(np.mean(900.0 + (np.arange(-124.0, -66.9, 1.0) + 124.0) * 2.0))
        self.assertNotAlmostEqual(
            surface, full_grid_mean, places=2,
            msg="the axis still describes the whole granule, not the analyzed region",
        )
        self.assertGreater(payload["vertical"]["pressure"]["spread"][-1], 0.0)


class ProfileExportTests(unittest.IsolatedAsyncioTestCase):
    """A profile exports from its own payload, never by re-reading the source
    granule. That is not a shortcut -- 24 numbers per axis IS the full
    resolution of this chart, so there is nothing a re-read could add, and the
    export keeps working after the source handle is evicted."""

    def setUp(self):
        self.payload = {
            "type": "profile",
            "title": "ozone_profile vertical profile over New Jersey",
            "export": {
                "type": "profile",
                "variable": "ozone_profile",
                "units": "DU",
                "region_name": "New Jersey",
                "aggregation": "mean",
                "source_handles": ["obs_prof"],
                "layers": [0, 1, 2],
                "values": [0.23, 12.0, 30.0],
                "valid_fraction": [0.5, 1.0, 1.0],
                "vertical": {
                    "pressure": {
                        "kind": "pressure", "units": "hPa",
                        "values": [0.175, 130.0, 902.0], "spread": [0.0, 56.6, 43.1],
                        "layer_order": "top_down",
                    },
                    "altitude": {
                        "kind": "altitude", "units": "km",
                        "values": [60.0, 13.0, 1.0], "spread": [0.06, 2.21, 0.42],
                        "layer_order": "top_down",
                    },
                },
                "default_axis": "pressure",
                "layer_order": "top_down",
            },
        }

    async def test_the_csv_carries_both_vertical_axes_and_their_spreads(self):
        """The spread column is what stops a reader treating the regional-mean
        axis as exact. It is exact aloft and approximate near the surface
        (finding 4), and only the per-layer number says which is which."""
        from tta_backend.services.export_service import ExportService

        rows = [row async for row in ExportService().iter_chart_csv_rows(self.payload, {})]

        self.assertEqual(rows[0], [
            "variable", "layer", "value", "units",
            "pressure", "pressure_units", "pressure_spread",
            "altitude", "altitude_units", "altitude_spread",
            "valid_fraction",
        ])
        self.assertEqual(rows[1], [
            "ozone_profile", 0, 0.23, "DU", 0.175, "hPa", 0.0, 60.0, "km", 0.06, 0.5,
        ])
        self.assertEqual(len(rows), 4)

    async def test_the_csv_never_writes_a_pressure_into_a_time_column(self):
        """D2's concrete reason for keeping ``profile`` a payload type of its
        own: routed as a timeseries, these rows would be labelled ``time``."""
        from tta_backend.services.export_service import ExportService

        rows = [row async for row in ExportService().iter_chart_csv_rows(self.payload, {})]

        self.assertNotIn("time", rows[0])

    async def test_the_png_draws_the_profile_without_reading_the_granule(self):
        from tta_backend.services.export_service import ExportService

        png_bytes = await ExportService().build_chart_png(self.payload, {})

        self.assertTrue(png_bytes.startswith(b"\x89PNG"))

    async def test_a_layer_with_no_surviving_data_does_not_break_the_png(self):
        """A layer the mask emptied is reported as absent, not zero -- which
        means the payload carries a null, and the export has to draw the gap
        rather than fail. Real profiles have these: the top layers are the
        thinnest-sampled part of the retrieval."""
        from tta_backend.services.export_service import ExportService

        payload = json.loads(json.dumps(self.payload))
        payload["export"]["values"] = [None, 12.0, 30.0]
        payload["export"]["vertical"]["pressure"]["values"] = [None, 130.0, 902.0]

        png_bytes = await ExportService().build_chart_png(payload, {})

        self.assertTrue(png_bytes.startswith(b"\x89PNG"))

    async def test_the_png_axis_falls_back_to_the_layer_index_with_no_physical_axis(self):
        from tta_backend.services.export_service import ExportService

        payload = json.loads(json.dumps(self.payload))
        payload["export"]["vertical"] = {}
        payload["export"]["default_axis"] = None

        png_bytes = await ExportService().build_chart_png(payload, {})

        self.assertTrue(png_bytes.startswith(b"\x89PNG"))


class RetrievingTheScienceVariableRetrievesItsVerticalAxesTests(unittest.TestCase):
    """A profile needs three variables in lockstep, and which three is a fact
    about the collection -- not a decision to leave to the agent.

    Left to the agent it went wrong on the first real try: it requested
    ``product/ozone_profile`` alone, Harmony obliged, and the chart came back
    with 24 values and no axis to plot them against. This is the same failure
    the quality flag already has a guard for, and the same reasoning: a
    companion the science variable cannot be honestly interpreted without is
    part of requesting it."""

    def test_asking_for_the_profile_asks_for_its_pressure_and_altitude_axes(self):
        from tta_backend.services.retrieval_composites import _pinned_companion_variables

        additions = _pinned_companion_variables(["product/ozone_profile"])

        self.assertEqual(
            sorted(additions),
            ["support_data/ozone_profile_altitude", "support_data/ozone_profile_pressure"],
        )

    def test_a_bare_leaf_request_gets_the_registrys_qualified_spelling(self):
        """The subsetter is handed one variable list; addressing part of it
        bare and part group-qualified is a request no collection's variable
        list looks like. The axes only exist under ``support_data``, so the
        qualified spelling is the only one that resolves."""
        from tta_backend.services.retrieval_composites import _pinned_companion_variables

        additions = _pinned_companion_variables(["ozone_profile"])

        self.assertIn("support_data/ozone_profile_pressure", additions)

    def test_axes_the_caller_already_asked_for_are_not_duplicated(self):
        from tta_backend.services.retrieval_composites import _pinned_companion_variables

        additions = _pinned_companion_variables([
            "product/ozone_profile", "support_data/ozone_profile_pressure",
        ])

        self.assertEqual(additions, ["support_data/ozone_profile_altitude"])

    def test_a_product_with_no_vertical_axis_gains_nothing(self):
        """The guard must not widen every other collection's retrieval."""
        from tta_backend.services.retrieval_composites import _pinned_companion_variables

        additions = _pinned_companion_variables(["product/column_amount_o3"])

        self.assertEqual([a for a in additions if "ozone_profile" in a], [])

    def test_the_quality_flag_guard_still_fires_alongside(self):
        """The two companions share one doctrine and one call site; adding the
        axes must not have displaced the flag."""
        from tta_backend.services.retrieval_composites import _pinned_companion_variables

        additions = _pinned_companion_variables(["product/vertical_column_troposphere"])

        self.assertIn("product/main_data_quality_flag", additions)


class ProfileIsReachableFromWhereResearchersAlreadyAreTests(unittest.TestCase):
    """Phase 6 routing. A capability nobody can find is not a capability, and
    the refusal researchers already hit -- "this variable has an extra
    dimension, pick a value" -- is exactly the moment they wanted the profile.
    Turning that refusal into the discovery path costs one string."""

    def test_the_tool_is_registered_for_the_agent(self):
        from tta_backend.tools.satellite_tools.factory import sanctioned_tool_names

        self.assertIn("plot_vertical_profile", sanctioned_tool_names())

    def test_refusing_a_vertical_dimension_points_at_the_profile_tool(self):
        import numpy as np
        import xarray as xr

        from tta_backend.utils.plotting import _dimension_choice_error

        da = xr.DataArray(
            np.zeros((2, 2, 3)),
            dims=("latitude", "longitude", "layer"),
            coords={
                "latitude": [10.0, 20.0],
                "longitude": [30.0, 40.0],
                "pressure": (("layer",), [1.0, 100.0, 900.0], {"units": "hPa"}),
            },
            name="ozone_profile",
        )

        error = _dimension_choice_error(da, "layer").to_dict()

        self.assertIn("plot_vertical_profile", error["suggestion"])

    def test_refusing_an_ordinary_extra_dimension_does_not(self):
        """Suggesting a profile for a wavelength or a retrieval-attempt index
        would send the researcher somewhere that cannot help them."""
        import numpy as np
        import xarray as xr

        from tta_backend.utils.plotting import _dimension_choice_error

        da = xr.DataArray(
            np.zeros((2, 2, 3)),
            dims=("latitude", "longitude", "wavelength"),
            coords={
                "latitude": [10.0, 20.0],
                "longitude": [30.0, 40.0],
                "wavelength": ("wavelength", [310.0, 340.0, 380.0], {"units": "nm"}),
            },
            name="radiance",
        )

        error = _dimension_choice_error(da, "wavelength").to_dict()

        self.assertNotIn("plot_vertical_profile", error["suggestion"])

    def test_the_profile_product_is_suggested_to_the_agent(self):
        from tta_backend.datasets.preset_collections import get_preset_collections

        short_names = {row["short_name"] for row in get_preset_collections()}
        self.assertIn("TEMPO_O3PROF_L3", short_names)

    def test_the_prompt_tells_the_model_when_to_reach_for_a_profile(self):
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        self.assertIn("plot_vertical_profile", get_earthdata_agent_prompt())

    def test_the_prompt_separates_naming_a_level_from_asking_which_level(self):
        """"At what altitude is the maximum?" is a profile. "The ozone at
        26 km" is a MAP at one level. The first wording of this guidance
        listed 'at what altitude/pressure' among the profile triggers, and the
        model duly answered "plot the ozone at 26 km over New Jersey" with the
        whole profile -- a reasonable reading of an ambiguous instruction.

        Pinned as a behaviour of the prompt because there is nothing else that
        can catch it: both tools succeed, so the wrong one produces a correct
        chart answering a question nobody asked."""
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()
        profile_section = prompt[prompt.index("plot_vertical_profile"):]
        profile_section = profile_section[: profile_section.index("`compare` requires")]

        # It must say that a NAMED level is a single-level map, not a profile.
        self.assertIn("plot_singular", profile_section)
        self.assertRegex(profile_section, r"(?i)names? a (specific|single|particular) (altitude|level)")
        # ...and that a coordinate-less vertical dim is selected by INDEX, so
        # "26 km" is not a value the selector accepts.
        self.assertRegex(profile_section, r"(?i)index")


class MaturityReachesTheArtifactsAPaperIsWrittenFromTests(unittest.TestCase):
    """T57 / D8. The ozone profile is published as BETA -- CMR's own entry
    title says so -- and its user guide states that publishing research on it
    is "not recommended and highly discouraged". Maturity is scientific
    provenance, not descriptive metadata: the place it has to arrive is the
    artifact somebody pulls while writing the paper, not a tooltip they saw
    once in a chat."""

    def test_the_registry_records_maturity_with_the_caveat_that_comes_with_it(self):
        from tta_backend.datasets.registry import load_registry

        cfg = load_registry()["TEMPO_O3PROF"]
        self.assertEqual(cfg.maturity, "beta")
        self.assertTrue(
            len(cfg.maturity_note) > 20,
            "a bare level says nothing a reader can act on; the note is the caveat",
        )

    def test_a_validated_product_is_not_silently_labelled_beta(self):
        """The field only means something if the common case reads differently
        -- and 'unknown' must stay distinguishable from a checked 'validated'."""
        from tta_backend.datasets.registry import load_registry

        self.assertEqual(load_registry()["TEMPO_NO2"].maturity, "unknown")

    def test_maturity_travels_in_a_charts_provenance(self):
        from tta_backend.tools.satellite_tools.plot_tools import _dataset_facts

        facts = _dataset_facts({"short_name": "X", "maturity": "beta", "maturity_note": "caveat here"})

        self.assertEqual(facts["maturity"], "beta")
        self.assertEqual(facts["maturity_note"], "caveat here")

    def test_the_methods_export_states_the_maturity_caveat(self):
        from tta_backend.services.methods_export_service import build_methods_markdown

        markdown = build_methods_markdown(
            artifact_title="ozone_profile vertical profile over New Jersey",
            aoi_description="New Jersey",
            time_window="2025-10-01",
            lineage={"nodes": []},
            citations=[],
            maturity="beta",
            maturity_note="Publication on this Beta product is highly discouraged.",
        )

        self.assertIn("Data maturity", markdown)
        self.assertIn("beta", markdown.lower())
        self.assertIn("Publication on this Beta product is highly discouraged.", markdown)

    def test_the_methods_export_stays_silent_when_maturity_is_unknown(self):
        """An unstated maturity must not become a reassuring one. Saying
        nothing is honest; printing 'unknown' as a section reads like a
        finding."""
        from tta_backend.services.methods_export_service import build_methods_markdown

        markdown = build_methods_markdown(
            artifact_title="t", aoi_description="a", time_window="w",
            lineage={"nodes": []}, citations=[],
        )

        self.assertNotIn("Data maturity", markdown)


if __name__ == "__main__":
    unittest.main()
