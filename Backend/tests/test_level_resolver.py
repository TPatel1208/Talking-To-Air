"""PRD T58 — asking for a level in the units the atmosphere has.

The failure this feature exists to prevent does not crash. It hands back an
ordinary-looking map of the wrong altitude. So the assertions here are weighted
toward what gets *refused* and what gets *disclosed*, not toward the happy path.

Numbers pinned from the Phase 1 spike (2026-08-08, live granules and live
UMM-Var) are named where they appear.
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

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]


class ParsingALevelRequestTests(unittest.TestCase):
    """D1/D11: the unit in the request is the only thing that says which axis
    was meant, on a product publishing more than one. A bare number is
    therefore not an under-specified request, it is a different question."""

    def _parse(self, text):
        from tta_backend.preprocessing.level_resolver import parse_level

        return parse_level(text)

    def test_a_pressure_request_carries_its_value_and_the_axis_it_names(self):
        request = self._parse("500 hPa")

        self.assertEqual(request.kind, "pressure")
        self.assertEqual(request.value, 500.0)
        self.assertEqual(request.units, "hPa")

    def test_an_altitude_request_names_the_altitude_axis(self):
        """The live failure that started T58: "the ozone at 26 km" had no
        spelling the tool accepted."""
        request = self._parse("26 km")

        self.assertEqual(request.kind, "altitude")
        self.assertEqual(request.value, 26.0)

    def test_a_bare_number_is_refused_rather_than_given_a_default_unit(self):
        """D11. On a product publishing both pressure and altitude -- which
        TEMPO_O3PROF does -- the unit is the ONLY thing saying which axis was
        meant. Defaulting would silently answer a different question, which is
        the same class of error as overloading ``dimension_value``."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._parse("500")

        self.assertIn("500", str(caught.exception))

    def test_an_unrecognised_unit_is_refused(self):
        from tta_backend.earthdata_mcp.results import MCPToolError

        with self.assertRaises(MCPToolError):
            self._parse("500 bananas")

    def test_a_request_with_no_number_at_all_is_refused(self):
        from tta_backend.earthdata_mcp.results import MCPToolError

        with self.assertRaises(MCPToolError):
            self._parse("the tropopause")

    def test_the_refusal_answers_with_the_products_own_vocabulary(self):
        """A refusal that lists units the file does not publish sends the
        researcher off to guess again."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._parse_with_axes("500 bananas", {"pressure": "hPa", "altitude": "km"})

        suggestion = caught.exception.to_dict().get("suggestion", "")
        self.assertIn("hPa", suggestion)
        self.assertIn("km", suggestion)

    def _parse_with_axes(self, text, available):
        from tta_backend.preprocessing.level_resolver import parse_level

        return parse_level(text, available=available)


# The regional-mean vertical axes MEASURED by the Phase 1 spike on a real
# 11-timestep TEMPO_O3PROF_L3 retrieval over New Jersey, 2025-10-01/02. Layer 0
# is the TOP of the atmosphere (~0.175 hPa / 60 km), which is what makes a bare
# index meaningless on this product.
MEASURED_PRESSURE_HPA = [
    0.1749, 0.416, 0.5884, 0.8321, 1.1767, 1.6641, 2.3534, 3.3283,
    4.7069, 6.6565, 9.4138, 13.3131, 18.8276, 26.6262, 37.6551, 53.2524,
    75.3103, 107.3553, 167.2165, 260.0476, 367.2513, 504.8014, 682.9116, 896.0633,
]
MEASURED_ALTITUDE_KM = [
    60.2146, 54.236, 51.6644, 49.0459, 46.4125, 43.7983, 41.2332, 38.7295,
    36.2888, 33.9031, 31.5568, 29.2459, 26.9749, 24.7501, 22.5718, 20.4352,
    18.3318, 16.185, 13.4236, 10.5367, 8.0879, 5.7026, 3.3105, 1.0823,
]


def _narrowed_profile(column_pressures=None, altitude=None):
    """A narrowed (time, lat, lon, layer) ozone field carrying its two vertical
    axes as per-pixel CF auxiliary coordinates -- the shape the real granule
    arrives in, and the shape the resolver must work on (finding 1: after
    aggregation the axes no longer exist).

    ``column_pressures`` is one pressure profile per LONGITUDE column, so a test
    can steer per-pixel disagreement exactly rather than approximately: with
    three columns each carries a third of the analyzed pixels. Defaults to the
    measured axis in every column, i.e. total agreement.
    """
    import numpy as np
    import xarray as xr

    columns = column_pressures or [MEASURED_PRESSURE_HPA]
    altitude = MEASURED_ALTITUDE_KM if altitude is None else altitude
    n_layers = len(columns[0])
    n_time, n_lat, n_lon = 2, 4, len(columns)
    shape = (n_time, n_lat, n_lon, n_layers)

    # (lon, layer) broadcast out to (time, lat, lon, layer).
    per_column = np.asarray(columns, dtype="float64")
    field = np.broadcast_to(per_column.reshape(1, 1, n_lon, n_layers), shape).copy()
    alt = np.broadcast_to(np.asarray(altitude, dtype="float64"), shape).copy()

    da = xr.DataArray(
        np.full(shape, 5.0),
        dims=("time", "latitude", "longitude", "layer"),
        coords={
            "time": np.array(["2025-10-01T12", "2025-10-01T13"], dtype="datetime64[ns]"),
            "latitude": [40.0, 40.5, 41.0, 41.5],
            "longitude": np.linspace(-75.0, -74.0, n_lon),
        },
        name="ozone_profile",
    )
    return da.assign_coords(
        ozone_profile_pressure=(da.dims, field, {"units": "hPa"}),
        ozone_profile_altitude=(da.dims, alt, {"units": "km"}),
    )


def _by_latitude(row_pressures):
    """A profile field whose vertical grid varies by LATITUDE rather than
    longitude, so a test can tell an area-weighted agreement fraction apart
    from a raw pixel count."""
    import numpy as np
    import xarray as xr

    n_layers = len(row_pressures[0])
    n_time, n_lat, n_lon = 1, len(row_pressures), 2
    shape = (n_time, n_lat, n_lon, n_layers)
    per_row = np.asarray(row_pressures, dtype="float64")
    field = np.broadcast_to(per_row.reshape(1, n_lat, 1, n_layers), shape).copy()
    alt = np.broadcast_to(np.asarray(MEASURED_ALTITUDE_KM, dtype="float64"), shape).copy()

    da = xr.DataArray(
        np.full(shape, 5.0),
        dims=("time", "latitude", "longitude", "layer"),
        coords={
            "time": np.array(["2025-10-01T12"], dtype="datetime64[ns]"),
            # A deliberately extreme span: cos(0 deg) = 1.0 against
            # cos(75 deg) = 0.259, so an equal pixel COUNT is a very unequal
            # area, and the two definitions cannot be confused.
            "latitude": [0.0, 75.0],
            "longitude": [-75.0, -74.0],
        },
        name="ozone_profile",
    )
    return da.assign_coords(
        ozone_profile_pressure=(da.dims, field, {"units": "hPa"}),
        ozone_profile_altitude=(da.dims, alt, {"units": "km"}),
    )


def _shifted(profile, layers):
    """The measured profile with its values rolled ``layers`` positions, so a
    column resolves a given request to a different layer than its neighbours
    while staying a physically ordered, monotonic axis."""
    import numpy as np

    base = np.asarray(profile, dtype="float64")
    ratio = base[-1] / base[0]
    # Geometric shift: multiply the whole column by the per-layer ratio raised
    # to ``layers``, which slides it along its own (near-exponential) grid.
    step = ratio ** (1.0 / (len(base) - 1))
    return (base * step ** layers).tolist()


class ResolvingAPhysicalLevelToALayerTests(unittest.TestCase):
    """D4: a requested level becomes a single layer INDEX, resolved against the
    regional-mean axis, so it can go through the existing selection path
    unchanged."""

    def _resolve(self, level, da=None, dim="layer"):
        from tta_backend.preprocessing.level_resolver import resolve_level

        return resolve_level(da if da is not None else _narrowed_profile(), dim, level)

    def test_a_pressure_request_resolves_to_the_nearest_layer(self):
        """Spike: 500 hPa over New Jersey resolves to layer 21 (504.8 hPa)."""
        resolution = self._resolve("500 hPa")

        self.assertEqual(resolution.index, 21)

    def test_an_altitude_request_resolves_against_the_altitude_axis(self):
        """Spike: 26 km resolves to layer 12 (26.97 km) -- the request that
        started T58, now answerable in the units it was asked in."""
        resolution = self._resolve("26 km")

        self.assertEqual(resolution.index, 12)

    def test_the_resolution_discloses_the_level_it_actually_landed_on(self):
        """D5/finding 4: level error is an axis of wrongness INDEPENDENT of
        agreement. 850 hPa lands on a layer whose regional mean is 896.06 hPa
        -- a 46 hPa discrepancy at 100% agreement. A result that reports only
        agreement would call that perfect."""
        resolution = self._resolve("850 hPa")

        self.assertEqual(resolution.index, 23)
        self.assertAlmostEqual(resolution.resolved_level, 896.0633, places=3)
        self.assertAlmostEqual(resolution.level_error, 46.0633, places=3)
        self.assertEqual(resolution.units, "hPa")
        self.assertEqual(resolution.kind, "pressure")
        self.assertEqual(resolution.requested, 850.0)

    def test_the_resolution_discloses_how_many_pixels_agree_with_the_chosen_layer(self):
        """D5: the honest artifact for a regional-mean resolution is a number,
        not a refusal -- "layer 21; 100% of analyzed pixels resolve to this
        same layer". Without a per-pixel spread every pixel agrees, which is
        the fact this pins."""
        resolution = self._resolve("500 hPa")

        self.assertEqual(resolution.dominant_fraction, 1.0)
        self.assertIsNone(resolution.runner_up)
        self.assertEqual(resolution.margin, 1.0)


    def test_every_layer_round_trips_to_itself(self):
        """Spike check 4: requesting the value that SITS at layer k must return
        k, for every layer and both axes. This tests the resolver, not just
        discovery -- an off-by-one or a silent unit scaling passes every other
        assertion here and fails this one."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = _narrowed_profile()
        for kind, units, axis in (
            ("pressure", "hPa", MEASURED_PRESSURE_HPA),
            ("altitude", "km", MEASURED_ALTITUDE_KM),
        ):
            for layer, value in enumerate(axis):
                with self.subTest(kind=kind, layer=layer):
                    resolution = resolve_level(narrowed, "layer", f"{value} {units}")
                    self.assertEqual(resolution.index, layer)

    def test_a_request_in_a_different_unit_of_the_same_kind_is_converted(self):
        """A request in Pa against an axis in hPa must be CONVERTED, not
        compared raw. Comparing raw is precisely the mistake
        ``_select_dim_nearest`` already refuses to make silently -- 50000
        compared against an axis running 0.17..896 would snap to the deepest
        layer and look entirely plausible."""
        resolution = self._resolve("50000 Pa")

        self.assertEqual(resolution.index, 21)
        self.assertAlmostEqual(resolution.resolved_level, 504.8014, places=3)

    def test_an_altitude_request_in_metres_is_converted(self):
        resolution = self._resolve("26000 m")

        self.assertEqual(resolution.index, 12)

    def test_a_level_outside_the_products_own_range_is_refused_not_snapped(self):
        """The axis runs 0.17-896 hPa. A 2000 hPa request is not "the bottom
        layer", it is a request this product cannot answer -- and snapping to
        the edge is how a units mismatch becomes a plausible wrong map."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("2000 hPa")

        message = caught.exception.to_dict()["message"]
        self.assertIn("2000", message)
        self.assertIn("896", message)

    def test_the_resolution_carries_a_selector_the_existing_path_understands(self):
        """The Architectural Constraint: the resolved layer goes through the
        EXISTING selection seam, unchanged. That seam reads a coordinate-less
        dimension by position, so on this product the selector is the index."""
        resolution = self._resolve("500 hPa")

        self.assertEqual(resolution.selector_value, 21)

    def test_a_dimension_that_has_a_coordinate_is_selected_by_its_coordinate_value(self):
        """MERRA-2's ``lev`` IS a coordinate, in hPa. The same selection seam
        reads a coordinate-bearing dimension by VALUE, so handing it an index
        there would silently select level 21 hPa instead of layer 21 -- an
        ordinary-looking map of the wrong altitude, which is the exact failure
        this feature exists to eliminate."""
        narrowed = _narrowed_profile().assign_coords(layer=("layer", MEASURED_PRESSURE_HPA))

        resolution = self._resolve("500 hPa", da=narrowed)

        self.assertEqual(resolution.index, 21)
        self.assertAlmostEqual(resolution.selector_value, MEASURED_PRESSURE_HPA[21])

    def test_asking_for_an_axis_the_product_does_not_publish_is_refused(self):
        """D12: pressure-only products exist. A km request against one has not
        been under-specified, it has been asked of the wrong file."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        narrowed = _narrowed_profile().drop_vars("ozone_profile_altitude")

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("26 km", da=narrowed)

        suggestion = caught.exception.to_dict().get("suggestion", "")
        self.assertIn("hPa", suggestion)


def _merra_shaped(lev_attrs, lev_values, edge_heights=None):
    """A MERRA-2-shaped field: the vertical dimension IS a coordinate, unlike
    TEMPO_O3PROF's bare index. Attributes reproduce what the live UMM-Var
    actually publishes for each product (Phase 1 gate, 2026-08-08)."""
    import numpy as np
    import xarray as xr

    shape = (1, 2, 2, len(lev_values))
    da = xr.DataArray(
        np.full(shape, 5.0),
        dims=("time", "lat", "lon", "lev"),
        coords={
            "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
            "lat": ("lat", [40.0, 41.0], {"units": "degrees_north"}),
            "lon": ("lon", [-75.0, -74.0], {"units": "degrees_east"}),
            "lev": ("lev", lev_values, lev_attrs),
        },
        name="O3",
    )
    if edge_heights is None:
        return da
    return da.assign_coords(H=(
        da.dims,
        np.broadcast_to(np.asarray(edge_heights, dtype="float64"), shape).copy(),
        {"units": "m", "long_name": "edge_heights"},
    ))


# MERRA-2 pressure levels and the geopotential edge heights that go with them.
_MERRA_LEV_HPA = [1000.0, 850.0, 500.0, 300.0, 100.0, 10.0]
_MERRA_H_METRES = [100.0, 1500.0, 5500.0, 9200.0, 16000.0, 31000.0]


class WorkingOnProductsBeyondTheOneItWasBuiltAgainstTests(unittest.TestCase):
    """The gate measured seven products; these pin the behaviour on the shapes
    that differ structurally from TEMPO_O3PROF. All attribute values are the
    live UMM-Var records, not invented ones."""

    def _resolve(self, narrowed, level):
        from tta_backend.preprocessing.level_resolver import resolve_level

        return resolve_level(narrowed, "lev", level)

    def test_a_pressure_level_product_resolves_against_its_real_coordinate(self):
        """M2I3NPASM: ``lev`` genuinely is hPa, and it is a dimension
        coordinate rather than a per-pixel field."""
        narrowed = _merra_shaped({"units": "hPa", "long_name": "vertical level"}, _MERRA_LEV_HPA)

        resolution = self._resolve(narrowed, "300 hPa")

        self.assertEqual(resolution.index, 3)
        self.assertEqual(resolution.axis_variable, "lev")
        self.assertEqual(resolution.selector_value, 300.0)

    def test_the_same_product_answers_an_altitude_request_from_its_height_field(self):
        """M2I3NPASM also publishes ``H`` (edge_heights, metres) spanning the
        same dimension. The two axes must agree about which layer a level is:
        300 hPa and 9.2 km are the same place."""
        narrowed = _merra_shaped(
            {"units": "hPa", "long_name": "vertical level"}, _MERRA_LEV_HPA,
            edge_heights=_MERRA_H_METRES,
        )

        by_pressure = self._resolve(narrowed, "300 hPa")
        by_altitude = self._resolve(narrowed, "9.2 km")

        self.assertEqual(by_pressure.index, by_altitude.index)
        self.assertEqual(by_altitude.axis_variable, "H")
        self.assertEqual(by_altitude.units, "m")

    def test_a_hybrid_model_level_product_is_refused_not_resolved(self):
        """M2I3NVASM is the case the PRD names as scientifically dangerous: its
        ``lev`` is a hybrid-eta level number 1..72, not a pressure. CF metadata
        refuses it for free -- ``standard_name: model_layers`` is not a vertical
        standard name, and it wins over the units outright."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        narrowed = _merra_shaped(
            {"units": "layer", "standard_name": "model_layers", "long_name": "vertical level"},
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

        with self.assertRaises(MCPToolError):
            self._resolve(narrowed, "500 hPa")

    def test_the_resolved_selector_round_trips_through_the_real_selection_seam(self):
        """The Architectural Constraint end to end on a coordinate-bearing
        dimension: whatever ``selector_value`` says must make the EXISTING
        ``_select_dim_nearest`` land on the layer the resolver chose."""
        from tta_backend.utils.plotting import _select_dim_nearest

        narrowed = _merra_shaped({"units": "hPa", "long_name": "vertical level"}, _MERRA_LEV_HPA)

        for requested, expected in (("1000 hPa", 0), ("300 hPa", 3), ("10 hPa", 5)):
            with self.subTest(requested=requested):
                resolution = self._resolve(narrowed, requested)
                selected = _select_dim_nearest(narrowed, "lev", resolution.selector_value)
                self.assertEqual(resolution.index, expected)
                self.assertEqual(float(selected["lev"]), _MERRA_LEV_HPA[expected])


class RefusingAnAxisWhoseScaleIsUnknownTests(unittest.TestCase):
    """The `resolved-WRONG` case, found by reviewing the conversion seam.

    ``vertical_axis_kind`` classifies a variable by ``standard_name`` FIRST, and
    only falls back to units. So a product declaring ``standard_name:
    air_pressure`` is accepted as a pressure axis whatever its units say --
    including a unit the converter has never heard of. Comparing the request to
    that axis unconverted is a silent factor-of-ten error that reports 100%
    dominance and looks like an ordinary map.
    """

    def _kpa_axis(self, levels_kpa):
        import numpy as np
        import xarray as xr

        shape = (1, 2, 2, len(levels_kpa))
        da = xr.DataArray(
            np.full(shape, 5.0),
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": [40.0, 41.0],
                "longitude": [-75.0, -74.0],
            },
            name="ozone",
        )
        return da.assign_coords(p=(
            da.dims,
            np.broadcast_to(np.asarray(levels_kpa, dtype="float64"), shape).copy(),
            {"units": "kPa", "standard_name": "air_pressure"},
        ))

    def test_kilopascals_are_converted_rather_than_refused(self):
        """kPa was the unit that exposed the original bug and it is a real CF
        unit, so the fix was to teach the converter, not to refuse it. 1..101 kPa
        is 10..1010 hPa -- a whole atmosphere -- and a 50 hPa request is 5 kPa,
        which is layer 1. Before the fix it answered layer 3 (50 kPa = 500 hPa),
        off by a factor of ten and disclosed as unanimous."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = self._kpa_axis([1.0, 5.0, 20.0, 50.0, 85.0, 101.0])

        resolution = resolve_level(narrowed, "layer", "50 hPa")

        self.assertEqual(resolution.index, 1)
        self.assertAlmostEqual(resolution.resolved_level, 5.0)
        self.assertEqual(resolution.units, "kPa")

    def test_an_axis_in_units_the_converter_cannot_read_is_refused(self):
        """The remaining route to an unconvertible axis is ``standard_name``,
        which wins over units outright -- so a variable can be classified as
        pressure while carrying a unit no table knows. Guessing parity there is
        the factor-of-ten bug; refuse instead."""
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = self._kpa_axis([1.0, 5.0, 20.0, 50.0, 85.0, 101.0])
        narrowed = narrowed.assign_coords(p=(
            narrowed["p"].dims, narrowed["p"].values,
            {"units": "torr", "standard_name": "air_pressure"},
        ))

        with self.assertRaises(MCPToolError) as caught:
            resolve_level(narrowed, "layer", "50 hPa")

        self.assertIn("torr", caught.exception.to_dict()["message"])

    def test_the_conversion_table_covers_every_unit_the_classifier_accepts(self):
        """Two hand-maintained vocabularies that must agree. A unit added to
        the classifier but not the converter is not a crash -- it is a silently
        unconverted comparison, which is the bug above."""
        from tta_backend.preprocessing.level_resolver import _TO_CANONICAL
        from tta_backend.utils.geo_utils import ALTITUDE_UNITS, PRESSURE_UNITS

        for kind, vocabulary in (("pressure", PRESSURE_UNITS), ("altitude", ALTITUDE_UNITS)):
            with self.subTest(kind=kind):
                self.assertEqual(
                    {unit.lower() for unit in vocabulary},
                    set(_TO_CANONICAL[kind]),
                    f"{kind} units known to geo_utils and to the converter have drifted",
                )


class RefusingALevelThatCannotHonestlyBeCalledOneTests(unittest.TestCase):
    """D6: the refusal rule is DERIVED from the Phase 1 spike, not invented.

    Two criteria, both falsifiable, neither a percentage somebody liked:

    1. A plurality that is not a MAJORITY means most of the analyzed region is
       on some other layer, so calling the answer "300 hPa" is affirmatively
       misleading. This sits below every dominance the spike measured (minimum
       83.1%), so it refuses nothing observed to be sound.
    2. If the regional-mean layer is not the layer most pixels would pick, that
       single index misrepresents the region. Needs no threshold at all. All 13
       spike measurements agreed.
    """

    def _resolve(self, level, columns):
        from tta_backend.preprocessing.level_resolver import resolve_level

        return resolve_level(_narrowed_profile(column_pressures=columns), "layer", level)

    def test_agreement_is_area_weighted_like_every_other_regional_fraction(self):
        """This codebase has ONE definition of "what fraction of the analyzed
        region" -- the cos(latitude)-weighted one (CONTEXT.md's QA pass rate,
        aggregation_service.cos_lat_weights). Dominance is the same kind of
        quantity, and counting pixels instead of area would make the disclosure
        disagree with the regional mean it is describing: half the CELLS can be
        a quarter of the AREA.

        Two latitude rows, equal in count. The equator row weighs cos(0) = 1.0,
        the 75 deg row cos(75) = 0.259, so the equator row's layer must come back
        as ~79% of the region rather than 50%.
        """
        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = _by_latitude([MEASURED_PRESSURE_HPA, _shifted(MEASURED_PRESSURE_HPA, 1)])

        resolution = resolve_level(narrowed, "layer", "300 hPa")

        import math

        equator, high = math.cos(0.0), math.cos(math.radians(75.0))
        self.assertAlmostEqual(resolution.dominant_fraction, equator / (equator + high), places=6)
        self.assertAlmostEqual(resolution.runner_up_fraction, high / (equator + high), places=6)

    def test_a_clearly_dominated_layer_resolves_with_its_disclosure_rather_than_refusing(self):
        """The spike's 300 hPa case: 83.1% dominant, 16.9% runner-up. Clearly
        dominated -- the honest artifact is a number, not a refusal (D5/D6)."""
        columns = [MEASURED_PRESSURE_HPA, MEASURED_PRESSURE_HPA, _shifted(MEASURED_PRESSURE_HPA, 1)]

        resolution = self._resolve("300 hPa", columns)

        self.assertAlmostEqual(resolution.dominant_fraction, 2 / 3, places=6)
        self.assertIsNotNone(resolution.runner_up)
        self.assertAlmostEqual(resolution.runner_up_fraction, 1 / 3, places=6)

    def test_a_level_no_majority_of_the_region_agrees_on_is_refused(self):
        """Three columns each resolving to a different layer: the plurality is
        33%, so two thirds of the region is somewhere else. Labelling the map
        with the requested level would be a claim about data it does not show."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        columns = [
            _shifted(MEASURED_PRESSURE_HPA, -3),
            MEASURED_PRESSURE_HPA,
            _shifted(MEASURED_PRESSURE_HPA, 3),
        ]

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("300 hPa", columns)

        message = str(caught.exception)
        self.assertIn("300", message)

    def test_the_refusal_reports_the_split_it_refused_on(self):
        """A refusal that does not say WHY leaves the researcher with no way to
        judge whether a narrower region would work."""
        from tta_backend.earthdata_mcp.results import MCPToolError

        columns = [
            _shifted(MEASURED_PRESSURE_HPA, -3),
            MEASURED_PRESSURE_HPA,
            _shifted(MEASURED_PRESSURE_HPA, 3),
        ]

        with self.assertRaises(MCPToolError) as caught:
            self._resolve("300 hPa", columns)

        payload = caught.exception.to_dict()
        self.assertIn("33", payload["message"] + payload.get("suggestion", ""))



@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "satellite tools factory test dependencies are not installed",
)
class AskingPlotSingularForAPhysicalLevelTests(unittest.IsolatedAsyncioTestCase):
    """Phase 4, and the Architectural Constraint's own test.

    Physical-level resolution must be *semantically equivalent* to direct layer
    selection: the resolver computes an index, and that index goes through the
    existing selection path. Two ways of asking for the same layer that
    disagreed about what they were built from would leave a researcher
    comparing results across tools with no way to see why.
    """

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
        self.volume.add_zarr("obs_profile", self._make_dataset)

    @staticmethod
    def _make_dataset():
        """A layered ozone field on a real-shaped grid, carrying its two
        per-pixel vertical axes as CF auxiliary coordinates -- TEMPO_O3PROF's
        actual structure, with the spike's measured axis values."""
        import numpy as np
        import xarray as xr

        n_time, n_lat, n_lon, n_layers = 2, 3, 4, len(MEASURED_PRESSURE_HPA)
        shape = (n_time, n_lat, n_lon, n_layers)
        # A value that identifies its own layer, so a wrong selection is
        # visible in the plotted numbers rather than only in the metadata.
        values = np.broadcast_to(
            np.arange(n_layers, dtype="float64") * 10.0, shape,
        ).copy()
        pressure = np.broadcast_to(np.asarray(MEASURED_PRESSURE_HPA), shape).copy()
        altitude = np.broadcast_to(np.asarray(MEASURED_ALTITUDE_KM), shape).copy()
        dims = ("time", "lat", "lon", "layer")
        return xr.Dataset(
            {"ozone_profile": (dims, values, {"units": "DU"})},
            coords={
                "time": np.array(["2025-10-01T12", "2025-10-01T13"], dtype="datetime64[ns]"),
                "lat": [39.0, 40.0, 41.0],
                "lon": [-76.0, -75.0, -74.0, -73.0],
                "ozone_profile_pressure": (dims, pressure, {"units": "hPa"}),
                "ozone_profile_altitude": (dims, altitude, {"units": "km"}),
            },
        )

    def _tool(self, name):
        from tta_backend.tools.satellite_tools.factory import build_satellite_tools

        return {t.name: t for t in build_satellite_tools(self.mcp_tools)}[name]

    async def _plot(self, **kwargs):
        emitted = {}
        plot_singular = self._tool("plot_singular")
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.update(payload=payload),
        ):
            result = await plot_singular.ainvoke({
                "handle": "obs_profile", "location": "global", **kwargs,
            })
        return json.loads(result), emitted.get("payload")

    async def test_a_physical_level_produces_the_same_artifact_as_the_layer_index(self):
        """The Architectural Constraint, asserted: same values, same statistics,
        same granule count, same date range."""
        by_level, level_payload = await self._plot(level="500 hPa")
        by_index, index_payload = await self._plot(dimension="layer", dimension_value=21)

        self.assertNotIn("error", by_level, by_level)
        self.assertNotIn("error", by_index, by_index)
        self.assertEqual(level_payload["values"], index_payload["values"])
        self.assertEqual(
            level_payload["aggregation_meta"]["n_granules"],
            index_payload["aggregation_meta"]["n_granules"],
        )
        self.assertEqual(
            level_payload["provenance"]["granule_dates"],
            index_payload["provenance"]["granule_dates"],
        )
        self.assertEqual(
            level_payload["provenance"]["start_date"],
            index_payload["provenance"]["start_date"],
        )

    async def test_the_plotted_values_are_the_requested_layers(self):
        """Each layer carries its own index x 10, so selecting 500 hPa (layer
        21) must plot 210 -- not a mean over layers, and not layer 21 of some
        other ordering."""
        _, payload = await self._plot(level="500 hPa")

        flat = [v for row in payload["values"] for v in row if v is not None]
        self.assertTrue(flat)
        self.assertTrue(all(v == 210.0 for v in flat), sorted(set(flat)))

    async def test_the_level_disclosure_reaches_provenance(self):
        """D5: dominance, runner-up, margin and level error are what let a
        reader judge whether this map would have looked different done
        properly. They belong with the result, not in a log line."""
        _, payload = await self._plot(level="850 hPa")

        disclosure = payload["provenance"]["level_resolution"]
        self.assertEqual(disclosure["index"], 23)
        self.assertEqual(disclosure["kind"], "pressure")
        self.assertEqual(disclosure["units"], "hPa")
        self.assertAlmostEqual(disclosure["requested"], 850.0)
        self.assertAlmostEqual(disclosure["resolved_level"], 896.0633, places=3)
        self.assertAlmostEqual(disclosure["level_error"], 46.0633, places=3)
        self.assertEqual(disclosure["dominant_fraction"], 1.0)

    async def test_an_unparseable_level_is_refused_before_any_data_is_reduced(self):
        by_level, _ = await self._plot(level="500")

        self.assertIn("error", by_level)

    async def test_asking_for_a_level_on_a_product_with_no_vertical_axis_is_refused(self):
        import xarray as xr

        self.volume.add_zarr("obs_flat", lambda: xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        ))
        emitted = {}
        plot_singular = self._tool("plot_singular")
        with patch(
            "tta_backend.tools.satellite_tools.plot_tools.emit_chart",
            lambda payload: emitted.update(payload=payload),
        ):
            result = await plot_singular.ainvoke({
                "handle": "obs_flat", "location": "global", "level": "500 hPa",
            })

        self.assertIn("error", json.loads(result))

    async def test_giving_both_a_level_and_a_dimension_value_is_refused(self):
        """They are different requests -- 'level' names a physical level,
        'dimension_value' names a coordinate value or an index. Silently
        preferring one is exactly the overloading D1 exists to prevent."""
        result, _ = await self._plot(level="500 hPa", dimension="layer", dimension_value=21)

        error = result["error"]
        self.assertIn("level", error["message"])
        self.assertIn("dimension_value", error["message"])

    async def test_a_level_outside_the_products_range_is_refused_end_to_end(self):
        """The refusal has to survive the whole tool path, not just the
        resolver -- an edge snap here is a plausible-looking map."""
        result, _ = await self._plot(level="2000 hPa")

        self.assertIn("error", result)

    async def test_an_ordinary_map_carries_no_level_disclosure(self):
        """A chart that never resolved a physical level must not grow an empty
        level section; the Metadata tab keys its whole block off its absence."""
        _, payload = await self._plot(dimension="layer", dimension_value=21)

        self.assertNotIn("level_resolution", payload["provenance"])


class MeasuringTheAnalyzedRegionAndNotItsBoundingBoxTests(unittest.TestCase):
    """Review finding A1, found by both reviewers independently.

    ``mask_data_by_geometry`` ends in ``data_array.where(mask)``, and xarray's
    ``.where`` masks the DATA while leaving auxiliary coordinates untouched. The
    vertical axis rides along as an auxiliary coordinate, so every number the
    disclosure reported -- the regional-mean level, the level error, the
    dominance vote, and the majority refusal that is this feature's safety
    mechanism -- was computed over the cropped BOUNDING BOX, including cells the
    map renders as no-data.

    T56's "valid_fraction denominator was the bbox" defect landing again in a
    new place, and a contradiction of CONTEXT.md's own definition of dominance
    as a fraction *of the analyzed region*.
    """

    def _split_region(self):
        """A bbox whose western half is outside the analyzed region. The
        in-region half sits at 850 hPa exactly; the out-of-region half is
        400-600 hPa, far enough that including it changes the answer."""
        import numpy as np
        import xarray as xr

        shape = (1, 2, 4, 3)
        axis = np.empty(shape, dtype="float64")
        axis[:, :, :2, :] = np.asarray([400.0, 500.0, 600.0])    # western, outside
        axis[:, :, 2:, :] = np.asarray([700.0, 850.0, 1000.0])   # eastern, inside

        science = np.full(shape, 5.0)
        science[:, :, :2, :] = np.nan  # what the geometry mask did to the data

        da = xr.DataArray(
            science,
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [40.0, 41.0], {"units": "degrees_north"}),
                "longitude": ("longitude", [-77.0, -76.0, -75.0, -74.0], {"units": "degrees_east"}),
            },
            name="ozone",
        )
        return da.assign_coords(p=(da.dims, axis, {"units": "hPa"}))

    def test_a_level_every_in_region_pixel_sits_on_resolves(self):
        """Every pixel the map actually shows is at exactly 850 hPa. Scored over
        the bounding box the axis runs 550-800 hPa, and an 850 hPa request was
        refused as out of range -- a refusal caused entirely by pixels that are
        not in the region."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(self._split_region(), "layer", "850 hPa")

        self.assertEqual(resolution.index, 1)
        self.assertAlmostEqual(resolution.resolved_level, 850.0)
        self.assertAlmostEqual(resolution.level_error, 0.0)

    def test_out_of_region_pixels_do_not_vote_on_dominance(self):
        """The majority rule is the safety mechanism, so the population it
        counts has to be the analyzed region."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(self._split_region(), "layer", "850 hPa")

        self.assertEqual(resolution.dominant_fraction, 1.0)
        self.assertIsNone(resolution.runner_up)
        # 2 latitudes x 2 in-region longitudes x 1 timestep.
        self.assertEqual(resolution.n_pixels, 4)


class ToleratingAnIncompleteVerticalColumnTests(unittest.TestCase):
    """Review finding A2. Dropping every column with ANY non-finite layer means
    one all-fill layer refuses every level request for the whole product --
    routine in profile retrievals, where the top layer fails at high solar
    zenith angle and the bottom sits below terrain. The regional mean already
    handled it correctly; only the all-or-nothing column filter broke."""

    def _with_dead_layer(self, dead):
        import numpy as np

        narrowed = _narrowed_profile()
        axis = narrowed["ozone_profile_pressure"].values.copy()
        axis[..., dead] = np.nan
        return narrowed.assign_coords(
            ozone_profile_pressure=(narrowed.dims, axis, {"units": "hPa"}),
        )

    def test_one_dead_layer_does_not_refuse_a_level_the_others_resolve(self):
        from tta_backend.preprocessing.level_resolver import resolve_level

        for dead in (0, 23):
            with self.subTest(dead_layer=dead):
                resolution = resolve_level(self._with_dead_layer(dead), "layer", "500 hPa")
                self.assertEqual(resolution.index, 21)
                self.assertGreater(resolution.n_pixels, 0)

    def test_a_region_with_no_usable_column_at_all_still_refuses(self):
        """The existing refusal is right for a region with nothing in it; it was
        only firing far too eagerly."""
        import numpy as np

        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = _narrowed_profile()
        narrowed = narrowed.assign_coords(ozone_profile_pressure=(
            narrowed.dims,
            np.full_like(narrowed["ozone_profile_pressure"].values, np.nan),
            {"units": "hPa"},
        ))

        with self.assertRaises(MCPToolError):
            resolve_level(narrowed, "layer", "500 hPa")


class RefusingWhenTheAxisScaleIsNotKnowableTests(unittest.TestCase):
    """Review finding #1: the companion door to the kPa bug already fixed.

    When the axis publishes NO units, ``axis_units`` fell back to the REQUEST's
    units, making the conversion an identity -- so a Pa-valued axis declaring
    only ``standard_name: air_pressure`` answered a 500 hPa request with the
    470 Pa layer, disclosed as 100% agreement and 29 hPa of error. The
    vocabulary drift-guard structurally cannot catch this: there is no unit to
    be missing from the table."""

    def _pascal_axis(self, attrs):
        import numpy as np
        import xarray as xr

        pascals = [17.49, 41.6, 58.84, 83.21, 117.67, 166.41, 235.34, 332.83, 470.69, 665.65]
        shape = (1, 2, 2, len(pascals))
        da = xr.DataArray(
            np.full(shape, 5.0),
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [40.0, 41.0], {"units": "degrees_north"}),
                "longitude": ("longitude", [-75.0, -74.0], {"units": "degrees_east"}),
            },
            name="ozone",
        )
        return da.assign_coords(p=(
            da.dims, np.broadcast_to(np.asarray(pascals), shape).copy(), attrs,
        ))

    def test_an_axis_that_publishes_no_units_is_refused(self):
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        with self.assertRaises(MCPToolError) as caught:
            resolve_level(self._pascal_axis({"standard_name": "air_pressure"}), "layer", "500 hPa")

        self.assertIn("unit", caught.exception.to_dict()["message"].lower())

    def test_an_axis_with_blank_units_is_refused_too(self):
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        with self.assertRaises(MCPToolError):
            resolve_level(
                self._pascal_axis({"standard_name": "air_pressure", "units": "   "}),
                "layer", "500 hPa",
            )


class ARequestThatTheProductCannotActuallyResolveTests(unittest.TestCase):
    """Review finding A4: level error is unbounded, and it is ANTI-correlated
    with the statistic that is bounded. Agreement is high precisely because the
    layers are far apart, so the worst level errors arrive at 100% dominance --
    a 130 hPa request returns a 107 hPa map, unanimously.

    The scale-free way to ask "could this product have done better?" is the
    error as a fraction of the local layer spacing. Above half a gap the request
    sits nearer the middle of the gap than to any layer the product has."""

    def test_a_level_almost_exactly_between_two_layers_says_so(self):
        """Layers 17 and 18 sit at 107.4 and 167.2 hPa, so 138 hPa is nearly
        halfway between them and neither layer is much of an answer.

        DISCLOSED, not refused. The PRD's own worked example -- 300 hPa
        resolving to layer 19 with 40 hPa of error -- has to keep resolving, and
        it scores 0.86 against this request's 0.98. Any refusal threshold lives
        in that narrow band, which is precisely the invented number D6 rejects.
        The honest artifact here is the number."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "138 hPa")

        self.assertEqual(resolution.index, 18)
        self.assertGreater(resolution.level_error_fraction_of_gap, 0.9)

    def test_the_prds_worked_example_still_resolves(self):
        """300 hPa -> layer 19, 40 hPa away. D5 names this as the case where the
        honest artifact is a number rather than a refusal, so a level-error gate
        that broke it would be a regression against the spec."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "300 hPa")

        self.assertEqual(resolution.index, 19)
        self.assertLess(resolution.level_error_fraction_of_gap, 0.9)

    def test_a_level_close_to_a_real_layer_still_resolves(self):
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "500 hPa")

        self.assertEqual(resolution.index, 21)

    def test_the_resolution_discloses_the_error_against_the_layer_spacing(self):
        """A bare "4.8 hPa" means nothing without knowing the layers are ~140
        hPa apart there. The normalized figure is what makes it readable, and it
        is what the refusal is gated on."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "500 hPa")

        self.assertLess(resolution.level_error_fraction_of_gap, 0.5)
        self.assertGreater(resolution.level_error_fraction_of_gap, 0.0)


class DisclosingHowFarApartThePixelsActuallyAreTests(unittest.TestCase):
    """Review findings A3 and B1/B8.

    D6's majority floor is necessary but not sufficient: the escape is the
    MULTI-modal region, not the bimodal one the derivation examined. A 51/25/24
    split across three vertical grids passes both criteria while 49% of the area
    sits ~150 hPa from what was asked, because the two minority groups pull the
    mean in opposite directions -- exactly when a mean is least representative.

    A non-adjacent runner-up needs no threshold to be alarming: on every real
    measurement the runner-up was the neighbouring layer."""

    def _tri_modal(self):
        """51% plain / 25% valley / 25% plateau -- an ordinary terrain-following
        product over a mountain front."""
        import numpy as np
        import xarray as xr

        plain = [1000.0, 850.0, 700.0]
        valley = [1100.0, 1000.0, 850.0]
        plateau = [850.0, 700.0, 550.0]
        columns = [plain] * 6 + [valley] * 3 + [plateau] * 3
        shape = (1, 1, len(columns), 3)
        axis = np.asarray(columns, dtype="float64").reshape(1, 1, len(columns), 3)

        da = xr.DataArray(
            np.full(shape, 5.0),
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [40.0], {"units": "degrees_north"}),
                "longitude": ("longitude", np.linspace(-77.0, -74.0, len(columns)), {"units": "degrees_east"}),
            },
            name="ozone",
        )
        return da.assign_coords(p=(da.dims, axis, {"units": "hPa"}))

    def test_a_multi_modal_region_is_refused_by_the_majority_floor(self):
        """The 51/25/24 escape the review constructed: three vertical grids, the
        two minority groups pulling the mean in opposite directions so it lands
        somewhere no large group actually is."""
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        with self.assertRaises(MCPToolError):
            resolve_level(self._tri_modal(), "layer", "850 hPa")

    def test_a_non_adjacent_runner_up_is_refused_even_with_a_clear_majority(self):
        """Isolated from the majority floor: 60/40 clears it comfortably, but
        the two groups are fourteen layers apart. On every real measurement the
        runner-up was the NEIGHBOURING layer, so a distant one is evidence the
        region has two unrelated vertical structures in it -- and no threshold
        is needed to say that one index cannot describe both."""
        import numpy as np
        import xarray as xr

        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.preprocessing.level_resolver import resolve_level

        near = list(MEASURED_PRESSURE_HPA)
        far = list(MEASURED_PRESSURE_HPA)
        # Six columns resolve 500 hPa to layer 21; four to layer 7, by shifting
        # those columns so layer 7 carries the 500 hPa value.
        far[7] = 500.0
        columns = [near] * 6 + [far] * 4
        shape = (1, 1, len(columns), len(near))
        axis = np.asarray(columns, dtype="float64").reshape(shape)
        da = xr.DataArray(
            np.full(shape, 5.0),
            dims=("time", "latitude", "longitude", "layer"),
            coords={
                "time": np.array(["2025-10-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [40.0], {"units": "degrees_north"}),
                "longitude": ("longitude", np.linspace(-77.0, -74.0, len(columns)), {"units": "degrees_east"}),
            },
            name="ozone",
        )
        da = da.assign_coords(p=(da.dims, axis, {"units": "hPa"}))

        with self.assertRaises(MCPToolError) as caught:
            resolve_level(da, "layer", "500 hPa")

        self.assertIn("apart", caught.exception.to_dict()["message"])

    def test_the_resolution_discloses_the_spread_of_the_layer_it_chose(self):
        """Dominance near 100% only bounds the spread relative to the layer GAP.
        On a product with 100 hPa gaps a layer ranging 260-340 hPa across the
        region still scores ~100%, and "270.9 hPa, 100% agreement" hides it.
        ``_profile_axis_block`` already discloses this for the profile."""
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "500 hPa")

        self.assertEqual(resolution.resolved_level_spread, 0.0)

    def test_a_layer_that_varies_across_the_region_reports_its_spread(self):

        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = _narrowed_profile(column_pressures=[MEASURED_PRESSURE_HPA] * 3)
        axis = narrowed["ozone_profile_pressure"].values.copy()
        # Layer 21 alone made to vary by +-20 hPa across longitude.
        axis[:, :, 0, 21] -= 20.0
        axis[:, :, -1, 21] += 20.0
        narrowed = narrowed.assign_coords(
            ozone_profile_pressure=(narrowed.dims, axis, {"units": "hPa"}),
        )

        resolution = resolve_level(narrowed, "layer", "500 hPa")

        self.assertEqual(resolution.index, 21)
        self.assertAlmostEqual(resolution.resolved_level_spread, 40.0, places=6)


class ReportingTheSampleWithoutMisstatingItTests(unittest.TestCase):
    """Review finding A5: ``dominant_fraction`` is cos(latitude)-weighted while
    ``n_pixels`` is a raw count, and both landed in one sentence -- "91.3% of 40
    analyzed pixels". By count it is 50%. The count is a fine field; it must not
    be the denominator of an area percentage."""

    def test_the_excluded_fraction_of_the_region_is_disclosed(self):
        """Review finding B3: with scattered fill, n_pixels fell 40 -> 4 while
        dominance stayed 1.000 and the message never changed. "100% of 4
        analyzed pixels" beside a map built from 40 needs its denominator."""
        import numpy as np

        from tta_backend.preprocessing.level_resolver import resolve_level

        narrowed = _narrowed_profile(column_pressures=[MEASURED_PRESSURE_HPA] * 4)
        axis = narrowed["ozone_profile_pressure"].values.copy()
        axis[:, :, 0, :] = np.nan  # one of four longitude columns entirely fill
        narrowed = narrowed.assign_coords(
            ozone_profile_pressure=(narrowed.dims, axis, {"units": "hPa"}),
        )

        resolution = resolve_level(narrowed, "layer", "500 hPa")

        self.assertGreater(resolution.excluded_fraction, 0.0)
        self.assertLess(resolution.excluded_fraction, 1.0)

    def test_a_complete_region_excludes_nothing(self):
        from tta_backend.preprocessing.level_resolver import resolve_level

        resolution = resolve_level(_narrowed_profile(), "layer", "500 hPa")

        self.assertEqual(resolution.excluded_fraction, 0.0)
