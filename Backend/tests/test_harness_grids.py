"""The grid opener: retrieval grids masked by area and quality rules, derived grids as stored."""
import datetime
import math
import tempfile
import unittest
from typing import Annotated, Any
from unittest import mock

import numpy as np
import xarray as xr

from tta_backend.datasets.qa_flags import QA_NO_FLAG_VARIABLE, QA_VERIFIED
from tta_backend.datasets.registry import COLLECTIONS_WITHOUT_QUALITY_FLAG, load_registry
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.errors import MCPToolError
from tta_backend.domain.earthdata.opening import combine
from tta_backend.harness.capabilities import Call, Capability, CapabilityResult, Params, handle_param, invoke
from tta_backend.harness.capabilities.grids import open_grid
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, HandleError, HandleRecord, ServicePayload
from tta_backend.harness.run_context import Region, RunEnv
from tta_backend.harness.testing import FakePlaces, box_region, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)
LATS = np.arange(38.5, 42.01, 0.5)
LONS = np.arange(-76.0, -72.99, 0.5)
NJ = box_region("New Jersey", -75.6, 38.9, -73.9, 41.4, region_type="polygon")
# A triangle inside the box, so the geometry mask removes cells the bounds crop keeps.
TRIANGLE = {
    "type": "Polygon",
    "coordinates": [[[-75.6, 38.9], [-73.9, 38.9], [-75.6, 41.4], [-75.6, 38.9]]],
}


def granule(
    *,
    short_name: str | None = "TEMPO_NO2_L3",
    science: str = "vertical_column_troposphere",
    flag: str | None = "main_data_quality_flag",
    flags: Any = 0,
    value: float = 2.0e15,
    day: str = "2024-06-01",
    lon360: bool = True,
    units: str = "molecules/cm^2",
) -> xr.Dataset:
    shape = (1, LATS.size, LONS.size)
    variables: dict[str, Any] = {science: (("time", "latitude", "longitude"), np.full(shape, value), {"units": units})}
    if flag is not None:
        flag_values = np.broadcast_to(np.asarray(flags), shape).astype("int16")
        variables[flag] = (("time", "latitude", "longitude"), flag_values)
    lons = LONS + 360 if lon360 else LONS
    attrs = {"short_name": short_name} if short_name else {}
    return xr.Dataset(
        variables,
        coords={"time": [np.datetime64(f"{day}T15:00:00")], "latitude": LATS, "longitude": lons},
        attrs=attrs,
    )


class _NoParams(Params):
    pass


async def _noop(call: Call[_NoParams]) -> CapabilityResult:
    return call.done()


PROBE = Capability(
    name="operations.probe", description="Probe.", params=_NoParams, output_kind=None,
    label="Probing", method="Probed.", retrieves=False, retry_safe=True, run=_noop,
)


class _GridParams(Params):
    grid: Annotated[str, handle_param("grid")]


async def _open(call: Call[_GridParams]) -> CapabilityResult:
    await open_grid(call, call.handle("grid"))
    return call.done()


OPEN = Capability(
    name="operations.open_probe", description="Open a grid.", params=_GridParams, output_kind=None,
    label="Opening", method="Opened.", retrieves=False, retry_safe=True, run=_open,
)


def layered_retrieval(*, axes_as_variables: bool, day: str = "2025-10-01") -> xr.Dataset:
    """A TEMPO_O3PROF-shaped retrieval; its vertical axes as data variables when asked."""
    from tests.test_vertical_profile import _o3prof_dataset

    ds = _o3prof_dataset(xr, np, layer_values=[1.0] * 6, times=(f"{day}T12:00:00",))
    if axes_as_variables:
        ds = ds.reset_coords(["ozone_profile_pressure", "ozone_profile_altitude"])
    return ds


def probe_call(env: RunEnv) -> Call[_NoParams]:
    return Call(capability=PROBE, params=_NoParams(), recorded={}, env=env, resolved={}, secret=None, seq=0)


class OpenerCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.files: dict[str, xr.Dataset] = {}
        self.places = FakePlaces({"new jersey": NJ})

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.files[record.id]

        self.env = await make_run_env(
            self._tmp.name, context=make_run_context(), open_retrieval=open_retrieval, places=self.places,
        )
        self.area = (await invoke(self.env, areas.resolve, {"place": "New Jersey"})).handle

    async def retrieval(
        self, handle: str, ds: xr.Dataset, *, product: str = "TEMPO_NO2", variable: str | None = None,
        area: str | None = None,
    ) -> HandleRecord:
        self.files[handle] = ds
        area_record = await self.env.resolver.get("user-a", area or self.area)
        region = {**area_record.descriptor.region, "area": area_record.id}  # type: ignore[dict-item]
        cfg = load_registry()[product]
        return await self.env.context.storage.handles.record_retrieval(HandleRecord(
            id=handle, kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
            descriptor=Descriptor(
                summary="A retrieval", product=product, unit=cfg.units, region=region,
                masking={"variable": variable or cfg.primary_var.rsplit("/", 1)[-1]},
            ),
            created_at=NOW, payload=ServicePayload(),
        ))


class RetrievalGridTests(OpenerCase):
    async def test_a_pinned_quality_flag_masks_failing_cells_and_is_disclosed(self):
        flags = np.zeros((1, LATS.size, LONS.size), dtype="int16")
        flags[0, 2, 2] = 1
        record = await self.retrieval("obs_tempo", granule(flags=flags))

        grid = await open_grid(probe_call(self.env), record)

        self.assertEqual(grid.masking["qa_status"], QA_VERIFIED)
        self.assertEqual(grid.masking["variable"], "vertical_column_troposphere")
        self.assertEqual(grid.masking["quality_flag"], "main_data_quality_flag")
        self.assertEqual(grid.masking["region"], "geometry")
        self.assertLess(grid.masking["qa_pass_rate"], 1.0)
        self.assertEqual(set(grid.data.dims), {"time", "lat", "lon"})
        self.assertEqual(grid.data.attrs["units"], "molecules/cm^2")
        # Longitudes were normalised, then cropped to the area.
        self.assertTrue((grid.data["lon"] >= -75.6).all() and (grid.data["lon"] <= -73.9).all())
        failing = grid.data.sel(lat=LATS[2], lon=LONS[2]).values
        self.assertTrue(np.isnan(failing).all())
        self.assertEqual(int(np.isfinite(grid.data).sum()), grid.region_cells - 1)

    async def test_cells_outside_the_area_geometry_are_masked(self):
        self.places.regions["triangle"] = Region(
            name="Triangle", bounds=(-75.6, 38.9, -73.9, 41.4), geometry=TRIANGLE, region_type="polygon",
        )
        triangle = (await invoke(self.env, areas.resolve, {"place": "Triangle"})).handle
        record = await self.retrieval("obs_tempo", granule(), area=triangle)
        grid = await open_grid(probe_call(self.env), record)
        # The north-east corner of the box is outside the triangle.
        self.assertTrue(np.isnan(grid.data.sel(lat=41.0, lon=-74.0).values).all())
        self.assertTrue(np.isfinite(grid.data.sel(lat=39.0, lon=-75.5).values).all())
        self.assertLess(grid.region_cells, grid.data.sizes["lat"] * grid.data.sizes["lon"])

    async def test_the_variable_on_the_handle_is_selected_and_never_another(self):
        ds = granule().assign(other=(("time", "latitude", "longitude"), np.ones((1, LATS.size, LONS.size))))
        record = await self.retrieval("obs_tempo", ds)
        grid = await open_grid(probe_call(self.env), record)
        self.assertEqual(grid.data.name, "vertical_column_troposphere")

        missing = await self.retrieval("obs_missing", ds.drop_vars("vertical_column_troposphere"))
        with self.assertRaises(MCPToolError) as caught:
            await open_grid(probe_call(self.env), missing)
        self.assertEqual(caught.exception.category, "contract")

    async def test_a_file_without_identity_uses_the_products_registry_entry(self):
        flags = np.zeros((1, LATS.size, LONS.size), dtype="int16")
        flags[0, 3, 3] = 1
        record = await self.retrieval("obs_plain", granule(short_name=None, flags=flags))
        grid = await open_grid(probe_call(self.env), record)
        self.assertEqual(grid.masking["qa_status"], QA_VERIFIED)
        self.assertEqual(grid.masking["registry_entry"], "chosen_product")
        self.assertTrue(np.isnan(grid.data.sel(lat=LATS[3], lon=LONS[3]).values).all())

    async def test_a_file_with_identity_says_so_in_the_provenance(self):
        record = await self.retrieval("obs_tempo", granule())
        grid = await open_grid(probe_call(self.env), record)
        self.assertEqual(grid.masking["registry_entry"], "file_identity")

    async def test_a_plan_with_unusable_bounds_is_refused(self):
        for bounds in ([-75.6, "38.9", -73.9, 41.4], [-75.6, 38.9, -73.9], [-75.6, 38.9, -73.9, math.nan]):
            with self.subTest(bounds=bounds):
                record = await self.retrieval("obs_tempo", granule())
                region = {**record.descriptor.region, "bounds": bounds}  # type: ignore[dict-item]
                broken = record.model_copy(update={"descriptor": record.descriptor.model_copy(update={"region": region})})
                with self.assertRaises(StepFailure) as caught:
                    await open_grid(probe_call(self.env), broken)
                self.assertEqual(caught.exception.category, FailureCategory.INTERNAL_ERROR)
                self.assertEqual(caught.exception.reason, (
                    "operations.probe: the retrieval grid obs_tempo does not record a usable bounds."
                ))

    async def test_an_evicted_area_or_retrieval_is_input_evicted(self):
        record = await self.retrieval("obs_tempo", granule())

        async def gone(user_id: str, handle: HandleRecord) -> Any:
            raise HandleError("evicted", handle.id)

        evicting = await make_run_env(
            self._tmp.name, context=self.env.context, open_retrieval=gone, places=self.places,
        )
        with self.assertRaises(StepFailure) as caught:
            await invoke(evicting, OPEN, {"grid": record.id})
        self.assertEqual(caught.exception.category, FailureCategory.INPUT_EVICTED)
        self.assertEqual(caught.exception.result, {"evicted": ["obs_tempo"]})

        await self.env.resolver.evict("user-a", self.area, NOW)
        with self.assertRaises(StepFailure) as area_gone:
            await invoke(self.env, OPEN, {"grid": record.id})
        self.assertEqual(area_gone.exception.result, {"evicted": [self.area]})

    async def test_pinned_vertical_axes_published_as_variables_ride_as_coordinates(self):
        ds = layered_retrieval(axes_as_variables=True)
        record = await self.retrieval("obs_prof", ds, product="TEMPO_O3PROF", area=await self.global_area())
        grid = await open_grid(probe_call(self.env), record)
        self.assertIn("ozone_profile_pressure", grid.data.coords)
        self.assertIn("ozone_profile_altitude", grid.data.coords)
        self.assertEqual(grid.data["ozone_profile_pressure"].attrs["units"], "hPa")

        second = await self.retrieval("obs_prof2", layered_retrieval(axes_as_variables=True, day="2025-10-02"),
                                      product="TEMPO_O3PROF", area=await self.global_area())
        joined = combine([grid, await open_grid(probe_call(self.env), second)], ["2025-10-01", "2025-10-02"])
        self.assertEqual(joined.data["ozone_profile_pressure"].sizes["time"], 2)
        stored = joined.dataset()
        self.assertIn("ozone_profile_altitude", stored.coords)

    async def global_area(self) -> str:
        self.places.regions["global"] = box_region("Global", -180.0, -90.0, 180.0, 90.0, region_type="bounding_box")
        return (await invoke(self.env, areas.resolve, {"place": "Global"})).handle

    async def test_heavy_work_goes_through_compute(self):
        record = await self.retrieval("obs_tempo", granule())
        before = self.env.compute.runs  # type: ignore[attr-defined]
        await open_grid(probe_call(self.env), record)
        self.assertEqual(self.env.compute.runs, before + 1)  # type: ignore[attr-defined]


class DerivedGridTests(OpenerCase):
    async def test_a_derived_grid_is_opened_as_stored_without_masking_again(self):
        stored = xr.Dataset(
            {"anomaly": (("lat", "lon"), np.array([[1.0, np.nan], [3.0, 4.0]]), {"units": "ppb"})},
            coords={"lat": [40.0, 40.5], "lon": [-75.0, -74.5]},
        )
        masking = {"variable": "anomaly", "qa_status": QA_VERIFIED, "region_cells": 4}
        record = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="grid", data=stored,
            descriptor=Descriptor(summary="An anomaly", masking=masking), operation="operations.anomaly",
            created_at=NOW,
        )
        with mock.patch.object(AggregationService, "resolve_and_mask", side_effect=AssertionError("masked again")):
            grid = await open_grid(probe_call(self.env), record)
        np.testing.assert_array_equal(grid.data.values, stored["anomaly"].values)
        self.assertEqual(grid.masking, masking)
        self.assertEqual(grid.region_cells, 4)

    async def test_a_non_grid_is_refused(self):
        area = await self.env.resolver.get("user-a", self.area)
        with self.assertRaises(TypeError):
            await open_grid(probe_call(self.env), area)


def _value(cfg: Any) -> float:
    low, high = cfg.valid_min, cfg.valid_max
    if math.isfinite(low) and math.isfinite(high):
        value = (low + high) / 2
    elif math.isfinite(low):
        value = low + 1.0
    else:
        value = 1.0
    return value if value != cfg.fill_value else value + 1.0


# The masking tier each registered dataset gets through the opener. A new
# registry entry fails here until its tier is stated.
EXPECTED_TIERS = {
    "OMI_NO2": QA_NO_FLAG_VARIABLE,
    "TROPOMI_NO2": QA_NO_FLAG_VARIABLE,
    "TEMPO_NO2": QA_VERIFIED,
    "TEMPO_NO2_NRT": QA_VERIFIED,
    "TEMPO_O3TOT": QA_NO_FLAG_VARIABLE,
    "TEMPO_O3PROF": QA_NO_FLAG_VARIABLE,
    "OMI_O3": QA_NO_FLAG_VARIABLE,
    "TEMPO_HCHO": QA_VERIFIED,
    "TEMPO_HCHO_V03": QA_VERIFIED,
    "TEMPO_HCHO_NRT": QA_VERIFIED,
    "OMI_HCHO": QA_VERIFIED,
    "OMI_SO2": QA_VERIFIED,
    "MODIS_AOD_TERRA": QA_NO_FLAG_VARIABLE,
    "VIIRS_AOD_SNPP": QA_NO_FLAG_VARIABLE,
    "VIIRS_AOD_NOAA20": QA_NO_FLAG_VARIABLE,
    "AERDA_AOD_NRT": QA_NO_FLAG_VARIABLE,
    "AERDA_AOD_NRT_3H": QA_NO_FLAG_VARIABLE,
    "MODIS_AOD_AQUA": QA_NO_FLAG_VARIABLE,
}


class RegistryTierTests(OpenerCase):
    def test_every_registered_dataset_has_a_stated_tier(self):
        self.assertEqual(set(EXPECTED_TIERS), set(load_registry()))
        for key, tier in EXPECTED_TIERS.items():
            with self.subTest(key):
                self.assertEqual(tier == QA_NO_FLAG_VARIABLE, key in COLLECTIONS_WITHOUT_QUALITY_FLAG)

    async def test_each_registered_dataset_gets_its_tier_through_the_opener(self):
        for key, cfg in load_registry().items():
            with self.subTest(key):
                science = cfg.primary_var.rsplit("/", 1)[-1]
                flag = cfg.quality_flag_var.rsplit("/", 1)[-1] if cfg.quality_flag_var else None
                failing = 1 if cfg.qa_good_values else (cfg.qa_bad_values or [1])[0]
                flags = np.zeros((1, LATS.size, LONS.size), dtype="int16")
                flags[0, 2, 2] = failing
                ds = granule(
                    short_name=cfg.short_name, science=science, flag=flag, flags=flags,
                    value=_value(cfg), units=cfg.units,
                )
                record = await self.retrieval(f"obs_{key.lower()}", ds, product=key)
                grid = await open_grid(probe_call(self.env), record)
                self.assertEqual(grid.masking["qa_status"], EXPECTED_TIERS[key])
                masked_cell = bool(np.isnan(grid.data.sel(lat=LATS[2], lon=LONS[2]).values).all())
                self.assertEqual(masked_cell, EXPECTED_TIERS[key] == QA_VERIFIED)


if __name__ == "__main__":
    unittest.main()
