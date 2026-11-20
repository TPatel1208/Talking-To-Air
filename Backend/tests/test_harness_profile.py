"""operations.vertical_profile and render.profile."""
import datetime
import os
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.operations import profile as operation
from tta_backend.harness.capabilities.render import profiles
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, HandleRecord, ServicePayload
from tta_backend.harness.handles.model import Period
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import (
    FakePlaces,
    MemoryArtifacts,
    RecordingEvents,
    box_region,
    make_run_context,
    make_run_env,
)
from tta_backend.harness.trace import FailureCategory, StepFailure

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from test_vertical_profile import ALTITUDES_TOP_DOWN, PRESSURES_TOP_DOWN, _o3prof_dataset  # noqa: E402

NOW = datetime.datetime(2025, 10, 2, tzinfo=datetime.UTC)
REGION = {"name": "Box", "bounds": [25.0, 5.0, 45.0, 25.0]}
GLOBAL = box_region("Global", -180.0, -90.0, 180.0, 90.0, region_type="bounding_box")


def layered_grid(layer_values: list[float], **kwargs: Any) -> xr.Dataset:
    """The profile fixture in the harness grid layout."""
    ds = _o3prof_dataset(xr, np, layer_values=layer_values, **kwargs)
    return ds.rename({"latitude": "lat", "longitude": "lon"})


class ProfileCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        self.events = RecordingEvents()
        self.artifacts = MemoryArtifacts()
        self.files: dict[str, xr.Dataset] = {}

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.files[record.id]

        context = make_run_context(stores=self.stores, events=self.events, artifacts=self.artifacts)
        self.env: RunEnv = await make_run_env(
            self._tmp.name, context=context, open_retrieval=open_retrieval, places=FakePlaces({"global": GLOBAL}),
        )

    async def seed_grid(self, ds: xr.Dataset) -> HandleRecord:
        return await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="grid", data=ds,
            descriptor=Descriptor(
                summary="Ozone profile", quantity="ozone partial column", unit="DU", product="TEMPO_O3PROF",
                region=REGION, period=Period(start="2025-10-01", end="2025-10-01"),
                masking={"variable": "ozone_profile", "qa_status": "verified", "region_cells": 4},
            ),
            operation="test.seed", created_at=NOW,
        )

    async def failure(self, capability: Any, params: dict[str, Any]) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, capability, params)
        return caught.exception


class VerticalProfileOperationTests(ProfileCase):
    async def test_a_layered_grid_becomes_a_table_of_layers_against_pressure_and_altitude(self):
        grid = await self.seed_grid(layered_grid([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = await invoke(self.env, operation.vertical_profile, {"grid": grid.id})

        self.assertEqual((result.status, result.kind), ("done", "table"))
        self.assertEqual(result.unit, "DU")
        frame = await self.env.resolver.open("user-a", result.handle, "table")
        self.assertEqual(list(frame["layer"]), [0, 1, 2, 3, 4, 5])
        self.assertEqual(list(frame["value"]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(list(frame["pressure"]), PRESSURES_TOP_DOWN)
        self.assertEqual(list(frame["altitude"]), ALTITUDES_TOP_DOWN)
        self.assertEqual(set(frame["pressure_unit"]), {"hPa"})
        self.assertEqual(list(frame["valid_fraction"]), [1.0] * 6)
        self.assertEqual(result.summary["default_axis"], "pressure")
        self.assertEqual(result.summary["layer_order"], "top_down")
        self.assertEqual(result.summary["vertical_dim"], "layer")

        record = await self.env.resolver.get("user-a", result.handle)
        self.assertEqual(record.parents, (grid.id,))
        self.assertEqual(record.operation, "operations.vertical_profile")
        self.assertEqual(record.descriptor.masking["qa_status"], "verified")
        [entry] = await self.stores.traces.entries("user-a", self.env.run_id)
        self.assertEqual((entry.capability, entry.status), ("operations.vertical_profile", "done"))
        self.assertIn("6 layers", entry.result["method"])

    async def test_a_retrieval_grid_keeps_axes_published_as_variables(self):
        area = (await invoke(self.env, areas.resolve, {"place": "global"})).handle
        area_record = await self.env.resolver.get("user-a", area)
        for axes_as_variables in (False, True):
            with self.subTest(axes_as_variables=axes_as_variables):
                ds = _o3prof_dataset(xr, np, layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                if axes_as_variables:
                    ds = ds.reset_coords(["ozone_profile_pressure", "ozone_profile_altitude"])
                handle = f"obs_prof_{int(axes_as_variables)}"
                self.files[handle] = ds
                await self.env.context.storage.handles.record_retrieval(HandleRecord(
                    id=handle, kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
                    descriptor=Descriptor(
                        summary="Ozone profile", product="TEMPO_O3PROF", unit="DU",
                        region={**area_record.descriptor.region, "area": area},  # type: ignore[dict-item]
                        masking={"variable": "ozone_profile"},
                    ),
                    created_at=NOW, payload=ServicePayload(),
                ))
                result = await invoke(self.env, operation.vertical_profile, {"grid": handle})
                frame = await self.env.resolver.open("user-a", result.handle, "table")
                self.assertEqual(list(frame["pressure"]), PRESSURES_TOP_DOWN)
                self.assertEqual(list(frame["altitude"]), ALTITUDES_TOP_DOWN)
                self.assertEqual(result.summary["default_axis"], "pressure")

    async def test_several_timesteps_are_averaged(self):
        grid = await self.seed_grid(layered_grid([2.0] * 6, times=("2025-10-01T12:00:00", "2025-10-01T13:00:00")))
        result = await invoke(self.env, operation.vertical_profile, {"grid": grid.id})
        self.assertEqual(result.summary["timesteps"], 2)
        self.assertTrue(any("mean over time" in d for d in result.disclosures))

    async def test_a_grid_without_a_vertical_dimension_is_refused(self):
        grid = await self.seed_grid(layered_grid([1.0] * 6).isel(layer=0, drop=True).drop_vars(
            ["ozone_profile_pressure", "ozone_profile_altitude"], errors="ignore",
        ))
        failure = await self.failure(operation.vertical_profile, {"grid": grid.id})
        self.assertEqual(failure.category, FailureCategory.UNSUPPORTED_SHAPE)
        self.assertEqual(failure.result, {"refusal": "no_vertical_dim"})

    async def test_two_extra_dimensions_are_refused_as_ambiguous(self):
        ds = layered_grid([1.0] * 6).expand_dims(member=[0, 1])
        grid = await self.seed_grid(ds)
        failure = await self.failure(operation.vertical_profile, {"grid": grid.id})
        self.assertEqual(failure.result, {"refusal": "ambiguous_vertical"})

    async def test_no_valid_layer_is_insufficient_evidence(self):
        grid = await self.seed_grid(layered_grid([np.nan] * 6))
        result = await invoke(self.env, operation.vertical_profile, {"grid": grid.id})
        self.assertEqual(result.status, "insufficient_evidence")

    async def test_an_oversized_grid_is_too_large(self):
        grid = await self.seed_grid(layered_grid([1.0] * 6))
        with patch("tta_backend.domain.charts.profile.MAX_PROFILE_CELLS", 3):
            failure = await self.failure(operation.vertical_profile, {"grid": grid.id})
        self.assertEqual(failure.category, FailureCategory.REQUEST_TOO_LARGE)

    async def test_only_a_grid_is_accepted(self):
        table = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="table", data=pd.DataFrame({"a": [1]}),
            descriptor=Descriptor(summary="t"), operation="test.seed", created_at=NOW,
        )
        failure = await self.failure(operation.vertical_profile, {"grid": table.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)

    def test_it_registers_under_the_operations_group_by_name(self):
        self.assertEqual([c.group for c in operation.CAPABILITIES], ["operations"])
        self.assertFalse(hasattr(operation, "GROUP"))


class RenderProfileTests(ProfileCase):
    async def profile_table(self, values: list[float]) -> str:
        grid = await self.seed_grid(layered_grid(values))
        result = await invoke(self.env, operation.vertical_profile, {"grid": grid.id})
        assert result.handle is not None
        return result.handle

    async def test_the_table_is_drawn_as_the_profile_chart(self):
        table = await self.profile_table([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = await invoke(self.env, profiles.render_profile, {"profile": table})

        [chart_id] = result.charts
        payload = self.artifacts.stored[chart_id]
        [emitted] = self.events.charts
        self.assertEqual(emitted["chart_id"], chart_id)
        self.assertEqual(payload["type"], "profile")
        self.assertEqual(payload["values"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(payload["layers"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(payload["units"], "DU")
        self.assertEqual(payload["default_axis"], "pressure")
        self.assertEqual(payload["layer_order"], "top_down")
        self.assertEqual(payload["vertical"]["pressure"]["values"], PRESSURES_TOP_DOWN)
        self.assertEqual(payload["vertical"]["altitude"]["units"], "km")
        self.assertEqual(payload["vertical"]["altitude"]["layer_order"], "top_down")
        self.assertEqual(payload["valid_fraction"], [1.0] * 6)
        self.assertEqual(payload["export"]["values"], payload["values"])
        self.assertEqual(payload["metadata"]["source_handles"], [table])
        self.assertEqual(payload["provenance"]["source_handles"], [table])
        self.assertEqual(result.summary["render_type"], "profile")
        self.assertEqual(result.method, "A vertical profile of 6 layers was drawn.")

    async def test_a_table_that_is_not_a_profile_is_refused(self):
        other = await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="table", data=pd.DataFrame({"date": ["2024-06-01"]}),
            descriptor=Descriptor(summary="t"), operation="test.seed", created_at=NOW,
        )
        failure = await self.failure(profiles.render_profile, {"profile": other.id})
        self.assertEqual(failure.result, {"refusal": "not_a_profile"})
        self.assertEqual(self.events.charts, [])

    async def test_a_grid_is_not_a_profile_table(self):
        grid = await self.seed_grid(layered_grid([1.0] * 6))
        failure = await self.failure(profiles.render_profile, {"profile": grid.id})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)


if __name__ == "__main__":
    unittest.main()
