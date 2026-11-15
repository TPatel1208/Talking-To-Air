"""render.map against plot_singular: the same retrieval gives the same map."""
import datetime
import importlib.util
import json
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import xarray as xr

from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.render import maps
from tta_backend.harness.capabilities.sources import areas
from tta_backend.harness.handles import Descriptor, HandleRecord, ServicePayload
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import FakePlaces, MemoryArtifacts, RecordingEvents, box_region, make_run_context, make_run_env

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)
GLOBAL = box_region("Global", -180.0, -90.0, 180.0, 90.0, region_type="bounding_box")
REQUIRED_MODULES = ["langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn", "zarr", "rasterio", "cartopy"]

# Keys whose values the harness payload shares with the plot tool's.
SHARED = ("type", "variable", "units", "lats", "lons", "values", "statistics", "vmin", "vmax", "scale", "colormap", "is_aggregated")


def bundle(times: list[str]) -> xr.Dataset:
    rng = np.random.default_rng(11)
    shape = (len(times), 8, 8)
    science = rng.uniform(1e15, 9e15, shape)
    science[0, 2, 2] = np.nan
    quality = np.zeros(shape, dtype="int8")
    quality[:, 5, 5] = 1
    return xr.Dataset(
        {
            "vertical_column_troposphere": (("time", "lat", "lon"), science, {"units": "molecules/cm^2"}),
            "main_data_quality_flag": (("time", "lat", "lon"), quality),
        },
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": np.linspace(35.0, 42.0, 8),
            "lon": np.linspace(-80.0, -73.0, 8),
        },
        attrs={"short_name": "TEMPO_NO2_L3"},
    )


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "the plot tool's fake earthdata server dependencies are not installed",
)
class MapAgreesWithThePlotToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from fake_earthdata_mcp import FakeEarthdataMCPServer, HandleVolume, build_fake_mcp
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.infrastructure.settings import Settings

        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.volume = HandleVolume(self._tmp.name + "/volume")
        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        self.mcp_tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None))

        self.files: dict[str, xr.Dataset] = {}

        async def open_retrieval(user_id: str, record: HandleRecord) -> Any:
            return self.files[record.id]

        self.artifacts = MemoryArtifacts()
        context = make_run_context(stores=memory_stores(), events=RecordingEvents(), artifacts=self.artifacts)
        self.env = await make_run_env(
            self._tmp.name, context=context, open_retrieval=open_retrieval, places=FakePlaces({"global": GLOBAL}),
        )

    async def old_payload(self, ds: xr.Dataset) -> dict[str, Any]:
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        self.volume.add_zarr("obs_1", lambda: ds)
        emitted: dict[str, Any] = {}
        tool = make_plot_singular(self.mcp_tools)
        with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", lambda p: emitted.update(payload=p)):
            raw = await tool.ainvoke({"handle": "obs_1", "location": "global"})
        self.assertNotIn("error", json.loads(raw))
        return emitted["payload"]

    async def new_payload(self, ds: xr.Dataset) -> dict[str, Any]:
        area = (await invoke(self.env, areas.resolve, {"place": "global"})).handle
        area_record = await self.env.resolver.get("user-a", area)
        self.files["obs_1"] = ds
        record = await self.env.context.storage.handles.record_retrieval(HandleRecord(
            id="obs_1", kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
            descriptor=Descriptor(
                summary="NO2", product="TEMPO_NO2", unit="molecules/cm^2",
                region={**area_record.descriptor.region, "area": area},  # type: ignore[dict-item]
                masking={"variable": "vertical_column_troposphere"},
            ),
            created_at=NOW, payload=ServicePayload(),
        ))
        result = await invoke(self.env, maps.render_map, {"grid": record.id})
        return self.artifacts.stored[result.charts[0]]

    async def assert_same_map(self, times: list[str]) -> None:
        old = await self.old_payload(bundle(times))
        new = await self.new_payload(bundle(times))
        for key in SHARED:
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new["overlay"]["bounds"], old["overlay"]["bounds"])
        self.assertEqual(new["aggregation_meta"]["n_granules"], old["aggregation_meta"]["n_granules"])
        self.assertEqual(new["aggregation_meta"]["aggregation_label"], old["aggregation_meta"]["aggregation_label"])
        self.assertEqual(new["provenance"]["masking"]["qa_status"], old["provenance"]["masking"]["qa_status"])

    async def test_a_single_granule(self):
        await self.assert_same_map(["2024-06-01T15:00"])

    async def test_a_time_mean_over_several_granules(self):
        await self.assert_same_map(["2024-06-01T15:00", "2024-06-02T15:00", "2024-06-03T15:00"])


if __name__ == "__main__":
    unittest.main()
