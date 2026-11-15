"""The app's host ports for the harness, against the fake earthdata MCP server."""
import datetime
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

from tests.test_harness_grids import granule
from tests.fake_earthdata_mcp import FakeEarthdataMCPServer, HandleVolume, build_fake_mcp
from tta_backend.datasets.qa_flags import QA_VERIFIED
from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
from tta_backend.earthdata_mcp.results import MCPToolError
from tta_backend.harness.capabilities import invoke
from tta_backend.harness.capabilities.sources import areas, satellite
from tta_backend.harness.handles import Descriptor, HandleError, HandleRecord, ServicePayload
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.run_context import IdempotencyKeyReused
from tta_backend.harness.testing import FixedClock, StaticCredentials, make_run_context, make_run_env
from tta_backend.infrastructure.settings import Settings, get_settings
from tta_backend.services.harness_host import (
    AdmissionCompute,
    McpEarthdataSession,
    ResolverPlaces,
    limits_from_settings,
    open_retrieval_over,
)
from tta_backend.services.open_handle import HandleGoneError, OpenHandleError
from tta_backend.utils.plotting import RegionResolver

NOW = datetime.datetime(2024, 7, 1, 12, tzinfo=datetime.UTC)


class NoGeocoder:
    """Answers nothing, so a test never reaches a live geocoder."""

    def geocode(self, location_name: str) -> None:
        return None

    async def ageocode(self, location_name: str) -> None:
        return None


def retrieval_record(handle: str, user: str = "user-a") -> HandleRecord:
    return HandleRecord(
        id=handle, kind="grid", origin="retrieval", user_id=user, conversation_id="conv-1",
        descriptor=Descriptor(summary="A retrieval"), created_at=NOW, payload=ServicePayload(),
    )


class HostAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"CUBE_STORE_DIR": os.path.join(self._tmp.name, "cubes")})
        patcher.start()
        self.addCleanup(patcher.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        self.volume = HandleVolume(os.path.join(self._tmp.name, "volume"))
        os.makedirs(self.volume.root, exist_ok=True)
        self.volume.add_zarr("obs_tempo", granule)
        self.submissions = 0

        async def search_datasets(**kwargs: Any) -> dict:
            if kwargs["query"] == "broken":
                return {"error": {"category": "user_input", "message": "The query is not valid."}}
            return {"datasets": [{
                "handle": "dataset_1", "concept_id": load_registry()["TEMPO_NO2"].collection_id,
                "short_name": "TEMPO_NO2_L3", "title": "TEMPO NO2",
            }]}

        async def define_area_of_interest(**kwargs: Any) -> dict:
            return {"handle": "aoi_1", "location": kwargs["location"]}

        async def check_coverage(**kwargs: Any) -> dict:
            return {"has_data": True, "granule_count": 2}

        async def estimate_retrieval_size(**kwargs: Any) -> dict:
            return {"estimated_bytes": 1000}

        async def retrieve_subset(**kwargs: Any) -> dict:
            self.submissions += 1
            return {"job_handle": "job_obs_tempo", "obs_handle": "obs_tempo"}

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "search_datasets": search_datasets,
            "define_area_of_interest": define_area_of_interest,
            "check_coverage": check_coverage,
            "estimate_retrieval_size": estimate_retrieval_size,
            "retrieve_subset": retrieve_subset,
            "get_retrieval_status": self.volume.get_retrieval_status,
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
        }))
        server.start()
        self.addCleanup(server.stop)
        self.tools = await load_raw_mcp_tools(Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None))

    async def test_the_session_parses_results_and_classifies_errors(self):
        session = McpEarthdataSession(self.tools)
        found = await session.call("search_datasets", {"query": "no2", "filters": None})
        self.assertEqual(found["datasets"][0]["handle"], "dataset_1")
        with self.assertRaises(MCPToolError) as caught:
            await session.call("search_datasets", {"query": "broken", "filters": None})
        self.assertEqual(caught.exception.category, "user_input")
        with self.assertRaises(MCPToolError) as unknown:
            await session.call("no_such_tool", {})
        self.assertEqual(unknown.exception.category, "contract")

    async def test_the_session_answers_a_repeated_idempotency_key_without_calling_again(self):
        session = McpEarthdataSession(self.tools)
        args = {"dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "x", "variables": [], "output_format": None}
        first = await session.call("retrieve_subset", args, idempotency_key="tta-1-0")
        again = await session.call("retrieve_subset", args, idempotency_key="tta-1-0")
        other = await session.call("retrieve_subset", args, idempotency_key="tta-1-1")
        self.assertEqual(first, again)
        self.assertEqual(other["job_handle"], "job_obs_tempo")
        self.assertEqual(self.submissions, 2)

    async def test_the_session_refuses_a_key_reused_for_another_request(self):
        session = McpEarthdataSession(self.tools, remembered=1)
        args = {"dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "x", "variables": [], "output_format": None}
        await session.call("retrieve_subset", args, idempotency_key="tta-1")
        with self.assertRaises(IdempotencyKeyReused) as caught:
            await session.call("retrieve_subset", {**args, "time_range": "y"}, idempotency_key="tta-1")
        self.assertEqual(caught.exception.category, "contract")
        self.assertEqual(self.submissions, 1)
        # Only the most recent key is remembered.
        await session.call("retrieve_subset", args, idempotency_key="tta-2")
        await session.call("retrieve_subset", args, idempotency_key="tta-1")
        self.assertEqual(self.submissions, 3)

    async def test_a_retrieval_gone_beyond_rematerialising_is_evicted(self):
        open_retrieval = open_retrieval_over(self.tools, "user-a")
        with self.assertRaises(HandleError) as caught:
            await open_retrieval("user-a", retrieval_record("obs_gone"))
        self.assertEqual((caught.exception.reason, caught.exception.handle_id), ("evicted", "obs_gone"))

    async def test_other_open_failures_keep_their_own_error(self):
        open_retrieval = open_retrieval_over(self.tools, "user-a")
        with mock.patch("tta_backend.services.harness_host.open_handle", side_effect=OpenHandleError("bad media")):
            with self.assertRaises(OpenHandleError) as caught:
                await open_retrieval("user-a", retrieval_record("obs_tempo"))
        self.assertNotIsInstance(caught.exception, HandleGoneError)

    async def test_retrievals_open_through_open_handle_for_their_owner_only(self):
        open_retrieval = open_retrieval_over(self.tools, "user-a")
        ds = await open_retrieval("user-a", retrieval_record("obs_tempo"))
        self.assertIn("vertical_column_troposphere", ds.data_vars)
        with self.assertRaises(HandleError) as caught:
            await open_retrieval("user-b", retrieval_record("obs_tempo", user="user-b"))
        self.assertEqual(caught.exception.reason, "not_found")

    async def test_compute_runs_on_the_admitted_pool(self):
        result = await AdmissionCompute().run(lambda a, *, b: a + b, 2, b=3)
        self.assertEqual(result, 5)

    def test_limits_come_from_settings(self):
        settings = Settings()
        limits = limits_from_settings(settings)
        self.assertEqual(limits.retrieval_soft_bytes, settings.retrieval_soft_cap_bytes)
        self.assertEqual(limits.retrieval_hard_bytes, settings.retrieval_hard_cap_bytes)
        self.assertEqual(limits.max_point_timeseries_days, settings.retrieval_max_timeseries_days)
        self.assertEqual(limits.retrieval_timeout_seconds, settings.await_retrieval_timeout_seconds)
        self.assertEqual((limits.poll_min_seconds, limits.poll_max_seconds),
                         (settings.await_retrieval_poll_min_seconds, settings.await_retrieval_poll_max_seconds))

    async def test_places_resolve_through_the_region_resolver(self):
        places = ResolverPlaces(RegionResolver(geocoding_service=NoGeocoder()))  # type: ignore[arg-type]
        region = await places.resolve("New Jersey")
        assert region is not None
        self.assertIn("New Jersey", region.name)
        west, south, east, north = region.bounds
        self.assertTrue(-76 < west < east < -73 and 38 < south < north < 42)
        self.assertIn(region.geometry["type"], ("Polygon", "MultiPolygon"))
        self.assertIsNone(await places.resolve("Nowhere-in-particular"))

    async def test_get_data_runs_end_to_end_over_the_host_ports(self):
        places = ResolverPlaces(RegionResolver(geocoding_service=NoGeocoder()))  # type: ignore[arg-type]
        context = make_run_context(credentials=StaticCredentials({"earthdata": "token"}), clock=FixedClock(NOW))
        env = await make_run_env(
            os.path.join(self._tmp.name, "payloads"), context=context,
            earthdata=McpEarthdataSession(self.tools), open_retrieval=open_retrieval_over(self.tools, "user-a"),
            places=places,
        )
        area = (await invoke(env, areas.resolve, {"place": "New Jersey"})).handle
        result = await invoke(env, satellite.get_data, {
            "quantity": "no2_tropospheric_column", "area": area, "dates": "2024-06-01", "date_role": "subject",
        }, step="event")
        self.assertEqual((result.status, result.handle, result.product), ("done", "obs_tempo", "TEMPO_NO2"))
        assert result.quality is not None
        self.assertEqual(result.quality["qa_status"], QA_VERIFIED)
        self.assertGreater(result.quality["valid_fraction"], 0.5)
        self.assertEqual(self.submissions, 1)


if __name__ == "__main__":
    unittest.main()
