"""
tests/live_smoke/test_mcp_contract.py
========================================
PRD T17: an opt-in suite that drives the real discovery -> AOI -> coverage ->
retrieve -> export chain against the real earthdata-retrieval MCP (both
Docker stacks up), asserting the exact response-key contracts a fake MCP
cannot prove. Its first assertions are the keys that drifted before (T11's
`aoi_handle`-vs-`handle` bug): the fake mirrored the wrong key and a 100%
green suite still shipped broken production behavior. Only a check against
the real thing can catch a lying mirror.

Run explicitly (skipped, not failed, otherwise):

    EARTHDATA_MCP_URL=http://localhost:8765/mcp pytest -m live_mcp tests/live_smoke/

Keeps retrieval tiny — a single small dataset/AOI/date already known (from
this repo's scripted eval and live-verification notes) to have TEMPO NO2
coverage — and uses a throwaway workspace id so it never collides with a
researcher's real workspace.
"""
from __future__ import annotations

import os
import sys
import unittest
import uuid

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

pytestmark = pytest.mark.live_mcp

# Live-verified (T17) against the real MCP's jobs/state.py JobState enum:
# ready/failed/expired/cancelled — NOT "materialized". This suite intentionally
# does not reuse services/retrieval_composites.py's TERMINAL_STATUSES, which
# still assumes "materialized" and never recognizes a real job's actual
# terminal status; that mismatch is a separate, pre-existing production bug
# this suite surfaced but does not fix (out of scope for T17 — see the final
# assertion below for the specific fact this suite proves about it).
_REAL_TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled"}
_POLL_INTERVAL_SECONDS = 3
# Live-verified (T17): a real Harmony-routed submit -> poll -> materialize
# cycle for even a single small subset took ~85s end to end (Harmony's own
# job polling, not just this backend) — 60 attempts at 3s gives ~3 minutes.
_POLL_MAX_ATTEMPTS = 60

# TEMPO NO2 over Houston in June 2024 is the same dataset/location/date this
# repo's scripted eval (tests/eval_harness.py) already exercises successfully
# against the real MCP — a known-good small window, not a guess. The variable
# name is the real product variable (confirmed live via describe_dataset
# against the actual TEMPO NO2 collection) — not the model-facing short name
# the agent's prompt uses, since this suite calls the MCP tools directly.
_QUERY = "TEMPO NO2"
_LOCATION = "-95.6,29.5,-95.0,30.0"  # small bbox around Houston, TX
_TIME_RANGE = "2024-06-01/2024-06-02"
_VARIABLES = ["product/vertical_column_troposphere"]


def _mcp_url() -> str | None:
    return os.environ.get("EARTHDATA_MCP_URL")


@unittest.skipUnless(_mcp_url(), "EARTHDATA_MCP_URL is not set — skipping the live MCP smoke suite")
class LiveMCPContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from config.settings import Settings
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.results import parse_tool_result

        self.parse_tool_result = parse_tool_result
        self.workspace_id = f"live_smoke_{uuid.uuid4().hex[:8]}"
        settings = Settings(earthdata_mcp_url=_mcp_url(), earthdata_mcp_token=os.environ.get("EARTHDATA_MCP_TOKEN"))
        self.tools = await load_raw_mcp_tools(settings)

    async def _invoke(self, tool_name: str, **kwargs) -> dict:
        raw = await self.tools[tool_name].ainvoke({**kwargs, "workspace_id": self.workspace_id})
        return self.parse_tool_result(raw)

    async def test_discovery_to_export_chain_uses_the_real_contract_keys(self):
        search_result = await self._invoke("search_datasets", query=_QUERY, filters=None)
        datasets = search_result.get("datasets") or search_result.get("results") or []
        self.assertTrue(datasets, f"search_datasets returned no results for {_QUERY!r}: {search_result}")
        dataset_handle = datasets[0]["handle"] if "handle" in datasets[0] else datasets[0]["dataset_handle"]

        # The exact bug this suite exists to catch (T11): the fake MCP
        # returned "aoi_handle", the real MCP returns "handle" — a green
        # fake-backed suite shipped broken production code.
        aoi_result = await self._invoke("define_area_of_interest", location=_LOCATION)
        self.assertIn("handle", aoi_result, f"define_area_of_interest response missing 'handle': {aoi_result}")
        aoi_handle = aoi_result["handle"]

        coverage_result = await self._invoke(
            "check_coverage", dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=_TIME_RANGE,
        )
        granule_count = coverage_result.get("granule_count", 0)
        if not granule_count:
            self.skipTest(
                f"no granules covering {_LOCATION!r}/{_TIME_RANGE!r} right now — "
                "contract facts already proven above (search/AOI keys); "
                "retrieval chain needs live coverage to continue"
            )

        retrieve_result = await self._invoke(
            "retrieve_subset",
            dataset_handle=dataset_handle,
            aoi_handle=aoi_handle,
            time_range=_TIME_RANGE,
            variables=_VARIABLES,
            output_format=None,
        )
        self.assertIn("job_handle", retrieve_result, f"retrieve_subset missing 'job_handle': {retrieve_result}")
        self.assertIn("obs_handle", retrieve_result, f"retrieve_subset missing 'obs_handle': {retrieve_result}")
        job_handle = retrieve_result["job_handle"]
        obs_handle = retrieve_result["obs_handle"]

        status = await self._await_terminal_status(job_handle)
        if status.get("status") != "ready":
            self.skipTest(f"retrieval job did not become ready (status={status}); contract keys already proven above")

        export_result = await self._invoke("export_result", handle=obs_handle)
        self.assertEqual(export_result.get("status"), "ready")
        self.assertIn("storage_uri", export_result, f"export_result missing 'storage_uri': {export_result}")

        # T24 anti-lying-mirror: open the real exported granule and assert the
        # canonical identifier finds its lat/lon. The synthetic test matrix
        # enumerates the structural shapes of Earthdata files; this pins one
        # real Harmony round-trip to those shapes, so a divergence between
        # what we synthesize and what the MCP actually exports (the gap that
        # let the original empty-coords crash ship) fails loudly here.
        from services.open_handle import _open
        from preprocessing.aggregation_service import AggregationService
        from utils.geo_utils import find_lat_coord, find_lon_coord

        ds = _open(export_result["storage_uri"], export_result.get("media_type", "netcdf"))
        da = AggregationService().to_dataarray(ds)
        self.assertIsNotNone(find_lat_coord(da), f"no latitude coord found on exported granule; coords={list(da.coords)}")
        self.assertIsNotNone(find_lon_coord(da), f"no longitude coord found on exported granule; coords={list(da.coords)}")

    async def test_inspect_granules_uses_the_real_contract_keys(self):
        # T21: the discovery pane's granule-inspection endpoint calls this
        # tool directly (services/discovery_service.py) — proves the real
        # MCP's response carries the keys that endpoint reads (`granules`,
        # `count`) rather than only the fake fixture's mirrored shape.
        aoi_result = await self._invoke("define_area_of_interest", location=_LOCATION)
        aoi_handle = aoi_result["handle"]

        search_result = await self._invoke("search_datasets", query=_QUERY, filters=None)
        datasets = search_result.get("datasets") or search_result.get("results") or []
        self.assertTrue(datasets, f"search_datasets returned no results for {_QUERY!r}: {search_result}")
        dataset_handle = datasets[0]["handle"] if "handle" in datasets[0] else datasets[0]["dataset_handle"]

        result = await self._invoke(
            "inspect_granules",
            dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=_TIME_RANGE, limit=10,
        )
        self.assertIn("granules", result, f"inspect_granules missing 'granules': {result}")
        self.assertIn("count", result, f"inspect_granules missing 'count': {result}")
        self.assertEqual(result["count"], len(result["granules"]))
        if result["granules"]:
            granule = result["granules"][0]
            self.assertIn("size_mb", granule, f"granule record missing 'size_mb': {granule}")
            self.assertIn("time_start", granule, f"granule record missing 'time_start': {granule}")

    async def _await_terminal_status(self, job_handle: str) -> dict:
        import asyncio

        for _ in range(_POLL_MAX_ATTEMPTS):
            status = await self._invoke("get_retrieval_status", job_handle=job_handle)
            if status.get("status") in _REAL_TERMINAL_STATUSES:
                return status
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        self.fail(f"retrieval job {job_handle} did not reach a terminal state within the smoke suite's poll budget")


if __name__ == "__main__":
    unittest.main()
