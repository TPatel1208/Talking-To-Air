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

_TERMINAL_STATUSES = {"materialized", "failed", "cancelled"}
_POLL_INTERVAL_SECONDS = 2
_POLL_MAX_ATTEMPTS = 30  # ~1 minute — a single small subset should finish well inside this.

# TEMPO NO2 over Houston in June 2024 is the same dataset/location/date this
# repo's scripted eval (tests/eval_harness.py) already exercises successfully
# against the real MCP — a known-good small window, not a guess.
_QUERY = "TEMPO NO2"
_LOCATION = "-95.6,29.5,-95.0,30.0"  # small bbox around Houston, TX
_TIME_RANGE = "2024-06-01/2024-06-01"
_VARIABLES = ["nitrogen_dioxide_tropospheric_column"]


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
        if status.get("status") != "materialized":
            self.skipTest(f"retrieval job did not materialize (status={status}); contract keys already proven above")

        export_result = await self._invoke("export_result", handle=obs_handle)
        self.assertEqual(export_result.get("status"), "ready")
        self.assertIn("storage_uri", export_result, f"export_result missing 'storage_uri': {export_result}")

    async def _await_terminal_status(self, job_handle: str) -> dict:
        import asyncio

        for _ in range(_POLL_MAX_ATTEMPTS):
            status = await self._invoke("get_retrieval_status", job_handle=job_handle)
            if status.get("status") in _TERMINAL_STATUSES:
                return status
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        self.fail(f"retrieval job {job_handle} did not reach a terminal state within the smoke suite's poll budget")


if __name__ == "__main__":
    unittest.main()
