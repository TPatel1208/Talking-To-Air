"""
tests/test_provenance_service.py
==================================
T10: the provenance panel's backend seam. Exercises provenance_service
against the fake earthdata-retrieval MCP (real wire protocol, fake tool
handlers) exactly like the T05/T09 service tests.
"""
import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class GetLineageTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_a_leaf_handle_with_no_ancestry_renders_as_a_single_leaf_node(self):
        from services.provenance_service import get_lineage

        async def get_provenance(handle, workspace_id):
            return {"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"}

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["dataset_tempo_no2"], tools)

        self.assertEqual(len(lineage["nodes"]), 1)
        self.assertEqual(lineage["nodes"][0]["handle"], "dataset_tempo_no2")
        self.assertEqual(lineage["nodes"][0]["kind"], "dataset")
        self.assertEqual(lineage["nodes"][0]["description"], "TEMPO NO2 L3")

    async def test_an_observation_handles_nested_inputs_are_flattened_ancestors_first(self):
        from services.provenance_service import get_lineage

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "kind": "observation",
                "events": [
                    {"stage": "routed", "at": "2026-07-01T00:00:00Z", "provider": "GES_DISC"},
                    {"stage": "materialized", "at": "2026-07-01T00:12:00Z", "granule_count": 24},
                ],
                "inputs": [
                    {"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"},
                    {"handle": "aoi_nj", "kind": "aoi", "description": "New Jersey"},
                ],
            }

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["obs_1"], tools)

        handles_in_order = [node["handle"] for node in lineage["nodes"]]
        self.assertEqual(set(handles_in_order), {"obs_1", "dataset_tempo_no2", "aoi_nj"})
        # Ancestors (leaf inputs, no events) render before the descendant that consumed them.
        self.assertLess(handles_in_order.index("dataset_tempo_no2"), handles_in_order.index("obs_1"))
        self.assertLess(handles_in_order.index("aoi_nj"), handles_in_order.index("obs_1"))

        obs_node = next(node for node in lineage["nodes"] if node["handle"] == "obs_1")
        self.assertEqual([event["stage"] for event in obs_node["events"]], ["routed", "materialized"])

    async def test_a_shared_ancestor_across_two_source_handles_is_deduplicated(self):
        # A T08 comparison artifact's two panels share the same AOI and the
        # same "align" intermediate — the merged lineage should list each
        # shared ancestor once, not once per panel that references it.
        from services.provenance_service import get_lineage

        provenance = {
            "obs_east": {
                "handle": "obs_east",
                "kind": "observation",
                "events": [{"stage": "materialized", "at": "2026-07-01T00:12:00Z"}],
                "inputs": [
                    {"handle": "aligned_1", "kind": "aligned", "events": [{"stage": "aligned", "at": "2026-07-01T00:10:00Z"}]},
                    {"handle": "aoi_nj", "kind": "aoi", "description": "New Jersey"},
                ],
            },
            "obs_west": {
                "handle": "obs_west",
                "kind": "observation",
                "events": [{"stage": "materialized", "at": "2026-07-01T00:13:00Z"}],
                "inputs": [
                    {"handle": "aligned_1", "kind": "aligned", "events": [{"stage": "aligned", "at": "2026-07-01T00:10:00Z"}]},
                    {"handle": "aoi_nj", "kind": "aoi", "description": "New Jersey"},
                ],
            },
        }

        async def get_provenance(handle, workspace_id):
            return provenance[handle]

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["obs_east", "obs_west"], tools)

        handles = [node["handle"] for node in lineage["nodes"]]
        self.assertEqual(len(handles), len(set(handles)), "shared ancestors must appear exactly once")
        self.assertEqual(set(handles), {"obs_east", "obs_west", "aligned_1", "aoi_nj"})
        self.assertLess(handles.index("aligned_1"), handles.index("obs_east"))
        self.assertLess(handles.index("aligned_1"), handles.index("obs_west"))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class GetCitationsTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_cites_the_distinct_dataset_behind_a_single_source_handle(self):
        from services.provenance_service import get_citations

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "kind": "observation",
                "events": [{"stage": "materialized", "at": "2026-07-01T00:12:00Z"}],
                "inputs": [{"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"}],
            }

        cite_calls = []

        async def cite_dataset(dataset_handle, workspace_id):
            cite_calls.append(dataset_handle)
            return {
                "dataset_handle": "dataset_tempo_no2",
                "doi": "10.5067/TEMPO/NO2/L3",
                "citation": "NASA, TEMPO NO2 Tropospheric Column L3, doi:10.5067/TEMPO/NO2/L3",
            }

        tools = await self._tools({"get_provenance": get_provenance, "cite_dataset": cite_dataset})

        citations = await get_citations(["obs_1"], tools)

        self.assertEqual(cite_calls, ["dataset_tempo_no2"])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["doi"], "10.5067/TEMPO/NO2/L3")

    async def test_a_dataset_shared_by_two_source_handles_is_cited_only_once(self):
        from services.provenance_service import get_citations

        provenance = {
            "obs_east": {
                "handle": "obs_east",
                "kind": "observation",
                "inputs": [{"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"}],
            },
            "obs_west": {
                "handle": "obs_west",
                "kind": "observation",
                "inputs": [{"handle": "dataset_tempo_no2", "kind": "dataset", "description": "TEMPO NO2 L3"}],
            },
        }

        async def get_provenance(handle, workspace_id):
            return provenance[handle]

        cite_calls = []

        async def cite_dataset(dataset_handle, workspace_id):
            cite_calls.append(dataset_handle)
            return {"dataset_handle": dataset_handle, "doi": "10.5067/TEMPO/NO2/L3", "citation": "NASA TEMPO NO2 L3"}

        tools = await self._tools({"get_provenance": get_provenance, "cite_dataset": cite_dataset})

        citations = await get_citations(["obs_east", "obs_west"], tools)

        self.assertEqual(cite_calls, ["dataset_tempo_no2"])
        self.assertEqual(len(citations), 1)


if __name__ == "__main__":
    unittest.main()
