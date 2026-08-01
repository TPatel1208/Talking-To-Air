"""
tests/test_provenance_service.py
==================================
T10: the provenance panel's backend seam. Exercises provenance_service
against the fake earthdata-retrieval MCP (real wire protocol, fake tool
handlers) exactly like the T05/T09 service tests.

Fixtures mirror the REAL MCP contract (harmony-retrieval-mcp
``tools/provenance.py``): ``get_provenance`` answers ``{handle, ancestors:
[{handle, depth}], events: [{event_type, detail, created_at}]}`` with events
newest-first, and ``cite_dataset`` answers CMR's own record. The previous
imagined ``inputs``/``kind``/``stage`` shape made citations silently empty
and methods.md crash against the live server (QA 2026-07-17).
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
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    async def test_a_leaf_handle_with_no_lineage_renders_as_a_single_noted_node(self):
        from tta_backend.services.provenance_service import get_lineage

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "dataset_tempo_no2",
                "ancestors": [],
                "events": [],
                "note": "no lineage found — ancestry attaches to obs_/cube_ handles",
            }

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["dataset_tempo_no2"], tools)

        self.assertEqual(len(lineage["nodes"]), 1)
        node = lineage["nodes"][0]
        self.assertEqual(node["handle"], "dataset_tempo_no2")
        self.assertEqual(node["kind"], "dataset", "kind derives from the handle prefix")
        self.assertIn("no lineage found", node["note"])

    async def test_an_observations_ancestors_render_before_it_with_events_oldest_first(self):
        from tta_backend.services.provenance_service import get_lineage

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "ancestors": [
                    {"handle": "dataset_tempo_no2", "depth": 1},
                    {"handle": "aoi_nj", "depth": 1},
                ],
                # The MCP answers newest-first.
                "events": [
                    {"event_type": "materialized", "detail": {"granule_count": 24}, "created_at": "2026-07-01T00:12:00Z"},
                    {"event_type": "routed", "detail": {"provider": "GES_DISC"}, "created_at": "2026-07-01T00:00:00Z"},
                ],
            }

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["obs_1"], tools)

        handles_in_order = [node["handle"] for node in lineage["nodes"]]
        self.assertEqual(set(handles_in_order), {"obs_1", "dataset_tempo_no2", "aoi_nj"})
        self.assertLess(handles_in_order.index("dataset_tempo_no2"), handles_in_order.index("obs_1"))
        self.assertLess(handles_in_order.index("aoi_nj"), handles_in_order.index("obs_1"))

        obs_node = next(node for node in lineage["nodes"] if node["handle"] == "obs_1")
        self.assertEqual(obs_node["kind"], "observation")
        self.assertEqual(
            [event["event_type"] for event in obs_node["events"]],
            ["routed", "materialized"],
            "events are re-ordered oldest-first for rendering",
        )

    async def test_a_job_handle_redirect_renders_under_its_resolved_obs_handle(self):
        from tta_backend.services.provenance_service import get_lineage

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "job_9",
                "resolved_handle": "obs_1",
                "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}],
                "events": [
                    {"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:12:00Z"},
                ],
            }

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["job_9"], tools)

        handles = {node["handle"] for node in lineage["nodes"]}
        self.assertEqual(handles, {"obs_1", "dataset_tempo_no2"})

    async def test_a_shared_ancestor_across_two_source_handles_is_deduplicated(self):
        # A T08 comparison artifact's two panels share the same AOI — the
        # merged lineage should list each shared ancestor once, not once per
        # panel that references it.
        from tta_backend.services.provenance_service import get_lineage

        provenance = {
            "obs_east": {
                "handle": "obs_east",
                "ancestors": [
                    {"handle": "cube_aligned_1", "depth": 1},
                    {"handle": "aoi_nj", "depth": 2},
                ],
                "events": [{"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:12:00Z"}],
            },
            "obs_west": {
                "handle": "obs_west",
                "ancestors": [
                    {"handle": "cube_aligned_1", "depth": 1},
                    {"handle": "aoi_nj", "depth": 2},
                ],
                "events": [{"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:13:00Z"}],
            },
        }

        async def get_provenance(handle, workspace_id):
            return provenance[handle]

        tools = await self._tools({"get_provenance": get_provenance})

        lineage = await get_lineage(["obs_east", "obs_west"], tools)

        handles = [node["handle"] for node in lineage["nodes"]]
        self.assertEqual(len(handles), len(set(handles)), "shared ancestors must appear exactly once")
        self.assertEqual(set(handles), {"obs_east", "obs_west", "cube_aligned_1", "aoi_nj"})
        self.assertLess(handles.index("cube_aligned_1"), handles.index("obs_east"))
        self.assertLess(handles.index("cube_aligned_1"), handles.index("obs_west"))
        # Deeper ancestors render before shallower ones.
        self.assertLess(handles.index("aoi_nj"), handles.index("cube_aligned_1"))


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class GetCitationsTests(unittest.IsolatedAsyncioTestCase):
    async def _tools(self, handlers):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        return await load_raw_mcp_tools(settings)

    @staticmethod
    def _citation_record(dataset_handle):
        return {
            "handle": dataset_handle,
            "concept_id": "C2930725014-LARC_CLOUD",
            "doi": "10.5067/TEMPO/NO2/L3",
            "doi_authority": "https://doi.org",
            "collection_citations": [{
                "Creator": "NASA/LARC/SD/ASDC",
                "Title": "TEMPO NO2 tropospheric and stratospheric columns V03",
                "Publisher": "NASA Atmospheric Science Data Center",
            }],
            "reference_citation_count": 12,
        }

    async def test_cites_the_distinct_dataset_ancestor_behind_a_single_source_handle(self):
        from tta_backend.services.provenance_service import get_citations

        async def get_provenance(handle, workspace_id):
            return {
                "handle": "obs_1",
                "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}],
                "events": [{"event_type": "materialized", "detail": {}, "created_at": "2026-07-01T00:12:00Z"}],
            }

        cite_calls = []

        async def cite_dataset(dataset_handle, workspace_id):
            cite_calls.append(dataset_handle)
            return self._citation_record(dataset_handle)

        tools = await self._tools({"get_provenance": get_provenance, "cite_dataset": cite_dataset})

        citations = await get_citations(["obs_1"], tools)

        self.assertEqual(cite_calls, ["dataset_tempo_no2"])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["doi"], "10.5067/TEMPO/NO2/L3")
        self.assertEqual(citations[0]["handle"], "dataset_tempo_no2")

    async def test_a_dataset_shared_by_two_source_handles_is_cited_only_once(self):
        from tta_backend.services.provenance_service import get_citations

        provenance = {
            "obs_east": {
                "handle": "obs_east",
                "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}],
                "events": [],
            },
            "obs_west": {
                "handle": "obs_west",
                "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}],
                "events": [],
            },
        }

        async def get_provenance(handle, workspace_id):
            return provenance[handle]

        cite_calls = []

        async def cite_dataset(dataset_handle, workspace_id):
            cite_calls.append(dataset_handle)
            return self._citation_record(dataset_handle)

        tools = await self._tools({"get_provenance": get_provenance, "cite_dataset": cite_dataset})

        citations = await get_citations(["obs_east", "obs_west"], tools)

        self.assertEqual(cite_calls, ["dataset_tempo_no2"])
        self.assertEqual(len(citations), 1)


if __name__ == "__main__":
    unittest.main()
