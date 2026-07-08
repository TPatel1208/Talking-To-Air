"""
T19 Testing Decisions: a scripted workflow driven through the real
composites/handle tools (fake MCP, no live model — the eval harness is
where model tokens get spent) at the streaming seam, asserting the
*sequence* of stage keys observed in the SSE stream — presence and order,
never timing internals: search -> aoi -> coverage -> estimate -> submit ->
progress(>=1) -> open -> render -> text.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn", "xarray", "zarr", "pyarrow"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "workflow stage sequence test dependencies are not installed",
)
class WorkflowStageSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import xarray as xr
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from earthdata_mcp.client import load_raw_mcp_tools
        from earthdata_mcp.workspace import bind_workspace
        from config.settings import Settings

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        def make_dataset():
            return xr.Dataset(
                {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
                coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
            )

        self.volume.add_zarr("obs_1", make_dataset)

        async def search_datasets(query, filters, workspace_id):
            return {"dataset_handle": "dataset_1", "short_name": query, "title": query}

        async def define_area_of_interest(location, workspace_id):
            return {"aoi_handle": "aoi_1", "location": location}

        async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"granule_count": 14, "coverage_pct": 100}

        async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
            return {"estimated_bytes": 100}

        async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
            return {"job_handle": "job_obs_1", "obs_handle": "obs_1"}

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "search_datasets": search_datasets,
            "define_area_of_interest": define_area_of_interest,
            "check_coverage": check_coverage,
            "estimate_retrieval_size": estimate_retrieval_size,
            "retrieve_subset": retrieve_subset,
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        raw_tools = await load_raw_mcp_tools(settings)
        self.tools = bind_workspace(raw_tools, lambda: "test-user")

    async def test_stage_sequence_covers_search_through_render_in_order(self):
        from services.retrieval_composites import await_retrieval, safe_retrieve
        from tools.satellite_tools.plot_tools import make_plot_singular
        from utils.streaming import stream_response
        from eval_harness import contains_subsequence

        plot_singular = make_plot_singular(self.tools)

        class ScriptedSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await self.tools["search_datasets"].ainvoke({"query": "no2", "filters": None})
                await self.tools["define_area_of_interest"].ainvoke({"location": "New Jersey"})
                await self.tools["check_coverage"].ainvoke({
                    "dataset_handle": "dataset_1", "aoi_handle": "aoi_1", "time_range": "2024-01-01/2024-01-02",
                })
                result = await safe_retrieve(
                    "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], self.tools,
                )
                await await_retrieval(result["job_handle"], self.tools)
                await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})
                yield "messages", (SimpleNamespace(content="Plotted NO2 over New Jersey.", type="ai", tool_calls=None), {})

        agent = ScriptedSatelliteAgent()
        agent.tools = self.tools

        events = [event async for event in stream_response(agent, "Plot NO2 over New Jersey", "thread-1")]

        stage_sequence = [
            data["stage"] for event_type, data in events
            if event_type == "status" and data.get("stage")
        ]
        self.assertTrue(
            contains_subsequence(
                stage_sequence,
                ["search", "aoi", "coverage", "estimate", "submit", "progress", "open", "render"],
            ),
            f"stage_sequence was {stage_sequence}",
        )

        text_events = [data for event_type, data in events if event_type == "text"]
        self.assertTrue(text_events, "expected a text event to end the turn")

    async def test_narration_stage_status_precedes_the_final_text_event(self):
        """User story #6: narration stops cleanly when the answer starts
        streaming — every stage status this turn observed arrives before
        the text event, never interleaved after it."""
        from services.retrieval_composites import await_retrieval, safe_retrieve
        from tools.satellite_tools.plot_tools import make_plot_singular
        from utils.streaming import stream_response

        plot_singular = make_plot_singular(self.tools)

        class ScriptedSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                await self.tools["search_datasets"].ainvoke({"query": "no2", "filters": None})
                result = await safe_retrieve(
                    "dataset_1", "aoi_1", "2024-01-01/2024-01-02", ["no2"], self.tools,
                )
                await await_retrieval(result["job_handle"], self.tools)
                await plot_singular.ainvoke({"handle": "obs_1", "location": "global"})
                yield "messages", (SimpleNamespace(content="Plotted NO2 over New Jersey.", type="ai", tool_calls=None), {})

        agent = ScriptedSatelliteAgent()
        agent.tools = self.tools

        events = [event async for event in stream_response(agent, "Plot NO2 over New Jersey", "thread-1")]

        event_types = [event_type for event_type, _ in events]
        first_text_index = event_types.index("text")
        self.assertNotIn("status", event_types[first_text_index + 1:])


if __name__ == "__main__":
    unittest.main()
